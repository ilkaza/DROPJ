"""Gradient-based trajectory optimisation for ReQueST.

Reimplemented to PyTorch and adapted with certain improvements from:
Reddy et al. (2020), "Learning human objectives by evaluating hypothetical behaviours"
"""

import torch
import torch.optim as optim
import numpy as np
import utils
from matplotlib import pyplot as plt
import time
import random


class GDTrajOptimizer:
    """Optimise action sequences via gradient descent for query generation.

    Args:
        env: Env providing n_z_dim, n_act_dim, rnn_size, max_ep_len
        reward_model: Learned reward model used for get_batch_reward_opt/get_uncertainty
        dynamics_model: MDN-RNN providing enc_traj_of_act_seq_keepgrads(...)
        traj_len: Length of action sequence (+1) to be optimised
        n_trajs: Number of trajectories to optimise in parallel
        query_loss_opt: One of {'rew_uncertainty','max_rew','min_rew','max_nov'}
        learning_rate: Adam learning rate for action variables
        use_random: If True, include a random initial action sequence
    """

    def __init__(self,
                 env,
                 reward_model,
                 dynamics_model,
                 traj_len=50,
                 n_trajs=1,
                 query_loss_opt='rew_uncertainty',
                 learning_rate=1e-2,
                 use_random=False):
        """Initialise optimiser and learnable action variables."""
        self.env = env
        self.reward_model = reward_model
        self.dynamics_model = dynamics_model
        self.traj_len = traj_len
        self.n_trajs = n_trajs
        self.query_loss_opt = query_loss_opt
        self.learning_rate = learning_rate
        self.obs_dim = self.env.n_z_dim
        self.use_random = use_random

        # Define initial action sequence as a learnable parameter
        self.act_seq_var = torch.zeros(self.n_trajs, self.traj_len - 1, self.env.n_act_dim, requires_grad=True)

        self.optimizer = optim.Adam([self.act_seq_var], lr=self.learning_rate)

    def rew_uncertainty_query_loss(self, traj, act_seq, sketch_data):
        """Loss for reward ensemble uncertainty (minimise to maximise uncertainty).

        Args:
            traj: Tensor [T, z|c|h]
            act_seq: Tensor [T-1, n_act_dim] actions used to generate traj
            sketch_data: Unused here (kept for consistency)

        Returns:
            Scalar loss tensor

        Notes:
            # traj and act_seq should have requires_grad=True
        """
        uncertainty = self.reward_model.get_uncertainty(traj[:-1, :], act_seq, traj[1:, :])
        return -torch.mean(uncertainty) # mean over the steps

    def max_rew_query_loss(self, traj, act_seq, sketch_data, gamma=1.):
        """Loss for maximum-reward queries (minimise to increase reward).

        Args:
            traj: Tensor [T, z|c|h]
            act_seq: Tensor [T-1, n_act_dim]
            sketch_data: Unused here (kept for consistency)
            gamma: Discount factor (currently unused in implementation)

        Returns:
            Scalar loss tensor (-mean reward)

        Notes:
            # Should be requires_grad=True (except for first observation)
        """
        rews = self.reward_model.get_batch_reward_opt(traj[:-1], act_seq, traj[1:])
        return - torch.mean(rews) # mean over the steps (following original tf implementation from ReQueST)

    def min_rew_query_loss(self, traj, act_seq, sketch_data):
        """Loss for minimum-reward queries (minimise to decrease reward).

        Args:
            traj: Tensor [T, z|c|h]
            act_seq: Tensor [T-1, n_act_dim]
            sketch_data: Unused here (kept for consistency)

        Returns:
            Scalar loss tensor (+mean reward)
        """
        return -self.max_rew_query_loss(traj, act_seq, sketch_data)

    def max_nov_query_loss(self, traj, act_seq, sketch_data):
        """Loss for novelty of observations.

        Args:
            traj: Tensor [T, z|c|h]
            act_seq: Tensor [T-1, n_act_dim]
            sketch_data: Dict with reference arrays

        Returns:
            Scalar loss tensor (minimisation encourages large latent distance)
        """
        ref_obses = sketch_data.get('next_obses' if self.reward_model.rew_func_input == "sa" else 'obses')
        max_ref_obses = 1000
        if len(ref_obses) > max_ref_obses:
            idxes = random.sample(list(range(len(ref_obses))), max_ref_obses)
            ref_obses = ref_obses[idxes, :]

        obses = traj.unsqueeze(0)
        ref_obses = torch.tensor(ref_obses, dtype=torch.float32).to(traj.device).unsqueeze(1)

        obses = obses[:, :, :self.env.n_z_dim]
        ref_obses = ref_obses[:, :, :self.env.n_z_dim]

        dist = torch.norm(ref_obses - obses, dim=2)
        dist_mult = 1. / torch.sqrt(torch.tensor(ref_obses.shape[2], dtype=torch.float32, device=ref_obses.device))
        loss = torch.mean(torch.exp(-dist_mult * dist))
        return loss

    def set_action_sequences(self):
        """Create initial candidate action sequences.

        Returns:
            List of numpy arrays each of shape [traj_len-1, n_act_dim]
        """
        steer_mag = 1.0
        acc_mag = 0.2
        break_mag = 0.2

        # set from original tf implementation
        predefined_actions = [
            [0., 0., 0.],
            [0., acc_mag, 0.],
            [0., 0., acc_mag],
            [steer_mag, acc_mag, 0.],
            [-steer_mag, acc_mag, 0.],
            [steer_mag, 0., break_mag],
            [-steer_mag, 0., break_mag],
            [steer_mag, 0., 0.],
            [-steer_mag, 0., 0.]
        ]

        act_seqs = [np.tile(act, (self.traj_len - 1, 1)) for act in predefined_actions]

        if self.use_random:
            random_act_seq = np.random.randn(self.traj_len - 1, self.env.n_act_dim)
            act_seqs.append(utils.normalize_act(random_act_seq, self.env))

        return act_seqs

    def run(self, init_obs, gd_iterations, n_queries_made, sketch_data=None, verbose=False):
        """Optimise from multiple initial action sequences and pick the best.

        Args:
            init_obs: 1D array/tensor [n_z_dim + 2*rnn_size] = [z, c, h]
            gd_iterations: Number of gradient steps per action sequence
            n_queries_made: Integer used for naming
            sketch_data: Optional dict passed to *_query_loss methods
            verbose: If True, plot loss traces and save a PNG

        Returns:
            Dict with keys:
                'traj': Encoded trajectory with optimised action sequence (numpy array)
                'act_seq': Optimised action sequence (numpy array)
                'loss': Final loss value (float)
        """
        if verbose:
            plt.figure()
            plt.suptitle(self.query_loss_opt + f'_que{n_queries_made}')

        init_act_seqs = self.set_action_sequences()

        best_eval = None
        for init_act_seq in init_act_seqs:
            data = self._run(init_obs, gd_iterations, init_act_seq, sketch_data, verbose)
            if best_eval is None or data['loss'] < best_eval['loss']:
                best_eval = data
            if verbose:
                plt.legend()
        if verbose:
            plt.savefig(self.query_loss_opt + f'_que{n_queries_made}' +  '.png')
            plt.close()
        return best_eval

    def _run(self, init_obs, gd_iterations, init_act_seq, sketch_data, verbose=False, ftol=1e-4, min_iters=2):
        """Optimise from a single initial action sequence.

        Args:
            init_obs: 1D array/tensor [z, c, h]
            gd_iterations: Number of gradient steps
            init_act_seq: Numpy array [traj_len-1, n_act_dim] initial actions
            sketch_data: Optional dict for *_query_loss methods
            verbose: If True, accumulate loss for plotting
            ftol: Tolerance placeholder for utils.converged()
            min_iters: Minimum steps placeholder for utils.converged()

        Returns:
            Dict with 'traj', 'act_seq', 'loss' as numpy/float
        """
        init_obs = torch.tensor(init_obs, dtype=torch.float32)
        self.act_seq_var.data = torch.tensor(init_act_seq, dtype=torch.float32)
        best_eval = None
        loss_evals = []

        start_time = time.time()
        for t in range(gd_iterations):
            self.optimizer.zero_grad()

            enc_traj = self.dynamics_model.enc_traj_of_act_seq_keepgrads(
                init_obs=init_obs[:self.env.n_z_dim].unsqueeze(0),
                act_seq=self.act_seq_var.unsqueeze(0),
                traj_len=self.traj_len,
                init_state=(init_obs[self.env.n_z_dim:self.env.n_z_dim+self.env.rnn_size].unsqueeze(0),
                            init_obs[-self.env.rnn_size:].unsqueeze(0))
            )

            loss = eval(f'self.{self.query_loss_opt}_query_loss(enc_traj, self.act_seq_var, sketch_data)')

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.act_seq_var.data = utils.clamp_act(self.act_seq_var, self.env)

            loss_val = loss.item()
            loss_evals.append(loss_val)

            if best_eval is None or loss_val < best_eval['loss']:
                best_eval = {
                    'traj': enc_traj.detach().cpu().numpy(),
                    'act_seq': self.act_seq_var.detach().cpu().numpy(),
                    'loss': loss_val
                }

            # if utils.converged(loss_evals, ftol, min_iters):
            #     break
        if verbose:
            print(f'call to update_op_{self.query_loss_opt}: %0.3f' % (time.time() - start_time))
            print('iterations: %d' % t)
            plt.plot(loss_evals, label=str([round(x, 2) for x in init_act_seq[0]]))
            plt.show(block=False)
            plt.pause(0.1)
        else:
            print(f'call to update_op_{self.query_loss_opt}: %0.3f' % (time.time() - start_time))
            print('iterations: %d' % t)
        return best_eval
