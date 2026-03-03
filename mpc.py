"""Model Predictive Control for all methods (STEP 4).

Reimplemented to PyTorch and adapted with certain improvements from:
Reddy et al. (2020), "Learning human objectives by evaluating hypothetical behaviours"
"""

import numpy as np
import utils
import time
import os


class MPCAgent:
    """MPC policy that replans every n_blind_steps.

    Args:
        env: Gym env providing n_z_dim, n_act_dim, rnn_size, action_space
        encoder: VAE used only for optional rendering/plots
        reward_model: Learned reward model (preference- or sparse-based)
        dynamics_model: MDN-RNN that predicts next latents (and c/h)
        obst: Optional obstacle tag to tune action magnitudes
        plan_horizon: Planning horizon of MPC
        n_blind_steps: Number of steps executed until replanning
        use_random: If True, include a fully random action sequence
    """
    def __init__(self, env, encoder, reward_model, dynamics_model, obst=None, plan_horizon=15, n_blind_steps=1,
                 use_random=False):
        self.env = env
        self.encoder = encoder
        self.dynamics_model = dynamics_model
        self.plan_horizon = plan_horizon
        self.n_blind_steps = n_blind_steps

        self.reward_model = reward_model
        self.traj_len = (self.plan_horizon + 1)

        self.init_act_seqs = None
        self.plot_step = 0
        self.use_random = use_random

        self.reset()
        self.obst = obst

    def reset(self):
        """Reset internal step counter and current plan."""
        self.steps = 0
        self.plan = None

    def __call__(self, obs):
        """Return the next action; replan every n_blind_steps.

        Args:
            obs: 1D array [n_z_dim + 2*rnn_size] = [z, c, h]

        Returns:
            Action vector [n_act_dim]
        """
        if self.steps % self.n_blind_steps == 0: # planning again
            self.set_action_sequences()  # action sequences
            data = self.run(init_obs=obs)
            self.plan = data['act_seq'][:self.n_blind_steps] # Only take the first n_blind_steps
            self.steps = 0
        self.steps += 1
        return self.plan[self.steps - 1]

    def set_action_sequences(self):
        """Builds sensible random action sequences as candidates for sample-based MPC planning.

        Notes:
            Creates (i) predefined sequences from a small sensible action set,
            (ii) two-stage randomly-generated combinations most common for CarRacing, and
            (iii) optionally one fully random sequence.
            Steering/accel/brake magnitudes depend on 'obst'.
            This sampling-based logic can be adapted to include more fully random action sequences or
            additional sensible action sequences for a specific environment.
        """
        plan_horizon = self.traj_len - 1
        steer_mag = 1.0
        acc_mag = 0.2
        if self.obst == '_chuckccarobst':
            brake_mag = 0.05
        elif self.obst == '_chuckcobst':
            brake_mag = 0.1
        else:
            brake_mag = 0.2

        predefined_actions = np.array([
            [0.0, acc_mag, 0.0],
            [-steer_mag, 0.1, 0.0],
            [steer_mag, 0.1, 0.0],
            [0.0, 0.1, 0.0],
            [-steer_mag, acc_mag, 0.0],
            [steer_mag, acc_mag, 0.0]
        ])

        act_seqs = [np.tile(act, (plan_horizon, 1)) for act in predefined_actions]

        # Define the available actions for combs in 2 stages
        actions = [
            [0.0, 0.1, 0.0],
            [0.0, acc_mag, 0.0],
            [-steer_mag, 0.1, 0.0],
            [steer_mag, 0.1, 0.0],
            [steer_mag, acc_mag, 0.0],
            [-steer_mag, acc_mag, 0.0],
            [-steer_mag, 0.1, brake_mag],
            [steer_mag, 0.1, brake_mag],
            [0.0, 0.1, brake_mag]
        ]

        # make the action sequences sensible for the environment
        weights = [
            0.13,  # 0 # [0.0, 0.0, 0.0]
            0.32,  # 1 # [0.0, 1.0, 0.0]
            0.30,  # 2 # [-1.0, 0.0, 0.0]
            0.20,  # 3 # [1.0, 0.0, 0.0]
            0.01,  # 4 # [1.0, 1.0, 0.0]
            0.01,  # 5 # [-1.0, 1.0, 0.0]
            0.01,  # 6 # [-1.0, 0.0, 0.8]
            0.01,  # 7 # [1.0, 0.0, 0.8]
            0.01   # 8 # [0.0, 0.0, 0.8]
        ]

        for stage_1_action in actions:
            full_seq = np.zeros((plan_horizon, 3))

            stage1_len = np.random.randint(1, plan_horizon)

            # Stage1: make sure all important actions are considered
            full_seq[:stage1_len] = stage_1_action
            # Stage 2: consider other meaningful actions based on weights
            stage2_action = np.random.choice(len(actions), p=weights)
            full_seq[stage1_len:] = actions[stage2_action]

            act_seqs.append(full_seq)

        # Completely random action sequence
        if self.use_random == True:
            random_act_seq = np.random.randn(plan_horizon, self.env.n_act_dim)
            random_act_seq = utils.normalize_act(random_act_seq, self.env)
            act_seqs.append(random_act_seq)

        self.act_seqs = act_seqs

    def run(self, init_obs, render=False):
        """Evaluate each predefined action sequence and return the best based on learned reward.

        Args:
            init_obs: 1D array [z, c, h] used to initialise dynamics state
            render: If True, render candidate trajectories for sampled-based MPC planning

        Returns:
            Dict with keys:
                'traj': Encoded trajectory (latent rollout)
                'act_seq': Best action sequence [plan_horizon, n_act_dim]
                'traj_ret': return of best action sequence (float)
        """
        best_eval = None

        traj_time = 0
        reward_time = 0
        #plan_start_time = time.time()

        if render:
            evaluations = []

        for idx, act_seq in enumerate(self.act_seqs):
            traj_time_start = time.time()
            enc_traj = self.dynamics_model.enc_traj_of_act_seq(
                init_obs=init_obs[:self.env.n_z_dim][np.newaxis, :],
                act_seq=act_seq[np.newaxis, :],
                traj_len=self.traj_len,
                init_state=(init_obs[self.env.n_z_dim:self.env.n_z_dim+self.env.rnn_size][np.newaxis, :],
                            init_obs[-self.env.rnn_size:][np.newaxis, :])
            )
            traj_time += time.time() - traj_time_start

            reward_time_start = time.time()
            traj_ret = self.get_enc_traj_return(enc_traj, act_seq)
            reward_time += time.time() - reward_time_start

            if render:
                act_seq_sum = utils.summarise_action_sequence(act_seq, self.use_random)
                evaluations.append({'act_seq_sum': act_seq_sum,
                                    'enc_traj': enc_traj[:,:self.env.n_z_dim],
                                    'traj_ret': traj_ret})

            if best_eval is None or traj_ret > best_eval['traj_ret']:
                best_eval = {'traj': enc_traj, 'act_seq': act_seq, 'traj_ret': traj_ret}

        if render: # plot dreamed trajectories with return
            if self.plot_step % 100 == 0:
                utils.plot_mpc_dreams(evaluations, self.encoder, save_path=os.path.join(os.getcwd(), 'mpc_dreams'),
                                      index=self.plot_step)
            self.plot_step +=1

        # plan_time = time.time() - plan_start_time
        # print(f"traj_time: {traj_time:.4f} seconds")
        # print(f"reward_time: {reward_time:.4f} seconds")
        # print(f"plan_time: {plan_time:.4f} seconds")

        return best_eval

    def get_enc_traj_return(self, traj, act_seq, gamma=1.):
        """Compute discounted return for an encoded trajectory.

        Args:
            traj: Array/Tensor of latents over time [T+1, z|c|h]
            act_seq: Array/Tensor [T, n_act_dim] of actions
            gamma: Discount factor

        Returns:
            Scalar return (float)
        """
        if hasattr(self.reward_model, 'get_batch_reward'): # sparse reward model
            rews = self.reward_model.get_batch_reward(traj[:-1], act_seq, traj[1:])
        else: # preference reward model
            act_seq = np.stack(utils.scale_actions(act_seq, self.env.action_space.low, self.env.action_space.high))
            rews = self.reward_model.r_hat(np.concatenate((traj[:-1], act_seq), axis=1))
        disc_rews = (gamma ** np.arange(len(act_seq))) * rews
        traj_ret = np.sum(disc_rews)
        return traj_ret

