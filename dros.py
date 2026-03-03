"""DROS (One-shot technique in STEP 2, with sparse labels)."""

import torch
from copy import deepcopy
import os
import pickle
import time
from collections import defaultdict
import re
import envs
from reward_model_sparse import RewardModelSparse
from dynamics_model import MDNRNNDynamicsModel
import utils
import reward_model_sparse
from traj_opt import GDTrajOptimizer
from vae_model import VAEModel
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk
import random
from mpc import MPCAgent
from utils import compute_perf_metrics


models_dir = os.path.join(os.getcwd(), 'models', 'carracing')
data_dir = os.path.join(os.getcwd(), 'data', 'carracing')

# environment
env = envs.make_carracing_env(n_z_dim=32, rnn_size=256, obst=False)
with open(os.path.join(data_dir, 'dream_user_rollouts.pkl'), 'rb') as f:
    dream_user_rollouts = pickle.load(f)

# VAE
encoder = VAEModel(
    env,
    kl_tolerance=0.5,
    size='L',
    ch='sesquialterate'  # double
)
encoder.load(os.path.join(models_dir, 'enc_user_lat32_ch15L_res84_ep600_epochs100_lrs0_00001.pt'))

# Sparse reward model
n_layers = 5
rew_func_input = "sa"
residual = False
dropout_rate = None # 0.5  # None # 0.2
temp_rm = RewardModelSparse(
    env,
    n_rew_nets_in_ensemble=4,
    n_layers=n_layers,
    layer_size=256,
    rew_func_input=rew_func_input,
    dropout_rate=dropout_rate,
    residual=residual)

dynamics_model = MDNRNNDynamicsModel(env, grad_clip=1.0)
dynamics_model.load(os.path.join(models_dir, 'dyn_gcT_ep600_ch15L_lat32_epochs300_lrs0_0001.pt'))

class TrajectoryFeedbackApp:
    """Tk GUI to label single-step queries with sparse rewards.

    Shows decoded frames and buttons for sparse reward labels, tracks per-query time,
    and stores a list of (s, a, r, s_next, None, None).
    """
    def __init__(self, root, encoder, env):
        """Build widgets and initialise state.

        Args:
            root: Tk root window
            encoder: VAE used to decode frames
            env: Gym env (for reward classes etc.)
        """
        self.root = root
        self.encoder = encoder
        self.env = env
        self.start_time = None  # To track the start time of a query
        self.total_queries = 0  # Total queries across all trajectories
        self.trajectory_query = 1  # Start counting queries from 1
        self.current_trajectory = 0

        self.root.title("Car Racing GUI")

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.trajectory_label = tk.Label(root, text=f"Trajectory: {self.current_trajectory}", font=("Helvetica", 14))
        self.trajectory_label.place(relx=0.9, rely=0.01, anchor="ne")

        self.query_label = tk.Label(root, text=f"Total Queries: {self.total_queries}", font=("Helvetica", 14))
        self.query_label.place(relx=0.9, rely=0.05, anchor="ne")

        self.trajectory_query_label = tk.Label(root, text=f"Traj's Query: {self.trajectory_query}",
                                               font=("Helvetica", 14))
        self.trajectory_query_label.place(relx=0.9, rely=0.09, anchor="ne")

        self.start_button = tk.Button(root, text="Start Trajectory", font=("Helvetica", 14),
                                      command=self.start_trajectory)
        self.start_button.pack(pady=10)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        skip_button = tk.Button(self.button_frame, text="Skip", command=self.skip_query,
                                font=("Helvetica", 16), width=12, height=2)
        skip_button.pack(side=tk.LEFT, padx=5)
        skip_button.config(state=tk.DISABLED)
        self.skip_button = skip_button

        self.reward_buttons = []
        for label, reward in zip(["Grass/Kerbs", "Road", "New Tile"], self.env.rew_classes):
            button = tk.Button(self.button_frame, text=label, command=lambda r=reward: self.record_feedback(r),
                               font=("Helvetica", 16), width=12, height=2)
            button.pack(side=tk.LEFT, padx=5)
            button.config(state=tk.DISABLED)
            self.reward_buttons.append(button)


        self.action_label = tk.Label(root, text="Action: None", font=("Helvetica", 14))
        self.action_label.pack(pady=10)

    def set_trajectory(self, traj, act_seq):
        """Load a new trajectory and action sequence.

        Args:
            traj: Encoded states over time (z|c|h per step)
            act_seq: Actions aligned with transitions
        """
        self.traj = traj
        self.act_seq = act_seq
        self.current_step = 0
        self.sketch = []
        self.query_times = []
        self.update_frame()
        self.start_button.config(state=tk.NORMAL)
        for button in self.reward_buttons:
            button.config(state=tk.DISABLED)
        self.skip_button.config(state=tk.DISABLED)

        self.update_trajectory_label()
        self.update_total_query_label()
        self.update_trajectory_query_label(reset=True)
        self.update_action_label(action="None")

    def decode_frame(self, latent):
        """Decode a latent z to a PIL image for display.

        Args:
            latent: 1D latent vector (z part of state)

        Returns:
            PIL Image resized for the UI
        """
        frame = self.encoder.decode_latent(latent)
        frame = (frame * 255.0).clip(0, 255).astype(np.uint8)

        frame = Image.fromarray(frame).resize((336, 336), Image.NEAREST)
        return frame

    def update_frame(self):
        """Render current step’s frame and start timing."""
        latent_obs = self.traj[self.current_step][:self.env.n_z_dim]
        frame = self.decode_frame(latent_obs)
        img = ImageTk.PhotoImage(frame)
        self.image_label.configure(image=img)
        self.image_label.image = img
        self.start_time = time.time()

    def start_trajectory(self):
        """Enable labeling and show the first query."""
        self.current_step = 1
        self.start_button.config(state=tk.DISABLED)
        for button in self.reward_buttons:
            button.config(state=tk.NORMAL)
        self.skip_button.config(state=tk.NORMAL)
        self.update_frame()

        self.update_action_label(action=str([round(x, 2) for x in self.act_seq[0]]))

    def record_feedback(self, reward):
        """Record a label for the current transition and advance.

        Args:
            reward: Chosen sparse reward label
        """
        end_time = time.time()
        query_duration = end_time - self.start_time
        self.query_times.append(query_duration)

        self.sketch.append((self.traj[self.current_step - 1], self.act_seq[self.current_step - 1], reward,
                            self.traj[self.current_step], None, None))

        self.current_step += 1
        if self.current_step == len(self.traj):  # Last action completed
            self.root.quit()
        else:
            self.update_total_query_label()
            self.update_trajectory_query_label()
            self.update_action_label(action=str([round(x, 2) for x in self.act_seq[self.current_step - 1]]))
            self.update_frame()

    def skip_query(self):
        """Skip current query without recording and advance."""
        self.current_step += 1
        if self.current_step == len(self.traj):
            self.root.quit()
        else:
            self.update_total_query_label()
            self.update_trajectory_query_label()
            self.update_action_label(action=str([round(x, 2) for x in self.act_seq[self.current_step - 1]]))
            self.update_frame()

    def update_total_query_label(self):
        """Increment and refresh the overall query counter."""
        self.total_queries += 1
        self.query_label.config(text=f"Overall Query: {self.total_queries}")

    def update_trajectory_query_label(self, reset=False):
        """Update the per-trajectory query counter.

        Args:
            reset: If True, start from 1 for a new trajectory
        """
        if reset:
            self.trajectory_query = 1
        else:
            self.trajectory_query += 1
        self.trajectory_query_label.config(text=f"Traj's Query: {self.trajectory_query}")

    def update_action_label(self, action):
        """Show the action associated with the current query.

        Args:
            action: Action to be displayed in the action label
        """
        self.action_label.config(text=f"Action: {action}")

    def update_trajectory_label(self):
        """Increment and refresh the trajectory index shown."""
        self.current_trajectory += 1
        self.trajectory_label.config(text=f"Trajectory: {self.current_trajectory}")


def gather_user_feedback(app, root, trajs, act_seqs):
    """Collect sparse reward labels via the GUI for multiple trajectories.

    Args:
        app: TrajectoryFeedbackApp instance
        root: Tk root window
        trajs: List of encoded trajectories
        act_seqs: List of action sequences

    Returns:
        (sketches, query_times)
    """
    sketches = []
    query_times = []
    for traj, act_seq in zip(trajs, act_seqs):
        app.set_trajectory(traj, act_seq)  # Update the trajectory
        root.mainloop()
        sketches.append(app.sketch)
        query_times.append(app.query_times)  # Collect query times for all trajectories
        print("Mean query time:", np.mean(np.concatenate(query_times)))

    return sketches, query_times

num_eval_episodes = 10
render_eval = True
def evaluate(reward_model):
    """Roll out MPC in the real env and compute metrics.

    Args:
        reward_model: Learned reward model used by the MPC agent

    Returns:
        (perf, rollouts, real_rollouts)
        where perf is a dict of aggregated metrics,
        rollouts are encoded transitions, and real_rollouts are raw frames
    """
    mpc_policy = MPCAgent(env, encoder, reward_model, dynamics_model, plan_horizon=30, n_blind_steps=4,
                          use_random=False)

    rollouts = []
    real_rollouts = []
    with torch.no_grad():
        average_episode_reward = 0

        for episode in range(num_eval_episodes):
            real_rollout = []

            obs = env.reset()  # calling the real env reset()
            real_rollout.append(obs)
            obs = utils.process_frame(obs)

            # agent.reset()
            mpc_policy.reset()

            curr_ch_states = (np.zeros((1, env.rnn_size)), np.zeros((1, env.rnn_size)))  # cell/hidden
            done = False
            episode_reward = 0

            rollout = []

            step = 0
            while not done:
                z = encoder.encode_frame(obs)

                if step % 50 == False:
                    curr_ch_states = (np.zeros((1, env.rnn_size)), np.zeros((1, env.rnn_size)))

                current_full_state = np.concatenate((z, curr_ch_states[0][0, :], curr_ch_states[1][0, :]))

                action = mpc_policy(current_full_state)
                obs, reward, done, extra = env.step(action)
                real_rollout.append(obs)
                obs = utils.process_frame(obs)

                episode_reward += reward

                if step > 0:  # Update previous step's full_state with current one
                    rollout[-1][3] = current_full_state  # Set full_state of the previous step

                rollout.append([current_full_state, action, reward, None, float(done), extra])

                _, curr_ch_states = dynamics_model.next_obs(
                    z[np.newaxis, :],
                    action[np.newaxis, :],
                    init_state=curr_ch_states,
                    temperature=0.1,
                    sample=True)

                if render_eval:
                    env.render()

                step += 1

            rollout[-1][3] = np.concatenate((encoder.encode_frame(obs), curr_ch_states[0][0, :],
                                             curr_ch_states[1][0, :]))  # add last full state
            average_episode_reward += episode_reward
            print("Episode_reward", episode_reward)

            rollouts.append(rollout)
            real_rollouts.append(real_rollout)
        average_episode_reward /= num_eval_episodes
        print("Average_episode_reward:", average_episode_reward)

        perf = compute_perf_metrics(rollouts, env)
        print("Average_episode_reward:", perf['rew'])
        print("Average_crash_rate (car in grass or kerbs):", perf['crash'])
        print("Average_success_rate (new tiles):", perf['succ'])

        return perf, rollouts, real_rollouts

def proc_rollouts(rollouts, traj_len=None):
    """Vectorise, trim/pad, and split rollouts into train/val.

    Args:
        rollouts: List of encoded episodes
        traj_len: Optional cap on per-episode length

    Returns:
        Dict with arrays and split indices from utils.split_rollouts
    """
    max_len = max(len(rollout) for rollout in rollouts)
    max_len = min(
        max_len if traj_len is None else max(traj_len - 1, max_len),
        env.max_ep_len)
    return utils.split_rollouts(utils.vectorize_rollouts(rollouts, max_len), train_frac=0.8)

# reward hyperparameters
epochs = 200
lrs = 0.001
weight_decay = 0   # 0 # 1e-6

def main():
    """Run DROS - Synthesises queries, gathers sparse labels via the GUI, trains the reward model."""
    residual_str = "res" if residual else "nores"
    dropout_rate_str = "nodrop" if dropout_rate is None else f"drop{str(dropout_rate).replace('.', '_')}"
    rew_name = (f"dros_"
        f"rew_lat{env.n_z_dim}_nl{n_layers}_inp{rew_func_input}_"
        f"{residual_str}_{dropout_rate_str}_"
        f"wd{str(weight_decay).replace('.', '_')}_epochs{epochs}_lrs{str(lrs).replace('.', '_')}"
    )

    # synthesising queries
    # Here should use the original dream_user_rollouts (not dream_user_rollouts_scaled as in contrast to prefs)
    with torch.no_grad():
        sa_t = temp_rm.create_sketches_queries(dream_user_rollouts, size_segment=50, num_queries=90)

    query_trajs = [sa_t[i, :, :-env.n_act_dim] for i in range(sa_t.shape[0])]
    query_act_seqs = [sa_t[i, :-1, -env.n_act_dim:] for i in range(sa_t.shape[0])]

    with torch.no_grad():
        sketch_rollouts, all_query_times = gather_user_feedback(app, root, query_trajs, query_act_seqs)

    with open(os.path.join(data_dir, f'dros_query_data_rewep{epochs}.pkl'), 'wb') as f:
        pickle.dump(sketch_rollouts, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(data_dir, f'dros_all_query_times_rewep{epochs}_.pkl'), 'wb') as f:
        pickle.dump(all_query_times, f, pickle.HIGHEST_PROTOCOL)

    # train different reward models for every 15 questions
    for sketch, n_queries_made in zip([15, 30, 45, 60, 75, 90], [735, 1470, 2205, 2940, 3675, 4410]):
        rew_name_ = rew_name + f"_que{n_queries_made}"

        # updating reward model
        sketch_data = proc_rollouts(sketch_rollouts[:sketch], traj_len=50)

        reward_model = RewardModelSparse(
            env,
            n_rew_nets_in_ensemble=4,
            n_layers=n_layers,
            layer_size=256,
            rew_func_input=rew_func_input,
            dropout_rate=dropout_rate,  # 0.2 - 0.5
            residual=residual)

        reward_model.learn(
            data=sketch_data,
            epochs=epochs,
            ftol=1e-4,
            batch_size=32,
            learning_rate=lrs,
            weight_decay=weight_decay,
            val_update_freq=5,
            verbose=True,
            rew_name=rew_name_)

        # Should use the model coming from the validation
        # reward_model.load(os.path.join(models_dir, rew_name_ + '.pt'))

    root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = TrajectoryFeedbackApp(root, encoder, env)

    main()
