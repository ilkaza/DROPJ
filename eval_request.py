# Evaluate ReQueST

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

from mpc import MPCAgent
from utils import compute_perf_metrics
from utils import compute_perf_metrics_

models_dir = os.path.join(os.getcwd(), 'models', 'carracing')
data_dir = os.path.join(os.getcwd(), 'data', 'carracing')

######### traj_opt_times ##########
def seconds_to_min_sec(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return minutes, remaining_seconds

with open(os.path.join(data_dir, f'request_traj_opt_times_guiTrue_rewep200_rewinitFalse_gdit200.pkl'), 'rb') as f:
    traj_opt_times = pickle.load(f)
all_values = []
for key in traj_opt_times.keys():
    all_values.extend(traj_opt_times[key])

all_values = np.array(all_values)

mean_all = np.mean(all_values)
std_all = np.std(all_values)

mean_min, mean_sec = seconds_to_min_sec(mean_all)
std_min, std_sec = seconds_to_min_sec(std_all)

print("Results (Minutes and Seconds):")
print(f"Mean: {mean_min} minutes and {mean_sec:.2f} seconds")
print(f"Standard Deviation: {std_min} minutes and {std_sec:.2f} seconds")
# Mean: 3 minutes and 13.58 seconds
# Standard Deviation: 0 minutes and 4.50 seconds

request_traj_time = mean_all * 45
mean_min, mean_sec = seconds_to_min_sec(request_traj_time)
print(f"ReQueST Trajectory Optimisation time until model with best Mean Return (45trajs):"
      f" {mean_min} minutes and {mean_sec:.2f} seconds")
# 145 minutes and 10.91 seconds

request_traj_time = mean_all * 75
mean_min, mean_sec = seconds_to_min_sec(request_traj_time)
print(f"ReQueST Trajectory Optimisation time until model with best Mean Return and Grass Rate (75trajs):"
      f" {mean_min} minutes and {mean_sec:.2f} seconds")
# 241 minutes and 58.19 seconds
################################################################################

# environment
env = envs.make_carracing_env(n_z_dim=32, rnn_size=256, obst=False)

# VAE
encoder = VAEModel(
    env,
    kl_tolerance=0.5,
    size='L',
    ch='sesquialterate'  # double
)

encoder.load(os.path.join(models_dir, 'enc_user_lat32_ch15L_res84_ep600_epochs100_lrs0_00001.pt'))

# Dynamics Model
dynamics_model = MDNRNNDynamicsModel(env, grad_clip=1.0)
dynamics_model.load(os.path.join(models_dir, 'dyn_gcT_ep600_ch15L_lat32_epochs300_lrs0_0001.pt'))

num_eval_episodes = 10
render_eval = True
def evaluate(reward_model):
    mpc_policy = MPCAgent(env, encoder, reward_model, dynamics_model, plan_horizon=25, n_blind_steps=4,
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

                # Construct the current full state
                current_full_state = np.concatenate((z, curr_ch_states[0][0, :], curr_ch_states[1][0, :]))

                action = mpc_policy(current_full_state)
                obs, reward, done, extra = env.step(action)  # take the action in the real env
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

        all_metrics = compute_perf_metrics_(rollouts)
        print("Average_episode_reward:", np.mean(all_metrics['rew']), '+/-', np.std(all_metrics['rew']))
        print("Average_success_rate (new tiles):", np.mean(all_metrics['succ']), '+/-', np.std(all_metrics['succ']))
        print("Average_crash_rate (grass or kerb):", np.mean(all_metrics['crash']), '+/-', np.std(all_metrics['crash']))

        return [all_metrics, rollouts, real_rollouts]

# ReQueST reward model
n_layers = 5
rew_func_input = "sa"
residual = False
dropout_rate = None  # None # 0.2

reward_model = RewardModelSparse(
    env,
    n_rew_nets_in_ensemble=4,
    n_layers=n_layers,
    layer_size=256,
    rew_func_input=rew_func_input,
    dropout_rate=dropout_rate,  # Typically set between 0.2 and 0.5
    residual=residual)

epochs = 200
lrs = 0.001
weight_decay = 0   # 0 #1e-6

residual_str = "res" if residual else "nores"
dropout_rate_str = "nodrop" if dropout_rate is None else f"drop{str(dropout_rate).replace('.', '_')}"
rew_name = (f"req_"
    f"rew_lat{env.n_z_dim}_nl{n_layers}_inp{rew_func_input}_"
    f"{residual_str}_{dropout_rate_str}_"
    f"wd{str(weight_decay).replace('.', '_')}_epochs{epochs}_lrs{str(lrs).replace('.', '_')}"
)

# perf_0 = evaluate(reward_model)
# perf_0.append(0)
#
# rew_name_1 = 'final_' + rew_name + '_noinit' + '_que735' # or # 'final_' + rew_name + '_noinit' + '_que735' + '_clean'
# reward_model.load(os.path.join(models_dir, 'request_reward_models', rew_name_1 + '.pt'))
# perf_1 = evaluate(reward_model)
# perf_1.append(735)
#
# rew_name_clean2 = 'final_' + rew_name + '_noinit' + '_que1470' + '_clean'
# reward_model.load(os.path.join(models_dir, 'request_reward_models', rew_name_clean2 + '.pt'))
# perf_clean2 = evaluate(reward_model)
# perf_clean2.append(1470)
#
# rew_name_clean3 = 'final_' + rew_name + '_noinit' + '_que2205' + '_clean'
# reward_model.load(os.path.join(models_dir, 'request_reward_models', rew_name_clean3 + '.pt'))
# perf_clean3 = evaluate(reward_model)
# perf_clean3.append(2205)
#
# rew_name_clean4 = 'final_' + rew_name + '_noinit' + '_que2940' + '_clean'
# reward_model.load(os.path.join(models_dir, 'request_reward_models', rew_name_clean4 + '.pt'))
# perf_clean4 = evaluate(reward_model)
# perf_clean4.append(2940)

rew_name_clean5 = 'final_' + rew_name + '_noinit' + '_que3675' + '_clean'
reward_model.load(os.path.join(models_dir, 'request_reward_models', rew_name_clean5 + '.pt'))
perf_clean5 = evaluate(reward_model)
perf_clean5.append(3675)

# rew_name_clean6 = 'final_' + rew_name + '_noinit' + '_que4410' + '_clean'
# reward_model.load(os.path.join(models_dir, 'request_reward_models', rew_name_clean6 + '.pt'))
# perf_clean6 = evaluate(reward_model)
# perf_clean6.append(4410)



