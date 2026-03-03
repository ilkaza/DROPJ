# Evaluate DROS

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
    mpc_policy = MPCAgent(env, encoder, reward_model, dynamics_model, plan_horizon=25, n_blind_steps=4, use_random=False)

    rollouts = []
    real_rollouts = []
    with torch.no_grad():
        average_episode_reward = 0

        for episode in range(num_eval_episodes):
            real_rollout = []

            obs = env.reset()  # calling the real env reset() #
            real_rollout.append(obs)
            obs = utils.process_frame(obs)

            # agent.reset()
            mpc_policy.reset()

            curr_ch_states = (np.zeros((1, env.rnn_size)), np.zeros((1, env.rnn_size)))
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

        all_metrics = compute_perf_metrics_(rollouts)
        print("Average_episode_reward:", np.mean(all_metrics['rew']), '+/-', np.std(all_metrics['rew']))
        print("Average_success_rate (new tiles):", np.mean(all_metrics['succ']), '+/-', np.std(all_metrics['succ']))
        print("Average_crash_rate (grass or kerb):", np.mean(all_metrics['crash']), '+/-', np.std(all_metrics['crash']))

        return [all_metrics, rollouts, real_rollouts]


# DROS reward model
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
rew_name = (f"dros_"
    f"rew_lat{env.n_z_dim}_nl{n_layers}_inp{rew_func_input}_"
    f"{residual_str}_{dropout_rate_str}_"
    f"wd{str(weight_decay).replace('.', '_')}_epochs{epochs}_lrs{str(lrs).replace('.', '_')}"
)

# perf_0 = evaluate(reward_model)
# perf_0.append(0)
#
# #######
# rew_name_1 = rew_name + '_que735'
# reward_model.load(os.path.join(models_dir, 'dros_reward_models', rew_name_1 + '.pt'))
# perf_1 = evaluate(reward_model)
# perf_1.append(735)
#
# rew_name_2 = rew_name + '_que1470'
# reward_model.load(os.path.join(models_dir, 'dros_reward_models', rew_name_2 + '.pt'))
# perf_2 = evaluate(reward_model)
# perf_2.append(1470)
#
# rew_name_3 = rew_name + '_que2205'
# reward_model.load(os.path.join(models_dir, 'dros_reward_models', rew_name_3 + '.pt'))
# perf_3 = evaluate(reward_model)
# perf_3.append(2205)

rew_name_4 = rew_name + '_que2940'
reward_model.load(os.path.join(models_dir, 'dros_reward_models', rew_name_4 + '.pt'))
perf_4 = evaluate(reward_model)
perf_4.append(2940)

# rew_name_5 = rew_name + '_que3675'
# reward_model.load(os.path.join(models_dir, 'dros_reward_models', rew_name_5 + '.pt'))
# perf_5 = evaluate(reward_model)
# perf_5.append(3675)
#
# rew_name_6 = rew_name + '_que4410'
# reward_model.load(os.path.join(models_dir, 'dros_reward_models', rew_name_6 + '.pt'))
# perf_6 = evaluate(reward_model)
# perf_6.append(4410)
