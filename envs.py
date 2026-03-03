""" Initialising Car Racing or Obstacle Car Racing environments."""

import numpy as np
import gym
from gym.envs.registration import register


def make_carracing_env(n_z_dim=64,
                       rnn_size=1024,
                       succ_rew_bonus=10.,
                       crash_rew_penalty=-1.,
                       obst=False):
    """Configures either the standard `CarRacing-v0` environment or
    'ObstCarRacing-v0' including obstacles (if available in your Gym installation).

    Args:
        n_z_dim: VAE latent size
        rnn_size: RNN hidden size for the world model
        succ_rew_bonus: Reward for driving on a new tile
        crash_rew_penalty: Penalty for driving on grass
        obst: If True, use the obstacle variant

    Returns:
        Gym environment configured for CarRacing/ObstCarRacing
    """
    if obst:
        register(
            id="ObstCarRacing-v0",
            entry_point="gym.envs.box2d:ObstCarRacing",
            max_episode_steps=1000,
            reward_threshold=900,
        )
        env = gym.make('ObstCarRacing-v0')
    else:
        env = gym.make('CarRacing-v0')
    env.n_act_dim = 3  # steer, gas, brake
    env.max_ep_len = 1000
    env.name = 'carracing'
    env.default_init_obs = None
    env.succ_rew_bonus = succ_rew_bonus
    env.crash_rew_penalty = crash_rew_penalty
    env.rew_classes = np.array([env.crash_rew_penalty, 0., env.succ_rew_bonus])
    env.only_terminal_reward = False
    env.n_z_dim = n_z_dim
    env.rnn_size = rnn_size
    env.n_obs_dim = n_z_dim + 2 * rnn_size

    return env
