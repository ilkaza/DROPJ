"""Extract real-world trajectories from keyboard control (STEP 1).

Uses pyglet to capture arrow keys and records (s, a, r, s', done, info) tuples.
"""

import numpy as np
import pyglet
pyglet.options["debug_gl"] = False

import utils
from copy import deepcopy
from gym.envs.box2d.obst_car_racing import ObstCarRacing


def extract_user_trajs(env, n_user_rollouts, max_ep_len=1000):
    """Record n_user_rollouts episodes with keyboard control.

    Args:
        env: Gym environment (CarRacing or ObstCarRacing)
        n_user_rollouts: Number of episodes to record
        max_ep_len: Max steps per episode

    Returns:
        List of episodes, each a list of (prev_obs, action, reward, obs, done, info)

    Notes:
        Arrow keys: ←/→ steer, ↑ gas, ↓ brake. Frames are preprocessed with utils.process_frame.
    """
    from pyglet.window import key

    if isinstance(env.unwrapped, ObstCarRacing):
        env.unwrapped.RECOVER = False
        env.unwrapped.CONTROL_SPEED = False
        env.unwrapped.INITIAL_ACC = False
        env.unwrapped.DRAWING_TRAJECTORIES = True

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    user_rollouts = []

    for ep in range(n_user_rollouts):
        obs = utils.process_frame(env.reset())
        done = False
        prev_obs = deepcopy(obs)
        rollout = []
        total_reward = 0
        for _ in range(max_ep_len):
            if done:
                break

            obs, r, done, info = env.step(a)
            total_reward += r
            env.render()

            obs = utils.process_frame(obs)
            rollout.append(deepcopy((prev_obs, a, r, obs, float(done), info)))
            prev_obs = deepcopy(obs)

        print("episode {} original_total_reward {:+0.2f} total_reward {:+0.2f}".format(
            ep, env.reward, total_reward))
        user_rollouts.append(rollout)
    env.close()

    if isinstance(env.unwrapped, ObstCarRacing):
        env.unwrapped.RECOVER = True
        env.unwrapped.CONTROL_SPEED = True
        env.unwrapped.INITIAL_ACC = True
        env.unwrapped.DRAWING_TRAJECTORIES = False

    return user_rollouts