"""Extract dream (latent-space) trajectories via keyboard control (STEP 2).

Uses pyglet to capture arrow keys and records (s, a, r, s', done, info) tuples.
"""

import numpy as np
import pyglet
pyglet.options["debug_gl"] = False
from gym.envs.box2d.obst_car_racing import ObstCarRacing


def extract_dream_user_trajs(env, n_user_rollouts, encoder, dynamics_model, true_reward_model,
                             default_init_obs, default_init_obses=None, max_ep_len=1000):
    """Record n_user_rollouts dream episodes with keyboard control.

    Args:
        env: Gym environment
        n_user_rollouts: Number of episodes to record
        encoder: VAE used to decode z for on-screen display
        dynamics_model: MDN-RNN used to step in latent space
        true_reward_model: Optional reward model for r(s,a,s')
        default_init_obs: Starting latent state [z|c|h]
        default_init_obses: Optional list of initial latents to cycle through
        max_ep_len: Max steps per episode

    Returns:
        List of episodes, each a list of (prev_obs, action, reward, obs, done, info)

    Notes:
        Arrow keys: ←/→ steer, ↑ gas, ↓ brake
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

    dream_env = DreamEnv(env, dynamics_model, true_reward_model, encoder)
    dream_env.env.default_init_obs = default_init_obs

    if default_init_obses:
        dream_env.env.default_init_obses = default_init_obses
        dream_env.init_obs_idx = 0

    render_first_time = True

    for ep in range(n_user_rollouts):
        env.reset()
        obs = dream_env.reset()

        done = False
        prev_obs = deepcopy(obs)
        rollout = []
        total_reward = 0

        dream_env.render_dream(render_first_time)
        render_first_time = False
        for _ in range(max_ep_len):
            if done:
                break
            env.step(a)
            obs, r, done, info = dream_env.step(a) # a taken from the keyboard
            total_reward += r

            rollout.append(deepcopy((prev_obs, a, r, obs, float(done), info)))
            prev_obs = deepcopy(obs)

            env.render()
            dream_env.render_dream()

        print("episode {} total_reward {:+0.2f}".format(ep, total_reward))
        user_rollouts.append(rollout)

    if isinstance(env.unwrapped, ObstCarRacing):
        env.unwrapped.RECOVER = True
        env.unwrapped.CONTROL_SPEED = True
        env.unwrapped.INITIAL_ACC = True
        env.unwrapped.DRAWING_TRAJECTORIES = False

    return user_rollouts


from copy import deepcopy
import random
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display, clear_output
from gym.spaces.box import Box
from gym.utils import seeding


class DreamEnv(gym.Env):
    """Latent-space environment driven by the learned world-model dynamics.

    Wraps a real env for metadata; steps happen in z|c|h space via the dynamics model;
    the encoder is used only for rendering.
    """

    def __init__(self, env, dynamics_model, true_reward_model, encoder=None):
        """Mirror env attributes and store models.

        Args:
            env: Source env providing n_obs_dim, n_z_dim, rnn_size, etc.
            dynamics_model: MDN-RNN used to predict next latent and next c/h
            encoder: VAE used only to render decoded frames
            true_reward_model: Optional model with get_reward(prev_obs, act, curr_obs)
        """
        super(DreamEnv, self).__init__()
        self.__dict__.update(env.__dict__)

        self.env = env  # <Timelimit<CarRacing is passed to self.env
        self.curr_obs = None
        self.curr_ch_states = None

        self.dynamics_model = dynamics_model
        self.true_reward_model = true_reward_model
        self.encoder = encoder
        self.observation_space = Box(low=-50., high=50., shape=(env.n_obs_dim,), dtype=np.float32)
        self._seed()
        self.use_random_init_obs = False

    def _seed(self, seed=None):
        """Initialise RNG for reproducibility.

        Args:
            seed: Optional integer seed

        Returns:
            [seed] as per Gym convention
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Reset to an initial latent state and RNN (c,h).

        Returns:
            1D latent observation concatenated as [z, c, h]
        """
        if hasattr(self.env, 'default_init_obses'):
            if self.use_random_init_obs:
                self.curr_obs = random.choice(self.env.default_init_obses).ravel()
            else:
                self.curr_obs = self.env.default_init_obses[self.init_obs_idx].ravel()
                self.init_obs_idx = (self.init_obs_idx + 1) % len(self.env.default_init_obses)
        else:
            self.curr_obs = self.env.default_init_obs.ravel()
        c = self.curr_obs[self.env.n_z_dim:self.env.n_z_dim + self.dynamics_model.rnn_size]
        h = self.curr_obs[-self.env.rnn_size:]
        self.curr_ch_states = (c[np.newaxis, :], h[np.newaxis, :])
        return self.curr_obs

    def step(self, act):
        """Advance one step using the dynamics model.

        Args:
            act: Action vector (same shape as env.n_act_dim)

        Returns:
            obs: Next latent observation [z, c, h]
            r: Scalar reward (0 if no true_reward_model)
            done: False (termination handled outside)
            info: {}

        Notes:
            # Here the action should be rescaled to original env range
        """
        act = act.ravel()
        obs = self.curr_obs
        obs = obs[:self.env.n_z_dim]

        next_obs, next_state = self.dynamics_model.next_obs(
            obs[np.newaxis, :],  # adding 1 dimension "batch" dim.
            act[np.newaxis, :],
            init_state=self.curr_ch_states)

        prev_obs = deepcopy(self.curr_obs)
        self.curr_obs = next_obs.ravel()
        self.curr_ch_states = next_state

        if self.true_reward_model is None:
            r = 0
        else:
            r = self.true_reward_model.get_reward(prev_obs, act, self.curr_obs)

        done = False
        info = {}

        return self.curr_obs, r, done, info

    def render_dream(self, render_first_time=False):
        """Display the decoded current frame from the dream env.

        Args:
            render_first_time: If True, initialise the figure; else update the image
        """
        with torch.no_grad():
            if render_first_time:
                plt.ion()
                self.fig, ax = plt.subplots()
                ax.axis('off')
                frame = self.encoder.decode_latent(self.curr_obs[:self.env.n_z_dim])
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                self.im = ax.imshow(frame, interpolation='none')
                display(self.fig)
            else:
                frame = self.encoder.decode_latent(self.curr_obs[:self.env.n_z_dim])
                self.im.set_data(frame)

                clear_output(wait=True)
                #display(self.fig)
                plt.pause(0.01)
