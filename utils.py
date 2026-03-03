"""Utility functions for preprocessing, rollouts, plotting, scaling, and metrics.

Some methods adapted from:
Reddy et al. (2020), "Learning human objectives by evaluating hypothetical behaviours"
"""

from copy import deepcopy
import collections
import os
import random
import gym
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from IPython.display import display, clear_output
from collections import defaultdict, Counter

inf = 999.


def process_frame(frame):
    """For CarRacing crop/resize a frame to 84x84 and replace black with white;
     for Obstacle Car Racing it remains the same.

    Args:
        frame: HxWx3 array

    Returns:
        84x84x3 array for CarRacing or
        HxWx3 array for Obstacle Car Racing
    """
    if frame.shape[0] == 96:
        obs = frame[0:84, :, :]  # crops the bottom part of the frame with the score

        obs_pil = Image.fromarray(obs)

        obs_resized = obs_pil.resize((84, 84), Image.ANTIALIAS)

        obs_resized_np = np.array(obs_resized)

        black_pixels_mask = np.all(obs_resized_np == 0, axis=-1)
        obs_resized_np[black_pixels_mask] = [255, 255, 255]
        return obs_resized_np
    else:
        frame = np.array(frame)
        return frame


def map_frames(rollouts, f, batch=False):
    """Apply a mapping f to all observations in rollouts.

    Args:
        rollouts: List of episodes, each a list of (s, a, r, s', done, info)
        f: Callable mapping obs -> obs'
        batch: If True, apply f in batches for speed

    Returns:
        Rollouts with mapped observations (and next_obs)
    """

    # Apply f to all observations in rollouts
    if not batch:
        return [[(f(s), a, r, f(ns), d, i)
                 for s, a, r, ns, d, i in rollout]
                for rollout in rollouts]
    obses = []
    for rollout in rollouts:
        for s, a, r, ns, d, i in rollout:
            obses.append(s)
    obses = np.array(obses)

    next_obses = []
    for rollout in rollouts:
        for s, a, r, ns, d, i in rollout:
            next_obses.append(ns)
    next_obses = np.array(next_obses)

    chunk_size = 512
    n_chunks = int(np.ceil(obses.shape[0] / chunk_size))
    mapped_obses = []
    mapped_next_obses = []
    for i in range(n_chunks):
        more_mapped_obses = f(obses[i * chunk_size:(i + 1) * chunk_size])
        mapped_obses.extend(more_mapped_obses)

        more_mapped_next_obses = f(next_obses[i * chunk_size:(i + 1) * chunk_size])
        mapped_next_obses.extend(more_mapped_next_obses)

    mapped_obses = np.array(mapped_obses)
    mapped_next_obses = np.array(mapped_next_obses)

    mapped_rollouts = deepcopy(rollouts)
    flat_idx = 0
    for rollout_idx, rollout in enumerate(mapped_rollouts):
        for t, (s, a, r, ns, d, i) in enumerate(rollout):
            mapped_rollouts[rollout_idx][t] = (mapped_obses[flat_idx], a, r, mapped_next_obses[flat_idx], d, i)
            flat_idx += 1
    return mapped_rollouts


def run_ep(policy,
           env,
           max_ep_len=None,
           proc_obs=(lambda x: x),
           render=True):
    """Run one episode in the real env and collect a rollout.

    Args:
        policy: Callable with .reset() optional, maps obs → action
        env: Gym environment
        max_ep_len: Cap on episode length (defaults to env.max_ep_len)
        proc_obs: Preprocessor applied to observations before policy
        render: If True, visualize (decoded for CarRacing)

    Returns:
        List of (obs, act, rew, next_obs, done, info)
    """
    if env.name == 'carracing' and not isinstance(env, 'DreamEnv'):
        proc_obs = process_frame

    if max_ep_len is None or max_ep_len > env.max_ep_len:
        max_ep_len = env.max_ep_len

    try:
        policy.reset()
    except:
        pass

    obs = proc_obs(env.reset())

    done = False
    prev_obs = deepcopy(obs)
    rollout = []

    for _ in range(max_ep_len):
        if done:
            break
        action = policy(prev_obs)
        obs, r, done, info = env.step(action)
        obs = proc_obs(obs)
        rollout.append(deepcopy((prev_obs, action, r, obs, float(done), info)))
        prev_obs = deepcopy(obs)
        if render:
            try:
                env.render()
            except NotImplementedError:
                pass
    return rollout


def run_ep_dream(policy,
                 env,
                 encoder,
                 max_ep_len=None,
                 render=True):
    """Roll out a policy in latent space ("dream") and optionally render.

    Args:
        policy: Callable taking obs and returning action
        env: Gym env with n_z_dim and max_ep_len
        encoder: EncoderModel for decode/encode utilities
        max_ep_len: Cap on episode length
        render: If True, display decoded frames

    Returns:
        List of (s, a, r, s', done, info)
    """
    if max_ep_len is None or max_ep_len > env.max_ep_len:
        max_ep_len = env.max_ep_len

    obs = env.reset()

    done = False
    prev_obs = deepcopy(obs)
    rollout = []

    if render:
        plt.ion()
        fig, ax = plt.subplots()
        ax.axis('off')

        first_point = obs[:env.n_z_dim]
        frame = encoder.decode_latent(first_point)
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

        im = ax.imshow(frame, interpolation='none')
        count = 0

    for _ in range(max_ep_len):
        if done:
            break
        action = policy(prev_obs)
        obs, r, done, info = env.step(action)
        rollout.append(deepcopy((prev_obs, action, r, obs, float(done), info)))

        prev_obs = deepcopy(obs)
        if render:
            next_point = obs
            if next_point is None or (max_ep_len is not None and count >= max_ep_len):
                break

            next_point = next_point[:env.n_z_dim]
            frame = encoder.decode_latent(next_point)
            im.set_data(frame)

            clear_output(wait=True)
            display(fig)
            plt.pause(0.01)

            count += 1

    plt.close(fig)
    return rollout


def set_dream_init_obs(obs_r, encoder):
    """Make initial latent obs for dreaming from a real RGB frame.

    Args:
        obs_r: HxWx3 array
        encoder: EncoderModel

    Returns:
        1D numpy array: [z, zeros_for_c_and_h]
    """
    obs_r = process_frame(obs_r)
    with torch.no_grad():
        dream_init_obs = encoder.encode_frame(obs_r)
    dream_init_obs = np.concatenate((dream_init_obs, np.zeros(512)))
    return dream_init_obs


def vectorize_rollouts(rollouts, max_ep_len, preserve_trajs=False):
    """Unzip rollouts into arrays (obs, act, rew, next_obs, done).

    Args:
        rollouts: List of episodes
        max_ep_len: Max steps per episode to keep
        preserve_trajs: If True, pad each episode to max_ep_len
                        If False -> flatten rollouts, lose episode structure

    Returns:
        Dict with arrays
    """
    data = {'obses': [], 'actions': [], 'rews': [], 'next_obses': [], 'dones': []}
    for rollout in rollouts:
        more_obses, more_actions, more_rews, more_next_obses, more_dones = [list(x) for x in
                                                                            zip(*rollout[:max_ep_len])][:5]
        if preserve_trajs:
            more_obses = pad(np.array(more_obses), max_ep_len)
            more_actions = pad(np.array(more_actions), max_ep_len)
            more_rews = pad(np.array(more_rews), max_ep_len)
            more_next_obses = pad(np.array(more_next_obses), max_ep_len)
            more_dones = pad(np.array(more_dones), max_ep_len)
        data['obses'].append(more_obses)
        data['actions'].append(more_actions)
        data['rews'].append(more_rews)
        data['next_obses'].append(more_next_obses)
        data['dones'].append(more_dones)

    if not preserve_trajs:
        data = {k: sum(v, []) for k, v in data.items()}

    data = {k: np.array(v) for k, v in data.items()}

    if preserve_trajs:
        data['traj_lens'] = np.array(
            [len(rollout[:max_ep_len]) + 1 for rollout in rollouts])  # remember where padding begins

    idxes = list(range(len(data['obses'])))
    random.shuffle(idxes)
    data = {k: v[idxes] for k, v in data.items()}

    return data


def rollouts_of_traj_data(traj_data):
    """Inverse of vectorize_rollouts(..., preserve_trajs=True).

    Args:
        traj_data: Dict produced by vectorize_rollouts with preserve_trajs=True

    Returns:
        List of rollouts (episodes)
    """
    rollouts = []
    for i in range(traj_data['obses'].shape[0]):
        ep_len = traj_data['traj_lens'][i] - 1
        rollout = list(
            zip(traj_data['obses'][i, :ep_len], traj_data['actions'][i, :ep_len], traj_data['rews'][i, :ep_len],
                traj_data['next_obses'][i, :ep_len], traj_data['dones'][i, :ep_len], [{}] * ep_len))
        rollouts.append(rollout)
    return rollouts


def traj_of_rollout(rollout):
    """Extract observation trajectory from a rollout.

    Args:
        rollout: List of (s, a, r, s', done, info)

    Returns:
        Array of observations over time (including final s' if present)
    """
    traj = [x[0] for x in rollout]
    last_obs = rollout[-1][3]
    if last_obs is not None:
        traj.append(last_obs)
    return np.array(traj)


def act_seq_of_rollout(rollout):
    """Extract action sequence from a rollout.

    Args:
        rollout: List of (s, a, r, s', done, info)

    Returns:
        Array of actions over time
    """
    return np.array([x[1] for x in rollout])


def reward_seq_of_rollout(rollout):
    """Extract reward sequence from a rollout.

    Args:
        rollout: List of (s, a, r, s', done, info)

    Returns:
        Array of rewards over time
    """
    return np.array([x[2] for x in rollout])


def split_rollouts(rollouts, train_frac=0.9):
    """Random train/val split for vectorised data.

    Args:
        rollouts: Dict with arrays (e.g. from vectorize_rollouts)
        train_frac: Fraction for training

    Returns:
        Dict rollouts updated with 'train_idxes', 'val_idxes',
        'train_idxes_of_rew_class' and 'train_idxes_of_act'
    """
    idxes = list(range(rollouts['obses'].shape[0]))
    random.shuffle(idxes)
    n_train_examples = int(train_frac * len(idxes))
    train_idxes = idxes[:n_train_examples]
    val_idxes = idxes[n_train_examples:]

    rews = rollouts.get('rews', None)
    if rews is not None and len(rews.shape) != 2 and all(r is not None for r in rews):

        def proc_idxes(idxes):
            idxes_of_rew_class = collections.defaultdict(list)
            for idx in idxes:
                idxes_of_rew_class[float(rews[idx])].append(idx)
            idxes_of_rew_class = dict(idxes_of_rew_class)
            return idxes_of_rew_class

        train_idxes_of_rew_class = proc_idxes(train_idxes)
    else:
        train_idxes_of_rew_class = None

    if 'actions' in rollouts:
        actions = rollouts['actions']

        def proc_idxes(idxes):
            idxes_of_act = collections.defaultdict(list)
            for idx in idxes:
                idxes_of_act[float(np.argmax(actions[idx]))].append(idx)
            idxes_of_act = dict(idxes_of_act)
            return idxes_of_act

        train_idxes_of_act = proc_idxes(train_idxes)
    else:
        train_idxes_of_act = None

    rollouts.update({
        'train_idxes': train_idxes,
        'val_idxes': val_idxes,
        'train_idxes_of_rew_class': train_idxes_of_rew_class,
        'train_idxes_of_act': train_idxes_of_act
    })
    return rollouts


def pad(arr, max_len):
    """Zero-pad a first-dimension-short array up to max_len; complete trajectories remain the same.

    Args:
        arr: Array to pad on axis 0
        max_len: Target length

    Returns:
        Padded array with length max_len on axis 0
    """
    n = arr.shape[0]
    if n > max_len:
        raise ValueError
    elif n == max_len:
        return arr
    else:
        shape = [max_len - n]
        shape.extend(arr.shape[1:])
        padding = np.zeros(shape)
        return np.concatenate((arr, padding), axis=0)


def make_random_policy(env):
    """Return a policy that samples actions from env.action_space.

    Args:
        env: Gym environment

    Returns:
        Callable obs -> action
    """
    policy = lambda _: env.action_space.sample()
    return policy


def plot_trajs(trajs, env, encoder=None, save_path=None):
    """Plot multiple trajectories (decoded if latent).

    Args:
        trajs: List of trajectories (either frames or latents z)
        env: Gym env (n_z_dim used to infer latent)
        encoder: VAE for decoding if trajectories are z
        save_path: Optional path to save as MP4

    Returns:
        None
    """
    global counter
    if env.name == 'carracing':
        for traj_eval in trajs:
            if encoder is not None:
                traj_eval = np.array(traj_eval)[:, :env.n_z_dim]
                frames = encoder.decode_batch_latents(traj_eval)
            else:
                frames = np.array(traj_eval)

            fig = plt.figure()
            ax = plt.axes()

            plt.axis('off')
            im = plt.imshow(frames[0], interpolation='none')

            def init():
                im.set_data(frames[0])
                return [im]

            def animate(i):
                im.set_array(frames[i])
                return [im]

            anim = animation.FuncAnimation(
                fig,
                animate,
                init_func=init,
                frames=len(frames),
                interval=40,
                blit=True)

            if save_path is not None:
                anim.save(save_path, writer='ffmpeg')

            plt.close(fig)


def plot_traj(traj, env, encoder=None, save_path=None):
    """Plot a single trajectory (decoded if latent).

    Args:
        traj: Sequence of frames or latents z
        env: Gym env (n_z_dim used to infer latent)
        encoder: EncoderModel for decoding if traj is z
        save_path: Optional path to save as MP4

    Returns:
        None
    """
    if encoder is not None:
        traj = np.array(traj)[:, :env.n_z_dim]
        frames = encoder.decode_batch_latents(traj)
    else:
        frames = np.array(traj)

    fig = plt.figure()
    ax = plt.axes()

    plt.axis('off')
    im = plt.imshow(frames[0], interpolation='none')

    def init():
        im.set_data(frames[0])
        return [im]

    def animate(i):
        im.set_array(frames[i])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(frames),
        interval=50,
        blit=False)

    if save_path is not None:
        anim.save(save_path, writer='ffmpeg')

    plt.close(fig)


def converged(val_losses, ftol, min_iters=2, eps=1e-9):
    """Relative-improvement convergence check.

    Args:
        val_losses: List of validation losses
        ftol: Relative tolerance
        min_iters: Minimum length before checking
        eps: Small constant

    Returns:
        True if converged else False
    """
    return len(val_losses) >= max(2, min_iters) and (
                val_losses[-1] == np.nan or abs(val_losses[-1] - val_losses[-2]) / (eps + abs(val_losses[-2])) < ftol)


def isinstance(obj, cls_name):
    """Shallow isinstance by class name.

    Args:
        obj: Object to test
        cls_name: Class name string

    Returns:
        True if obj.__class__.__name__ equals cls_name
    """
    return obj.__class__.__name__ == cls_name


def rnn_encode_rollouts(raw_rollouts, env, encoder, dynamics_model):
    """Encode rollouts to latents and attach RNN c/h states.

    Args:
        raw_rollouts: List of episodes with raw frames
        env: Env providing n_z_dim and max_ep_len
        encoder: EncoderModel with encode_batch_frames
        dynamics_model: MDN-RNN with rnn_encode_rollouts(...)

    Returns:
        Dict with encoded trajectories concatenated with c/h states
    """
    enc_rollouts = map_frames(raw_rollouts, encoder.encode_batch_frames, batch=True)
    enc_traj_data = split_rollouts(vectorize_rollouts(enc_rollouts, env.max_ep_len, preserve_trajs=True))
    return dynamics_model.rnn_encode_rollouts(enc_traj_data)


class DreamEnv(gym.Env):
    """Latent-space environment driven by the learned dynamics.

    Wraps a real env's metadata; steps are performed by the dynamics model
    in latent space.

    Args:
        env: Source env whose attributes are mirrored (e.g. n_z_dim, max_ep_len)
        dynamics_model: MDN-RNN providing next_obs(...)
        true_reward_model: Optional model with get_reward(prev_obs, act, curr_obs)
    """
    def __init__(self, env, dynamics_model, true_reward_model):
        super(DreamEnv, self).__init__()
        self.__dict__.update(env.__dict__)
        self.env = env
        self.dynamics_model = dynamics_model
        self.true_reward_model = true_reward_model

        self.curr_obs = None

        self.curr_ch_states = None

    def reset(self):
        """Reset to default_init_obs and initialise RNN states.

        Returns:
            obs: 1D latent observation (concatenated with initial c/h)
        """
        self.curr_obs = self.default_init_obs.ravel()
        c = self.curr_obs[self.env.n_z_dim:self.env.n_z_dim + self.dynamics_model.rnn_size]
        h = self.curr_obs[-self.env.rnn_size:]
        self.curr_ch_states = (c[np.newaxis, :], h[np.newaxis, :])
        return self.curr_obs

    def step(self, act):
        """Advance one step using the dynamics model.

        Args:
            act: Action vector

        Returns:
            obs: Next latent observation [z|c|h]
            rew: Scalar reward (0 if no true_reward_model)
            done: Always False here (handled at higher level)
            info: Empty dict
        """
        act = act.ravel()
        obs = self.curr_obs
        obs = obs[:self.env.n_z_dim]

        next_obs, next_state = self.dynamics_model.next_obs(
            obs[np.newaxis, :],
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


def rollout_in_dream(policy,
                     env,
                     dynamics_model,
                     true_reward_model,
                     encoder,
                     init_obs=None,
                     max_ep_len=None):
    """Roll out a policy in DreamEnv and return trajectories.

    Args:
        policy: Callable mapping obs -> action
        env: Real env whose metadata is mirrored
        dynamics_model: MDN-RNN for latent transitions
        true_reward_model: Optional reward model
        encoder: VAE used for decoding in rendering
        init_obs: Latent start state for DreamEnv
        max_ep_len: Episode length cap

    Returns:
        (traj, act_seq, reward_seq)
    """
    dream_env = DreamEnv(env, dynamics_model, true_reward_model)
    if init_obs is not None:
        dream_env.env.default_init_obs = init_obs
    rollout = run_ep_dream(policy, dream_env, encoder, max_ep_len=max_ep_len, render=True)
    init_act_seq = act_seq_of_rollout(rollout)
    init_traj = traj_of_rollout(rollout)
    init_reward_seq = reward_seq_of_rollout(rollout)
    return init_traj, init_act_seq, init_reward_seq


def perm_tf2pt(image):
    """Permute image tensors from TF layout to PyTorch layout.

    Args:
        image: Tensor [N,H,W,C] or [H,W,C] (RGB assumed)

    Returns:
        Tensor [N,C,H,W] or [C,H,W]
    """
    if image.dim() == 4 and image.shape[1] != 3:  # Assuming RGB images with 3 channels
        image_tensor = image.permute(0, 3, 1, 2)
    elif image.dim() == 3 and image.shape[0] != 3:
        image_tensor = image.permute(2, 0, 1)
    return image_tensor


class CustomDataset(Dataset):
    """Subset view over a dict of arrays/tensors keyed by data_keys.

    Args:
        data: Dict of arrays/tensors
        data_keys: Keys to include from data
        idxes: Indices selecting the subset
    """
    def __init__(self, data, data_keys, idxes):
        self.data = {key: data[key] for key in data_keys}
        self.idxes = idxes

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, index):
        """Return a dict of tensors for the given index.

        Args:
            index: Integer in [0, len(self))

        Returns:
            Dict {key: tensor} for the sample
        """
        idx = self.idxes[index]
        return {key: torch.tensor(self.data[key][idx], dtype=torch.float) for key in self.data}


def get_pi_idx(x, pdf):
    """Sample a mixture/component index from cumulative probs.

    Args:
        x: Uniform sample in [0, 1]
        pdf: 1D array of mixture probabilities summing to 1

    Returns:
        Integer index in [0, len(pdf)-1]
    """

    # samples from a categorical distribution
    N = pdf.shape[0]
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
            return i
    print('error with sampling ensemble')
    return -1


def balance_classes(rollouts):
    """Downsample steps so each reward class has equal count.

    Args:
        rollouts: List of episodes; each step's reward class at index 2

    Returns:
        New list of episodes with per-class counts balanced to the minimum
    """

    # Initialise counters for each reward class
    class_counts = defaultdict(int)
    # Count the occurrences of each reward class in the dataset
    for rollout in rollouts:
        for step in rollout:
            reward_class = step[2]
            class_counts[reward_class] += 1
    # Find the minimum count across the classes to balance to this number
    min_count = min(class_counts.values())

    # Initialise the new balanced rollouts list
    balanced_rollouts = []
    # Initialise counters for how many samples we've kept from each class
    kept_counts = defaultdict(int)
    for rollout in rollouts:
        balanced_rollout = []
        for step in rollout:
            reward_class = step[2]
            # Only add the step if we haven't reached the limit for this class
            if kept_counts[reward_class] < min_count:
                balanced_rollout.append(step)
                kept_counts[reward_class] += 1
        balanced_rollouts.append(balanced_rollout)

    return balanced_rollouts


def clamp_act(act, env):
    """Clamp action tensor to env.action_space bounds.

    Args:
        act: Tensor action
        env: Gym env with action_space.low/high

    Returns:
        Clamped action tensor
    """
    act_low = torch.tensor(env.action_space.low, device=act.device)
    act_high = torch.tensor(env.action_space.high, device=act.device)

    clamped_act = torch.clamp(act, min=act_low, max=act_high)

    clamping_applied = act != clamped_act

    num_clamped_elements = clamping_applied.sum().item()

    return clamped_act


def unnormalize_act(act, env, eps=0):
    """Map action from env space -> logits via inverse-sigmoid.

    Args:
        act: Action in env bounds
        env: Gym env
        eps: Small padding added/subtracted to bounds

    Returns:
        Real-valued logits (same shape as act)
    """
    act_low = env.action_space.low - eps
    act_range = env.action_space.high + eps - act_low
    p = (act - act_low) / act_range
    return logit(p)


def logit(p):
    """Elementwise logit (numpy)."""
    return np.log(p / (1.0 - p))


def normalize_act(act, env, eps=0):
    """Map logits -> action in env bounds via sigmoid.

    Args:
        act: Real-valued logits
        env: Gym env
        eps: Small padding added/subtracted to bounds

    Returns:
        Action in env bounds
    """
    act_low = env.action_space.low - eps
    act_range = env.action_space.high + eps - act_low
    return act_low + sigmoid(act) * act_range


def sigmoid(x):
    """Elementwise sigmoid (numpy)."""
    return 1 / (1 + np.exp(-x))


def compute_perf_metrics(rollouts, env):
    """Aggregate rollout metrics (reward, success/crash rate, length) used on Car Racing experiments.

    Args:
        rollouts: List of episodes
        env: Gym env

    Returns:
        Dict with keys {'rew','succ','crash'(='grass'),'rolloutlen'}
    """
    metrics = {}
    metrics['rew'] = np.mean([sum(r for s, a, r, ns, d, i in rollout) for rollout in rollouts])
    for key in ['succ', 'crash']:
        if env.only_terminal_reward:
            inds = [1 if rollout[-1][-1].get(key, False) else 0 for rollout in rollouts]
        else:
            inds = [1 if x[-1].get(key, False) else 0 for rollout in rollouts for x in rollout]  # carracing
        metrics[key] = np.mean(inds)
    metrics['rolloutlen'] = np.mean([len(rollout) for rollout in rollouts])
    return metrics


def compute_perf_metrics_(rollouts):
    """Aggregate Car Racing metrics.

    Args:
        rollouts: List of episodes

    Returns:
        Dict with keys {'rew','succ','crash'(='grass'),'rolloutlen'}
    """
    metrics = {}
    metrics['rew'] = [sum(r for s, a, r, ns, d, i in rollout) for rollout in rollouts]
    metrics['succ'] = [np.mean([1 if x[-1].get('succ', False) else 0 for x in rollout]) for rollout in
                       rollouts]  # x[-1] is the 'extra'
    metrics['crash'] = [np.mean([1 if x[-1].get('crash', False) else 0 for x in rollout]) for rollout in rollouts]
    metrics['rolloutlen'] = [len(rollout) for rollout in rollouts]
    return metrics


def compute_perf_metrics_obst(rollouts):
    """Aggregate Obstacle Car Racing metrics.

    Args:
        rollouts: List of episodes

    Returns:
        Dict with keys {'rew','succ','crash'(='grass'),'rolloutlen', 'chuck', 'chuck_passed','car','car_passed'}
    """
    metrics = {}
    metrics['rew'] = [sum(r for s, a, r, ns, d, i in rollout) for rollout in rollouts]
    metrics['succ'] = [np.mean([1 if x[-1].get('succ', False) else 0 for x in rollout]) for rollout in
                       rollouts]  # x[-1] is the 'extra'
    metrics['crash'] = [np.mean([1 if x[-1].get('crash', False) else 0 for x in rollout]) for rollout in
                        rollouts]  # absolute number of steps in the grass
    metrics['rolloutlen'] = [len(rollout) for rollout in rollouts]

    metrics['chuck'] = [np.sum([1 if x[-1].get('chuck', False) else 0 for x in rollout]) for rollout in
                        rollouts]  # absolute number of chuckholes stepped
    metrics['chuck_passed'] = [np.sum([1 if x[-1].get('chuck_passed', False) else 0 for x in rollout]) for rollout in
                               rollouts]  # absolute number of chuckholes passed

    metrics['car'] = [np.sum([1 if x[-1].get('car', False) else 0 for x in rollout]) for rollout in
                      rollouts]  # absolute number of cars crashed
    metrics['car_passed'] = [np.sum([1 if x[-1].get('car_passed', False) else 0 for x in rollout]) for rollout in
                             rollouts]  # absolute number of cars passed

    return metrics


def summarise_action_sequence(act_seq, is_random=False):
    """Run-length summary of a 3D continuous action sequence.

    Args:
        act_seq: Array [T, 3] of actions (steer, gas, brake)
        is_random: If True, annotate random action sequence

    Returns:
        Multiline string summarising repeated actions and counts
    """
    if is_random:
        return "Random sequence"

    prev_action = act_seq[0]
    count = 1
    sequence_summary = []

    for i in range(1, len(act_seq)):
        if (act_seq[i] == prev_action).all():  # If the action is the same as the previous one
            count += 1
        else:
            sequence_summary.append(f"[{prev_action[0]:.1f} {prev_action[1]:.1f} {prev_action[2]:.1f}]: {count}")
            prev_action = act_seq[i]
            count = 1

    sequence_summary.append(f"[{prev_action[0]:.1f} {prev_action[1]:.1f} {prev_action[2]:.1f}]: {count}")

    return "\n".join(sequence_summary)


def plot_mpc_dreams(evaluations, encoder, save_path=None, index=None):
    """Visualise decoded dream rollouts from MPC evaluations.

    Args:
        evaluations: List of dicts with 'enc_traj', 'act_seq', 'traj_ret'
        encoder: VAE to decode latents
        save_path: MP4 filepath
        index: Figure index/title helper

    Returns:
        None
    """
    n_cols = min(5, len(evaluations))
    n_rows = math.ceil(len(evaluations) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

    dec_trajs = []
    ims = []

    max_return = max(eval['traj_ret'] for eval in evaluations)

    for idx, (eval, ax) in enumerate(zip(evaluations, axes.flatten())):
        enc_traj = eval['enc_traj']
        dec_traj = encoder.decode_batch_latents(enc_traj)
        dec_trajs.append(dec_traj)

        im = ax.imshow(dec_traj[0])
        ax.axis('off')

        if eval['traj_ret'] == max_return:
            ax.set_title(f"{eval['act_seq_sum']} \n Ret: {eval['traj_ret']:.3f}", color='red')
        else:
            ax.set_title(f"{eval['act_seq_sum']} \n Ret: {eval['traj_ret']:.3f}")

        ims.append(im)
    plt.tight_layout(pad=5.0)

    def update(frame):
        for im, dec_traj in zip(ims, dec_trajs):
            im.set_data(dec_traj[frame])
        return ims

    for i in range(idx + 1, n_rows * n_cols):
        fig.delaxes(axes.flatten()[i])

    anim = FuncAnimation(fig, update, frames=range(enc_traj.shape[0]), interval=400, blit=True)

    if save_path is not None:
        if index is not None:
            file_name = os.path.join(save_path, f'{index}.mp4')
            anim.save(file_name, writer='ffmpeg')
            index += 1
        else:
            anim.save(os.path.join(save_path, 'mpc_dreams'), writer='ffmpeg')

    plt.close(fig)


def scale_actions(actions, action_space_low, action_space_high):
    """Min–max scale actions to [0,1] given env bounds.

    Args:
        actions: Array [..., A]
        action_space_low: Array [A] lower bounds
        action_space_high: Array [A] upper bounds

    Returns:
        Scaled actions in [0,1]
    """
    scaled_actions = []
    for action in actions:
        scaled_action = 2 * (action - action_space_low) / (action_space_high - action_space_low) - 1
        scaled_action = np.clip(scaled_action, -1, 1)
        scaled_actions.append(scaled_action)
    return scaled_actions


def rescale_action(action, eval_env):
    """Rescale a single action from [0,1] to env bounds.

    Args:
        action: Array/Tensor [A] scaled to [0,1]
        eval_env: Gym env providing action_space.low/high

    Returns:
        Action in env bounds
    """
    if not isinstance(action, np.ndarray):
        action = np.array(action)

    lb = eval_env.action_space.low
    ub = eval_env.action_space.high

    rescaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
    rescaled_action = np.clip(rescaled_action, lb, ub)
    return rescaled_action


def scale_action(action, action_space_low, action_space_high):
    """Min–max scale a single action to [0,1].

    Args:
        action: Array [A]
        action_space_low: Array [A]
        action_space_high: Array [A]

    Returns:
        Scaled action in [0,1]
    """
    scaled_action = 2 * (action - action_space_low) / (action_space_high - action_space_low) - 1
    scaled_action = np.clip(scaled_action, -1, 1)
    return scaled_action


def torch_logit(p):
    """Elementwise logit (PyTorch)."""
    return torch.log(p / (1. - p))
