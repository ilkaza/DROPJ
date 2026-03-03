"""Main file to run DROPJ in Car Racing.

STEP 1: Learn a world model (needs trajectories from the real world)
STEP 2: Create dream user trajectories in one-shot
STEP 3: Provide human preferences and justifications to build a reward model
STEP 4: Deploy with MPC

Notes:
    Controlled by the DO_STEP*_FROM_SCRATCH, EXTRACT_RAW_USER_ROLLOUTS, CREATE_NEW_QUERIES flags and paths below.
"""

from matplotlib import pyplot as plt
import pickle
import random
import os
import numpy as np
import warnings
import torch
import utils
import envs
from dynamics_model import MDNRNNDynamicsModel
from vae_model import VAEModel
from user_extract import extract_user_trajs
import sys


DO_STEP1_FROM_SCRATCH = False
EXTRACT_RAW_USER_ROLLOUTS = False
DO_STEP2_FROM_SCRATCH = False
DO_STEP3_FROM_SCRATCH = False
CREATE_NEW_QUERIES = False

DEBUG = True

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

models_dir = os.path.join(os.getcwd(), 'models', 'carracing')
data_dir = os.path.join(os.getcwd(), 'data', 'carracing')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = envs.make_carracing_env(n_z_dim=32, rnn_size=256, obst=False)
random_policy = utils.make_random_policy(env)


# STEP 1: Train World Model
#--------------------------
# Skip training of World Model (VAE and MDN-RNN), if you don't want to train from scratch,
# but instead use the pretrained models provided

if DO_STEP1_FROM_SCRATCH:
    # extract trajectories for the real-world dataset R
    if EXTRACT_RAW_USER_ROLLOUTS:
        n_user_rollouts = 10  # 600
        raw_user_rollouts = extract_user_trajs(env, n_user_rollouts, max_ep_len=1000)
        with open(os.path.join(data_dir, f'raw_user_rollouts_res84_ep{n_user_rollouts}.pkl'), 'wb') as f:
            pickle.dump(raw_user_rollouts, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(data_dir, f'raw_user_rollouts_res84_ep{n_user_rollouts}.pkl'), 'rb') as f:
            raw_user_rollouts = pickle.load(f)
    else: # use pre-collected trajectories
        n_user_rollouts = 600
        with open(os.path.join(data_dir, f'raw_user_rollouts_res84_ep{n_user_rollouts}.pkl'), 'rb') as f:
            raw_user_rollouts = pickle.load(f)
    if DEBUG:
        user_perf = utils.compute_perf_metrics(raw_user_rollouts, env)

        # plot and save a trajectory
        init_act_seq = utils.act_seq_of_rollout(raw_user_rollouts[0])
        init_traj = utils.traj_of_rollout(raw_user_rollouts[0])
        init_reward_seq = utils.reward_seq_of_rollout(raw_user_rollouts[0])
        utils.plot_trajs([init_traj], env, save_path = 'real_traj.mp4')

    #-------- Train VAE (encoder) ------------
    # preprocess data for VAE
    raw_user_obses = np.array([x[0] for rollout in raw_user_rollouts for x in rollout])
    raw_user_obs_data = utils.split_rollouts({'obses': raw_user_obses}) # only give observations (no rewards)
    with open(os.path.join(data_dir, f'raw_user_obs_data_res84_ep{n_user_rollouts}.pkl'), 'wb') as f:
        pickle.dump(raw_user_obs_data, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_dir, f'raw_user_obs_data_res84_ep{n_user_rollouts}.pkl'), 'rb') as f:
        raw_user_obs_data = pickle.load(f)

    encoder = VAEModel(
        env,
        kl_tolerance=0.5,
        size='L',
        ch='sesquialterate'  # double
    )

    encoder.learn(
        raw_user_obs_data,
        epochs= 100,
        ftol=1e-6,
        learning_rate=1e-3,
        batch_size=32,
        val_update_freq=1,
        verbose=True
        )

    if DEBUG:
        encoder.evaluate(raw_user_obs_data)

        # inspect reconstruction
        with torch.no_grad():
            obs = raw_user_rollouts[0][100][0]
            plt.figure()
            plt.imshow(obs)
            plt.axis('off')
            plt.show()

            latent = encoder.encode_frame(obs)

            recon = encoder.decode_latent(latent)
            plt.figure()
            plt.imshow(recon)
            plt.axis('off')
            plt.show()

    #----------------- Train MDN-RNN (dynamics_model) ---------------------------------
    # preprocess data for MDN-RNN
    raw_user_traj_data = utils.split_rollouts(utils.vectorize_rollouts(raw_user_rollouts,
                                                                       env.max_ep_len, preserve_trajs=True))
    with open(os.path.join(data_dir, f'raw_user_traj_data_res84_ep{n_user_rollouts}.pkl'), 'wb') as f:
        pickle.dump(raw_user_traj_data, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_dir, f'raw_user_traj_data_res84_ep{n_user_rollouts}.pkl'), 'rb') as f:
        raw_user_traj_data = pickle.load(f)

    dynamics_model = MDNRNNDynamicsModel(env, grad_clip=1.0)

    dynamics_model.learn(
        encoder,
        raw_user_traj_data,
        epochs= 300,
        learning_rate=1e-3,
        ftol=1e-6,
        batch_size=32,
        val_update_freq=10,
        verbose=True
        )

    if DEBUG:
        dynamics_model.evaluate(
            encoder,
            raw_user_traj_data,
            batch_size=1000
            )

    #---------- Encoding trajectories as z|c|h - not necessary step if default obs exists --------------------
    def batchify(data, batch_size):
        """Yield chunks of 'data' with size up to 'batch_size'.

        Args:
            data: Sequence or list
            batch_size: Max chunk size

        Returns:
            Generator over slices of data
        """
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    batch_size = 10

    with torch.no_grad():
        rnn_enc_user_rollouts = []
        for batch in batchify(raw_user_rollouts, batch_size):
            encoded_batch = utils.rnn_encode_rollouts(batch, env, encoder, dynamics_model)
            rollouts_batch = utils.rollouts_of_traj_data(encoded_batch)
            rnn_enc_user_rollouts.extend(rollouts_batch)

        with open(os.path.join(data_dir, f"rnn_enc_user_rollouts_ep{n_user_rollouts}.pkl"), 'wb') as f:
            pickle.dump(rnn_enc_user_rollouts, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(data_dir, f"rnn_enc_user_rollouts_ep{n_user_rollouts}.pkl"), 'rb') as f:
            rnn_enc_user_rollouts = pickle.load(f)

else: # load pretrained world model (encoder and dynamics model)
    encoder = VAEModel(
        env,
        kl_tolerance=0.5,
        size='L',
        ch='sesquialterate'  # double
    )
    encoder.load(os.path.join(models_dir, 'enc_user_lat32_ch15L_res84_ep600_epochs100_lrs0_00001.pt'))

    dynamics_model = MDNRNNDynamicsModel(env, grad_clip=1.0)
    dynamics_model.load(os.path.join(models_dir, 'dyn_gcT_ep600_ch15L_lat32_epochs300_lrs0_0001.pt'))

# STEP 2: Play the game in the dream world to extract dream user trajectories
#----------------------------------------------------------------------------
if DO_STEP2_FROM_SCRATCH:

    # Store a dream init obs as the starting point to extract dream_user_rollouts
    if "rnn_enc_user_rollouts" in globals():
        with torch.no_grad():
            default_init_obs = rnn_enc_user_rollouts[0][50][0] # z|c|h
            if DEBUG:
                plt.figure()
                plt.imshow(encoder.decode_latent(default_init_obs[:env.n_z_dim]))
            with open(os.path.join(data_dir, 'default_init_obs.pkl'), 'wb') as f:
                pickle.dump(default_init_obs, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_dir, 'default_init_obs.pkl'), 'rb') as f:
        default_init_obs = pickle.load(f)

    # no trained reward model at this point returns r=0)
    reward_model = None

    if DEBUG:
        # sanity check
        env.default_init_obs = default_init_obs
        with torch.no_grad():
            traj_len = 1000
            # run a rollout in the dream with a random policy
            init_traj, init_act_seq, init_reward_seq = utils.rollout_in_dream(
                random_policy,
                env,
                dynamics_model,
                reward_model,
                encoder,
                init_obs=default_init_obs, # z|c|h
                max_ep_len=(traj_len-1)
                )
            utils.plot_trajs([init_traj], env, encoder, save_path = 'dream.mp4')

    # Click on the gym window, and play the game with the arrows in the dream-world window
    from user_extract_dream import extract_dream_user_trajs
    n_user_rollouts = 10
    with torch.no_grad():
        dream_user_rollouts = extract_dream_user_trajs(env, n_user_rollouts, encoder, dynamics_model, reward_model,
                                                       default_init_obs, max_ep_len=1000)
    with open(os.path.join(data_dir, 'dream_user_rollouts.pkl'), 'wb') as f:
        pickle.dump(dream_user_rollouts, f, protocol=4)
    with open(os.path.join(data_dir, 'dream_user_rollouts.pkl'), 'rb') as f:
        dream_user_rollouts = pickle.load(f)

    # Scale dream_user_rollouts actions for preference reward models in [-1, 1]
    action_space_low = np.array([-1, 0, 0])
    action_space_high = np.array([1, 1, 1])
    for rollout in dream_user_rollouts:
        for i, step in enumerate(rollout):
            observation, action, reward, *rest = step
            scaled_action = utils.scale_action(action, action_space_low, action_space_high)
            rollout[i] = (observation, scaled_action, reward, *rest)
    with open(os.path.join(data_dir, 'dream_user_rollouts_scaled.pkl'), 'wb') as f:
        pickle.dump(dream_user_rollouts, f, protocol=4)

else:
    # load provided dream user trajectories drawn in the pre-trained world model, if they exist
    if DO_STEP3_FROM_SCRATCH:
        dream_rollout_path = os.path.join(data_dir, 'dream_user_rollouts_scaled.pkl')
        try:
            with open(dream_rollout_path, 'rb') as f:
                dream_user_rollouts_scaled = pickle.load(f) # scaled actions in [-1, 1]
        except FileNotFoundError:
            print(f"[Info] {dream_rollout_path} not found. "
                  f"To do STEP 3 from scratch you need 'dream_user_rollouts_scaled'. "
                  f"Otherwise, set DO_STEP3_FROM_SCRATCH = False to load the pre-trained reward model.")
            sys.exit(1)


# STEP 3: Train reward model from human preferences from GUI
# ----------------------------------------------------------
from reward_model_pref import RewardModelPref

ensemble_size = 3
segment = 20
activation = 'tanh'
reward_lr = 0.0003
train_batch_size = 128
reward_update = 60

reward_model = RewardModelPref(
    env.n_obs_dim,
    env.n_act_dim,
    ensemble_size=ensemble_size,
    size_segment=segment,
    activation=activation,
    lr=reward_lr,
    train_batch_size=train_batch_size)

use_justifications = False  # True for DROPJ, False for DROP and DROPe
use_equal_in_plain_prefs = True  # False for DROP, True for DROPe; arbitrary for DROPJ
resp = 500  # number of responses
w_s = 1 # safety justification weight
w_def = 0.75 # default justification weight

if DO_STEP3_FROM_SCRATCH: # create new queries, provide preferences and train reward model from scratch

    # create preference queries - do this once
    if CREATE_NEW_QUERIES:
        with torch.no_grad():
            reward_model.create_pref_queries(env, encoder, dream_user_rollouts_scaled, num_queries=400,
                                             save_path=os.path.join(data_dir, 'queries_s20'))

    # videos (queries)
    with open(os.path.join(data_dir, 'queries_s20','data.pkl'), 'rb') as f:
        queries = pickle.load(f)

    if DEBUG:
        print("STOP: Please run pref_gui.py separately to collect human preferences and justifications.")
        print("Once done, make sure the responses file has the correct path. Then you can ignore this message.")

    # replace with the name of the responses file you created
    if use_justifications==False and use_equal_in_plain_prefs==False:
        responses_file = os.path.join(data_dir, 'responses_s20', f"responses_user26_s20_q600_drop.pkl")
    elif use_justifications==False and use_equal_in_plain_prefs==True:
        responses_file = os.path.join(data_dir, 'responses_s20', f"responses_user27_s20_q600_drope.pkl")
    elif use_justifications==True:
        responses_file = os.path.join(data_dir, 'responses_s20', f"responses_user27_s20_q600_dropj.pkl")

    with open(responses_file, 'rb') as f:
        user_responses = pickle.load(f)

    # time statistics from responses
    total_times = [response['total_time'] for response in user_responses if 'total_time' in response]
    mean_total_time = np.mean(total_times)
    std_total_time = np.std(total_times)

    # train preference reward model
    reward_model.learn_with_GUI(queries, user_responses[:resp], reward_update,
                                use_justifications, w_s, w_def, use_equal_in_plain_prefs)
    if use_justifications == False and use_equal_in_plain_prefs == False:
        reward_model.save(os.getcwd(), note=f'GUI_s20_nojustnoeq_resp{resp}')
    elif use_justifications == False and use_equal_in_plain_prefs == True:
        reward_model.save(os.getcwd(), note=f'GUI_s20_nojusteq_resp{resp}')
    elif use_justifications == True:
        reward_model.save(os.getcwd(), note=f'GUI_s20_just_resp{resp}')

else: # load trained reward models
    if use_justifications==False and use_equal_in_plain_prefs==False:
        reward_model.load(os.path.join(models_dir, 'pref_reward_models'), note=f'GUI_s20_nojustnoeq_resp{resp}')
    elif use_justifications==False and use_equal_in_plain_prefs==True:
        reward_model.load(os.path.join(models_dir, 'pref_reward_models'), note=f'GUI_s20_nojusteq_resp{resp}')
    elif use_justifications==True:
        reward_model.load(os.path.join(models_dir, 'pref_reward_models'), note=f'GUI_s20_just_resp{resp}')


# STEP 4: Deployment (evaluation)
# ------------------------------
from mpc import MPCAgent
from utils import compute_perf_metrics_

num_eval_episodes = 10
render_eval = True
record_video = False

def evaluate(mpc_policy):
    """Roll out MPC in the real env and report metrics.

    Args:
        mpc_policy: MPCAgent callable obs -> action

    Returns:
        [all_metrics, rollouts, real_rollouts]
        where all_metrics has keys {'rew','succ','crash'(='grass')} averaged across episodes

    Notes:
        Uses global settings: num_eval_episodes, render_eval, record_video.
        Relies on globals env, encoder, dynamics_model for stepping/encoding.
    """
    rollouts = []
    real_rollouts = []
    with torch.no_grad():
        average_episode_reward = 0

        for episode in range(num_eval_episodes):
            if record_video:
                import cv2
                VIDEO_W, VIDEO_H = 600, 400
                CROP_H = 350  # Keep top 350px (no score)
                video_writer = cv2.VideoWriter(
                    f"dropj_{episode}.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (VIDEO_W, CROP_H)
                )

            real_rollout = []

            obs = env.reset()  # calling the real env reset()
            real_rollout.append(obs)
            obs = utils.process_frame(obs)

            mpc_policy.reset()

            curr_ch_states = (np.zeros((1, env.rnn_size)), np.zeros((1, env.rnn_size)))  # cell/hidden - init with zeros
            done = False
            episode_reward = 0

            rollout = []

            step = 0
            while not done:
                z = encoder.encode_frame(obs)

                if step % 50 == False:  # zero the memory every 50 steps
                    curr_ch_states = (np.zeros((1, env.rnn_size)), np.zeros((1, env.rnn_size)))

                # Construct the current full state
                current_full_state = np.concatenate((z, curr_ch_states[0][0, :], curr_ch_states[1][0, :]))

                action = mpc_policy(current_full_state) # plan
                obs, reward, done, extra = env.step(action) # take the action in the real env
                real_rollout.append(obs)
                obs = utils.process_frame(obs)

                episode_reward += reward

                if step > 0:  # Update previous step's full_state with current one
                    rollout[-1][3] = current_full_state  # Set full_state of the previous step

                # Append data with a placeholder (None) for full_state
                rollout.append([current_full_state, action, reward, None, float(done), extra])

                _, curr_ch_states = dynamics_model.next_obs( # only RNN state needed
                    z[np.newaxis, :],
                    action[np.newaxis, :],
                    init_state=curr_ch_states,
                    temperature=0.1,
                    sample=True)

                if render_eval:
                    env.render()

                if record_video:
                    frame = env.render(mode="rgb_array")  # Grab the 600x400 RGB frame
                    frame = frame[:CROP_H, :, :]
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                step += 1

            rollout[-1][3] = np.concatenate((encoder.encode_frame(obs), curr_ch_states[0][0, :],
                                             curr_ch_states[1][0, :]))  # add last full state
            average_episode_reward += episode_reward
            print("Episode_reward", episode_reward)

            rollouts.append(rollout)
            real_rollouts.append(real_rollout)

            if record_video:
                video_writer.release
        average_episode_reward /= num_eval_episodes

        all_metrics = compute_perf_metrics_(rollouts)
        print("Average_episode_reward:", np.mean(all_metrics['rew']), '+/-', np.std(all_metrics['rew']))
        print("Average_success_rate (new tiles):", np.mean(all_metrics['succ']), '+/-', np.std(all_metrics['succ']))
        print("Average_crash_rate (grass or kerb):", np.mean(all_metrics['crash']), '+/-', np.std(all_metrics['crash']))

        return [all_metrics, rollouts, real_rollouts]

# create MPC policy
mpc_policy = MPCAgent(env, encoder, reward_model, dynamics_model, plan_horizon=15, n_blind_steps=4, use_random=False)

# evaluate
perf = evaluate(mpc_policy)
