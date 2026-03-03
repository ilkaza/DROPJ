"""Main file to run DROPJ in Obstacle Car Racing with multiple justifications.

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

models_dir = os.path.join(os.getcwd(), 'models', 'obstcarracing')
data_dir = os.path.join(os.getcwd(), 'data', 'obstcarracing')
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

env = envs.make_carracing_env(n_z_dim=64, rnn_size=1024, obst=True)

res = 128 # resolution
# choose chuckholes or chuckholes + cars
OBST_MODE = 'chuckccar' # options: 'chuckc', 'chuckccar'
if OBST_MODE == 'chuckc':
    obst = '_chuckcobst'
elif OBST_MODE == 'chuckccar':
    obst = '_chuckccarobst'
else:
    obst = ''

# STEP 1: Train World Model
#--------------------------
# Skip training of World Model (VAE and MDN-RNN), if you don't want to train from scratch,
# but instead use the pretrained models provided

if DO_STEP1_FROM_SCRATCH:
    # extract trajectories for the real-world dataset R
    if EXTRACT_RAW_USER_ROLLOUTS:
        n_user_rollouts = 1  # 600
        raw_user_rollouts = extract_user_trajs(env, n_user_rollouts, max_ep_len=1000)
        with open(os.path.join(data_dir, f'raw_user_rollouts_res{res}_ep{n_user_rollouts}{obst}.pkl'), 'wb') as f:
            pickle.dump(raw_user_rollouts, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(data_dir, f'raw_user_rollouts_res{res}_ep{n_user_rollouts}{obst}.pkl'), 'rb') as f:
            raw_user_rollouts = pickle.load(f)
    else: # use pre-collected trajectories
        if OBST_MODE == 'chuckc':
            n_user_rollouts = 600
        elif OBST_MODE == 'chuckccar':
            n_user_rollouts = 800
        with open(os.path.join(data_dir, f'raw_user_rollouts_res{res}_ep{n_user_rollouts}{obst}.pkl'), 'rb') as f:
            raw_user_rollouts = pickle.load(f)
    if DEBUG:
        user_perf = utils.compute_perf_metrics_obst(raw_user_rollouts)

    #-------- Train VAE (encoder) ------------
    # preprocess data for VAE
    raw_user_obses = np.array([x[0] for rollout in raw_user_rollouts for x in rollout])
    raw_user_obs_data = utils.split_rollouts({'obses': raw_user_obses}) # only give observations (no rewards)
    with open(os.path.join(data_dir, f'raw_user_obs_data_res{res}_ep{n_user_rollouts}{obst}.pkl'), 'wb') as f:
        pickle.dump(raw_user_obs_data, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_dir, f'raw_user_obs_data_res{res}_ep{n_user_rollouts}{obst}.pkl'), 'rb') as f:
        raw_user_obs_data = pickle.load(f)

    encoder = VAEModel(
        env,
        kl_tolerance=2,
        size='res128', # 'L', 'XL', 'res128'
        ch='quadruple' # 'sesquialterate', 'double', 'quadruple'
    )

    encoder.learn(
        raw_user_obs_data,
        epochs=90,
        ftol=1e-6,
        learning_rate=1e-4,
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
    with open(os.path.join(data_dir, f'raw_user_traj_data_res{res}_ep{n_user_rollouts}{obst}.pkl'), 'wb') as f:
        pickle.dump(raw_user_traj_data, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_dir, f'raw_user_traj_data_res{res}_ep{n_user_rollouts}{obst}.pkl'), 'rb') as f:
        raw_user_traj_data = pickle.load(f)

    dynamics_model = MDNRNNDynamicsModel(env, grad_clip=1.0, num_mixture=7)

    dynamics_model.learn(
        encoder,
        raw_user_traj_data,
        epochs=300,
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

    #---------- Encoding trajectories as z|c|h --------------------
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

        with open(os.path.join(data_dir, f"rnn_enc_user_rollouts_ep{n_user_rollouts}{obst}_lat{env.n_z_dim}.pkl"), 'wb'
                  ) as f:
            pickle.dump(rnn_enc_user_rollouts, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(data_dir, f"rnn_enc_user_rollouts_ep{n_user_rollouts}{obst}_lat{env.n_z_dim}.pkl"), 'rb'
                  ) as f:
            rnn_enc_user_rollouts = pickle.load(f)

else:
    encoder = VAEModel(
        env,
        kl_tolerance=2,
        size='res128',
        ch='quadruple'
    )

    if OBST_MODE == 'chuckc':
        encoder.load(
            os.path.join(models_dir, f'enc_user_lat64_ch4res128_res{res}_ep600{obst}_epochs90_lrs0.0001_kl2.pt'))
    elif OBST_MODE == 'chuckccar':
        encoder.load(
            os.path.join(models_dir, f'enc_user_lat64_ch4res128_res{res}_ep800{obst}_epochs90_lrs0.0001_kl2.pt'))

    dynamics_model = MDNRNNDynamicsModel(env, grad_clip=1.0, num_mixture=7)

    if OBST_MODE == 'chuckc':
        dynamics_model.load(os.path.join(models_dir, f'dyn_gcT_ep600{obst}_ch4res128_lat64'
                                                     f'_encep90_enclr0.0001_epochs50_lrs0.0001_rnn1024_mix7.pt'))
    elif OBST_MODE == 'chuckccar':
        dynamics_model.load(os.path.join(models_dir, f'dyn_gcT_ep800{obst}_ch4res128_lat64'
                                                     f'_encep90_enclr0.0001_epochs30_lrs0.0001_rnn1024_mix7.pt'))

# STEP 2: Play the game in the dream world to extract dream user trajectories
# ----------------------------------------------------------------------------
if DO_STEP2_FROM_SCRATCH:

    # Store a dream init obs(es) as the starting point(s) to extract dream_user_rollouts
    if "rnn_enc_user_rollouts" in globals():
        with torch.no_grad():
            default_init_obs = rnn_enc_user_rollouts[0][50][0] # z|c|h
            if DEBUG:
                plt.figure()
                plt.imshow(encoder.decode_latent(default_init_obs[:env.n_z_dim]))
            with open(os.path.join(data_dir, f'default_init_obs{obst}.pkl'), 'wb') as f:
                pickle.dump(default_init_obs, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_dir, f'default_init_obs{obst}.pkl'), 'rb') as f:
        default_init_obs = pickle.load(f)

    # can gather more default observations
    if "rnn_enc_user_rollouts" in globals():
        with torch.no_grad():
            default_init_obses = []
            for i in range(len(rnn_enc_user_rollouts)):
                rollout = rnn_enc_user_rollouts[i]
                for j in range(100, len(rollout), 100):
                    if len(default_init_obses) >= 60:
                        break
                    default_init_obses.append(rollout[j][0])
                if len(default_init_obses) >= 60:
                    break

            # for init_obs in default_init_obses:
            #     plt.figure()
            #     plt.imshow(encoder.decode_latent(init_obs[:env.n_z_dim]))
            #     plt.show()
            with open(os.path.join(data_dir, f'default_init_obses{obst}.pkl'), 'wb') as f:
                pickle.dump(default_init_obses, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_dir, f'default_init_obses{obst}.pkl'), 'rb') as f:
        default_init_obses = pickle.load(f) # z|c|h

    reward_model = None

    # Click on the gym window, and play the game with the arrows in the dream-world window
    from user_extract_dream import extract_dream_user_trajs
    n_user_rollouts = 60

    with torch.no_grad():
        dream_user_rollouts = extract_dream_user_trajs(env, n_user_rollouts, encoder, dynamics_model, reward_model,
                                                       default_init_obs, default_init_obses, max_ep_len=1000) # z|c|h
    with open(os.path.join(data_dir, f'dream_user_rollouts{obst}.pkl'), 'wb') as f:
        pickle.dump(dream_user_rollouts, f, protocol=4)
    with open(os.path.join(data_dir, f'dream_user_rollouts{obst}.pkl'), 'rb') as f:
        dream_user_rollouts = pickle.load(f)

    # Scale dream_user_rollouts actions for preference reward models in [-1, 1]
    action_space_low = np.array([-1, 0, 0])
    action_space_high = np.array([1, 1, 1])
    for rollout in dream_user_rollouts:
        for i, step in enumerate(rollout):
            observation, action, reward, *rest = step
            scaled_action = utils.scale_action(action, action_space_low, action_space_high)
            rollout[i] = (observation, scaled_action, reward, *rest)
    with open(os.path.join(data_dir, f'dream_user_rollouts{obst}_scaled.pkl'), 'wb') as f:
        pickle.dump(dream_user_rollouts, f, protocol=4)

else:
    # load provided dream user trajectories drawn in the pre-trained world model, if they exist
    if DO_STEP3_FROM_SCRATCH:
        dream_rollout_path = os.path.join(data_dir, f'dream_user_rollouts{obst}_scaled.pkl')
        try:
            with open(dream_rollout_path, 'rb') as f:
                dream_user_rollouts_scaled = pickle.load(f) # scaled actions in [-1, 1]
        except FileNotFoundError:
            print(f"[Info] {dream_rollout_path} not found. "
                  f"To do STEP 3 from scratch you need 'dream_user_rollouts{obst}_scaled'. "
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
reward_update = 40

reward_model = RewardModelPref(
    env.n_obs_dim,
    env.n_act_dim,
    ensemble_size=ensemble_size,
    size_segment=segment,
    activation=activation,
    lr=reward_lr,
    train_batch_size=train_batch_size)

resp = 1000
if OBST_MODE == 'chuckc':
    w_def = 0.75
    w_grass = 1.0
    w_chuck = 1.0
    just_weights = {'Default': w_def, 'Grass': w_grass, 'Chuckhole': w_chuck}
elif OBST_MODE == 'chuckccar':
    w_def = 1.0
    w_grass = 1.0
    w_chuck = 1.0
    w_car = 1.0
    just_weights = {'Default':w_def, 'Grass':w_grass, 'Chuckhole':w_chuck, 'Car':w_car}

if DO_STEP3_FROM_SCRATCH: # create new queries, provide preferences and train reward model from scratch

    # create preference queries - do this once (attention many windows will open with num_queries=1000)
    if CREATE_NEW_QUERIES:
        with torch.no_grad():
            reward_model.create_pref_queries(env, encoder, dream_user_rollouts_scaled, num_queries=10, # 1000, 1500
                                             save_path=os.path.join(data_dir, f'queries_s20{obst}'))

    # videos (queries)
    with open(os.path.join(data_dir, f'queries_s20{obst}','data.pkl'), 'rb') as f:
        queries = pickle.load(f)

    if DEBUG:
        print("STOP: Please run obst_pref_gui.py separately to collect human preferences and justifications.")
        print("Once done, make sure the responses file has the correct path. Then you can ignore this message.")

    # replace with the name of the responses file you created
    if OBST_MODE == 'chuckc':
        responses_file = os.path.join(data_dir, f'responses_s20{obst}', f"responses_chuckcobst_user44_mjust.pkl")
    elif OBST_MODE == 'chuckccar':
        responses_file = os.path.join(data_dir, f'responses_s20{obst}', f"responses_chuckccarobst_user45_mjust.pkl")

    with open(responses_file, 'rb') as f:
        user_responses = pickle.load(f)

    # time statistics from responses
    total_times = [response['total_time'] for response in user_responses if 'total_time' in response]
    mean_total_time = np.mean(total_times)
    std_total_time = np.std(total_times)

    # train preference reward model
    reward_model.learn_with_GUI_multi_justs(queries, user_responses, reward_update, just_weights)

    if OBST_MODE == 'chuckc':
        reward_model.save(os.getcwd(), note=f'GUI_s20{obst}_mjust_wdef{w_def}_wgrass{w_grass}_wchuck{w_chuck}')
    elif OBST_MODE == 'chuckccar':
        reward_model.save(os.getcwd(), note=f'GUI_s20{obst}_mjust'
                                            f'_wdef{w_def}_wgrass{w_grass}_wchuck{w_chuck}_wcar{w_car}')

else: # load pretrained reward models
    if OBST_MODE == 'chuckc':
        reward_model.load(os.path.join(models_dir, 'pref_reward_models'),
                          note=f'GUI_s20{obst}_mjust_wdef{w_def}_wgrass{w_grass}_wchuck{w_chuck}')
    elif OBST_MODE == 'chuckccar':
        reward_model.load(os.path.join(models_dir, 'pref_reward_models'),
                          note=f'GUI_s20{obst}_mjust_wdef{w_def}_wgrass{w_grass}_wchuck{w_chuck}_wcar{w_car}')


# STEP 4: Deployment (evaluation)
# ------------------------------
from mpc import MPCAgent
from utils import compute_perf_metrics_obst

num_eval_episodes = 10
render_eval = True
record_video = False

def evaluate(mpc_policy):
    """Roll out MPC in the real env (obstacle mode) and report metrics.

    Args:
        mpc_policy: MPCAgent callable obs -> action

    Returns:
        [all_metrics, rollouts, real_rollouts]
        where all_metrics include 'rew','succ','crash' (='grass') and obstacle counts like
        'chuck','chuck_passed' (and 'car','car_passed' when present) averaged across episodes.

    Notes:
        Uses globals num_eval_episodes, render_eval, record_video.
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
                CROP_H = 350  # Keep top 350px
                video_writer = cv2.VideoWriter(
                    f"chuckccar_{episode}.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (VIDEO_W, CROP_H)
                )

            real_rollout = []

            obs = env.reset()  # calling the real env reset()
            real_rollout.append(obs)
            obs = utils.process_frame(obs)

            # agent.reset()
            mpc_policy.reset()

            curr_ch_states = (np.zeros((1, env.rnn_size)), np.zeros((1, env.rnn_size)))  # cell/hidden - init with zeros
            done = False
            episode_reward = 0

            chuckholes_stepped = 0
            chuckholes_passed = 0
            cars_hit = 0
            cars_passed = 0

            rollout = []

            step = 0
            while not done:
                z = encoder.encode_frame(obs)

                if step % 50 == False:  # zero the memory every 50 steps
                    curr_ch_states = (np.zeros((1, env.rnn_size)), np.zeros((1, env.rnn_size)))

                # Construct the current full state
                current_full_state = np.concatenate((z, curr_ch_states[0][0, :], curr_ch_states[1][0, :]))

                action = mpc_policy(current_full_state)  # plan
                obs, reward, done, extra = env.step(action)  # take the action in the real env
                real_rollout.append(obs)
                obs = utils.process_frame(obs)

                episode_reward += reward

                if extra.get("chuck") is True:
                    chuckholes_stepped += 1
                if extra.get("chuck_passed") is True:
                    chuckholes_passed += 1
                if extra.get("car") is True:
                    cars_hit += 1
                if extra.get("car_passed") is True:
                    cars_passed += 1

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
            print("Chuckholes stepped", chuckholes_stepped)
            print("Chuckholes passed", chuckholes_passed)
            print("Cars hit", cars_hit)
            print("Cars passed", cars_passed)

            rollouts.append(rollout)
            real_rollouts.append(real_rollout)

            if record_video:
                video_writer.release
        average_episode_reward /= num_eval_episodes

        all_metrics = compute_perf_metrics_obst(rollouts)
        print("Average_episode_reward:", np.mean(all_metrics['rew']), '+/-', np.std(all_metrics['rew']))
        print("Average_success_rate (new tiles):", np.mean(all_metrics['succ']), '+/-', np.std(all_metrics['succ']))
        print("Average_grass_rate:", np.mean(all_metrics['crash']), '+/-', np.std(all_metrics['crash']))
        print("Average chuckholes stepped:", np.mean(all_metrics['chuck']), '+/-', np.std(all_metrics['chuck']))
        print("Average chuckholes passed:", np.mean(all_metrics['chuck_passed']),
              '+/-', np.std(all_metrics['chuck_passed']))
        print("Norm average chuckholes stepped:",
              np.mean([x / y for x, y in zip(all_metrics['chuck'], all_metrics['chuck_passed'])]),
              '+/-', np.std([x / y for x, y in zip(all_metrics['chuck'], all_metrics['chuck_passed'])]))

        if "car" in all_metrics:
            print("Average cars hit:", np.mean(all_metrics['car']), '+/-', np.std(all_metrics['car']))
            print("Average cars passed:", np.mean(all_metrics['car_passed']), '+/-', np.std(all_metrics['car_passed']))
            print("Norm average cars stepped:",
                  np.mean([x / y for x, y in zip(all_metrics['car'], all_metrics['car_passed'])]),
                  '+/-', np.std([x / y for x, y in zip(all_metrics['car'], all_metrics['car_passed'])]))

        return [all_metrics, rollouts, real_rollouts]

# create MPC policy
mpc_policy = MPCAgent(env, encoder, reward_model, dynamics_model, obst, plan_horizon=15, n_blind_steps=4,
                      use_random=False)

# evaluate
perf = evaluate(mpc_policy)
