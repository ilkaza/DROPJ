"""Reward model from human preferences (and justifications) for STEP 3.

Some parts are adapted from: Lee et al. (2021), "PEBBLE: Feedback-Efficient Interactive
Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training".

Trains in one-shot, so buffer fills only once. Simulated teacher is not used.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.animation import FuncAnimation

device = 'cuda'


def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    """Build a simple MLP.

    Args:
        in_dim: Input size
        out_dim: Output size
        hidden_dims: List of hidden layer sizes
        activation: 'tanh' or 'relu'

    Returns:
        nn.Sequential MLP
    """
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())
    return net


class RewardModelPref():
    """Ensemble reward model trained from preferences (and justifications).

    Maintains segment pairs and trains an ensemble of MLPs to predict rewards
    such that preferred segments have higher cumulative reward.

    Args:
        ds: State feature dimension
        da: Action feature dimension
        ensemble_size: Number of models in the ensemble
        size_segment: Segment length (timesteps) for each query
        activation: Nonlinearity for the MLPs ('tanh' or 'relu')
        lr: Optimiser learning rate
        train_batch_size: Batch size for reward training
        capacity: Max number of stored queries
    """

    def __init__(self, ds, da,
                 ensemble_size=3, size_segment=50, activation='tanh',
                 lr=3e-4, train_batch_size=128, capacity=1e5):

        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.activation = activation
        self.size_segment = size_segment

        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
        self.train_batch_size = train_batch_size

        self.construct_ensemble()

    def construct_ensemble(self):
        """Create the ensemble and optimiser.

        Returns:
            None
        """
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds + self.da,
                                           out_size=1, H=256, n_layers=3,
                                           activation=self.activation)).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    def load(self, model_dir, note):
        """Load ensemble weights from disk.

        Args:
            model_dir: Directory containing saved states
            note: Suffix used in filenames
        """
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, member, note))
            )

    def learn_with_GUI(self, queries, user_responses, reward_update, use_justifications, w_s=1, w_def=0.75,
                       use_equal_in_plain_prefs=False):
        """Train the reward model from GUI-labeled queries.

        Args:
            queries: Dict mapping from video filename to the two segments
            user_responses: List with GUI responses for each query
            reward_update: Number of training epochs
            use_justifications: True for DROPJ, False for DROP and DROPe
            w_s: Safety justification weight (when only one segment is safe)
            w_def: Default justification weight (when both are safe and a preference is given)
            use_equal_in_plain_prefs: False for DROP, True for DROPe (allow 'Equally')

        Notes:
            Plots loss (and accuracy when prefs-only) for quick inspection.
        """
        labeled_queries = self.sampling_gui(queries, user_responses, use_justifications, w_s, w_def,
                                            use_equal_in_plain_prefs)

        losses = []
        accuracies = []
        for epoch in range(reward_update):
            if use_justifications:
                train_loss = self.train_soft_reward_gui(use_justifications)
                total_loss = np.mean(train_loss)
                print(f"Epoch {epoch + 1}/{reward_update}, Loss: {total_loss:.4f}")
                losses.append(total_loss)
            else:  # only prefs
                train_acc, train_loss = self.train_soft_reward_gui(use_justifications, use_equal_in_plain_prefs)
                total_loss = np.mean(train_loss)
                total_acc = np.mean(train_acc)
                print(f"Epoch {epoch + 1}/{reward_update}, Loss: {total_loss:.4f}, Accuracy: {total_acc:.4f}")
                losses.append(total_loss)
                accuracies.append(total_acc)

            # if total_acc > 0.97: # possibly was used to avoid overfitting
            #     break;

        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()

        if accuracies:
            plt.subplot(1, 2, 2)
            plt.plot(accuracies, label='Total Accuracy', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training Accuracy over Epochs')
            plt.legend()

        plt.tight_layout()
        plt.show()
        #plt.savefig('training_loss_accuracy.png')

        if use_justifications:
            print("Reward function is updated!! LOSS: " + str(total_loss))
        else:
            print("Reward function is updated!! LOSS: " + str(total_loss))
            print("Reward function is updated!! ACC: " + str(total_acc))

    def learn_with_GUI_multi_justs(self, queries, user_responses, reward_update, just_weights):
        """Train from GUI labels with multiple justification types.

        Args:
            queries: Dict mapping from video filename to the two segments
            user_responses: List with GUI responses for each query
            reward_update: Number of training epochs
            just_weights: Dict mapping justification names to their corresponding weights

        Returns:
            None
        """
        labeled_queries = self.sampling_gui_multi_justs(queries, user_responses, just_weights)

        losses = []
        for epoch in range(reward_update):
            train_loss = self.train_soft_reward_gui(use_justifications=True)
            total_loss = np.mean(train_loss)
            print(f"Epoch {epoch + 1}/{reward_update}, Loss: {total_loss:.4f}")
            losses.append(total_loss)

            # if total_acc > 0.97:
            #     break;

        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()

        print("Reward function is updated!! LOSS: " + str(total_loss))

    def sampling_gui(self, queries, user_responses, use_justifications, w_s, w_def, use_equal_in_plain_prefs):
        """Convert GUI responses into numeric labels and fill the buffer.

        Args:
            queries: Dict mapping from video filename to the two segments
            user_responses: List with GUI responses for each query
            use_justifications: True for DROPJ, False for DROP and DROPe
            w_s: Safety justification weight (when only one segment is safe)
            w_def: Default justification weight (when both are safe and a preference is given)
            use_equal_in_plain_prefs: False for DROP, True for DROPe (allow 'Equally')

        Returns:
            Number of labeled queries added
        """
        sa_t_1 = []
        sa_t_2 = []
        labels = []
        skipped_count = 0

        for response in user_responses:
            video = response['video']
            if response['responses'] == 'Skipped':
                skipped_count +=1
                continue

            sa_t_1.append(queries[video]['segment1'])
            sa_t_2.append(queries[video]['segment2'])

            if use_justifications:
                q1 = response['responses'][0][1]
                q2 = response['responses'][1][1]
                preference = response['responses'][2][1] if len(response['responses']) > 2 else None

                if q1 == 'Yes' and q2 == 'No':
                    labels.append(w_s)
                elif q1 == 'No' and q2 == 'Yes':
                    labels.append(1.0 - w_s)
                elif q1 == 'Yes' and q2 == 'Yes':
                    if preference == 'Left': # Left is first
                        labels.append(w_def)
                    elif preference == 'Right': # Right is second
                        labels.append(1.0 - w_def)
                    elif preference == 'Equally':
                        labels.append(0.5)
                elif q1 == 'No' and q2 == 'No':
                        labels.append(0.5)
                else:
                    raise NotImplementedError()
            else: # plain preferences
                preference = response['responses'][0][1] # 'Left'/'Right'
                if preference == 'Left':  # Left is first
                    labels.append(1.0)
                elif preference == 'Right':  # Right is second
                    labels.append(0.0)
                elif use_equal_in_plain_prefs and preference == 'Equally':
                    labels.append(0.5)
                else:
                    raise NotImplementedError()

        sa_t_1 = np.stack(sa_t_1)
        sa_t_2 = np.stack(sa_t_2)
        labels = np.array(labels).reshape(-1, 1)

        unique_values, counts = np.unique(labels, return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Value {value} appears {count} times")
        print("Skipped", skipped_count, "times")

        self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def sampling_gui_multi_justs(self, queries, user_responses, just_weights):
        """Convert GUI responses with multiple justifications into numeric labels and fill the buffer.

        Args:
            queries: Dict mapping from video filename to the two segments
            user_responses: List with GUI responses for each query
            just_weights: Dict mapping justification names to their corresponding weights

        Returns:
            Number of labeled queries added
        """
        sa_t_1 = []
        sa_t_2 = []
        labels = []
        skipped_count = 0

        for response in user_responses:
            video = response['video']
            if response['responses'] == 'Skipped':
                skipped_count +=1
                continue  # Skip adding this query

            sa_t_1.append(queries[video]['segment1'])
            sa_t_2.append(queries[video]['segment2'])

            justification = response['responses'][0][1]
            preference = response['responses'][1][1]

            if justification in just_weights:
                if preference == 'Left':
                    labels.append(just_weights[justification])
                elif preference == 'Right':
                    labels.append(1.0 - just_weights[justification])
                elif preference == 'Equally':
                    labels.append(0.5)

        sa_t_1 = np.stack(sa_t_1)
        sa_t_2 = np.stack(sa_t_2)
        labels = np.array(labels).reshape(-1, 1)

        unique_values, counts = np.unique(labels, return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Value {value} appears {count} times")

        self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def put_queries(self, sa_t_1, sa_t_2, labels):
        """Append labeled segment pairs to the buffer.

        Args:
            sa_t_1: Array [N, size_segment, ds+da] for first segments
            sa_t_2: Array [N, size_segment, ds+da] for second segments
            labels: Array [N, 1] with preference weights in [0,1]

        Returns:
            None
        """
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index

    def train_soft_reward_gui(self, use_justifications, use_equal_in_plain_prefs=False):
        """One epoch of reward training on the buffered segment pairs.

        Args:
            use_justifications: True for DROPJ, False for DROP and DROPe
            use_equal_in_plain_prefs: False for DROP, True for DROPe (allow 'Equally')

        Returns:
            If use_justifications: array of loss values per ensemble member for inspection
            Else: arrays of (approximate) accuracy, and loss per ensemble member for inspection
        """
        ensemble_loss = np.array([0.0 for _ in range(self.de)])
        if not use_justifications:
            ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = torch.tensor(0.0, device=device)

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs] # obs are from the env range and actions are scaled in [-1, 1]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).to(device)  # (batch_size,) tensor

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1) # summing the r's (logits) of a segment
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], dim=-1) # (batch_size, 2)

                # Compute loss (3 different ways to do it - all should return the same or very close value)

                # (C) # reverse order
                target_onehot = torch.zeros_like(r_hat)
                target_onehot[:, 0] = labels
                target_onehot[:, 1] = 1 - labels
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)

                # (A) reversed the order - now 1 in labels means that the first is preferred
                logits_diff = r_hat[:, 0] - r_hat[:, 1]
                bce_loss = nn.BCEWithLogitsLoss()
                labelsA = labels.float()
                curr_lossA = bce_loss(logits_diff, labelsA)

                # (B) reversed the order - now 1 in labels means that the first is preferred
                probs = torch.softmax(r_hat, dim=1)
                pi0_prob = probs[:, 0]
                bce_loss = nn.BCELoss()
                labelsB = labels.float()
                curr_lossB = bce_loss(pi0_prob, labelsB)

                # Assert that the difference between any two computed losses does not exceed 0.001
                assert abs(curr_loss.item() - curr_lossA.item()) <= 0.001, "|curr_loss - curr_lossA| > 0.001"
                # assert abs(curr_loss.item() - curr_lossB.item()) <= 0.001, "|curr_loss - curr_lossB| > 0.001"
                # assert abs(curr_lossA.item() - curr_lossB.item()) <= 0.001, "|curr_lossA - curr_lossB| > 0.001"

                loss += curr_loss
                ensemble_loss[member] += curr_loss.item() * len(r_hat)

                if not use_justifications:
                    if use_equal_in_plain_prefs:
                        equal_mask = torch.abs(probs[:, 0] - probs[:, 1]) <= 0.3
                        negative_one = torch.tensor(-1, device=device)
                        predicted = torch.where(equal_mask, negative_one, torch.argmax(probs, dim=1))
                        labels_updated = torch.where(labels == 0.5, negative_one, (1 - labels).long())
                        correct = (predicted == labels_updated).sum().item()
                    else:
                        _, predicted = torch.max(r_hat.data, 1)
                        correct = (predicted == (1-labels)).sum().item() # if label is 0, then 2nd one is preferred
                    ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        if not use_justifications:
            ensemble_acc = ensemble_acc / total
        ensemble_loss = ensemble_loss / total

        if not use_justifications:
            return ensemble_acc, ensemble_loss
        else:
            return ensemble_loss

    def r_hat_member(self, x, member=-1):
        """Compute per-timestep reward predictions for one ensemble member.

        Args:
            x: Tensor [B, size_segment, ds+da] segment
            member: Index of the ensemble model

        Returns:
            Tensor [B, size_segment, 1] of predicted rewards
        """
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    def softXEnt_loss(self, logits, target):
        """Two-class soft cross-entropy loss helper.

        Args:
            logits: Tensor [B, 2] of preference scores
            target: Tensor [B, 2] of preference targets

        Returns:
            Scalar loss (mean over batch)
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim=1) # convert logits to log-probabilities
        return -(target * logprobs).sum() / logits.shape[0]

    def save(self, model_dir, note=None):
        """Save ensemble weights from disk.

        Args:
            model_dir: Directory containing saved states
            note: Suffix used in filenames
        """
        for member in range(self.de):
            torch.save(self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, member, note))

    def r_hat(self, x):
        """Compute per-timestep reward predictions, averaged over the ensemble.

        Args:
            x: Array [size_segment, ds+da]

        Returns:
            Numpy array [size_segment] of mean predicted rewards (averaged across ensemble)

        Notes:
            Calls r_hat_member(...) for each model, stacks to [E, size_segment, 1],
            averages over ensemble (axis=0), then squeezes the last dim.
        """
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.squeeze(np.mean(r_hats, axis=0))

    def create_pref_queries(self, env, encoder, dream_user_rollouts_scaled, num_queries=600, save_path=None):
        """Sample random segment pairs from dream trajectories and save preference queries in .mp4 and .pkl.

        Args:
            dream_user_rollouts_scaled: List of dream trajectories
            num_queries: Number of query pairs to create
            env: Environment for rendering (used in videos creation)
            encoder: VAE to decode latents for rendering
            save_path: Directory to save clips and data

        Notes:
            Uncomment the line np.random.seed(None) to create random segments in each run
        """
        inputs_dream = []
        for rollout in dream_user_rollouts_scaled:
            inputs_dream_rollout = []
            for step in rollout:
                observation, action, reward, *rest = step
                input_dream = np.concatenate((observation, action))
                inputs_dream_rollout.append(input_dream)
            inputs_dream_rollout = np.array(inputs_dream_rollout).reshape(1000, -1)
            inputs_dream.append(inputs_dream_rollout)

        len_traj, num_trajs = len(inputs_dream[0]), len(inputs_dream)
        train_inputs = np.array(inputs_dream)

        # np.random.seed(None)
        batch_index_1 = np.random.choice(num_trajs, size=num_queries, replace=True)
        batch_index_2 = np.random.choice(num_trajs, size=num_queries, replace=True)

        sa_t_1 = train_inputs[batch_index_1]
        sa_t_2 = train_inputs[batch_index_2]

        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1])

        # Generate time index
        time_index = np.array([list(range(i * len_traj, i * len_traj + self.size_segment)) for i in range(num_queries)])
        time_index_1 = (time_index +
                        np.random.choice(len_traj - self.size_segment, size=num_queries, replace=True).reshape(-1, 1))
        time_index_2 = (time_index +
                        np.random.choice(len_traj - self.size_segment, size=num_queries, replace=True).reshape(-1, 1))

        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)  # Batch x size_seg x dim of s&a
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0)  # Batch x size_seg x dim of s&a

        self.save_query_segments_data(sa_t_1, sa_t_2, env, encoder, index=0,
                                           save_path=os.path.join(os.getcwd(), save_path))

    def save_query_segments_data(self, segments1, segments2, env, encoder, index, save_path=None):
        """Render and serialise query segment pairs for the GUI.

        Args:
            segments1: Array of first segments [num_queries, size_segment, ds+da]
            segments2: Array of second segments [num_queries, size_segment, ds+da]
            env: Environment used for rendering
            encoder: VAE to decode latents for rendering
            index: Starting index for filenames
            save_path: Directory to save videos and data.pkl (if not None)

        Notes:
            Writes videos as mp4 and a pickled dict mapping filenames to segments.

            Opens a preview window per clip. These windows auto-close, but it may take
            several seconds, so many windows will briefly appear. If that’s a problem,
            generate fewer queries per batch and adjust `index` to continue from the last
            saved filename. To avoid repeating the same segments across runs, call
            'np.random.seed(None)' inside 'create_pref_queries()' before sampling.
        """
        data_dict = {}

        for i, (segment1, segment2) in enumerate(zip(segments1, segments2)):
            original_segment1 = segment1
            original_segment2 = segment2

            segment1 = segment1[:, :env.n_z_dim]
            frames1 = encoder.decode_batch_latents(segment1)

            segment2 = segment2[:, :env.n_z_dim]
            frames2 = encoder.decode_batch_latents(segment2)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].axis('off')
            axes[1].axis('off')

            im1 = axes[0].imshow(frames1[0])
            im2 = axes[1].imshow(frames2[0])

            def update(frame):
                im1.set_data(frames1[frame])
                im2.set_data(frames2[frame])
                return im1, im2

            anim = FuncAnimation(fig, update, frames=range(self.size_segment), interval=400, blit=True)

            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                file_name = f'{index + i}.mp4'
                anim.save(os.path.join(save_path, file_name), writer='ffmpeg')

            plt.close(fig)

            data_dict[file_name] = {
                'segment1': original_segment1,
                'segment2': original_segment2
            }

        if save_path is not None:
            with open(os.path.join(save_path, 'data.pkl'), 'wb') as f:
                pickle.dump(data_dict, f)
