"""Reward model from sparse reward labels (used in ReQueST and DROS).

Reimplemented in PyTorch and adapted with certain improvements from:
Reddy et al. (2020), "Learning human objectives by evaluating hypothetical behaviours"
"""

from tqdm import tqdm
import matplotlib.pyplot as plt
from models import PyTorchModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import os
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, accuracy_score


models_dir = os.path.join(os.getcwd(), 'models', 'carracing')
data_dir = os.path.join(os.getcwd(), 'data', 'carracing')


class RewardDataset(Dataset):
    """Dataset wrapper for sparse reward labels training.

    Stores obs, act, next_obs, reward and per-sample weights, with a mapping from
    raw rewards to class indices.
    """

    def __init__(self, data, idxes):
        """Initialise data and idxes, and map original rewards to class indices.

        Args:
            data: Dict containing 'obses', 'actions', 'next_obses', 'rews', 'weights', etc.
            idxes: Indices selecting the split (train or val)
        """
        self.data = data
        self.idxes = idxes
        self.target_mapping = {-1: 0, 0: 1, 10: 2}

    def __len__(self):
        """Return number of samples in the split."""
        return len(self.idxes)

    def __getitem__(self, index):
        """Return a single sample as tensors.

        Args:
            index: Integer index within this split

        Returns:
            Dict with 'obses', 'act', 'next_obses', 'target', 'weights'
        """
        idx = self.idxes[index]
        mapped_target = self.target_mapping[self.data['rews'][idx]]
        return {
            'obses': torch.tensor(self.data['obses'][idx], dtype=torch.float32),
            'act': torch.tensor(self.data['actions'][idx], dtype=torch.float32),
            'next_obses': torch.tensor(self.data['next_obses'][idx], dtype=torch.float32),
            'target': torch.tensor(mapped_target, dtype=torch.long),
            'weights': torch.tensor(self.data['weights'][idx], dtype=torch.float32)
        }

    @classmethod
    def bal_weights(cls, rews, train_idxes, val_idxes):
        """Compute per-sample and per-class weights from training distribution.

        Args:
            rews: Array of raw rewards (class labels)
            train_idxes: Indices for training split
            val_idxes: Indices for validation split

        Returns:
            weights: Per-sample weights aligned with rews
            class_weights: Per-class weights for loss
        """

        # Initialise weights array with 0.0 for all entries
        weights = np.zeros_like(rews, dtype=float)

        # Extract training rewards
        train_rews = rews[train_idxes]

        # Compute weight for each class based on training rewards
        unique_classes = np.unique(train_rews)
        class_sample_counts = np.array([np.sum(train_rews == cls) for cls in unique_classes])

        weight_per_class = 1. / class_sample_counts

        # Map weights to all data points based on their class
        for cls, weight in zip(unique_classes, weight_per_class):
            weights[rews == cls] = weight

        # Set validation weights to 1 (or neutral)
        weights[val_idxes] = 1

        # Create class weights for loss function - ensuring the order matches model's output
        # Assuming model outputs classes in the order of [-1, 0, 10]
        expected_class_order = [-1, 0, 10]

        # Create a mapping from class label to weight
        weight_mapping = {cls: np.max(class_sample_counts) / count for cls, count
                          in zip(unique_classes, class_sample_counts)}

        # Order weights according to expected_class_order
        class_weights = np.array([weight_mapping[cls] for cls in expected_class_order if cls in weight_mapping])

        return weights, class_weights


class ResidualBlock(nn.Module):
    """Two-layer MLP block with residual connection and optional dropout.

    Args:
        input_size: Input feature size
        layer_size: Hidden/output size
        dropout_rate: Dropout probability or None
    """

    def __init__(self, input_size, layer_size, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_size, layer_size)
        self.relu = nn.ReLU()
        self.dropout_rate = dropout_rate
        if self.dropout_rate: self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(layer_size, layer_size)

        if input_size != layer_size:
            self.residual_adjust = nn.Linear(input_size, layer_size)
        else:
            self.residual_adjust = None

    def forward(self, x):
        """Apply residual MLP block.

        Args:
            x: Tensor [..., input_size]

        Returns:
            Tensor [..., layer_size]
        """
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        if self.dropout_rate:
            out = self.dropout(out)
        out = self.fc2(out)

        # Adjust the residual if necessary
        if self.residual_adjust is not None:
            residual = self.residual_adjust(residual)

        out += residual
        out = self.relu(out)
        return out


class RewardModelSparse(PyTorchModel):
    """Ensemble reward classifier trained on sparse labels.

    Supports different input choices (s, sa, s', ss', sas'), residual MLPs,
    and class-balanced training.

    Args:
        env: Env carrying n_obs_dim, n_act_dim, rew_classes
        n_rew_nets_in_ensemble: Number of models in the ensemble
        n_layers: Number of MLP layers (per model)
        layer_size: Hidden size
        rew_func_input: One of {'s','sa',"s'","ss'","sas'"}
        dropout_rate: Dropout probability or None
        residual: If True, use residual blocks
    """

    def __init__(self, env, n_rew_nets_in_ensemble=4, n_layers=1, layer_size=64, rew_func_input="s'",
                 dropout_rate=None, residual=True):
        super().__init__(env)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_keys = ['obses', 'actions', 'next_obses', 'rews']

        self.n_rew_nets_in_ensemble = n_rew_nets_in_ensemble
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.rew_func_input = rew_func_input
        self.dropout_rate = dropout_rate
        self.residual = residual

        self.reward_networks = nn.ModuleList([
            self.build_raw_net() for _ in range(n_rew_nets_in_ensemble)
        ])

        self.to(self.device)

    def build_raw_net(self):
        """Build a single reward network according to config.

        Returns:
            nn.Module mapping chosen input to logits over classes
        """
        layers = []
        input_size = self.env.n_obs_dim
        if self.rew_func_input == 'sa':
            input_size += self.env.n_act_dim
        elif self.rew_func_input == "ss'":
            input_size += self.env.n_obs_dim
        elif self.rew_func_input == "sas'":
            input_size += self.env.n_act_dim + self.env.n_obs_dim

        if self.residual:
            for _ in range(self.n_layers):
                layers.append(ResidualBlock(input_size, self.layer_size, self.dropout_rate))
                input_size = self.layer_size
            layers.append(nn.Linear(self.layer_size, len(self.env.rew_classes)))
        else:
            for _ in range(self.n_layers):
                layers.append(nn.Linear(input_size, self.layer_size))
                layers.append(nn.ReLU())
                if self.dropout_rate: layers.append(nn.Dropout(self.dropout_rate))
                input_size = self.layer_size
            layers.append(nn.Linear(self.layer_size, len(self.env.rew_classes)))

        return nn.Sequential(*layers)

    def forward(self, obs, act, next_obs):
        """Run ensemble forward pass and return per-model logits.

        Args:
            obs: Tensor [B, n_obs_dim]
            act: Tensor [B, n_act_dim]
            next_obs: Tensor [B, n_obs_dim]

        Returns:
            List of length ensemble_size with logits [B, num_classes]
        """
        inputs = {
            's': obs,
            'sa': torch.cat([obs, act], dim=1),
            "s'": next_obs,
            "ss'": torch.cat([obs, next_obs], dim=1),
            "sas'": torch.cat([obs, act, next_obs], dim=1)
        }
        x = inputs[self.rew_func_input]
        outputs = [net(x) for net in self.reward_networks]
        return outputs

    def custom_loss_function(self, outputs, targets, weights, class_weights, weighted_loss=None):
        """Cross-entropy loss with optional weighting.

        Args:
            outputs: Tensor [B, ensemble, num_classes] of logits
            targets: Tensor [B] class indices
            weights: Tensor [B] per-sample weights
            class_weights: Tensor [num_classes] per-class weights
            weighted_loss: None, 'max', or 'inverse' to choose scheme

        Returns:
            Scalar loss tensor
        """
        averaged_outputs = outputs.mean(dim=1)
        if weighted_loss == 'max':
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            loss = criterion(averaged_outputs, targets)
        elif weighted_loss == 'inverse':
            criterion = nn.CrossEntropyLoss(reduction='none')
            losses = criterion(averaged_outputs, targets)
            weighted_losses = losses * weights
            loss = weighted_losses.mean()
        else:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(averaged_outputs, targets)
        return loss

    def format_batch(self, batch):
        """Move batch dict to the current device.

        Args:
            batch: Dict of tensors from DataLoader

        Returns:
            Dict with tensors on self.device
        """
        batch_tensors = {key: value.to(self.device) for key, value in batch.items()}
        return batch_tensors

    def initialize_optimizer(self, learning_rate=1e-3, weight_decay=1e-6):
        """Create Adam optimiser if not already initialised.

        Args:
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def learn(self, data, epochs=400, learning_rate=1e-3, weight_decay=1e-6, ftol=1e-6,
              batch_size=32, val_update_freq=1, verbose=True, rew_name="model"):
        """Train on sparse reward labels.

        Args:
            data: Dict with arrays for 'obses','actions','next_obses','rews' and split idxes
            epochs: Max epochs
            learning_rate: Adam learning rate
            weight_decay: L2 regularization
            ftol: For early stop (not used here)
            batch_size: Training batch size
            val_update_freq: Validate every N epochs
            verbose: If True, print progress
            rew_name: Label for plots/checkpoints

        Returns:
            None
        """
        data['weights'], data['class_weights'] = RewardDataset.bal_weights(data['rews'],
                                                                           data['train_idxes'], data['val_idxes'])

        train_dataset = RewardDataset(data, data['train_idxes'])
        val_dataset = RewardDataset(data, data['val_idxes'])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
        print("Number of batches/iterations in training dataset:", len(train_loader))

        self.train()
        self.initialize_optimizer(learning_rate, weight_decay)

        val_avg_losses = []
        val_avg_accs = []

        best_val_loss = None
        # best_val_loss = self.eval_model(data, val_loader)

        for epoch in range(epochs):
            losses = []
            accs = []
            train_preds, train_targets = [], []

            # train_loader = tqdm(train_loader)

            for batch in train_loader:

                self.optimizer.zero_grad()

                batch_tensors = self.format_batch(batch)

                outputs = self.forward(batch_tensors['obses'], batch_tensors['act'], batch_tensors['next_obses'])

                loss = self.custom_loss_function(torch.stack(outputs, dim=1), batch_tensors['target'],
                                                 batch_tensors['weights'],
                                                 torch.tensor(data['class_weights'],
                                                              dtype=torch.float32).to(self.device))
                acc = accuracy(torch.stack(outputs, dim=1).mean(dim=1), batch_tensors['target'])

                losses.append(loss.item())
                accs.append(acc.item())

                loss.backward()
                self.optimizer.step()

                predictions = torch.stack(outputs, dim=1).mean(dim=1).argmax(dim=1)  # torch.max(outputs, dim=1)
                train_preds.extend(predictions.cpu().numpy())
                train_targets.extend(batch_tensors['target'].cpu().numpy())

                # train_loader.set_description(f"Epoch {epoch}/{epochs}")
                # train_loader.set_postfix(loss=np.mean(losses), epoch=epoch)
            if epoch % val_update_freq == 0:
                self.eval()
                total_val_loss = 0
                total_val_acc = 0
                total_samples = 0
                val_preds, val_targets = [], []

                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch_tensors = self.format_batch(val_batch)
                        outputs = self.forward(val_batch_tensors['obses'], val_batch_tensors['act'],
                                               val_batch_tensors['next_obses'])

                        # probabilities = [F.softmax(output, dim=1) for output in outputs]

                        batch_size = val_batch[self.data_keys[0]].size(0)
                        total_samples += batch_size

                        val_loss = self.custom_loss_function(torch.stack(outputs, dim=1), val_batch_tensors['target'],
                                                             val_batch_tensors['weights'],
                                                             torch.tensor(data['class_weights'],
                                                                          dtype=torch.float32).to(self.device))
                        val_acc = accuracy(torch.stack(outputs, dim=1).mean(dim=1), val_batch_tensors['target'])

                        total_val_loss += val_loss.item() * batch_size
                        total_val_acc += val_acc.item() * batch_size

                        predictions = torch.stack(outputs, dim=1).mean(dim=1).argmax(dim=1)
                        val_preds.extend(predictions.cpu().numpy())
                        val_targets.extend(val_batch_tensors['target'].cpu().numpy())

                    val_avg_loss = total_val_loss / total_samples
                    val_avg_acc = total_val_acc / total_samples

                    val_avg_losses.append(val_avg_loss)
                    val_avg_accs.append(val_avg_acc)

                    if verbose:
                        print(f"Epoch {epoch}/{epochs}, Train loss: {np.mean(losses)}, Validation Loss: {val_avg_loss},"
                              f" Train Acc: {np.mean(accs)}, Validation Acc: {val_avg_acc},"
                              f" Learning Rate: {self.optimizer.param_groups[0]['lr']}")

                    if best_val_loss is None or val_avg_loss < best_val_loss:
                        best_val_loss = val_avg_loss
                        self.save(os.path.join(models_dir, rew_name + '.pt'))

                    # if self.converged(val_avg_losses, ftol):
                    #   if verbose:
                    #     print("Convergence criteria met")
                    #   break

                # scheduler.step(val_loss)
                self.train()

            # if self.converged(val_avg_losses, ftol):
            #   break
        if verbose:
            plt.figure()
            epochs_x = range(0, epochs, val_update_freq)
            plt.plot(epochs_x, val_avg_losses)
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.title('Reward Validation Loss Over Time')
            plt.xticks(epochs_x)
            plt.show(block=False)
            plt.pause(0.1)
            plt.savefig('loss_' + rew_name + '.png')
            plt.close()
        if verbose:
            plt.figure()
            epochs_x = range(0, epochs, val_update_freq)
            plt.plot(epochs_x, val_avg_accs)
            plt.xlabel('Epoch')
            plt.ylabel('Validation Accuracy')
            plt.suptitle('Reward Validation Accuracy Over Time')
            plt.xticks(epochs_x)
            plt.show(block=False)
            plt.pause(0.1)
            plt.savefig('acc_' + rew_name + '.png')
            plt.close()

    def evaluate(self, data, batch_size=1000):
        """Evaluate on the validation split.

        Args:
            data: Dict with arrays and split idxes
            batch_size: Eval batch size

        Returns:
            None
        """
        data['weights'], data['class_weights'] = RewardDataset.bal_weights(data['rews'], data['train_idxes'],
                                                                           data['val_idxes'])

        val_dataset = RewardDataset(data, data['val_idxes'])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.eval_model(data, val_loader)

    def eval_model(self, data, val_loader):
        """Compute validation metrics.

        Args:
            data: Original data dict (for rew_classes etc.)
            val_loader: DataLoader for validation split

        Returns:
            None
        """
        self.eval()
        total_val_loss = 0
        total_val_acc = 0
        total_samples = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for val_batch in val_loader:
                val_batch_tensors = self.format_batch(val_batch)
                outputs = self.forward(val_batch_tensors['obses'], val_batch_tensors['act'],
                                       val_batch_tensors['next_obses'])

                batch_size = val_batch[self.data_keys[0]].size(0)
                total_samples += batch_size

                val_loss = self.custom_loss_function(torch.stack(outputs, dim=1), val_batch_tensors['target'],
                                                     val_batch_tensors['weights'],
                                                     torch.tensor(data['class_weights'],
                                                                  dtype=torch.float32).to(self.device))
                val_acc = accuracy(torch.stack(outputs, dim=1).mean(dim=1), val_batch_tensors['target'])

                total_val_loss += val_loss.item() * batch_size
                total_val_acc += val_acc.item() * batch_size

                predictions = torch.stack(outputs, dim=1).mean(dim=1).argmax(dim=1)
                val_preds.extend(predictions.cpu().numpy())
                val_targets.extend(val_batch_tensors['target'].cpu().numpy())

            val_avg_loss = total_val_loss / total_samples
            val_avg_acc = total_val_acc / total_samples

            val_confusion = confusion_matrix(val_targets, val_preds)
            val_class_report = classification_report(val_targets, val_preds)
            val_f1 = f1_score(val_targets, val_preds, average='weighted')
            val_recall = recall_score(val_targets, val_preds, average='weighted')
            val_accuracy = accuracy_score(val_targets, val_preds)

            print(f"Validation Loss: {val_avg_loss}, Validation Acc: {val_avg_acc}")

            print("Validation Confusion Matrix:\n", val_confusion)
            print("Validation Classification Report:\n", val_class_report)
            print("Validation F1 Score:", val_f1)
            print("Validation Recall:", val_recall)
            print("Validation Accuracy:", val_accuracy)

            return val_avg_loss

    def get_reward(self, prev_obs, act, curr_obs):
        """Predict a single-step reward class and map to scalar reward.

        Args:
            prev_obs: Array/Tensor [n_obs_dim]
            act: Array/Tensor [n_act_dim]
            curr_obs: Array/Tensor [n_obs_dim]

        Returns:
            Scalar reward from env.rew_classes
        """
        self.eval()
        with torch.no_grad():
            prev_obs = torch.tensor(prev_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            act = torch.tensor(act, dtype=torch.float32).unsqueeze(0).to(self.device)
            curr_obs = torch.tensor(curr_obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            outputs = self.forward(prev_obs, act, curr_obs)
            averaged_output = torch.stack(outputs, dim=1).mean(dim=1)  # Average the outputs from the ensemble
            predicted_class = torch.argmax(averaged_output, dim=1).item()  # Get the predicted class

        reward = self.env.rew_classes[predicted_class]
        return reward

    def get_batch_reward(self, prev_obs, act, curr_obs):
        """Predict rewards for a batch, and map rewards to classes, and classes constant rewards.

        Args:
            prev_obs: Array/Tensor [B, n_obs_dim]
            act: Array/Tensor [B, n_act_dim]
            curr_obs: Array/Tensor [B, n_obs_dim]

        Returns:
            Numpy array [B] of rewards
        """
        self.eval()
        with torch.no_grad():
            prev_obs = torch.tensor(prev_obs, dtype=torch.float32).to(self.device)
            act = torch.tensor(act, dtype=torch.float32).to(self.device)
            curr_obs = torch.tensor(curr_obs, dtype=torch.float32).to(self.device)

            outputs = self.forward(prev_obs, act, curr_obs)
            averaged_output = torch.stack(outputs, dim=1).mean(dim=1)
            predicted_classes = torch.argmax(averaged_output, dim=1)

        rewards = [self.env.rew_classes[pred_class] for pred_class in predicted_classes]
        return np.array(rewards)

    def get_batch_reward_opt(self, prev_obs, act, curr_obs, order='soft_av'):
        """Differentiable reward from ensemble logits.

        Args:
            prev_obs: Tensor [B, n_obs_dim]
            act: Tensor [B, n_act_dim]
            curr_obs: Tensor [B, n_obs_dim]
            order: 'soft_av' (default) or 'av_soft' for combining probabilities

        Returns:
            Tensor [B] of rewards
        """
        self.eval()

        prev_obs = prev_obs.to(self.device) # requires_grad=True (except 1st obs)
        act = act.to(self.device)
        curr_obs = curr_obs.to(self.device)

        outputs = self.forward(prev_obs, act, curr_obs)

        reward_weights = torch.tensor(self.env.rew_classes, device=self.device)
        if order == 'av_soft':
            averaged_output = torch.stack(outputs, dim=1).mean(dim=1)
            # Apply softmax-weighted average of class rewards
            # torch.argmax is a non-differentiable operation, meaning it doesn’t support gradient computation
            class_probs = torch.softmax(averaged_output, dim=1)
            rewards = torch.sum(class_probs * reward_weights, dim=1)
        elif 'soft_av':
            # Apply softmax-weighted reward calculation to each model in the ensemble first
            individual_rewards = [
                torch.sum(torch.softmax(output, dim=1) * reward_weights, dim=1)
                for output in outputs
            ]

            # Average the rewards across the ensemble
            rewards = torch.stack(individual_rewards, dim=1).mean(dim=1)

        return rewards

    def get_uncertainty(self, prev_obs, act, curr_obs):
        """Estimate epistemic uncertainty via ensemble KL divergence.

        Args:
            prev_obs: Tensor [B, n_obs_dim]
            act: Tensor [B, n_act_dim]
            curr_obs: Tensor [B, n_obs_dim]

        Returns:
            Tensor [B] of uncertainty scores
        """
        self.eval()

        prev_obs = prev_obs.to(self.device)
        act = act.to(self.device)
        curr_obs = curr_obs.to(self.device)

        outputs = self.forward(prev_obs, act, curr_obs)

        probs = [F.softmax(output, dim=1) for output in outputs]

        probs = torch.stack(probs, dim=1)

        mean_probs = probs.mean(dim=1, keepdim=True)

        kl_divs = probs * (torch.log(probs + 1e-9) - torch.log(mean_probs + 1e-9))
        kl_divs = kl_divs.sum(dim=2)

        uncertainty = kl_divs.mean(dim=1)

        return uncertainty

    def create_sketches_queries(self, dream_user_rollouts, size_segment=50, num_queries=90):
        """Extract fixed-length segments from dream rollouts for GUI queries.

        Args:
            dream_user_rollouts: List of trajectories of (s, a, r, s', ...)
            size_segment: Segment length
            num_queries: Number of segments to sample

        Returns:
            Numpy array [num_queries, size_segment, ds+da] of segments
        """
        inputs_dream = []
        for rollout in dream_user_rollouts:
            inputs_dream_rollout = []
            for step in rollout:
                observation, action, reward, *rest = step
                input_dream = np.concatenate((observation, action))
                inputs_dream_rollout.append(input_dream)
            inputs_dream_rollout = np.array(inputs_dream_rollout).reshape(1000, -1)
            inputs_dream.append(inputs_dream_rollout)

        len_traj, num_trajs = len(inputs_dream[0]), len(inputs_dream)
        train_inputs = np.array(inputs_dream)

        batch_index = np.random.choice(num_trajs, size=num_queries, replace=True)

        sa_t = train_inputs[batch_index]

        sa_t = sa_t.reshape(-1, sa_t.shape[-1])

        time_index = np.array([list(range(i * len_traj, i * len_traj + size_segment)) for i in range(num_queries)])
        time_index = (time_index + np.random.choice(len_traj - size_segment, size=num_queries, replace=True)
                      .reshape(-1, 1))

        sa_t = np.take(sa_t, time_index, axis=0)  # Batch x size_seg x dim of s&a

        return sa_t

def accuracy(outputs, labels):
    """Compute accuracy.

    Args:
        outputs: Logits [B, num_classes]
        labels: Class indices [B]

    Returns:
        Scalar tensor accuracy in [0,1]
    """
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).float()
    acc = correct.sum() / len(labels)
    return acc

def synth_sketch(traj, act_seq, true_reward_model):
    """Create synthetic sketch tuples from a trajectory and a true reward model.

    Args:
        traj: Sequence of latent states
        act_seq: Sequence of actions (aligned with traj[:-1])
        true_reward_model: Callable with get_batch_reward(...)

    Returns:
        List of (s, a, r, s_next, None, None) tuples
    """
    sketch = true_reward_model.get_batch_reward(traj[:-1], act_seq, traj[1:])
    return [(s, a, r, ns, None, None)
            for s, a, r, ns in zip(traj[:-1], act_seq, sketch, traj[1:])]
