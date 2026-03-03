"""MDN-RNN (Dynamics model) for latent transitions (STEP 1).

Reimplemented in PyTorch and adapted with certain improvements from:
- Ha and Schmidhuber (2018), "World Models"
- Reddy et al. (2020), "Learning human objectives by evaluating hypothetical behaviours"
"""

from models import PyTorchModel
import utils
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from utils import CustomDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_value_
import os
import pickle


class MDNRNNDynamicsModel(PyTorchModel):
    """LSTM + Mixture Density Network in latent space.

    Models next latent state given current latent state and action. Supports optional
    layer norm and dropout, and provides methods for single-step prediction
    and rollout encoding.

    Args:
        env: Environment with n_z_dim, n_act_dim, rnn_size
        grad_clip: Clip value for gradients (None to disable)
        num_mixture: Number of Gaussian components in the MDN
        use_layer_norm: Apply layer normalisation on LSTM outputs
        use_input_dropout: Apply dropout to inputs
        input_dropout_prob: Dropout probability for inputs
        use_output_dropout: Apply dropout to outputs
        output_dropout_prob: Dropout probability for outputs

    Notes:
        Uses nn.LSTMCell (instead of nn.LSTM) to store the cell state (c) at each step.
    """

    def __init__(self, env, grad_clip=1.0, num_mixture=5, use_layer_norm=False, use_input_dropout=False,
                 input_dropout_prob=0.90, use_output_dropout=False, output_dropout_prob=0.90):
        super().__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_keys = ['obses', 'logvars', 'actions', 'traj_lens']

        self.grad_clip = grad_clip
        self.num_mixture = num_mixture
        self.rnn_size = self.env.rnn_size

        rnn_input_size = self.env.n_z_dim + self.env.n_act_dim
        #self.lstm = nn.LSTM(input_size=rnn_input_size, hidden_size=self.rnn_size, num_layers=1, batch_first=True)
        self.lstm_cell = nn.LSTMCell(input_size=rnn_input_size, hidden_size=self.rnn_size)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.rnn_size)
        self.input_dropout = nn.Dropout(p=input_dropout_prob) if use_input_dropout else None
        self.output_dropout = nn.Dropout(p=output_dropout_prob) if use_output_dropout else None

        # Define output layers for MDN
        NOUT = self.env.n_z_dim * self.num_mixture * 3  # For mix, mean, and logstd
        self.output_w = nn.Linear(self.rnn_size, NOUT)

        self.to(self.device)

    def preproc_rollouts(self, encoder, raw_rollout_data):
        """Encode raw rollouts to latent space and pack tensors.

        Args:
            encoder: Trained VAE encoder that maps obs to (z, logvar)
            raw_rollout_data: Dict of numpy arrays for obses, next_obses

        Returns:
            Dict with encoded arrays obses, next_obses, logvars, next_logvars, etc.
        """
        rollout_data = deepcopy(raw_rollout_data)
        with torch.no_grad():
            for prefix in ['', 'next_']:
                raw_obses = raw_rollout_data['%sobses' % prefix]

                flat_raw_obses = raw_obses.reshape(
                    (raw_obses.shape[0] * raw_obses.shape[1], raw_obses.shape[2],
                     raw_obses.shape[3], raw_obses.shape[4]))

                mus = []
                logvars = []

                chunk_size = 1000
                n_chunks = int(np.ceil(flat_raw_obses.shape[0] / chunk_size))
                for i in range(n_chunks):
                    flat_raw_obses_ = torch.tensor(flat_raw_obses[i * chunk_size:(i + 1) * chunk_size],
                                                   dtype=torch.float).to(self.device)
                    flat_raw_obses_ = encoder.squash_obs(flat_raw_obses_)
                    flat_raw_obses_ = utils.perm_tf2pt(flat_raw_obses_)

                    more_mus, more_logvars = encoder.encode(flat_raw_obses_)
                    mus.extend(more_mus)
                    logvars.extend(more_logvars)

                mus_cpu_numpy = [tensor.cpu().detach().numpy() for tensor in mus]
                mus = np.array(mus_cpu_numpy)
                logvars_cpu_numpy = [tensor.cpu().detach().numpy() for tensor in logvars]
                logvars = np.array(logvars_cpu_numpy)

                rollout_data['%sobses' % prefix] = mus.reshape(
                    (raw_obses.shape[0], raw_obses.shape[1], mus.shape[1]))
                rollout_data['%slogvars' % prefix] = logvars.reshape(
                    (raw_obses.shape[0], raw_obses.shape[1], logvars.shape[1]))

        return rollout_data

    def run_lstmcell(self, input_x, seq_lens, initial_state):
        """Run LSTMCell over variable-length sequences.

        Args:
            input_x: Tensor of shape [B, T, n_z_dim + n_act_dim]
            seq_lens: 1D tensor of per-sequence lengths
            initial_state: Tuple (c, h) or None

        Returns:
            outputs: Tensor [B, T, rnn_size] of hidden states (h)
            c_states: Tensor [B, T, rnn_size] of cell states (c)
            h_n: Final hidden state [B, rnn_size]
            c_n: Final cell state [B, rnn_size]
        """
        batch_size, max_seq_len, _ = input_x.size()

        seq_lens, sort_indices = seq_lens.sort(descending=True)
        input_x = input_x[sort_indices]

        if initial_state is None:
            h, c = (torch.zeros(batch_size, self.rnn_size, device=input_x.device),
                    torch.zeros(batch_size, self.rnn_size, device=input_x.device))
        else: # e.g. in next_obs()
            c, h = initial_state
            if h.shape[0]>1: # more than 1 trajectory
                h = h[sort_indices, :]
                c = c[sort_indices, :]


        outputs = []
        c_states = []

        for t in range(max_seq_len):
            active_batch_size = sum([l > t for l in seq_lens]) if seq_lens.shape[0]>1 else 1
            # Compute updates for the active part of the batch
            h_update, c_update = self.lstm_cell(
                input_x[:active_batch_size, t, :], (h[:active_batch_size], c[:active_batch_size]))

            h = torch.cat((h_update, h[active_batch_size:]), dim=0)
            c = torch.cat((c_update, c[active_batch_size:]), dim=0)

            # Store the current hidden state for all sequences
            outputs.append(h.unsqueeze(1).clone())
            c_states.append(c.unsqueeze(1).clone())

        # Concatenate outputs to match expected shape
        outputs = torch.cat(outputs, dim=1)
        c_states = torch.cat(c_states, dim=1)

        # Unsort the batch to restore original order
        _, unsort_indices = sort_indices.sort()
        outputs = outputs[unsort_indices]
        c_states = c_states[unsort_indices]
        h_n = outputs[:,-1,:]
        c_n = c_states[:,-1,:]

        return outputs, c_states, h_n, c_n

    def forward(self, input_x, seq_lens=None, initial_state=None):
        """Compute LSTM states and MDN parameters for next-latent prediction.

        Args:
            input_x: Tensor [B, T, n_z_dim + n_act_dim]
            seq_lens: 1D tensor of sequence lengths
            initial_state: Tuple (c, h) or None

        Returns:
            c_states: Tensor [B, T, rnn_size]
            h_states: Tensor [B, T, rnn_size]
            last_state: Tuple (c_T, h_T) with final states
            logmix: Tensor [B*T*n_z_dim, num_mixture]
            mean: Tensor [B*T*n_z_dim, num_mixture]
            logstd: Tensor [B*T*n_z_dim, num_mixture]
        """
        if self.input_dropout:
            input_x = self.input_dropout(input_x)

        # this is slow, but necessary if you want to get the c_states for all steps.
        outputs, c_states, h_n, c_n = self.run_lstmcell(input_x, seq_lens-1, initial_state) # outputs are h_states

        if self.use_layer_norm:
            outputs = self.layer_norm(outputs)

        if self.output_dropout:
            outputs = self.output_dropout(outputs)

        h_states = outputs

        outputs = outputs.reshape(-1, self.rnn_size)
        outputs = self.output_w(outputs)
        outputs = outputs.reshape(-1, self.num_mixture * 3)
        out_logmix, out_mean, out_logstd = torch.split(outputs, self.num_mixture, dim=1)
        out_logmix = out_logmix - torch.logsumexp(out_logmix, dim=1, keepdim=True)

        return (c_states,
            h_states,
            (c_n, h_n),
            out_logmix,
            out_mean,
            out_logstd)

    def mdn_loss_function(self, logmix, mean, logstd, y):
        """Negative log-likelihood of targets y under the MDN outputs.

        Args:
            logmix: Mixture logits [B*T*n_z_dim, num_mixture]
            mean: Component means [B*T*n_z_dim, num_mixture]
            logstd: Component log-stds [B*T*n_z_dim, num_mixture]
            y: Target next latents [B, T, n_z_dim]

        Returns:
            Scalar loss tensor
        """
        n_trajs, traj_len, _ = y.shape

        shape = [n_trajs, traj_len, self.env.n_z_dim, self.num_mixture]
        mean = mean.reshape(shape)
        logmix = logmix.reshape(shape)
        logstd = logstd.reshape(shape)
        y = torch.unsqueeze(y, 3)

        def torch_lognormal(y, mean, logstd):
            log_sqrt_two_pi = torch.log(torch.sqrt(torch.tensor(2.0 * np.pi)))
            return -0.5 * ((y - mean) / torch.exp(logstd)) ** 2 - logstd - log_sqrt_two_pi

        v = logmix + torch_lognormal(y, mean, logstd) # Combine log probabilities with log mixture coefficients
        v = torch.logsumexp(v, dim=3, keepdim=False) # Sum over mixtures (next observation if we had 1 sample only)

        # 3 averages overall
        v = torch.mean(v, dim=2, keepdim=False) # Average over feature dimension
        v = torch.mean(v, dim=1, keepdim=False)  # Average over sequence length
        v = torch.mean(v)  # scalar - Average over batch_size
        loss = -v
        return loss

    def format_batch(self, batch):
        """Prepare a DataLoader batch for the model.

        Args:
            batch: Dict of tensors from the dataset

        Returns:
            Dict with 'inputs', 'outputs', 'traj_lens' ready for forward()
        """
        batch_tensors = {key: value.to(self.device) for key, value in batch.items()}

        # Generate stochastic samples 'z'
        eps = torch.randn_like(batch_tensors['logvars'])
        z = batch_tensors['obses'] + torch.exp(batch_tensors['logvars'] / 2) * eps

        # Offset inputs/outputs by one timestep
        batch_tensors['inputs'] = torch.cat((z[:, :-1, :], batch_tensors['actions'][:, :-1, :]), dim=2)
        batch_tensors['outputs'] = z[:, 1:, :]
        batch_tensors['traj_lens'] -= 1  # Adjust trajectory lengths due to the offset

        return batch_tensors

    def learn(self, encoder, raw_rollout_data, epochs=400, learning_rate=1e-3, ftol=1e-6,
              batch_size=32, val_update_freq=1, verbose=True):
        """Train the MDN-RNN on encoded rollouts.

        Args:
            encoder: VAE encoder used to map observations to z
            raw_rollout_data: Dict with raw rollouts and splits
            epochs: Max epochs
            learning_rate: Adam learning rate
            ftol: For early stop (not used here)
            batch_size: Training batch size
            val_update_freq: Validate every N epochs
            verbose: If True, print progress
        """

        # Preprocess rollouts (encode them to latent space)
        rollout_data = self.preproc_rollouts(encoder, raw_rollout_data)

        # Prepare data loaders
        train_dataset = CustomDataset(rollout_data, self.data_keys, rollout_data['train_idxes'])
        val_dataset = CustomDataset(rollout_data, self.data_keys, rollout_data['val_idxes'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(rollout_data['val_idxes']), shuffle=False)
        print("Number of batches/iterations in training dataset:", len(train_loader))

        self.train()
        self.initialize_optimizer(learning_rate)

        val_avg_losses = []
        best_val_loss = None


        for epoch in range(epochs):
            losses = []
            train_loader = tqdm(train_loader)
            for batch in train_loader:
                self.optimizer.zero_grad()
                batch_tensors = self.format_batch(batch)
                c_states,  h_states, _, out_logmix, out_mean, out_logstd = self.forward(batch_tensors['inputs'],
                                                                                        batch_tensors['traj_lens'])
                loss = self.mdn_loss_function(out_logmix, out_mean, out_logstd, batch_tensors['outputs'])
                loss.backward()

                # Gradient clipping
                if self.grad_clip:
                    clip_grad_value_(self.parameters(), self.grad_clip)

                self.optimizer.step()
                #scheduler.step()

                losses.append(loss.item())

                #train_loader.set_description(f"Epoch {epoch}/{epochs}")
                train_loader.set_postfix(loss=np.mean(losses), epoch=epoch)
            if epoch % val_update_freq == 0:
                self.eval()
                total_val_loss = 0
                total_samples = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch_tensors = self.format_batch(val_batch)
                        c_states, h_states, _, out_logmix, out_mean, out_logstd = self.forward(
                            val_batch_tensors['inputs'], val_batch_tensors['traj_lens'])
                        val_loss = self.mdn_loss_function(out_logmix, out_mean, out_logstd,
                                                          val_batch_tensors['outputs'])
                        batch_size = val_batch[self.data_keys[0]].size(0)
                        total_val_loss += val_loss.item() * batch_size
                        total_samples += batch_size

                    val_avg_loss = total_val_loss / total_samples
                    val_avg_losses.append(val_avg_loss)
                    if verbose:
                        print(f"Epoch {epoch}/{epochs}, Train loss: {np.mean(losses)}, Validation Loss: {val_avg_loss}"
                              f", Learning Rate: {self.optimizer.param_groups[0]['lr']}")

                    if best_val_loss is None or val_avg_loss < best_val_loss:
                        best_val_loss = val_avg_loss
                        # self.save(os.path.join(os.getcwd(), 'dyn_gcT_ep600_ch15L_epochs20_lrs0_0001.pt'))

                    # if self.converged(val_avg_losses, ftol):
                    #     if verbose:
                    #         print("Convergence criteria met")
                    #     break

                #scheduler.step(val_loss)
                self.train()

            # if self.converged(val_avg_losses, ftol):
            #     break

        if verbose:
            plt.figure()
            epochs_x = range(0, epochs, val_update_freq)
            plt.plot(epochs_x, val_avg_losses)
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.title('MDN-RNN Validation Loss Over Time')
            plt.xticks(epochs_x)
            plt.show()

    def evaluate(self, encoder, val_raw_rollout_data, batch_size=1000):
        """Compute validation loss on held-out rollouts.

        Args:
            encoder: VAE encoder
            val_raw_rollout_data: Dict of raw rollouts for validation
            batch_size: Eval batch size

        Returns:
            Mean validation loss (float)
        """

        # Preprocess rollouts (encode them to latent space (32,))
        val_rollout_data = self.preproc_rollouts(encoder, val_raw_rollout_data)

        val_dataset = CustomDataset(val_rollout_data, self.data_keys, val_rollout_data['val_idxes'])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.eval()
        total_val_loss = 0
        total_samples = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch_tensors = self.format_batch(val_batch)
                c_states, h_states, _, out_logmix, out_mean, out_logstd = self.forward(val_batch_tensors['inputs'],
                                                                                       val_batch_tensors['traj_lens'])
                val_loss = self.mdn_loss_function(out_logmix, out_mean, out_logstd, val_batch_tensors['outputs'])
                batch_size = val_batch[self.data_keys[0]].size(0)  # 9
                total_val_loss += val_loss.item() * batch_size
                total_samples += batch_size

            val_avg_loss = total_val_loss / total_samples

            print(f"Val/Test Loss: {val_avg_loss}")

    def rnn_encode_rollouts(self, rollout_data):
        """Attach RNN c/h states to each encoded rollout.

        Args:
            rollout_data: Dict with obses, actions, next_obses, traj_lens, etc.

        Returns:
            Dict where 'obses' and 'next_obses' are concatenated with c/h states
        """
        input_x = np.concatenate((rollout_data['obses'], rollout_data['actions']), axis=2)
        input_x = torch.tensor(input_x, dtype=torch.float).to(self.device)
        seq_lens = rollout_data['traj_lens']
        seq_lens = torch.tensor(seq_lens, dtype=torch.float).to(self.device)

        self.eval()
        with torch.no_grad():
            c_states, h_states, _, _, _, _ = self.forward(input_x, seq_lens)
        c_states = c_states.cpu().numpy()
        h_states = h_states.cpu().numpy()


        ch_states = np.concatenate((c_states, h_states), axis=2)
        ch_states = np.concatenate((np.zeros((ch_states.shape[0], 1, ch_states.shape[2])),
                                    ch_states),
                                       axis=1)

        data = deepcopy(rollout_data)

        data['obses'] = np.zeros((data['obses'].shape[0], data['obses'].shape[1],
                                  self.env.n_z_dim + 2 * self.env.rnn_size))

        data['next_obses'] = np.zeros(
            (data['next_obses'].shape[0], data['next_obses'].shape[1],
             self.env.n_z_dim + 2 * self.env.rnn_size))

        for rollout_idx in range(data['obses'].shape[0]):
          data['obses'][rollout_idx] = np.concatenate(
              (rollout_data['obses'][rollout_idx], ch_states[rollout_idx, :-1]),
              axis=1)

          data['next_obses'][rollout_idx] = np.concatenate(
              (rollout_data['next_obses'][rollout_idx], ch_states[rollout_idx, 1:]), axis=1)

        return data

    def next_obs(self, obs, act, init_state, temperature=1.0, mixact_seq=None, sample=True):
        """Predict next latent and next RNN state for one step.

        Args:
            obs: Current latent observation
            act: Action at the current step
            init_state: Tuple (c, h) for the RNN
            temperature: Sampling temperature
            mixact_seq: Optional mix/action coefficients (not used)
            sample: If True, sample a mode; else use mean

        Returns:
            next_obs_pred: Next latent
            next_state_pred: Tuple (c_next, h_next)

        Notes:
            Uses MDN to either sample or take the mixture mean.
        """
        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        act = torch.tensor(act, dtype=torch.float).to(self.device)
        if init_state is not None:
            init_state = (torch.tensor(init_state[0], dtype=torch.float).to(self.device),
                          torch.tensor(init_state[1], dtype=torch.float).to(self.device))

        obs = obs.unsqueeze(1) # (batch=n_trajs, seq_len, latent_dim) set another dim for the seq_len
        act = act.unsqueeze(1)

        input_x = torch.cat([obs, act], dim=-1)  # Concatenate obs and actions along the feature dimension
        seq_lens = torch.tensor(input_x.shape[1] + 1, dtype=torch.float).unsqueeze(0).to(self.device)
        _, _, last_state, out_logmix, out_mean, out_logstd = self.forward(input_x, seq_lens, initial_state=init_state)

        if not sample:
            traj_len = 1
            n_trajs = obs.shape[0]
            shape = [n_trajs, traj_len, self.env.n_z_dim, self.num_mixture]

            mean = torch.reshape(out_mean, shape)

            def f(logmix):
                """Numerically stable softmax over mixture logits.

                Args:
                    logmix: Logits tensor

                Returns:
                    Normalised mixture coefficients
                """
                logmix = torch.reshape(logmix, shape)
                logmix = logmix - torch.logsumexp(logmix, dim=3, keepdim=True)
                return torch.exp(logmix)

            if mixact_seq is not None:
              mix_coeffs = f(mixact_seq)
            else:
              mix_coeffs = f(out_logmix / temperature)

            mixed_means = torch.sum(mix_coeffs * mean, dim=3)

            next_obs_pred = mixed_means[:, -1, :]
        else: # as in WME, sample and use only one of the K modes of the MDN
            # adjust temperatures
            logmix2 = out_logmix / temperature # the bigger the temperature, the closer the mixture coefficients come

            # print(logmix2.max())
            logmix2 -= logmix2.max()
            logmix2 = torch.exp(logmix2)
            logmix2 /= logmix2.sum(axis=1).reshape(self.env.n_z_dim, 1)  # Normalisation (they become probabilities)

            mixture_idx = torch.zeros(self.env.n_z_dim).to(self.device)
            chosen_mean = torch.zeros(self.env.n_z_dim).to(self.device)
            chosen_logstd = torch.zeros(self.env.n_z_dim).to(self.device)
            for j in range(self.env.n_z_dim):
                idx = utils.get_pi_idx(torch.rand(1).to(self.device), logmix2[j]) # choose one of the 5 modes
                mixture_idx[j] = idx
                chosen_mean[j] = out_mean[j][idx]
                chosen_logstd[j] = out_logstd[j][idx]

            # the bigger the temperature, the wider range of the chosen mode can be
            rand_gaussian = (torch.randn(self.env.n_z_dim).to(self.device) *
                             torch.sqrt(torch.tensor(temperature).to(self.device)))
            next_x = chosen_mean + torch.exp(chosen_logstd).to(self.device) * rand_gaussian

            next_z = next_x.reshape(self.env.n_z_dim)

            next_obs_pred = next_z.reshape(1, self.env.n_z_dim)

        next_obs_pred = next_obs_pred.cpu().numpy()
        next_state_pred = (last_state[0].cpu().numpy(), last_state[1].cpu().numpy())

        next_obs_pred = np.concatenate((next_obs_pred, next_state_pred[0], next_state_pred[1]), axis=1)  # latent|c|h

        return next_obs_pred, next_state_pred

    def next_obs_gpu(self, obs, act, init_state=None, temperature=1.0, mixact_seq=None, sample=True):
        """GPU variant of next_obs.

        Args:
            obs: Current latent observation
            act: Action at the current step
            init_state: Tuple (c, h)
            temperature: Sampling temperature
            mixact_seq: Optional mix/action coefficients
            sample: If True, sample; else use mean

        Returns:
            next_obs_pred: Next latent
            next_state_pred: Tuple (c_next, h_next)
        """
        obs = obs.unsqueeze(1)  # (batch=n_trajs, seq_len, latent_dim) set another dim for the seq_len
        act = act.unsqueeze(1)

        input_x = torch.cat([obs, act], dim=-1)  # Concatenate observations and actions along feature dimension
        seq_lens = torch.tensor(input_x.shape[1] + 1, dtype=torch.float).unsqueeze(0).to(self.device)
        _, _, last_state, out_logmix, out_mean, out_logstd = self.forward(input_x, seq_lens, initial_state=init_state)

        if not sample:
            traj_len = 1
            n_trajs = obs.shape[0]
            shape = [n_trajs, traj_len, self.env.n_z_dim, self.num_mixture]

            mean = torch.reshape(out_mean, shape)

            def f(logmix):
                """Numerically stable softmax over mixture logits."""
                logmix = torch.reshape(logmix, shape)
                logmix = logmix - torch.logsumexp(logmix, dim=3, keepdim=True)
                return torch.exp(logmix)

            if mixact_seq is not None:
              mix_coeffs = f(mixact_seq)
            else:
              mix_coeffs = f(out_logmix / temperature)

            mixed_means = torch.sum(mix_coeffs * mean, dim=3)

            next_obs_pred = mixed_means[:, -1, :]
        else: # as in WME, sample and use only one of the K modes of the MDN
            # adjust temperatures
            logmix2 = out_logmix / temperature # the bigger the temperature, the closer the mixture coefficients come

            # print(logmix2.max())
            logmix2 -= logmix2.max()
            logmix2 = torch.exp(logmix2)
            logmix2 /= logmix2.sum(axis=1).reshape(self.env.n_z_dim, 1)  # Normalisation (they become probabilities)

            mixture_idx = torch.zeros(self.env.n_z_dim).to(self.device)
            chosen_mean = torch.zeros(self.env.n_z_dim).to(self.device)
            chosen_logstd = torch.zeros(self.env.n_z_dim).to(self.device)
            for j in range(self.env.n_z_dim):
                idx = utils.get_pi_idx(torch.rand(1).to(self.device), logmix2[j])  # scalar # choose one of the 5 modes
                mixture_idx[j] = idx
                chosen_mean[j] = out_mean[j][idx]
                chosen_logstd[j] = out_logstd[j][idx]

            # the bigger the temperature, the wider range of the chosen mode can be
            rand_gaussian = (torch.randn(self.env.n_z_dim).to(self.device) *
                             torch.sqrt(torch.tensor(temperature).to(self.device)))
            next_x = chosen_mean + torch.exp(chosen_logstd).to(self.device) * rand_gaussian

            next_z = next_x.reshape(self.env.n_z_dim)

            next_obs_pred = next_z.reshape(1, self.env.n_z_dim)

        next_state_pred = (last_state[0], last_state[1])

        return next_obs_pred, next_state_pred

    def enc_traj_of_act_seq(self,
                          init_obs,
                          act_seq,
                          traj_len,
                          init_state=None):
        """Generate encoded trajectory (z|c|h) for an action sequence.

        Args:
            init_obs: Initial latent observation
            act_seq: Tensor of actions
            traj_len: Number of steps
            init_state: (c, h) tensors or None

        Returns:
            traj: Numpy array with fully-enc trajectory [T, z|c|h]
        """
        obs = init_obs
        ch_states = init_state

        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        act_seq = torch.tensor(act_seq, dtype=torch.float).to(self.device)
        if ch_states is not None:
            ch_states = (torch.tensor(ch_states[0], dtype=torch.float).to(self.device),
                          torch.tensor(ch_states[1], dtype=torch.float).to(self.device))

        traj = [torch.cat((obs, ch_states[0], ch_states[1]), dim=1)]

        for t in range(traj_len - 1):
            obs, ch_states = self.next_obs_gpu(
                obs,
                act_seq[:, t, :],
                init_state=ch_states,
                temperature=0.1,
                sample=False
            )
            traj.append(torch.cat((obs, ch_states[0], ch_states[1]), dim=1))

        traj = torch.stack(traj).squeeze(1).cpu().numpy()

        return traj

    def enc_traj_of_act_seq_keepgrads(self,
                          init_obs,
                          act_seq,
                          traj_len,
                          init_state=None):
        """Like enc_traj_of_act_seq, but keeps gradients in PyTorch.

        Args:
            init_obs: Initial latent observation (tensor)
            act_seq: Tensor of actions
            traj_len: Number of steps
            init_state: (c, h) tensors or None

        Returns:
            traj: Tensor with fully-enc trajectory [T, z|c|h]
        """
        obs = init_obs
        ch_states = init_state

        obs = obs.to(self.device)
        act_seq = act_seq.to(self.device)
        if ch_states is not None:
            ch_states = (ch_states[0].to(self.device),
                          ch_states[1].to(self.device))

        traj = [torch.cat((obs, ch_states[0], ch_states[1]), dim=1)]

        for t in range(traj_len - 1):
            obs, ch_states = self.next_obs_gpu(
                obs,
                act_seq[:, t, :],
                init_state=ch_states,
                temperature=0.1,
                sample=False
            )
            traj.append(torch.cat((obs, ch_states[0], ch_states[1]), dim=1))

        traj = torch.stack(traj).squeeze(1)

        return traj


    def traj_of_act_seq(self,
                          init_obs,
                          act_seq,
                          traj_len,
                          init_state=None):
        """Generate latent-only trajectory for an action sequence.

        Args:
            init_obs: Initial latent observation
            act_seq: Sequence of actions
            traj_len: Number of steps
            init_state: Optional (c, h)

        Returns:
            Numpy array of latents z over time
        """
        obs = init_obs
        ch_states = init_state

        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        act_seq = torch.tensor(act_seq, dtype=torch.float).to(self.device)
        if ch_states is not None:
            ch_states = (torch.tensor(ch_states[0], dtype=torch.float).to(self.device),
                          torch.tensor(ch_states[1], dtype=torch.float).to(self.device))

        traj = [obs]

        for t in range(traj_len - 1):
            obs, ch_states = self.next_obs_gpu(
                obs,
                act_seq[:, t, :],
                init_state=ch_states,
                temperature=0.1,
                sample=False
            )

            traj.append(obs)

        traj = torch.stack(traj).squeeze(1).unsqueeze(0).cpu().numpy()

        return traj

    def trajs_of_act_seqs(self,
                          init_obs,
                          act_seq,
                          traj_len,
                          init_state=None):
        """Generate multiple latent trajectories for multiple action sequences.

        Args:
            init_obs: Initial latent observation
            act_seqs: List/array of action sequences
            traj_len: Number of steps
            init_state: Optional (c, h)

        Returns:
            List or array of latent trajectories
        """
        obs = init_obs
        ch_states = init_state

        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        act_seq = torch.tensor(act_seq, dtype=torch.float).to(self.device)
        if ch_states is not None:
            ch_states = (torch.tensor(ch_states[0], dtype=torch.float).to(self.device),
                          torch.tensor(ch_states[1], dtype=torch.float).to(self.device))

        traj = [obs]

        for t in range(traj_len - 1):
            obs, ch_states = self.next_obs_gpu(
                obs,
                act_seq[:, t, :],
                init_state=ch_states,
                temperature=0.1,
                sample=False
            )

            traj.append(obs)

        traj = torch.stack(traj).squeeze(1).unsqueeze(0).cpu().numpy()

        return traj

    def rnn_encode_traj(self, traj, act_seq, init_state=None):
        """Compute RNN c/h for a given latent trajectory and actions.

        Args:
            traj: Latent trajectory [T, n_z_dim]
            act_seq: Action sequence [T-1, n_act_dim]
            init_state: Optional (c, h)

        Returns:
            Numpy arrays with c/h over time and concatenated outputs as used elsewhere
        """
        if init_state is not None:
            torch_init_state = (torch.tensor(init_state[0], dtype=torch.float).to(self.device),
                          torch.tensor(init_state[1], dtype=torch.float).to(self.device))

        input_x = np.concatenate((traj[:,:-1,:], act_seq), axis=2)
        input_x = torch.tensor(input_x, dtype=torch.float).to(self.device)
        seq_lens = torch.tensor(input_x.shape[1] + 1, dtype=torch.float).unsqueeze(0).to(self.device)

        self.eval()
        with torch.no_grad():
            h_states, c_states, _, _ = self.run_lstmcell(input_x, seq_lens - 1, initial_state=torch_init_state)
            # c_states, h_states, _, _, _, _ = self.forward(input_x, seq_lens, initial_state=torch_init_state)

        c_states = c_states.cpu().numpy()
        h_states = h_states.cpu().numpy()

        c_states = c_states[0]
        h_states = h_states[0]

        c_states = np.concatenate([init_state[0], c_states], axis=0)
        h_states = np.concatenate([init_state[1], h_states], axis=0)

        ch_states = np.concatenate([c_states, h_states], axis=1)

        init_full_obs = np.concatenate([traj[0, 0][np.newaxis,:], ch_states[:1, :]], axis=1)

        suffixes = np.concatenate([traj[0, 1:], ch_states[1:, :]], axis=1)

        return np.concatenate([init_full_obs, suffixes], axis=0)
