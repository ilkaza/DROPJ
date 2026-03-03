"""Variational Autoencoder (encoder) for STEP 1.

Reimplemented in PyTorch and adapted with certain improvements from:
- Ha & Schmidhuber (2018), "World Models"
- Reddy et al. (2020), "Learning human objectives by evaluating hypothetical behaviours"
"""

import os
import numpy as np
from models import PyTorchModel
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import CustomDataset
from torch.utils.data import DataLoader


class EncoderModel(PyTorchModel):
    """Utilities around the VAE to encode/decode frames and batches."""
    def encode_frame(self, obs):
        """Encode a single RGB frame to its latent mean.

        Args:
            obs: HxWx3 array or tensor in [0,1]

        Returns:
            1D numpy array latent mean (z)
        """
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device).unsqueeze(0)
        else:
            obs_tensor = obs.to(self.device).unsqueeze(0)
        mu, _ = self.encode(utils.perm_tf2pt(self.squash_obs(obs_tensor)))
        mu = mu[0, :].cpu().numpy()
        return mu

    def encode_batch_frames(self, obses):
        """Encode a batch of frames to latent means.

        Args:
            obses: NxHxWx3 array or tensor in [0,1]

        Returns:
            NxZ numpy array of latent means
        """
        if isinstance(obses, np.ndarray):
            obs_tensor = torch.tensor(obses, dtype=torch.float).to(self.device)
        else:
            obs_tensor = obses.to(self.device)
        mu, _ = self.encode(utils.perm_tf2pt(self.squash_obs(obs_tensor)))
        mu = mu.cpu().numpy()
        return mu

    def decode_latent(self, latent):
        """Decode a single latent vector to an RGB frame.

        Args:
            latent: 1D array/tensor (z)

        Returns:
            HxWx3 numpy array in [0,1]
        """
        if isinstance(latent, np.ndarray):
            latent = torch.tensor(latent, dtype=torch.float).to(self.device).unsqueeze(0)
            recon = self.decode(latent)[0, :, :, :]
        else:
            recon = self.decode(latent.unsqueeze(0))[0, :, :, :]
        recon =  recon.cpu().numpy().transpose(1, 2, 0)
        return recon

    def decode_batch_latents(self, latents):
        """Decode a batch of latent vectors to RGB frames.

        Args:
            latents: NxZ array/tensor

        Returns:
            NxHxWx3 numpy array in [0,1]
        """
        if isinstance(latents, np.ndarray):
            latents = torch.tensor(latents, dtype=torch.float).to(self.device)
        else:
            latents = latents.to(self.device)
        recons = self.decode(latents).cpu().numpy().transpose(0, 2, 3, 1)
        return recons

    def squash_obs(self, obs):
        """Ensure obs are float tensors in [0,1].

        Args:
            obs: Tensor in RGB

        Returns:
            Tensor in [0,1]
        """
        obs_max = obs.max()
        obs_norm = 255. if obs_max > 1. else 1.
        return obs / obs_norm


class VAEModel(EncoderModel):
    """Convolutional VAE encoder/decoder for CarRacing frames.

    Supports multiple size presets (M/L/XL/res128) and channel multipliers.
    Provides helpers for training, evaluation, and validation printing.

    Args:
        env: Object providing n_z_dim and image size info
        size: Determines the number of conv layers (One of {'M','L','XL','res128'})
             'res128' is  for Obstacle Car Racing, when the original image resolution is 128x128
        ch: Channel multiplier preset for 'res128' (one of {'sesquialterate', 'double', 'quadruple'})

    Notes:
        size='res128' is given for Obstacle Car Racing, when the original image resolution is 128x128.
    """

    def __init__(self, env, kl_tolerance=0.5, size='L', ch='sesquialterate'):
        super().__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kl_tolerance = kl_tolerance
        self.size = size
        self.ch = ch
        self.data_keys = ['obses']

        if self.size=='M':
            self.enc_conv1 = nn.Conv2d(3, 48, 4, stride=2)
            self.enc_conv2 = nn.Conv2d(48, 96, 4, stride=2)
            self.enc_conv3 = nn.Conv2d(96, 192, 4, stride=2)
            self.enc_conv4 = nn.Conv2d(192, 384, 4, stride=4)
            self.fc_mu = nn.Linear(2*2*384, self.env.n_z_dim)
            self.fc_log_var = nn.Linear(2*2*384, self.env.n_z_dim)

            self.dec_fc = nn.Linear(self.env.n_z_dim, 2*2*384)
            self.dec_deconv1 = nn.ConvTranspose2d(2*2*384, 192, 5, stride=2)
            self.dec_deconv2 = nn.ConvTranspose2d(192, 96, 5, stride=2)
            self.dec_deconv3 = nn.ConvTranspose2d(96, 48, 6, stride=2)
            self.dec_deconv4 = nn.ConvTranspose2d(48, 3, 26, stride=2)
        elif self.size=='L':
            if self.ch=='sesquialterate':
                ch1 = 24
            elif self.ch=='double':
                ch1 = 32
            self.enc_conv1 = nn.Conv2d(3, ch1, 4, stride=2)
            self.enc_conv2 = nn.Conv2d(ch1, 2*ch1, 4, stride=2)
            self.enc_conv3 = nn.Conv2d(2*ch1, 4*ch1, 4, stride=2)
            self.enc_conv4 = nn.Conv2d(4*ch1, 8*ch1, 4, stride=1)
            self.enc_conv5 = nn.Conv2d(8*ch1, 16*ch1, 4, stride=1)
            self.fc_mu = nn.Linear(2*2*16*ch1, self.env.n_z_dim)
            self.fc_log_var = nn.Linear(2*2*16*ch1, self.env.n_z_dim)

            self.dec_fc = nn.Linear(self.env.n_z_dim, 2*2*16*ch1)
            self.dec_deconv1 = nn.ConvTranspose2d(16*ch1, 8*ch1, 4, stride=1)
            self.dec_deconv2 = nn.ConvTranspose2d(8*ch1, 4*ch1, 4, stride=1)
            self.dec_deconv3 = nn.ConvTranspose2d(4*ch1, 2*ch1, 5, stride=2)
            self.dec_deconv4 = nn.ConvTranspose2d(2*ch1, ch1, 5, stride=2)
            self.dec_deconv5 = nn.ConvTranspose2d(ch1, 3, 4, stride=2)
        elif self.size=='XL':
            self.enc_conv1 = nn.Conv2d(3, 16, 5, stride=2)
            self.enc_conv2 = nn.Conv2d(16, 32, 4, stride=2)
            self.enc_conv3 = nn.Conv2d(32, 64, 4, stride=2)
            self.enc_conv4 = nn.Conv2d(64, 128, 3, stride=1)
            self.enc_conv5 = nn.Conv2d(128, 256, 3, stride=1)
            self.enc_conv6 = nn.Conv2d(256, 512, 3, stride=1)
            self.fc_mu = nn.Linear(2*2*512, self.env.n_z_dim)
            self.fc_log_var = nn.Linear(2*2*512, self.env.n_z_dim)

            self.dec_fc = nn.Linear(self.env.n_z_dim, 2*2*512)
            self.dec_deconv1 = nn.ConvTranspose2d(512, 256, 3, stride=1)
            self.dec_deconv2 = nn.ConvTranspose2d(256, 128, 3, stride=1)
            self.dec_deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=1)
            self.dec_deconv4 = nn.ConvTranspose2d(64, 32, 3, stride=2)
            self.dec_deconv5 = nn.ConvTranspose2d(32, 16, 5, stride=2)
            self.dec_deconv6 = nn.ConvTranspose2d(16, 3, 4, stride=2)
        elif self.size=='res128': # for Obstacle Car Racing
            if self.ch=='sesquialterate':
                ch1 = 24
            elif self.ch=='double':
                ch1 = 32
            elif self.ch=='quadruple':
                ch1 = 64
            self.enc_conv1 = nn.Conv2d(3, ch1, kernel_size=4, stride=2, padding=1)
            self.enc_conv2 = nn.Conv2d(ch1, 2 * ch1, kernel_size=4, stride=2, padding=1)
            self.enc_conv3 = nn.Conv2d(2 * ch1, 4 * ch1, kernel_size=4, stride=2, padding=1)
            self.enc_conv4 = nn.Conv2d(4 * ch1, 8 * ch1, kernel_size=4, stride=2, padding=1)
            self.enc_conv5 = nn.Conv2d(8 * ch1, 16 * ch1, kernel_size=4, stride=2, padding=1)
            self.fc_mu = nn.Linear(4 * 4 * 16 * ch1, self.env.n_z_dim)
            self.fc_logvar = nn.Linear(4 * 4 * 16 * ch1, self.env.n_z_dim)

            self.dec_fc = nn.Linear(self.env.n_z_dim, 4 * 4 * 16 * ch1)
            self.dec_deconv1 = nn.ConvTranspose2d(16 * ch1, 8 * ch1, kernel_size=4, stride=2, padding=1)
            self.dec_deconv2 = nn.ConvTranspose2d(8 * ch1, 4 * ch1, kernel_size=4, stride=2, padding=1)
            self.dec_deconv3 = nn.ConvTranspose2d(4 * ch1, 2 * ch1, kernel_size=4, stride=2, padding=1)
            self.dec_deconv4 = nn.ConvTranspose2d(2 * ch1, ch1, kernel_size=4, stride=2, padding=1)
            self.dec_deconv5 = nn.ConvTranspose2d(ch1, 3, kernel_size=4, stride=2, padding=1)

        self.to(self.device)

    def format_batch(self, batch):
        """Prepare a DataLoader batch for forward().

        Args:
            batch: Dict with images under self.data_keys[0]

        Returns:
            Dict with 'inputs' tensor ready for the model
        """
        batch = {key: value.to(self.device) for key, value in batch.items()}
        formatted_batch = self.squash_obs(batch['obses'])
        formatted_batch = utils.perm_tf2pt(formatted_batch)
        return formatted_batch

    def forward(self, x):
        """Run VAE forward pass and return reconstruction and params.

        Args:
            input_x: Tensor [N,3,H,W] in [0,1]

        Returns:
            recon_x: Tensor [N,3,H,W] reconstruction
            mu: Tensor [N,Z] latent mean
            logvar: Tensor [N,Z] latent log-variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input_x):
        """Encode images to latent parameters.

        Args:
            input_x: Tensor [N,3,H,W] in [0,1]

        Returns:
            mu: Tensor [N,Z]
            logvar: Tensor [N,Z]
        """
        if self.size=='M':
            h = F.relu(self.enc_conv1(input_x))
            h = F.relu(self.enc_conv2(h))
            h = F.relu(self.enc_conv3(h))
            h = F.relu(self.enc_conv4(h))
            h = h.reshape(h.size(0), -1)
            mu = self.fc_mu(h)
            logvar = self.fc_log_var(h)
            return mu, logvar
        elif self.size=='L':
            h = F.relu(self.enc_conv1(input_x))
            h = F.relu(self.enc_conv2(h))
            h = F.relu(self.enc_conv3(h))
            h = F.relu(self.enc_conv4(h))
            h = F.relu(self.enc_conv5(h))
            h = h.reshape(h.size(0), -1)
            mu = self.fc_mu(h)
            logvar = self.fc_log_var(h)
            return mu, logvar
        elif self.size=='XL':
            h = F.relu(self.enc_conv1(input_x))
            h = F.relu(self.enc_conv2(h))
            h = F.relu(self.enc_conv3(h))
            h = F.relu(self.enc_conv4(h))
            h = F.relu(self.enc_conv5(h))
            h = F.relu(self.enc_conv6(h))
            h =  h.reshape(h.size(0), -1)
            mu = self.fc_mu(h)
            logvar = self.fc_log_var(h)
            return mu, logvar
        elif self.size=='res128':
            h = F.relu(self.enc_conv1(input_x))
            h = F.relu(self.enc_conv2(h))
            h = F.relu(self.enc_conv3(h))
            h = F.relu(self.enc_conv4(h))
            h = F.relu(self.enc_conv5(h))
            h = h.reshape(h.size(0), -1)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample z using the reparameterization trick.

        Args:
            mu: Tensor [N,Z]
            logvar: Tensor [N,Z]

        Returns:
            z: Tensor [N,Z]
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        """Decode latent vectors to images.

        Args:
            z: Tensor [N,Z]

        Returns:
            recon_x: Tensor [N,3,H,W] in [0,1]
        """
        if self.size=='M':
            h = self.dec_fc(z)
            h = h.reshape(h.size(0), 2 * 2 * 384, 1, 1)
            h = F.relu(self.dec_deconv1(h))
            h = F.relu(self.dec_deconv2(h))
            h = F.relu(self.dec_deconv3(h))
            output_y = torch.sigmoid(self.dec_deconv4(h))
            output_y = torch.clamp(output_y, min=1e-8, max=1 - 1e-8)
            return output_y
        elif self.size=='L':
            h = self.dec_fc(z)
            if self.ch=='sesquialterate':
                h = h.reshape(h.size(0), 384, 2, 2)
            elif self.ch=='double':
                h = h.reshape(h.size(0), 512, 2, 2)
            h = F.relu(self.dec_deconv1(h))
            h = F.relu(self.dec_deconv2(h))
            h = F.relu(self.dec_deconv3(h))
            h = F.relu(self.dec_deconv4(h))
            output_y = torch.sigmoid(self.dec_deconv5(h))
            output_y = torch.clamp(output_y, min=1e-8, max=1 - 1e-8)
            return output_y
        elif self.size=='XL':
            h = self.dec_fc(z)
            h = h.reshape(h.size(0), 512, 2, 2)
            h = F.relu(self.dec_deconv1(h))
            h = F.relu(self.dec_deconv2(h))
            h = F.relu(self.dec_deconv3(h))
            h = F.relu(self.dec_deconv4(h))
            h = F.relu(self.dec_deconv5(h))
            output_y = torch.sigmoid(self.dec_deconv6(h))
            output_y = torch.clamp(output_y, min=1e-8, max=1 - 1e-8)
            return output_y
        elif self.size=='res128':
            h = self.dec_fc(z)
            if self.ch=='sesquialterate':
                h = h.reshape(h.size(0), 384, 4, 4)
            elif self.ch=='double':
                h = h.reshape(h.size(0), 512, 4, 4)
            elif self.ch=='quadruple':
                h = h.reshape(h.size(0), 1024, 4, 4)
            h = F.relu(self.dec_deconv1(h))
            h = F.relu(self.dec_deconv2(h))
            h = F.relu(self.dec_deconv3(h))
            h = F.relu(self.dec_deconv4(h))
            output_y = torch.sigmoid(self.dec_deconv5(h))
            output_y = torch.clamp(output_y, min=1e-8, max=1 - 1e-8)
            return output_y

    def loss_function(self, recon_x, x, mu, logvar):
        """Compute reconstruction + KL losses.

        Args:
            recon_x: Tensor [N,3,H,W] reconstruction
            x: Tensor [N,3,H,W] original input
            mu: Tensor [N,Z]
            logvar: Tensor [N,Z]

        Returns:
            If self.training: scalar loss tensor (MSE + KL)
            Else: tuple (BCE + KL, MSE + KL)
        """
        MSE = torch.sum((recon_x - x) ** 2, dim=[1, 2, 3])
        MSE = torch.mean(MSE)


        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        BCE = BCE / recon_x.size(0)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        KLD = torch.max(KLD, torch.tensor(self.kl_tolerance * self.env.n_z_dim).to(KLD.device))
        KLD = torch.mean(KLD)

        if self.training:
            return MSE + KLD
        else:
            return BCE + KLD, MSE + KLD

    def train_model(self, data):
        """One training step using raw image tensor.

        Args:
            data: Tensor [N,3,H,W] in [0,1]

        Returns:
            Scalar loss tensor
        """
        recon_batch, mu, logvar = self.forward(data)
        loss = self.loss_function(recon_batch, data, mu, logvar)
        return loss

    def eval_model(self, data):
        """Compute validation losses.

        Args:
            data: Tensor [N,3,H,W] in [0,1]

        Returns:
            (bce_loss, mse_loss): Two scalar tensors
        """
        recon_batch, mu, logvar = self.forward(data)
        val_loss_bce, val_loss_mse = self.loss_function(recon_batch, data, mu, logvar)
        return val_loss_bce, val_loss_mse

    def evaluate(self, data, batch_size=1000):
        """Print average validation losses over the validation split.

        Args:
            data: Dict with 'obses' array and split idxes
            batch_size: Eval batch size

        Returns:
            None
        """
        val_dataset = CustomDataset(data, self.data_keys, data['val_idxes'])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.eval()
        total_val_loss_mse = 0
        total_val_loss_bce = 0
        total_samples = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_loss_bce, val_loss_mse = self.eval_model(self.format_batch(val_batch))
                batch_size = val_batch[self.data_keys[0]].size(0)
                total_val_loss_mse += val_loss_mse.item() * batch_size
                total_val_loss_bce += val_loss_bce.item() * batch_size
                total_samples += batch_size
            val_avg_loss_mse = total_val_loss_mse / total_samples
            val_avg_loss_bce = total_val_loss_bce / total_samples

        print(f'Validation MSE: {val_avg_loss_mse}, Validation BCE: {val_avg_loss_bce}')
