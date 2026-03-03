"""PyTorch base class."""

import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from tqdm import tqdm
from utils import CustomDataset


class PyTorchModel(nn.Module):
    """Base model with standard training/eval utilities.

    Args:
        env: Environment passed down to subclasses

    Notes:
        Subclasses must implement:
            - format_batch(batch): return inputs in the form train_model/eval_model expect
            - train_model(formatted_batch): compute a scalar loss for a training step
            - eval_model(formatted_batch): return (bce_loss, mse_loss) for validation
        Also set self.data_keys to the keys used by the dataset.
    """

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.loss = None
        self.optimizer = None

    def initialize_optimizer(self, learning_rate=1e-3):
        """Create Adam optimiser if not already initialised.

        Args:
            learning_rate: Learning rate for Adam
        """
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def learn(self, data, epochs=40, ftol=1e-6, learning_rate=1e-3, batch_size=32, val_update_freq=10, verbose=True):
        """Generic training loop with periodic validation.

        Args:
            data: Dict with array and split idxes
            epochs: Maximum number of epochs
            ftol: Convergence tolerance (optional use)
            learning_rate: Adam learning rate
            batch_size: Training batch size
            val_update_freq: Validate every N epochs
            verbose: If True, print progress

        Returns:
            None

        Notes:
            Expects self.data_keys to select inputs from the dataset.
        """
        train_dataset = CustomDataset(data, self.data_keys,
                                      data['train_idxes'])
        val_dataset = CustomDataset(data, self.data_keys, data['val_idxes'])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
        print("Number of batches/iterations of one epoch in training dataset:", len(train_loader))

        self.train()

        self.initialize_optimizer(learning_rate)

        val_avg_losses_bce = []
        val_avg_losses_mse = []
        best_val_loss_mse = None

        for epoch in range(epochs):
            losses = []
            train_loader = tqdm(train_loader)
            for batch in train_loader:
                self.optimizer.zero_grad()

                loss = self.train_model(self.format_batch(batch))
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                train_loader.set_postfix(loss=np.mean(losses), epoch=epoch)
            if epoch % val_update_freq == 0:
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
                    val_avg_losses_mse.append(val_avg_loss_mse)
                    val_avg_losses_bce.append(val_avg_loss_bce)
                    if verbose:
                        print(f'Epoch {epoch}/{epochs}, Train loss: {np.mean(losses)},'
                              f' Validation MSE: {val_avg_loss_mse}, Validation BCE: {val_avg_loss_bce}')

                    if best_val_loss_mse is None or val_avg_loss_mse < best_val_loss_mse:
                        best_val_loss_mse = val_avg_loss_mse
                        #self.save(os.path.join(os.getcwd(), 'enc_user_lat32_ch15L_res84_ep600_epochs60_lrs0_00001.pt'))

                    # if self.converged(val_avg_losses_bce, ftol):
                    #   if verbose:
                    #     print("Convergence criteria met")
                    #   break
                self.train()

            # if self.converged(val_avg_losses_bce, ftol):
            #   break

        if verbose:
            plt.figure()
            plt.plot(val_avg_losses_bce)
            plt.xlabel('Epoch')
            plt.ylabel('Validation BCE Loss')
            plt.title('Validation BCE Loss Over Time')
            plt.show()

            plt.figure()
            plt.plot(val_avg_losses_mse)
            plt.xlabel('Epoch')
            plt.ylabel('Validation MSE Loss')
            plt.title('Validation MSE Loss Over Time')
            plt.show()

    def converged(self, val_losses, ftol, min_iters=2, eps=1e-9):
        """Relative-improvement convergence check.

        Args:
            val_losses: List of validation losses
            ftol: Relative tolerance
            min_iters: Minimum length before checking
            eps: Small constant

        Returns:
            True if converged else False
        """
        if len(val_losses) >= max(2, min_iters):
            if val_losses[-1] == np.nan or abs(val_losses[-1] - val_losses[-2]) / (eps + abs(val_losses[-2])) < ftol:
                return True
        return False

    def save(self, file):
        """Save model and optimiser state dicts.

        Args:
            file: Path to a checkpoint file
        """
        torch.save({
            'model_state': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict()}, file)

    def load(self, file):
        """Load model and optimiser state dicts.

        Args:
            file: Path to a checkpoint file
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(file, map_location=device)
        self.load_state_dict(checkpoint['model_state'])
        self.to(device)
        self.initialize_optimizer()
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
