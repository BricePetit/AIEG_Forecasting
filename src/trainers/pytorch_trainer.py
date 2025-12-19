"""
In this module, we define the trainer in PyTorch.
"""
__title__: str = "pytorch_trainer"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
import gc
import os
from typing import NoReturn

# Imports third party libraries
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchmetrics.regression import MeanAbsoluteError
import tqdm

# Imports from src
from .base_trainer import BaseTrainer
from config import DASH_NB, MODELS_DIR
from metrics import print_metrics
from models import (
    PyTorchGRU, PyTorchLSTM, PyTorchRNN, PyTorchSeq2Point, PyTorchUNetNilm, PyTorchCNNGRU,
    PyTorchComplexCNNGRU, PyTorchTimeSeriesTransformer, PyTorchMLP, PyTorchSimpleMLP
)
from utils import plot_predictions

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- CLASSES ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


class PyTorchTrainer(BaseTrainer):
    """
    Class for a PyTorchSeq2Models model. The idea of this class is to create, train, validate and
    test the model. This class will plot the history and the results of the model. It is also
    possible to save and load the model.
    According to the model that we want, it will be possible to create a Seq2Point model or a
    Seq2Seq model.
    """
    def __init__(
            self, model_name: str, model_type: str, seq_length: int, out_length: int,
            learning_rate: float, epochs: int, batch_size: int, patience: int, input_size: int = 1,
            features: int = 1, hidden_size: int = 1, num_layers: int = 3, d_model: int = 64,
            num_heads: int = 4, dim_feedforward: int = 128, drop_prob: float = 0.2,
            shuffle: bool = False
    ):
        """
        Constructor / Initializer of the Seq2Point class.

        :param model_name:      Name of the model.
        :param model_type:      Type of the model (Seq2Point, UNetNilm or).
        :param seq_length:      Length of the sequence.
        :param out_length:      Length of the output sequence.
        :param learning_rate:   Learning rate.
        :param epochs:          Number of epochs.
        :param batch_size:      Size of the batch.
        :param patience:        Patience for the early stopping.
        :param input_size:      Number of input features.
        :param features:        Number of features.
        :param hidden_size:     Number of hidden units (number of hidden features).
        :param num_layers:      Number of layers.
        :param d_model:         Dimension of the model.
        :param num_heads:       Number of heads.
        :param dim_feedforward: Dimension of the feedforward layer.
        :param drop_prob:       Dropout probability.
        :param shuffle:         Boolean to shuffle the indices.
        """
        super().__init__(
            model_name, model_type, seq_length, out_length, learning_rate,
            epochs, batch_size, patience, 'PyTorch'
        )
        if self.model_type == "Seq2Point":
            self.model = PyTorchSeq2Point(self.seq_length, self.out_length, features)
        elif self.model_type == "UNetNilm":
            self.model = PyTorchUNetNilm(features)
        elif self.model_type == 'LSTM':
            self.model = PyTorchLSTM(
                self.input_size, self.out_length, hidden_size, num_layers, drop_prob
            )
        elif self.model_type == 'GRU':
            self.model = PyTorchGRU(
                self.input_size, self.out_length, hidden_size, num_layers, drop_prob
            )
        elif self.model_type == 'RNN':
            self.model = PyTorchRNN(
                self.input_size, self.out_length, hidden_size, num_layers, drop_prob
            )
        elif self.model_type == 'CNNGRU':
            self.model = PyTorchCNNGRU(
                self.seq_length, input_size,
            )
        elif self.model_type == 'ComplexCNNGRU':
            self.model = PyTorchComplexCNNGRU(
                self.seq_length, input_size, hidden_size, num_layers, drop_prob
            )
        elif self.model_type == 'TimeSeriesTransformer':
            self.model = PyTorchTimeSeriesTransformer(
                input_size, d_model, num_heads, num_layers, dim_feedforward, self.out_length,
                drop_prob
            )
        elif self.model_type == 'MLP':
            self.model = PyTorchMLP(
                input_size, output_size=self.out_length, dropout_rate=drop_prob
            )
        elif self.model_type == 'SimpleMLP':
            self.model = PyTorchSimpleMLP(
                input_size, hidden_size, self.out_length, drop_prob
            )
        self.workers = 8
        self.shuffle = shuffle
        self.metrics = {
            'loss_train': [],
            'loss_val': []
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_epoch(self, train_gen, metrics_fn, optimizer, epoch, example_seen):
        """
        Function to train the model for one epoch.

        :param train_gen:       Data generator for the training set.
        :param metrics_fn:      Dictionary of metrics functions.
        :param optimizer:       Optimizer.
        :param epoch:           Current epoch.
        :param example_seen:    Number of examples seen.
        """
        # Set training mode
        self.model.train()
        # Get the size of the dataset
        ds_size = len(train_gen)
        # Create a progress bar
        # pbar_train = tqdm.tqdm(
        #     enumerate(train_gen), total=ds_size, desc=f'Epoch [{epoch + 1}/{self.epochs}] - Training'
        # )
        # Initialize loss, MAE, and MAPE
        train_loss, train_mae = 0, 0
        # For each data in the generator
        for x_train, y_train in train_gen:
        # for i, (x_train, y_train) in pbar_train:
            # Send data to CPU/GPU
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            # Forward pass
            # Zero gradients.
            optimizer.zero_grad()
            y_pred = self.model(x_train)
            # Compute loss and metrics
            loss = metrics_fn['loss'](y_pred, y_train)
            # Backward pass.
            loss.backward()
            # Gradient clipping (helps stabilize training).
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # Step the optimizer.
            optimizer.step()
            # Add loss, MAE, and MAPE.
            train_loss += loss.item()
            # train_mae += metrics_fn['mae'](y_pred, y_train).item()
            # example_seen += x_train.size(0)
            # pbar_train.set_postfix_str(
            #     f"Loss (MSE): {train_loss / (i + 1):.4f}, MAE: {train_mae / (i + 1):.4f}"
            # )
            # if i % 25 == 0:
                # train_log(loss, example_seen, epoch + 1)
        # Update metrics
        self.metrics['loss_train'].append(round(train_loss/ds_size, 4))
        return example_seen

    def val_epoch(self, val_gen, metrics_fn, epoch):
        """
        Function to validate the model for one epoch.

        :param val_gen:     Data generator for the validation set.
        :param metrics_fn:  Dictionary of metrics functions.
        :param epoch:       Current epoch.
        """
        # Set evaluation mode.
        self.model.eval()
        # Get the size of the dataset.
        ds_size = len(val_gen)
        # Create a progress bar.
        # pbar_val = tqdm.tqdm(
        #     enumerate(val_gen), total=ds_size, desc=f'Epoch [{epoch + 1}/{self.epochs}] - Validation'
        # )
        # Initialize loss and MAE
        total_loss, total_mae = 0, 0
        val_loss = 0
        # Safe way to do the validation
        with torch.no_grad():
            # For each data in the generator
            for x_val, y_val in val_gen:
            # for i, (x_val, y_val) in pbar_val:
                # Send data to CPU/GPU
                x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                # Forward pass and compute loss.
                y_pred = self.model(x_val)

                val_loss += metrics_fn['loss'](y_pred, y_val).item()
                # Add loss, MAE and MAPE
                # total_loss += metrics_fn['loss'](y_pred, y_val).item()
                # total_mae += metrics_fn['mae'](y_pred, y_val).item()
                # pbar_val.set_postfix_str(
                #     f"Loss (MSE): {total_loss / (i + 1):.4f}, MAE: {total_mae / (i + 1):.4f}"
                # )
        # Update metrics
        self.metrics[f'loss_val'].append(round(val_loss/ds_size, 4))

    def train(self, train, val=None, example_seen=0):
        """
        Function to train the model.

        :param train:           Data for the training set.
        :param val:             Data for the validation set.
        :param example_seen:    Number of examples seen.
        """
        print('-' * DASH_NB, 'Training the model...', '-' * DASH_NB)
        # Create the data loaders.
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.workers,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True
        )
        # Send the model to the device.
        self.model.to(self.device)
        # Autotuner runs a short benchmark and selects the kernel with the best performance on
        # a given hardware for a given input size. (Only for NVIDIA GPUs and convolutional networks)
        # torch.backends.cudnn.benchmark = True
        # Initialize the metrics functions.
        # metrics_fn = {
        #     'loss': nn.MSELoss(),
        #     'mae': MeanAbsoluteError(),
        # }
        # Send metrics to the device.
        # for key in metrics_fn.keys():
        #     metrics_fn[key].to(self.device)
        # Initialize the optimizer.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Add a learning rate scheduler.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.patience//2, verbose=True
        )
        # Initialize variables for early stopping.
        early_stopping = 0
        best_loss = float('inf')
        # wandb.watch(self.model, mse_fn, log="all", log_freq=10)
        # For each epoch.
        for epoch in range(self.epochs):
            # Train the model.
            example_seen = self.train_epoch(
                train_gen, metrics_fn, optimizer, epoch, example_seen
            )
            # Validate the model.
            self.val_epoch(val_gen, metrics_fn, epoch)
            # Get the last loss from the validation set.
            val_loss = self.metrics['loss_val'][-1]
            # Store current learning rate.
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                break
            # Update learning rate scheduler.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            scheduler.step(self.metrics['loss_val'][-1])
            # Check if the loss improved.
            if val_loss < best_loss:
                best_loss = val_loss
                early_stopping = 0
                self.save_model()
            else:
                early_stopping += 1
            
            # Cleanup after each epoch.
            gc.collect()
            torch.cuda.empty_cache()

            # Check if we need to stop the training.
            if self.patience == early_stopping:
                print(f"Loss did not improve after {self.patience} epochs. Stopping training at epoch {epoch + 1}/{self.epochs}.")
                break
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                loss_train = self.metrics['loss_train'][-1]
                print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {loss_train:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}')
        
        self.load_model()
        print('-' * DASH_NB, 'Model trained!', '-' * DASH_NB)
        print('-' * DASH_NB, 'Plotting history...', '-' * DASH_NB)
        # self.plot_history({
        #     'loss': self.metrics['loss_train'],
        #     'val_loss': self.metrics['loss_val']
        # })
        print('-' * DASH_NB, 'History plotted!', '-' * DASH_NB)

    def evaluate(self, test):
        """
        Function to evaluate the model.

        :param test:    Test data.

        :return:        Return the evaluation result.
        """
        print('-' * DASH_NB, 'Evaluating the model...', '-' * DASH_NB)
        # Create lists to store predictions and true values.
        all_preds, all_true = [], []
        # Create the data generator.
        test_gen = torch.utils.data.DataLoader(
            test, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True
        )
        # Get the size of the dataset.
        ds_size = len(test_gen)
        # Turn on evaluation mode.
        self.model.eval()
        # Send the model to the device.
        self.model.to(self.device)
        # Autotuner runs a short benchmark and selects the kernel with the best performance on
        # a given hardware for a given input size. (Only for NVIDIA GPUs and convolutional networks)
        torch.backends.cudnn.benchmark = True
        # Initialize the metrics.
        metrics_fn = {
            'loss': nn.MSELoss(),
            'mae': MeanAbsoluteError(),
        }
        # Send metrics to the device.
        for key in metrics_fn.keys():
            metrics_fn[key].to(self.device)
        # Initialize loss, MAE, and MAPE.
        total_loss, total_mae = 0, 0
        # Create a progress bar.
        # pbar_test = tqdm.tqdm(enumerate(test_gen), total=ds_size, desc='Evaluating')
        # Safe way to do the prediction.
        with torch.inference_mode():
            # Make predictions on the test set.
            for x_test, y_test in test_gen:
            # for i, (x_test, y_test) in pbar_test:
                # Forward pass and compute loss.
                x_test = x_test.to(self.device)
                # Do the prediction.
                y_preds = self.model(x_test)
                # Add loss, MAE, and MAPE.
                # total_loss += metrics_fn['loss'].item()
                # total_mae += metrics_fn['mae'].item()
                # pbar_test.set_postfix_str(
                #     f"Loss: {total_loss / (i + 1):.3f}, MAE: {total_mae / (i + 1):.3f}"
                # )
                # If y_test is a tensor, convert it to numpy array. Else, convert it to numpy array.
                y_test = (
                    y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else np.array(y_test)
                )
                # Append the predictions and true values to the lists.
                all_preds.append(y_preds.cpu().numpy())
                all_true.append(y_test)
        # Concatenate all predictions and true values.
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_true)
        print('-' * DASH_NB, 'Evaluation Done!', '-' * DASH_NB)
        print('-' * DASH_NB, 'Evaluation Result:', '-' * DASH_NB)
        print_metrics(y_true, y_pred)

    def predict(self, test, index=None):
        """
        Function to predict with the model, and return the predictions.
        Predictions will be saved in a file, and the results will be plotted.

        :param test:    Test data generator.
        :param index:   Index of the test data.

        :return:        Return the prediction result.
        """
        print('-' * DASH_NB, 'Predicting the model...', '-' * DASH_NB)
        # Create the data generator.
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True
        )
        # Turn on evaluation mode.
        self.model.eval()
        # Send the model to the device.
        self.model.to(self.device)
        # Autotuner runs a short benchmark and selects the kernel with the best performance on
        # a given hardware for a given input size. (Only for NVIDIA GPUs and convolutional networks)
        torch.backends.cudnn.benchmark = True
        # Create a progress bar.
        # pbar_test = tqdm.tqdm(enumerate(test_gen), total=len(test_gen), desc='Predicting')
        # Create lists to store predictions and true values.
        all_preds, all_true = [], []
        # Safe way to do the prediction.
        with torch.no_grad():
            # Make predictions on the test set.
            for x_test, y_test in test_loader:
            # for _, (x_test, y_test) in pbar_test:
                x_test = x_test.to(self.device)
                # Do the prediction.
                y_preds = self.model(x_test).cpu().numpy()
                # If y_test is a tensor, convert it to numpy array. Else, convert it to numpy array.
                y_test = (
                    y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else np.array(y_test)
                )
                # Append the predictions and true values to the lists.
                all_preds.append(y_preds)
                all_true.append(y_test)
        # Concatenate all predictions and true values.
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_true)
        # Check if the predictions and true values are multi-step or single-step.
        multi_step = y_pred.shape[1] > 1 or y_true.shape[1] > 1
        if multi_step:
            # For the multi-step predictions.
            # Convert to DataFrame for multi-step predictions.
            if index is not None:
                true_df = pd.DataFrame(
                    y_true, index=index, columns=[f't+{i+1}' for i in range(y_true.shape[1])]
                )
            else:
                true_df = pd.DataFrame(
                    y_true, columns=[f't+{i+1}' for i in range(y_true.shape[1])]
                )
            
            # Show the metrics for each step.
            for i in range(min(y_true.shape[1], y_pred.shape[1])):
                col = f't+{i+1}'
                print(f"\n--- Metrics for {col} ---")
                print_metrics(true_df[col], y_pred[:, i])
            # Show the metrics for all steps.
            print("\n--- Metrics for all steps ---")
            print_metrics(true_df.values.flatten(), y_pred.flatten())
            # Plot the predictions.
            plot_predictions(true_df, y_pred, self.out_length, index=true_df.index)
        else:
            # For single-step predictions.
            # Flatten the predictions and true values.
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
            # Convert to DataFrame for single-step predictions.
            if index is not None:
                results = pd.DataFrame({'true': y_true, 'predicted': y_pred}, index=index)
            else:
                results = pd.DataFrame({'true': y_true, 'predicted': y_pred})
            # Show the metrics.
            print_metrics(results['true'], results['predicted'])
            # Plot the predictions.
            plot_predictions(
                results['true'], results['predicted'], self.out_length, index=results.index
            )
        print('-' * DASH_NB, 'Predictions Done!', '-' * DASH_NB)
        print('-' * DASH_NB, 'Predictions Result:', '-' * DASH_NB)
        self.save_predictions(y_pred)
        

    def save_model(self) -> NoReturn:
        """
        Function to save the model.
        """
        # print('-' * DASH_NB, 'Saving the model...', '-' * DASH_NB)
        path = f"{MODELS_DIR}/{self.model_type}"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(
            self.model.state_dict(),
            f"{path}/{self.model_name}_{self.model_type}_PyTorch.pth"
        )
        # print('-' * DASH_NB, 'Model saved!', '-' * DASH_NB)

    def load_model(self) -> NoReturn:
        """
        Function to load the model.
        :return:
        """
        # print('-' * DASH_NB, 'Loading the model...', '-' * DASH_NB)
        self.model.load_state_dict(torch.load(
            f"{MODELS_DIR}/{self.model_type}/{self.model_name}_{self.model_type}_PyTorch.pth",
            weights_only=True
        ))
        # print('-' * DASH_NB, 'Model loaded!', '-' * DASH_NB)


def custom_loss(y_pred, y_true, alpha=1):
    """
    Custom loss function.

    :param y_pred:  Predictions.
    :param y_true:  True values.
    :param alpha:   Alpha.
    """
    # Base loss.
    base_loss = nn.MSELoss()(y_pred, y_true)
    # Penalize negative predictions.
    penalty = torch.sum(torch.relu(-y_pred))
    return base_loss + penalty * alpha
