"""
This module is used to train and test the models.
"""

__title__: str = "training_testing"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
from itertools import product
import os
from typing import (
    Dict, List, NoReturn, Tuple
)

# Imports third party libraries
import cudf
# import lightgbm as lgb
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# import shap

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.stattools import adfuller, acf, pacf
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import xgboost as xgb


# Imports from src
from config import (
    DASH_NB,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    NB_WORKERS,
    SAVED_MODELS_DIR,
    consumption_sites_grouped,
    consumption_drop_period,
    production_sites_grouped,
    production_drop_period,
)

from data_generator import (
    PyTorchDataGenerator,
)

from metrics import (
    compute_metrics,
    compute_mae,
    compute_mape,
    compute_wmape,
    compute_mse,
    print_metrics,
)

from models import (
    PyTorchSimpleMLP,
    PyTorchMLP,
    PyTorchGRU,
    PyTorchCNNGRU,
    PyTorchComplexCNNGRU,
    PyTorchTimeSeriesTransformer,
)

from preprocessing import (
    denormalize_data,
    drop_useless_perdiod,
    load_data,
    prepare_data_ml,
)

from trainers import (
    PyTorchTrainer,
    TFTrainer,
)


from evaluation import plot_predictions

from data.preprocessing import (
    kfold_temporal_split,
    temporal_split,
)

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNCTIONS ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #

def train_model(
        model, train_gen, val_gen, criterion, optimizer, model_name, batch_size=64, num_epochs=100,
        patience=10, scheduler=None, device='cpu', save_dir='saved_models'
    ):
    """
    """
    # Create DataLoaders
    train_loader = DataLoader(
        train_gen, batch_size=batch_size, num_workers=NB_WORKERS, pin_memory=True, shuffle=True
    )
    val_loader = DataLoader(
        val_gen, batch_size=batch_size, num_workers=NB_WORKERS, pin_memory=True, shuffle=False
    )
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'best_{model_name}_model.pt')
    
    # Move model to device
    model = model.to(device)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            # with torch.amp.autocast(device.type):
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            # Backward pass
            loss.backward()
            # Gradient clipping (helps stabilize training)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        val_loss = val_loss / len(val_loader)
        history['val_loss'].append(val_loss)

        # Store current learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            break
        history['learning_rate'].append(current_lr)

        # Update learning rate scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}/{num_epochs}")
            break
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}')
    
    # Load the best model
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Best model saved with validation loss: {best_val_loss:.6f}")
    
    return model, history


def predict_model(
    model, test_gen, out_size, batch_size=64, device='cpu', index=None, kfold_cv=False
):
    """
    Function to make predictions using the model on a given dataset.

    :param model:        Trained model.
    :param test_gen:     Test data generator.
    :param out_size:     Output size.
    :param batch_size:   Batch size for the DataLoader.
    :param device:       Device to use for computation (CPU or GPU).
    :param index:        Optional index for the predictions.
    :param kfold_cv:     Whether to use k-fold cross-validation or not.
    """
    # Set the model to evaluation mode.
    model.eval()
    
    # Create the test DataLoader.
    test_loader = DataLoader(
        test_gen, batch_size=batch_size, shuffle=False, num_workers=NB_WORKERS, pin_memory=True
    )
    # Create lists to hold the predictions and true values.
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            # Move data to device.
            X_test = X_test.to(device)
            # Make predictions.
            y_preds = model(X_test).cpu().numpy()
            # If y_test is a tensor, convert it to numpy array. Else, convert it to numpy array.
            y_test = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else np.array(y_test)
            # Append the predictions and true values to the lists.
            all_preds.append(y_preds)
            all_true.append(y_test)
    # Concatenate all predictions and true values.
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_true)
    if not kfold_cv:
        # Check if the predictions and true values are multi-step or single-step.
        if out_size > 1:
            # For the multi-step predictions.
            # Convert to DataFrame for multi-step predictions.
            if index is not None:
                true_df = pd.DataFrame(
                    y_true, index=index,
                    columns=[f'ap+{i+1}' for i in range(y_true.shape[1])]
                )
            else:
                true_df = pd.DataFrame(
                    y_true, 
                    columns=[f'ap+{i+1}' for i in range(y_true.shape[1])]
                )
            # Show the metrics for each step.
            for i in range(min(y_true.shape[1], y_pred.shape[1])):
                col = f'ap+{i+1}'
                print(f"\n--- Metrics for {col} ---")
                print_metrics(true_df[col], y_pred[:, i])
            # Show the metrics for all steps.
            print("\n--- Metrics for all steps ---")
            print_metrics(true_df.values.flatten(), y_pred.flatten())
            # Plot the predictions.
            plot_predictions(true_df, y_pred, out_size=y_true.shape[1], index=true_df.index)
        else:
            # For single-step predictions.
            # Flatten the predictions and true values.
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
            # Convert to DataFrame for single-step predictions.
            if index is not None:
                results = pd.DataFrame(
                    {'true': y_true, 'ap+1': y_pred}, index=index
                )
            else:
                results = pd.DataFrame({'true': y_true, 'ap+1': y_pred})
            # Show the metrics.
            print_metrics(results['true'], results['predicted'])
            # Plot the predictions.
            plot_predictions(
                results['true'], results['predicted'], out_size=1, index=results.index
            )
    return y_pred


def explain_shap(train_gen, test_gen, batch_size, model) -> NoReturn:
    """
    Function to explain the model using SHAP values.

    :param train_gen:   Training data generator.
    :param test_gen:    Test data generator.
    :param batch_size:  Batch size for the DataLoader.
    :param model:       Trained model.
    """
    # counter = 0
    # for feature in test_gen.data.columns:
    #     print(f"Feature {counter}: {feature}")
    #     counter += 1
    # Get first values.
    batch_train = next(iter(DataLoader(
        train_gen, batch_size=batch_size, num_workers=NB_WORKERS, pin_memory=True, shuffle=True
    )))[0]
    batch_test = next(iter(DataLoader(
        test_gen, batch_size=batch_size, num_workers=NB_WORKERS, pin_memory=True, shuffle=False
    )))[0]
    # Explain the gradient.
    explainer = shap.GradientExplainer(model.to('cpu'), batch_train.to('cpu'))
    # Get the SHAP values.
    shap_values = explainer.shap_values(batch_test.to('cpu'))
    # Plot the SHAP values.
    shap_values = shap_values.squeeze(-1)
    shap.summary_plot(
        shap_values.reshape(-1, shap_values[0].shape[-1]),
        batch_test.numpy().reshape(-1, batch_test.shape[-1]),
        feature_names=test_gen.data.columns,
        max_display=50,
    )


def kfold_cross_validation_dl(
    dataset: pd.HDFStore, target_sites: dict, period2drop: dict, n_splits: int, 
    in_size: int, out_size: int, features_nb: int, batch_size: int, num_epochs: int, patience: int,
    learning_rate: float, criterion: nn.Module, device: str, model_name: str, max_value: float,
    model_type: str, weather: bool = False, ts: bool = False, stats: bool = False,
    selected_features: List[str] = None, lag: int = 0, previous_days: int = 0,
    step: int = 1, explain: bool = False, plot: bool = True
) -> Tuple[dict, str]:
    """
    Function to apply k-fold cross-validation on the data using deep learning models.

    :param dataset:         Dataset to use.
    :param site_name:       Name of the site.
    :param sn:              Serial number.
    :param prediction_type: Type of prediction (consumption or production).
    :param n_splits:        Number of folds for cross-validation.
    :param in_size:         Input size.
    :param out_size:        Output size.
    :param features_nb:     Number of features.
    :param batch_size:      Batch size.
    :param num_epochs:      Number of epochs.
    :param patience:        Patience for early stopping.
    :param learning_rate:   Learning rate.
    :param criterion:       Loss function.
    :param device:          Device to use (CPU or GPU).
    :param model_name:      Name of the model.
    :param max_value:       Maximum value for the output.
    :param model_type:      Type of model (simple_mlp or cnn_gru).
    :param weather:         Whether to use weather data or not.
    :param ts:              Whether to use time series features or not.
    :param stats:           Whether to use statistics features or not.
    :param select_features: Selected features to use.
    :param lag:             Lag for the model.
    :param previous_days:   Number of previous days to include in the data.
    :param step:            Step size for shifting the data.
    :param explain:         Whether to explain the model or not.
    :param plot:            Whether to plot the predictions or not.
    """
    # Create list of target sites to train on.
    target = [
        (site_id, site) if len(site) > 1 else (site_id, site[0])
        for site_id, site in target_sites.items()
    ]
    k_folds = kfold_temporal_split(
        dataset, in_size, out_size, k=n_splits,
        target=target, period2drop=period2drop, #train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )
    test_gen = PyTorchDataGenerator(
        dataset, k_folds[-1][2], in_size, out_size, is_weather=weather, is_ts_features=ts,
        is_stats_features=stats, lag=lag, previous_days=previous_days, step=step,
        specific_features=selected_features, period2drop=period2drop
    )
    # Define the predictions and metrics.
    all_preds = {}
    all_metrics = {}
    best_fold = ""
    best_mae = float('inf')
    fold = 1
    # Get the true values for the test set.
    true_values = []
    for _, y in test_gen:
        true_values.append(y.numpy())
    true_values = np.vstack(true_values)
    true_df = pd.DataFrame(
        true_values, index=test_gen.data.index[-len(true_values):],
        columns=[f'ap+{i+1}' for i in range(out_size)] if lag == 0 else [f'ap+{lag}']
    )
    for train_set, val_set, _ in k_folds:
        print('-' * DASH_NB, f"Fold {fold}/{n_splits}", '-' * DASH_NB)
        all_metrics[f'fold_{fold}'] = {}
        print('-' * DASH_NB, 'Training the model...', '-' * DASH_NB)
        # Create the data generators.
        train_gen = PyTorchDataGenerator(
            dataset, train_set, in_size, out_size, is_weather=weather, is_ts_features=ts,
            is_stats_features=stats, lag=lag, previous_days=previous_days, step=step,
            specific_features=selected_features, period2drop=period2drop
        )
        val_gen = PyTorchDataGenerator(
            dataset, val_set, in_size, out_size, is_weather=weather, is_ts_features=ts,
            is_stats_features=stats, lag=lag, previous_days=previous_days, step=step,
            specific_features=selected_features, period2drop=period2drop
        )
        # Train the model.
        if model_type == 'simple_mlp':
            y_preds = train_predict_simple_mlp(
                train_gen, val_gen, test_gen, in_size, out_size, features_nb, batch_size,
                num_epochs, patience, learning_rate, criterion, device, model_name, max_value,
                lag=lag, explain=explain, kfold_cv=True
            )
        elif model_type == 'complex_mlp':
            y_preds = train_predict_complex_mlp(
                train_gen, val_gen, test_gen, in_size, out_size, features_nb, batch_size,
                num_epochs, patience, learning_rate, criterion, device, model_name, max_value,
                lag=lag, kfold_cv=True
            )
        elif model_type == 'gru':
            y_preds = train_predict_gru(
                train_gen, val_gen, test_gen, in_size, out_size, features_nb, batch_size,
                num_epochs, patience, learning_rate, criterion, device, model_name, max_value,
                lag=lag, previous_days=previous_days, explain=explain, kfold_cv=True
            )
        elif model_type == 'cnn_gru':
            y_preds = train_predict_cnn_gru(
                train_gen, val_gen, test_gen, in_size, out_size, features_nb, batch_size,
                num_epochs, patience, learning_rate, criterion, device, model_name, max_value,
                lag=lag, explain=explain, kfold_cv=True
            )
        elif model_type == "complex_cnn_gru":
            y_preds = train_predict_complex_cnn_gru(
                train_gen, val_gen, test_gen, in_size, out_size, features_nb, batch_size,
                num_epochs, patience, learning_rate, criterion, device, model_name, max_value,
                lag=lag, kfold_cv=True
            )
        elif model_type == "transformer":
            y_preds = train_predict_transformer(
                train_gen, val_gen, test_gen, in_size, out_size, features_nb, batch_size,
                num_epochs, patience, learning_rate, criterion, device, model_name, max_value,
                lag=lag, kfold_cv=True
            )
        
        all_metrics, all_preds, best_fold, best_mae = kfold_metrics(
            out_size, all_metrics, all_preds, true_df, fold, y_preds, best_mae, best_fold,
            y_scaler={site_id: None for site_id, _ in target_sites.items()}
        )
        fold += 1
    show_kfold_metrics_plots(
        out_size, all_metrics, all_preds, true_df, best_fold,
        y_scaler={site_id: None for site_id, _ in target_sites.items()}, plot=plot
    )
    return all_metrics, best_fold


def train_predict_simple_mlp(
    train_gen: PyTorchDataGenerator, val_gen: PyTorchDataGenerator, test_gen: PyTorchDataGenerator,
    in_size: int, out_size: int, features_nb: int, batch_size: int, num_epochs: int, patience: int,
    learning_rate: float, criterion: nn.Module, device: str, model_name: str, max_value: float,
    lag: int = 0, previous_days: int = 0, explain: bool = False, kfold_cv: bool = False
):
    """
    Function to train and predict with a simple MLP model.

    :param train_gen:       Training data generator.
    :param val_gen:         Validation data generator.
    :param test_gen:        Test data generator.
    :param in_size:         Input size.
    :param out_size:        Output size.
    :param features_nb:     Number of features.
    :param batch_size:      Batch size.
    :param num_epochs:      Number of epochs.
    :param patience:        Patience for early stopping.
    :param learning_rate:   Learning rate.
    :param criterion:       Loss function.
    :param device:          Device to use (CPU or GPU).
    :param model_name:      Name of the model.
    :param max_value:       Maximum value for the output.
    :param lag:             Lag for the model.
    :param previous_days:   Number of previous days to include in the data.
    :param explain:         Whether to explain the model or not.
    :param kfold_cv:        Whether to use k-fold cross-validation or not.

    :return:               Predictions.
    """
    if out_size == 1:
        index = test_gen.data[in_size+lag+(previous_days * 96):].index
    else:
        index = test_gen.data[in_size+lag+(previous_days * 96):-out_size+1].index
    # Define model and optimizer.
    simple_mlp_model = PyTorchSimpleMLP(in_size * features_nb, output_size=out_size)
    simple_mlp_model = torch.compile(simple_mlp_model)
    # Define the optimizer.
    simple_mlp_optimizer = optim.Adam(simple_mlp_model.parameters(), lr=learning_rate)
    # Set up learning rate scheduler.
    simple_mlp_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        simple_mlp_optimizer, mode='min', factor=0.5, patience=patience // 2
    )
    # Train the model.
    simple_mlp_model, simple_mlp_history = train_model(
        simple_mlp_model,
        train_gen,
        val_gen,
        criterion,
        simple_mlp_optimizer,
        model_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        patience=patience,
        scheduler=simple_mlp_scheduler,
        device=device
    )
    # Make predictions.
    y_preds = predict_model(
        simple_mlp_model, test_gen, out_size, batch_size=batch_size, device=device,
        index=index, kfold_cv=kfold_cv
    )
    if explain:
        explain_shap(train_gen, test_gen, batch_size, simple_mlp_model)
    return y_preds


def train_predict_complex_mlp(
    train_gen: PyTorchDataGenerator, val_gen: PyTorchDataGenerator, test_gen: PyTorchDataGenerator,
    in_size: int, out_size: int, features_nb: int, batch_size: int, num_epochs: int, patience: int,
    learning_rate: float, criterion: nn.Module, device: str, model_name: str, max_value: float,
    lag: int = 0, kfold_cv: bool = False
):
    """
    Function to train and predict with a complex MLP model.

    :param train_gen:       Training data generator.
    :param val_gen:         Validation data generator.
    :param test_gen:        Test data generator.
    :param in_size:         Input size.
    :param out_size:        Output size.
    :param features_nb:     Number of features.
    :param batch_size:      Batch size.
    :param num_epochs:      Number of epochs.
    :param patience:        Patience for early stopping.
    :param learning_rate:   Learning rate.
    :param criterion:       Loss function.
    :param device:          Device to use (CPU or GPU).
    :param model_name:      Name of the model.
    :param max_value:       Maximum value for the output.
    :param lag:             Lag for the model.
    :param kfold_cv:        Whether to use k-fold cross-validation or not.

    :return:               Predictions.
    """
    if out_size == 1:
        index = test_gen.data[in_size+lag:].index
    else:
        index = test_gen.data[in_size+lag:-out_size+1].index
    mlp_model = PyTorchMLP(in_size * features_nb, output_size=out_size,)
    mlp_model = torch.compile(mlp_model)
    # Define the optimizer.
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)
    # Set up learning rate scheduler.
    mlp_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        mlp_optimizer, mode='min', factor=0.5, patience=patience // 2
    )
    # Train the model.
    mlp_model, mlp_history = train_model(
        mlp_model,
        train_gen,
        val_gen,
        criterion,
        mlp_optimizer,
        model_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        patience=patience,
        scheduler=mlp_scheduler,
        device=device
    )
    # Make predictions.
    y_preds = predict_model(
        mlp_model, test_gen, out_size, batch_size=batch_size, device=device,
        index=index, kfold_cv=kfold_cv
    )
    return y_preds


def train_predict_gru(
    train_gen: PyTorchDataGenerator, val_gen: PyTorchDataGenerator, test_gen: PyTorchDataGenerator,
    in_size: int, out_size: int, features_nb: int, batch_size: int, num_epochs: int, patience: int,
    learning_rate: float, criterion: nn.Module, device: str, model_name: str, max_value: float,
    lag: int = 0, previous_days: int = 0, explain: bool = False, kfold_cv: bool = False
):
    """
    Function to train and predict with a GRU model.

    :param train_gen:       Training data generator.
    :param val_gen:         Validation data generator.
    :param test_gen:        Test data generator.
    :param in_size:         Input size.
    :param out_size:        Output size.
    :param features_nb:     Number of features.
    :param batch_size:      Batch size.
    :param num_epochs:      Number of epochs.
    :param patience:        Patience for early stopping.
    :param learning_rate:   Learning rate.
    :param criterion:       Loss function.
    :param device:          Device to use (CPU or GPU).
    :param model_name:      Name of the model.
    :param max_value:       Maximum value for the output.
    :param lag:             Lag for the model.
    :param previous_days:   Number of previous days to include in the data.
    :param explain:         Whether to explain the model or not.
    :param kfold_cv:        Whether to use k-fold cross-validation or not.

    :return:               Predictions.
    """
    if out_size == 1:
        index = test_gen.data[in_size+lag+(previous_days * 96):].index
    else:
        index = test_gen.data[in_size+lag+(previous_days * 96):-out_size+1].index
    # Define model and optimizer.
    gru_model = PyTorchGRU(features_nb, out_size)
    gru_model = torch.compile(gru_model)
    gru_optimizer = optim.Adam(gru_model.parameters(), lr=learning_rate)

    # Set up learning rate scheduler.
    gru_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gru_optimizer, mode='min', factor=0.5, patience=patience // 2
    )

    # Train the model.
    gru_model, gru_history = train_model(
        gru_model, 
        train_gen, 
        val_gen, 
        criterion, 
        gru_optimizer, 
        model_name, 
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        patience=patience, 
        scheduler=gru_scheduler,
        device=device
    )
    
     # Make predictions.
    y_preds = predict_model(
        gru_model, test_gen, out_size, batch_size=batch_size, device=device,
        index=index, kfold_cv=kfold_cv
    )
    if explain:
        explain_shap(train_gen, test_gen, batch_size, gru_model)
    return y_preds


def train_predict_cnn_gru(
    train_gen: PyTorchDataGenerator, val_gen: PyTorchDataGenerator, test_gen: PyTorchDataGenerator,
    in_size: int, out_size: int, features_nb: int, batch_size: int, num_epochs: int, patience: int,
    learning_rate: float, criterion: nn.Module, device: str, model_name: str, max_value: float,
    lag: int = 0, explain: bool = False, kfold_cv: bool = False
):
    """
    Function to train and predict with a CNN-GRU model.

    :param train_gen:       Training data generator.
    :param val_gen:         Validation data generator.
    :param test_gen:        Test data generator.
    :param in_size:         Input size.
    :param out_size:        Output size.
    :param features_nb:     Number of features.
    :param batch_size:      Batch size.
    :param num_epochs:      Number of epochs.
    :param patience:        Patience for early stopping.
    :param learning_rate:   Learning rate.
    :param criterion:       Loss function.
    :param device:          Device to use (CPU or GPU).
    :param model_name:      Name of the model.
    :param max_value:       Maximum value for the output.
    :param lag:             Lag for the model.
    :param explain:         Explain the model (True) or not (False).
    :param kfold_cv:        Whether to use k-fold cross-validation or not.

    :return:               Predictions.
    """
    if out_size == 1:
        index = test_gen.data[in_size+lag:].index
    else:
        index = test_gen.data[in_size+lag:-out_size+1].index
    # Define model and optimizer.
    cnn_gru_model = PyTorchCNNGRU(n_features=features_nb, output_size=out_size)
    cnn_gru_optimizer = optim.Adam(cnn_gru_model.parameters(), lr=learning_rate)

    # Set up learning rate scheduler.
    cnn_gru_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        cnn_gru_optimizer, mode='min', factor=0.5, patience=patience // 2
    )

    # Train the model.
    cnn_gru_model, cnn_gru_history = train_model(
        cnn_gru_model, 
        train_gen, 
        val_gen, 
        criterion, 
        cnn_gru_optimizer, 
        model_name, 
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        patience=patience, 
        scheduler=cnn_gru_scheduler,
        device=device
    )
    # Make predictions.
    y_preds = predict_model(
        cnn_gru_model, test_gen, out_size, batch_size=batch_size, device=device,
        index=index, kfold_cv=kfold_cv
    )
    if explain:
        explain_shap(train_gen, test_gen, batch_size, cnn_gru_model)
    return y_preds


def train_predict_complex_cnn_gru(
    train_gen: PyTorchDataGenerator, val_gen: PyTorchDataGenerator, test_gen: PyTorchDataGenerator,
    in_size: int, out_size: int, features_nb: int, batch_size: int, num_epochs: int, patience: int,
    learning_rate: float, criterion: nn.Module, device: str, model_name: str, max_value: float,
    lag: int = 0, kfold_cv: bool = False
):
    if out_size == 1:
        index = test_gen.data[in_size+lag:].index
    else:
        index = test_gen.data[in_size+lag:-out_size+1].index
    # Define model and optimizer.
    complex_cnn_gru_model = PyTorchComplexCNNGRU(n_features=features_nb, output_size=out_size)
    complex_cnn_gru_optimizer = optim.Adam(complex_cnn_gru_model.parameters(), lr=learning_rate)

    # Set up learning rate scheduler.
    complex_cnn_gru_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        complex_cnn_gru_optimizer, mode='min', factor=0.5, patience=patience // 2
    )

    # Train the model.
    complex_cnn_gru_model, complex_cnn_gru_history = train_model(
        complex_cnn_gru_model, 
        train_gen, 
        val_gen, 
        criterion, 
        complex_cnn_gru_optimizer, 
        model_name, 
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        patience=patience, 
        scheduler=complex_cnn_gru_scheduler,
        device=device
    )
    # Make predictions.
    y_preds = predict_model(
        complex_cnn_gru_model, test_gen, out_size, batch_size=batch_size, device=device,
        index=index, kfold_cv=kfold_cv
    )
    return y_preds


def train_predict_transformer(
    train_gen: PyTorchDataGenerator, val_gen: PyTorchDataGenerator, test_gen: PyTorchDataGenerator,
    in_size: int, out_size: int, features_nb: int, batch_size: int, num_epochs: int, patience: int,
    learning_rate: float, criterion: nn.Module, device: str, model_name: str, max_value: float,
    lag: int = 0, kfold_cv: bool = False
):
    if out_size == 1:
        index = test_gen.data[in_size+lag:].index
    else:
        index = test_gen.data[in_size+lag:-out_size+1].index
    # Model dimensions.
    d_model = 64
    # Number of attention heads. Needs to divide d_model.
    num_heads = 4
    # Number of encoder layers.
    num_layers = 3
    # Dimensions of the feedforward network.
    dim_feedforward = 128
    # Dropout rate.
    dropout = 0.2
    # Create the transformer model.
    transformer_model = PyTorchTimeSeriesTransformer(
        input_size=features_nb,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        output_size=out_size,
        dropout=dropout
    )
    transformer_model = torch.compile(transformer_model)

    # Define the optimizer.
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=learning_rate)
    # Set up learning rate scheduler.
    transformer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        transformer_optimizer, mode='min', factor=0.5, patience=patience // 2
    )
    # Train the model.
    transformer_model, transformer_history = train_model(
        transformer_model, 
        train_gen,
        val_gen,
        criterion, 
        transformer_optimizer,
        model_name, 
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        patience=patience, 
        scheduler=transformer_scheduler,
        device=device
    )
    # Make predictions.
    y_preds = predict_model(
        transformer_model, test_gen, out_size, batch_size=batch_size, device=device,
        index=index, kfold_cv=kfold_cv
    )
    return y_preds


def kfold_metrics(
    out_size: int, all_metrics: dict, all_preds: dict, true_df: pd.DataFrame, fold: int,
    y_preds: np.ndarray, best_mae: float = float('inf'), best_fold: str = '', y_scaler=None
) -> tuple:
    """
    Function to compute the k-fold metrics.

    :param out_size:    Output size.
    :param all_metrics: All metrics for each fold.
    :param all_preds:   All predictions for each fold.
    :param true_df:     True values DataFrame.
    :param fold:        Current fold number.
    :param site_id:     Site ID for the current fold.
    :param y_preds:     Predictions for the current fold.
    :param best_mae:    Best MAE so far.
    :param best_fold:   Best fold based on MAE.
    :param y_scaler:    Optional scaler for the target variable.

    :return:            Updated all_metrics, all_preds, best_fold, best_mae.
    """
    all_preds[f'fold_{fold}'] = y_preds
    df_preds = pd.DataFrame(
        y_preds, index=true_df.index, columns=true_df.columns
    )
    # Apply inverse scaling if scalers are provided
    
    true_df, df_preds = denormalize_data(y_scaler, true_df, df_preds, out_size)
    all_metrics[f'fold_{fold}']['overall'] = {'all' : compute_metrics(
        true_df.values.flatten(), df_preds.values.flatten()
    )}
    for site_id in true_df.index.get_level_values('site_id').unique():
        all_metrics[f'fold_{fold}'][site_id] = {'all' : compute_metrics(
            true_df.xs(site_id, level='site_id').values.flatten(),
            df_preds.xs(site_id, level='site_id').values.flatten()
        )}
    for col in true_df.columns:
        all_metrics[f'fold_{fold}']['overall'][col] = compute_metrics(
            true_df[col].values.flatten(), df_preds[col].values.flatten()
        )
        for site_id in true_df.index.get_level_values('site_id').unique():
            all_metrics[f'fold_{fold}'][site_id][col] = compute_metrics(
                true_df.xs(site_id, level='site_id')[col].values.flatten(),
                df_preds.xs(site_id, level='site_id')[col].values.flatten()
            )
    mae = all_metrics[f'fold_{fold}']['overall']['all']['MAE']
    if mae < best_mae:
        best_fold = f'fold_{fold}'
        best_mae = mae
    return all_metrics, all_preds, best_fold, best_mae


def show_kfold_metrics_plots(
    out_size: int, all_metrics: dict, all_preds: dict, true_df: pd.DataFrame, best_fold: str,
    y_scaler=None, plot: bool = True
) -> NoReturn:
    """
    Function to show the k-fold metrics and plots.

    :param out_size:     Output size.
    :param all_metrics:  All metrics for each fold.
    :param all_preds:    All predictions for each fold.
    :param true_df:      True values DataFrame.
    :param best_fold:    Best fold based on MAE.
    :param y_scaler:     Optional scaler for the target variable.
    :param plot:         Whether to plot the predictions or not.
    """

    # Calculate the average metrics across all folds and sites.
    print(DASH_NB * "-", "Average Metrics Across All Folds and sites", DASH_NB * "-")
    for metric in ['MAE', 'NMAE', 'MAPE', 'MSE', 'RMSE', 'NRMSE']:
        metric_values = [all_metrics[fold]['overall']['all'][metric] for fold in all_metrics]
        print(f"{metric}: {sum(metric_values) / len(metric_values):.6f}")
    
    # Calculate the average metrics across all folds for each site.
    print(DASH_NB * "-", "Average Metrics Across All Folds for each site", DASH_NB * "-")
    for site_id in true_df.index.get_level_values('site_id').unique():
        print(DASH_NB * "-", f"Site: {site_id}", DASH_NB * "-")
        for metric in ['MAE', 'NMAE', 'MAPE', 'MSE', 'RMSE', 'NRMSE']:
            metric_values = [
                all_metrics[fold][site_id]['all'][metric] for fold in all_metrics
            ]
            print(f"{metric}: {sum(metric_values) / len(metric_values):.6f}")
    # Calculate the average metrics across all folds for each target site.
    print(DASH_NB * "-", "Average Metrics Across All Folds for each target ", DASH_NB * "-")
    for col in true_df.columns:
        print(DASH_NB * "-", f"Target: {col}", DASH_NB * "-")
        for metric in ['MAE', 'NMAE', 'MAPE', 'MSE', 'RMSE', 'NRMSE']:
            metric_values = [
                all_metrics[fold]['overall'][col][metric] for fold in all_metrics
            ]
            print(f"{metric}: {sum(metric_values) / len(metric_values):.6f}")
    # Calculate the average metrics across all folds for each target site and sites.
    print(
        DASH_NB * "-", "Average Metrics Across All Folds for each target site and sites",
        DASH_NB * "-"
    )
    for col in true_df.columns:
        print(DASH_NB * "-", f"Target: {col}", DASH_NB * "-")
        for site_id in true_df.index.get_level_values('site_id').unique():
            print(DASH_NB * "-", f"Site: {site_id}", DASH_NB * "-")
            print(f"Mean: {true_df.xs(site_id, level='site_id')[col].mean():.6f}")
            for metric in ['MAE', 'NMAE', 'MAPE', 'MSE', 'RMSE', 'NRMSE']:
                metric_values = [
                    all_metrics[fold][site_id][col][metric] for fold in all_metrics
                ]
                print(f"{metric} : {sum(metric_values) / len(metric_values):.6f}")
    if plot:
        best_fold_preds = pd.DataFrame(
            all_preds[best_fold], index=true_df.index, columns=true_df.columns
        )
        true_df, best_fold_preds = denormalize_data(y_scaler, true_df, best_fold_preds, out_size)
        for site in true_df.index.get_level_values('site_id').unique():
            print(DASH_NB * "-", f"Site: {site}", DASH_NB * "-")
            true_site_df = true_df.xs(site, level='site_id')
            preds_site_df = best_fold_preds.xs(site, level='site_id')
            plot_predictions(
                true_site_df, preds_site_df, out_size=out_size, index=true_site_df.index,
            )


def expanding_window_split(total_len, val_size):
    """
    Génère des indices pour un rolling/expanding window cross-validation.
    total_len : longueur totale de la série
    val_size : taille du fold de validation
    """
    folds = []
    train_start = 0

    i = 0
    while True:
        train_end = val_size * (i + 1)
        val_start = train_end
        val_end = val_start + val_size

        # si on dépasse la longueur totale, on ajuste val_end
        if val_start >= total_len:
            break  # plus de données pour la validation
        if val_end > total_len:
            val_end = total_len  # dernier fold plus petit

        folds.append((list(range(train_start, train_end)), list(range(val_start, val_end))))
        i += 1

    return folds


def kfold_cross_validation_ml(
    dataset: pd.HDFStore, target_sites: dict, period2drop: dict, in_size: int, out_size: int,
    n_splits: int, model_type: str, model_name: str, weather: bool = False, ts: bool = False,
    stats: bool = False, selected_features: List[str] = None, selected_target: List[str] = None,
    previous_days: int = 0, step: int = 1, device: str = 'cuda', n_estimators: int = 1000,
    max_depth: int = 5, early_stopping_rounds: int = 10, normalize: bool = False,
    ft_importances: bool = False, explain: bool = False, plot: bool = True
) -> NoReturn:
    """
    Function to apply k-fold cross-validation on the data using XGBoost.

    :param dataset:                 Dataset to use.
    :param target_sites:            Dictionary of target sites.
    :param period2drop:             Dictionary of periods to drop for each site.
    :param in_size:                 Input size.
    :param out_size:                Output size.
    :param n_splits:                Number of splits for cross-validation.
    :param model_type:              Type of model (xgb or svm).
    :param model_name:              Name of the model.
    :param weather:                 Whether to use weather data or not.
    :param ts:                      Whether to use time series features or not.
    :param stats:                   Whether to use statistics features or not.
    :param selected_features:       Selected features to use.
    :param selected_target:         Selected target to use.
    :param previous_days:           Number of previous days to include in the data.
    :param step:                    Step size for the rolling window.
    :param device:                  Device to use (cuda or cpu).
    :param n_estimators:            Number of estimators for XGBoost.
    :param max_depth:               Maximum depth of the trees.
    :param early_stopping_rounds:   Early stopping rounds.
    :param normalize:               Whether to normalize the data or not.
    :param ft_importances:          Whether to show feature importances or not.
    :param explain:                 Whether to explain the model or not.
    :param plot:                    Whether to plot the predictions or not.
    """
    # Define the predictions and metrics.
    all_preds = {}
    all_metrics = {}
    best_fold = ""
    best_mae = float('inf')
    all_y_scaler = {}
    all_train, all_test, true_df = [], [], []
    
    for site_id, sites in target_sites.items():
        processed_data, site_name, sn, prediction_type = load_data(
            dataset, sites, in_size, out_size, weather, ts, stats,
            False, selected_target
        )
        # Drop rows based on period2drop.
        processed_data = drop_useless_perdiod(period2drop, processed_data, site_id)
        # Get the data, features, and target.
        processed_train = (
            processed_data[:int(len(processed_data) * 0.9)][['ap-0']]
            .rename(columns={'ap-0': 'ap'})
        )
        processed_test = (
            processed_data[int(len(processed_data) * 0.9):][['ap-0']]
            .rename(columns={'ap-0': 'ap'})
        )
        train, _, _, _ = prepare_data_ml(
            dataset, site_name, sn, prediction_type, in_size, out_size, processed_train, weather,
            ts, stats, normalize, selected_target, previous_days, step, site_id
        )
        all_train.append(train)
        test, features, target, y_scaler = prepare_data_ml(
            dataset, site_name, sn, prediction_type, in_size, out_size, processed_test, weather,
            ts, stats, normalize, selected_target, previous_days, step, site_id
        )
        all_y_scaler[site_id] = y_scaler
        all_test.append(test)
        # Get the true values for the test set.
        true_df.append(pd.DataFrame(
            test[target], index=test.index, columns=target
        ))
    all_train = pd.concat(all_train)
    all_test = pd.concat(all_test)
    true_df = pd.concat(true_df)
    # For each fold, we need to split the data.
    val_size = int(len(all_train) / (n_splits + 1))
    folds = expanding_window_split(len(all_train), val_size)
    for fold, (train_index, val_index) in enumerate(folds, start=1):
        all_metrics[f'fold_{fold}'] = {}
        train = all_train.iloc[train_index]
        val = all_train.iloc[val_index]
        if selected_features is not None:
            features = selected_features
        # Apply the time series split.
        if model_type == 'xgb':
            y_preds = apply_xgb(
                train, val, all_test, features, target, model_name, device, n_estimators, max_depth, 
                early_stopping_rounds, ft_importances=ft_importances, explain=explain,
                test_implementation=True
            )
        elif model_type == 'svm':
            y_preds = apply_svm(
                train, all_test, features, target
            )
        elif model_type == 'knn':
            y_preds = apply_knn(
                train, all_test, features, target
            )
        # Compute the metrics.
        if out_size > 1 and len(target) == out_size:
            all_metrics, all_preds, best_fold, best_mae = kfold_metrics(
                out_size, all_metrics, all_preds, true_df, fold, y_preds, best_mae,
                best_fold, all_y_scaler
            )
        else:
            all_metrics, all_preds, best_fold, best_mae = kfold_metrics(
                1, all_metrics, all_preds, true_df, fold, y_preds, best_mae,
                best_fold, all_y_scaler
            )
        fold += 1
    if out_size > 1 and len(target) == out_size:
        show_kfold_metrics_plots(
            out_size, all_metrics, all_preds, true_df, best_fold, all_y_scaler, plot
        )
    else:
        show_kfold_metrics_plots(
            1,  all_metrics, all_preds, true_df, best_fold, all_y_scaler, plot
        )


def apply_knn(train: pd.DataFrame, test: pd.DataFrame, features: List[str], target: List[str]):
    """
    Function to apply KNN on the data.

    :param train:                   Training data.
    :param val:                     Validation data.
    :param test:                    Test data.
    :param features:                Features to use.
    :param target:                  Target to predict.
    :param model_name:              Name of the model to save.
    :param ft_importances:          Feature importances.
    :param explain:                 Explain the model.
    :param test_implementation:     Whether to return predictions for the test set.

    :return:                        Predictions for the test set if test_implementation is True.
    """

    model = MultiOutputRegressor(KNeighborsRegressor(
        n_neighbors=50, weights='distance', metric='euclidean', algorithm='auto', n_jobs=-1, p=1
    ))
    model.fit(train[features], train[target].values.ravel())
    knn_preds = model.predict(test[features])
    return knn_preds


def apply_xgb(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, features: List[str],
    target: List[str], model_name: str, device: str = 'cuda', n_estimators: int = 1000,
    max_depth: int = 5, early_stopping_rounds: int = 10, SEED: int = 42,
    ft_importances: bool = False, explain: bool = False, test_implementation: bool = False
):
    """
    Function to apply XGBoost on the data.

    :param train:                   Training data.
    :param val:                     Validation data.
    :param test:                    Test data.
    :param features:                Features to use.
    :param target:                  Target to predict.
    :param model_name:              Name of the model to save.
    :param ft_importances:          Feature importances.
    :param explain:                 Explain the model.
    :param test_implementation:     Whether to return predictions for the test set.

    :return:                        Predictions for the test set if test_implementation is True.
    """
    # Define the XGBoost model.
    model = xgb.XGBRegressor(
        device=device,
        n_estimators=n_estimators,
        max_depth=max_depth,
        early_stopping_rounds=early_stopping_rounds,
        random_state=SEED,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1
    )
    # Fit the model.
    model.fit(
        cudf.DataFrame.from_pandas(train[features]),
        cudf.DataFrame.from_pandas(train[target]),
        eval_set=[(
            cudf.DataFrame.from_pandas(val[features]),
            cudf.DataFrame.from_pandas(val[target])
        )],
        verbose=0
    )
    # Save the model.
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    model.save_model(f'{SAVED_MODELS_DIR}/{model_name}.json')
    # Print the feature importances if requested.
    if ft_importances:
        for feature in features:
            print(f"{feature}: {model.feature_importances_[features.index(feature)]}")
    # Plot the feature importances.
    if explain:
        # Explain trees.
        explainer = shap.TreeExplainer(model)
        explanation = explainer(train[features])
        # Get the SHAP values.
        shap_values = explanation.values
        # Plot the SHAP values.
        shap.summary_plot(shap_values, train[features], max_display=50)
    # Do predictions.
    if test_implementation:
        return model.predict(cudf.DataFrame.from_pandas(test[features]))


def apply_svm(train, test, features, target):
    """
    Function to apply SVM on the data.

    :param train:       Training data.
    :param test:        Test data.
    :param features:    Features to use.
    :param target:      Target to predict.

    :return:            Predictions.
    """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.01, epsilon=0.1, verbose=True, shrinking=False)))
    ])
    model.fit(train[features], train[target])
    svr_preds = model.predict(test[features])
    return svr_preds


def sarima(data):
    dataset = pd.HDFStore(f'{PROCESSED_DATA_DIR}/aieg.h5', "r")
    # data = dataset['/aieg_AMR_ANDENN_0/consumption'].set_index('ts')['ap']
    data = dataset['/aieg_ADMCOMMU_212303507/production'].set_index('ts')['ap']
    data.index = pd.to_datetime(data.index)
    data = data.asfreq('15min')
    data_size = len(data)
    train = data[:int(data_size * 0.8)]
    test = data[int(data_size * 0.8):]
    order = (4, 2, 1)  # (p, d, q)
    seasonal_order = (1, 1, 0, 48)  # (P, D, Q, S) need to try 672
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    y_hat = model_fit.forecast(steps=1)
    print('Predicting the next value:', y_hat)
    mae = mean_absolute_error(np.array([test.iloc[0]]), y_hat)
    mse = mean_squared_error(np.array([test.iloc[0]]), y_hat)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((np.array([test.iloc[0]]) - y_hat) / np.array([test.iloc[0]]))) * 100
    print(f'MAE: {mae:.3f}')
    print(f'MSE: {mse:.3f}')
    print(f'RMSE: {rmse:.3f}')
    print(f'MAPE: {mape:.3f}%')


def arima():
    dataset = pd.HDFStore(f'{PROCESSED_DATA_DIR}/aieg.h5', "r")
    # data = dataset['/aieg_AMR_ANDENN_0/consumption'].set_index('ts')['ap']
    data = dataset['/aieg_ADMCOMMU_212303507/production'].set_index('ts')['ap']
    data.index = pd.to_datetime(data.index)
    data = data.asfreq('15min')
    data_size = len(data)
    train = data[:int(data_size * 0.8)]
    test = data[int(data_size * 0.8):]
    test_shifted = test.copy()
    # 288, 384, 480, 576, 672
    seq_length, out = 672, 1

    for t in range(1, out):
        test_shifted['ap+'+str(t)] = test_shifted['ap'].shift(-t)
    test_shifted = test_shifted.dropna()
    history = [x for x in train]
    history = history[(-seq_length):]
    # p, d, q, = 1, determine_d(train), 1
    p, d, q = 4, 2, 1
    # (4,2,1) -> better mape 0.540
    # (2,1,2) -> better mape 0.566
    # (4,1,1) -> 0.633%
    # (4,1,2) -> 0.549
    predictions=[]
    train_pbar = tqdm(enumerate(test_shifted), total=len(test_shifted), desc=f'Training')
    for t, _ in train_pbar:
        # model = ARIMA(history, order=(p, d, q))
        model = SARIMAX(history, order=(p, d, q), seasonal_order=(1, 1, 0, 48))
        model_fit = model.fit()
        yhat = model_fit.forecast(steps=out)
        predictions.append(yhat[0])
        obs = test[t]
        history.append(obs)
        history.pop(0)
        # print(model_fit.summary())

    # forecast_steps = len(test)
    # test_pred = model_fit.forecast(steps=1)
    # #
    # plt.figure(figsize=(10, 6))
    # #
    # plt.plot(range(len(train)), train, label="Train data", color="blue")
    # plt.plot(range(len(train)), train_pred, label="Train Predictions", color="orange",
    #          linestyle="--")
    
    # # Affichage des vraies valeurs et des prédictions sur les données de test
    # plt.plot(range(len(train), len(data)), test, label="Test data", color="green")
    # plt.plot(range(len(train), len(data)), test_pred, label="Test Predictions", color="red",
    #          linestyle="--")
    
    # plt.legend()
    # plt.title("ARIMA Predictions")
    # plt.show()
    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    
    print(f'MAE: {mae:.3f}')
    print(f'MSE: {mse:.3f}')
    print(f'RMSE: {rmse:.3f}')
    print(f'MAPE: {mape:.3f}%')
    # Best MAPE: 0.020
    # Best p: 0
    # Best d: 2
    # Best q: 0

    # Best MAPE: 0.019
    # Best p: 0
    # Best d: 3
    # Best q: 1



    # best_mape = 100
    # best_p, best_d, best_q = 0, 0, 0
    # for p in range(3, 13, 1):
    #     print(20 * '-', f'p: {p}', 20 * '-')
    #     for d in range(0, 4, 1):
    #         print(20 * '-', f'd: {d}', 20 * '-')
    #         for q in range(0, 13, 1):
    #             print(20 * '-', f'q: {q}', 20 * '-')
    #             model = ARIMA(train, order=(p, d, q))
    #             model_fit = model.fit()
    #             y_hat = model_fit.forecast(steps=1)
    #             mape = np.mean(
    #                 np.abs((np.array([test.iloc[0]]) - y_hat) / np.array([test.iloc[0]]))
    #             ) * 100
    #             if mape < best_mape:
    #                 best_mape = mape
    #                 best_p, best_d, best_q = p, d, q
    #                 print(f'Best MAPE: {best_mape:.3f}')
    #                 print(f'Best p: {best_p}')
    #                 print(f'Best d: {best_d}')
    #                 print(f'Best q: {best_q}')
    # print('Best results overall:')
    # print('Best MAPE:', best_mape)
    # print('Best p:', best_p)
    # print('Best d:', best_d)
    # print('Best q:', best_q)
    # print('Predicting the next value:', y_hat)
    # mae = mean_absolute_error(np.array([test.iloc[0]]), y_hat)
    # mse = mean_squared_error(np.array([test.iloc[0]]), y_hat)
    # rmse = np.sqrt(mse)
    # mape = np.mean(np.abs((np.array([test.iloc[0]]) - y_hat) / np.array([test.iloc[0]]))) * 100
    # print(f'MAE: {mae:.3f}')
    # print(f'MSE: {mse:.3f}')
    # print(f'RMSE: {rmse:.3f}')
    # print(f'MAPE: {mape:.3f}%')
    dataset.close()


def determine_d(data: pd.Series) -> int:
    """
    Function to determine the number of differences needed to make the series stationary.

    :param data: Time series data.

    :return: Number of differences needed (d).
    """
    d = 0
    adf_result = adfuller(data)
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')

    while adf_result[1] > 0.05:
        d += 1
        data = data.diff().dropna()
        adf_result = adfuller(data)
        print(f'ADF Statistic after {d} differencing: {adf_result[0]}')
        print(f'p-value after {d} differencing: {adf_result[1]}')
    return d


def test_stats_arima():
    """
    Function to find ARIMA parameters.

    p is the number of lag observations included in the model, also called the lag order
    (number of auto-regressive terms).

    d is the number of times that the raw observations are differenced, also called the degree of
    differencing (number of nonseasonal differences).

    q is the size of the moving average window, also called the order of moving average
    (number of lagged forecast errors in the prediction equation).

    First of all, we need to test for stationarity (using the Dickey-Fuller test). If the series is
    not stationary, we need to differentiate it.

    We need to draw the ACF (AutoCorrelation Function) and PACF (Partial AutoCorrelation Function)
    of the data (results of the Dickey-Fuller test, test of stationarity). This will help us in
    finding the value of p because the cut-off point to the PACF is p and the cut-off point to the
    ACF is q.

    To find p (or AR term):
    The lollipop plot that you see above is the ACF and PACF results. To estimate the amount of AR
    terms, you need to look at the PACF plot. First, ignore the value at lag 0. It will always show
    a perfect correlation, since we are estimating the correlation between today’s value with
    itself. Note that there is a blue area in the plot, representing the confidence interval.
    To estimate how much AR terms you should use, start counting how many “lollipop” are above or
    below the confidence interval before the next one enter the blue area.

    To find d (or I term):
    This is an easy part. All you need to do to estimate the amount of I (d) terms is to know how
    many Differencing was used to make the series stationary. For example, if you used log
    difference or first difference to transform a time series, the amount of I terms will be 1,
    since Arauto takes the difference between the actual value (e.g. today’s value) and 1 previous
    value (e.g. yesterday’s value).

    To find q (or MA term):
    Just like the PACF function, to estimate the amount of MA terms, this time you will look at
    ACF plot. The same logic is applied here: how much lollipops are above or below the confidence
    interval before the next lollipop enters the blue area?

    :return:
    """
    dataset = pd.HDFStore(f'{PROCESSED_DATA_DIR}/aieg.h5', "r")
    # Get the data
    data = dataset['/aieg_AMR_ANDENN_0/consumption'].set_index('ts')['ap']
    # Test de stationnarité (Dickey-Fuller)
    adf_result = adfuller(data)
    print('ADF Statistic:', adf_result[0])
    print('p-value:', adf_result[1])

    # Compute ACF and PACF
    lag_acf = acf(data, nlags=20)
    lag_pacf = pacf(data, nlags=20, method='ols')

    # Plot ACF
    plot_acf(data, lags=20)
    plot_acf(data.diff().dropna(), lags=20)
    # plot_acf(data.diff().diff(12).dropna(), lags=20)
    # Plot PACF
    plot_pacf(data, lags=20)
    plot_pacf(data.diff().dropna(), lags=20)
    # plot_pacf(data.diff().diff(12).dropna(), lags=20)
    plt.show()
    print(determine_d(data))
    dataset.close()


def train_selected_models(site_name: str, sn: str, prediction_type: str, params: dict, weather_data: bool = False, ts_features: bool = False) -> NoReturn:
    """
    Function to train selected models with specific parameters.

    :param site_name:       Name of the site.
    :param sn:              Serial Number.
    :param prediction_type: Type of prediction (consumption or production).
    :param params:          Parameters for the models.
    """
    # Create a DataFrame to store the results.
    results_df = pd.DataFrame(columns=["model", "in", "out", "mae", "mape", "wmape", "mse", "rmse"])
    # Load the data.
    dataset = pd.HDFStore(f'{PROCESSED_DATA_DIR}/aieg.h5', "r")
    features_nb = 18 if weather_data else 1
    # data = dataset['/aieg_AMR_ANDENN_0/consumption'].set_index('ts')[['ap']]
    # For each model type.
    for model_type, param in params.items():
        print('-' * DASH_NB, f'Training model type: {model_type}', '-' * DASH_NB)
        # Check if the model needs standardization.
        # standardize = False if model_type in ['Seq2Point', 'UNetNilm', 'xgboost'] else True
        standardize = True
        # For each sequence length (window size).
        for i in range(len(param['in'])):
            print('-' * DASH_NB, f'In: {param["in"][i]}, Out: {param["out"][i]}', '-' * DASH_NB)
            # Check if the model is a deep learning model.
            if model_type in ['GRU', 'CNN-GRU', 'LSTM', 'RNN', 'Seq2Point', 'UNetNilm']:
                # Split the dataset.
                train, val, test = temporal_split(
                    dataset, param['in'][i], param['out'][i],
                    target=[f'/aieg_{site_name}_{sn}/{prediction_type}']
                )
                # Create the generators for different sets.
                train_gen = PyTorchDataGenerator(
                    model_type, dataset, train, param['in'][i], param['out'][i],
                    'train', standardize=standardize, is_weather=weather_data
                )
                val_gen = PyTorchDataGenerator(
                    model_type, dataset, val, param['in'][i], param['out'][i],
                    'val', standardize=standardize, is_weather=weather_data
                )
                test_gen = PyTorchDataGenerator(
                    model_type, dataset, test, param['in'][i], param['out'][i],
                    'test', standardize=standardize, is_weather=weather_data
                )
                # Create the trainer.
                trainer = PyTorchTrainer(
                    f'aieg_model_{model_type}_{prediction_type}_in{param["in"][i]}'
                    f'_out{param["out"][i]}', model_type, features_nb, param['in'][i],
                    param['in'][i], param['out'][i], param['lr'], param['epochs'],
                    param['batch'][i], param['patience'], param['hidden'][i], param['layers'][i]
                )
                # Train the model.
                trainer.train(train_gen, val_gen)
                # Save the model.
                trainer.save_model()
                # Load the model.
                # trainer.load_model()
                # Evaluate the model.
                evaluation_result = trainer.evaluate(test_gen)
                print(evaluation_result)
                if results_df.empty:
                    results_df = pd.DataFrame([[
                            model_type, param['in'][i], param['out'][i], evaluation_result[1],
                            evaluation_result[2], evaluation_result[3], evaluation_result[0],
                            np.sqrt(evaluation_result[0])
                        ]],
                        columns=["model", "in", "out", "mae", "mape", "wmape", "mse", "rmse"]
                    )
                else:
                    results_df = pd.concat(
                        [
                            results_df,
                            pd.DataFrame([[
                                model_type, param['in'][i], param['out'][i], evaluation_result[1],
                                evaluation_result[2], evaluation_result[3], evaluation_result[0],
                                np.sqrt(evaluation_result[0])]],
                                columns=["model", "in", "out", "mae", "mape", "wmape", "mse", "rmse"]
                            )
                        ], ignore_index=True
                    )
                # Do the predictions.
                predictions = trainer.predict(test_gen)
                # predictions = [y if y > 0 else 0 for y in predictions]
                # Convert predictions into a numpy array.
                y_true = np.array([y for _, y in tqdm(test_gen, desc='Loading y_true')])
                print_metrics(y_true, predictions)
                # predictions = trainer.load_predictions()
                # Plot predictions.
                trainer.plot_results(predictions, test_gen)
            # Check if the model is a tree-based model.
            elif model_type == 'xgboost':
                # Get the data, features, and target.
                data, features, target = prepare_data_ml(
                    dataset, site_name, sn, prediction_type, param['in'][i], param['out'][i],
                    weather_data=weather_data, ts_features=ts_features
                )
                data = data.drop(data["2024-01-29":"2024-03-23"].index)
                # Get the size of the dataset to split it in different sets (train, val, test).
                data_size = len(data)
                train = data[:int(data_size * 0.8)]
                val = data[int(data_size * 0.8):int(data_size * 0.9)]
                test = data[int(data_size * 0.9):]
                # Create the trainer.
                trainer = PyTorchTrainer(
                    f'aieg_model_{model_type}_{prediction_type}_in{param["in"][i]}_'
                    f'out{param["out"][i]}', model_type, features_nb, param['in'][i], 0,
                    param['out'][i], 0, 0, 0, param['patience'], 0, 0
                )
                # Create the model.
                model = xgb.XGBRegressor(
                    device='cuda', n_estimators=param['n_estimators'],
                    max_depth=param['max_depth'], early_stopping_rounds=param['patience']
                )
                # Train the model.
                model.fit(
                    train[features], train[target], eval_set=[(val[features], val[target])],
                    verbose=0
                )
                # model_features = model.feature_importances_
                # print(features)
                # # Check the features importances.
                # for j in range(len(features)):
                #     if model_features[j] != 0:
                #         print(f'{features[j]}: {model_features[j]}')
                # Do predictions.
                y_pred = model.predict(test[features])
                # y_pred = [y if y > 0 else 0 for y in y_pred]
                print_metrics(test[target], y_pred)
                # Create a df with results.
                if results_df.empty:
                    results_df = pd.DataFrame([[
                        model_type, param['in'][i], param['out'][i],
                        compute_mae(test[target], y_pred),
                        compute_mape(test[target], y_pred),
                        compute_wmape(test[target], y_pred),
                        compute_mse(test[target], y_pred, squared=False),
                        compute_mse(test[target], y_pred)]],
                        columns=["model", "in", "out", "mae", "mape", "wmape", "mse", "rmse"]
                    )
                else:
                    results_df = pd.concat(
                        [
                            results_df,
                            pd.DataFrame([[
                                model_type, param['in'][i], param['out'][i],
                                compute_mae(test[target], y_pred),
                                compute_mape(test[target], y_pred),
                                compute_wmape(test[target], y_pred),
                                compute_mse(test[target], y_pred, squared=False),
                                compute_mse(test[target], y_pred)]],
                                columns=["model", "in", "out", "mae", "mape", "wmape", "mse", "rmse"]
                            )
                        ], ignore_index=True
                    )
                plot_predictions(
                    test[target], y_pred, param['out'][i]
                )
    # Save the results in csv file.
    results_df.to_csv(f'{PROCESSED_DATA_DIR}/results_{site_name}_{sn}_{prediction_type}.csv')
    dataset.close()


def get_features_nb(
    dataset: pd.HDFStore, sites_grouped: Dict[int, List[str]],
    drop_period: Dict[int, List[Tuple[str, str]]], in_size: int, out_size: int, stats: bool,
    ts: bool, weather: bool
) -> int:
    """
    Function to get the number of features from the dataset.
    
    :param dataset:     Dataset to use.
    :param in_size:     Size of the input.
    :param out_size:    Size of the output.
    :param stats:       Whether to use statistics features or not.
    :param ts:          Whether to use time series features or not.
    :param weather:     Whether to use weather data or not.

    :return:            Returns the number of features.
    """
    target_site = [
        (site_id, site) if len(site) > 1 else (site_id, site[0])
        for site_id, site in sites_grouped.items()
    ]
    k_folds_sites = kfold_temporal_split(
        dataset, in_size, out_size, k=2,
        target=target_site, period2drop=drop_period
    )
    based_train_gen = PyTorchDataGenerator(
        dataset, k_folds_sites[-1][0], in_size, out_size, is_stats_features=stats,
        is_ts_features=ts, is_weather=weather, previous_days=1
    )
    return based_train_gen[0][0].shape[1]


def get_max_value(
    dataset: pd.HDFStore, site_id: int, site: List[str],
    drop_period: Dict[int, List[Tuple[str, str]]], in_size: int, out_size: int
) -> float:
    """
    Function to get the maximum value from the dataset for a specific site.
    
    :param dataset:         The dataset to use.
    :param site_id:         Identifier of the site.
    :param site:            List of site names.
    :param drop_period:     Dictionary of periods to drop.
    :param in_size:         Size of the input.
    :param out_size:        Size of the output.

    :return:                Maximum value from the dataset for the specific site.
    """
    processed_data, _, _, _ = load_data(dataset, site, in_size, out_size)
    processed_data = drop_useless_perdiod(drop_period, processed_data, site_id)
    return processed_data['ap-0'].max()


def selected_models_kfold_cv() -> NoReturn:
    """
    Function to perform k-fold cross-validation on selected models with different configurations.

    :return:    None
    """
    # Define the parameters for the training.
    batch_size, num_epochs, patience = 128, 300, 10
    learning_rate = 1e-3
    criterion = nn.MSELoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kfold = 10
    # Load the dataset.
    dataset = pd.HDFStore(f'{PROCESSED_DATA_DIR}/aieg.h5', "r")
    # Define the sites grouped by energy type.
    energy_type = {
        'consumption': [consumption_sites_grouped, consumption_drop_period],
        'production': [production_sites_grouped, production_drop_period]
    }
    results_df = pd.DataFrame(columns=["model", "energy", "siteid", "in", "out", "ts", "stats", "weather", "mae", "nmae"])
    for model in ['cnn_gru', 'transformer', 'xgb']:
        print('-' * DASH_NB, f'Starting model: {model}', '-' * DASH_NB)
        for energy, (sites, drop_period) in energy_type.items():
            print('-' * DASH_NB, f'Energy type: {energy}', '-' * DASH_NB)
            for in_size in [8, 96, 672]:
                print('-' * DASH_NB, f'Input size: {in_size}', '-' * DASH_NB)
                for out_size in [1, 4, 8, 12]:
                    print('-' * DASH_NB, f'Output size: {out_size}', '-' * DASH_NB)
                    for ts, stats, weather in product([False, True], repeat=3):
                        print(
                            '-' * DASH_NB,
                            f'TS: {ts}, Stats: {stats}, Weather: {weather}',
                            '-' * DASH_NB
                        )
                        for site_id, site in sites.items():
                            print('-' * DASH_NB, f'Site ID: {site_id}', '-' * DASH_NB)
                            # Build model_name including only the present feature flags
                            base_name = f'{model}_site{site_id}_{energy}_in{in_size}_out{out_size}'
                            suffixes = []
                            if ts:
                                suffixes.append('ts')
                            if stats:
                                suffixes.append('stats')
                            if weather:
                                suffixes.append('weather')
                            model_name = base_name + ('_' + '_'.join(suffixes) if suffixes else '')
                            if model in ['xgb', 'svm', 'knn']:
                                all_metrics, best_fold = kfold_cross_validation_ml(
                                    dataset, {site_id : site}, {site_id: drop_period[site_id]},
                                    in_size, out_size, kfold, model, model_name,
                                    device=device, stats=stats, ts=ts, weather=weather,
                                    previous_days=1, plot=False
                                )
                            else:
                                features_nb = get_features_nb(
                                    dataset, {site_id : site}, drop_period, in_size, out_size,
                                    stats, ts, weather
                                )
                                max_value = get_max_value(
                                    dataset, site_id, site, drop_period, in_size, out_size
                                )
                                all_metrics, best_fold = kfold_cross_validation_dl(
                                    dataset, {site_id : sites[site_id]},
                                    {site_id: drop_period[site_id]}, kfold, in_size, out_size,
                                    features_nb, batch_size, num_epochs, patience, learning_rate,
                                    criterion, device, model_name, max_value, model,
                                    stats=stats, ts=ts, weather=weather, previous_days=1,
                                    plot=False
                                )

                            if results_df.empty:
                                results_df = pd.DataFrame([[
                                    model, energy, site_id, in_size, out_size, ts, stats, weather,
                                    all_metrics[best_fold]['overall']['all']['MAE'],
                                    all_metrics[best_fold]['overall']['all']['NMAE']]],
                                    columns=[
                                        "model", "energy", "siteid", "in", "out", "ts", "stats",
                                        "weather", "mae", "nmae"
                                    ]
                                )
                            else:
                                results_df = pd.concat(
                                    [
                                        results_df,
                                        pd.DataFrame([[
                                            model, energy, site_id, in_size, out_size, ts, stats,
                                            weather,
                                            all_metrics[best_fold]['overall']['all']['MAE'],
                                            all_metrics[best_fold]['overall']['all']['NMAE']]],
                                            columns=[
                                                "model", "energy", "siteid", "in", "out", "ts", "stats",
                                                "weather", "mae", "nmae"
                                            ]
                                        )
                                    ], ignore_index=True
                                )
                            results_df.to_csv(f'{PROCESSED_DATA_DIR}/results.csv')


def main() -> NoReturn:
    """
    Main function to train and test the models.
    """
    selected_models_kfold_cv()


if __name__ == '__main__':
    main()