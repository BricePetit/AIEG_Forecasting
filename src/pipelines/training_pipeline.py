"""
Training pipeline.
"""

__title__: str = "training_pipeline"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports from standard library
import logging
import os
from pathlib import Path
import yaml

# Imports from third party libraries
import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import xgboost as xgb

# Imports from src
from configs.config_loader import ConfigLoader
from data.dataset_loader import build_group_data
from data.preprocessing import prepare_data_ml, temporal_split
from data_generator.pytorch_generators import PyTorchDataGenerator
from evaluation.metrics import print_metrics
from models.pytorch_models import PyTorchComplexCNNGRU
from utils.logging import setup_logger
from utils.seeds import set_seed
from xgboost_utils import clean_data_for_xgb

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------- Globals --------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    setup_logger(log_file="training_pipeline.log", level=logging.INFO)

config_loader = ConfigLoader()
config = config_loader.load_global()

DASH = '-' * 20

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ Functions ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #


def save_best_params(
    site_id: int, best_params: dict, prediction_type: str, config_dir: str = "src/configs"
) -> None:
    """
    Function to save the best parameters found by Optuna for a given site.

    :param site_id:     ID of the site for which the best parameters were found.
    :param best_params: Best parameters found by Optuna.
    :param prediction_type:     Type of prediction to make.
    :param config_dir:  Directory where the best parameters will be saved.
    """
    logger.info("Saving best parameters for site %s: %s", site_id, best_params)
    path = Path(config_dir) / f"best_params/{prediction_type}/site_{site_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(best_params, f)


def train_model(
        model: PyTorchComplexCNNGRU, train_gen: PyTorchDataGenerator,
        val_gen: PyTorchDataGenerator, criterion: nn.MSELoss, optimizer: optim.Adam,
        model_name: str, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None
    ) -> tuple[PyTorchComplexCNNGRU, dict[str, list], float]:
    """
    Function to train the model.

    :param model:           Model to train.
    :param train_gen:       Training data generator.
    :param val_gen:         Validation data generator.
    :param criterion:       Loss function.
    :param optimizer:       Optimizer.
    :param model_name:      Name of the model.
    :param scheduler:       Learning rate scheduler.

    :return:                Trained model and training history.
    """
    # Create DataLoaders.
    train_loader = DataLoader(
        train_gen, batch_size=config['model']['model']['cnn_gru']['batch_size'],
        num_workers=config['model']['model']['cnn_gru']['num_workers'], pin_memory=True,
        shuffle=True, persistent_workers=True, prefetch_factor=4, drop_last=True
    )
    val_loader = DataLoader(
        val_gen, batch_size=config['model']['model']['cnn_gru']['batch_size'],
        num_workers=config['model']['model']['cnn_gru']['num_workers'], pin_memory=True,
        shuffle=False, persistent_workers=True, prefetch_factor=4, drop_last=False
    )
    # Create save directory if it doesn't exist.
    os.makedirs(config['paths']['paths']['saved_models_dir'], exist_ok=True)
    model_path = os.path.join(
        config['paths']['paths']['saved_models_dir'], f'best_{model_name}_model.pt'
    )
    amp_enabled = (torch.cuda.is_available() and str(device).startswith("cuda"))
    amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    # Move model to device.
    model = model.to(device, non_blocking=True)
    # Initialize variables for early stopping.
    best_val_loss = float('inf')
    early_stopping_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
    # Training loop.
    for epoch in range(config['model']['model']['cnn_gru']['num_epochs']):
        # Training phase.
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            # Move data to device.
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # Forward pass.
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(inputs)
            loss = criterion(outputs.float(), targets.float())
            # Backward pass.
            # loss.backward()
            scaler.scale(loss).backward()
            # Gradient clipping (helps stabilize training).
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        # Average training loss.
        train_loss = train_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        # Validation phase.
        model.eval()
        val_loss = 0.0
        # Disable gradient calculation for validation.
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device.
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                # Forward pass.
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    outputs = model(inputs)
                loss = criterion(outputs.float(), targets.float())
                val_loss += loss.item()
        # Average validation loss.
        val_loss = val_loss / len(val_loader)
        history['val_loss'].append(val_loss)
        # Store current learning rate.
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
        # Early stopping.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if early_stopping_counter >= config['model']['model']['cnn_gru']['patience']:
            logger.warning(
                "Early stopping triggered at epoch %d/%d",epoch + 1,
                config['model']['model']['cnn_gru']['num_epochs']
            )
            break
        # Print progress.
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d, Train Loss: %.6f, Val Loss: %.6f, LR: %.6f",
                epoch + 1,
                config['model']['model']['cnn_gru']['num_epochs'],
                train_loss,
                val_loss,
                current_lr,
            )
    # Load the best model.
    model.load_state_dict(torch.load(model_path, weights_only=True))
    logger.info("Best model saved with validation loss: %.6f", best_val_loss)
    return model, history, best_val_loss


# def predict_model(
#     model: PyTorchComplexCNNGRU, test_gen: PyTorchDataGenerator, out_size: int, index=None,
#     kfold_cv: bool = False
# ):
#     """
#     Function to make predictions using the model on a given dataset.

#     :param model:       Trained model.
#     :param test_gen:    Test data generator.
#     :param out_size:    Output size.
#     :param index:       Optional index for the predictions.
#     :param kfold_cv:    Whether to use k-fold cross-validation or not.

#     :return:            Predictions.
#     """
#     # Set the model to evaluation mode.
#     model.eval()
#     # AMP settings for inference.
#     amp_enabled = (torch.cuda.is_available() and str(device).startswith("cuda"))
#     amp_dtype = torch.bfloat16
    
#     # Create the test DataLoader.
#     test_loader = DataLoader(
#         test_gen, batch_size=64, shuffle=False, num_workers=16, pin_memory=True,
#         persistent_workers=True, prefetch_factor=4, drop_last=False
#     )
#     # Create lists to hold the predictions and true values.
#     all_preds, all_true = [], []
#     with torch.no_grad():
#         for X_test, y_test in test_loader:
#             # Move data to device.
#             X_test = X_test.to(device, non_blocking=True)
#             # Make predictions.
#             with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
#                 y_preds = model(X_test)
#             y_preds = (
#                 y_preds.detach().float().cpu().numpy()
#                 if isinstance(y_preds, torch.Tensor)
#                 else np.asarray(y_preds, dtype=np.float32)
#             )
#             y_test = (
#                 y_test.detach().float().cpu().numpy()
#                 if isinstance(y_test, torch.Tensor)
#                 else np.asarray(y_test, dtype=np.float32)
#             )
#             # Append the predictions and true values to the lists.
#             all_preds.append(y_preds)
#             all_true.append(y_test)
#     # Concatenate all predictions and true values.
#     y_pred = np.vstack(all_preds)
#     y_true = np.vstack(all_true)
#     if not kfold_cv:
#         # Check if the predictions and true values are multi-step or single-step.
#         if out_size > 1:
#             # For the multi-step predictions.
#             # Convert to DataFrame for multi-step predictions.
#             if index is not None:
#                 true_df = pd.DataFrame(
#                     y_true, index=index,
#                     columns=[f't+{i+1}' for i in range(y_true.shape[1])]
#                 )
#             else:
#                 true_df = pd.DataFrame(
#                     y_true, 
#                     columns=[f't+{i+1}' for i in range(y_true.shape[1])]
#                 )
#             # Show the metrics for each step.
#             for i in range(min(y_true.shape[1], y_pred.shape[1])):
#                 col = f't+{i+1}'
#                 logger.info("--- Metrics for %s ---", col)
#                 print_metrics(true_df[col], y_pred[:, i])
#             # Show the metrics for all steps.
#             logger.info("--- Metrics for all steps ---")
#             print_metrics(true_df.values.flatten(), y_pred.flatten())
#             # Plot the predictions.
#             # plot_predictions(true_df, y_pred, out_size=y_true.shape[1], index=true_df.index)
#         else:
#             # For single-step predictions.
#             # Flatten the predictions and true values.
#             y_pred = y_pred.flatten()
#             y_true = y_true.flatten()
#             # Convert to DataFrame for single-step predictions.
#             if index is not None:
#                 results = pd.DataFrame(
#                     {'true': y_true, 't+1': y_pred}, index=index
#                 )
#             else:
#                 results = pd.DataFrame({'true': y_true, 't+1': y_pred})
#             # Check if the index is a MultiIndex.
#             if isinstance(index, pd.MultiIndex):
#                 # If so, group by the site_id and show the metrics for each house.
#                 logger.info("--- Metrics for all houses ---")
#                 print_metrics(results['true'], results['t+1'])
#                 for site_id in results.index.get_level_values("site_id").unique():
#                     house_df = results.xs(site_id, level="site_id")
#                     logger.info("--- Metrics for house id %s ---", site_id)
#                     # Show the metrics.
#                     print_metrics(house_df['true'], house_df['t+1'])
#                     # Plot the predictions.
#                     # plot_predictions(
#                     #     house_df['true'], house_df['t+1'], out_size=1, index=house_df.index
#                     # )
#     return y_pred


def evaluate_cnn_gru_model(
    model: PyTorchComplexCNNGRU, test_gen: PyTorchDataGenerator, kfold_cv: bool = False,
    index: pd.Index = None
) -> None:
    """
    Function to evaluate the CNN-GRU model on the test set.

    :param model:       Trained CNN-GRU model.
    :param test_gen:    Test data generator.
    :param kfold_cv:    Whether to use k-fold cross-validation.
    :param index:       Index for the results DataFrame.
    """
    # Set the model to evaluation mode.
    model.eval()
    # AMP settings for inference.
    amp_enabled = (torch.cuda.is_available() and str(device).startswith("cuda"))
    amp_dtype = torch.bfloat16
    # Create the test DataLoader.
    test_loader = DataLoader(
        test_gen, batch_size=config['model']['model']['cnn_gru']['batch_size'], 
        shuffle=False, num_workers=config['model']['model']['cnn_gru']['num_workers'],
        pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=False
    )
    # Create lists to hold the predictions and true values.
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            # Move data to device.
            X_test = X_test.to(device, non_blocking=True)
            # Make predictions.
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                y_preds = model(X_test)
            y_preds = (
                y_preds.detach().float().cpu().numpy()
                if isinstance(y_preds, torch.Tensor)
                else np.asarray(y_preds, dtype=np.float32)
            )
            y_test = (
                y_test.detach().float().cpu().numpy()
                if isinstance(y_test, torch.Tensor)
                else np.asarray(y_test, dtype=np.float32)
            )
            # Append the predictions and true values to the lists.
            all_preds.append(y_preds)
            all_true.append(y_test)
    # Concatenate all predictions and true values.
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_true)
    if not kfold_cv:
        # Check if the predictions and true values are multi-step or single-step.
        if config['data']['data']['horizon'] > 1:
            # For the multi-step predictions.
            # Convert to DataFrame for multi-step predictions.
            if index is not None:
                true_df = pd.DataFrame(
                    y_true, index=index,
                    columns=[f't+{i+1}' for i in range(y_true.shape[1])]
                )
            else:
                true_df = pd.DataFrame(
                    y_true, 
                    columns=[f't+{i+1}' for i in range(y_true.shape[1])]
                )
            # Show the metrics for each step.
            for i in range(min(y_true.shape[1], y_pred.shape[1])):
                col = f't+{i+1}'
                logger.info("--- Metrics for %s ---", col)
                print_metrics(true_df[col], y_pred[:, i])
            # Show the metrics for all steps.
            logger.info("--- Metrics for all steps ---")
            print_metrics(true_df.values.flatten(), y_pred.flatten())
            # Plot the predictions.
            # plot_predictions(true_df, y_pred, out_size=y_true.shape[1], index=true_df.index)
        else:
            # For single-step predictions.
            # Flatten the predictions and true values.
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
            # Convert to DataFrame for single-step predictions.
            if index is not None:
                results = pd.DataFrame(
                    {'true': y_true, 't+1': y_pred}, index=index
                )
            else:
                results = pd.DataFrame({'true': y_true, 't+1': y_pred})
            # Check if the index is a MultiIndex.
            if isinstance(index, pd.MultiIndex):
                # If so, group by the site_id and show the metrics for each house.
                logger.info("--- Metrics for all houses ---")
                print_metrics(results['true'], results['t+1'])
                for site_id in results.index.get_level_values("site_id").unique():
                    house_df = results.xs(site_id, level="site_id")
                    logger.info("--- Metrics for house id %s ---", site_id)
                    # Show the metrics.
                    print_metrics(house_df['true'], house_df['t+1'])
                    # Plot the predictions.
                    # plot_predictions(
                    #     house_df['true'], house_df['t+1'], out_size=1, index=house_df.index
                    # )


def tuning_cnn_gru(
    train_gen: PyTorchDataGenerator, val_gen: PyTorchDataGenerator, features_nb: int,
    criterion: nn.Module, model_name: str
) -> dict:
    """
    Function to perform hyperparameter tuning for the CNN-GRU model using Optuna.

    :param train_gen:       Training data generator.
    :param val_gen:         Validation data generator.
    :param features_nb:     Number of features.
    :param criterion:       Loss function.
    :param model_name:      Name of the model.

    :return:                Best hyperparameters found by Optuna.
    """
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna to minimize the validation loss of the CNN-GRU model.

        :param trial:   Optuna trial object.

        :return:        Validation loss for the current set of hyperparameters.
        """
        # Suggest hyperparameters for the CNN-GRU model.
        cnn_depth = trial.suggest_int("cnn_depth", 2, 4)
        cnn_channels = [
            trial.suggest_int(f"cnn_channels_{i}", 32, 256)
            for i in range(cnn_depth)
        ]
        params = {
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
            "gru_hidden_size": trial.suggest_int("gru_hidden", 64, 512),
            "gru_layers": trial.suggest_int("gru_layers", 1, 3),
            "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "pool_size": trial.suggest_int("pool_size", 2, 8),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }
        # Create the model with the suggested hyperparameters.
        model = PyTorchComplexCNNGRU(
            n_features=features_nb,
            output_size=config['data']['data']['horizon'],
            cnn_channels=cnn_channels,
            kernel_size=params["kernel_size"],
            gru_hidden_size=params["gru_hidden_size"],
            gru_layers=params["gru_layers"],
            bidirectional=params["bidirectional"],
            dropout=params["dropout"],
            pool_size=params["pool_size"]
        ).to(device, non_blocking=True)
        # Define the optimizer and learning rate scheduler.
        cnn_gru_optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        cnn_gru_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            cnn_gru_optimizer, mode='min', factor=0.5,
            patience=config['model']['model']['cnn_gru']['patience'] // 2
        )
        # Train the model and return the validation loss.
        model, _, val_loss = train_model(
            model, 
            train_gen, 
            val_gen, 
            criterion, 
            cnn_gru_optimizer, 
            model_name, 
            scheduler=cnn_gru_scheduler,
            device=device
        )
        return val_loss
    # Create an Optuna study and optimize the objective function.
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config['model']['model']['num_trials'])
    return study.best_params


def train_final_cnn_gru(
    train_gen: PyTorchDataGenerator, val_gen: PyTorchDataGenerator, features_nb: int,
    criterion: nn.Module, model_name: str, best_params: dict = None
) -> PyTorchComplexCNNGRU:
    """
    Function to train and predict with a CNN-GRU model.

    :param train_gen:   Training data generator.
    :param val_gen:     Validation data generator.
    :param features_nb: Number of features.
    :param criterion:   Loss function.
    :param model_name:  Name of the model.
    :param best_params: Dictionary containing the best parameters for the model.

    :return:            Trained CNN-GRU model.
    """
    # Create the model with the best parameters found by Optuna.
    final_model = PyTorchComplexCNNGRU(
        n_features=features_nb,
        output_size=config['data']['data']['horizon'],
        cnn_channels=[best_params[f"cnn_channels_{i}"] for i in range(best_params["cnn_depth"])],
        kernel_size=best_params["kernel_size"],
        gru_hidden_size=best_params["gru_hidden"],
        gru_layers=best_params["gru_layers"],
        bidirectional=best_params["bidirectional"],
        dropout=best_params["dropout"],
        pool_size=best_params["pool_size"]
    ).to(device, non_blocking=True)
    # Define the optimizer and learning rate scheduler.
    cnn_gru_optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    # Define a learning rate scheduler that reduces the learning rate when the validation loss
    # plateaus.
    cnn_gru_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        cnn_gru_optimizer, mode='min', factor=0.5,
        patience=config['model']['model']['cnn_gru']['patience'] // 2
    )
    # Train the model.
    final_model, _, _ = train_model(
        final_model, 
        train_gen, 
        val_gen, 
        criterion, 
        cnn_gru_optimizer, 
        model_name, 
        scheduler=cnn_gru_scheduler,
        device=device
    )
    return final_model


def main_cnn_gru(
    target_sites: dict, period2drop: dict, model_name: str, etype: str,
    test_implementation: bool = True
) -> None:
    """
    Function to train (and test) the CNN-GRU model.

    :param target_sites:        Dictionary of target sites.
    :param period2drop:         Dictionary of periods to drop.
    :param model_name:          Name of the model.
    :param etype:               Type of energy (consumption or production).
    :param test_implementation: Whether to test the implementation or not.
    """
    # Load the dataset.
    dataset = pd.HDFStore(
        f"{config['paths']['paths']['processed_data_dir']}/aieg.h5", mode='r'
    )
    # Create list of target sites to train on.
    target = [
        (site_id, site) if len(site) > 1 else (site_id, site[0])
        for site_id, site in target_sites.items()
    ]
    site_id = list(target_sites.keys())[0]
    # Check if we want to test the implementation. If so, we will use the test data.
    if test_implementation:
        train_sites, val_sites, test_sites = temporal_split(
            dataset, config["data"]["data"]["window_size"],
            config["data"]["data"]["horizon"], target=target, period2drop=period2drop,
        )
        test_gen = PyTorchDataGenerator(
            dataset, test_sites, config["data"]["data"]["window_size"],
            config["data"]["data"]["horizon"], period2drop=period2drop,
            is_stats_features=config['data']['features']['use_stat_features'],
            is_ts_features=config['data']['features']['use_time_features'],
            is_weather=config['data']['features']['use_weather_features'],
            previous_days=config['data']['features']['previous_days']
        )
    # Otherwise, we will use the validation data.
    else:
        train_sites, val_sites, _ = temporal_split(
            dataset, config["data"]["data"]["window_size"],
            config["data"]["data"]["horizon"], target=target, period2drop=period2drop,
            train_ratio=0.8, val_ratio=0.2, test_ratio=0.0
        )
        test_gen = None
    # Create the data generators.
    train_gen = PyTorchDataGenerator(
        dataset, train_sites, config["data"]["data"]["window_size"],
        config["data"]["data"]["horizon"], period2drop=period2drop,
        is_stats_features=config['data']['features']['use_stat_features'],
        is_ts_features=config['data']['features']['use_time_features'],
        is_weather=config['data']['features']['use_weather_features'],
        previous_days=config['data']['features']['previous_days']
    )
    val_gen = PyTorchDataGenerator(
        dataset, val_sites, config["data"]["data"]["window_size"],
        config["data"]["data"]["horizon"], period2drop=period2drop,
        is_stats_features=config['data']['features']['use_stat_features'],
        is_ts_features=config['data']['features']['use_time_features'],
        is_weather=config['data']['features']['use_weather_features'],
        previous_days=config['data']['features']['previous_days']
    )
    # Get the number of features.
    features_nb = train_gen[0][0].shape[1]
    # Define the criterion.
    criterion = nn.MSELoss()
    # Perform hyperparameter tuning with Optuna to find the best parameters for the model.
    best_params = tuning_cnn_gru(train_gen, val_gen, features_nb, criterion, model_name, device)
    # Save the best parameters found by Optuna.
    save_best_params(site_id, best_params, etype)
    # Train the model with the best parameters found by Optuna.
    best_model = train_final_cnn_gru(
        train_gen, val_gen, features_nb, criterion, model_name, best_params=best_params
    )
    evaluate_cnn_gru_model(best_model, test_gen, config["data"]["data"]["horizon"])
    os.makedirs(config['paths']['paths']['saved_models_dir'], exist_ok=True)
    model_path = os.path.join(
        config['paths']['paths']['saved_models_dir'], f'best_{model_name}_model.pt'
    )
    torch.save(best_model.state_dict(), model_path, weights_only=True)
    dataset.close()


def save_features(
    site_id: int, selected_features: list, prediction_type: str, config_dir: str = "src/configs"
) -> None:
    """
    Function to save the selected features for a given site.

    :param site_id:             ID of the site for which the features were selected.
    :param selected_features:   List of selected features for the site.
    :param prediction_type:     Type of prediction to make.
    :param config_dir:          Directory where the selected features will be saved.
    """
    logger.info("Saving selected features for site %s: %s", site_id, selected_features)
    path = Path(config_dir) / f"features/{prediction_type}/site_{site_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "site_id": site_id,
        "selected_features": selected_features,
        "n_features": len(selected_features),
    }
    with open(path, "w") as f:
        yaml.dump(data, f)




def xgb_features_importance(dtrain: xgb.DMatrix) -> list:
    """
    Function to get the feature importance from an XGBoost model trained on the training data.

    :param dtrain:  Training data in DMatrix format.

    :return:        List of the top 50 most important features.
    """
    logger.info("Training XGBoost model to get feature importance...")
    # Train a simple XGBoost model to get the feature importance.
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "seed": config['model']['model']['seed'],
    }
    model = xgb.train(params, dtrain, num_boost_round=200)
    # Get the feature importance.
    importance = model.get_score(importance_type="gain")
    importance_df = pd.DataFrame({
        "feature": list(importance.keys()),
        "importance": list(importance.values())
    }).sort_values(by="importance", ascending=False)
    # Return the top 50 most important features.
    return importance_df.head(50)["feature"].tolist()


def xgb_optimization(dtrain: xgb.DMatrix, dval: xgb.DMatrix) -> dict:
    """
    Optimize XGBoost hyperparameters using Optuna.

    :param dtrain:  Training data in DMatrix format.
    :param dval:    Validation data in DMatrix format.

    :return:        Best hyperparameters found by Optuna.
    """
    logger.info("Starting XGBoost hyperparameter optimization with Optuna...")
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna to minimize the validation MAE of the XGBoost model.

        :param trial:   Optuna trial object.

        :return:        Validation MAE for the current set of hyperparameters.
        """
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "tree_method": "hist",
        }
        model = xgb.train(
            params,
            dtrain,
            evals=[(dval, "val")],
            num_boost_round=1000,
            early_stopping_rounds=20,
            verbose_eval=False,
        )
        return model.best_score
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config['model']['model']['n_trials'])
    return study.best_params


def main_xgboost(
    target_sites: dict, period2drop: dict, model_name: str, etype: str,
    test_implementation: bool = True,
) -> None:
    """
    Function to train and test the XGBoost model.

    :param target_sites:        Dictionary of target sites.
    :param period2drop:         Dictionary of periods to drop.
    :param model_name:          Name of the model.
    :param etype:               Type of energy (consumption or production).
    :param test_implementation: Whether to test the implementation or not.
    """
    # Initialize an empty list to store dataframes.
    dataset = pd.HDFStore(
        f"{config['paths']['paths']['processed_data_dir']}/aieg.h5", mode='r'
    )
    site_name, sn, prediction_type = "", "", ""
    # Concatenate data for sites in the group if the list has more than one site.
    site_id = list(target_sites.keys())[0]
    sites = target_sites[site_id]
    if len(sites) > 1:
        group_data = build_group_data(dataset, sites)
        processed_data, features, target, _ = prepare_data_ml(
            site_name, sn, prediction_type, config["data"]["data"]["window_size"],
            config["data"]["data"]["horizon"], df=group_data,
            is_weather=config["data"]['features']['use_weather_features'],
            is_ts=config["data"]['features']['use_time_features'],
            is_stats=config["data"]['features']['use_stat_features'],
            previous_days=config['data']['features']['previous_days']
        )
    else:
        split = sites[0].split('_')
        if len(split) > 3:
            site_name = f"{split[1]}_{split[2]}"
        else:
            site_name = split[1]
        sn, prediction_type = split[-1].split('/')

        processed_data, features, target, _ = prepare_data_ml(
            site_name, sn, prediction_type, config["data"]["data"]["window_size"],
            config["data"]["data"]["horizon"],
            is_weather=config["data"]['features']['use_weather_features'],
            is_ts=config["data"]['features']['use_time_features'],
            is_stats=config["data"]['features']['use_stat_features'],
            previous_days=config['data']['features']['previous_days']
        )
    # Remove the ts index to create a multiindex with site_id.
    processed_data = processed_data.reset_index(names=['ts'])
    processed_data['site_id'] = site_id
    processed_data.set_index(['ts', 'site_id'], inplace=True)
    processed_data['site_id'] = site_id
    # Drop rows based on period2drop.
    processed_data = clean_data_for_xgb(processed_data, period2drop, site_id)
    # Append the processed group data to the datasets lists.
    train = processed_data.iloc[:int(len(processed_data) * 0.8)]
    val = processed_data.iloc[int(len(processed_data) * 0.8):int(len(processed_data) * 0.9)]
    test = processed_data.iloc[int(len(processed_data) * 0.9):] if test_implementation else None
    # Concatenate all group datasets into a single dataframe.
    dtrain = xgb.DMatrix(train[features], label=train[target])
    dval = xgb.DMatrix(val[features], label=val[target])
    # Get the feature importance from the XGBoost model trained on the training data and
    # select the top 50 most important features.
    selected_features = xgb_features_importance(dtrain)
    # Save the selected features for the site.
    save_features(site_id, selected_features, etype)
    # Train the XGBoost model with hyperparameter optimization using Optuna.
    dtrain = xgb.DMatrix(train[selected_features], label=train[target])
    dval = xgb.DMatrix(val[selected_features], label=val[target])
    best_params = xgb_optimization(dtrain, dval)
    save_best_params(site_id, best_params, etype)
    # Retrain the model on the combined training and validation data.
    logger.info(
        "Retraining XGBoost model with the best hyperparameters on the combined " \
        "training and validation data..."
    )
    df_final = pd.concat([train, val])
    dfinal = xgb.DMatrix(df_final[selected_features], label=df_final[target])
    final_model = xgb.train(
        params=best_params,
        dtrain=dfinal,
        num_boost_round=500
    )
    dtest = (
        xgb.DMatrix(test[selected_features], label=test[target]) if test_implementation else None
    )
    y_preds = final_model.predict(dtest)
    print_metrics(test[target].values, y_preds)
    # Save the model.
    os.makedirs(config["paths"]["paths"]["saved_models_dir"], exist_ok=True)
    final_model.save_model(f'{config["paths"]["paths"]["saved_models_dir"]}/{model_name}.json')
    # Close the dataset.
    dataset.close()


def run_training_pipeline() -> None:
    """
    Function to run the training pipeline for both the CNN-GRU and XGBoost models. It iterates over
    the energy types (consumption and production) and the site groups defined in the configuration,
    trains the models, and saves the best models and their parameters.
    """
    logger.info("%s Starting training pipeline %s", DASH, DASH)
    set_seed(config['model']['model']['seed'])
    # Define the sites grouped by energy type.
    energy_type = {
        'consumption': [
            config['domain']['consumption_sites_grouped'],
            config['domain']['consumption_drop_period']
        ],
        'production': [
            config['domain']['production_sites_grouped'],
            config['domain']['production_drop_period']
        ]
    }
    for etype, (sites_grouped, drop_period) in energy_type.items():
        logger.info("%s Starting training for %s sites %s", DASH, etype, DASH)
        for site_id, sites in sites_grouped.items():
            logger.info("%s Sites for site_id %s: %s %s", DASH, site_id, sites, DASH)
            if config['model']['model']['cnn_gru']['selected']:
                main_cnn_gru(
                    {site_id : sites}, {site_id: drop_period[site_id]}, f'cnn_gru_{etype}', etype,
                    True
                )
            else:
                main_xgboost(
                    {site_id : sites}, {site_id: drop_period[site_id]},
                    f'xgb_{etype}_siteid_{site_id}', etype, True
                )
    logger.info("%s Training pipeline completed %s", DASH, DASH)


if __name__ == "__main__":
    run_training_pipeline()
