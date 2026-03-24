"""
This module is used to train and test the models.
"""

__title__: str = "AIEG_training"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

import sys, os
sys.path.append(os.path.abspath("src"))
from data.preprocessing import prepare_data_ml, temporal_split
from data.dataset_loader import concat_production_sites
from evaluation.metrics import print_metrics
from evaluation.plots import plot_predictions
from data_generator.pytorch_generators import PyTorchDataGenerator
from models.pytorch_models import PyTorchComplexCNNGRU
import optuna

# Imports standard libraries
from io import StringIO
import os
import requests
from typing import List, NoReturn, Tuple

# Imports third party libraries
import cudf
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import xgboost as xgb

# Imports from src
from config import (
    DASH_NB,
    IN_DATA,
    OUT_DATA,
    # BATCH_SIZE,
    # NUM_EPOCHS,
    # PATIENCE,
    # LEARNING_RATE,
    # NB_WORKERS,
    PROCESSED_DATA_DIR,
    # SAVED_MODELS_DIR,
    # IS_WEATHER_FEATURES,
    # IS_TS_FEATURES,
    # IS_STATS_FEATURES,
    consumption_sites_grouped,
    consumption_drop_period,
    production_sites_grouped,
    production_drop_period,
)

# from metrics import print_metrics
# from models import PyTorchCNNGRU
# from AIEG_utils import (
#     add_ts_features,
#     add_stats_features,
#     concat_production_sites,
#     plot_predictions,
#     temporal_split,
# )
# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ GLOBAL VAR ----------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNCTIONS ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #
BASE_DIR: str = "/home/iridia-tower/Bureau/bripetit_phd/aieg"
SAVED_MODELS_DIR: str = f"{BASE_DIR}/saved_models"


def train_model(
        model: PyTorchComplexCNNGRU, train_gen: PyTorchDataGenerator, val_gen: PyTorchDataGenerator,
        criterion: nn.MSELoss, optimizer: optim.Adam, model_name: str,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None, device: str = 'cpu'
    ):
    """
    Function to train the model.

    :param model:           Model to train.
    :param train_gen:       Training data generator.
    :param val_gen:         Validation data generator.
    :param criterion:       Loss function.
    :param optimizer:       Optimizer.
    :param model_name:      Name of the model.

    :param scheduler:       Learning rate scheduler.
    :param device:          Device to use (CPU or GPU).

    :return:                Trained model and training history.
    """
    # Create DataLoaders.
    train_loader = DataLoader(
        train_gen, batch_size=64, num_workers=16, pin_memory=True, shuffle=True, persistent_workers=True,
        prefetch_factor=4, drop_last=True
    )
    val_loader = DataLoader(
        val_gen, batch_size=64, num_workers=16, pin_memory=True, shuffle=False, persistent_workers=True,
        prefetch_factor=4, drop_last=False
    )
    # Create save directory if it doesn't exist.
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    model_path = os.path.join(SAVED_MODELS_DIR, f'best_{model_name}_model.pt')
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
    for epoch in range(100):
        # Training phase.
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            # Move data to device.
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
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
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
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
        if early_stopping_counter >= 10:
            print(f"Early stopping triggered at epoch {epoch+1}/{100}")
            break
        # Print progress.
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f'Epoch {epoch+1}/{100}, Train Loss: {train_loss:.6f}, '
                f'Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}'
            )
    # Load the best model
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Best model saved with validation loss: {best_val_loss:.6f}")
    return model, history, best_val_loss


def predict_model(
    model: PyTorchComplexCNNGRU, test_gen: PyTorchDataGenerator, out_size: int,
    device: str = 'cpu', index=None, kfold_cv: bool = False
):
    """
    Function to make predictions using the model on a given dataset.

    :param model:       Trained model.
    :param test_gen:    Test data generator.
    :param out_size:    Output size.
    :param device:      Device to use for computation (CPU or GPU).
    :param index:       Optional index for the predictions.
    :param kfold_cv:    Whether to use k-fold cross-validation or not.

    :return:            Predictions.
    """
    # Set the model to evaluation mode.
    model.eval()
    # AMP settings for inference.
    amp_enabled = (torch.cuda.is_available() and str(device).startswith("cuda"))
    amp_dtype = torch.bfloat16
    
    # Create the test DataLoader.
    test_loader = DataLoader(
        test_gen, batch_size=64, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True,
        prefetch_factor=4, drop_last=False
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
        if out_size > 1:
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
                print(f"\n--- Metrics for {col} ---")
                print_metrics(true_df[col], y_pred[:, i])
            # Show the metrics for all steps.
            print("\n--- Metrics for all steps ---")
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
                print("\n--- Metrics for all houses ---")
                print_metrics(results['true'], results['t+1'])
                for site_id in results.index.get_level_values("site_id").unique():
                    house_df = results.xs(site_id, level="site_id")
                    print(f"\n--- Metrics for house id {site_id} ---")
                    # Show the metrics.
                    print_metrics(house_df['true'], house_df['t+1'])
                    # Plot the predictions.
                    # plot_predictions(
                    #     house_df['true'], house_df['t+1'], out_size=1, index=house_df.index
                    # )
    return y_pred


def train_predict_cnn_gru(
    train_gen: PyTorchDataGenerator, val_gen: PyTorchDataGenerator, test_gen: PyTorchDataGenerator,
    features_nb: int, criterion: nn.Module, device: str, model_name: str, max_value: float,
    lag: int = 0, kfold_cv: bool = False, test_implementation: bool = False
):
    """
    Function to train and predict with a CNN-GRU model.

    :param train_gen:           Training data generator.
    :param val_gen:             Validation data generator.
    :param test_gen:            Test data generator.
    :param features_nb:         Number of features.
    :param criterion:           Loss function.
    :param device:              Device to use (CPU or GPU).
    :param model_name:          Name of the model.
    :param max_value:           Maximum value for the output.
    :param lag:                 Lag for the model.
    :param kfold_cv:            Whether to use k-fold cross-validation or not.
    :param test_implementation: Whether to test the implementation or not.

    :return:                    Predictions.
    """
    def objective(trial):

        # =========================
        # HYPERPARAMETERS
        # =========================

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
        model = PyTorchComplexCNNGRU(
            n_features=features_nb,
            output_size=OUT_DATA,
            cnn_channels=cnn_channels,
            kernel_size=params["kernel_size"],
            gru_hidden_size=params["gru_hidden_size"],
            gru_layers=params["gru_layers"],
            bidirectional=params["bidirectional"],
            dropout=params["dropout"],
            pool_size=params["pool_size"]
        ).to(device, non_blocking=True)

        cnn_gru_optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        cnn_gru_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            cnn_gru_optimizer, mode='min', factor=0.5, patience=10 // 2
        )

        model, cnn_gru_history, val_loss = train_model(
            model, 
            train_gen, 
            val_gen, 
            criterion, 
            cnn_gru_optimizer, 
            model_name, 
            scheduler=cnn_gru_scheduler,
            device=device
        )

        # val_loss = train_model(model, train_gen, val_gen, params['lr'], device=device)

        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params

    final_model = PyTorchComplexCNNGRU(
        n_features=features_nb,
        output_size=OUT_DATA,
        cnn_channels=[best_params[f"cnn_channels_{i}"] for i in range(best_params["cnn_depth"])],
        kernel_size=best_params["kernel_size"],
        gru_hidden_size=best_params["gru_hidden"],
        gru_layers=best_params["gru_layers"],
        bidirectional=best_params["bidirectional"],
        dropout=best_params["dropout"],
        pool_size=best_params["pool_size"]
    ).to(device, non_blocking=True)

    cnn_gru_optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    cnn_gru_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        cnn_gru_optimizer, mode='min', factor=0.5, patience=10 // 2
    )

    final_model, cnn_gru_history, val_loss = train_model(
        final_model, 
        train_gen, 
        val_gen, 
        criterion, 
        cnn_gru_optimizer, 
        model_name, 
        scheduler=cnn_gru_scheduler,
        device=device
    )
    # Define model and optimizer.
    # cnn_gru_model = PyTorchComplexCNNGRU(
    #     features_nb,
    # )
    # cnn_gru_optimizer = optim.Adam(cnn_gru_model.parameters(), lr=0.001)
    # Set up learning rate scheduler.
    # cnn_gru_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     cnn_gru_optimizer, mode='min', factor=0.5, patience=10 // 2
    # )
    # Train the model.
    # cnn_gru_model, cnn_gru_history = train_model(
    #     cnn_gru_model, 
    #     train_gen, 
    #     val_gen, 
    #     criterion, 
    #     cnn_gru_optimizer, 
    #     model_name, 
    #     scheduler=cnn_gru_scheduler,
    #     device=device
    # )
    # Check if we want to test the implementation and if we have a test generator to do
    # predictions.
    if test_implementation:
        if OUT_DATA == 1:
            index = test_gen.data[IN_DATA+lag:].index
        else:
            index = test_gen.data[IN_DATA+lag:-OUT_DATA+1].index
        index = test_gen.data.iloc[test_gen.indices].index
        # Make predictions.
        y_preds = predict_model(
            final_model, test_gen, OUT_DATA, device=device, index=index, kfold_cv=kfold_cv
        )
        return y_preds


def main_cnn_gru(
    target_sites: dict, period2drop: dict, model_name: str, test_implementation: bool = True,
) -> NoReturn:
    """
    Function to train (and test) the CNN-GRU model.

    :param target_sites:        Dictionary of target sites.
    :param period2drop:         Dictionary of periods to drop.
    :param model_name:          Name of the model.
    :param test_implementation: Whether to test the implementation or not.
    :param weather_features:    Whether to use weather features or not.
    :param ts_features:         Whether to use time series features or not.
    :param stats_features:      Whether to use statistics features or not.
    """
    # Load the dataset.
    dataset = pd.HDFStore(f'{PROCESSED_DATA_DIR}/aieg.h5', mode='r')
    # Create list of target sites to train on.
    target = [
        (site_id, site) if len(site) > 1 else (site_id, site[0])
        for site_id, site in target_sites.items()
    ]

    # Check if we want to test the implementation. If so, we will use the test data.
    if test_implementation:
        train_sites, val_sites, test_sites = temporal_split(
            dataset, IN_DATA, OUT_DATA, target=target, period2drop=period2drop,
        )
        test_gen = PyTorchDataGenerator(
            dataset, test_sites, IN_DATA, OUT_DATA, period2drop=period2drop,
            is_stats_features=True, is_ts_features=True,
            is_weather=True, previous_days=0
        )
    # Otherwise, we will use the validation data.
    else:
        train_sites, val_sites, _ = temporal_split(
            dataset, IN_DATA, OUT_DATA, target=target, period2drop=period2drop,
            train_ratio=0.8, val_ratio=0.2, test_ratio=0.0
        )
        test_gen = None
    # Create the data generators.
    train_gen = PyTorchDataGenerator(
        dataset, train_sites, IN_DATA, OUT_DATA, period2drop=period2drop,
        is_stats_features=True, is_ts_features=True,
        is_weather=True, previous_days=0
    )
    val_gen = PyTorchDataGenerator(
        dataset, val_sites, IN_DATA, OUT_DATA, period2drop=period2drop,
        is_stats_features=True, is_ts_features=True,
        is_weather=True, previous_days=0
    )
    # Find the maximum value of the target variable in the training, validation and test sets.
    max_values = [
        train_gen.data['ap'].max(),
        val_gen.data['ap'].max(),
        test_gen.data['ap'].max() if test_gen else 0
    ]
    # Get the number of features.
    features_nb = train_gen[0][0].shape[1]
    # Define the criterion.
    criterion = nn.MSELoss()

    # Train the model.
    train_predict_cnn_gru(
        train_gen, val_gen, test_gen, features_nb, criterion, device, model_name, max(max_values),
        test_implementation=test_implementation, lag=0
    )
    dataset.close()


def main_xgboost(
    target_sites, period2drop, model_name, test_implementation: bool = True,
) -> NoReturn:
    """
    Function to train and test the XGBoost model.

    :param target_sites:        Dictionary of target sites.
    :param period2drop:         Dictionary of periods to drop.
    :param model_name:          Name of the model.
    :param test_implementation: Whether to test the implementation or not.
    :param weather_features:    Whether to use weather features or not.
    :param ts_features:         Whether to use time series features or not.
    :param stats_features:      Whether to use statistics features or not.
    """
    # Initialize an empty list to store dataframes.
    dataset = pd.HDFStore(f'{PROCESSED_DATA_DIR}/aieg.h5', mode='r')
    site_name, sn, prediction_type = "", "", ""
    # Concatenate data for sites in the group if the list has more than one site.
    sites = list(target_sites.values())
    site_id = list(target_sites.keys())[0]
    if len(sites) > 1:
        sn_list = [site.split('_')[2].split('/')[0] for site in sites]
        group_data = concat_production_sites(dataset, sn_list)
        processed_data, features, target, _ = prepare_data_ml(
            site_name, sn, prediction_type, IN_DATA, 8, df=group_data,
            is_weather=True, is_ts=True, is_stats=True, previous_days=0
        )
    else:
        split = sites[0][0].split('_')
        if len(split) > 3:
            site_name = f"{split[1]}_{split[2]}"
        else:
            site_name = split[1]
        sn, prediction_type = split[-1].split('/')

        processed_data, features, target, _ = prepare_data_ml(
            site_name, sn, prediction_type, IN_DATA, 8,
            is_weather=True, is_ts=True, is_stats=True, previous_days=0
        )
    # Remove the ts index to create a multiindex with site_id.
    processed_data = processed_data.reset_index(names=['ts'])
    processed_data['site_id'] = site_id
    processed_data.set_index(['ts', 'site_id'], inplace=True)
    processed_data['site_id'] = site_id
    # Drop rows based on period2drop.
    if site_id in period2drop:
        for period in period2drop[site_id]:
            start, end = period
            if start == "start":
                start = processed_data.index.get_level_values("ts").min()
            if end == "end":
                end = processed_data.index.get_level_values("ts").max()
            processed_data = processed_data.drop(
                processed_data.loc[
                    (processed_data.index.get_level_values("ts") >= start) &
                    (processed_data.index.get_level_values("ts") <= end)
                ].index
            )
        # Append the processed group data to the datasets lists.
        train = processed_data.iloc[:int(len(processed_data) * 0.8)]
        val = processed_data.iloc[int(len(processed_data) * 0.8):int(len(processed_data) * 0.9)]
        test = processed_data.iloc[int(len(processed_data) * 0.9):] if test_implementation else None
    # Concatenate all group datasets into a single dataframe.
    dtrain = xgb.DMatrix(train[features], label=train[target])
    dval = xgb.DMatrix(val[features], label=val[target])
    dtest = xgb.DMatrix(test[features], label=test[target]) if test_implementation else None
    # Save or process the final dataset as needed

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "seed": 42,
    }

    model = xgb.train(params, dtrain, num_boost_round=200)

    importance = model.get_score(importance_type="gain")

    importance_df = pd.DataFrame({
        "feature": list(importance.keys()),
        "importance": list(importance.values())
    }).sort_values(by="importance", ascending=False)

    selected_features = importance_df.head(50)["feature"].tolist()

    print(selected_features)

    dtrain = xgb.DMatrix(train[selected_features], label=train[target])
    dval = xgb.DMatrix(val[selected_features], label=val[target])

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
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
            verbose_eval=False
        )

        return model.best_score
    # dtest = xgb.DMatrix(test[selected_features], label=test[target]) if test_implementation else None

    # # Train the model.
    # xgb_model = xgb.train(
    #     params=params, dtrain=dtrain, evals=[(dval, "val")], num_boost_round=2000,
    #     verbose_eval=50, early_stopping_rounds=10
    # )
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    print(study.best_params)
    # # Check if we want to test the implementation.
    # if test_implementation:
    #     # If so, we will use the test data.
    #     y_preds = xgb_model.predict(dtest)

    #     print(f"Site ID: {site_id}")
    #     print("Average:", test[target].mean())
    #     print_metrics(test[target].values, y_preds)
    #     # plot_predictions(
    #     #     test[target].values, y_preds, out_size=OUT_DATA,
    #     #     index=test.index.get_level_values("ts")
    #     # )
    # Retrain the model on the combined training and validation data.
    df_final = pd.concat([train, val])
    dfinal = xgb.DMatrix(df_final[features], label=df_final[target])
    final_model = xgb.train(
        params=study.best_params,
        dtrain=dfinal,
        num_boost_round=500
    )
    y_preds = final_model.predict(dtest)
    print_metrics(test[target].values, y_preds)
    # Save the model.
    # os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    # final_model.save_model(f'{SAVED_MODELS_DIR}/{model_name}.json')
    # Close the dataset.
    dataset.close()


if __name__ == "__main__":
    # Define the sites grouped by energy type.
    energy_type = {
        # 'consumption': [consumption_sites_grouped, consumption_drop_period],
        'production': [production_sites_grouped, production_drop_period]
    }
    for etype, (sites_grouped, drop_period) in energy_type.items():
        print(DASH_NB * '=', f" Starting training for {etype} sites ", DASH_NB * '=')
        for site_id, sites in sites_grouped.items():
            print(DASH_NB * '-', f" Sites for site_id {site_id}: {sites} ", DASH_NB * '-')
            # main_xgboost({site_id : sites}, {site_id: drop_period[site_id]}, f'xgb_{etype}', True)
            main_cnn_gru({site_id : sites}, {site_id: drop_period[site_id]}, f'cnn_gru_{etype}', True)
