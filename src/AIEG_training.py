"""
This module is used to train and test the models.
"""

__title__: str = "AIEG_training_testing"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

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
    BASE_URL_WEATHER,
    DASH_NB,
    IN_DATA,
    OUT_DATA,
    BATCH_SIZE,
    NUM_EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    NB_WORKERS,
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    IS_WEATHER_FEATURES,
    IS_TS_FEATURES,
    IS_STATS_FEATURES,
    consumption_sites_grouped,
    consumption_drop_period,
    production_sites_grouped,
    production_drop_period,
)
from data_generator import PyTorchDataGenerator
from metrics import print_metrics
from models import PyTorchCNNGRU
from AIEG_utils import (
    add_ts_features,
    add_stats_features,
    concat_production_sites,
    plot_predictions,
    temporal_split,
)
# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ GLOBAL VAR ----------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNCTIONS ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #


def download_irm_data(
    output_file: str = f'{PROCESSED_DATA_DIR}/aws_10min.csv', start_year=2010, end_year=None
):
    """
    Function to get the data from the IRM API. The data is in CSV format and is read into a pandas
    dataframe where the data are filtered to get the data for the station HUMAIN.

    :return:    A pandas dataframe with the weather data.
    """
    first = True
    # Remove the file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
    # Check if the end year is None, if so, set it to the current year.
    if end_year is None:
        end_year = datetime.now(timezone.utc).year
    # For each year, download the data from the IRM API.
    for year in range(start_year, end_year + 1):
        print(DASH_NB * '-', f"Downloading data for {year}...", DASH_NB * '-')

        start_time = f"{year}-01-01T00:00:00Z"
        end_time = f"{year}-12-31T23:59:59Z"

        params = {
            "service": "wfs",
            "version": "2.0.0",
            "request": "getFeature",
            "typenames": "aws:aws_10min",
            "outputformat": "csv",
            "CQL_FILTER": f"timestamp DURING {start_time}/{end_time}"
        }

        response = requests.get(BASE_URL_WEATHER, params=params, stream=True)

        if response.status_code == 200 and len(response.text) > 0:
            df = pd.read_csv(StringIO(response.text), sep=",")
            
            df.to_csv(output_file, mode='a', index=False, header=first)
            first = False
            print(DASH_NB * '-', f"Data for {year} downloaded with {len(df)} lines.", DASH_NB * '-')
        else:
            print(DASH_NB * '-', f'Error {response.status_code} or empty file for {year}', DASH_NB * '-')

    print(DASH_NB * '-', f"Data downloaded from IRM API in the file: {output_file}", DASH_NB * '-')


def get_weather_data():
    """
    Function to get the weather data.

    :return:    Weather data.
    """
    # Load the weather data.
    weather = pd.read_csv(f'{PROCESSED_DATA_DIR}/aws_10min.csv')
    # Load the station data for the weather.
    station = pd.read_csv(f'{PROCESSED_DATA_DIR}/aws_station.csv')
    # Get the weather data for the station.
    weather = weather[
        weather['the_geom'] == station[station['name'] ==  'HUMAIN']['the_geom'].iloc[0]
    ]
    # Set the index to the timestamp.
    weather = weather.set_index('timestamp')
    # Convert the index to datetime.
    weather.index = pd.to_datetime(weather.index)
    # Select the columns of interest.
    weather = weather[[
        'precip_quantity', 'temp_dry_shelter_avg', 'temp_grass_pt100_avg', 'temp_soil_avg',
        'temp_soil_avg_5cm', 'temp_soil_avg_10cm', 'temp_soil_avg_20cm', 'temp_soil_avg_50cm',
        'wind_speed_10m', 'wind_speed_avg_30m', 'wind_direction', 'wind_gusts_speed',
        'humidity_rel_shelter_avg', 'pressure', 'sun_duration', 'short_wave_from_sky_avg',
        'sun_int_avg'
    ]]
    # Resample the data to 15 minutes.
    weather = weather.resample('15min').first()
    return weather


def normalize_data(
    data: pd.DataFrame, features: List[str], target: List[str]
) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    Function to normalize the data using the RobustScaler.

    :param data:      DataFrame containing the data.
    :param features:  List of features to normalize.
    :param target:    List of targets to normalize.

    :return:          Normalized DataFrame and the scaler used for normalization.
    """
    # Create a copy of the data to avoid modifying the original data.
    df = data.copy()
    # Create a RobustScaler object.
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    # Fit the scaler on the features and transform them.
    df[features] = x_scaler.fit_transform(df[features])
    # Transform the target.
    df[target] = y_scaler.fit_transform(df[target])
    return df, y_scaler


def generate_shifted_data(
    data: pd.DataFrame,
    in_data: int,
    out_data: int,
    previous_days:int = 0,
    selected_target: List[str] = None
):
    """
    Function to generate the shifted data. The function will shift the data by the number of
    input data and the number of output data. The function will also shift the data by the
    number of previous days.

    :param data:            DataFrame containing the data.
    :param in_data:         The number of input data.
    :param out_data:        The number of output data.
    :param previous_days:   Number of previous days to consider for the shifted data.
    :param selected_target: List of selected targets to use. If None, use the default targets.

    :return:                Shifted DataFrame, list of features, and list of targets.
    """
    shifted = {}
    features_list = []
    target_list = []
    day_shift = previous_days * 96
    stats_features = [
        "rolling_mean", "rolling_median", "rolling_std", "rolling_min", "rolling_max",
        "diff_1", "diff_2", "ewm_mean"
    ]
    # For each column in the data, we shift the data to create the window of the sequence.
    for col in data.columns:
        # If the column is ap or a statistics feature, we shift the data by the number of input
        # data plus the day shift.
        if col.startswith('ap') or col in stats_features:
            for j in range(in_data):
                new_col_name = f'{col}-{j + day_shift}'
                shifted[new_col_name] = data[col].shift(j + day_shift)
                # Add the feature to the list of features.
                features_list.append(new_col_name)
        # Otherwise, we shift the data by the number of input data for the current day.
        else:
            for j in range(in_data):
                new_col_name = f'{col}-{j}'
                shifted[new_col_name] = data[col].shift(j)
                # Add the feature to the list of features.
                features_list.append(new_col_name)
    # If the selected target is provided, we create the target by shifting the data.
    if selected_target:
        for target in selected_target:
            try:
                shifted[target] = data['ap'].shift(-int(target.split('+')[1]))
            except Exception as e:
                print(f"Invalid selected_target format: {target}. Skipping.")
        target_list = list(selected_target)
    else:
        # If the selected target is not provided, we create the target by shifting the data.
        for k in range(1, out_data + 1):
            new_col_name = f'ap+{k}'
            shifted[new_col_name] = data['ap'].shift(-k)
            # Add the target to the list of targets.
            target_list.append(new_col_name)
    shifted = pd.DataFrame.from_dict(shifted)
    return shifted, features_list, target_list



def prepare_data_ml(
    dataset: pd.HDFStore, site_name: str, sn: str, prediction_type: str, in_data: int,
    out_data: int, df: pd.DataFrame = None, weather_data: bool = False, ts_features: bool = False,
    stats_features: bool = False, normalize: bool = False, selected_target: List[str] = None,
    previous_days: int = 0
) -> Tuple[pd.DataFrame, List[str], List[str], RobustScaler]:
    """
    Function to prepare the data before using them for the training/testing. In this function,
    we can use the weather data to add them to the features. We can also create new data from the
    timestamp (e.g. holidays, season, etc.).

    :param dataset:         HDF5 dataset.
    :param site_name:       Name of the site.
    :param sn:              Serial Number.
    :param prediction_type: Type of prediction (consumption/production).
    :param in_data:         The number of input data.
    :param out_data:        The number of output data.
    :param df:              DataFrame containing the data.
    :param weather_data:    Use weather data (True) or not (False).
    :param ts_features:     Use timestamp features (True) or not (False).
    :param stats_features:  Use statistics features (True) or not (False).
    :param normalize:       Normalize the data (True) or not (False).
    :param selected_target: List of selected targets to use. If None, use the default targets.
    :param previous_days:   Number of previous days to consider for the shifted data.

    :return:                Processed data, list of features, list of targets,
                            and the scaler used for normalization.
    """
    # Check if the data is already in the dataframe.
    if df is not None:
        data = df.copy()
    else:
        # Get the data.
        data = dataset[f'/aieg_{site_name}_{sn}/{prediction_type}'].set_index('ts')[['ap']]
    # Set the index to datetime.
    data.index = pd.to_datetime(data.index)
    data = data.asfreq('15min')
    # Check if we want to use weather data.
    if 'q1' in data.columns:
        data.drop(columns=['q1'], inplace=True)
    if 'q4' in data.columns:
        data.drop(columns=['q4'], inplace=True)
    if weather_data:
        # Get the weather data.
        weather = get_weather_data()
        # Select the weather data according to the data info ts (in index)
        weather = weather.loc[data.index]
        data = pd.concat([data, weather], axis=1)
    # Check if we want to use timestamp features.
    if ts_features:
        # Add time features in the data.
        data = add_ts_features(data)
    # Check if we want to use statistics features.
    if stats_features:
        # Add weather features in the data.
        data = add_stats_features(data, in_data)
    # Shift the data to create the window of the sequence as a feature.
    data, features, target = generate_shifted_data(
        data, in_data, out_data, previous_days, selected_target
    )
    # Drop the NaN values.
    data.dropna(inplace=True)
    data = data[features + target]
    if normalize:
        # Normalize the data.
        data, y_scaler = normalize_data(data, features, target)
    return data, features, target, y_scaler if normalize else None


def apply_xgb(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, features: List[str],
    target: List[str], model_name: str, device: str = 'cuda', n_estimators: int = 1000,
    max_depth: int = 5, early_stopping_rounds: int = 10, SEED: int = 42,
    test_implementation: bool = False
):
    """
    Function to apply XGBoost on the data.

    :param train:                   Training data.
    :param val:                     Validation data.
    :param test:                    Test data.
    :param features:                Features to use.
    :param target:                  Target to predict.
    :param model_name:              Name of the model.
    :param device:                  Device to use (cuda or cpu).
    :param n_estimators:            Number of estimators.
    :param max_depth:               Maximum depth of the trees.
    :param early_stopping_rounds:   Early stopping rounds.
    :param SEED:                    Random seed.
    :param test_implementation:     Whether to test the implementation or not.

    :return:                        Predictions.
    """
    # Define the model.
    xgb_model = xgb.XGBRegressor(
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
    # Train the model.
    xgb_model.fit(
        cudf.DataFrame.from_pandas(train[features]),
        cudf.DataFrame.from_pandas(train[target]),
        eval_set=[(
            cudf.DataFrame.from_pandas(val[features]),
            cudf.DataFrame.from_pandas(val[target])
        )],
        verbose=0,
    )
    # Save the model.
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    xgb_model.save_model(f'{SAVED_MODELS_DIR}/{model_name}.json')
    # Do predictions.
    if test_implementation:
        return xgb_model.predict(cudf.DataFrame.from_pandas(test[features]))


def train_model(
        model: PyTorchCNNGRU, train_gen: PyTorchDataGenerator, val_gen: PyTorchDataGenerator,
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
        train_gen, batch_size=BATCH_SIZE, num_workers=NB_WORKERS, pin_memory=True, shuffle=True
    )
    val_loader = DataLoader(
        val_gen, batch_size=BATCH_SIZE, num_workers=NB_WORKERS, pin_memory=True, shuffle=False
    )
    # Create save directory if it doesn't exist.
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    model_path = os.path.join(SAVED_MODELS_DIR, f'best_{model_name}_model.pt')
    # Move model to device.
    model = model.to(device)
    # Initialize variables for early stopping.
    best_val_loss = float('inf')
    early_stopping_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
    # Training loop.
    for epoch in range(NUM_EPOCHS):
        # Training phase.
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            # Move data to device.
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass.
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward pass.
            loss.backward()
            # Gradient clipping (helps stabilize training).
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
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
                inputs, targets = inputs.to(device), targets.to(device)
                # Forward pass.
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
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
        if early_stopping_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}/{NUM_EPOCHS}")
            break
        # Print progress.
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.6f}, '
                f'Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}'
            )
    # Load the best model
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Best model saved with validation loss: {best_val_loss:.6f}")
    return model, history


def predict_model(
    model: PyTorchCNNGRU, test_gen: PyTorchDataGenerator, out_size: int,
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
    
    # Create the test DataLoader.
    test_loader = DataLoader(
        test_gen, batch_size=BATCH_SIZE, shuffle=False, num_workers=NB_WORKERS, pin_memory=True
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
            plot_predictions(true_df, y_pred, out_size=y_true.shape[1], index=true_df.index)
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
                    plot_predictions(
                        house_df['true'], house_df['t+1'], out_size=1, index=house_df.index
                    )
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
    # Define model and optimizer.
    cnn_gru_model = PyTorchCNNGRU(
        n_features=features_nb, output_size=OUT_DATA, min_value=0, max_value=max_value * 2
    )
    cnn_gru_optimizer = optim.Adam(cnn_gru_model.parameters(), lr=LEARNING_RATE)
    # Set up learning rate scheduler.
    cnn_gru_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        cnn_gru_optimizer, mode='min', factor=0.5, patience=PATIENCE // 2
    )
    # Train the model.
    cnn_gru_model, cnn_gru_history = train_model(
        cnn_gru_model, 
        train_gen, 
        val_gen, 
        criterion, 
        cnn_gru_optimizer, 
        model_name, 
        scheduler=cnn_gru_scheduler,
        device=device
    )
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
            cnn_gru_model, test_gen, OUT_DATA, device=device, index=index, kfold_cv=kfold_cv
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
            is_stats_features=IS_STATS_FEATURES, is_ts_features=IS_TS_FEATURES,
            is_weather=IS_WEATHER_FEATURES
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
        is_stats_features=IS_STATS_FEATURES, is_ts_features=IS_TS_FEATURES,
        is_weather=IS_WEATHER_FEATURES
    )
    val_gen = PyTorchDataGenerator(
        dataset, val_sites, IN_DATA, OUT_DATA, period2drop=period2drop,
        is_stats_features=IS_STATS_FEATURES, is_ts_features=IS_TS_FEATURES,
        is_weather=IS_WEATHER_FEATURES
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
        test_implementation=test_implementation
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
    train_data, val_data, test_data = [], [], []
    site_name, sn, prediction_type = "", "", ""
    # Iterate over production_sites_grouped to create datasets.
    for site_id, sites in target_sites.items():
        # Concatenate data for sites in the group if the list has more than one site.
        if len(sites) > 1:
            sn_list = [site.split('_')[2].split('/')[0] for site in sites]
            group_data = concat_production_sites(dataset, sn_list)
            processed_data, features, target, _ = prepare_data_ml(
                dataset, site_name, sn, prediction_type, IN_DATA, OUT_DATA, df=group_data,
            )
        else:
            split = sites[0].split('_')
            if len(split) > 3:
                site_name = f"{split[1]}_{split[2]}"
            else:
                site_name = split[1]
            sn, prediction_type = split[-1].split('/')
            processed_data, features, target, _ = prepare_data_ml(
                dataset, site_name, sn, prediction_type, IN_DATA, OUT_DATA,
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
        train_data.append(processed_data.iloc[:int(len(processed_data) * 0.8)])
        if test_implementation:
            val_data.append(processed_data.iloc[int(len(processed_data) * 0.8):int(len(processed_data) * 0.9)])
            test_data.append(processed_data.iloc[int(len(processed_data) * 0.9):])
        else:
            # Split the data into training, validation, and test sets.
            val_data.append(processed_data.iloc[int(len(processed_data) * 0.8):])
    # Concatenate all group datasets into a single dataframe.
    final_train_data = pd.concat(train_data)
    final_val_data = pd.concat(val_data)
    final_test_data = pd.concat(test_data) if test_implementation else None
    # Save or process the final dataset as needed
    y_preds = apply_xgb(
        final_train_data, final_val_data, final_test_data, features, target, model_name,
        device='cuda', n_estimators=1500, max_depth=6, early_stopping_rounds=10,
        test_implementation=test_implementation
        
    )
    # Check if we want to test the implementation.
    if test_implementation:
        # If so, we will use the test data.
        final_test_data["y_preds"] = y_preds
        # For each site_id, we will show the metrics and plot the predictions.
        for site_id in final_test_data['site_id'].unique():
            site_data = final_test_data[final_test_data['site_id'] == site_id]
            for_metrics = site_data[site_data[target].values > 0]

            print(f"Site ID: {site_id}")
            print("Average:", site_data[target].mean())
            print_metrics(for_metrics[target].values, for_metrics["y_preds"].values)
            plot_predictions(
                site_data[target].values, site_data["y_preds"].values, out_size=OUT_DATA,
                index=site_data.index.get_level_values("ts")
            )
        # Print the metrics for all sites.
        print_metrics(final_test_data[target].values, final_test_data["y_preds"].values)
    # Close the dataset.
    dataset.close()


if __name__ == "__main__":
    main_xgboost(production_sites_grouped, production_drop_period, 'xgb_production', True)
    # main_xgboost(consumption_sites_grouped, consumption_drop_period, 'xgb_consumption', True)
    # main_cnn_gru(production_sites_grouped, production_drop_period, 'cnn_gru_production', True)
    # main_cnn_gru(consumption_sites_grouped, consumption_drop_period, 'cnn_gru_consumption', True)
