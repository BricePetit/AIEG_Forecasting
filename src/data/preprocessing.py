"""
Module for data preprocessing. This module contains functions to preprocess the data before training the model.
The main function of this module is the preprocess_data function, which takes as input the raw data and returns the preprocessed data.
The preprocessing steps include:
- Adding timestamp features (hour, day of the week, month, etc.)
- Adding statistics features (rolling mean, rolling std, etc.)
- Adding weather features (temperature, humidity, etc.)
- Normalizing the data using the RobustScaler.
"""

__title__: str = "preprocessing"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
import logging
from typing import List, Tuple

# Imports third party libraries
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

# Imports from src
from .dataset_loader import load_complete_data, build_group_data
from utils.logging import setup_logger

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- Globals ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    setup_logger(log_file="preprocessing.log", level=logging.INFO)

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- FUNCTIONS ----------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

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
) -> Tuple[pd.DataFrame, List[str], List[str]]:
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
            except Exception:
                logger.warning("Invalid selected_target format: %s. Skipping.", target)
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
    site_name: str, sn: str, prediction_type: str, in_data: int, out_data: int,
    df: pd.DataFrame = None, is_weather: bool = False, is_ts: bool = False, is_stats: bool = False,
    normalize: bool = False, selected_target: List[str] = None, previous_days: int = 0
) -> Tuple[pd.DataFrame, List[str], List[str], RobustScaler]:
    """
    Function to prepare the data before using them for the training/testing. In this function,
    we can use the weather data to add them to the features. We can also create new data from the
    timestamp (e.g. holidays, season, etc.).

    :param site_name:       Name of the site.
    :param sn:              Serial Number.
    :param prediction_type: Type of prediction (consumption/production).
    :param in_data:         The number of input data.
    :param out_data:        The number of output data.
    :param df:              DataFrame containing the data.
    :param is_weather:      Use weather data (True) or not (False).
    :param is_ts:           Use timestamp features (True) or not (False).
    :param is_stats:        Use statistics features (True) or not (False).
    :param normalize:       Normalize the data (True) or not (False).
    :param selected_target: List of selected targets to use. If None, use the default targets.
    :param previous_days:   Number of previous days to consider for the shifted data.

    :return:                Processed data, list of features, list of targets,
                            and the scaler used for normalization.
    """
    # Get the data.
    data = load_complete_data(
        sn, site_name, prediction_type, in_data, is_weather, is_stats, is_ts, df
    )
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


def create_split(
    site, dataset, min_length, train, val, test, train_ratio=0.8, val_ratio=0.1,
    test_ratio=0.1, k=None, k_th=None, period2drop=None
):
    """
    Function to create the split of the dataset. It returns the train, validation and test sets.
    We can split the dataset temporally in k-folds.

    :param site:        Site to consider.
    :param dataset:     Dataset in hdf5 format.
    :param min_length:
    :param train:       Train set.
    :param val:         Validation set.
    :param test:        Test set.
    :param train_ratio: Train ratio.
    :param val_ratio:   Validation ratio.
    :param test_ratio:  Test ratio.
    :param k:           Number of folds.
    :param k_th:        Fold number.
    :param period2drop: Period to drop.

    :return:            Return the train, validation and test sets.
    """
    if isinstance(site, tuple):
        site_id, site_name = site
        if isinstance(site_name, list):
            df = build_group_data(dataset, site_name)
            site_name = '*'.join(site_name)
        else:
            df = dataset[site_name].set_index('ts')[['ap']]
            df.index = pd.to_datetime(df.index)
            df = df.asfreq('15min').fillna(0)
        df = df.reset_index(names=['ts'])
        df['site_id'] = site_id
        df.set_index(['ts', 'site_id'], inplace=True)
        if site_id in period2drop:
            for period in period2drop[site_id]:
                start, end = period
                if start == "start":
                    start = df.index.get_level_values("ts").min()
                if end == "end":
                    end = df.index.get_level_values("ts").max()
                df = df.drop(df.loc[
                    (df.index.get_level_values("ts") >= start) &
                    (df.index.get_level_values("ts") <= end)
                ].index)
    else:
        if 'FLOGVE2' in site and '205603983' in site and 'consumption' in site:
            return train, val, test
        if 'CASPOMP' in site and '202015241' in site and 'consumption' in site:
            return train, val, test
        if (
            '1904GPRS' in site and ('212648428' in site or '212303520' in site)
            and 'consumption' in site
        ):
            return train, val, test
        # Get the DataFrame.
        df = dataset[site].set_index('ts')[['ap']]
        df.index = pd.to_datetime(df.index)
        df = df.asfreq('15min')
        df = df.reset_index(names=['ts'])
        df['site_id'] = site_id
        df.set_index(['ts', 'site_id'], inplace=True)
        site_name = site
    # Get the size of the dataset.
    df_size = len(df)
    # Get the size of train, validation and test sets.
    train_size = int(df_size * train_ratio)
    val_size = int(df_size * val_ratio)
    test_size = int(df_size * test_ratio)
    # Do not consider the site if the size of the dataset is less than the minimum length.
    if df_size < min_length or train_size < min_length or val_size < min_length:
        return train, val, test
    # Split the data into train, validation and test sets.
    # if "CHAMAIEG" in site_name:
    if k:
        k_size = int((train_size + val_size) / k)
        train[site_name] = {
            'start': df[:k_size * (k_th + 1)].index[0],
            'stop': df[:k_size * (k_th + 1)].index[-1]
        }
        val[site_name] = {
            'start': df[k_size * (k_th + 1):k_size * (k_th + 2)].index[0],
            'stop': df[k_size * (k_th + 1):k_size * (k_th + 2)].index[-1]
        }
    else:
        train[site_name] = {
            'start': df.index[0],
            'stop': df[:train_size].index[-1]
        }
        val[site_name] = {
            'start': df[train_size:train_size + val_size].index[0],
            'stop': df[train_size:train_size + val_size].index[-1]
        }
    if test_size > 0:
        test[site_name] = {
            'start': df[df_size - test_size:df_size].index[0],
            'stop': df[df_size - test_size:df_size].index[-1]
        }
    else:
        test = None
    return train, val, test


def temporal_split(
        dataset: pd.HDFStore, in_length: int, out_length: int, train_ratio: float = 0.8,
        val_ratio: float = 0.1, test_ratio: float = 0.1, target: list = None,
        period2drop: dict = None
):
    """
    Function to split the dataset temporally.

    :param dataset:     Dataset in hdf5 format.
    :param in_length:   Length of the input sequence.
    :param out_length:  Length of the output sequence.
    :param train_ratio: Size of the training set.
    :param val_ratio:   Size of the validation set.
    :param test_ratio:  Size of the test set.
    :param target:      List of sites to consider.
    :param period2drop: Period to drop.

    :return:            Return the train, validation and test sets.
    """
    # Initialize the train, validation and test sets.
    train, val, test = {}, {}, {}
    # Get the minimum length of the dataset.
    min_length = in_length + out_length
    # For each site in the dataset.
    target = target if target else dataset.keys()
    for site in tqdm(target, desc='Creating temporal split'):
        train, val, test = create_split(
            site, dataset, min_length, train, val, test, train_ratio, val_ratio, test_ratio,
            period2drop=period2drop
        )
    return train, val, test


def kfold_temporal_split(
        dataset: pd.HDFStore, in_length: int, out_length: int, k: int = 10,
        train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
        target: list = None, period2drop: dict = None
):
    """
    Function to split the dataset temporally in k-folds.

    :param dataset:     Dataset in hdf5 format.
    :param in_length:   Length of the input sequence.
    :param out_length:  Length of the output sequence.
    :param k:           Number of folds.
    :param train_ratio: Size of the training set.
    :param val_ratio:   Size of the validation set.
    :param test_ratio:  Size of the test set.
    :param target:      List of sites to consider.
    :param period2drop: Period to drop.

    :return:            Return the k-folds.
    """
    # Initialize the k-folds.
    k_folds = []
    # Get the minimum length of the dataset.
    min_length = in_length + out_length
    # For each fold.
    for i in range(k):
        # Initialize the counter.
        counter = 1
        # Initialize the train, validation and test sets.
        train, val, test = {}, {}, {}
        # For each site in the dataset.
        target = target if target else dataset.keys()
        for site in tqdm(target, desc=f'Creating {i + 1}th fold'):
            site = (counter, site) if isinstance(site, str) else site
            train, val, test = create_split(
                site, dataset, min_length, train, val, test, train_ratio, val_ratio,
                test_ratio, k, i, period2drop=period2drop
            )
            counter += 1
        fold = (train, val, test)
        if any(fold):
            k_folds.append(fold)
    return k_folds
