"""
This module contains functions to add statistics features to the data. These features are based on
the rolling window of the data and the differences of the data. The features added are:
- rolling_mean:     Rolling mean of the data.
- rolling_median:   Rolling median of the data.
- rolling_std:      Rolling standard deviation of the data.
- rolling_min:      Rolling minimum of the data.
- rolling_max:      Rolling maximum of the data.
- diff_1:           First difference of the data.
- diff_2:           Second difference of the data.
- ewm_mean:         Exponential weighted mean of the data.
"""

__title__: str = "stats_features"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries


# Imports third party libraries
import pandas as pd

# Imports from src


# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- FUNCTIONS ----------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


def add_stats_features(data: pd.DataFrame, in_data) -> pd.DataFrame:
    """
    Function to add statistics features to the data. The function will add the following features:
    - rolling_mean:     Rolling mean of the data.
    - rolling_median:   Rolling median of the data.
    - rolling_std:      Rolling standard deviation of the data.
    - rolling_min:      Rolling minimum of the data.
    - rolling_max:      Rolling maximum of the data.
    - diff_1:           First difference of the data.
    - diff_2:           Second difference of the data.
    - ewm_mean:         Exponential weighted mean of the data.

    :param data:    DataFrame containing the data.
    :param in_data: Number of input data.

    :return:        DataFrame with added features.
    """
    # Copy the data to avoid modifying the original data.
    df = data.copy()
    # Add the statistics features to the data for the rolling window.
    df['rolling_mean'] = df['ap'].rolling(window=in_data).mean()
    df['rolling_median'] = df['ap'].rolling(window=in_data).median()
    df['rolling_std'] = df['ap'].rolling(window=in_data).std()
    # Min and max features for the rolling window.
    df['rolling_min'] = df['ap'].rolling(window=in_data).min()
    df['rolling_max'] = df['ap'].rolling(window=in_data).max()
    # Differential features.
    df['diff_1'] = df['ap'].diff(1)
    df['diff_2'] = df['ap'].diff(2)
    # Exponential weighted mean.
    df['ewm_mean'] = df['ap'].ewm(span=in_data).mean()
    return df
