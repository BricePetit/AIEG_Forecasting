"""
This module contains the functions to compute the metrics for the evaluation of the model.
"""

__title__: str = "metrics"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
import logging
import math

# Imports third party libraries
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

# Imports from src
from utils.logging import setup_logger


# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------- GLOBALS --------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    setup_logger(log_file="evaluation_metrics.log", level=logging.INFO)


# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNCTIONS ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #


def create_indexed_dataframe(pbar_test, predictions) -> pd.DataFrame:
    """
    Function to create a DataFrame with indexed values (true and predicted). We will compute the
    mean, the standard deviation, the margin of error, the lower bound and the upper bound.

    :param pbar_test:   Progress bar for the test set.
    :param predictions: Predictions of the model.

    :return: DataFrame with indexed values.
    """
    # Dictionary to hold the indexed values.
    index_dict_true = {}
    index_dict_hat = {}
    idx = []
    # Loop through the test set.
    for i, (_, y) in pbar_test:
        # For each value in the prediction's window and add the value in the dict.
        for j in range(len(predictions[i])):
            if i + j not in index_dict_hat.keys():
                index_dict_hat[i + j] = [[predictions[i][j]]]
            else:
                index_dict_hat[i + j][0].append(predictions[i][j])
        # Get the true value for the index.
        index_dict_true[i] = float(y.iloc[0])
        if i == 0 or y.name not in idx:
            idx.append(y.name)
    # Create the dataframes
    df_pred = pd.DataFrame.from_dict(index_dict_hat, orient='index', columns=['predicted'])
    df_true = pd.DataFrame.from_dict(index_dict_true, orient='index', columns=['true'])
    # Concatenate the two dataframes.
    df = pd.concat([df_true, df_pred], axis=1)
    # Compute the mean.
    df['mean'] = df['predicted'].apply(lambda x: np.mean(x))
    # Compute the standard deviation.
    df['std'] = df['predicted'].apply(lambda x: np.std(x))
    # Compute the number of samples.
    df['n'] = df['predicted'].apply(len)
    # Compute the interquartile range (IQR).
    df['iqr'] = df['predicted'].apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    # Compute the lower bound using the IQR.
    df['lower_bound'] = df['mean'] - 1.5 * df['iqr']
    # Compute the upper bound using the IQR.
    df['upper_bound'] = df['mean'] + 1.5 * df['iqr']
    # Drop the NaN values.
    df.dropna(inplace=True)
    # Reset the index.
    df.index = idx
    return df


def compute_metrics(y_true, y_hat) -> dict:
    """
    Function to compute the metrics.

    :param y_true:  True values.
    :param y_hat:   Predicted values

    :return:        Return the metrics.
    """
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    metrics = {
        "MAE": mean_absolute_error(y_true, y_hat),
        "NMAE": mean_absolute_error(y_true, y_hat) / y_true.mean(),
        "MAPE": mean_absolute_percentage_error(y_true, y_hat),
        "MSE": mean_squared_error(y_true, y_hat),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_hat)),
        "NRMSE": math.sqrt(mean_squared_error(y_true, y_hat)) / y_true.mean(),
        "R2": r2_score(y_true, y_hat),
    }
    return metrics


def print_metrics(y_true, y_hat) -> None:
    """
    Function that computes and logs the metrics.

    :param y_true:  True values.
    :param y_hat:   Predicted values
    """
    metrics = compute_metrics(y_true, y_hat)
    logger.info("")
    logger.info("Metrics:")
    logger.info("MAE sklearn (lower is better): %s", metrics["MAE"])
    logger.info("Normalized MAE (lower is better): %s", metrics["NMAE"])
    logger.info("MAPE sklearn (lower is better): %s %%", metrics["MAPE"])
    logger.info("MSE (lower is better): %s", metrics["MSE"])
    logger.info("RMSE (lower is better): %s", metrics["RMSE"])
    logger.info("NRMSE (lower is better): %s", metrics["NRMSE"])
    logger.info("R2 (higher is better): %s", metrics["R2"])
    logger.info("")
