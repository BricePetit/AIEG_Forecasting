"""
In this module, we define the metrics used to evaluate the performance of the model.
"""
__title__: str = "metrics"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# Imports standard libraries.
import math
from typing import NoReturn

# Imports third party libraries.
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

# Imports from src.


def compute_mae(y, y_pred):
    """
    Function to compute the Mean Absolute Error (MAE).
    It is the average of the absolute difference between the predicted values and the actual values.

    :param y:           True Y values.
    :param y_pred:      Predicted Y values.

    :return:            Return the MAE.
    """
    # Ensure that the input is a numpy array.
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y - y_pred))


def compute_mape(y, y_pred, epsilon=1e-10):
    """
    Function to compute the Mean Absolute Percentage Error (MAPE).
    It is the average of the absolute percentage difference between the predicted values and the actual values.

    :param y:           True Y values.
    :param y_pred:      Predicted Y values.
    :param epsilon:     Small value to avoid division by zero.

    :return:            Return the MAPE.
    """
    # Ensure that the input is a numpy array.
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y - y_pred) / (np.abs(y) + epsilon))) * 100


def compute_smape(y, y_pred, epsilon=1e-10):
    """
    Function to compute the Symmetric Mean Absolute Percentage Error (SMAPE).
    It is the average of the absolute percentage difference between the predicted values and the actual values.

    :param y:           True Y values.
    :param y_pred:      Predicted Y values.
    :param epsilon:     Small value to avoid division by zero.

    :return:            Return the SMAPE.
    """
    # Ensure that the input is a numpy array.
    y = np.array(y)
    y_pred = np.array(y_pred)
    # return np.sum(np.abs(y_pred - y)) / np.sum(y + y_pred + epsilon) * 100
    # return np.mean(2.0 * np.abs(y - y_pred) / (np.abs(y) + np.abs(y_pred) + epsilon)) * 100
    # return np.mean(np.abs(y - y_pred) / ((np.abs(y) + np.abs(y_pred)) / 2)) * 100
    return np.mean(2 * np.abs(y - y_pred) / (np.abs(y) + np.abs(y_pred) + epsilon)) * 100


def modified_smape(y_true, y_pred, epsilon=1.0):
    """
    Modified Symmetric Mean Absolute Percentage Error avec epsilon
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true) + epsilon
    
    return np.mean(numerator / denominator) * 100


def compute_wmape(y, y_pred, epsilon=1e-10):
    """
    Function to compute the Weighted Mean Absolute Percentage Error (WMAPE).
    It is the sum of the absolute percentage difference between the predicted values and the
    actual values.

    :param y:           True Y values.
    :param y_pred:      Predicted Y values.
    :param epsilon:     Small value to avoid division by zero.
    
    :return:            Return the WMAPE.
    """
    # Ensure that the input is a numpy array.
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.sum(np.abs(y - y_pred)) / np.sum(np.abs(y) + epsilon) * 100


def compute_mse(y, y_pred, squared=True):
    """
    Function to compute the Mean Squared Error (MSE) or the Root Mean Squared Error (RMSE).
    It is the average of the squared difference between the predicted values and the actual values. 

    :param y:           True Y values.
    :param y_pred:      Predicted Y values.
    :param squared:     If True, return the squared error.

    :return:            Return the MSE or the RMSE.
    """
    # Ensure that the input is a numpy array.
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.sqrt(np.sum((y - y_pred) ** 2) / len(y)) if squared else np.sum((y - y_pred) ** 2) / len(y)


def compute_r2(y, y_pred):
    """
    Function to compute the R2 score.
    It is the proportion of the variance in the dependent variable that is predictable from the independent variable.

    :param y:           True Y values.
    :param y_pred:      Predicted Y values.

    :return:            Return the R2 score.
    """
    # Ensure that the input is a numpy array.
    y = np.array(y)
    y_pred = np.array(y_pred)
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


def index_of_agreement(y_true, y_pred):
    """
    Index of Agreement en pourcentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mean_y_true = np.mean(y_true)
    numerator = np.sum(np.square(y_pred - y_true))
    denominator = np.sum(np.square(
        np.abs(y_pred - mean_y_true) + np.abs(y_true - mean_y_true)
    ))
    
    # Éviter division par zéro
    if denominator == 0:
        return 100.0 if numerator == 0 else 0.0
        
    d = 1 - (numerator / denominator)
    return d * 100  # en pourcentage


def percent_error_non_zero(y_true, y_pred, threshold=0.1):
    """
    Pourcentage d'erreur sur les périodes non nulles (ou > threshold)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Créer un masque pour les valeurs au-dessus du seuil
    mask = y_true > threshold
        
    # Si aucune valeur ne dépasse le seuil
    if not np.any(mask):
        return np.nan

    # Convertir les données en arrays 1D si nécessaires
    if hasattr(y_true, 'values'):
        y_true_values = y_true.values.flatten()
        y_pred_values = y_pred.values.flatten()
    else:
        y_true_values = np.array(y_true).flatten()
        y_pred_values = np.array(y_pred).flatten()

    # Appliquer le masque sur les valeurs 1D
    mask_values = mask.values.flatten() if hasattr(mask, 'values') else mask.flatten()
        
    total_true = np.sum(y_true_values[mask_values])
        
    # Éviter division par zéro
    if total_true == 0:
        return np.inf
            
    error = np.sum(np.abs(y_pred_values[mask_values] - y_true_values[mask_values])) / total_true * 100
        
    return error


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
    }
    return metrics


def print_metrics(y_true, y_hat) -> NoReturn:
    """
    Function that computes and prints the metrics.

    :param y_true:  True values.
    :param y_hat:   Predicted values
    """
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    print()
    print("Metrics:")
    print("MAE sklearn (lower is better): ", mean_absolute_error(y_true, y_hat))
    print("Normalized MAE (lower is better): ", mean_absolute_error(y_true, y_hat) / y_true.mean())
    print("MAPE sklearn (lower is better): ", mean_absolute_percentage_error(y_true, y_hat), "%")
    print("MSE (lower is better): ", mean_squared_error(y_true, y_hat))
    print("RMSE (lower is better): ", math.sqrt(mean_squared_error(y_true, y_hat)))
    print("NRMSE (lower is better): ", math.sqrt(mean_squared_error(y_true, y_hat)) / y_true.mean())
    print("R2 (higher is better): ", r2_score(y_true, y_hat))
    # print("Modified SMAPE (lower is better): ", modified_smape(y_true, y_hat))
    print()
