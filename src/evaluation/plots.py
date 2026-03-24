"""
Module for plotting the predictions.
"""

__title__: str = "plots"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
from typing import NoReturn

# Imports third party libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

# Imports from src
from .metrics import create_indexed_dataframe

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNCTIONS ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #


def plot_predictions(y_true, y_hat, out_size: int, index=None) -> NoReturn:
    """
    Function to plot the predictions.

    :param y_true:      True values.
    :param y_hat:       Predicted values.
    :param out_size:    Size of the output.
    :param index:       Index of the data.
    """
    fig = go.Figure()
    # Case where we have multiple outputs (out_size > 1, Multi-Step-Ahead).
    if out_size > 1:
        # Normalise inputs in DataFrame if necessary.
        if not isinstance(y_true, pd.DataFrame):
            cols = [f't+{i+1}' for i in range(y_true.shape[1])]
            y_true = pd.DataFrame(y_true, columns=cols)
        # Use the index of the y_true DataFrame if it exists.
        if index is not None:
            y_true.index = index[:len(y_true)]
        # Create the DataFrame with the indexed predictions and the confidence interval.
        processed_df = create_indexed_dataframe(
            tqdm(enumerate(y_true.iterrows()), total=len(y_true), desc='Creating visualization'),
            y_hat
        )
        # Add the real values trace.
        fig.add_trace(go.Scatter(
            x=processed_df.index, 
            y=processed_df['true'], 
            mode='lines', 
            name='Valeurs réelles',
            line=dict(width=1.5)
        ))
        
        # Add the predictions trace.
        fig.add_trace(go.Scatter(
            x=processed_df.index, 
            y=processed_df['mean'], 
            mode='lines', 
            name='Prédictions',
            line=dict(width=1.5)
        ))
        
        # Add the confidence interval traces (upper and lower bounds).
        fig.add_trace(go.Scatter(
            x=processed_df.index,
            y=processed_df['upper_bound'],
            mode='lines',
            line=dict(width=0),
            name='Écart interquartile',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=processed_df.index,
            y=processed_df['lower_bound'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(width=0),
            name='Écart interquartiles'
        ))
        # Add the absolute error trace.
        error = np.abs(processed_df['true'] - processed_df['mean'])
        fig.add_trace(go.Scatter(
            x=processed_df.index, 
            y=error,
            mode='lines', 
            name='|Erreur|',
            line=dict(color='green', dash='dot'),
            visible='legendonly'
        ))
    # Case where we have only one output (out_size = 1, One-Step-Ahead).
    else:
        # Normalise inputs in DataFrame if necessary.
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.iloc[:,0]
            index_values = y_true.index
        elif isinstance(y_true, pd.Series):
            index_values = y_true.index
        else:
            y_true = y_true.flatten() if hasattr(y_true, 'flatten') else y_true
            index_values = index if index is not None else np.arange(len(y_true))
        # Convert y_true and y_hat to Series if not already.
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true, index=index_values[:len(y_true)])
        if not isinstance(y_hat, pd.Series) and not isinstance(y_hat, pd.DataFrame):
            y_hat = pd.Series(
                y_hat.flatten() if hasattr(y_hat, 'flatten') else y_hat, 
                index=index_values[:len(y_hat)]
            )
        elif isinstance(y_hat, pd.DataFrame):
            y_hat = y_hat.iloc[:, 0]
        # Add the real values trace.
        fig.add_trace(go.Scatter(
            x=y_true.index, 
            y=y_true, 
            mode='lines', 
            name='Valeurs réelles',
            line=dict(width=1.5)
        ))
        # Add the predictions trace.
        fig.add_trace(go.Scatter(
            x=y_hat.index,
            y=y_hat, 
            mode='lines',
            name='Prédictions',
            line=dict(width=1.5)
        ))
        
        # Add the absolute error trace.
        error = np.abs(y_true.values - y_hat.values)
        fig.add_trace(go.Scatter(
            x=y_true.index,
            y=error,
            mode='lines',
            name='|Erreur|',
            line=dict(color='green', dash='dot'),
            visible='legendonly'
        ))
    # Update the layout of the figure.
    fig.update_layout(
        title='Prédictions vs Valeurs Réelles',
        xaxis_title='Temps',
        yaxis_title='Valeur',
        template='plotly_white',
        hovermode="x unified",
        legend={'traceorder':'normal'},
        # legend=dict(
        #     # orientation="h",
        #     yanchor="bottom",
        #     y=1.02,
        #     xanchor="right",
        #     x=1
        # )
    )
    fig.show()
    # return fig
