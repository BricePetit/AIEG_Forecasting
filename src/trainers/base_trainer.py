"""
In this module, we defined the base class for our trainers.
"""
__title__: str = "BaseTrainer"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
from abc import ABC, abstractmethod
import os
from tqdm import tqdm
from typing import (
    NoReturn,
)

# Imports third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import seaborn as sns

# Imports from src
from config import (
    DASH_NB,
    PLOTS_DIR,
    PREDICTIONS_DIR
)

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- CLASSES ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


class BaseTrainer(ABC):
    """
    Base class for a trainer. The idea is to train/val/test any model in TensorFlow and
    PyTorch.
    """
    def __init__(
        self, model_name: str, model_type: str, seq_length: int, out_length: int,
        learning_rate: float, epochs: int, batch_size: int, patience: int, api_name: str
    ):
        """
        Constructor of the BaseTrainer class.

        :param model_name:      Name of the model.
        :param model_type:      Type of the model (Seq2Point, Seq2Seq, etc.).
        :param seq_length:      Length of the sequence.
        :param out_length:      Length of the output sequence.
        :param learning_rate:   Learning rate.
        :param epochs:          Number of epochs.
        :param batch_size:      Size of the batch.
        :param patience:        Patience for the early stopping.
        :param api_name:        Name of the API (TensorFlow or PyTorch).
        """
        self.model_name = model_name
        self.model_type = model_type
        self.seq_length = seq_length
        self.out_length = out_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.api_name = api_name

    @abstractmethod
    def train(self, train, val = None) -> NoReturn:
        """
        Function to train the model.

        :param train:   Training data.
        :param val:     Validation data.
        """

    @abstractmethod
    def evaluate(self, test):
        """
        Function to evaluate the model.

        :param test:    Test data.

        :return:        Return the evaluation result.
        """

    @abstractmethod
    def predict(self, test):
        """
        Function to predict with the model.

        :param test:    Test data.

        :return:        Return the prediction result.
        """

    @abstractmethod
    def save_model(self) -> NoReturn:
        """
        Function to save a model.
        """

    @abstractmethod
    def load_model(self) -> NoReturn:
        """
        Function to load a model.
        """

    def plot_history(self, history) -> NoReturn:
        """
        Function to plot the history of the model.

        :param history: History of the model.
        """
        pred_fig = go.Figure()
        pred_fig.add_trace(
            go.Scatter(
                y=history['loss'],
                name='MSE (training)'
            )
        )
        pred_fig.add_trace(
            go.Scatter(
                y=history['val_loss'],
                name='MSE (validation)'
            )
        )
        pred_fig.update_layout(
            title='Training loss',
            xaxis_title='Epoch', yaxis_title='Loss'
        )
        path = f"{PLOTS_DIR}/history/{self.model_type}"
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the graph in html format
        pred_fig.write_html(
            f"{path}/{self.model_name}_{self.model_type}_{self.api_name}"
            f"_train_history.html"
        )


    def plot_results(self, predictions, test, fig_format='png') -> NoReturn:
        """
        Function to plot the results of the model.

        :param predictions: Predictions of the model.
        :param test:        Test data generator.
        :param fig_format:  Extension of the figure.
        """
        

        # 75776 represents more or less a week of data with a sampling rate of 8 seconds with a
        # batch of 512 samples.
        # offset = 672
        offset = 96
        if isinstance(test, pd.DataFrame):
            scaler = None
            if self.out_length > 1:
                pbar_test = tqdm(enumerate(test.iterrows()), total=len(test), desc='Create dataframe')
                df = create_indexed_dataframe(pbar_test, predictions, scaler)
            else:
                df = pd.DataFrame(
                    {
                        'true': test['t-1'],
                        'predicted': predictions
                    }
                )
        else:
            scaler = test.get_scaler() if test.is_standardized() else None
            pbar_test = tqdm(enumerate(test), total=len(test), desc='Create dataframe')
            # Create a dataframe with indexed values.
            if self.out_length > 1:
                df = create_indexed_dataframe(pbar_test, predictions, scaler)
            else:
                df = pd.DataFrame(
                    {
                        'true': np.concatenate([y for _, (_, y) in pbar_test]),
                        'predicted': predictions
                    },
                    index=test.data[self.seq_length + 1:].index
                )
                # Inverse the scaling.
                if scaler:
                    df['true'] = scaler.inverse_transform(df[['true']])
        print(
            '-' * DASH_NB, 'Getting the main data and the ground truth and plotting it...',
            '-' * DASH_NB
        )
        path = f"{PLOTS_DIR}/predictions/{self.model_type}/in{self.seq_length}_out{self.out_length}"
        if not os.path.exists(path):
            os.makedirs(path)
        # Create a graph for each week
        pbar_plot = tqdm(range(len(df) // offset), total=len(df) // offset, desc='Plotting results')
        for i in pbar_plot:
            # Get the current data.
            current_data = df.iloc[i * offset:(i + 1) * offset]
            if fig_format == 'html':
                # Create a new figure.
                pred_fig = go.Figure()
                # Plot the ground truth.
                pred_fig.add_trace(go.Scatter(
                    x=current_data.index,
                    y=current_data['true'],
                    mode='lines',
                    name='Ground Truth',
                    line=dict(color='green')
                ))
                if self.out_length > 1:
                    # Fill the standard deviation.
                    pred_fig.add_trace(go.Scatter(
                        x=current_data.index + current_data.index[::-1],
                        y=list(current_data['mean'] + current_data['std']) + list(
                            (current_data['mean'] - current_data['std']))[::-1],
                        fill='toself',
                        hoverinfo="skip",
                        fillcolor='rgba(0,100,250,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        name='Standard Deviation'
                    ))
                    # Plot the prediction.
                    pred_fig.add_trace(go.Scatter(
                        x=current_data.index,
                        y=current_data['mean'],
                        mode='lines',
                        name='Prediction',
                        line=dict(color='blue')
                    ))
                else:
                    # Plot the prediction.
                    pred_fig.add_trace(go.Scatter(
                        x=current_data.index,
                        y=current_data['predicted'],
                        mode='lines',
                        name='Prediction',
                        line=dict(color='blue')
                    ))
                # Update the layout.
                pred_fig.update_layout(
                    title=f'Predictions {i + 1} of the dataset',
                    xaxis_title='Time',
                    yaxis_title='Power (W)',
                    legend_title='Legend',
                    template='plotly_white'
                )
                # Save the graph in html format.
                pred_fig.write_html(
                    f"{path}/{self.model_name}_{self.model_type}_{self.api_name}"
                    f"_prediction_{i + 1}.{fig_format}"
                )
            else:
                # Create a new figure.
                plt.figure(figsize=(12, 8))
                # Plot the ground truth.
                sns.lineplot(x=current_data.index, y=current_data['true'], label='Ground Truth')
                # If the output length is greater than 1, we will plot the mean and the confidence
                # interval.
                if self.out_length > 1:
                    # Plot the prediction.
                    sns.lineplot(x=current_data.index, y=current_data['mean'], label='Prediction')
                    # Fill the confidence interval.
                    plt.fill_between(
                        current_data.index,
                        current_data['mean'] + current_data['upper_bound'],
                        current_data['mean'] - current_data['lower_bound'],
                        # current_data['upper_bound'],
                        # current_data['lower_bound'],
                        color='blue',
                        alpha=0.3,
                        label='Confidence interval (95%)'
                    )
                else:
                    # Plot the prediction.
                    sns.lineplot(
                        x=current_data.index, y=current_data['predicted'], label='Prediction'
                    )
                # Add a horizontal line at 0.
                plt.axhline(y=0, color='black', linestyle='--', linewidth=1.2)
                # Add a title to the graph.
                plt.title(f'Predictions {i + 1} of the dataset')
                # Add labels to the graph.
                plt.xlabel('Time')
                plt.ylabel('Power (W)')
                plt.legend()
                # Save the graph in png format.
                plt.savefig(
                    f"{path}/{self.model_name}_{self.model_type}_{self.api_name}"
                    f"_prediction_{i + 1}.{fig_format}", format=fig_format
                )
                plt.close()
        print('-' * DASH_NB, 'Results plotted!', '-' * DASH_NB)

    def save_predictions(self, predictions):
        """
        Function to save the predictions of the model.

        :param predictions: Predictions of the model.
        """
        path = f"{PREDICTIONS_DIR}/{self.model_type}"
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = (
            f"{path}/{self.model_name}_{self.model_type}_{self.api_name}_predictions.npy"
        )
        with open(file_name, 'wb') as f:
            np.save(f, predictions)

    def load_predictions(self):
        """
        Function to load the predictions of the model.

        :return: Return the predictions of the model.
        """
        path = f"{PREDICTIONS_DIR}/{self.model_type}"
        file_name = (
            f"{path}/{self.model_name}_{self.model_type}_{self.api_name}_predictions.npy"
        )
        with open(file_name, 'rb') as f:
            predictions = np.load(f)
        return predictions


def create_indexed_dataframe(pbar_test, predictions, scaler):
    """
    Function to create a DataFrame with indexed values (true and predicted). We will compute the
    mean, the standard deviation, the margin of error, the lower bound and the upper bound.

    :param pbar_test:   Progress bar for the test set.
    :param predictions: Predictions of the model.
    :param scaler:      Scaler used to scale the data.

    :return: DataFrame with indexed values.
    """
    # Dictionary to hold the indexed values.
    index_dict_true = {}
    index_dict_hat = {}
    idx = []
    # Create the bar.
    for i, (_, y) in pbar_test:
        # For each value in the prediction's window and add the value in the dict.
        for j in range(len(predictions[i])):
            if i + j not in index_dict_hat.keys():
                index_dict_hat[i + j] = [[predictions[i][j]]]
            else:
                index_dict_hat[i + j][0].append(predictions[i][j])
        # Get the true value for the index.
        index_dict_true[i] = float(y['t+1'])
        if i != 0:
            idx.append(y.name)
    # Create the dataframes
    df_pred = pd.DataFrame.from_dict(index_dict_hat, orient='index', columns=['predicted'])
    df_true = pd.DataFrame.from_dict(index_dict_true, orient='index', columns=['true'])
    # Inverse the scaling.
    if scaler:
        df_true = pd.DataFrame(scaler.inverse_transform(df_true), columns=['true'])
    # Concatenate the two dataframes.
    df = pd.concat([df_true, df_pred], axis=1)
    # Compute the mean.
    df['mean'] = df['predicted'].apply(lambda x: np.mean(x))
    # Compute the standard deviation.
    df['std'] = df['predicted'].apply(lambda x: np.std(x))
    # Compute the number of samples.
    df['n'] = df['predicted'].apply(len)
    # Compute the margin of error (95% confidence interval).
    df['margin_of_error'] = df.apply(
        lambda row:
        stats.t.ppf((1 + 0.95) / 2., row['n'] - 1) * (row['std'] / np.sqrt(row['n'])),
        axis=1
    )
    # Compute the lower bound.
    df['lower_bound'] = df['mean'] - df['margin_of_error']
    # Compute the upper bound.
    df['upper_bound'] = df['mean'] + df['margin_of_error']
    # Drop the NaN values.
    df.dropna(inplace=True)
    # Reset the index.
    df.index = idx
    return df
