"""
This module contains utility functions for the AIEG project.
"""

__title__: str = "training_testing"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
from typing import List, NoReturn

# Imports third party libraries
import numpy as np
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, Easter, EasterMonday
from pandas.tseries.offsets import Day
import plotly.graph_objects as go
from tqdm import tqdm

# Imports from src

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNCTIONS ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #

def concat_production_sites(
    dataset: pd.HDFStore, sites_sn: List[str], info: bool = False
) -> pd.DataFrame:
    """
    Function to concatenate the production data of the sites. In this case, we only need to
    concatenate the production data of the sites (CHAMAIEG et CHAMPAIEG).

    :param dataset:     HDF5 dataset.
    :param sites_sn:    List of serial numbers of the sites.
    :param info:        Display information (True) or not (False).

    :return:            Concatenated production data.
    """
    # Get the first timestamp of the data.
    start_date = min(
        dataset[f"/aieg_CHAMAIEG_{sn}/{prediction_type}"].set_index('ts').index.min()
        for sn in sites_sn
        for prediction_type in ['consumption', 'production']
        if not (prediction_type == 'consumption' and sn in ['217158317', '217158406', '250692408'])
    )
    # Get the last timestamp of the data.
    end_date = max(
        dataset[f"/aieg_CHAMAIEG_{sn}/{prediction_type}"].set_index('ts').index.max()
        for sn in sites_sn
        for prediction_type in ['consumption', 'production']
        if not (prediction_type == 'consumption' and sn in ['217158317', '217158406', '250692408'])
    )

    # Create a dataframe with zeros for the same index as the chamaieg_data.
    index = pd.date_range(start=start_date, end=end_date, freq='15min')
    df = pd.DataFrame(0, index=index, columns=['ap'])

    # Add all chamaieg data to the new dataframe.
    for sn in sites_sn:
        for prediction_type in ['consumption', 'production']:
            if prediction_type == 'consumption' and sn in ['217158317', '217158406', '250692408']:
                continue
            temp_data = dataset[f"/aieg_CHAMAIEG_{sn}/{prediction_type}"].set_index('ts')[['ap']]
            temp_data.index = pd.to_datetime(temp_data.index)
            temp_data = temp_data.asfreq('15min').fillna(0)
            df = df.add(temp_data, fill_value=0)
            if info:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=temp_data.index, y=temp_data['ap'], mode='lines',
                        name=f'{sn} - {prediction_type}'
                    )
                )
                fig.update_layout(title=f'{sn} - {prediction_type}')
                fig.show()
    if info:
        print(df)
    return df


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


def add_ts_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Function to add features to the data. Based on the timestamp, the function will add the
    following features:
    - hour:         Hour of the day and cyclic encoding.
    - day:          Day of the month.
    - month:        Month of the year and cyclic encoding.
    - year:         Year.
    - dayofweek:    Day of the week (0 = Monday, 6 = Sunday) and cyclic encoding.
    - weekofyear:   Week of the year.
    - weekend:      Weekend (1) or not (0).
    - season:       Season of the year (0 = Winter, 1 = Spring, 2 = Summer, 3 = Autumn) and
                    cyclic encoding.
    - holiday:      Holiday (1) or not (0).

    :param data:    DataFrame containing the data.

    :return:        DataFrame with added features.
    """
    class BelgiumHolidays(AbstractHolidayCalendar):
        """
        Class to define the Belgium holidays.
        """
        rules = [
            Holiday('New Year', month=1, day=1),
            EasterMonday,
            Holiday('Labour Day', month=5, day=1),
            Holiday('Ascension', month=1, day=1, offset=[Easter(), Day(39)]),
            Holiday('Pentecost Monday', month=1, day=1, offset=[Easter(), Day(50)]),
            Holiday('National Day', month=7, day=21),
            Holiday('Assumption', month=8, day=15),
            Holiday('All Saints', month=11, day=1),
            Holiday('Armistice', month=11, day=11),
            Holiday('Christmas', month=12, day=25)
        ]
    calendar = BelgiumHolidays()
    holidays = calendar.holidays(start=data.index.min(), end=data.index.max())
    df = data.copy()
    df['hour'] = df.index.hour
    # Cyclic encoding for the hour of the day.
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dayofweek'] = df.index.dayofweek
    # Cyclic encoding for the day of the week.
    df['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['month'] = df.index.month
    # Cyclic encoding for the month of the year.
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['day'] = df.index.day
    df['year'] = df.index.year
    df['weekofyear'] = df.index.isocalendar().week
    df['weekend'] = df['dayofweek'].apply(lambda x: 1 if x > 4 else 0)
    df['season'] = df['month'].apply(
        lambda x:
        0 if x in [12, 1, 2] else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3
    )
    # Cyclic encoding for the season of the year.
    df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4)
    df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4)
    df['holiday'] = df.index.isin(holidays).astype(int)
    df['is_peak_hour'] = (
        (df['hour'] >= 7) & (df['hour'] <= 22) & (df['weekend'] == 0) & (df['holiday'] == 0)
    ).astype(int)
    return df


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
            sn_list = [sn.split('_')[2].split('/')[0] for sn in site_name]
            df = concat_production_sites(
                dataset, sn_list, info=False
            )
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
