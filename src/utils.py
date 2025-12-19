"""
This module is the main module of the project.
"""

__title__: str = "py_to_mysql"
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
from typing import List, NoReturn

# Imports third party libraries
# import datetime
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Imports from src
from config import (
    BASE_URL_WEATHER,
    PROCESSED_DATA_DIR,
    DASH_NB
)

# ----------------------------------------------------------------------------------------------- #
# -------------------------------------- GLOBAL VARIABLES --------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# -------------------------------------------- PATHS -------------------------------------------- #


# ------------------------------------- Execution Variables ------------------------------------- #
HOST: str = "192.168.0.69"
PORT: int = 3306
CREDENTIALS_FILE: str = "credentials.json"


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
        # start_time = f"{year}-01-01T00%3A00%3A00Z%2F"
        # end_time = f"{year}-12-31T23%3A50%3A00Z"
        for month in range(1, 4):
            # Format the month to be two digits.
            start_month = f"{1+(month-1)*4:02d}"
            end_month = f"{4*month:02d}"
            print(
                DASH_NB * '-',
                f"Downloading data for {year}-{start_month} to {year}-{end_month}...",
                DASH_NB * '-'
            )
            # Create the start and end time for the request.
            start_time = f"{year}-{start_month}-01T00%3A00%3A00Z%2F"
            end_time = f"{year}-{end_month}-31T23%3A50%3A00Z"

            url = f"https://opendata.meteo.be/service/ows?service=wfs&version=2.0.0&request=getFeature&typenames=aws:aws_10min&outputformat=csv&CQL_FILTER=code+IN+(6472%2C6455%2C6414%2C6459)+AND+timestamp+DURING+{start_time}{end_time}"
            response = requests.get(url, stream=True)

            if response.status_code == 200 and len(response.text) > 0:
                df = pd.read_csv(StringIO(response.text), sep=",")
                
                df.to_csv(output_file, mode='a', index=False, header=first)
                first = False
                print(DASH_NB * '-', f"Data for {year} downloaded with {len(df)} lines.", DASH_NB * '-')
            else:
                print(DASH_NB * '-', f'Error {response.status_code} or empty file for {year}-{start_month} to {year}-{end_month}', DASH_NB * '-')
                # print(DASH_NB * '-', f'Error {response.status_code} or empty file for {year}', DASH_NB * '-')

    print(DASH_NB * '-', f"Data downloaded from IRM API in the file: {output_file}", DASH_NB * '-')


def concat_production_sites(
    dataset: pd.HDFStore, sites: List[str], info: bool = False
) -> pd.DataFrame:
    """
    Function to concatenate the production data of the sites. In this case, we only need to
    concatenate the production data of the sites (CHAMAIEG et CHAMPAIEG).

    :param dataset:     HDF5 dataset.
    :param sites_sn:    List of sites.
    :param info:        Display information (True) or not (False).

    :return:            Concatenated production data.
    """
    # Get the first timestamp of the data.
    start_date = min(
        dataset[f"/{site}"].set_index('ts').index.min()
        for site in sites
        for prediction_type in ['consumption', 'production']
        if not (prediction_type == 'consumption' and site.split('_')[-1].split('/')[0] in ['217158317', '217158406', '250692408'])
    )
    # Get the last timestamp of the data.
    end_date = max(
        dataset[f"/{site}"].set_index('ts').index.max()
        for site in sites
        for prediction_type in ['consumption', 'production']
        if not (prediction_type == 'consumption' and site.split('_')[-1].split('/')[0] in ['217158317', '217158406', '250692408'])
    )

    # Create a dataframe with zeros for the same index as the chamaieg_data.
    index = pd.date_range(start=start_date, end=end_date, freq='15min')
    df = pd.DataFrame(0, index=index, columns=['ap'])

    # Add all chamaieg data to the new dataframe.
    for site in sites:
        sn = site.split('_')[-1].split('/')[0]
        for prediction_type in ['consumption', 'production']:
            if prediction_type == 'consumption' and sn in ['217158317', '217158406', '250692408']:
                continue
            temp_data = dataset[f"/{site}"].set_index('ts')[['ap']]
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
            df = concat_production_sites(
                dataset, site_name, info=False
            )
            site_name = '*'.join(site_name)
        else:
            df = dataset[site_name].set_index('ts')[['ap']]
            df.index = pd.to_datetime(df.index)
            df = df.asfreq('15min').fillna(0)
        df = df.reset_index(names=['ts'])
        df['site_id'] = site_id
        df.set_index(['ts', 'site_id'], inplace=True)
        if period2drop and site_id in period2drop:
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
        val_window = int((train_size + val_size) / k)
        train_end_idx = val_window * (k_th + 1)
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + val_window

        if val_end_idx >= len(df):  # safety check
            return train, val, test

        train[site_name] = {
            'start': df.index[0],
            'stop': df.iloc[train_end_idx - 1].name
        }
        val[site_name] = {
            'start': df.iloc[val_start_idx].name,
            'stop': df.iloc[val_end_idx - 1].name
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
    # Initialize the counter.
    counter = 1
    # Initialize the train, validation and test sets.
    train, val, test = {}, {}, {}
    # Get the minimum length of the dataset.
    min_length = in_length + out_length
    # For each site in the dataset.
    target = target if target else dataset.keys()
    for site in tqdm(target, desc='Creating temporal split'):
        site = (counter, site) if isinstance(site, str) else site
        train, val, test = create_split(
            site, dataset, min_length, train, val, test, train_ratio, val_ratio,
            test_ratio, period2drop=period2drop
        )
        counter += 1
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


def spatial_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Function to split the dataset spatially.

    :return:    Return the train, validation and test sets.
    """
    train, val, test = {}, {}, {}
    consumption_homes = [
        (key, len(dataset[key])) for key in dataset.keys() if key.split('/')[2] == 'consumption'
    ]
    total_size = sum(points for _, points in consumption_homes)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    consumption_homes = [
        (key, len(dataset[key])) for key in dataset.keys() if key.split('/')[2] == 'consumption'
    ]
    home_ids, points = zip(*consumption_homes)
    # Create train and test set
    x_train, x_test, y_train, y_test = train_test_split(
        home_ids, points, test_size=0.1, random_state=42
    )
    # From the train set (0.9 of the dataset), create the train and validation sets (0.8 and 0.1)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.115, random_state=42
    )
    print(x_train, x_val, x_test)
    print(y_train, y_val, y_test)


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
        for j in range(len(predictions.iloc[i])):
            if i + j not in index_dict_hat.keys():
                index_dict_hat[i + j] = [[predictions.iloc[i, j]]]
            else:
                index_dict_hat[i + j][0].append(predictions.iloc[i, j])
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
    # # Compute the margin of error (95% confidence interval).
    # df['margin_of_error'] = df.apply(
    #     lambda row:
    #     stats.t.ppf((1 + 0.95) / 2., max(1, row['n'] - 1)) * (row['std'] / np.sqrt(max(1, row['n']))),
    #     axis=1
    # )
    # # Compute the lower bound.
    # df['lower_bound'] = df['mean'] - df['margin_of_error']
    # # Compute the upper bound.
    # df['upper_bound'] = df['mean'] + df['margin_of_error']
    # Drop the NaN values.
    df.dropna(inplace=True)
    # Reset the index.
    df.index = idx
    return df


def plot_predictions(y_true, y_hat, out_size: int, index=None) -> NoReturn:
    """
    Function to plot the predictions.

    :param y_true:      True values.
    :param y_hat:       Predicted values.
    :param out_size:    Size of the output.
    :param index:       Index of the data.
    :param y_scaler:    Scaler used to normalize the data (optional).
    """
    fig = go.Figure()
    # Case where we have multiple outputs (out_size > 1, Multi-Step-Ahead).
    if out_size > 1 and len(y_true.columns) > 1:
        if not isinstance(y_true, pd.DataFrame):
            cols = [f'ap+{i+1}' for i in range(y_true.shape[1])]
            y_true = pd.DataFrame(y_true, columns=cols)
        # Use the index of the y_true DataFrame if it exists.
        if index is not None:
            y_true.index = index[:len(y_true)]
        index = y_true.index.get_level_values('ts')
        # Create the DataFrame with the indexed predictions and the confidence interval.
        processed_df = create_indexed_dataframe(
            tqdm(enumerate(y_true.iterrows()), total=len(y_true), desc='Creating visualization'),
            y_hat
        )

        # Add the real values trace.
        fig.add_trace(go.Scatter(
            x=index, 
            y=processed_df['true'], 
            mode='lines', 
            name='Valeurs réelles',
            line=dict(width=1.5)
        ))
        
        # Add the predictions trace.
        fig.add_trace(go.Scatter(
            x=index, 
            y=processed_df['mean'], 
            mode='lines', 
            name='Prédictions',
            line=dict(width=1.5)
        ))
        
        # Add the confidence interval traces (upper and lower bounds).
        fig.add_trace(go.Scatter(
            x=index,
            y=processed_df['upper_bound'],
            mode='lines',
            line=dict(width=0),
            name='Écart interquartile',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=index,
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
            x=index, 
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
        index = y_true.index.get_level_values('ts')
        # Add the real values trace.
        fig.add_trace(go.Scatter(
            x=index, 
            y=y_true, 
            mode='lines', 
            name='Valeurs réelles',
            line=dict(width=1.5)
        ))
        # Add the predictions trace.
        fig.add_trace(go.Scatter(
            x=index,
            y=y_hat, 
            mode='lines',
            name='Prédictions',
            line=dict(width=1.5)
        ))
        
        # Add the absolute error trace.
        error = np.abs(y_true - y_hat)
        fig.add_trace(go.Scatter(
            x=index,
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


def save_predictions(predictions: pd.DataFrame, path: str) -> NoReturn:
    """
    Function to save the predictions in a csv file.

    :param predictions: Predictions of the model.
    :param path:        Path to save the predictions.
    """
    # Save the predictions in a csv file.
    predictions.to_csv(path)


def get_historical_features(
    selected_features: List[str], train_all: pd.DataFrame
) -> List[str]:
    """
    Function to get the historical features of the selected features.

    :param selected_features:   List of selected features.
    :param train_all:          Dataframe with all the data.

    :return:                  List of all the selected features.
    """
    all_selected_features = []
    
    # Loop over all the selected features.
    for feat in selected_features:
        # First add the current feature (if it exists in the dataset)
        if feat in train_all.columns:
            all_selected_features.append(feat)
        
        # Then add all historical values (t-0 to t-7)
        for i in range(8):
            historical_feat = f"{feat}-{i}"
            if historical_feat in train_all.columns:
                all_selected_features.append(historical_feat)
    return all_selected_features

