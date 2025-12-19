"""
This module is used to preprocess the datasets. It will convert the datasets to h5 format, resample
the data.
"""
__title__: str = "run_export"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


# Imports standard libraries
import math
import multiprocessing
import os.path
from typing import (
    List, NoReturn, Tuple
)

# Imports third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, EasterMonday
from pandas.tseries.offsets import Easter, Day
import plotly.graph_objects as go
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

# Imports from src
from config import (
    CREATE_H5,
    DASH_NB,
    NB_WORKERS,
    PLOT_WEEKS,
    PLOTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

from utils import concat_production_sites

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNCTIONS ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #


def get_files_in_directory(directory: str) -> list:
    """
    Function to get the files in a directory.

    :param directory:    Directory to check.

    :return:            Return the files in the directory.
    """
    return [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]


def get_site_name_sn_production(site):
    """
    Function to get the name of the site, if it is a production site or not and the serial number.

    :param site:    Site to get the name and if it is a production site or not.
 
    :return:        Return the name of the site, if it is a production site or not and the serial
                    number. 
    """
    # Split the file name and do not consider the empty string.
    split = [i for i in site.split('_') if i]
    # Set the production to False by default, because if there is no _C or _P or + or -,
    # it is a consumption site.
    production = False
    # Set the site name to an empty string by default.
    site_name = ''
    # Get the serial number of the site.
    sn = split[-1][:-4]
    # Only consider the site name.
    if split[1] in [
        '2010', '2011', '2012', '2013', '2014', '2015',
        '2016', '2017', '2018', '2019', '2020', '2021'
    ]:
        current_split = split[2:-1]
    else:
        current_split = split[1:-1]
    if len(current_split) > 0:
        # Check if the site is a consumption or production site with _C or _P.
        if current_split[-1] in ['C', 'P']:
            production = True if current_split[-1] == 'P' else False
            current_split = current_split[:-1]
        # Create the site name and check if the site is a consumption or production
        # site with + or - at the end of the name.
        site_name = '_'.join(current_split)
        production = True if '-' in site_name or production else False
        site_name = site_name.replace('-', '').replace('+', '')
        if site_name == 'AIEGINPV' and sn == '205024201':
            production = True
    return site_name, sn, production


def preprocess_dataframe_and_save(h5, df, site_name, sn, production) -> NoReturn:
    """
    This function will preprocess the dataframe and save it. To preprocess the dataframe, we 
    resample the data to 15 minutes and sum the values of the columns 'ap', 'q1' and 'q4'. If the
    site is a production site, we multiply the values of the columns 'ap', 'q1' and 'q4' by -1.
    In addition, we change the name of the columns 'dls' to 'ts' and the name of the site to the
    correct name.
    
    :param h5:          HDF5 file to save the dataframe.
    :param df:          Dataframe to save.
    :param site_name:   Name of the site.
    :param sn:          Serial number of the site.
    :param production:  Boolean to check if the site is a production site or not.
    """
    # Replace the dots and the dashes by underscores.
    site_name = site_name.replace('.', '_').replace('-', '_').replace('~', '_')
    # Multiply the values of the columns 'ap', 'q1' and 'q4' by -1 if it is a production site.
    if production:
        for col in ['ap', 'q1', 'q4']:
            df[col] = df[col].apply(lambda x: -x if x < 0 else x)
    # Remove a row if all the row is 0.
    df = df[(df['ap'] != 0) | (df['q1'] != 0) | (df['q4'] != 0)]
    # Check if the DataFrame is not empty. We do not consider sites where there are only zeros.
    if len(df) > 0:
        # Set the index to 'dls' and convert it to datetime.
        df = df.set_index('dls')
        df.index = pd.to_datetime(df.index)
        # Sort the index.
        df.sort_index(inplace=True)
        # Resample the data to 15 minutes and sum the values of the columns 'ap', 'q1' and 'q4'.
        df = (
            df
            .resample('15min')
            .agg({'site': 'first', 'sn': 'first', 'ap': 'sum', 'q1': 'sum', 'q4': 'sum'})
        )
        # Fill the NaN values with the previous values.
        df[['site', 'sn']] = df[['site', 'sn']].ffill().bfill()
        # Give the correct type to site and sn
        df = df.astype({'site': 'str', 'sn': 'int'}, copy=False)
        # Fill the other NaN values with 0.
        df.fillna(0, inplace=True)
        # Reset the index and rename the columns.
        df.reset_index(inplace=True)
        df.rename(columns={'dls': 'ts'}, inplace=True)
        # Replace the site name by the correct name.
        df.replace({df['site'][0]: site_name}, inplace=True)
        # Save the dataframe.
        h5.put(
            f"/aieg_{site_name}_{sn}/{'production' if production else 'consumption'}",
            df,
            format='table'
        )
    

def create_h5() -> NoReturn:
    """
    Function to create the h5 files. We will go through all the sites and create the h5 files.
    
    The h5 file will have the following structure:
    - site_name_sn/consumption/DataFrame with the columns 'ts', 'site', 'sn', 'ap', 'q1', 'q4'.
    - site_name_sn/production/DataFrame with the columns 'ts', 'site', 'sn', 'ap', 'q1', 'q4'.
    """
    print('-' * DASH_NB, 'Creating the h5 files...', '-' * DASH_NB)
    columns = ['site', 'sn', 'dls', 'ap', 'q1', 'q4']
    unique_sites = set()
    sites = sorted(get_files_in_directory(RAW_DATA_DIR))
    h5_file = pd.HDFStore(
        f"{PROCESSED_DATA_DIR}/aieg.h5", "w", complevel=9, complib='blosc'
    )
    # We go through all the sites.
    for site in sites:
        print('-' * DASH_NB, f"Processing {site}...", '-' * DASH_NB)
        # Skip the .DS_Store file.
        if site == '.DS_Store':
            continue
        # Get the name of the site and if it is a production site or not.
        site_name, sn, prod = get_site_name_sn_production(site)
        # Check if the site is already processed. If it is the case, we continue.
        if (site_name, sn, prod) in unique_sites:
            continue
        # If the site is not processed, we add it to the set.
        unique_sites.add((site_name, sn, prod))
        # Create the dataframes for the consumption and production sites.
        current_df = pd.read_csv(f"{RAW_DATA_DIR}/{site}")
        if prod:
            concat_data = pd.DataFrame(columns=columns)
            concat_data_pv = current_df if len(current_df) > 1 else pd.DataFrame(columns=columns)
        else:
            concat_data = current_df if len(current_df) > 1 else pd.DataFrame(columns=columns)
            concat_data_pv = pd.DataFrame(columns=columns)
        # We go through all the sites to check if there is another site with the same name.
        for other_site in sites:
            # Get the name of the site and if it is a production site or not.
            other_site_name, other_sn, other_site_prod = get_site_name_sn_production(other_site)
            # Check if the site is the same as the current site.
            if (
                    site_name == other_site_name and sn == other_sn
                    and site != other_site and prod == other_site_prod
            ):
                # Load the other dataframe.
                other_df = pd.read_csv(f"{RAW_DATA_DIR}/{other_site}")
                # Skip if the other dataframe is empty or has only one row. We skip when there is
                # only one row because it means that the only data has not the good format.
                if len(other_df) <= 1:
                    continue
                # Concatenate the dataframes.
                if other_site_prod:
                    concat_data_pv = (
                        pd.concat([concat_data_pv, other_df], axis=0)
                        if not concat_data_pv.empty else other_df
                    )
                else:
                    concat_data = (
                        pd.concat([concat_data, other_df], axis=0)
                        if not concat_data.empty else other_df
                    )
        # Save the dataframes.
        if prod:
            if len(concat_data_pv) > 0:
                print(DASH_NB * '-', 'Saving production...', DASH_NB * '-')
                preprocess_dataframe_and_save(h5_file, concat_data_pv, site_name, sn, prod)
        else:
            if len(concat_data) > 0:
                print(DASH_NB * '-', 'Saving consumption...', DASH_NB * '-')
                preprocess_dataframe_and_save(h5_file, concat_data, site_name, sn, prod)
    h5_file.close()
    print('-' * DASH_NB, 'H5 file created!', '-' * DASH_NB)


def plot_by_week(site, df, pic_format='svg') -> NoReturn:
    """
    Function to plot all the data by week of a site.
    
    :param site:        Site to plot.
    :param df:          Dataframe of the site.
    :param pic_format:  Format of the picture to save.
    """
    # Offset to display week by week. 4 is the number of 15min in 1h.
    offset_app = int(4 * 24 * 7)
    if 'FLOGVE2' in site and '205603983' in site:
        return
    if 'CASPOMP' in site and '202015241' in site:
        return
    if '1904GPRS' in site and ('212648428' in site or '212303520' in site):
        return
    # Display the data for each week
    if pic_format == 'html':
        for i in range(math.ceil(len(df) / offset_app)):
            # Use plotly to plot the data
            fig = go.Figure()
            # Plot the data.
            fig.add_trace(
                go.Scatter(
                    x=df['ts'][i * offset_app: (i + 1) * offset_app],
                    y=df['ap'][i * offset_app: (i + 1) * offset_app]
                )
            )
            fig.add_hline(y=0)
            # Update the layout.
            fig.update_layout(
                title=f'{site} - Semaine {i + 1}',
                xaxis_title='Time',
                yaxis_title='Power (W)'
            )
            # Save the graph in html format.
            fig.write_html(
                f"{PLOTS_DIR}/html_weeks/{site.replace('/', '_')}_semaine_{i + 1}.html"
            )
    else:
        for i in range(math.ceil(len(df) / offset_app)):
            # Create the figure
            plt.figure(figsize=(12, 8))
            # Plot the data
            sns.lineplot(
                x=df['ts'][i * offset_app: (i + 1) * offset_app],
                y=df['ap'][i * offset_app: (i + 1) * offset_app]
            )
            # Update the layout.
            plt.title(f'{site} - Semaine {i + 1}')
            plt.xlabel('Time')
            plt.ylabel('Power (W)')
            plt.axhline(y=0, color='black', linestyle='--')
            # Save the graph in svg format.
            plt.savefig(
                f"{PLOTS_DIR}/html_weeks/{site.replace('/', '_')}_semaine_{i + 1}.{pic_format}",
                format=pic_format
            )
            # Close the figure.
            plt.close()


def parallelize_function(func) -> NoReturn:
    """
    Function to parallelize the given function.

    :param func:    Function to parallelize.
    """
    dataset = pd.HDFStore(f'{PROCESSED_DATA_DIR}/aieg.h5', "r")
    with multiprocessing.Pool(NB_WORKERS) as p:
        # Create a dictionary containing the dataset name and the dataset path.
        func_map = {
            site: p.apply_async(
                # Apply the function
                func,
                (
                    site, dataset[site]
                )
            )
            # For each dataset
            for site in dataset
        }
        # For each process, we wait the end of the execution
        for _, mapped_func in tqdm(func_map.items(), total=len(func_map), desc='Plotting by week'):
            mapped_func.wait()
    dataset.close()


def add_ts_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Function to add features to the data. Based on the timestamp, the function will add the
    following features:
    - hour:         Hour of the day.
    - day:          Day of the month.
    - month:        Month of the year.
    - year:         Year.
    - dayofweek:    Day of the week.
    - weekofyear:   Week of the year.
    - weekend:      Weekend (1) or not (0).
    - season:       Season of the year.
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
    df = data.copy()
    ts = df.index.get_level_values("ts")
    holidays = calendar.holidays(start=ts.min(), end=ts.max())
    df['hour'] = ts.hour
    # Cyclic encoding for the hour of the day.
    df['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24)
    df['dayofweek'] = ts.dayofweek
    # Cyclic encoding for the day of the week.
    df['dayofweek_sin'] = np.sin(2 * np.pi * ts.dayofweek / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * ts.dayofweek / 7)
    df['month'] = ts.month
    # Cyclic encoding for the month of the year.
    df['month_sin'] = np.sin(2 * np.pi * ts.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * ts.month / 12)
    df['day'] = ts.day
    df['year'] = ts.year
    df['weekofyear'] = ts.isocalendar().week.values
    df['weekend'] = df['dayofweek'].apply(lambda x: 1 if x > 4 else 0)
    df['season'] = df['month'].apply(
        lambda x: 0 if x in [12, 1, 2] else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3
    )
    # Cyclic encoding for the season of the year.
    df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4)
    df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4)
    df['holiday'] = ts.isin(holidays).astype(int)
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


def denormalize_data(
    y_scaler: dict, true_df: pd.DataFrame, df_preds: pd.DataFrame, out_size: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to denormalize the data using the scaler for each site_id.

    :param y_scaler:    Dictionary containing the scaler for each site_id.
    :param true_df:     DataFrame containing the true values.
    :param df_preds:    DataFrame containing the predicted values.
    :param out_size:    Size of the output (1 for single-step, >1 for multi-step).

    :return:            Denormalized true_df and df_preds DataFrames.
    """
    # Get unique site_ids from the multi-index.
    site_ids = true_df.index.get_level_values('site_id').unique()
    all_true_df = []
    all_df_preds = []
    is_scaler = False
    for scaler in y_scaler.values():
        if scaler is not None:
            is_scaler = True
            break
    if not is_scaler:
        return true_df, df_preds
    for site_id in site_ids:
        # Check if scaler exists for this site_id.
        if site_id not in y_scaler or y_scaler[site_id] is None:
            continue
        # Create mask for this site_id to avoid repeated index operations.
        site_data_true = true_df.xs(site_id, level='site_id')
        site_data_preds = df_preds.xs(site_id, level='site_id')

        if out_size > 1:
            # For multi-step predictions, process each column separately
            # Extract values, reshape for scaler, and inverse transform
            true_values = site_data_true.values
            pred_values = site_data_preds.values
            try:
                # Inverse transform the values using the scaler for this site_id.
                true_scaled = y_scaler[site_id].inverse_transform(true_values)
                pred_scaled = y_scaler[site_id].inverse_transform(pred_values)
                # Update the true_df and df_preds with the inverse transformed values.
                all_true_df.append(true_scaled)
                all_df_preds.append(pred_scaled)
            except Exception as e:
                print(
                    f"Warning: Could not inverse transform for site {site_id}: {e}"
                )
        else:
            # For single-step predictions, there is only one column
            try:
                # Extract values, reshape for scaler, and inverse transform.
                true_values = site_data_true.values#.reshape(-1, 1)
                pred_values = site_data_preds.values#.reshape(-1, 1)
                true_scaled = y_scaler[site_id].inverse_transform(true_values).flatten()
                pred_scaled = y_scaler[site_id].inverse_transform(pred_values).flatten()
                # Update the true_df and df_preds with the inverse transformed values.
                all_true_df.append(true_scaled)
                all_df_preds.append(pred_scaled)
            except Exception as e:
                print(f"Warning: Could not inverse transform for site {site_id}: {e}")
    denormalized_true_df = pd.DataFrame(
        np.concatenate(all_true_df),
        index=true_df.index,
        columns=[f'{col}' for col in true_df.columns]
    )
    denormalized_df_preds = pd.DataFrame(
        np.concatenate(all_df_preds),
        index=df_preds.index,
        columns=[f'{col}' for col in true_df.columns]
    )
    return denormalized_true_df, denormalized_df_preds


def generate_shifted_data(
    data: pd.DataFrame,
    in_data: int,
    out_data: int,
    previous_days:int = 0,
    selected_target: List[str] = None,
    step: int = 1
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
    :param step:            Step size for shifting the data.

    :return:                Shifted DataFrame, list of features, and list of targets.
    """
    features_list = []
    target_list = []
    day_shift = previous_days * 96
    stats_features = [
        "rolling_mean", "rolling_median", "rolling_std", "rolling_min", "rolling_max",
        "diff_1", "diff_2", "ewm_mean"
    ]
    shifted_data = pd.concat(
        [
            data[col].shift(
                j + (day_shift if col.startswith('ap') or col in stats_features else 0)
            ).rename(
                f"{col}-{j + (day_shift if col.startswith('ap') or col in stats_features else 0)}"
            )
            for col in data.columns
            for j in range(in_data)
        ],
        axis=1
    )
    features_list = shifted_data.columns.tolist()
    # If the selected target is provided, we create the target by shifting the data.
    if selected_target:
        target_df = [
            data['ap'].shift(-int(t.split('+')[1])).rename(t)
            for t in selected_target
            if '+' in t
        ]
        target_list = list(selected_target)
    else:
        target_df = [
            data['ap'].shift(-k).rename(f'ap+{k}')
            for k in range(1, out_data + 1)
        ]
        target_list = [f'ap+{k}' for k in range(1, out_data + 1)]
    shifted_data = pd.concat([shifted_data] + target_df, axis=1)
    # Remove the rows with NaN values.
    shifted_data.dropna(inplace=True)
    shifted_data = shifted_data.iloc[::step, :]
    return shifted_data, features_list, target_list



def prepare_data_ml(
    dataset: pd.HDFStore, site_name: str, sn: str, prediction_type: str, in_data: int,
    out_data: int, df: pd.DataFrame = None, weather_data: bool = False, ts_features: bool = False,
    stats_features: bool = False, normalize: bool = False, selected_target: List[str] = None,
    previous_days: int = 0, step: int = 1, site_id: int = 0
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
    :param step:            Step size for shifting the data.
    :param site_id:         Site ID to add as a multi-index.

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
    data.index.set_names(['ts'], inplace=True)
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
        data, in_data, out_data, previous_days, selected_target, step
    )
    # Normalize the data if requested.
    if normalize:
        data, y_scaler = normalize_data(data, features, target)
    # Check if we want to add the multi-index with the site_id.
    if site_id:
        data = add_multi_index(data, site_id)
    return data, features, target, y_scaler if normalize else None


def select_features(
    site_name, sn, prediction_type, in_data: int, out_data: int, weather_data: bool = False,
    ts_features: bool = False, stats_features: bool = False, normalize: bool = False
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Function to select the features of a site.

    :param site_name:       Name of the site.
    :param sn:              Serial number.
    :param prediction_type: Type of prediction (consumption/production).
    :param in_data:         The number of input data.
    :param out_data:        The number of output data.
    :param weather_data:    Use weather data (True) or not (False).
    :param ts_features:     Use timestamp features (True) or not (False).

    :return:                Processed data, list of features, and list of targets.
    """
    data, features, target, y_scaler = prepare_data_ml(
        pd.HDFStore(f'{PROCESSED_DATA_DIR}/aieg.h5', "r"), site_name, sn, prediction_type, in_data,
        out_data, weather_data=weather_data, ts_features=ts_features, stats_features=stats_features,
        normalize=normalize
    )
    data.drop(columns=['ap'], inplace=True)
    columns2keep = [f for f in data.columns if not (f.startswith('t-'))]
    data2corr = data[columns2keep]
    # Plot the correlation matrix.
    corr = data2corr.corr(method='spearman')
    # Select the K best features.
    selector = SelectKBest(f_regression, k=13).fit(data[columns2keep], data[target])
    info = selector.get_support()
    selector_scores = selector.scores_
    print(f"\nSelected features for {site_name} - {sn} - {prediction_type} with the K best technique:\n")
    for i in range(len(columns2keep)):
        if info[i]:
            print(f"{columns2keep[i]}: {selector_scores[i]:.2f}")
    print(f"\nSelected features for {site_name} - {sn} - {prediction_type} with the correlation:\n")
    print(corr['ap-0'][(corr['ap-0'] > 0.1) | (corr['ap-0'] < -0.1)].drop(columns=['ap-0']))
    # Get the K best features.
    k_best_list = list(selector.get_feature_names_out())
    # Get features with a correlation higher than 0.1 or lower than -0.1.
    corr_list = list(corr['ap-0'][(corr['ap-0'] > 0.1) | (corr['ap-0'] < -0.1)].index)
    # Print the selected features.
    print("\nIntersection with features of the K best technique and the correlation:\n")
    print(set(corr_list) & set(k_best_list))
    print()
    selected_features = set(corr_list) & set(k_best_list)
    for f in data.columns:
        if f.startswith('t-'):
            selected_features.add(f)
    print(f"Selected features: {selected_features}")
    print(target)
    return data[list(selected_features) + target], list(selected_features), target, y_scaler


def add_multi_index(data: pd.DataFrame, site_id: int) -> pd.DataFrame:
    """
    Function to add a multi-index to the data. The function will reset the index and add the
    site_id as a new column. The index will be set to the timestamp and the site_id.

    :param data:    DataFrame containing the data.
    :param site_id: Site ID to add to the data.

    :return:        DataFrame with the multi-index.
    """
    # Add the site_id as a new column and set the index to the timestamp and the site_id.
    data['site_id'] = site_id
    # Set the index to the timestamp and the site_id.
    data.set_index('site_id', append=True, inplace=True)
    # Force names of levels of the multi-index.
    data.index.set_names(['ts', 'site_id'], inplace=True)
    # Add the site_id to the data.
    data['site_id'] = site_id
    return data


def load_data(
    dataset: pd.HDFStore, sites: List[str], in_size: int, out_size: int,
    weather_data: bool = False, ts_features: bool = False, stats_features: bool = False,
    normalize: bool = False, selected_target: List[str] = None, previous_days: int = 0,
    step: int = 1, site_id: int = 0
) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Function to load the data from the dataset. The function will concatenate the data for the
    sites in the group if the list has more than one site. If the list has only one site, it will
    load the data for that site.

    :param dataset:         HDF5 dataset.
    :param sites:           List of sites to load the data for.
    :param in_size:         The number of input data.
    :param out_size:        The number of output data.
    :param weather_data:    Use weather data (True) or not (False).
    :param ts_features:     Use timestamp features (True) or not (False).
    :param stats_features:  Use statistics features (True) or not (False).
    :param normalize:       Normalize the data (True) or not (False).
    :param selected_target: List of selected targets to use. If None, use the default targets.
    :param previous_days:   Number of previous days to consider for the shifted data.
    :param step:            Step size for shifting the data.

    :return:                Processed data, site name, serial number, and prediction type.
    """
    # Concatenate data for sites in the group if the list has more than one site.
    group_data = concat_production_sites(dataset, sites) if len(sites) > 1 else None
    split = sites[0].split('_')
    if len(split) > 3:
        site_name = f"{split[1]}_{split[2]}"
    else:
        site_name = split[1]
    sn, prediction_type = split[-1].split('/')
    processed_data, _, _, _ = prepare_data_ml(
        dataset, site_name, sn, prediction_type, in_size, out_size, group_data, weather_data,
        ts_features, stats_features, normalize, selected_target, previous_days, step, site_id
    )
    return processed_data, site_name, sn, prediction_type


def drop_useless_perdiod(period2drop: dict, processed_data: pd.DataFrame, site_id: str) -> pd.DataFrame:
    """
    Function to drop the useless periods from the processed data. The periods to drop are defined
    in the period2drop dictionary. The periods are defined as a tuple of start and end timestamps.
    If the start or end is "start" or "end", it will use the minimum or maximum timestamp
    of the processed data respectively.

    :param period2drop:      Dictionary containing the periods to drop.
    :param processed_data:   DataFrame containing the processed data.
    :param site_id:          Site ID to drop the periods for.

    :return:                DataFrame with the useless periods dropped.
    """
    # Check if the site_id is in the period2drop dictionary.
    if site_id in period2drop:
        # If it is, we drop the periods from the processed data.
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
    return processed_data


def main() -> NoReturn:
    """
    Main function to preprocess the datasets.
    """
    if CREATE_H5:
        create_h5()
    if PLOT_WEEKS:
        parallelize_function(plot_by_week)
    # dataset = pd.HDFStore(f'{PROCESSED_DATA_DIR}/aieg.h5', "r")

    # plot_by_week('/aieg_CRECHCOM_212303541/production', dataset['/aieg_CRECHCOM_212303541/production'], 'png')
    # plot_by_week('/aieg_AIEGSEIL_212303559/production', dataset['/aieg_AIEGSEIL_212303559/production'], 'png')
    # for key in dataset:
    #     print(DASH_NB * '-', f'site: {key}', DASH_NB * '-')
    #     df = dataset[key]
    #     if 'consumption' in key:
    #         print('this is a consumption site')
    #         percentage = len(df[df['ap'] > 0]) / len(df) * 100
    #         print(f"there are {len(df[df['ap'] > 0])} non zeros values")
    #         print(f"there are {len(df[df['ap'] <= 0])} zeros values")
    #         print(f"the percentage of non zeros values is {percentage}%")
    #     else:
    #         print("this is a production site site")
    #         percentage = len(df[df['ap'] < 0]) / len(df) * 100
    #         print(f"there are {len(df[df['ap'] < 0])} non zeros values")
    #         print(f"there are {len(df[df['ap'] >= 0])} zeros values")
    #         print(f"the percentage of non zeros values is {percentage}%")
    # dataset.close()
