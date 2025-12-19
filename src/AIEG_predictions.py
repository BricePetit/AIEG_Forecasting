"""
This module is used to train and test the models.
"""

__title__: str = "training_testing"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
from io import StringIO
import os
from typing import List, NoReturn, Tuple
import xml.etree.ElementTree as ET


# Imports third party libraries
import cudf
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader
import xgboost as xgb

# Imports from src
from config import (
    BASE_URL_WEATHER,
    IN_DATA,
    OUT_DATA,
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    IS_WEATHER_FEATURES,
    IS_TS_FEATURES,
    IS_STATS_FEATURES,
)

from AIEG_utils import (
    add_ts_features,
    add_stats_features,
    concat_production_sites,
)

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ GLOBAL VAR ----------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNCTIONS ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #

def save_predictions_to_xml(
    predictions_cons: dict, predictions_prod: dict, violations: dict,
    output_file: str="predictions.xml"
):
    """
    In this function, we create the XML file with the predictions. The XML file is created with
    the following structure:
    <predictions>
        <consumption>
            <site name="/aieg_EP_AND_0/consumption">
                <entry timestamp="2024-06-01 10:00">12100.0</entry>
            </site>
            ...
        </consumption>
        <production>
            <site name="aieg_CHAMAIEG_217158317/production">
                <entry timestamp="2024-06-01 10:00">4500.0</entry>
            </site>
            ...
        </production>
        <violation>
            <entry timestamp="2024-06-01 10:00" status="true">
                <deload_sites>
                    <site>aieg_CHAMAIEG_250692408/production</site>
                    ...
                </deload_sites>
            </entry>
            ...
        </violation>
    </predictions>

    :param predictions_cons:    Dictionary with the consumption predictions.
    :param predictions_prod:    Dictionary with the production predictions.
    :param violations:          Dictionary with the violations.
    :param output_file:         Name of the output XML file.
    """
    root = ET.Element("predictions")

    # === Consumption ===
    conso_elem = ET.SubElement(root, "consumption")
    for site, values in predictions_cons.items():
        site_elem = ET.SubElement(conso_elem, "site", attrib={"name": site})
        for date, value in values.items():
            entry_elem = ET.SubElement(site_elem, "entry", attrib={"timestamp": date})
            entry_elem.text = str(value)

    # === Production ===
    prod_elem = ET.SubElement(root, "production")
    for site, values in predictions_prod.items():
        site_elem = ET.SubElement(prod_elem, "site", attrib={"name": site})
        for date, value in values.items():
            entry_elem = ET.SubElement(site_elem, "entry", attrib={"timestamp": date})
            entry_elem.text = str(value)

    # === Violation section ===
    viol_elem = ET.SubElement(root, "violation")
    for ts, (has_violation, sites_to_deload) in sorted(violations.items()):
        entry_elem = ET.SubElement(viol_elem, "entry", attrib={
            "timestamp": ts,
            "status": str(has_violation).lower()
        })
        if has_violation and sites_to_deload:
            deload_elem = ET.SubElement(entry_elem, "deload_sites")
            for site in sites_to_deload:
                ET.SubElement(deload_elem, "site").text = site

    # Save the XML file
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"XML file saved to: {output_file}")


def get_irm_data():
    """
    Function to get the data from the IRM API. The data is in CSV format and is read into a pandas
    dataframe where the data are filtered to get the data for the station HUMAIN.

    :return:    A pandas dataframe with the weather data.
    """
    # Compute the start and end time for the request.
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=IN_DATA * 15 + OUT_DATA * 15)
    # Format the start and end time to the required format.
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Create the request to the IRM API.
    # The request is a WFS request to get the data from the AWS database.
    params = {
        "service": "wfs",
        "version": "2.0.0",
        "request": "getFeature",
        "typenames": "aws:aws_10min",
        "outputformat": "csv",
        "CQL_FILTER": f"timestamp DURING {start_time_str}/{end_time_str}"
    }
    # Send the request to the IRM API.
    response = requests.get(BASE_URL_WEATHER, params=params)
    # Check if the request was successful.
    if response.status_code == 200:
        # Read the CSV data into a pandas dataframe.
        weather = pd.read_csv(StringIO(response.text))
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
    else:
        print(f"Error in the request : {response.status_code}")


def prepare_data_ml(
    dataset: pd.HDFStore, site_name: str, sn: str, prediction_type: str, df: pd.DataFrame = None
) -> Tuple[pd.DataFrame, List[str], List[str]]:
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

    :return:                Processed data, list of features, and list of targets.
    """
    # Create the features and target.
    features, target = [], []
    # Check if the data is already in the dataframe.
    if df is not None:
        data = df.copy()
    else:
        # Get the data.
        data = dataset[f'/aieg_{site_name}_{sn}/{prediction_type}'].set_index('ts')[['ap']]
    # Set the index to datetime.
    data.index = pd.to_datetime(data.index)
    data = data.asfreq('15min')
    # Remove Q1 and Q4 from the data.
    if 'q1' in data.columns:
        data.drop(columns=['q1'], inplace=True)
    if 'q4' in data.columns:
        data.drop(columns=['q4'], inplace=True)
    # Check if we want to use weather data.
    if IS_WEATHER_FEATURES:
        # Get the weather data.
        weather = get_irm_data()
        # Select the weather data according to the data info ts (in index)
        weather = weather.loc[data.index]
        data = pd.concat([data, weather], axis=1)
        # Add weather features.
        features += weather.columns.tolist()
    # Check if we want to use timestamp features.
    if IS_TS_FEATURES:
        # Add time features in the data.
        data = add_ts_features(data)
        # Add the features' names to the list of features.
        features += [
            'hour','hour_sin', 'hour_cos', 'dayofweek', 'dayofweek_sin', 'dayofweek_cos',
            'month', 'month_sin', 'month_cos','day',  'year', 'weekofyear', 'weekend',
            'season', 'season_sin', 'season_cos', 'holiday', 'is_peak_hour'
        ]
    # Check if we want to use statistics features.
    if IS_STATS_FEATURES:
        # Add weather features in the data.
        data = add_stats_features(data, IN_DATA)
        # Add the features' names to the list of features.
        features += [
            'rolling_mean', 'rolling_median', 'rolling_std', 'rolling_min', 'rolling_max',
            'diff_1', 'diff_2', 'ewm_mean'
        ]
    # Shift the data to create the window of the sequence as a feature.
    shifted_data = {
        f'{col}-{j}': data[col].shift(j) for col in data.columns for j in range(IN_DATA)
    }
    # shifted_data = {f't-{j}': data['ap'].shift(j) for j in range(in_data)}
    features.extend(shifted_data.keys())
    shifted_data.update({f't+{k}': data['ap'].shift(-k) for k in range(1, OUT_DATA + 1)})
    # shifted_data = {f't+{k}': data['ap'].shift(-k) for k in range(out_data)}
    target.extend([f't+{k}' for k in range(1, OUT_DATA + 1)])
    data = pd.concat([data, pd.DataFrame(shifted_data)], axis=1)
    data.drop(columns=['ap'], inplace=True)
    # Drop the NaN values.
    data.dropna(inplace=True)
    return data, features, target


def main_xgboost(target_sites, model_name: str = "xgboost_model") -> NoReturn:
    dataset = pd.HDFStore(f'{PROCESSED_DATA_DIR}/aieg.h5', mode='r')
    test_data = []
    site_name, sn, prediction_type = "", "", ""
    for site_id, sites in target_sites.items():
        # Concatenate data for sites in the group if the list has more than one site.
        if len(sites) > 1:
            sn_list = [site.split('_')[2].split('/')[0] for site in sites]
            group_data = concat_production_sites(dataset, sn_list)
            processed_data, features, _ = prepare_data_ml(
                dataset, site_name, sn, prediction_type, IN_DATA, OUT_DATA, df=group_data,
                weather_data=IS_WEATHER_FEATURES, ts_features=IS_TS_FEATURES,
                stats_features=IS_STATS_FEATURES
            )
        else:
            split = sites[0].split('_')
            if len(split) > 3:
                site_name = f"{split[1]}_{split[2]}"
            else:
                site_name = split[1]
            sn, prediction_type = split[-1].split('/')
            processed_data, features, _ = prepare_data_ml(
                dataset, site_name, sn, prediction_type, IN_DATA, OUT_DATA,
                weather_data=IS_WEATHER_FEATURES, ts_features=IS_TS_FEATURES,
                stats_features=IS_STATS_FEATURES
            )
        # Remove the ts index to create a multiindex with site_id.
        processed_data = processed_data.reset_index(names=['ts'])
        processed_data['site_id'] = site_id
        processed_data.set_index(['ts', 'site_id'], inplace=True)
        processed_data['site_id'] = site_id
        # Append the processed group data to the datasets lists.
        test_data.append(processed_data.iloc[int(len(processed_data) * 0.9):])

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(f'{SAVED_MODELS_DIR}/{model_name}.json')

    return xgb_model.predict(cudf.DataFrame.from_pandas(test_data[features]))



def main_cnn_gru():
    cnn_gru_model = PyTorchCNNGRU(
        n_features=features_nb,
        output_size=out_size,
        min_value=0,
        max_value=max_value * 2
    )
    cnn_gru_model.load_state_dict(torch.load(model_path, weights_only=True))

    # Load the new data from the DB.

    # Load the new data from IRM.

    # Create the generator for the new data.

    # Do predictions.

    # Save the predictions in a specific file.
    xgb_model.predict(cudf.DataFrame.from_pandas(test[features]))


def main():
    pass


if __name__ == "__main__":
    # main_cnn_gru()
    main_xgboost()