"""
In this module, we define the functions to load the weather data. The data is downloaded from the IRM API and is stored in a CSV file. The data is then loaded into a pandas dataframe and is resampled to 15 minutes.
"""

__title__: str = "weather_loader"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
from io import StringIO
import logging
import os
import requests
from typing import NoReturn

# Imports third party libraries
from datetime import datetime, timezone
import pandas as pd

# Imports from src
from configs.config_loader import ConfigLoader
from utils.logging import setup_logger

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- Globals ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

config_loader = ConfigLoader()
config = config_loader.load_global()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    setup_logger(log_file="weather_loader.log", level=logging.INFO)

DASH = '-' * 20

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ Functions ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #

def download_irm_data(
    output_file: str = f"{config['paths']['paths']['processed_data_dir']}/aws_10min.csv",
    start_year: int=2010,
    end_year: int=None
) -> NoReturn:
    """
    Function to get the data from the IRM API. The data is in CSV format and is read into a pandas
    dataframe where the data are filtered to get the data for the station HUMAIN.

    :param output_file: Path to the output file.
    :param start_year:  Start year for the data download.
    :param end_year:    End year for the data download. If None, the current year is used.
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
        logger.info("Downloading data for %d...", year)

        start_time = f"{year}-01-01T00:00:00Z"
        end_time = f"{year}-12-31T23:59:59Z"

        params = {
            "service": "wfs",
            "version": "2.0.0",
            "request": "getFeature",
            "typenames": "aws:aws_10min",
            "outputformat": "csv",
            "CQL_FILTER": f"timestamp DURING {start_time}/{end_time} AND code in (6472)"
        }

        response = requests.get("https://opendata.meteo.be/service/ows", params=params, stream=True)

        if response.status_code == 200 and len(response.text) > 0:
            df = pd.read_csv(StringIO(response.text), sep=",")
            
            df.to_csv(output_file, mode='a', index=False, header=first)
            first = False
            logger.info("%s Data for %d downloaded with %d lines. %s", DASH, year, len(df), DASH)
        else:
            logger.error(
                "%s Error %d or empty file for %d %s",
                DASH,
                response.status_code,
                year,
                DASH,
            )

    logger.info("%s Data downloaded from IRM API in the file: %s %s", DASH, output_file, DASH)


def get_weather_data() -> pd.DataFrame:
    """
    Function to get the weather data.

    :return:    Weather data.
    """
    # Load the weather data.
    weather = pd.read_csv(f"{config['paths']['paths']['processed_data_dir']}/aws_10min.csv")
    # Load the station data for the weather.
    # station = pd.read_csv(f"{config['paths']['paths']['processed_data_dir']}/aws_station.csv")
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
