"""
Init file for data module.

This module contains functions to preprocess the data. The main function of this module is the
preprocess_data function, which takes as input the raw data and returns the preprocessed data. The
preprocessing steps include:
- Adding weather features (temperature, humidity, etc.)
- Normalizing the data using the RobustScaler.
- Adding timestamp features (hour, day of the week, month, etc.)
- Adding statistics features (rolling mean, rolling std, etc.)
"""

__title__: str = "data"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries

# Imports third party libraries

# Imports from src

from .dataset_loader import load_complete_data, concat_production_sites, build_group_data

from .download_data import download_database

from .mysql_client import MySQLClient

from .preprocessing import (
    create_split,
    generate_shifted_data,
    normalize_data,
    prepare_data_ml,
    temporal_split
)

from .weather_loader import get_weather_data, download_irm_data

__all__ = [
    "load_complete_data",
    "concat_production_sites",
    "build_group_data",
    "download_database",
    "MySQLClient",
    "create_split",
    "generate_shifted_data",
    "normalize_data",
    "prepare_data_ml",
    "temporal_split",
    "get_weather_data",
    "download_irm_data",
]
