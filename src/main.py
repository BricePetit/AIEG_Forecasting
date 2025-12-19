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
from typing import (
    NoReturn,
)

# Imports third party libraries

# Imports from src
from config import (
    DASH_NB,
    DOWNLOAD_DATA,
    DOWNLOAD_WEATHER,
    PREPROCESSING,
    TRAINING,
)
import py_to_mysql as ptm
import preprocessing
import training_testing
from utils import download_irm_data

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNCTIONS ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #


def main() -> NoReturn:
    """
    Main function to run the project.
    """
    if DOWNLOAD_DATA:
        print('-' * DASH_NB, 'Downloading data...', '-' * DASH_NB)
        ptm.download_database()
        print('-' * DASH_NB, 'Downloading data done!', '-' * DASH_NB)
    if DOWNLOAD_WEATHER:
        print('-' * DASH_NB, 'Downloading weather data...', '-' * DASH_NB)
        download_irm_data()
        print('-' * DASH_NB, 'Downloading weather data done!', '-' * DASH_NB)
    if PREPROCESSING:
        print('-' * DASH_NB, 'Preprocessing...', '-' * DASH_NB)
        preprocessing.main()
        print('-' * DASH_NB, 'Preprocessing done!', '-' * DASH_NB)
    if TRAINING:
        print('-' * DASH_NB, 'Training...', '-' * DASH_NB)
        training_testing.main()
        print('-' * DASH_NB, 'Training done!', '-' * DASH_NB)


if __name__ == '__main__':
    main()
