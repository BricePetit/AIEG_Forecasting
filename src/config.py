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
import platform

# Imports third party libraries

# Imports from src


# ----------------------------------------------------------------------------------------------- #
# -------------------------------------- GLOBAL VARIABLES --------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# -------------------------------------------- PATHS -------------------------------------------- #
if platform.system() == 'Darwin':
    # Path to the working directory
    WORKING_DIR: str = (
        "/Users/bricepetitulb/Library/CloudStorage/OneDrive-UniversitéLibredeBruxelles/ULB/PhD/"
        "VdE/AIEG/repartition_key"
    )
    # Path to the base directory.
    BASE_DIR: str = "/Users/bricepetitulb/AIEG"
elif platform.system() == 'Linux':
    # Path to the base directory.
    BASE_DIR: str = "/home/iridia-tower/Bureau/bripetit_phd/aieg"
    # Path to the working directory
    WORKING_DIR: str = f"{BASE_DIR}/repartition_key"
else:
    raise Exception('OS not supported.')
# Path to the data folder.
DATA_DIR: str = f"{BASE_DIR}/data"
# Path to the raw data folder.
RAW_DATA_DIR: str = f"{DATA_DIR}/raw"
# Path to the processed data folder.
PROCESSED_DATA_DIR: str = f"{DATA_DIR}/processed"
# Path to the models' folder.
MODELS_DIR: str = f"{BASE_DIR}/models"
# Path to the plots' folder.
PLOTS_DIR: str = f"{BASE_DIR}/repartition_key/plots"
# Path to the saved predictions' folder.
PREDICTIONS_DIR: str = f"{BASE_DIR}/predictions"
# Path to the saved models' folder.
SAVED_MODELS_DIR: str = f"{BASE_DIR}/saved_models"

# ------------------------------------- Execution Variables ------------------------------------- #
# MySQL Host and Port
LOCAL = True
HOST: str = "192.168.0.69"
PORT: int = 3306
# Credentials file
CREDENTIALS_FILE: str = f"{WORKING_DIR}/credentials.json"
TABLES_NAMES: dict = {
    "regie": ["std"],
    "regie_archives": [
        "std_2010", "std_2011", "std_2012", "std_2013", "std_2014", "std_2015", "std_2016",
        "std_2017", "std_2018", "std_2019", "std_2020", "std_2021", "cpt6"
    ],
}
# Pour les tables std, on veut site, sn, dls, ap, q1, q4.
# Pour les tables std smart, on veut site, sn, dls, ap
# Pour Std pv, on s'en fiche pour l'instant
# Pour la table cpt6, on veut Site, SN, msT6, DR6, S00
DOWNLOAD_WEATHER: bool = True
# Base URL for the IRM API.
BASE_URL_WEATHER: str = "https://opendata.meteo.be/service/ows"
# Set to true if you want to preprocess the data. 
PREPROCESSING: bool = False
# Set to true if you want to download the data.
DOWNLOAD_DATA: bool = False
# Set to True if you want to create the h5 file.
CREATE_H5: bool = False
# Set to True if you want to plot the data by week.
PLOT_WEEKS: bool = False
# Set to true if you want to train the models.
TRAINING: bool = False
# The number of dash in the text.
DASH_NB: int = 20
# The number of workers.
NB_WORKERS: int = 8
# Size of the input data.
IN_DATA: int = 8
# Size of the output data.
OUT_DATA: int = 96
# Size of the batch.
BATCH_SIZE: int = 64
# Number of epochs.
NUM_EPOCHS: int = 100
# Patience for the training.
PATIENCE: int = 10
# Learning rate for the training.
LEARNING_RATE: float = 0.001
# Set to True if you want to use the weather features.
IS_WEATHER_FEATURES: bool = True
# Set to True if you want to use the time series features.
IS_TS_FEATURES: bool = True
# Set to True if you want to use the stats features.
IS_STATS_FEATURES: bool = True
# The production sites are grouped by their site_id.
production_sites_grouped = {
    1 : [
        "aieg_CHAMAIEG_217158317/production", "aieg_CHAMAIEG_217158406/production",
        "aieg_CHAMAIEG_219792511/consumption", "aieg_CHAMAIEG_219792511/production",
        "aieg_CHAMAIEG_250692408/production"
    ],
    2 : ["aieg_CHAAIEG2_250692408/consumption"],
    3 : ["aieg_SPORAIEG_217158317/consumption"],
    4 : ["aieg_AIEGINPV_250692460/consumption"],
    5 : ["aieg_FABLANTO_213310817/production"],
    6 : ["aieg_DEPOTVIR_212303594/production", "aieg_DEPOTVIR_0/production"]
}
# The production drop periods are defined for each site_id.
production_drop_period = {
    1 : [["2022-01-01","2024-03-22"], ["2024-01-29","2024-03-08"], ["2024-12-10", "2025-03-31"]],
    2 : [["2022-01-01","2024-03-22"], ["2024-01-29","2024-03-08"], ["2024-12-10", "2025-03-31"]],
    3 : [
            ["2023-08-07","2023-09-26"], ["2024-01-29","2024-03-22"], ["2024-04-16","2024-07-16"],
            ["2024-09-08", "2024-12-10"], ["2024-12-22","2025-03-28"]
    ],
    4 : [
            ["2023-07-30","2023-08-08"], ["2023-09-18","2023-09-24"], ["2024-01-29","2024-03-21"],
            ["2024-04-27","2024-05-13"], ["2024-06-15","2024-06-25"], ["2025-03-14","2025-03-16"],
            ["2025-03-28","2025-04-04"], ["2025-05-12","2025-05-21"]
    ],
    5 : [["start", "2023-09-06"]],
    6 : [["2024-01-29","2024-03-08"], ["2024-12-10", "2025-03-31"]]
}
# The consumption sites are grouped by their site_id.
consumption_sites_grouped = {
    1 : ["aieg_EP_AND_0/consumption"],
    2 : ["aieg_EP_RUM_0/consumption"],
    3 : ["aieg_EP_OHE_0/consumption"],
    4 : ["aieg_EP_VIR_0/consumption"],
    5 : ["aieg_EP_GVE_0/consumption"],
    6 : ["aieg_AIEGMAR_0/production"],
    7 : ["aieg_FABLANTO_213310817/consumption"],
}
# The consumption drop periods are defined for each site_id.
consumption_drop_period = {
    1 : [["start", "2023-01-01"], ["2025-01-01", "end"]],
    2 : [["start", "2023-01-01"], ["2025-01-01", "end"]],
    3 : [["start", "2023-01-01"], ["2025-01-01", "end"]],
    4 : [["start", "2023-01-01"], ["2025-01-01", "end"]],
    5 : [["start", "2023-01-01"], ["2025-01-01", "end"]],
    6 : [["start", "2023-05-01"]],
    7 : []
}

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNCTIONS ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #
