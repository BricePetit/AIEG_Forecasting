"""
Data pipeline module for the AIEG forecasting project. This module contains the main data pipeline
for the project, which includes data loading, preprocessing, and feature engineering. The data
pipeline is designed to be flexible and modular, allowing for easy integration of new data sources
and preprocessing steps as needed.
"""

__title__: str = "data_pipeline"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports from standard library
from datetime import datetime
import logging
import os

# Imports from third party libraries
import pandas as pd

# Imports from src
from configs.config_loader import ConfigLoader
from data.download_data import download_database
from data.synergrid_slp_loader import download_and_build_synergrid_consumption_csv
from data.weather_loader import download_irm_data
from utils.logging import setup_logger
from utils.site_keys import build_site_key, normalize_site_name, parse_domain_site_key

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------- Globals --------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    setup_logger(log_file="data_pipeline.log", level=logging.INFO)

DASH = '-' * 20

config_loader = ConfigLoader()
config = config_loader.load_global()

RAW_DATA_DIR = config['paths']['paths']['raw_data_dir']
PROCESSED_DATA_DIR = config['paths']['paths']['processed_data_dir']

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ Functions ------------------------------------------ #
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


def get_site_name_sn_production(site: str) -> tuple[str, str, bool]:
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


def get_domain_site_keys() -> set[str]:
    """
    Extract allowed site keys from domain.yaml grouped sections.

    :return: Set of site keys in H5 format.
    """
    domain_cfg = config['domain']
    grouped_sections = [
        domain_cfg.get('production_sites_grouped', {}),
        domain_cfg.get('consumption_sites_grouped', {}),
    ]
    allowed_keys = set()
    for grouped in grouped_sections:
        for _, sites in grouped.items():
            for site in sites:
                allowed_keys.add(f"/{site}" if not str(site).startswith('/') else str(site))
    return allowed_keys


def get_domain_site_tuples() -> set[tuple[str, str]]:
    """
    Build the set of (site_name, sn) tuples from domain grouped sections.

    :return: Set of normalized tuples for download filtering.
    """
    return {parse_domain_site_key(site_key) for site_key in get_domain_site_keys()}


def preprocess_dataframe_and_save(h5, df, site_name, sn, production) -> None:
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
    site_name = normalize_site_name(site_name)
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
        h5.put(build_site_key(site_name, sn, production), df, format='table')
    

def create_h5(only_domain_sites: bool = False) -> None:
    """
    Function to create the h5 files. We will go through all the sites and create the h5 files.
    
    The h5 file will have the following structure:
    - site_name_sn/consumption/DataFrame with the columns 'ts', 'site', 'sn', 'ap', 'q1', 'q4'.
    - site_name_sn/production/DataFrame with the columns 'ts', 'site', 'sn', 'ap', 'q1', 'q4'.

    :param only_domain_sites:   Whether to only include sites present in the domain.yaml groups.
    """
    logger.info("%s Creating the h5 file from raw CSV files %s", DASH, DASH)
    allowed_site_keys = get_domain_site_keys() if only_domain_sites else set()
    if only_domain_sites:
        logger.info(
            "Domain filter enabled: %d site keys loaded from domain.yaml",
            len(allowed_site_keys)
        )
    columns = ['site', 'sn', 'dls', 'ap', 'q1', 'q4']
    unique_sites = set()
    sites = sorted(get_files_in_directory(RAW_DATA_DIR))
    logger.info("Found %d files in raw directory: %s", len(sites), RAW_DATA_DIR)
    selected_count = 0
    skipped_count = 0
    h5_file = pd.HDFStore(
        f"{PROCESSED_DATA_DIR}/aieg.h5", "w", complevel=9, complib='blosc'
    )
    # We go through all the sites.
    for site in sites:
        logger.info("%s Processing file: %s %s", DASH, site, DASH)
        # Skip the .DS_Store file.
        if site == '.DS_Store':
            logger.debug("Skipping ignored file: %s", site)
            continue
        # Get the name of the site and if it is a production site or not.
        site_name, sn, prod = get_site_name_sn_production(site)
        site_key = build_site_key(site_name, sn, prod)
        if only_domain_sites and site_key not in allowed_site_keys:
            skipped_count += 1
            logger.debug("Skipping site not present in domain groups: %s", site_key)
            continue
        if only_domain_sites:
            selected_count += 1
        # Check if the site is already processed. If it is the case, we continue.
        if (site_name, sn, prod) in unique_sites:
            logger.debug("Skipping duplicate site tuple: (%s, %s, %s)", site_name, sn, prod)
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
                logger.info(
                    "%s Saving production data for site=%s sn=%s (%d rows) %s",
                    DASH,
                    site_name,
                    sn,
                    len(concat_data_pv),
                    DASH,
                )
                preprocess_dataframe_and_save(h5_file, concat_data_pv, site_name, sn, prod)
            else:
                logger.warning("No production rows to save for site=%s sn=%s", site_name, sn)
        else:
            if len(concat_data) > 0:
                logger.info(
                    "%s Saving consumption data for site=%s sn=%s (%d rows) %s",
                    DASH,
                    site_name,
                    sn,
                    len(concat_data),
                    DASH,
                )
                preprocess_dataframe_and_save(h5_file, concat_data, site_name, sn, prod)
            else:
                logger.warning("No consumption rows to save for site=%s sn=%s", site_name, sn)
    h5_file.close()
    logger.info("%s H5 file created: %s/aieg.h5 %s", DASH, PROCESSED_DATA_DIR, DASH)
    if only_domain_sites:
        logger.info(
            "Domain filter summary: selected=%d, skipped=%d",
            selected_count,
            skipped_count,
        )


def run_data_pipeline(only_domain_sites: bool = True) -> None:
    """
    Function to run the data pipeline. The data pipeline will download the data from the database
    and create the h5 files. The data pipeline is designed to be flexible and modular, allowing for
    easy integration of new data sources and preprocessing steps as needed.

    :param only_domain_sites:   Whether to only include sites present in the domain.yaml groups. If
                                True, only sites whose keys are present in the domain.yaml grouped
                                sections will be included in the download and processing. This
                                allows for focused data handling based on domain-specific
                                configurations.
    """
    logger.info("%s Starting data pipeline %s", DASH, DASH)
    logger.info("Mode only_domain_sites=%s", only_domain_sites)

    # Force full weather refresh from 2010 to current year for consistency.
    now_year = datetime.now().year
    weather_output_file = f"{PROCESSED_DATA_DIR}/aws_10min.csv"
    logger.info(
        "%s Downloading weather data from 2010 to %d into %s %s",
        DASH,
        now_year,
        weather_output_file,
        DASH,
    )
    download_irm_data(output_file=weather_output_file, start_year=2010, end_year=now_year)
    logger.info("%s Weather data downloaded %s", DASH, DASH)

    # Build synthetic household consumption curve from Synergrid SLP files (2010 -> now).
    synthetic_output_file = f"{PROCESSED_DATA_DIR}/synthetic_consumption_w.csv"
    synthetic_raw_dir = f"{RAW_DATA_DIR}/synergrid_slp"
    logger.info(
        "%s Building Synergrid synthetic household consumption curve into %s %s",
        DASH,
        synthetic_output_file,
        DASH,
    )
    download_and_build_synergrid_consumption_csv(
        output_csv=synthetic_output_file,
        raw_dir=synthetic_raw_dir,
        start_year=2016,
        end_year=now_year,
        profile_kind="rlp0n",
        rlp_start_year=2016,
        dso_target="AIEG",
        small_dso_fallback="SMALL",
    )
    logger.info("%s Synthetic consumption curve ready %s", DASH, DASH)

    allowed_site_tuples = get_domain_site_tuples() if only_domain_sites else None
    if only_domain_sites:
        logger.info(
            "Download filter enabled: %d (site, sn) tuples from domain.yaml",
            len(allowed_site_tuples),
        )
    logger.info("%s Downloading the data from the database %s", DASH, DASH)
    download_database(allowed_sites=allowed_site_tuples)
    logger.info("%s Data downloaded %s", DASH, DASH)
    create_h5(only_domain_sites=only_domain_sites)
    logger.info("%s Data pipeline completed %s", DASH, DASH)


if __name__ == "__main__":
    run_data_pipeline()
