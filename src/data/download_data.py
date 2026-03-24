"""
Module to download the data from the database.
"""

__title__: str = "download_data"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# Imports standard libraries
import json
import logging
import os
from pathlib import Path

# Imports third party libraries
from tqdm import tqdm

# Imports from src
from configs.config_loader import ConfigLoader
from utils.site_keys import normalize_site_name
from .mysql_client import MySQLClient

# ----------------------------------------------------------------------------------------------- #
# -------------------------------------- GLOBAL VARIABLES --------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

config_loader = ConfigLoader()
config = config_loader.load_global()
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNCTIONS ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #

def load_credentials(file: str) -> dict:
    """
    Function to load the credentials from a JSON file.

    :param file:    str, the path to the JSON file.

    :return:        dict, the credentials.
    """
    # Check if the file exists
    if not os.path.exists(file):
        raise FileNotFoundError(f"The file '{file}' does not exist.")
    # Load the credentials
    with open(file, 'r', encoding='utf-8') as f:
        credentials = json.load(f)
    return credentials


def download_database(allowed_sites: set[tuple[str, str]] | None = None) -> None:
    """
    Function to download the database.

    :param allowed_sites: Optional set of (site_name, sn) tuples to restrict downloads.
    """
    tables_name: dict = {
        "regie": ["std"],
        "regie_archives": [
            "std_2010", "std_2011", "std_2012", "std_2013", "std_2014", "std_2015", "std_2016",
            "std_2017", "std_2018", "std_2019", "std_2020", "std_2021", "cpt6"
        ],
    }
    creds = load_credentials(f"{config['paths']['paths']['base_dir']}/credentials.json")
    user, password = creds['user'], creds['password']
    db = MySQLClient("192.168.0.69", 3306, user, password)
    # Create the raw data directory if it does not exist.
    raw_dir = Path(config['paths']['paths']['data_dir']) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # For each keyspace and tables.
    for keyspace, tables in tables_name.items():
        # For each table in the keyspace.
        for table in tables:
            # If the table is a std table.
            if "std" in table:
                # Select the columns to download according to the table.
                if "smart" in table:
                    columns = ["site", "sn", "dls", "ap"]
                elif "pv" in table:
                    columns = ["*"]
                else:
                    columns = ["site", "sn", "dls", "ap", "q1", "q4"]
                # Table ref
                table_ref = f"{keyspace}.{table}"
                # Get distinct users.
                users = db.query(f"SELECT DISTINCT site, sn FROM {table_ref}")
                # Drop the empty users.
                users.drop(users[(users["site"] == "") & (users["sn"] == 0)].index, inplace=True)
                initial_users_count = users.shape[0]
                if allowed_sites is not None:
                    users = users[
                        users.apply(
                            lambda row: (
                                normalize_site_name(row["site"]),
                                str(row["sn"]),
                            ) in allowed_sites,
                            axis=1,
                        )
                    ]
                    logger.info(
                        "Filtering %s users in %s: kept %d/%d based on domain list",
                        table_ref,
                        table,
                        users.shape[0],
                        initial_users_count,
                    )
                # For each user, download the data and save it to a CSV file.
                for _, user in tqdm(
                    users.iterrows(),
                    total=users.shape[0], desc=f"Downloading {table} in keyspace {keyspace}"
                ):
                    df = db.query(
                        f"SELECT {', '.join(columns)} FROM {table_ref} WHERE site=%s AND sn=%s",
                        (user['site'], user['sn'])
                    )
                    df.to_csv(
                        f"{raw_dir}/{table}_{user['site']}_{user['sn']}.csv", index=False
                    )
            # If the table is a cpt6 table.
            elif table == "cpt6":
                columns = ["Site", "SN", "msT6", "DR6", "S00"]
                db.query(f"SELECT {', '.join(columns)}FROM {keyspace}.{table}")
            else:
                db.query(f"SELECT * FROM {keyspace}.{table} LIMIT 10")

