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
import json
import logging
import os.path

# Imports third party libraries

import mysql.connector
from mysql.connector.abstracts import MySQLConnectionAbstract
from mysql.connector.pooling import PooledMySQLConnection
import pandas as pd
import tqdm

# Imports from src

from config import (
    CREDENTIALS_FILE,
    DATA_DIR,
    HOST,
    LOCAL,
    PORT,
    TABLES_NAMES,
)

# ----------------------------------------------------------------------------------------------- #
# -------------------------------------- GLOBAL VARIABLES --------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# -------------------------------------------- PATHS -------------------------------------------- #


# ------------------------------------- Execution Variables ------------------------------------- #


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


def connect_to_mysql(
        host: str, port: int, credentials: dict
) -> PooledMySQLConnection | MySQLConnectionAbstract:
    """
    Function to connect to a MySQL database.

    :param host:         str, the hostname of the database.
    :param port:         int, the port of the database.
    :param credentials:  dict, the credentials to connect to the database.

    :return:             MySQLConnection, the connection to the database.
    """
    # Connect to the database
    if LOCAL:
        connection = None
    else:
        connection = mysql.connector.connect(
            host=host,
            port=port,
            user=credentials['user'],
            password=credentials['password'],
            # database=credentials['database']
        )
    return connection


SESSION = connect_to_mysql(HOST, PORT, load_credentials(CREDENTIALS_FILE))


def select_query(
        keyspace,
        table_name,
        columns,
        where_clause="",
        limit=None,
        distinct=False,
        join_clause="",
        group_by="",
        order_by=""
):
    """
    General function to construct a SQL SELECT query.


    :param keyspace:        str, the keyspace/database name.
    :param table_name:      str, the name of the table to query.
    :param columns:         list or str, columns to select ('*' or list of columns).
    :param where_clause:    str, optional WHERE clause for filtering.
    :param limit:           int, optional limit on the number of rows to return.
    :param distinct:        bool, if True, adds the DISTINCT keyword to the query.
    :param join_clause:     str, optional JOIN clause for joining tables.
    :param group_by:        str, optional GROUP BY clause for grouping results.
    :param order_by:        str, optional ORDER BY clause for ordering results.
    
    :return:                DataFrame, the result of the query as a pandas DataFrame.
    """
    distinct = "DISTINCT" if distinct else ""
    columns = ", ".join(columns) if isinstance(columns, list) else columns
    join_clause = f"{join_clause}" if join_clause else ""
    where = f"WHERE {where_clause}" if where_clause else ""
    group_by = f"GROUP BY {group_by}" if group_by else ""
    order_by = f"ORDER BY {order_by}" if order_by else ""
    limit = f"LIMIT {limit}" if limit else ""

    query = (
        f"SELECT {distinct} {columns} "
        f"FROM {keyspace}.{table_name} "
        f"{join_clause} "
        f"{where} "
        f"{group_by} {order_by} {limit};"
    )

    logging.debug("===> select query : " + query)
    
    # Assuming select_res_to_df is a predefined function to execute the query and return a DataFrame
    res_df = execute_query(query)
    return res_df


def execute_query(query: str, modify: bool = False):
    """
    Function to execute a query.

    :param query:   str, the query to execute.
    :param modify:  bool, if True, the query is a modification query (INSERT, UPDATE, DELETE).

    :return:        DataFrame, the result of the query as a pandas DataFrame. If the query is a
                    modification query, returns None.
    """
    res = None
    with SESSION.cursor() as cursor:
        cursor.execute(query)
        if modify:
            SESSION.commit()
        else:
            columns = [i[0] for i in cursor.description]
            res = pd.DataFrame(cursor.fetchall(), columns=columns)
    return res


def download_database():
    """
    Function to download the database.
    """
    # For each keyspace and tables.
    for keyspace, tables in TABLES_NAMES.items():
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
                # Get distinct users.
                users = select_query(keyspace, table, ["site", "sn"], distinct=True)
                # Drop the empty users.
                users.drop(users[(users["site"] == "") & (users["sn"] == 0)].index, inplace=True)
                # For each user, download the data and save it to a CSV file.
                for index, user in (
                    tqdm.tqdm(
                        users.iterrows(),
                        total=users.shape[0], desc=f"Downloading {table} in keyspace {keyspace}"
                    )
                ):
                    where_clause = f"site = '{user['site']}' AND sn = {user['sn']}"
                    df = select_query(keyspace, table, columns,  where_clause)
                    df.to_csv(
                        f"{DATA_DIR}/raw/{table}_{user['site']}_{user['sn']}.csv", index=False
                    )
            # If the table is a cpt6 table.
            elif table == "cpt6":
                columns = ["Site", "SN", "msT6", "DR6", "S00"]
                select_query(keyspace, table, columns)
            else:
                columns = ["*"]
                select_query(keyspace, table, columns, limit=10)
