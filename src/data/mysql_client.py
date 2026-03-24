"""
Module to connect to a MySQL database and execute queries.
"""

__title__: str = "mysql_client"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries

# Imports third party libraries
import mysql.connector
import pandas as pd

# Imports from src


# ----------------------------------------------------------------------------------------------- #
# -------------------------------------- GLOBAL VARIABLES --------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# -------------------------------------------- PATHS -------------------------------------------- #


# ------------------------------------- Execution Variables ------------------------------------- #


# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- CLASSES ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

class MySQLClient:
    """
    Class to connect to a MySQL database and execute queries.
    """
    def __init__(self, host: str, port: int, user: str, password: str):
        """
        Initialization of the MySQLClient class.

        :param host:      The host of the MySQL database.
        :param port:      The port of the MySQL database.
        :param user:      The user of the MySQL database.
        :param password:  The password of the MySQL database.
        """
        self.connection = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password
        )
    
    def query(self, query: str) -> pd.DataFrame | None:
        """
        Function to execute a query.

        :param query:   str, the query to execute.

        :return:        DataFrame, the result of the query as a pandas DataFrame.
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            columns = [i[0] for i in cursor.description]
            return pd.DataFrame(cursor.fetchall(), columns=columns)
