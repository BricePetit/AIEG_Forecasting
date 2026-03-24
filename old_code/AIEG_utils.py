"""
This module contains utility functions for the AIEG project.
"""

__title__: str = "training_testing"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
from typing import List, NoReturn

# Imports third party libraries
import numpy as np
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, Easter, EasterMonday
from pandas.tseries.offsets import Day
import plotly.graph_objects as go
from tqdm import tqdm

# Imports from src

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNCTIONS ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #


