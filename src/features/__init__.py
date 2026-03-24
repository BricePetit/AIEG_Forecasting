"""
Init file for features module.

This module contains functions to add features to the data. The main function of this module is the
add_ts_features function, which takes as input the data and returns the data with added features. The
features added are:
- Timestamp features (hour, day of the week, month, etc.)
- Statistics features (rolling mean, rolling std, etc.)
"""

__title__: str = "features"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries

# Imports third party libraries

# Imports from src
from .stats_features import add_stats_features
from .ts_features import add_ts_features

__all__ = [
    "add_stats_features",
    "add_ts_features",
]
