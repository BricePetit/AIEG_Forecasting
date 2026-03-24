"""
Init file for evaluation module.
"""

__title__: str = "evaluation"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries


# Imports third party libraries


# Imports from src
from .metrics import create_indexed_dataframe

from .plots import plot_predictions

__all__ = [
    "create_indexed_dataframe",
    "plot_predictions",
]
