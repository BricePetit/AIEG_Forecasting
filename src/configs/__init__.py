"""
Init file for configs module.

This module contains functions to load and manage configuration files for the project.
"""


__title__: str = "configs"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries

# Imports third party libraries

# Imports from src

from .config_loader import ConfigLoader

__all__ = [
    "ConfigLoader"
]
