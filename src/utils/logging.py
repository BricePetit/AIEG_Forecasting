"""
Logging utilities for the project. This module contains functions for setting up and configuring
logging for the project. It provides a function to set up a logger that logs to both the console
and a file, with a specified logging level and format.
"""

__title__: str = "logging"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports from standard library
import logging
from pathlib import Path

# Imports from third party libraries

# Imports from src

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ Functions ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #

def setup_logger(
    log_dir: str = "logs",
    log_file: str = "pipeline.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure project logger.

    :param log_dir:    Directory where the log file will be saved.
    :param log_file:   Name of the log file.
    :param level:      Logging level (e.g., logging.INFO, logging.DEBUG).

    :return:           Configured logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    # File handler
    file_handler = logging.FileHandler(Path(log_dir) / log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
