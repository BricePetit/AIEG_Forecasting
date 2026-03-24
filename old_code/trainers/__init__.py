"""
Init file for trainers module.
"""

__title__: str = "trainers"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries

# Imports third party libraries

# Imports from src
from .tf_trainer import TFTrainer
from .pytorch_trainer import PyTorchTrainer

__all__ = [
    "TFTrainer",
    "PyTorchTrainer",
]
