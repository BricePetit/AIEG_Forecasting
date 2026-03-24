"""
Init file for models module.
"""

__title__: str = "models"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries

# Imports third party libraries

# Imports from src
from .pytorch_models import (
    PyTorchSimpleMLP, PyTorchMLP, PyTorchSeq2Point, PyTorchUNetNilm, PyTorchRNN, PyTorchLSTM,
    PyTorchGRU, PyTorchCNNGRU, PyTorchComplexCNNGRU, PyTorchTimeSeriesTransformer

)
from .tf_models import TFSeq2Point, TFUNetNilm

from .xgb_model import XGBoostModel

__all__ = [
    "PyTorchSimpleMLP",
    "PyTorchMLP",
    "PyTorchSeq2Point",
    "PyTorchUNetNilm",
    "PyTorchRNN",
    "PyTorchLSTM",
    "PyTorchGRU",
    "PyTorchCNNGRU",
    "PyTorchComplexCNNGRU",
    "PyTorchTimeSeriesTransformer",
    "TFSeq2Point",
    "TFUNetNilm",
    "XGBoostModel",
]
