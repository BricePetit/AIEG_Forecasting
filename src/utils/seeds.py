"""
Utils for seeding the random number generators for reproducibility.
"""

__title__: str = "seeds"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports from standard library
import os
import random

# Imports from third party libraries
import numpy as np
import torch

# Imports from src

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ Functions ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    """
    Function to set the seed for reproducibility.

    :param seed:    The seed to set.
    """
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch CPU
    torch.manual_seed(seed)
    # Torch GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Python hashing
    os.environ["PYTHONHASHSEED"] = str(seed)
