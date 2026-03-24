"""
In this module, we define a generator of data in PyTorch.
"""

__title__: str = "pytorch_generators"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
from typing import Tuple

# Imports third party libraries
import numpy as np
import torch

# Imports from src
from .base_generators import (
    BaseDataGenerator,
)

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- CLASSES ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


class PyTorchDataGenerator(
    BaseDataGenerator, torch.utils.data.Dataset
):
    """
    Generator for all sites.
    """

    def __len__(self) -> int:
        """
        Function to get the number of indices.

        :return:    Return the number of indices.
        """
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function to get one batch of data.

        :param index: Index of the batch.

        :return:    Return the batch of data.
        """
        # Get the idx.
        idx = self.indices[index]
        # Concatenate past and current windows.
        windows_x = np.concatenate(
            (
                self.x_past[idx - self.previous_days:idx - self.previous_days + self.seq_length],
                self.x_now[idx:idx + self.seq_length])
        , axis=1
        ).astype('float32')
        # Compute the start and stop indices for the target variable.
        y_start = idx + self.seq_length + self.lag
        y_stop  = y_start + self.out_length
        # Get the target variable.
        windows_y = self.y[y_start:y_stop]
        return torch.from_numpy(windows_x), torch.from_numpy(windows_y)
