"""
In this module, we define a generator of data for all houses and a single appliance in TensorFlow.
"""
__title__: str = "tf_generators"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
from typing import (
    NoReturn,
)

# Imports third party libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Imports from src
from .base_generators import (
    BaseDataGenerator,
)

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- CLASSES ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


class TFDataGeneratorInMem(
    BaseDataGenerator, tf.keras.utils.Sequence
):
    """
    Data generator for TensorFlow.
    """
    def __init__(
        self, model_type: str, dataset: pd.HDFStore, sites: list or dict, seq_length: int,
        out_length: int, mode: str, split_type: str = 'temporal', standardize: bool = False,
        batch_size: int = 512, shuffle: bool = False
    ):
        """
        Constructor of the TFDataGeneratorHouses1ApplianceInMem class.

        :param model_type:  Type of the model (Seq2Point, RNN, LSTM, GRU).
        :param dataset:     Dataset in h5 format.
        :param sites:       List of sites or dictionary of sites (with start stop values). It
                            depends on the self.split_mode. If the split_mode is temporal, the sites
                            will be a dictionary with the start and stop values. If the split_mode
                            is spatial, the sites will be a list of sites.
        :param seq_length:   Length of the input sequence.
        :param out_length:  Length of the output sequence.
        :param mode:        Mode of the data generator (train, val, test).
        :param split_type:  Type of split (spatial or temporal).
        :param standardize: Boolean to know if we need to standardize the data.
        :param batch_size:  Batch size.
        :param shuffle:     Boolean to shuffle the indices.
        """
        super().__init__(
            model_type, dataset, sites, seq_length, out_length, mode, split_type, standardize
        )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self) -> int:
        """
        Function to get the number of batches per epoch.

        :return:    Return the number of batches per epoch.
        """
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index: int) -> tuple:
        """
        Function to get one batch of data.

        :param index: Index of the batch.

        :return:    Return the batch of data.
        """
        # Get the random index, ds_name, bid and the offset
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        # Initialize windows
        # windows_x = np.empty((self.batch_size, 3, self.in_length), dtype='float32')
        # windows_y = np.empty((self.batch_size, 3, self.out_length), dtype='float32')
        windows_x = np.empty((self.batch_size, self.in_length), dtype='float32')
        windows_y = np.empty((self.batch_size, self.out_length), dtype='float32')
        # Extract data for all indices at once
        for i, batch_idx in enumerate(batch_indices):
            x_stop = batch_idx + self.in_length
            # windows_x[i] = self.data.iloc[batch_idx:x_stop, [3, 4, 5]].to_numpy(dtype='float32')
            # windows_y[i] = (
            # self.data.iloc[x_stop:x_stop + self.out_length, [3, 4, 5]].to_numpy(dtype='float32')
            # )
            windows_x[i] = self.data.iloc[batch_idx:x_stop]['ap'].to_numpy(dtype='float32')
            windows_y[i] = (
                self.data.iloc[x_stop:x_stop + self.out_length]['ap'].to_numpy(dtype='float32')
            )
        return windows_x, windows_y

    def on_epoch_end(self) -> NoReturn:
        """
        Function to shuffle the indices at the end of each epoch.
        """
        # Shuffle the indices if asked
        if self.shuffle:
            np.random.shuffle(self.indices)
