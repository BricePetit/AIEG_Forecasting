"""
In this module, we define a generator of data. First, we define an abstract data generator that
will be used by the other data generators. Then, we define the data generator for
TensorFlow and PyTorch in other files.

Basically, both data generators are the same. The only difference is that the PyTorch data
generator inherits from torch.utils.data.Dataset and the TensorFlow data generator inherits
from tf.keras.utils.Sequence.
"""

__title__: str = "base_generators"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
from abc import ABC, abstractmethod
from typing import Tuple, Union

# Imports third party libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# Imports from src
from preprocessing import add_ts_features, add_stats_features, get_weather_data
from utils import concat_production_sites

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- CLASSES ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


class BaseDataGenerator(ABC):
    """
    Very generic data generator. We use this class to define the abstract methods.
    """

    def __init__(
        self, dataset: pd.HDFStore, sites, seq_length: int, out_length: int,
        split_type: str = 'temporal', is_weather: bool = False, is_ts_features: bool = False,
        is_stats_features: bool = False, lag: int = 0, previous_days: int = 0, step: int = 1,
        specific_features: list = None, period2drop=None,
    ):
        """
        Constructor of the BaseDataGenerator class.

        :param dataset:             Dataset in h5 format.
        :param sites:               List of sites or dictionary of sites (with start stop values).
                                    It depends on the self.split_mode. If the split_mode is
                                    temporal, the sites will be a dictionary with the start and
                                    stop values. If the split_mode is spatial, the sites will be a
                                    list of sites.
        :param seq_length:          Length of the input sequence (window size).
        :param out_length:          Length of the output sequence (window size).
        :param split_type:          Type of split (temporal or spatial).
        :param is_weather:          Boolean to indicate if we want to add the weather data.
        :param is_ts_features:      Boolean to indicate if we want to add the time series features.
        :param is_stats_features:   Boolean to indicate if we want to add the statistics features.
        :param lag:                 Lag to add to the data.
        :param previous_days:       Number of previous days to include in the data.
        :param step:                Step size for shifting the data.
        :param specific_features:   List of specific features to add to the data.
        :param period2drop:        Period to drop from the data.
        """
        super().__init__()

        self.dataset = dataset
        self.seq_length = seq_length
        self.out_length = out_length
        self.split_type = split_type
        self.is_weather = is_weather
        self.is_ts_features = is_ts_features
        self.is_stats_features = is_stats_features
        self.lag = lag
        self.previous_days = previous_days
        self.step = step
        self.specific_features = specific_features
        self.weather = get_weather_data()
        self.weather_features = self.weather.columns.tolist() if self.is_weather else []
        self.ts_features = []
        self.stats_features = []
        self.data, self.indices = self.init_data_and_indices(sites, period2drop)

    @abstractmethod
    def __len__(self) -> int:
        """
        Function to get the number of batches per epoch.

        :return:    Return the number of batches per epoch.
        """

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple:
        """
        Function to get one batch of data.

        :param idx: Index of the batch.

        :return:    Return the batch of data.
        """

    def init_data_and_indices(self, sites, period2drop=None):
        """
        Function to concatenate all the dataframes of the sites and create the indices.
        If we have 3 sites = [[1,2,3,4,5],[10,20,30,40,50],[100,200,300,400,500]]
        and a sequence length of 3, we will have the following indices:
        indices = [0, 1, 2, 5, 6, 7, 10, 11, 12].

        :param sites:   List of sites or dictionary of sites (with start stop values). It depends
                        on the self.split_mode. If the split_mode is temporal, the sites will be a
                        dictionary with the start and stop values. If the split_mode is spatial, the
                        sites will be a list of sites.

        :return:        Return concatenated dataframe of all sites and the indices.
        """
        indices = []
        df_list = []
        start_idx = 0
        previous_days = self.previous_days * 96
        min_required_len = self.seq_length + self.out_length + self.lag + previous_days
        # For each site.
        for site in tqdm(sites, desc='Initialization of the data and indices'):
            # Load the data.
            if 'CHAMAIEG' in site or 'DEPOTVIR' in site:
                sites_list = site.split('*')
                tmp_df = concat_production_sites(self.dataset, sites_list, info=False)
            else:
                tmp_df = pd.DataFrame(self.dataset[site]).set_index('ts')[['ap']]
                tmp_df.index = pd.to_datetime(tmp_df.index)
            site_id = sites[site]["start"][1]
            tmp_df = tmp_df.reset_index(names=['ts'])
            tmp_df['site_id'] = site_id
            tmp_df.set_index(['ts', 'site_id'], inplace=True)
            tmp_df['site_id'] = site_id
            
            if period2drop and site_id in period2drop:
                for start_p, end_p in period2drop[site_id]:
                    if start_p == "start":
                        start_p = tmp_df.index.get_level_values("ts").min()
                    if end_p == "end":
                        end_p = tmp_df.index.get_level_values("ts").max()
                    mask = (tmp_df.index.get_level_values("ts") >= start_p) & \
                        (tmp_df.index.get_level_values("ts") <= end_p)
                    tmp_df = tmp_df.drop(tmp_df[mask].index)

            if self.split_type == 'temporal':
                tmp_df = tmp_df.loc[sites[site]['start']:sites[site]['stop']]
            # Add the weather data if needed.
            if self.is_weather:
                tmp_weather = self.weather.loc[tmp_df.index.get_level_values("ts")]
                tmp_weather = tmp_weather.reset_index(names=['ts'])
                tmp_weather['site_id'] = site_id
                tmp_weather.set_index(['ts', 'site_id'], inplace=True)
                tmp_df = pd.concat([tmp_df, tmp_weather], axis=1)
            # Add the time features if needed.
            if self.is_ts_features:
                tmp_df = add_ts_features(tmp_df)
                self.ts_features = list(set(tmp_df.columns) - {'ap'} - set(self.weather_features))
            # Add the statistics features if needed.
            if self.is_stats_features:
                tmp_df = add_stats_features(tmp_df, self.seq_length)
                self.stats_features = list(
                    set(tmp_df.columns) - {'ap'} - set(self.weather_features)
                    - set(self.ts_features)
                )
            # Select the specific features if needed.
            if self.specific_features is not None:
                tmp_df = tmp_df[self.specific_features]
            # Append the dataframe to the list.
            tmp_df = tmp_df.dropna()
            df_list.append(tmp_df)
            # Create indices.
            df_size = len(tmp_df)
            if df_size >= min_required_len:
                n_windows = (df_size - min_required_len) // self.step + 1
                first_idx = start_idx + previous_days
                starts = range(first_idx, first_idx + n_windows * self.step, self.step)
                indices.extend(starts)
            start_idx += df_size
        # Concatenate all the dataframes.
        df = pd.concat(df_list, axis=0)
        return df, indices

    def close_dataset(self):
        """
        Function to close the dataset file.
        """
        self.dataset.close()
