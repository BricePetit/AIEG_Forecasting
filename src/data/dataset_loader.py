"""
Module to load the complete data. The function will load the data from the HDF5 file, add the
timestamp features and the statistics features.
"""

__title__: str = "dataset_loader"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports from standard library
from typing import List

# Imports from third party libraries
import pandas as pd

# Imports from src
from .weather_loader import get_weather_data
from configs.config_loader import ConfigLoader
from features import add_ts_features, add_stats_features

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- Globals ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

config_loader = ConfigLoader()
config = config_loader.load_global()

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ Functions ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #

def load_complete_data(
    site_sn: str,
    site_name: str,
    prediction_type: str,
    in_data: int,
    is_weather: bool = True,
    is_stats: bool = True,
    is_ts: bool = True,
    old_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Function to load the complete data. The function will load the data from the HDF5 file, add the
    timestamp features and the statistics features.

    :param site_sn:         The site SN.
    :param site_name:       The site name.
    :param prediction_type: The prediction type (ap, msT6, DR6, S00).
    :param in_data:         The number of input data.
    :param is_weather:      Use weather data (True) or not (False).
    :param is_stats:        Use statistics features (True) or not (False).
    :param is_ts:           Use timestamp features (True) or not (False).
    :param old_df:          DataFrame containing the data. If None, the data will
                            be loaded from the HDF5 file.

    :return:                The complete data with added features.
    """
    if old_df is not None:
        data = old_df.copy()
    else:
        dataset = pd.HDFStore(
            f"{config['paths']['paths']['processed_data_dir']}/aieg.h5", mode='r'
        )
        data = dataset[f'/aieg_{site_name}_{site_sn}/{prediction_type}'].set_index('ts')[['ap']]
        dataset.close()
    # Check if we want to use weather data.
    if 'q1' in data.columns:
        data.drop(columns=['q1'], inplace=True)
    if 'q4' in data.columns:
        data.drop(columns=['q4'], inplace=True)
    # Set the index to datetime.
    data.index = pd.to_datetime(data.index)
    data = data.asfreq('15min')
    if is_weather:
        # Get the weather data.
        weather = get_weather_data()
        # Align weather on the target index without hard failure when coverage is partial.
        weather = weather.reindex(data.index)
        if weather.isna().all(axis=1).all():
            raise ValueError(
                "Weather data has no overlap with target timestamps. "
                "Please refresh IRM data with a wider date range."
            )
        weather = weather.ffill().bfill()
        data = pd.concat([data, weather], axis=1)
    if is_ts:
        data = add_ts_features(data)
    if is_stats:
        data = add_stats_features(data, in_data)
    return data


def build_group_data(dataset: pd.HDFStore, sites: List[str]) -> pd.DataFrame:
    """
    Build one aggregated dataframe from grouped domain site keys.

    Each item in ``sites`` is expected to be a full key like:
    - aieg_CHAMAIEG_219792511/production
    - aieg_DEPOTVIR_212303594/production
    - aieg_CHAAIEG2_250692408/consumption

    The function loads every existing source series, aligns on timestamp labels
    (not on row positions), and sums them into a single ``ap`` signal.

    :param dataset: HDF5 dataset.
    :param sites:   List of full site keys.

    :return:        Aggregated DataFrame with an ``ap`` column.
    """
    dataframes: list[pd.DataFrame] = []
    for site_key in sites:
        h5_key = f"/{site_key}"
        if h5_key not in dataset:
            continue
        temp_data = dataset[h5_key].set_index('ts')[['ap']]
        temp_data.index = pd.to_datetime(temp_data.index)
        # Safety for unordered inputs and potential duplicated timestamps.
        temp_data = temp_data.sort_index()
        if temp_data.index.has_duplicates:
            temp_data = temp_data.groupby(level=0, sort=True).sum()
        temp_data = temp_data.asfreq('15min').fillna(0)
        dataframes.append(temp_data)

    if not dataframes:
        raise ValueError(
            "No valid source series found in H5 for grouped site. "
            f"Expected one of: {sites}"
        )

    aggregated = dataframes[0].copy()
    for temp_data in dataframes[1:]:
        # Index-aware addition: timestamps are matched by labels, so disjoint
        # periods across sites are handled correctly.
        aggregated = aggregated.add(temp_data, fill_value=0)
    # Guarantee deterministic chronological order for downstream consumers.
    aggregated = aggregated.sort_index()
    return aggregated


def concat_production_sites(dataset: pd.HDFStore, sites_sn: List[str]) -> pd.DataFrame:
    """
    Function to concatenate the production data of the sites. In this case, we only need to
    concatenate the production data of the sites (CHAMAIEG et CHAMPAIEG).

    :param dataset:     HDF5 dataset.
    :param sites_sn:    List of serial numbers of the sites.

    :return:            Concatenated production data.
    """
    # Backward-compatible wrapper kept for legacy code paths.
    # This preserves the historical CHAMAIEG convention while delegating
    # aggregation logic to the generic grouped-data function.
    sites = []
    for sn in sites_sn:
        if sn in ['217158317', '217158406', '250692408']:
            sites.append(f"aieg_CHAMAIEG_{sn}/production")
        else:
            sites.append(f"aieg_CHAMAIEG_{sn}/consumption")
            sites.append(f"aieg_CHAMAIEG_{sn}/production")
    return build_group_data(dataset, sites)
