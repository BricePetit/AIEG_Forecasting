"""
This module contains functions to add features to the data. These features are based on the
timestamp and include:
- hour:         Hour of the day and cyclic encoding.
- day:          Day of the month.
- month:        Month of the year and cyclic encoding.
- year:         Year.
- dayofweek:    Day of the week (0 = Monday, 6 = Sunday) and cyclic encoding.
- weekofyear:   Week of the year.
- weekend:      Weekend (1) or not (0).
- season:       Season of the year (0 = Winter, 1 = Spring, 2 = Summer, 3 = Autumn) and cyclic
                encoding.
- holiday:      Holiday (1) or not (0).
- is_peak_hour: Peak hour (1) or not (0).
"""

__title__: str = "ts_features"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries


# Imports third party libraries
import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    EasterMonday,
    Easter,
    Day,
)

# Imports from src


# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- FUNCTIONS ----------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


def add_ts_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Function to add features to the data. Based on the timestamp, the function will add the
    following features:
    - hour:         Hour of the day and cyclic encoding.
    - day:          Day of the month.
    - month:        Month of the year and cyclic encoding.
    - year:         Year.
    - dayofweek:    Day of the week (0 = Monday, 6 = Sunday) and cyclic encoding.
    - weekofyear:   Week of the year.
    - weekend:      Weekend (1) or not (0).
    - season:       Season of the year (0 = Winter, 1 = Spring, 2 = Summer, 3 = Autumn) and
                    cyclic encoding.
    - holiday:      Holiday (1) or not (0).

    :param data:    DataFrame containing the data.

    :return:        DataFrame with added features.
    """
    class BelgiumHolidays(AbstractHolidayCalendar):
        """
        Class to define the Belgium holidays.
        """
        rules = [
            Holiday('New Year', month=1, day=1),
            EasterMonday,
            Holiday('Labour Day', month=5, day=1),
            Holiday('Ascension', month=1, day=1, offset=[Easter(), Day(39)]),
            Holiday('Pentecost Monday', month=1, day=1, offset=[Easter(), Day(50)]),
            Holiday('National Day', month=7, day=21),
            Holiday('Assumption', month=8, day=15),
            Holiday('All Saints', month=11, day=1),
            Holiday('Armistice', month=11, day=11),
            Holiday('Christmas', month=12, day=25)
        ]
    df = data.copy()
    if isinstance(df.index, pd.MultiIndex):
        ts_index = pd.to_datetime(df.index.get_level_values('ts'))
    else:
        ts_index = pd.to_datetime(df.index)
    calendar = BelgiumHolidays()
    holidays = calendar.holidays(start=ts_index.min(), end=ts_index.max())
    df['hour'] = ts_index.hour
    # Cyclic encoding for the hour of the day.
    df['hour_sin'] = np.sin(2 * np.pi * ts_index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * ts_index.hour / 24)
    df['dayofweek'] = ts_index.dayofweek
    # Cyclic encoding for the day of the week.
    df['dayofweek_sin'] = np.sin(2 * np.pi * ts_index.dayofweek / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * ts_index.dayofweek / 7)
    df['month'] = ts_index.month
    # Cyclic encoding for the month of the year.
    df['month_sin'] = np.sin(2 * np.pi * ts_index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * ts_index.month / 12)
    df['day'] = ts_index.day
    df['year'] = ts_index.year
    df['weekofyear'] = ts_index.isocalendar().week
    df['weekend'] = df['dayofweek'].apply(lambda x: 1 if x > 4 else 0)
    df['season'] = df['month'].apply(
        lambda x:
        0 if x in [12, 1, 2] else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3
    )
    # Cyclic encoding for the season of the year.
    df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4)
    df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4)
    df['holiday'] = ts_index.isin(holidays).astype(int)
    df['is_peak_hour'] = (
        (df['hour'] >= 7) & (df['hour'] <= 22) & (df['weekend'] == 0) & (df['holiday'] == 0)
    ).astype(int)
    return df
