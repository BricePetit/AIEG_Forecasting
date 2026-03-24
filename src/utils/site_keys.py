"""
Utilities to normalize and manipulate AIEG site keys.
"""

__title__: str = "site_keys"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"


# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# Imports standard libraries

# Imports third party libraries

# Imports from src

# ----------------------------------------------------------------------------------------------- #
# -------------------------------------- GLOBAL VARIABLES --------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNCTIONS ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #


def normalize_site_name(site_name: str) -> str:
    """
    Normalize site name so it can be compared across raw files, DB and config keys.

    :param site_name: Raw site name.

    :return:          Normalized site name.
    """
    return str(site_name).replace('.', '_').replace('-', '_').replace('~', '_').replace('+', '')


def build_site_key(site_name: str, sn: str, production: bool) -> str:
    """
    Build the H5 key used in the project.

    :param site_name:   Site name.
    :param sn:          Serial number.
    :param production:  Whether the site is production.

    :return:            H5 site key.
    """
    normalized_site = normalize_site_name(site_name)
    return f"/aieg_{normalized_site}_{sn}/{'production' if production else 'consumption'}"


def parse_domain_site_key(site_key: str) -> tuple[str, str]:
    """
    Parse a domain key into a (site_name, sn) tuple.

    :param site_key: Domain key like /aieg_SITE_SN/consumption.

    :return:         Tuple (normalized_site_name, sn).
    """
    normalized_key = str(site_key).lstrip('/')
    prefix_part = normalized_key.split('/')[0]
    without_prefix = prefix_part[len('aieg_'):] if prefix_part.startswith('aieg_') else prefix_part
    site_name, sn = without_prefix.rsplit('_', 1)
    return normalize_site_name(site_name), str(sn)
