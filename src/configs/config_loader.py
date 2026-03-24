"""
This module defines the `load_yaml` function, which is used to load a yaml file and return its
content as a dictionary.
"""

__title__: str = "config_loader"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports from standard library
from copy import deepcopy
from pathlib import Path
import yaml

# Imports from third party libraries

# Imports from src

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ Functions ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #

def deep_update(base: dict, updates: dict) -> dict:
    """
    Function to recursively update a dictionary with another dictionary. This is used to update the
    global configuration with the site-specific configuration.

    :param base:    Base dictionary to update.
    :param updates: Dictionary with the updates.

    :return: Updated dictionary.
    """
    for k, v in updates.items():
        if isinstance(v, dict) and k in base:
            base[k] = deep_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base


# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- Classes ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


class ConfigLoader:
    """
    Class to load the configuration files. The configuration files are organized in a hierarchical
    structure, with a global configuration file and site-specific configuration files. The global
    configuration file contains the default values for all the sites, while the site-specific
    configuration files contain the values that override the global configuration for each site.
    
    The class also handles the loading of the best parameters from Optuna, which are stored in a
    separate directory. The best parameters are loaded and merged with the global configuration,
    allowing for a seamless integration of the optimized parameters into the training pipeline.
    """
    def __init__(self, config_dir="src/configs"):
        """
        Class to load the configuration files.
        
        :param config_dir:  Directory where the configuration files are located.
        """
        self.config_dir = Path(config_dir)

    def load_yaml(self, path):
        """
        Method to load a yaml file and return its content as a dictionary.

        :param path:    Path to the yaml file.

        :return:        Content of the yaml file as a dictionary.
        """
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def load_global(self):
        """
        Method to load the global configuration files.

        :return:        Global configuration as a dictionary.
        """
        model = self.load_yaml(self.config_dir / "global/model.yaml")
        data = self.load_yaml(self.config_dir / "global/data.yaml")
        domain = self.load_yaml(self.config_dir / "global/domain.yaml")
        paths = self.load_yaml(self.config_dir / "global/paths.yaml")

        return {
            "model": model,
            "data": data,
            "domain": domain,
            "paths": paths,
        }

    def load_site_config(self, site_id: int):
        """
        Method to load the configuration for a specific site.

        :param site_id: ID of the site for which to load configuration.

        :return:        Configuration for the specified site as a dictionary.
        """
        global_config = self.load_global()

        site_path = self.config_dir / f"sites/site_{site_id}.yaml"
        best_params_path = self.config_dir / f"best_params/site_{site_id}.yaml"

        config = deepcopy(global_config)

        # Override site-specific
        if site_path.exists():
            site_config = self.load_yaml(site_path)
            config = deep_update(config, site_config)

        # Override best params (Optuna)
        if best_params_path.exists():
            best_params = self.load_yaml(best_params_path)
            config["model"]["best_params"] = best_params

        return config

    def load_features(self, site_id) -> dict | None:
        """
        Method to load the features for a specific site. The features are stored in a yaml file in
        the "features" directory, with the name "site_{site_id}.yaml". If the file does not exist,
        the method returns None.
        """
        path = self.config_dir / f"features/site_{site_id}.yaml"
        if path.exists():
            return self.load_yaml(path)
        return None
