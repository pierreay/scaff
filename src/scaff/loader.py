"""Loader module.

The main purpose of loader.py is to provides facilities to load from / save to
disk different kind of data.

"""

# * Importation

# Standard import.

# External import.
import numpy as np

# Internal import.
from scaff import logger as l
from scaff import config

# * Variables

# * Classes

class LoaderConf():
    """Configuration for the loader.py module."""

    # Path from which data should be loaded or stored.
    data_path = None
    # Pattern in which the first {} will be replaced by trace index.
    data_pattern = None

    def __init__(self, data_path = None, data_pattern = None):
        self.data_path = data_path
        self.data_pattern = data_pattern

    @staticmethod
    def _get_from_toml(toml):
        """Allows to bypass the first 2-level dictionnary access.

        :returns: The value stored in the TOML file, or None if it was an empty
        string.

        """
        assert isinstance(toml, dict)
        return toml[__name__.split(".")[0]][__name__.split(".")[1]]

    def load(self, appconf):
        """Load the configuration from an AppConf."""
        assert isinstance(appconf, config.AppConf)
        self.data_path = LoaderConf._get_from_toml(appconf.toml)["data_path"]
        self.data_path = None if self.data_path == "" else self.data_path
        self.data_pattern = LoaderConf._get_from_toml(appconf.toml)["data_pattern"]

class Loader():
    """Load data from disk."""

    # Configuration [LoaderConf].
    conf = None

    def __init__(self, conf):
        assert isinstance(conf, LoaderConf)
        self.conf = conf
