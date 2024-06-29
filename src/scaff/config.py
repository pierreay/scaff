"""Global configuration."""

# * Importation

# Standard import.

# Compatibility import.
try:
    import tomllib
# NOTE: For Python <= 3.11:
except ModuleNotFoundError as e:
    import tomli as tomllib

# External import.

# Internal import.
from scaff import logger as l

# * Global variables

# Reference to the AppConf object used to configure the application.
CONFIG = None

# * Classes

class AppConf():
    """Application configuration."""

    # Path to the configuration file [str].
    path = None
    # TOML structure of the configuration file.
    toml = None

    def __init__(self, path):
        # Set the application-wide global variable to the last instanciated configuration.
        global CONFIG
        CONFIG = self
        # Get parameters.
        self.path = path
        # Load the configuration file.
        with open(self.path, "rb") as f:
            try:
                self.toml = tomllib.load(f)
            except Exception as e:
                l.LOGGER.error("Bad TOML configuration file format!")
                raise e

# * Functions

def loaded():
    return CONFIG is not None

def get():
    assert CONFIG is not None, "Configuration has not been loaded!"
    return CONFIG.toml
