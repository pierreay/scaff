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
APPCONF = None

# * Classes

class AppConf():
    """Application configuration."""

    # Path to the configuration file [str].
    path = None
    # TOML structure of the configuration file.
    toml = None

    def __init__(self, path):
        # Set the application-wide global variable to the last instanciated configuration.
        global APPCONF
        APPCONF = self
        # Get parameters.
        self.path = path
        # Load the configuration file.
        with open(self.path, "rb") as f:
            try:
                self.toml = tomllib.load(f)
            except Exception as e:
                l.LOGGER.error("Bad TOML configuration file format!")
                raise e

class ModuleConf():
    """Module configuration."""

    # Package name.
    _pkg_name = None
    # Module name.
    _mod_name = None

    def __init__(self, name, appconf = None):
        """Initialize a module configuration.

        :param name: Should be the __name__ of the respective module.
        :param appconf: If set to an AppConf, run self.load(appconf).

        Usage in child classes:
        super().__init__(__name__, appconf)

        """
        assert appconf is None or isinstance(appconf, AppConf)
        self._pkg_name = name.split(".")[0]
        self._mod_name = name.split(".")[1]
        if appconf is not None:
            self.load(appconf)
    
    def get_dict(self, appconf):
        """Get module configuration from an AppConf.

        Allows to automatically bypass the first 2 levels of the dictionnary to
        access the pattern: '[package.module]'

        :returns: The dictionnary level for the module.

        """
        assert isinstance(appconf, AppConf)
        return appconf.toml[self._pkg_name][self._mod_name]

    def load(self, appconf):
        """Load the module configuration from an AppConf."""
        assert False, "This method should be overridden!"

    def check(self):
        """Check that the module configuration is correct."""
        assert False, "This method should be overridden!"

# * Functions

def loaded():
    return APPCONF is not None

def get():
    assert APPCONF is not None, "Configuration has not been loaded!"
    return APPCONF.toml
