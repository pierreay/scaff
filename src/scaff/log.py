"""Logging facilities."""

# Core modules.
import logging
import os

# External modules.
import colorlog

# * Objects

# Logger and its handler used accross all modules.
LOGGER = None
HANDLER_INF = None
HANDLER_DBG = None

# * Constants

# Default logging level.
LEVEL_DEFAULT = "INFO"
# Packane name.
PKG_NAME="scaff"
# Formatters depending on level.
FORMAT_INF = "%(log_color)s[{}] [%(levelname)s] %(message)s".format(PKG_NAME)
FORMAT_DBG = "%(log_color)s[{}] [%(asctime)s] [%(process)d] [%(threadName)s] [%(module)s] [%(levelname)s] %(message)s".format(PKG_NAME)

# * Functions

def _set_level(level):
    """Set logging level.

    :param level: String or logging level object.

    """
    LOGGER.removeHandler(HANDLER_INF)
    LOGGER.removeHandler(HANDLER_DBG)
    if level == "DEBUG":
        LOGGER.addHandler(HANDLER_DBG)
    else:
        LOGGER.addHandler(HANDLER_INF)
    LOGGER.setLevel(level)

def _disable():
    """Disable logging messages."""
    set_level(logging.CRITICAL + 1)

def init(level=LEVEL_DEFAULT):
    """Initialize the logging system."""
    global LOGGER, HANDLER_INF, HANDLER_DBG
    if LOGGER is None:
        LOGGER = colorlog.getLogger(__name__)
        HANDLER_INF = colorlog.StreamHandler()
        HANDLER_DBG = colorlog.StreamHandler()
        HANDLER_INF.setFormatter(colorlog.ColoredFormatter(FORMAT_INF))
        HANDLER_DBG.setFormatter(colorlog.ColoredFormatter(FORMAT_DBG))
        LOGGER.propagate = False # We want a custom handler and don't want its
                                 # messages also going to the root handler.
        _set_level(level)
        LOGGER.debug("Logger initialized: {}".format(__name__))

def configure(enable=True, level=LEVEL_DEFAULT):
    """Re-configure the logging system after being initialized()."""
    _set_level(level)
    if enable is False:
        _disable()

# * Script

# Initialize the module at importation time.
init(LEVEL_DEFAULT)
