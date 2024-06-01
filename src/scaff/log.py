"""Logging facilities."""

# Core modules.
import logging
import os

# External modules.
import colorlog

# * Objects

# Logger used accross all modules.
LOGGER = None

# * Constants

# Default logging level.
LOGGER_DEFAULT_LEVEL = "DEBUG"

# * Functions

def log_n_exit(str, code, e=None, traceback=True):
    """Log a critical error and exit.

    :param str: Log message.
    :param code: Exit code.
    :param e: Exception object.

    """
    assert LOGGER, "No initialized logger"
    if e:
        LOGGER.critical(e, exc_info=traceback, stack_info=traceback)
    LOGGER.critical(str)
    exit(code)

def set_level(level):
    """Set the logger of the logging system to LEVEL (string or logging
    levels).

    """
    LOGGER.setLevel(level)

def disable():
    """Disable the logging messages."""
    set_level(logging.CRITICAL + 1)

def init(level):
    """Initialize the logging system.

    Initialize the stream type (stderr) and the logging format depending on the
    later in the global LOGGER variable, alert program start.

    """
    global LOGGER
    if LOGGER is None:
        handler = colorlog.StreamHandler()
        format = "%(log_color)s{}%(levelname)-5s - %(message)s".format("[%(asctime)s] [%(process)d] [%(threadName)s] [%(module)s] " if level == "DEBUG" else "")
        formatter = colorlog.ColoredFormatter(format)
        LOGGER = colorlog.getLogger(__name__)
        handler.setFormatter(formatter)
        LOGGER.propagate = False # We want a custom handler and don't want its
                                 # messages also going to the root handler.
        LOGGER.setLevel(level)
        LOGGER.addHandler(handler)

def configure(enable=True, level=LOGGER_DEFAULT_LEVEL):
    """Configure the logging system after init()."""
    set_level(level)
    if enable is False:
        disable()

# Initialized at import time.
init(LOGGER_DEFAULT_LEVEL)
