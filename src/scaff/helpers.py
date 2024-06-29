"""Helpers functions wrapping classes or doing small computations."""

# * Importation

# Standard import.
from os import path

# External import.
import numpy as np

# Internal import.
from scaff import logger as l

# * Functions for command-line interface

def hello_world():
    l.LOGGER.info("Hello world!")
