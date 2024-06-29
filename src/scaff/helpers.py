"""Helpers functions wrapping classes or doing small computations."""

# * Importation

# Standard import.
from os import path

# External import.
import numpy as np

# Internal import.
from scaff import logger as l

# * Functions for command-line interface

# * Miscellaneous classes

class ExecOnce():
    """Execute someething only once by switching off the flag."""

    # Status flag.
    state = None

    def __init__(self, default=True):
        self.state = default

    def pop(self):
        val = self.get()
        self.off()
        return val

    def get(self):
        return self.state is True

    def off(self):
        self.state = False
