"""I/O module.

The main purpose of io.py is to provides facilities to handle different kind of
data on disk (load, save, count).

"""

# * Importation

# Standard import.
import sys
from os import path

# External import.
import numpy as np

# Internal import.
from scaff import logger as l
from scaff import config

# * Variables

# * Classes

class IOConf(config.ModuleConf):
    """Configuration for the io.py module."""

    # Path from which data should be loaded or stored.
    data_path = None
    # Pattern in which the first {} will be replaced by trace index.
    data_pattern = None

    def __init__(self, appconf = None, data_path = None, data_pattern = None):
        super().__init__(__name__, appconf)
        self.data_path = data_path
        self.data_pattern = data_pattern
        if appconf is not None:
            self.load(appconf)

    def load(self, appconf):
        self.data_path = self.get_dict(appconf)["data_path"]
        self.data_path = None if self.data_path == "" else self.data_path
        self.data_pattern = self.get_dict(appconf)["data_pattern"]

    def check(self):
        assert data_path is not None
        assert data_pattern is not None

class IO():
    # Configuration [IOConf].
    conf = None

    def __init__(self, conf):
        assert isinstance(conf, IOConf)
        self.conf = conf

    def load(self, i):
        """Load the data stored at given index.

        :param i: Index of desired data.

        """
        return np.load(path.join(self.conf.data_path, self.conf.data_pattern.format(i)))

    def count(self):
        """Count the number of data stored on-disk (last index + 1)."""
        for i in range(0, sys.maxsize):
            if path.exists(path.join(self.conf.data_path, self.conf.data_pattern.format(i))):
                continue
            else:
                return i
            assert(i < 1e6), "Infinite loop?"

if __name__ == "__main__":
    io = IO(IOConf(config.APPCONF))
    io.conf.data_path = "/tmp/data_path"
    import ipdb; ipdb.set_trace()
