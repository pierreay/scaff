"""I/O module.

The main purpose of io.py is to provides facilities to load from / save to
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

class IOConf(config.ModuleConf):
    """Configuration for the io.py module."""

    # Path from which data should be loaded or stored.
    data_path = None
    # Pattern in which the first {} will be replaced by trace index.
    data_pattern = None

    def __init__(self, data_path = None, data_pattern = None):
        super().__init__(__name__)
        self.data_path = data_path
        self.data_pattern = data_pattern

    def load(self, appconf):
        self.data_path = self.get_dict(appconf)["data_path"]
        self.data_path = None if self.data_path == "" else self.data_path
        self.data_pattern = self.get_dict(appconf)["data_pattern"]

    def check(self):
        assert data_path is not None
        assert data_pattern is not None

class IO():
    """Load data from disk."""

    # Configuration [IOConf].
    conf = None

    def __init__(self, conf):
        assert isinstance(conf, IOConf)
        self.conf = conf

if __name__ == "__main__":
    # Configure the IO from configuration file.
    iocnf, savercnf = io.IOConf(), io.IOConf()
    if config.loaded() is True:
        iocnf.load(config.APPCONF)
        savercnf.load(config.APPCONF)
    # Configure the IO from arguments.
    iocnf.data_path = load_path
    savercnf.data_path = load_path
    # Create the IO and the Saver.
    io = io.IO(iocnf)
    saver = io.IO(savercnf)    
