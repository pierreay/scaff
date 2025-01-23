#!/bin/bash

# Install the Histogram Enumeration Library (HEL) used for bruteforcing the key.

# Install the dependencies.
sudo apt-get install -yq libntl-dev libgmp-dev
# Clone the library.
git clone https://github.com/pierreay/python_hel.git
# Compile and install the C++ library using Intel's AES-NI instructions.
cd python_hel/hel_wrapper
make AES_TYPE=aes_ni
sudo make install
sudo ldconfig
# Install the Python interface.
cd ../python_hel
sudo python3 setup.py install
