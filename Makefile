# * Variables

# ** Project configuration

# Package name.
PKG_NAME=scaff

# Package directory.
# NOTE: This corresponds to the so-called "src" layout of Python documentation.
PKG_DIR=src

# * Recipes

# ** Installing/Uninstalling to user using Pip

# Install the project user-wide using dynamic installation method (symlinking files).
install:
	pip install --break-system-packages --user --editable .

# Uninstall the project user-wide.
uninstall: clean
	pip uninstall --break-system-packages $(PKG_NAME)

# ** Cleaning

clean: 
	rm -rf build
	rm -rf $(PKG_DIR)/$(PKG_NAME).egg-info
	rm -rf $(PKG_DIR)/$(PKG_NAME)/__pycache__
