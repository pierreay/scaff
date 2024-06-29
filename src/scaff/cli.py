#!/usr/bin/python3

# * Importation

# Standard import.

# External import.
import click

# Internal import.
from scaff import logger as l
from scaff import config

# * Command-line interface

@click.group(context_settings={'show_default': True})
@click.option("--config", "config_path", type=click.Path(), default="", help="Path of a TOML configuration file.")
@click.option("--log/--no-log", default=True, help="Enable or disable logging.")
@click.option("--loglevel", default="INFO", help="Set the logging level.")
def cli(config_path, log, loglevel):
    """Side-channel analysis tool."""
    l.configure(log, loglevel)
    if config_path != "":
        l.LOGGER.info("Configuration file loaded: {}".format(path.abspath(config_path)))
        if path.exists(config_path):
            try:
                config.AppConf(config_path)
            except Exception as e:
                l.LOGGER.error("Configuration file cannot be loaded: {}".format(path.abspath(config_path)))
                raise e
        else:
            l.LOGGER.warn("Configuration file does not exists: {}".format(path.abspath(config_path)))

@cli.command()
def hello_world():
    helpers.hello_world()

if __name__ == "__main__":
    cli()
