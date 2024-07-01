#!/usr/bin/python3

# * Importation

# Standard import.
from os import path
from functools import partial

# External import.
import click
import matplotlib.pyplot as plt

# Internal import.
from scaff import logger as l
from scaff import config
from scaff import helpers
from scaff import legacy
from scaff import processors
from scaff import io

# * Command-line interface

@click.group(context_settings={'show_default': True})
@click.option("--config", "config_path", type=click.Path(), default="", help="Path of a TOML configuration file.")
@click.option("--log/--no-log", default=True, help="Enable or disable logging.")
@click.option("--loglevel", default="INFO", help="Set the logging level.")
def cli(config_path, log, loglevel):
    """Side-channel analysis tool."""
    l.configure(log, loglevel)
    if config_path != "":
        l.LOGGER.info("Load configuration file: {}".format(config_path))
        if path.exists(config_path):
            try:
                config.AppConf(config_path)
            except Exception as e:
                l.LOGGER.error("Configuration file cannot be loaded: {}".format(path.abspath(config_path)))
                raise e
        else:
            l.LOGGER.warn("Configuration file does not exists: {}".format(config_path))


@cli.command()
@click.argument("load_path", type=click.Path())
@click.argument("save_path", type=click.Path())
def copy(load_path, save_path):
    """Copy traces of a dataset from one location to another."""
    processor = processors.Processor(
        processors.ProcessingCopy(load_path=load_path, save_path=save_path),
        helpers.ExecOnce()
    ).start()

@cli.command()
@click.argument("load_path", type=click.Path())
@click.argument("save_path", type=click.Path())
@click.option("--skip/--no-skip", "skip_flag", default=False, help="If set to True, disable any skip / filter configured for the extraction processing.")
def extract(load_path, save_path, skip_flag):
    """Extract traces (amplitude and phase rotation) from signals (IQs)."""
    # Sanity-check.
    load_path = path.abspath(load_path)
    save_path = path.abspath(save_path)
    if not path.exists(load_path):
        l.LOGGER.critical("Directory does not exists: {}".format(load_path))
        exit(1)
    if not path.exists(save_path):
        l.LOGGER.critical("Directory does not exists: {}".format(save_path))
        exit(1)
    # Loader.
    loader = io.IO(io.IOConf(config.APPCONF))
    loader.conf.data_path = load_path
    # Processing configuration.
    processing = processors.ProcessingExtract(load_path=load_path, save_path=save_path)
    processing.config = legacy.ExtractConf().load(config.APPCONF)
    # Configure according to CLI flags.
    if skip_flag is True:
        processing.config.num_traces_per_point_min = 0
        processing.config.min_correlation = 0
    # Processing execution.
    try:
        processor = processors.Processor(processing, helpers.ExecOnce(), stop=partial(loader.count)).start()
    except Exception as e:
        l.LOGGER.critical("Error during extraction processing: {}".format(e))
        raise e

@cli.command()
@click.argument("target_path", type=str)
@click.argument("comp", type=str)
@click.option("--base", type=int, default=0, show_default=True,
              help="Base start index.")
@click.option("--offset", type=int, default=0, show_default=True,
              help="Added to base index to obtain end index.")
@click.option("--cumulative/--no-cumulative", type=bool, default=False, show_default=True,
              help="Show a cumulative plot or a single plot per traces.")
def show(target_path, comp, base, offset, cumulative):
    # Sanity-check.
    target_path = path.abspath(target_path)
    if not path.exists(target_path):
        l.LOGGER.critical("Directory does not exists: {}".format(target_path))
        exit(1)
    # Loader.
    loader = io.IO(io.IOConf(data_path=target_path, data_pattern="{{}}_{}.npy".format(comp)))
    # Plotting.
    for i in list(range(base, base + offset)):
        plt.plot(loader.load(i))
        if cumulative is False:
            plt.show()
    if cumulative is True:
        plt.show()    

# ** CHES20

@cli.command()
# Old-general:
@click.option("--data-path", type=click.Path(exists=True, file_okay=False),
              help="Directory where the traces are stored.")
@click.option("--num-traces", default=0, show_default=True,
              help="The number of traces to use, or 0 to use the maximum available.")
@click.option("--start-point", default=0, show_default=True,
              help="Index of the first point in each trace to use.")
@click.option("--end-point", default=0, show_default=True,
              help="Index of the last point in each trace to use, or 0 for the maximum.")
@click.option("--plot/--no-plot", default=False, show_default=True,
              help="Visualize relevant data (use only with a small number of traces.")
@click.option("--save-images/--no-save-images", default=False, show_default=True,
              help="Save images (when implemented).")
@click.option("--bruteforce/--no-bruteforce", default=False, show_default=True,
              help="Attempt to fix a few wrong key bits with informed exhaustive search.")
@click.option("--norm/--no-norm", default=False, show_default=True,
              help="Normalize each trace individually: x = (x-avg(x))/std(x).")
@click.option("--norm2/--no-norm2", default=False, show_default=True,
              help="Normalize each trace set: traces = (traces-avg(traces))/std(traces).")
@click.option("--comp", default="amp",
              help="Choose component to load (e.g., amplitude is 'amp'")
# Old-specific:
@click.option("--variable", default="hw_sbox_out", show_default=True,
              help="Variable to attack (hw_sbox_out, hw_p_xor_k, sbox_out, p_xor_k, p, hd)")
@click.option("--lr-type", default=None, show_default=True,
              help="Variable to attack (n_p_xor_k, n_sbox_out)")
@click.option("--pois-algo", default="snr", show_default=True,
              help="Algo used to find pois (snr, soad, r, t)")
@click.option("--k-fold", default=10, show_default=True,
              help="k-fold cross validation.")
@click.option("--num-pois", default=1, show_default=True,
              help="Number of points of interest.")
@click.option("--poi-spacing", default=5, show_default=True,
              help="Minimum number of points between two points of interest.")
@click.option("--pois-dir", default="", type=click.Path(file_okay=False, writable=True),
              help="Reduce the trace using the POIS in this folder")
@click.option("--align/--no-align", default=False, show_default=True,
             help="Align the training traces before computing the profile.")
@click.option("--fs", default=0, type=float, show_default=True,
             help="Sampling rate used when aligning traces")
@click.argument("template_dir", type=click.Path(file_okay=False, writable=True))
def profile(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp,
        variable, lr_type, pois_algo, k_fold, num_pois, poi_spacing, pois_dir, align, fs, template_dir):
    """Profiled template creation.

    Build a template using a chosen technique. The template directory is where
    we store multiple files comprising the template; beware that existing files
    will be overwritten!

    """
    legacy.profile(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp, variable, lr_type, pois_algo, k_fold, num_pois, poi_spacing, pois_dir, align, fs, template_dir)

@cli.command()
# Old-general:
@click.option("--data-path", type=click.Path(exists=True, file_okay=False),
              help="Directory where the traces are stored.")
@click.option("--num-traces", default=0, show_default=True,
              help="The number of traces to use, or 0 to use the maximum available.")
@click.option("--start-point", default=0, show_default=True,
              help="Index of the first point in each trace to use.")
@click.option("--end-point", default=0, show_default=True,
              help="Index of the last point in each trace to use, or 0 for the maximum.")
@click.option("--plot/--no-plot", default=False, show_default=True,
              help="Visualize relevant data (use only with a small number of traces.")
@click.option("--save-images/--no-save-images", default=False, show_default=True,
              help="Save images (when implemented).")
@click.option("--bruteforce/--no-bruteforce", default=False, show_default=True,
              help="Attempt to fix a few wrong key bits with informed exhaustive search.")
@click.option("--norm/--no-norm", default=False, show_default=True,
              help="Normalize each trace individually: x = (x-avg(x))/std(x).")
@click.option("--norm2/--no-norm2", default=False, show_default=True,
              help="Normalize each trace set: traces = (traces-avg(traces))/std(traces).")
@click.option("--comp", default="amp",
              help="Choose component to load (e.g., amplitude is 'amp'")
# Old-specific:
@click.option("--variable", default="hw_sbox_out", show_default=True,
              help="Variable to attack (hw_sbox_out, hw_p_xor_k, p_xor_k)")
@click.option("--pois-algo", default="", show_default=True,
              help="Algo used to find pois (snr, soad)")
@click.option("--num-pois", default=1, show_default=True,
              help="Number of points of interest.")
@click.option("--poi-spacing", default=5, show_default=True,
              help="Minimum number of points between two points of interest.")
@click.argument("template_dir", type=click.Path(file_okay=False, writable=True))
@click.option("--attack-algo", default="pcc", show_default=True,
              help="Algo used to rank the guesses (pdf, pcc)")
@click.option("--k-fold", default=2, show_default=True,
              help="k-fold cross validation.")
@click.option("--average-bytes/--no-average-bytes", default=False, show_default=True,
              help="Average the profile of the 16 bytes into one, for now it works only with pcc.")
@click.option("--pooled-cov/--no-pooled-cov", default=False, show_default=True,
              help="Pooled covariance for template attacks.")
@click.option("--window", default=0, show_default=True,
              help="Average poi-window to poi+window samples.")
@click.option("--align/--no-align", default=False, show_default=True,
             help="Align the attack traces with the profile before to attack.")
@click.option("--fs", default=0, type=float, show_default=True,
             help="Sampling rate used when aligning traces")
def attack(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp,
           variable, pois_algo, num_pois, poi_spacing, template_dir, attack_algo, k_fold, average_bytes, pooled_cov, window, align, fs):
    """
    Template attack or profiled correlation attack.

    The template directory is where we store multiple files comprising the
    template.
    """
    legacy.attack(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp,
           variable, pois_algo, num_pois, poi_spacing, template_dir, attack_algo, k_fold, average_bytes, pooled_cov, window, align, fs)

@cli.command()
# Old-general:
@click.option("--data-path", type=click.Path(exists=True, file_okay=False),
              help="Directory where the traces are stored.")
@click.option("--num-traces", default=0, show_default=True,
              help="The number of traces to use, or 0 to use the maximum available.")
@click.option("--start-point", default=0, show_default=True,
              help="Index of the first point in each trace to use.")
@click.option("--end-point", default=0, show_default=True,
              help="Index of the last point in each trace to use, or 0 for the maximum.")
@click.option("--plot/--no-plot", default=False, show_default=True,
              help="Visualize relevant data (use only with a small number of traces.")
@click.option("--save-images/--no-save-images", default=False, show_default=True,
              help="Save images (when implemented).")
@click.option("--bruteforce/--no-bruteforce", default=False, show_default=True,
              help="Attempt to fix a few wrong key bits with informed exhaustive search.")
@click.option("--norm/--no-norm", default=False, show_default=True,
              help="Normalize each trace individually: x = (x-avg(x))/std(x).")
@click.option("--norm2/--no-norm2", default=False, show_default=True,
              help="Normalize each trace set: traces = (traces-avg(traces))/std(traces).")
@click.option("--comp", default="amp",
              help="Choose component to load (e.g., amplitude is 'amp'")
# Old-specific:
@click.option("--variable", default="hw_sbox_out", show_default=True,
              help="Variable to attack (hw_sbox_out, hw_p_xor_k, p_xor_k)")
@click.option("--pois-algo", default="", show_default=True,
              help="Algo used to find pois (snr, soad)")
@click.option("--num-pois", default=1, show_default=True,
              help="Number of points of interest.")
@click.option("--poi-spacing", default=5, show_default=True,
              help="Minimum number of points between two points of interest.")
@click.argument("template_dir", type=click.Path(file_okay=False, writable=True))
@click.option("--attack-algo", default="pcc", show_default=True,
              help="Algo used to rank the guesses (pdf, pcc)")
@click.option("--k-fold", default=2, show_default=True,
              help="k-fold cross validation.")
@click.option("--average-bytes/--no-average-bytes", default=False, show_default=True,
              help="Average the profile of the 16 bytes into one, for now it works only with pcc.")
@click.option("--pooled-cov/--no-pooled-cov", default=False, show_default=True,
              help="Pooled covariance for template attacks.")
@click.option("--window", default=0, show_default=True,
              help="Average poi-window to poi+window samples.")
@click.option("--align/--no-align", default=False, show_default=True,
             help="Align the attack traces with the profile before to attack.")
@click.option("--fs", default=0, type=float, show_default=True,
             help="Sampling rate used when aligning traces.")
@click.option("--corr-method", default="add", type=str, show_default=True,
             help="Correlation recombination method [add | mul].")
def attack_recombined(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp,
        variable, pois_algo, num_pois, poi_spacing, template_dir, attack_algo, k_fold, average_bytes, pooled_cov, window, align, fs, corr_method):
    legacy.attack_recombined(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp, variable, pois_algo, num_pois, poi_spacing, template_dir, attack_algo, k_fold, average_bytes, pooled_cov, window, align, fs, corr_method)

# ** CCS18

@cli.command()
# Old-general:
@click.option("--data-path", type=click.Path(exists=True, file_okay=False),
              help="Directory where the traces are stored.")
@click.option("--num-traces", default=0, show_default=True,
              help="The number of traces to use, or 0 to use the maximum available.")
@click.option("--start-point", default=0, show_default=True,
              help="Index of the first point in each trace to use.")
@click.option("--end-point", default=0, show_default=True,
              help="Index of the last point in each trace to use, or 0 for the maximum.")
@click.option("--plot/--no-plot", default=False, show_default=True,
              help="Visualize relevant data (use only with a small number of traces.")
@click.option("--save-images/--no-save-images", default=False, show_default=True,
              help="Save images (when implemented).")
@click.option("--bruteforce/--no-bruteforce", default=False, show_default=True,
              help="Attempt to fix a few wrong key bits with informed exhaustive search.")
@click.option("--norm/--no-norm", default=False, show_default=True,
              help="Normalize each trace individually: x = (x-avg(x))/std(x).")
@click.option("--norm2/--no-norm2", default=False, show_default=True,
              help="Normalize each trace set: traces = (traces-avg(traces))/std(traces).")
@click.option("--comp", default="amp",
              help="Choose component to load (e.g., amplitude is 'amp'")
# Old-specific:
@click.argument("template_dir", type=click.Path(file_okay=False, writable=True))
@click.option("--num-pois", default=2, show_default=True,
              help="Number of points of interest.")
@click.option("--poi-spacing", default=5, show_default=True,
              help="Minimum number of points between two points of interest.")
def tra_create(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp,
        template_dir, num_pois, poi_spacing):
    """
    Template Radio Analysis; create a template.

    The data set should have a considerable size in order to allow for the
    construction of an accurate model. In general, the more data is used for
    template creation the less is needed to apply the template.

    The template directory is where we store multiple files comprising the
    template; beware that existing files will be overwritten!
    """
    legacy.tra_create(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp, template_dir, num_pois, poi_spacing)

@cli.command()
# Old-general:
@click.option("--data-path", type=click.Path(exists=True, file_okay=False),
              help="Directory where the traces are stored.")
@click.option("--num-traces", default=0, show_default=True,
              help="The number of traces to use, or 0 to use the maximum available.")
@click.option("--start-point", default=0, show_default=True,
              help="Index of the first point in each trace to use.")
@click.option("--end-point", default=0, show_default=True,
              help="Index of the last point in each trace to use, or 0 for the maximum.")
@click.option("--plot/--no-plot", default=False, show_default=True,
              help="Visualize relevant data (use only with a small number of traces.")
@click.option("--save-images/--no-save-images", default=False, show_default=True,
              help="Save images (when implemented).")
@click.option("--bruteforce/--no-bruteforce", default=False, show_default=True,
              help="Attempt to fix a few wrong key bits with informed exhaustive search.")
@click.option("--norm/--no-norm", default=False, show_default=True,
              help="Normalize each trace individually: x = (x-avg(x))/std(x).")
@click.option("--norm2/--no-norm2", default=False, show_default=True,
              help="Normalize each trace set: traces = (traces-avg(traces))/std(traces).")
@click.option("--comp", default="amp",
              help="Choose component to load (e.g., amplitude is 'amp'")
# Old-specific:
@click.argument("template_dir", type=click.Path(exists=True, file_okay=False))
def tra_attack(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp,
        template_dir):
    """
    Template Radio Analysis; apply a template.

    Use the template to attack the key in a new data set (i.e. different from
    the one used to create the template). The template directory must be the
    location of a previously created template with compatible settings (e.g.
    same trace length).
    """
    legacy.tra_attack(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp, template_dir)

@cli.command()
# Old-general:
@click.option("--data-path", type=click.Path(exists=True, file_okay=False),
              help="Directory where the traces are stored.")
@click.option("--num-traces", default=0, show_default=True,
              help="The number of traces to use, or 0 to use the maximum available.")
@click.option("--start-point", default=0, show_default=True,
              help="Index of the first point in each trace to use.")
@click.option("--end-point", default=0, show_default=True,
              help="Index of the last point in each trace to use, or 0 for the maximum.")
@click.option("--plot/--no-plot", default=False, show_default=True,
              help="Visualize relevant data (use only with a small number of traces.")
@click.option("--save-images/--no-save-images", default=False, show_default=True,
              help="Save images (when implemented).")
@click.option("--bruteforce/--no-bruteforce", default=False, show_default=True,
              help="Attempt to fix a few wrong key bits with informed exhaustive search.")
@click.option("--norm/--no-norm", default=False, show_default=True,
              help="Normalize each trace individually: x = (x-avg(x))/std(x).")
@click.option("--norm2/--no-norm2", default=False, show_default=True,
              help="Normalize each trace set: traces = (traces-avg(traces))/std(traces).")
@click.option("--comp", default="amp",
              help="Choose component to load (e.g., amplitude is 'amp'")
def cra(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp):
    """
    Correlation Radio Analysis.

    Run a "standard" correlation attack against a data set, trying to recover
    the key used for the observed AES operations. The attack works by
    correlating the amplitude-modulated signal of the screaming channel with the
    power consumption of the SubBytes step in the first round of AES, using a
    Hamming-weight model.
    """
    legacy.cra(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp)

if __name__ == "__main__":
    cli()
