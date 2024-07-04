"""Multi-processing module.

The main purpose of processors.py is to provides fast processing facilities
when working with large datasets (e.g., > 100 GB).

"""

# * Importation

# Standard import.
import signal
import os
from os import path
from multiprocessing import Process, Queue
from functools import partial

# External import.
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Internal import.
from scaff import logger as l
from scaff import helpers
from scaff import legacy
from scaff import dsp
from scaff import config

# * Variables

# Current running processor self-reference for signal handler.
PROCESSOR = None

# * Classes

class ProcessingInterface():
    """Processing to be applied on one element by a Processor."""

    #  Processing title.
    title = None
    # Path for processing load.
    load_path = None
    # Loaded data.
    load_data = None
    # Path for processing save.
    save_path = None
    # Saved data.
    save_data = None
    # Processing index.
    idx = None
    # Set to True if Processing failed.
    failed = None
    
    def __init__(self, load_path, save_path):
        # Default values.
        self.failed = False
        # Get parameters.
        self.load_path = load_path
        self.save_path = save_path

    def load(self, i):
        """Load data to process from disk for given index inside self.load_data.

        :param i: Index of processing.

        """
        pass

    def exec(self, plot_flag):
        """Run the processing on loaded data and save the result inside self.save_data.

        :param plot_flag: Flag indicating to plot the result.
        :returns: True if processing succeed, False if processing failed.

        """
        return self.failed == False

    def on_error(self, plot_flag):
        """Executed if the exec() function failed.

        :param plot_flag: Flag indicating to plot the result.

        """
        pass

    def save(self, i):
        """Save data resulting of processing for given index to disk.

        :param i: Index to processing.

        """
        pass

    def plot(self, plot_flag):
        """Plot data resulting of processing for given index."""
        pass

class ProcessingCopy(ProcessingInterface):
    """Processing that copy one dataset from one location to another.

    Created for demonstration purpose.

    """
    title = "Copy"
    def load(self, i):
        l.LOGGER.debug("[{}] Load data for index {}".format(type(self).__name__, i))
        self.load_data = np.load(path.join(self.load_path, "{}_iq.npy".format(i)))
    def exec(self, plot_flag):
        l.LOGGER.debug("[{}] Exec processing for current index".format(type(self).__name__))
        self.save_data = self.load_data
        return True
    def save(self, i):
        l.LOGGER.debug("[{}] Save data for index {}".format(type(self).__name__, i))
        np.save(path.join(self.save_path, "{}_iq.npy".format(i)), self.save_data)
    def plot(self, plot_flag):
        if plot_flag is True:
            l.LOGGER.debug("[{}] Plot data for current index".format(type(self).__name__))
            plt.plot(np.abs(self.save_data))
            plt.show()

class ProcessingExtract(ProcessingInterface):
    """Processing that extract traces from IQs.

    The traces are amplitude and phase rotation of the targeted part of the
    AES. A template from the amplitude is used to align all the traces.
    Optionally, post-processing (filters) can be applied on the traces before
    saving.

    """
    title = "Extract"
    # Template used for traces.
    template = None
    # Extraction configuration [ExtractConf].
    config = None
    # If set to a filter by an external code [LHPFilter], it will be applied on
    # the IQ signal for trace extraction/alignement/averaging (but not for
    # triggering).
    filter_amp = None
    filter_phr = None

    def load(self, i):
        l.LOGGER.debug("[{}] Load data for index {}".format(type(self).__name__, i))
        self.load_data = np.load(path.join(self.load_path, "{}_iq.npy".format(i)))
        self.idx = i

    def exec(self, plot_flag):
        l.LOGGER.debug("[{}] Exec extraction for current index".format(type(self).__name__))
        # Save a template only if first processing.
        if self.idx == 0:
            average_file_name = path.join(self.save_path, "template.npy")
        else:
            average_file_name = None
        # Create unfiltered and filtered components.
        trace_amp = np.absolute(self.load_data)
        trace_phr = dsp.phase_rot(self.load_data)
        if self.filter_amp is not None:
            trace_amp_filt = np.absolute(self.filter_amp.apply(self.load_data, self.config.sampling_rate, force_dtype=True))
        if self.filter_phr is not None:
            trace_phr_filt = dsp.phase_rot(self.filter_phr.apply(self.load_data, self.config.sampling_rate, force_dtype=True))
        # Try one extraction on unfiltered amplitude to perform the triggering.
        try:
            self.save_data_amp, self.template, res_amp = legacy.extract(
                trace_amp, self.template, self.config, average_file_name=average_file_name,
            )
        except Exception as e:
            l.LOGGER.error("Error during extraction processing: {}".format(e))
            self.failed = True
        # Re-execute extractions with potentially filtered signals and with
        # phase rotation using previous results.
        if self.filter_amp is not None:
            self.save_data_amp, _, res_amp = legacy.extract(
                trace_amp_filt, self.template, self.config, average_file_name=average_file_name, results_old=res_amp
            )
        self.save_data_phr, _, res_phr = legacy.extract(
            trace_phr if self.filter_phr is None else trace_phr_filt, self.template, self.config, average_file_name=None, results_old=res_amp
        )
        # Plot results if asked.
        if plot_flag is True:
            legacy.plot_results(self.config, trace_amp, res_amp.trigger, res_amp.trigger_avg, res_amp.trace_starts, res_amp.traces,
                                target_path=None, plot=False, savePlot=False, title="Amplitude", final=False)
            legacy.plot_results(self.config, trace_phr, res_phr.trigger, res_phr.trigger_avg, res_phr.trace_starts, res_phr.traces,
                                target_path=self.save_path, plot=plot_flag, savePlot=plot_flag, title="Phase rotation", final=True)
        return self.failed == False

    def on_error(self, plot_flag):
        self.save_data_amp = np.zeros(self.template.shape, dtype=self.template.dtype)
        self.save_data_phr = np.zeros(self.template.shape, dtype=self.template.dtype)

    def save(self, i):
        l.LOGGER.debug("[{}] Save data for index {}".format(type(self).__name__, i))
        np.save(path.join(self.save_path, "{}_amp.npy".format(i)), self.save_data_amp)
        np.save(path.join(self.save_path, "{}_phr.npy".format(i)), self.save_data_phr)
    def plot(self, plot_flag):
        if plot_flag is True:
            l.LOGGER.debug("[{}] Plot data for current index".format(type(self).__name__))
            plt.plot(self.save_data_amp)
            plt.plot(self.save_data_phr)
            plt.show()

class Processor():
    """Processor allowing to apply a ProcessingInterface on several elements in parallel.

    Functions:
    - __init__: Create a new processing.
    - start: Execute the previously created processing.
    - disable_parallel: Disable the processing parallelization.
    - restore_parallel: Restore the previous processing parallelization.
    - is_parallel: To know if parallelization is enabled.

    """

    # Index of start for the processing.
    start_idx = None
    # Index of stop for the processing (-1 means infinite).
    stop_idx = None
    # Processing class [ProcessingInterface].
    processing = None
    # Plotting switch [PlotOnce].
    plot_once = None
    # Number of workers.
    # < 0 = maximum available processes.
    # > 0 = specified number of processes.
    # 0 = no process, run sequentially.
    ncpu = None
    ncpu_bak = None # Backup to restore is needed.
    # List of failed processes.
    bad_list = None

    def __init__(self, processing, plot_once, ncpu=-1, stop=-1):
        """Initialize a processing.

        It will run the ProcessingInterface using the plot switch PLOT.

        If NCPU is set to negative number, use the maximum number of workers.
        If set to a positive number, use this as number of workers. If set to
        0, disable multi-process processing and use a single-process
        processing.

        :param stop: If set to integer, use this as stop index. If set to
        function reference (partial), execute it returning the stop index.

        """
        assert isinstance(processing, ProcessingInterface), "Bad parameter class!"
        assert isinstance(plot_once, helpers.ExecOnce), "Bad parameter class!"
        assert (isinstance(stop, int) or isinstance(stop, partial)), "Bad parameter class!"
        # Initialize variables with default values.
        self.start_idx = 0
        self.bad_list = []
        # Install the signal handler to properly quit.
        self.__signal_install()
        # Set stop index.
        if isinstance(stop, int) is True:
            self.stop_idx = stop
        elif isinstance(stop, partial):
            self.stop_idx = stop()
        # Set number of CPUs.
        if ncpu < 0:
            self.ncpu = os.cpu_count() - 1
            l.LOGGER.info("[{}] Select {} processes for parallelization".format(type(self).__name__, self.ncpu))
        else:
            self.ncpu = ncpu
        self.ncpu_bak = self.ncpu
        # Save other given parameters.
        self.processing = processing
        self.plot_once = plot_once

    def start(self):
        """Start the (parallelized) processing.

        As a best practice, the first processing may need to be executed in the
        main process to modify an external object on which remaning processings
        could rely on.

        """
        def _init(i, stop_idx):
            """Initialize the processing starting at given index.

            :param i: Current processing index. 
            :param stop_idx: Upper bound for i to execute the processing.
            :returns: A tuple composed of the Queue for result transfer and a
                      list of processes.

            """
            # Disable parallelization for first processing.
            self.disable_parallel(i == 0)
            # Queue for transferring results from processing function (parallelized or not).
            q = Queue()
            # List of processes. Only create necessary processes.
            ps_len = self.ncpu
            if i + self.ncpu >= stop_idx:
                ps_len -= i + self.ncpu - stop_idx
            ps = [None] * ps_len
            # Initialize the processes if needed (but do not run them).
            for idx, _ in enumerate(ps):
                l.LOGGER.debug("[{}] Process index #{} for processing index #{} is created!".format(type(self).__name__, idx, i + idx))
                ps[idx] = Process(target=self.__process_fn, args=(q, i + idx, self.plot_once.pop(), self.processing))
            return q, ps

        def _run(i, q, ps):
            """Run the processing.

            :param i: Processing index. 
            :param q: Processing queue.
            :param ps: Process list.

            """
            # Create the processes and perform the parallelized processing, or...
            if self.is_parallel():
                for idx, proc in enumerate(ps):
                    proc.start()
                    l.LOGGER.debug("[{}] Started process: idx={}".format(type(self).__name__, idx))
            # ...or perform process sequentially.
            else:
                self.__process_fn(q, i, self.plot_once.pop(), self.processing)

        def _get(q, ps):
            """Get the processing results using the queue.

            :param q: Processing queue.
            :param ps: Process list.

            """
            # Check the result.
            for _, __ in enumerate(ps):
                l.LOGGER.debug("[{}] Wait result from queue...".format(type(self).__name__))
                check, i_processed = q.get()
                if check is False:
                    self.bad_list.append(i_processed)

        def _end(i_done, ps, pbar=None):
            """Terminate the processing for given index.

            If parallelized, terminated the processing contained in the PS
            list. If set, update tqdm's bar just like given index.

            :param i_done: Finished processing index.
            :param ps: Process list.
            :param pbar: tqdm's bar.
            :returns: Return the new index for next processing.

            """
            # Terminate the processes.
            for idx, proc in enumerate(ps):
                l.LOGGER.debug("[{}] Join process... idx={}".format(type(self).__name__, idx))
                proc.join()
            # Update the progress index and bar.
            i_step = len(ps) if self.ncpu > 0 else 1
            i = i_done + i_step
            pbar.update(i_step)
            # Restore parallelization after first processing if needed.
            # NOTE: Should be at the end since it will modify self.ncpu.
            self.restore_parallel(i_done == 0)
            l.LOGGER.debug("[{}] Finished processing: index #{} -> #{}".format(type(self).__name__, i_done, i - 1))
            return i
            
        # Setup progress bar.
        with (logging_redirect_tqdm(loggers=[l.LOGGER]),
              tqdm(initial=self.start_idx, total=self.stop_idx, desc=self.processing.title) as pbar,):
            i = self.start_idx
            while i < self.stop_idx:
                # Initialize processing for indexes starting at index i.
                q, ps = _init(i, self.stop_idx)
                # Run the processing.
                _run(i, q, ps)
                # Get and check the results.
                _get(q, ps)
                # Terminate the processing.
                i = _end(i, ps, pbar=pbar)
        return self

    def disable_parallel(self, cond=True):
        """Disable the parallel processing.

        Inverse of this function is restore_parallel() to restore the previous
        parallelization value.

        :param cond: Must be True to execute the function.

        """
        if cond is True and self.is_parallel() is True:
            l.LOGGER.debug("[{}] Disable parallelization for next processings!".format(type(self).__name__))
            self.ncpu_bak = self.ncpu
            self.ncpu = 0

    def restore_parallel(self, cond=True):
        """Restore the process parallelization.

        Will restore as before disable_parallel() call.

        :param cond: Must be True to execute the function.

        """
        if cond is True and self.is_parallel(was=True):
            self.ncpu = self.ncpu_bak
            l.LOGGER.debug("[{}] Restore {} processes for next processings!".format(type(self).__name__, self.ncpu))

    def is_parallel(self, was=False):
        """Return True if parallelization is enabled, False otherwise.

        :param was: If set to True, test against the value before the
        disable_parallel() call.

        """
        return self.ncpu > 0 if was is False else self.ncpu_bak > 0

    def __signal_install(self):
        """Install the signal handler.

        Catch the SIGINT signal.

        """
        global PROCESSOR
        PROCESSOR = self
        signal.signal(signal.SIGINT, self.__signal_handler)

    def __process_fn(self, q, i, plot_flag, processing):
        """Main function for processes of the Process class.

        It is usually ran by a caller process from the Processor._run()
        function. It may be run in the main proces too. It will load one index
        value from the data, execute the processing, may check and plot the
        result, and save the resulting data.

        :param q: Queue to transmit the results.
        :param i: Index to load and process.
        :param plot_flag: Flag indicating to plot the result.
        :param processing: A ProcessingInterface class.

        :raises: Can raise an exception if the first processing fail.

        """
        # Sanity-check.
        assert isinstance(i, int) and isinstance(plot_flag, bool) and isinstance(processing, ProcessingInterface)
        l.LOGGER.debug("[{}] Start __process_fn() for index #{}...".format(type(self).__name__, i))
        # Load the data to process.
        processing.load(i)
        # Apply the processing.
        check = processing.exec(plot_flag)
        # Terminate the processing if the first one failed.
        if check == False and i == 0:
            raise Exception("First processing encountered an error!")
        # Execute the fallback if a processing failed but is not the first one.
        elif check == False and i != 0:
            processing.on_error(plot_flag)
        # Plot the results if needed.
        processing.plot(plot_flag)
        # * Save the processed data and transmit result to caller process.
        processing.save(i)
        q.put((check, i))
        l.LOGGER.debug("[{}] End __process_fn() for index #{}".format(type(self).__name__, i))

    @staticmethod
    def __signal_handler(sig, frame):
        """Catch signal properly exiting process.

        Signal handler supposed to catch SIGINT to properly quit the processing
        by setting the stop index to 0.

        """
        global PROCESSOR
        PROCESSOR.stop_idx = 0

    def __str__(self):
        string = "Processor:\n"
        string += "\tstart_idx: {}\n".format(self.start_idx)
        string += "\tstop_idx: {}\n".format(self.stop_idx)
        return string + self.processing.__str__()
