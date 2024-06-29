class DatasetProcessing():
    """Processing workflow using a Dataset.

    Allows to use friendly interface functions for scripting and advanced
    processing functions (e.g. parallelization).

    Functions:
    - resume: Resume from previous processing.
    - create: Create a new processing.
    - process: Execute the previously created processing.
    - disable_plot: Disable the plot(s) for next processing.
    - disable_parallel: Disable the processing parallelization.
    - restore_parallel: Restore the previous processing parallelization.
    - is_parallel: To know if parallelization is enabled.

    """

    # * List all public variables.
    # Dataset of the processing.
    # NOTE: Mandatory to not be None.
    dset = None
    # Subset of the processing.
    # NOTE: Mandatory to not be None.
    sset = None
    # Index of start trace for the processing.
    start = 0
    # Index of stop trace for the processing (-1 means infinite).
    stop = -1
    # Processing title.
    process_title = None
    # Processing function. Must have this signature:
    # FUNC_NAME(dset, sset, plot, args)
    # Where the FUNC_NAME can get supplementary arguments from the ARGS list/tuple.
    process_fn = None
    # Processing plotting switch (a PlotOnce class).
    process_plot = None
    # Processing function arguments.
    process_args = None
    # Processing number of workers.
    # < 0 = maximum available processes.
    # > 0 = specified number of processes.
    # 0 = no process, run sequentially.
    process_nb = None
    _process_nb = None # Backup.

    def __init__(self, indir, subset, outdir=None, stop=-1):
        """Initialize a dataset processing.

        Load a dataset from INDIR and load the SUBSET subset. If OUTDIR is not
        None, set it as the savedir of the dataset. On error during the dataset
        loading, quit the programm.

        """
        # Install the signal handler.
        self.__signal_install()
        # Get dataset and subset.
        self.dset = Dataset.pickle_load(indir, quit_on_error=True)
        self.sset = self.dset.get_subset(subset)
        assert self.dset is not None and self.sset is not None
        # Set the outdir directory for saving.
        if outdir is not None:
            self.dset.set_dirsave(outdir)
        # Set stop trace.
        if stop == -1:
            self.stop = self.sset.get_nb_trace_ondisk()
        else:
            self.stop = stop
        # Set the dirty flag to True after loading.
        self.dset.dirty = True

    def resume(self, from_zero=False):
        """Resume the processing of a dataset...

        If:
        - The FROM_ZERO parameter is set to False.
        - The DIRTY flag of the previsouly saved dataset is set to True.

        By:
        1. Fetching the template previously saved.
        2. Fetching the bad entries previously saved.
        3. Using the dirty idx previously saved as start index.

        """
        if from_zero is False and self.dset.get_savedir_dirty():
            self.dset.resume_from_savedir(self.sset.subtype)
            self.start = self.dset.dirty_idx
            l.LOGGER.info("Resume at trace {} using template from previous processing".format(self.start))
            l.LOGGER.debug("Template: shape={}".format(self.sset.template.shape))

    def create(self, title, fn, plot, args, nb = -1):
        """Create a processing.

        The processing will be titled TITLE, running the function FN using the
        plot switch PLOT and custom arguments ARGS.

        If NB is set to negative number, use the maximum number of workers. If
        set to a positive number, use this as number of workers. If set to 0,
        disable multi-process processing and use a single-process processing.

        """
        assert isinstance(plot, libplot.PlotOnce), "plot parameter must be a PlotOnce class!"
        self.process_title = title
        self.process_fn = fn
        self.process_plot = plot
        self.process_args = args
        if nb < 0:
            self.process_nb = os.cpu_count() - 1
            l.LOGGER.info("Automatically select {} processes for parallelization".format(self.process_nb))
        else:
            self.process_nb = nb
        self._process_nb = self.process_nb

    def process(self):
        """Run the (parallelized) processing.

        The processing must be configured using Dataset
        Processing.create()
        before to use this function.

        """
        # Check that self.create() function has been called.
        assert self.process_title is not None
        assert self.process_fn is not None
        assert self.process_plot is not None
        assert self.process_args is not None
        assert self.process_nb >= 0
        
        def _init(i, stop):
            """Initialize the processing starting at trace index I.

            Return a tuple composed of the Queue for result transfer and a list
            of processes.

            """
            # NOTE: The first processing needs to be executed in the main
            # process to modify the dataset object. Remaning processings could
            # rely on this one to get some parameters (e.g. the template
            # signal).
            self.disable_parallel(i == 0)
            # Queue for transferring results from processing function (parallelized or not).
            q = Queue()
            # List of processes. Only create necessary processes.
            ps_len = self.process_nb
            if i + self.process_nb >= stop:
                ps_len -= i + self.process_nb - stop
            ps = [None] * ps_len
            # Initialize the processes if needed (but do not run them).
            for idx, _ in enumerate(ps):
                l.LOGGER.debug("Create process index #{} for trace index #{}".format(idx, i + idx))
                ps[idx] = Process(target=self.__process_fn, args=(q, self.dset, self.sset, i + idx, self.process_plot.pop(), self.process_args,))
            return q, ps

        def _run(i, q, ps):
            """Run the processing starting at trace index I using the Queue Q
            and the processes of list PS.

            """
            # Create the processes and perform the parallelized processing...
            if self.is_parallel():
                for idx, proc in enumerate(ps):
                    proc.start()
                    l.LOGGER.debug("Started process: idx={}".format(idx))
            # ...or perform process sequentially.
            else:
                self.__process_fn(q, self.dset, self.sset, i, self.process_plot.pop(), self.process_args)

        def _get(q, ps):
            """Get the processing results using the Queue Q and the processes of list PS."""
            # Check the result.
            for _, __ in enumerate(ps):
                l.LOGGER.debug("Wait result from queue...")
                check, i_processed = q.get()
                if check is True:
                    self.sset.bad_entries.append(i_processed)

        def _end(i_done, ps, pbar=None):
            """Terminate the processing for trace index I_DONE.

            1. Update the processing loop information to prepare the next
               processing.
            2. Save the processing state in the dataset for further
               resuming.
            3. If parallelized, terminated the processing contained in
               the PS list.
            4. If specified, update TQDM's PBAR just like index I_DONE.

            Return the new index I for next processing.

            """
            # Terminate the processes.
            for idx, proc in enumerate(ps):
                l.LOGGER.debug("Join process... idx={}".format(idx))
                proc.join()
            # Update the progress index and bar.
            # NOTE: Handle case where process_nb == 0 for single-process processing.
            i_step = len(ps) if self.process_nb > 0 else 1
            i = i_done + i_step
            pbar.update(i_step)
            # Save dataset for resuming if not finishing the loop.
            self.dset.dirty_idx = i
            self.dset.pickle_dump(unload=False, log=False)
            # Restore parallelization after first trace processing if needed.
            # NOTE: Should be at the end since it will modify self.process_nb.
            self.restore_parallel(i_done == 0)
            l.LOGGER.debug("Finished processing: trace #{} -> #{}".format(i_done, i - 1))
            return i
            
        # Setup progress bar.
        with (logging_redirect_tqdm(loggers=[l.LOGGER]),
              tqdm(initial=self.start, total=self.stop, desc=self.process_title) as pbar,):
            i = self.start
            while i < self.stop:
                # Initialize processing for trace(s) starting at index i.
                q, ps = _init(i, self.stop)
                # Run the processing.
                _run(i, q, ps)
                # Get and check the results.
                _get(q, ps)
                # Terminate the processing.
                i = _end(i, ps, pbar=pbar)

    def disable_plot(self, cond=True):
        """Disable the plotting parameter if COND is True."""
        if cond is True and self.process_plot is True:
            l.LOGGER.debug("Disable plotting for next processings")
            self.process_plot = False

    def disable_parallel(self, cond=True):
        """Disable the parallel processing if COND is True.

        One can call restore_parallel() to restore the previous parallelization
        value.

        """
        if cond is True and self.is_parallel() is True:
            l.LOGGER.debug("Disable parallelization for next processings")
            self._process_nb = self.process_nb
            self.process_nb = 0

    def restore_parallel(self, cond=True):
        """Restore the process parallelization as before disable_parallel()
        call if COND is True.

        """
        if cond is True and self.is_parallel(was=True):
            l.LOGGER.debug("Restore previous parallelization value for next processings")
            self.process_nb = self._process_nb

    def is_parallel(self, was=False):
        """Return True if parallelization is enabled, False otherwise.

        Set WAS to True to test against the value before the disable_parallel()
        call.

        """
        return self.process_nb > 0 if was is False else self._process_nb > 0

    def __signal_install(self):
        """Install the signal handler.

        Catch the SIGINT signal.

        """
        global DPROC
        DPROC = self
        signal.signal(signal.SIGINT, self.__signal_handler)

    def __process_fn(self, q, dset, sset, i, plot, args):
        """Main function for processes.

        It is usually ran by a caller process from the self.process/_run()
        function. It may be run in the main proces too. It will load a trace,
        execute the processing based on the self.process_fn function pointer,
        may check and plot the result, and save the resulting trace.

        Q is a Queue to transmit the results, DSET a Dataset, SSET a Subset, I
        the trace index to load and process, PLOT a flag indicating to plot the
        result, and ARGS additionnal arguments transmitted to the
        self.process_fn function.

        """
        l.LOGGER.debug("Start __process_fn() for trace #{}...".format(i))
        # * Load the trace to process.
        # NOTE: We choose to always load traces one by one since raw traces can
        # be large (> 30 MB).
        sset.load_trace(i, nf=False, ff=True, check=True, log=False)
        # * Apply the processing and get the resulting trace.
        # NOTE: ff can be None if the processing fails.
        ff = self.process_fn(dset, sset, plot, args)
        # * Check the trace is valid.
        check = False
        if i > 0:
            check, ff_checked = analyze.fill_zeros_if_bad(sset.template, ff, log=True, log_idx=i)
        elif i == 0 and ff is not None:
            l.LOGGER.info("Trace #0 processing (e.g. creating a template) is assumed to be valid!")
            ff_checked = ff
        else:
            raise Exception("Trace #0 processing encountered an error!")
        sset.replace_trace(ff_checked, TraceType.FF)
        # * Plot the averaged trace if wanted and processing succeed.
        if sset.ff[0] is not None:
            libplot.plot_time_spec_sync_axis(sset.ff[0:1], samp_rate=dset.samp_rate, cond=plot, comp=complex.CompType.AMPLITUDE)
        # * Save the processed trace and transmit result to caller process.
        sset.save_trace(nf=False, custom_dtype=False)
        q.put((check, i))
        l.LOGGER.debug("End __process_fn() for trace #{}".format(i))

    @staticmethod
    def __signal_handler(sig, frame):
        """Catch signal properly exiting process.

        Signal handler supposed to catch SIGINT to properly quit the processing
        by setting the stop index to 0.

        """
        global DPROC
        DPROC.stop = 0

    def __str__(self):
        """Return the __str__ from the dataset."""
        string = "dataset_processing:\n"
        string += "- start: {}\n".format(self.start)
        string += "- stop: {}\n".format(self.stop)
        return string + self.dset.__str__()
