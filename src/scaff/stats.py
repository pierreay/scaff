"""Statistics."""

# * Importation

# External import.
import numpy as np
import matplotlib.pyplot as plt

# * Classes

class Profile():
    # Reference to the parent dataset (used to resolve path). Can be None.
    dataset = None
    # Name of the (sub)directory containing the profile if linked to a parent
    # dataset.
    dir = None
    # Full path of the profile directory. Arbitrary if not linked to a parent
    # dataset, otherwise set according to self.dataset.dir.
    fp = None

    # Profile's filenames.
    POIS_FN       = "POIS.npy"
    RS_FN         = "PROFILE_RS.npy"
    RZS_FN        = "PROFILE_RZS.npy"
    MEANS_FN      = "PROFILE_MEANS.npy"
    STDS_FN       = "PROFILE_STDS.npy"
    COVS_FN       = "PROFILE_COVS.npy"
    MEAN_TRACE_FN = "PROFILE_MEAN_TRACE.npy"

    # Profile's data.
    POIS        = None
    RS          = None
    RZS         = None
    MEANS       = None
    STDS        = None
    COVS        = None
    MEAN_TRACE  = None
    # Starting point used in original trace.
    POINT_START = None
    # Ending point used in original trace.
    POINT_END   = None
    
    def __init__(self, dataset = None, fp = None):
        """Initialize a profile.

        Set EITHER the DATASET parameter to a Dataset reference or the FP
        parameter to a full valid path.

        """
        # Safety-check of using either DATASET or FP.
        assert fp is None if dataset is not None else True
        assert dataset is None if fp is not None else True

        # Attach a dataset if needed.
        if dataset is not None:
            self.attach_dataset(dataset)
        # Attach a full path if needed.
        elif fp is not None:
            self.attach_path(fp)

    def get_path(self, save=False, fp=False):
        """Return the absolute path of the Profile.

        Assert that the dirname of the returned path exists. If FP is set to
        True, force returning path based on self.fp.

        """
        # If a dataset is attached to the profile, return a path based on the
        # dataset path.
        if self.dataset is not None and fp is False:
            assert self.dataset.dir is not None and path.exists(self.dataset.dir)
            return path.abspath(path.join(self.dataset.dir, self.dir))
        # If a full path is registered, return it.
        elif self.fp is not None or fp is True:
            assert path.exists(path.dirname(self.fp))
            return path.abspath(self.fp)
        else:
            assert False, "Profile has not been configured correctly!"

    def save(self, full_path=None):
        """Store traces and points from the Profile."""
        # NOTE: Feature to test.
        fp = False
        # if full_path is not None:
        #     self.fp = path.abspath(full_path)
        #     fp = True
        os.makedirs(self.get_path(fp=fp), exist_ok=True)
        np.save(path.join(self.get_path(fp=fp), Profile.POIS_FN), self.POIS)
        np.save(path.join(self.get_path(fp=fp), Profile.RS_FN), self.RS)
        np.save(path.join(self.get_path(fp=fp), Profile.RZS_FN), self.RZS)
        np.save(path.join(self.get_path(fp=fp), Profile.MEANS_FN), self.MEANS)
        np.save(path.join(self.get_path(fp=fp), Profile.STDS_FN), self.STDS)
        np.save(path.join(self.get_path(fp=fp), Profile.COVS_FN), self.COVS)
        np.save(path.join(self.get_path(fp=fp), Profile.MEAN_TRACE_FN), self.MEAN_TRACE)

    # Load the profile, for comparison or for attacks.
    def load(self):
        self.POIS       = np.load(path.join(self.get_path(), Profile.POIS_FN))
        self.RS         = np.load(path.join(self.get_path(), Profile.RS_FN))
        self.RZS        = np.load(path.join(self.get_path(), Profile.RZS_FN))
        self.MEANS      = np.load(path.join(self.get_path(), Profile.MEANS_FN))
        self.COVS       = np.load(path.join(self.get_path(), Profile.COVS_FN))
        self.STDS       = np.load(path.join(self.get_path(), Profile.STDS_FN))
        self.MEAN_TRACE = np.load(path.join(self.get_path(), Profile.MEAN_TRACE_FN))

    def plot(self, delim=False, save=None, plt_param_dict={}):
        # Code taken from attack.py:find_pois().
        # Plot the POIs.
        plt.subplots_adjust(hspace = 1)
        plt.subplot(2, 1, 1)
        plt.xlabel("Samples")
        plt.ylabel("Correlation coeff. (r)")
        for i, snr in enumerate(self.RS):
            plt.plot(snr, label="subkey %d"%i, **plt_param_dict)
        for bnum in range(16):
            plt.plot(self.POIS[bnum], self.RS[bnum][self.POIS[bnum]], '.')
        # Plot the mean trace.
        plt.subplot(2, 1, 2)
        plt.plot(self.MEAN_TRACE, **plt_param_dict)
        plt.xlabel("Samples")
        plt.ylabel("Mean trace")
        plt.tight_layout()
        if save is None:
            plt.show()
        else:
            plt.savefig(save)

        # Advanced plot by printing the delimiters using the FF trace #0.
        # NOTE: This part imply that profile has been built with FF and not NF.
        if delim is not False and self.dataset.train_set.get_nb_trace_ondisk() > 0:
            if self.dataset.train_set.ff is None:
                self.dataset.train_set.load_trace(0, nf=False, ff=True, check=True)
            libplot.plot_time_spec_sync_axis(self.dataset.train_set.ff[0:1], samp_rate=self.dataset.samp_rate, peaks=[self.POINT_START, self.POINT_END])
   
    def __str__(self):
        string = "profile:\n"
        string += "- dataset: {}\n".format(self.dataset is not None)
        string += "- dir: {}\n".format(self.dir)
        string += "- fp: {}\n".format(self.fp)
        string += "- get_path(): {}\n".format(self.get_path())
        if self.POIS is not None:
            string += "- pois shape: {}\n".format(self.POIS.shape)
        if self.MEAN_TRACE is not None:
            string += "- profile trace shape: {}\n".format(self.MEAN_TRACE.shape)
        if self.POINT_START:
            string += "- profile start point: {}\n".format(self.POINT_START)
        if self.POINT_END:
            string += "- profile end point: {}\n".format(self.POINT_END)
        return string

    def attach_dataset(self, dataset):
        """Attach a dataset to the Profile."""
        assert dataset is not None and type(dataset) == Dataset
        assert self.dataset is None, "Cannot attach a new dataset while a dataset is still attached!"
        assert self.fp is None, "Cannot attach a new dataset while a full path is already set!"
        self.dir = "profile"   # Fixed subdirectory.
        self.dataset = dataset # Parent. Don't need to save the subset as the
                               # subset is always train for a profile.

    def attach_path(self, fp):
        """Attach a full path to the Profile."""
        assert path.exists(fp)
        assert self.dataset is None, "Cannot attach a new path while a dataset is still attached!"
        assert self.fp is None, "Cannot attach a new path while a full path is already set!"
        self.fp = fp

