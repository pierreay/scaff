"""Statistics."""

# * Importation

# Interal import.
from os import path

# External import.
import numpy as np
import matplotlib.pyplot as plt

# * Classes

class Profile():
    # Filenames.
    POIS_FN       = "POIS.npy"
    RS_FN         = "PROFILE_RS.npy"
    RZS_FN        = "PROFILE_RZS.npy"
    MEANS_FN      = "PROFILE_MEANS.npy"
    STDS_FN       = "PROFILE_STDS.npy"
    COVS_FN       = "PROFILE_COVS.npy"
    MEAN_TRACE_FN = "PROFILE_MEAN_TRACE.npy"

    # Data.
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
    
    def __init__(self):
        pass

    def save(self, save_dir):
        np.save(path.join(save_dir, Profile.POIS_FN), self.POIS)
        np.save(path.join(save_dir, Profile.RS_FN), self.RS)
        np.save(path.join(save_dir, Profile.RZS_FN), self.RZS)
        np.save(path.join(save_dir, Profile.MEANS_FN), self.MEANS)
        np.save(path.join(save_dir, Profile.STDS_FN), self.STDS)
        np.save(path.join(save_dir, Profile.COVS_FN), self.COVS)
        np.save(path.join(save_dir, Profile.MEAN_TRACE_FN), self.MEAN_TRACE)

    # Load the profile, for comparison or for attacks.
    def load(self, load_dir):
        self.POIS       = np.load(path.join(load_dir, Profile.POIS_FN))
        self.RS         = np.load(path.join(load_dir, Profile.RS_FN))
        self.RZS        = np.load(path.join(load_dir, Profile.RZS_FN))
        self.MEANS      = np.load(path.join(load_dir, Profile.MEANS_FN))
        self.COVS       = np.load(path.join(load_dir, Profile.COVS_FN))
        self.STDS       = np.load(path.join(load_dir, Profile.STDS_FN))
        self.MEAN_TRACE = np.load(path.join(load_dir, Profile.MEAN_TRACE_FN))
        return self

    def plot(self, show=False, save=None, clear=True, plt_param_dict={}):
        # Plot the mean trace.
        plt.subplot(2, 1, 1)
        plt.plot(self.MEAN_TRACE, **plt_param_dict)
        plt.xlabel("Samples")
        plt.ylabel("Mean trace")
        # Plot the POIs.
        plt.subplot(2, 1, 2)
        plt.xlabel("Samples")
        plt.ylabel("Correlation coeff. (r)")
        for i, snr in enumerate(self.RS):
            plt.plot(snr, label="subkey %d"%i, **plt_param_dict)
        for bnum in range(16):
            plt.plot(self.POIS[bnum], self.RS[bnum][self.POIS[bnum]], '.', **plt_param_dict)
        if save is not None:
            figure = plt.gcf() # Get current figure
            figure.set_size_inches(32, 18) # Set figure's size manually to your full screen (32x18).
            plt.savefig(save, bbox_inches='tight', dpi=100)
        if show is True:
            plt.show()
        if clear is True:
            plt.clf()
