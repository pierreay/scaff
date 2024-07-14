"""Digital Signal Processing (DSP) functions."""

# * Importation

import builtins

# External import.
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter

# * Classes

class LHPFilter():
    """Low/High-pass digital Butterworth FIR filter."""

    # Filter type ["low" | "high"]
    type = None
    # Cut-off frequnecy [Hz].
    cutoff = None
    # Order.
    order = None
    # Toggle switch.
    enabled = None
    
    @staticmethod
    def _butter_highpass_filter(sigin, cutoff, fs, order=5):
        """Alternative implementation of the highpass filter."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="high", analog=False)
        if type(sigin) == np.floating:
            return lfilter(b, a, sigin)
        elif type(singin) == np.complexing:
            return lfilter(b, a, sigin.real) + 1j * lfilter(b, a, sigin.imag)

    @staticmethod
    def _butter_lowpass_filter(sigin, cutoff, fs, order=5):
        """Alternative implementation of the lowpass filter."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        if type(sigin) == np.floating:
            return lfilter(b, a, sigin)
        elif type(singin) == np.complexing:
            return lfilter(b, a, sigin.real) + 1j * lfilter(b, a, sigin.imag)

    def __init__(self, type, cutoff, order=1, enabled=True):
        """Configure a filter."""
        # Input check.
        assert type in ["low", "high"], "Bad filter type!"
        assert builtins.type(cutoff) == int or builtins.type(cutoff) == float, "Bad cutoff type!"
        # Get parameters.
        self.type = type
        self.cutoff = cutoff
        self.order = order
        self.enabled = enabled

    def apply(self, sigin, fs, force_dtype=False):
        """Apply the configured filter to a signal.

        :param sigin: Signal to filter, may be real-valued (e.g., float) or
        complex-valued (filter applied on real and imag independantly).

        :param fs: Sampling rate of the signal.

        :returns: The filtered signal.

        """
        # Return input signal if disabled.
        if self.enabled is False:
            return sigin
        # Filter the signal.
        sigout = signal.sosfilt(signal.butter(self.order, self.cutoff, self.type, fs=fs, output="sos"), sigin)
        # Convert back to orignal dtype if requested, otherwise, return as it.
        if force_dtype is True:
            return np.array(sigout, dtype=sigin.dtype)
        else:
            return sigout

# * Functions

def phase_rot(trace):
    """Get the phase rotation of one or multiple traces."""
    dtype_in = np.complex64
    dtype_out = np.float32
    assert type(trace) == np.ndarray
    assert trace.dtype == dtype_in
    if trace.ndim == 1:
        # NOTE: Phase rotation from expe/240201/56msps.py without filter:
        # Compute unwraped (remove modulos) instantaneous phase.
        trace = np.unwrap(np.angle(trace))
        # Set the signal relative to 0.
        trace = [trace[i] - trace[0] for i in range(len(trace))]
        # Compute the phase rotation of instantenous phase.
        # NOTE: Manually add first [0] sample.
        trace = [0] + [trace[i] - trace[i - 1] for i in range(1, len(trace), 1)]
        # Convert back to np.ndarray.
        trace = np.array(trace, dtype=dtype_out)
        assert trace.dtype == dtype_out
        return trace
    elif trace.ndim == 2:
        trace_rot = np.empty_like(trace, dtype=dtype_out)
        for ti, tv in enumerate(trace):
            trace_rot[ti] = phase_rot(tv)
        return trace_rot
