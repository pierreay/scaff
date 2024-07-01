#!/usr/bin/python3

# * Importations

import sys
from enum import Enum
import os
from os import path
import pickle
import itertools
import binascii
import math

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.stats import multivariate_normal, linregress, norm, pearsonr, entropy
from scipy.stats import ttest_ind, f
import statsmodels.api as sm
from tqdm import tqdm

from scaff import logger as l
from scaff import config

# * Analyze

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    assert(cutoff < fs / 2) # Nyquist
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    assert(cutoff < fs / 2) # Nyquist
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    assert(lowcut < fs / 2 and highcut < fs / 2) # Nyquist
    assert(lowcut < highcut)                     # Logic
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def shift(sig, val):
    """Shift a signal SIG from the VAL offset.

    Shift a signal SIG to left (positive VAL) or right (negative
    VAL). Empty parts of the signal are completed using np.zeros of same
    dtype as SIG.

    VAL can be the output of the signal.correlate() function.

    """
    if val > 0:
        sig = sig[val:]
        sig = np.append(sig, np.zeros(val, dtype=sig.dtype))
    elif val < 0:
        sig = sig[:val]
        sig = np.insert(sig, 0, np.zeros(-val, dtype=sig.dtype))
    return sig

def get_shift_corr(arr_1, arr_2):
    """Get the shift maximizing cross-correlation between arr_1 and arr_2."""
    corr = signal.correlate(arr_1, arr_2)
    return np.argmax(corr) - (len(arr_2) - 1)

def align(template, target, sr, ignore=True, log=False, get_shift_only=False, normalize=False):
    """Align a signal against a template.

    Return the TARGET signal aligned (1D np.array) using cross-correlation
    along the TEMPLATE signal, where SR is the sampling rates of signals. The
    shift is filled with zeros shuch that shape is not modified.

    - If IGNORE is set to false, raise an assertion for high shift values.
    - If LOG is set to True, log the shift produced by the cross-correlation.

    NOTE: The cross-correlation shift is computed based on amplitude
    (np.float64) of signals.

    """
    # +++===+++++++++
    # +++++++===+++++ -> shift > 0 -> shift left target -> shrink template from right or pad target to right
    # ===++++++++++++ -> shift < 0 -> shift right target -> shrink template from left or pad target to left
    # Safety-check to prevent weird exception inside the function.
    assert template.shape > (1,) and target.shape > (1,), "Cannot align empty traces!"
    # NOTE: Disabled this assertation because I'm not sure why it was necessary.
    # assert template.shape == target.shape, "Traces to align should have the same length!"
    assert template.ndim == 1 and target.ndim == 1, "Traces to align should be 1D-ndarray!"
    # Compute the cross-correlation and find shift across amplitude.
    lpf_freq     = sr / 4
    template_lpf = butter_lowpass_filter(get_amplitude(template), lpf_freq, sr)
    target_lpf   = butter_lowpass_filter(get_amplitude(target), lpf_freq, sr)
    if normalize is True:
        template_lpf = normalize(template_lpf)
        target_lpf = normalize(target_lpf)
    shiftv        = get_shift_corr(target_lpf, template_lpf)
    if get_shift_only is True:
        return shiftv
    # Log and check shift value.
    if log:
        l.LOGGER.debug("Shift to maximize cross correlation: {}".format(shiftv))
    if not ignore:
        assert np.abs(shiftv) < len(template/10), "shift is too high, inspect"
    # Apply shift on the raw target signal.
    return shift(target, shiftv)

def align_nb(s, nb, sr, template, tqdm_log=True):
    s_aligned = [0] * nb
    if tqdm_log:
        lrange = tqdm(range(0, nb), desc="Align")
    else:
        lrange = list(range(0, nb))
    for idx in lrange:
        s_aligned[idx] = align(template, s[idx], sr)
    s_aligned = np.array(s_aligned, dtype=s.dtype)
    return s_aligned

def align_all(s, sr, template=None, tqdm_log=True):
    """Align the signals contained in the S 2D np.array of sampling rate
    SR. Use TEMPLATE signal (1D np.array) as template/reference signal if
    specified, otherwise use the first signal of the S array.

    """
    return align_nb(s, len(s), sr, template if template is not None else s[0], tqdm_log)

# Enumeration of components type of a signal.
CompType = Enum('CompType', ['AMPLITUDE', 'PHASE', 'PHASE_ROT'])

NormMethod = Enum('NormMethod', ['MINMAX', 'ZSCORE', 'COMPLEX_ABS', 'COMPLEX_ANGLE'])

def is_iq(s):
    """Return True is the signal S is composed of IQ samples, False otherwise."""
    return s.dtype == np.complex64

def get_amplitude(traces):
    """Get the amplitude of one or multiples traces.

    From the TRACES 2D np.array of shape (nb_traces, nb_samples) or the 1D
    np.array of shape (nb_samples) containing IQ samples, return an array with
    the same shape containing the amplitude of the traces.

    If traces contains signals in another format than np.complex64, silently
    return the input traces such that this function can be called multiple
    times.

    """
    if traces.dtype == np.complex64:
        return np.abs(traces)
    else:
        return traces

def get_phase(traces):
    """Get the phase of one or multiples traces.

    From the TRACES 2D np.array of shape (nb_traces, nb_samples) or the 1D
    np.array of shape (nb_samples) containing IQ samples, return an array with
    the same shape containing the phase of the traces.

    If traces contains signals in another format than np.complex64, silently
    return the input traces such that this function can be called multiple
    times.

    """
    if traces.dtype == np.complex64:
        return np.angle(traces)
    else:
        return traces

def get_phase_rot(trace):
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
            trace_rot[ti] = get_phase_rot(tv)
        return trace_rot

def get_comp(traces, comp):
    """Get a choosen component.

    Return the choosen component of signals contained in the 1D or 2D ndarray
    TRACES according to COMP set to CompType.AMPLITUDE, CompType.PHASE or
    CompType.PHASE_ROT.

    If the signals contained in TRACES are already of the given component, this
    function will do nothing.

    """
    assert type(traces) == np.ndarray, "Traces should be numpy array"
    assert (type(comp) == str or comp in CompType), "COMP is set to a bad type or bad enum value!"
    if (type(comp) == CompType and comp == CompType.AMPLITUDE) or (type(comp) == str and CompType[comp] == CompType.AMPLITUDE):
        return get_amplitude(traces)
    elif (type(comp) == CompType and comp == CompType.PHASE) or (type(comp) == str and CompType[comp] == CompType.PHASE):
        return get_phase(traces)
    elif (type(comp) == CompType and comp == CompType.PHASE_ROT) or (type(comp) == str and CompType[comp] == CompType.PHASE_ROT):
        return get_phase_rot(traces)
    assert False, "Bad COMP string!"

def is_p2r_ready(radii, angles):
    """Check if polar complex can be converted to regular complex.

    Return True if values contained in RADII and ANGLES are in the acceptable
    ranges for the P2R (polar to regular) conversion. Without ensuring this,
    the conversion may lead to aberrant values.

    RADII and ANGLES can be ND np.ndarray containing floating points values.

    """
    # Check that RADII and ANGLES are not normalized.
    norm = is_normalized(radii) or is_normalized(angles)
    # Check that 0 <= RADII <= 2^16. NOTE: RADII is computed like the following
    # with maximum value of 16 bits integers (because we use CS16 from
    # SoapySDR):
    # sqrt((2^16)*(2^16) + (2^16)*(2^16)) = 92681
    # Hence, should we use 2^17 instead?
    radii_interval = radii[radii < 0].shape == (0,) and radii[radii > np.iinfo(np.uint16).max].shape == (0,)
    # Check that -PI <= ANGLES <= PI.
    angles_interval = angles[angles < -np.pi].shape == (0,) and angles[angles > np.pi].shape == (0,)
    return not norm and radii_interval and angles_interval

def p2r(radii, angles):
    """Complex polar to regular.

    Convert a complex number from Polar coordinate to Regular (Cartesian)
    coordinates.

    The input and output is symmetric to the r2p() function. RADII is
    the magnitude while ANGLES is the angles in radians (default for
    np.angle()).

    NOTE: This function will revert previous normalization as the range of
    values of RADII and ANGLES are mathematically important for the conversion.

    Example using r2p for a regular-polar-regular conversion:
    > polar = r2p(2d_ndarray_containing_iq)
    > polar[0].shape
    (262, 2629)
    > polar[1].shape
    (262, 2629)
    > regular = p2r(polar[0], polar[1])
    > regular.shape
    (262, 2629)
    > np.array_equal(arr, regular)
    False
    > np.isclose(arr, regular)
    array([[ True,  True,  True, ...,  True,  True,  True], ..., [ True,  True,  True, ...,  True,  True,  True]])

    Source: https://stackoverflow.com/questions/16444719/python-numpy-complex-numbers-is-there-a-function-for-polar-to-rectangular-co?rq=4

    """
    if not is_p2r_ready(radii, angles):
        radii  = normalize(radii,  method=NormMethod.COMPLEX_ABS)
        angles = normalize(angles, method=NormMethod.COMPLEX_ANGLE)
    return radii * np.exp(1j * angles)

def r2p(x):
    """Complex regular to polar.

    Convert a complex number from Regular (Cartesian) coordinates to Polar
    coordinates.

    The input X can be a 1) single complex number 2) a 1D ndarray of complex
    numbers 3) a 2D ndarray of complex numbers. The returned output is a tuple
    composed of a 1) two scalars (float32) representing magnitude and phase 2)
    two ndarray containing the scalars.

    Example using a 2D ndarray as input:
    r2p(arr)[0][1][0] -> magnitude of 1st IQ of 2nd trace.2
    r2p(arr)[1][0][1] -> phase of 2nd IQ of 1st trace.

    Source: https://stackoverflow.com/questions/16444719/python-numpy-complex-numbers-is-there-a-function-for-polar-to-rectangular-co?rq=4
    """
    # abs   = [ 0   ; +inf ] ; sqrt(a^2 + b^2)
    # angle = [ -PI ; +PI  ] ; angle in rad
    return np.abs(x), np.angle(x)

def normalize(arr, method=NormMethod.MINMAX, arr_complex=False):
    """Return a normalized ARR array.

    Set method to NormMethod.MINMAX to normalize using min-max feature scaling.

    Set method to NormMethod.ZSCORE to normalize using zscore normalization.

    Set method to NormMethod.COMPLEX_ABS to normalize between range of absolute
    value of a complex number.

    Set method to NormMethod.COMPLEX_ANGLE to normalize between range of angle
    of a complex number.

    By default, ARR is a ND np.ndarray containing floating points numbers. It
    should not contains IQ, as normalizing complex numbers doesn't makes sense
    (leads to artifacts). The normalization has to be applied on the magnitude
    and angle of the complex numbers, obtained using polar representation with
    complex.r2p(). Normalizing and converting back to regular representation
    just after doesn't make sense, since the normalization is reverted in the
    complex.p2r() function. Hence, we offer the optional ARR_COMPLEX option. If
    ARR_COMPLEX is set to True, ARR must contains complex numbers, and it will
    be returned a tuple composed of the normalized amplitude and the normalized
    angle. We use an explicit option to more easily show what is the input and
    output in the code that will use this function.

   """
    assert method in NormMethod
    if arr_complex is True:
        assert is_iq(arr), "normalization input should be complex numbers"
        arr_polar = r2p(arr)
        return normalize(arr_polar[0], method=method), normalize(arr_polar[1], method=method)
    else:
        assert arr.dtype == np.float32 or arr.dtype == np.float64, "normalization input should be floating points numbers"
        if method == NormMethod.MINMAX:
            return normalize_minmax(arr)
        elif method == NormMethod.ZSCORE:
            return normalize_zscore(arr)
        elif method == NormMethod.COMPLEX_ABS:
            # Refer to is_p2r_ready() and r2p() for bounds reference.
            return normalize_generic(arr, {'actual': {'lower': arr.min(), 'upper': arr.max()}, 'desired': {'lower': 0, 'upper': np.iinfo(np.int16).max}})
        elif method == NormMethod.COMPLEX_ANGLE:
            # Refer to is_p2r_ready() and r2p() for bounds reference.
            return normalize_generic(arr, {'actual': {'lower': arr.min(), 'upper': arr.max()}, 'desired': {'lower': -np.pi, 'upper': np.pi}})

def normalize_minmax(arr):
    """Apply min-max feature scaling normalization to a 1D np.array ARR
    representing the amplitude of a signal.

    Min-Max Scaling will scales data between a range of 0 to 1 in float.

    """
    assert arr.dtype == np.float32 or arr.dtype == np.float64
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def normalize_zscore(arr, set=False):
    """Normalize a trace using Z-Score normalization.

    Z-Score Normalization will converts data into a normal distribution with a
    mean of 0 and a standard deviation of 1.

    If SET is set to TRUE, apply normalization on the entire set instead of on
    each trace individually.

    Source: load.py from original Screaming Channels.

    """
    # Do not normalize I/Q samples (complex numbers).
    assert arr.dtype == np.float32 or arr.dtype == np.float64
    mu = np.average(arr) if set is False else np.average(arr, axis=0)
    std = np.std(arr) if set is False else np.std(arr, axis=0)
    if set is True or std != 0:
        arr = (arr - mu) / std
    return arr

def normalize_generic(values, bounds):
    """Normalize VALUES between BOUNDS.

    VALUES is a ND np.ndarray. BOUNDS is a dictionnary with two entries,
    "desired" and "actual", each one having the "upper" and "lower"
    bounds. This dictionnary is used to rescale the values from the "actual"
    bounds to the "desired" ones.

    Source:
    https://stackoverflow.com/questions/48109228/normalizing-data-to-certain-range-of-values

    """
    assert values.dtype == np.float32 or values.dtype == np.float64
    return bounds['desired']['lower'] + (values - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower'])

def is_normalized(values):
    """Return True if values contained in VALUES are normalized.

    VALUES is a 1D ndarray containing floating-points numbers.

    NOTE: In this function, we assume normalization means min-max feature
    scaling (floats between 0 and 1) and that a zeroed signal is not a
    normalized signal.

    NOTE: VALUES cannot contains IQ (complex numbers) as it doesn't make sense
    to have a normalized signal (assuming 0 and 1) in the cartesian / regular
    representation.

    """
    assert type(values) == np.ndarray
    assert values.ndim == 1
    assert values.dtype == np.float32 or values.dtype == np.float64
    zeroed = values.nonzero()[0].shape == (0,)
    interval = values[values < 0].shape == (0,) and values[values > 1].shape == (0,)
    return not zeroed and interval

def process_iq(sig, amplitude=False, phase=False, norm=False, log=False):
    """Return a processed signal depending on basic parameters.

    By default, all processing are disabled.

    :param sig: Signal to process (np.complex64).

    :param amplitude: If set to True, process and return only the amplitude
    component (np.float32).

    :param phase: If set to True, process and return only the phase component
    (np.float32).

    :param norm: If set to True, normalize the signal.

    :param log: If set to True, log processing to the user.

    :returns: The processed signal in I/Q (np.complex64) if both AMPLITUDE and
    PHASE are False, otherwise the specified component (np.float32).

    """
    if amplitude is True:
        if log is True:
            l.LOGGER.info("Get the amplitude of the processed signal")
        sig = get_comp(sig, CompType.AMPLITUDE)
    elif phase is True:
        if log is True:
            l.LOGGER.info("Get the phase of the processed signal")
        sig = get_comp(sig, CompType.PHASE)
    else:
        if log is True:
            l.LOGGER.info("Keep I/Q of the processed signal")
    # Safety-check between options and nature of signal.
    sig_is_iq = is_iq(sig)
    assert sig_is_iq == (amplitude is False and phase is False)
    # NOTE: Normalize after getting the correct component.
    if norm is True:
        if log is True:
            l.LOGGER.info("Normalize the processed signal")
        sig = normalize(sig, arr_complex=sig_is_iq)
        # If signal was complex before normalization, we must convert the polar
        # representation to cartesian representation before returning.
        if sig_is_iq is True:
            sig = p2r(sig[0], sig[1])
    # Safety-check of signal type.
    if amplitude is False and phase is False:
        assert is_iq(sig) == True, "Bad signal type after processing!"
    else:
        assert is_iq(sig) == False, "Bad signal type after processing!"
    return sig

def truncate_min(arr):
    """Truncate traces to minimum of the array in place.

    Truncate all the traces (1D np.array) contained in ARR (list) to the length
    of the smaller one. Usefull to create a 2D np.array.

    This function work in place, but returns the new array ARR with truncated
    traces for scripting convenience.

    """
    target_len = sys.maxsize
    for s in arr:
        target_len = len(s) if len(s) < target_len else target_len
    for idx, s in enumerate(arr):
        arr[idx] = s[:target_len]
    return arr

def plot_results(config, data, trigger, trigger_average, starts, traces, target_path=None, plot=True, savePlot=False, title="", final=True):
    index_base = 1 if final is False else 2
    plt.subplot(4, 2, index_base)

    t = np.linspace(0,len(data) / config.sampling_rate, len(data))
    plt.plot(t, data)
    plt.title(title)
    plt.xlabel("time [s]")
    plt.ylabel("normalized amplitude")
   
    plt.plot(t, trigger*100)
    plt.axhline(y=trigger_average*100, color='y')
    trace_length = int(config.signal_length * config.sampling_rate)
    for start in starts:
        stop = start + trace_length
        plt.axvline(x=start / config.sampling_rate, color='r', linestyle='--')
        plt.axvline(x=stop / config.sampling_rate, color='g', linestyle='--')

    plt.subplot(4, 2, index_base + 2)
    
    plt.specgram(
        data, NFFT=256, Fs=config.sampling_rate, Fc=0, noverlap=127, cmap=None, xextent=None,
        pad_to=None, sides='default', scale_by_freq=None, mode='default',
        scale='default')
    plt.axhline(y=config.bandpass_lower, color='b', lw=0.2)
    plt.axhline(y=config.bandpass_upper, color='b', lw=0.2)
    plt.title("Spectrogram")
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

    # Don't continue to plot if no encryption was extracted.
    if len(traces) != 0:
        t = np.linspace(0,len(traces[0]) / config.sampling_rate, len(traces[0]))
        plt.subplot(4, 2, index_base + 4)
        for trace in traces:
            plt.plot(t, trace / max(trace))
        plt.title("{} aligned traces".format(len(traces)))
        plt.xlabel("time [s]")
        plt.ylabel("normalized amplitude")

        plt.subplot(4, 2, index_base + 6)
        avg = np.average(traces, axis=0)
        plt.plot(t, avg / max(avg))
        plt.title("Average of {} traces".format(len(traces)))
        plt.xlabel("time [s]")
        plt.ylabel("normalized amplitude")

    if final is True:
        plt.tight_layout()
        plt.subplots_adjust(hspace = 0.5, left = 0.1)

        # NOTE: Fix savefig() layout.
        figure = plt.gcf() # Get current figure
        figure.set_size_inches(32, 18) # Set figure's size manually to your full screen (32x18).

        if savePlot and target_path != None:
            plt.savefig(target_path + "/template.png", dpi=100, bbox_inches='tight')
        if plot:
            plt.show()

        plt.clf()

def find_starts(config, data):
    """
    Find the starts of interesting activity in the signal.

    The result is a list of indices where interesting activity begins, as well
    as the trigger signal and its average.
    """
    
    trigger = butter_bandpass_filter(
        data, config.bandpass_lower, config.bandpass_upper,
        config.sampling_rate, 6)
    
    trigger = np.absolute(trigger)
    # Use an SOS filter because the old one raised exception when using small
    # lowpass values:
    lpf = signal.butter(5, config.lowpass_freq, 'low', fs=config.sampling_rate, output='sos')
    trigger = np.array(signal.sosfilt(lpf, trigger), dtype=trigger.dtype)
    # trigger = butter_lowpass_filter(trigger, config.lowpass_freq,config.sampling_rate, 6)

    # transient = 0.0005
    # start_idx = int(transient * config.sampling_rate)
    start_idx = 0
    average = np.average(trigger[start_idx:])
    maximum = np.max(trigger[start_idx:])
    minimum = np.min(trigger[start_idx:])
    middle = (np.max(trigger[start_idx:]) - min(trigger[start_idx:])) / 2
    if average < 1.1*middle:
        l.LOGGER.debug("Adjusting average to avg + (max - avg) / 2")
        average = average + (maximum - average) / 2
    offset = -int(config.trigger_offset * config.sampling_rate)

    if config.trigger_rising:
        trigger_fn = lambda x, y: x > y
    else:
        trigger_fn = lambda x, y: x < y

    if config.trigger_threshold is not None and config.trigger_threshold > 0:
        l.LOGGER.debug("Use config trigger treshold instead of average!")
        average = config.trigger_threshold / 100 # NOTE: / 100 because of *100 in plot_results().

    # The cryptic numpy code below is equivalent to looping over the signal and
    # recording the indices where the trigger crosses the average value in the
    # direction specified by config.trigger_rising. It is faster than a Python
    # loop by a factor of ~1000, so we trade readability for speed.
    trigger_signal = trigger_fn(trigger, average)[start_idx:]
    starts = np.where((trigger_signal[1:] != trigger_signal[:-1])
                      * trigger_signal[1:])[0] + start_idx + offset + 1
    # if trigger_signal[0]:
    #     starts = np.insert(starts, 0, start_idx + offset)

    return starts, trigger, average

class ExtractConf(config.ModuleConf):
    """Configuration for the extract() function."""

    def __init__(self):
        super().__init__(__name__)

    def load(self, appconf):
        self.signal_length = self.get_dict(appconf)["signal_length"]
        self.sampling_rate = self.get_dict(appconf)["sampling_rate"]
        self.num_traces_per_point = self.get_dict(appconf)["num_traces_per_point"]
        self.num_traces_per_point_min = self.get_dict(appconf)["num_traces_per_point_min"]
        self.min_correlation = self.get_dict(appconf)["min_correlation"]
        self.bandpass_lower = self.get_dict(appconf)["bandpass_lower"]
        self.bandpass_upper = self.get_dict(appconf)["bandpass_upper"]
        self.lowpass_freq = self.get_dict(appconf)["lowpass_freq"]
        self.trigger_offset = self.get_dict(appconf)["trigger_offset"]
        self.trigger_rising = self.get_dict(appconf)["trigger_rising"]
        self.trigger_threshold = self.get_dict(appconf)["trigger_threshold"]
        # NOTE: Allows chaining.
        return self

def extract(data, template, config, average_file_name=None, plot=False, target_path=None, savePlot=False):
    """Post-process an IQ signal to get a clean and well-aligned amplitude and
    phase rotation trace."""
    # Compute needed parameters.
    # Length of a trace.
    trace_length = int(config.signal_length * config.sampling_rate)
    num_traces_per_point = config.num_traces_per_point
    # Sanity-check.
    if len(data) == 0:
        raise Exception("Empty data!")
    if template is not None and len(template) != trace_length:
        raise Exception("Template length doesn't match collection parameters: {} != {}".format(len(template), trace_length))
    # Get signal components.
    data_amp = np.absolute(data)
    data_phr = get_phase_rot(data)

    # Create starts based on trigger frequency.x
    # NOTE: find_starts() will work with the amplitude, but we will use the
    # starts indexes against the raw I/Q.
    trace_starts, trigger, trigger_avg = find_starts(config, data_amp)

    # Extract at trigger + autocorrelate with the template to align.
    traces_amp = [] # Extracted amplitude traces.
    traces_phr = [] # Extracted phase rotation traces.
    skip_nb = 0     # Number of skipped starts.
    corrs = []      # Correlations coefficients stats (during autocorrelation).
    for start_idx, start in enumerate(trace_starts):
        stop = start + trace_length
        # Don't try to extract more traces than configured AES.
        if len(traces_amp) >= num_traces_per_point:
            break
        # Don't try to extract out of the trace index.
        if stop > len(data_amp):
            break
        # Compute current trace candidate.
        trace_amp = data_amp[start:stop]
        # If template is not provided, use the first trace instead.
        if template is None or len(template) == 0:
            l.LOGGER.debug("Use first trace as template!")
            template = trace_amp
            continue
        # Perform the autocorrelation between trace candidate and template.
        trace_lpf    = butter_lowpass_filter(trace_amp, config.sampling_rate / 4, config.sampling_rate)
        template_lpf = butter_lowpass_filter(template, config.sampling_rate  / 4, config.sampling_rate)
        # NOTE: Arbitrary but gives better alignment result.
        correlation = signal.correlate(trace_lpf ** 2, template_lpf ** 2)
        # correlation = signal.correlate(trace_lpf, template_lpf)
        corrs.append(max(correlation))
        # Check correlation if required.
        if config.min_correlation > 0 and max(correlation) < config.min_correlation:
            l.LOGGER.debug("Skip trace start: {} < {}".format(max(correlation), config.min_correlation))
            skip_nb += 1
            continue
        # Save extracted traces.
        shift = np.argmax(correlation) - (len(template) - 1)
        traces_amp.append(data_amp[start + shift : stop + shift])
        traces_phr.append(data_phr[start + shift : stop + shift])

    # Average the extracted traces.
    avg_amp = np.average(traces_amp, axis=0)
    avg_phr = np.average(traces_phr, axis=0)

    # Print the results.
    l.LOGGER.info("num_traces_per_point > starts > extracted > min ; skip_by_corr : {} > {} > {} > {} ; {}".format(num_traces_per_point, len(trace_starts), len(traces_amp), config.num_traces_per_point_min, skip_nb))
    if len(corrs) != 0:
        l.LOGGER.info("percentile(corrs) : 1% / 5% / 10% / 25% : {:.2e} / {:.2e} / {:.2e} / {:.2e}".format(np.percentile(corrs, 1), np.percentile(corrs, 5), np.percentile(corrs, 10), np.percentile(corrs, 25)))

    # Plot the results.
    if plot or savePlot:
        plot_results(config, data_amp, trigger, trigger_avg, trace_starts, traces_amp, target_path, plot, savePlot, "amp", final=False)
        plot_results(config, data_phr, trigger, trigger_avg, trace_starts, traces_phr, target_path, plot, savePlot, "phr", final=True)

    # Check for errors, otherwise, save averaged amplitude trace.
    if (np.shape(avg_amp) == () or np.shape(avg_phr) == ()):
        raise Exception("Trigger or correlation configuration excluded all starts!")
    elif len(traces_amp) < config.num_traces_per_point_min:
        raise Exception("Not enough traces have been averaged: {} < {}".format(len(traces_amp), config.num_traces_per_point_min))
    elif average_file_name:
        np.save(average_file_name, avg_amp)

    return data, avg_amp, avg_phr, template

# * Attack

SMALL_SIZE = 8*4
MEDIUM_SIZE = 10*4
BIGGER_SIZE = 12*4

# Configuration that applies to all attacks; set by the script entry point (cli()).
# Plus other global variables
PLOT = None
SAVE_IMAGES = None
# Number of bytes in the key to attack.
NUM_KEY_BYTES = 16
BRUTEFORCE = None
# Set upper bound to key rank when bruteforcing.
BIT_BOUND_END = 40
PLAINTEXTS = None
KEYS = None
CIPHERTEXTS = None
FIXED_KEY = None
FIXED_PLAINTEXT = None
TRACES = None
KEYFILE = None
VARIABLES = None
VARIABLE_FUNC = None
CLASSES = None
SETS = None
MEANS = None
MEANS_TEST = None
MEANS_PROFILE = None
VARS = None
STDS = None
SNRS = None
TTESTS = None
PTTESTS = None
CORRS = None
SOADS = None
RS = None
RZS = None
PS = None
POIS = None
TRACES_REDUCED = None
TRACES_TEST = None
TRACES_PROFILE = None
PROFILE_RS = None
PROFILE_RZS = None
PROFILE_MEANS = None
PROFILE_COVS = None
PROFILE_STDS = None
PROFILE_MEAN_TRACE = None
LOG_PROBA = None
COMP = None
NUM_TRACES = None
START_POINT = None
END_POINT = None
NORM = None
NORM2 = None

# Per-trace pre-processing:
# 1. z-score normalization
def pre_process(trace, norm):
    if norm:
        mu = np.average(trace)
        std = np.std(trace)
        if std != 0:
            trace = (trace - mu) / std
    return trace

# Load all plaintexts and key(s) from the respective files
def load_all(filename, number=0):
    with open(filename, "r") as f:
        if number == 0:
            data = f.read()
        else:
            data = ''
            for i in range(0, number):
                data += f.readline()
            if data[len(data)-1] == '\n':
                 data = data[0:len(data)-1]
    return [[int(c) for c in bytearray.fromhex(line)]
            for line in data.split('\n')]

# Smart loading of the traces from files
# 1. Discard empty traces (errors occurred during collection)
# 2. Apply pre-processing techniques
def generic_load(data_path,number,wstart=0,wend=0,
                 norm=False, norm2=False, comp="amp"):
    """
    Function that loads plainext, key(s), and (raw) traces.
    """

    empty = 0
    p = load_all(path.join(data_path, 'pt.txt'), number)
    k = load_all(path.join(data_path, 'key.txt'), number)
    fixed_key = False
    if len(k) == 1:
        fixed_key = True
    
    fixed_plaintext = False
    if len(p) == 1:
        fixed_plaintext = True

    plaintexts = []
    keys = []
    traces = []

    for i in range(number):
        # read average or raw traces from file
        raw_traces = np.load(
                path.join(data_path, '%d_%s.npy' % (i, comp))
        )

        if np.shape(raw_traces) == () or not raw_traces.any():
            print("WARN: Empty trace: #{}".format(i))
            empty += 1
            continue

        raw_traces = [raw_traces]

        if wend != 0:
            raw_traces = np.asarray(raw_traces)[:,wstart:wend]

        # iterate over traces
        for trace in raw_traces:
            if trace.all() == 0:
                continue
            trace = pre_process(trace, norm)
            traces.append(trace)
            plaintexts.append(p[i])
            if fixed_key:
                keys.append(k[0])
            else:
                keys.append(k[i])
    if empty > 0:
        l.LOGGER.warn("Number of empty traces: {}".format(empty))
    traces = np.asarray(traces)

    # Apply z-score normalization on the set
    if norm2:
        mu = np.average(traces, axis=0)
        std = np.std(traces, axis=0)
        traces = (traces - mu) / std

    return fixed_key, plaintexts, keys, traces

def attack_global_configure(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp):
    global PLOT, BRUTEFORCE, PLAINTEXTS, TRACES, KEYFILE, DATAPATH
    global KEYS, FIXED_KEY, SAVE_IMAGES, CIPHERTEXTS
    global COMP
    global NUM_TRACES, START_POINT, END_POINT, NORM, NORM2
    SAVE_IMAGES = save_images
    PLOT = plot
    BRUTEFORCE = bruteforce
    KEYFILE = path.join(data_path, 'key.txt')
    DATAPATH = data_path
    COMP = comp
    NUM_TRACES = num_traces
    START_POINT = start_point
    END_POINT = end_point
    NORM = norm
    NORM2 = norm2    
    FIXED_KEY, PLAINTEXTS, KEYS, TRACES = generic_load(
        data_path, num_traces, start_point, end_point, norm, norm2, comp=COMP
    )    
    CIPHERTEXTS = list(map(aes, PLAINTEXTS, KEYS))
    variable_func = None
    PLAINTEXTS = np.asarray(PLAINTEXTS)
    KEYS = np.asarray(KEYS)
    CIPHERTEXTS = np.asarray(CIPHERTEXTS)

# ** CCS18 UTILS (from ChipWhisper)

def cov(x, y):
    # Find the covariance between two 1D lists (x and y).
    # Note that var(x) = cov(x, x)
    return np.cov(x, y)[0][1]
 
hw = [bin(n).count("1") for n in range(256)]

sbox=(
0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16)

def intermediate(pt, keyguess):
    return sbox[pt ^ keyguess]

def hamw(n):
    """Return the Hamming Weight of the number N."""
    # NOTE: Alternative implementation using str built-in class.
    # return bin(n).count("1")
    return int(n).bit_count()

def hamd(n, m):
    """Return the Hamming Distance between numbers N and M."""
    return hamw(n ^ m)

# NOTE: Set "flush=True" for all "print()" calls in this function, otherwise,
# the result using grep and tee in bash script is unreliable.
def print_result(bestguess,knownkey,pge):
    # Hamming distance between all known subkeys and best guess subkeys.
    hd = [hamd(g, k) for g, k in zip(bestguess, knownkey)]

    print("Best Key Guess: ", end=' ', flush=True)
    for b in bestguess: print(" %02x "%b, end=' ')
    print("")
    
    print("Known Key:      ", end=' ', flush=True)
    for b in knownkey: print(" %02x "%b, end=' ')
    print("")
    
    print("PGE:            ", end=' ', flush=True)
    for b in pge: print("%03d "%b, end=' ')
    print("")

    print("HD:             ", end=' ', flush=True)
    for hd_i in hd: print("%03d "% hd_i, end=' ')
    print("")

    print("SUCCESS:        ", end=' ', flush=True)
    tot = 0
    for g,r in list(zip(bestguess,knownkey)):
        if(g==r):
            print("  1 ", end=' ')
            tot += 1
        else:
            print("  0 ", end=' ')
    print("")
    print("NUMBER OF CORRECT BYTES: %d"%tot, flush=True)
    print("HD SUM:                  %d"%np.sum(hd), flush=True)

# ** CHES20 Utils

# Compute the leak variable starting from the plaintext and key
def compute_variables(variable):
    global VARIABLES, CLASSES, VARIABLE_FUNC, FIXED_PLAINTEXT
    VARIABLES = np.zeros((NUM_KEY_BYTES, len(TRACES)), dtype=int)
    FIXED_PLAINTEXT = False
    if variable == "hw_sbox_out":
        CLASSES = list(range(0, 9))
        VARIABLE_FUNC = lambda p, k : hw[sbox[p ^ k]]
    elif variable == "hw_p_xor_k":
        CLASSES = list(range(0, 9))
        VARIABLE_FUNC = lambda p, k : hw[p ^ k]
    elif variable == "sbox_out":
        CLASSES = list(range(0, 256))
        VARIABLE_FUNC = lambda p, k : sbox[p ^ k]
    elif variable == "p_xor_k":
        CLASSES = list(range(0, 256))
        VARIABLE_FUNC = lambda p, k : p ^ k
    elif variable == "p":
        CLASSES = list(range(0, 256))
        VARIABLE_FUNC = lambda p, k : p
        FIXED_PLAINTEXT = True
    elif variable == "hw_p":
        CLASSES = list(range(0, 9))
        VARIABLE_FUNC = lambda p, k : hw[p]
        FIXED_PLAINTEXT = True
    elif variable == "hw_k":
        CLASSES = list(range(0, 9))
        VARIABLE_FUNC = lambda p, k : hw[k]
    elif variable == "k":
        CLASSES = list(range(0, 256))
        VARIABLE_FUNC = lambda p, k : k
    elif variable == "hw_k":
        CLASSES = list(range(0, 9))
        VARIABLE_FUNC = lambda p, k : hw[k]
    elif variable == "hd":
        CLASSES = list(range(0, 7))
        VARIABLE_FUNC = lambda p, k : hw[(p ^ k) ^ sbox[p ^ k]] - 1
    elif variable == "fixed_vs_fixed":
        CLASSES = list(range(0, 2))
        VARIABLE_FUNC = lambda p, k: 1 if p ^ k == 48 else 0
    elif variable == "c":
        CLASSES = list(range(0, 256))
    elif variable == "hw_c":
        CLASSES = list(range(0, 9))
    else:
        raise Exception("Variable type %s is not supported" % variable)

    if variable == "c":
        for bnum in range(NUM_KEY_BYTES):
            VARIABLES[bnum] = CIPHERTEXTS[:,bnum]
    elif variable == "hw_c":
        for bnum in range(NUM_KEY_BYTES):
            VARIABLES[bnum] = [hw[c] for c in CIPHERTEXTS[:,bnum]]
    else:
        for bnum in range(NUM_KEY_BYTES):
            VARIABLES[bnum] = list(map(VARIABLE_FUNC, PLAINTEXTS[:, bnum], KEYS[:, bnum]))

# Classify the traces according to the leak variable
def classify():
    global SETS
    SETS = [[[] for _ in CLASSES] for b in range(NUM_KEY_BYTES)]
    for bnum in range(NUM_KEY_BYTES):
        for cla, trace in list(zip(VARIABLES[bnum], TRACES)):
            SETS[bnum][cla].append(trace)

        SETS[bnum] = [np.array(SETS[bnum][cla]) for cla in CLASSES]

# Estimate mean, variance, and standard deviation for each class, and the
# average trace for all traces
def estimate():
    global MEANS, VARS, STDS
    global PROFILE_MEAN_TRACE

    PROFILE_MEAN_TRACE = np.average(TRACES, axis=0)
    MEANS = np.zeros((NUM_KEY_BYTES, len(CLASSES), len(TRACES[0])))
    VARS = np.zeros((NUM_KEY_BYTES, len(CLASSES), len(TRACES[0])))
    STDS = np.zeros((NUM_KEY_BYTES, len(CLASSES), len(TRACES[0])))
    
    for bnum in range(NUM_KEY_BYTES):
        for cla in CLASSES:
            MEANS[bnum][cla] = np.average(SETS[bnum][cla], axis=0)
            VARS[bnum][cla] = np.var(SETS[bnum][cla], axis=0)
            STDS[bnum][cla] = np.std(SETS[bnum][cla], axis=0)

# Estimate the side-channel SNR
def estimate_snr():
    global SNRS
    SNRS = np.zeros((NUM_KEY_BYTES, len(TRACES[0])))
    for bnum in range(NUM_KEY_BYTES):
        SNRS[bnum] = np.var(MEANS[bnum], axis=0) / np.average(VARS[bnum], axis=0)

# Estimate the t-test
def estimate_ttest():
    global TTESTS, PTTESTS
    TTESTS = np.zeros((NUM_KEY_BYTES, len(TRACES[0])))
    PTTESTS = np.zeros((NUM_KEY_BYTES, len(TRACES[0])))
    for bnum in range(NUM_KEY_BYTES):
        TTESTS[bnum], PTTESTS[bnum] = ttest_ind(SETS[bnum][1],
                SETS[bnum][0], axis=0, equal_var=False)
    
    tmax = np.max(np.absolute(TTESTS[0]))
    p = PTTESTS[0][np.argmax(np.absolute(TTESTS[0]))]
    print("tmax", tmax, "p", p, "p < 0.00001", p < 0.00001)

# Split a set of traces into a k-folds, 1 for test and k-1 for profiling
# fold indicates which of the k-folds is the test set
def split(fold, k_fold):
    global TRACES_TEST, TRACES_PROFILE
    global VARIABLES_TEST, VARIABLES_PROFILE
    Ntraces = len(TRACES)
    Ntest = Ntraces / k_fold
    Nprofiling = Ntraces - Ntest
 
    test_range = [i for i in range(0,Ntraces) if i >= fold*Ntest 
            and i < fold*Ntest + Ntest]
    profiling_range = [i for i in range(0,Ntraces) if i < fold*Ntest 
            or i >= fold*Ntest + Ntest]

    TRACES_TEST = TRACES[test_range]
    TRACES_PROFILE = TRACES[profiling_range]

    VARIABLES_TEST = VARIABLES[:, test_range]
    VARIABLES_PROFILE = VARIABLES[:, profiling_range]

# Classify the profiling set based on the leak variable and estimate the
# average of each class
def classify_and_estimate_profile():
    global MEANS_PROFILE
    MEANS_PROFILE = np.zeros((NUM_KEY_BYTES, len(CLASSES), len(TRACES[0])))
    sets = [[[] for _ in CLASSES] for b in range(NUM_KEY_BYTES)]
    for bnum in range(NUM_KEY_BYTES):
        for cla, trace in list(zip(VARIABLES_PROFILE[bnum], TRACES_PROFILE)):
            sets[bnum][cla].append(trace)

        sets[bnum] = [np.array(sets[bnum][cla]) for cla in CLASSES]
    for bnum in range(NUM_KEY_BYTES):
        for cla in CLASSES:
            MEANS_PROFILE[bnum][cla] = np.average(sets[bnum][cla], axis=0)

# Assign to each test trace the trace estimated with the profiling set for the
# same value of the leak variable
def estimate_test():
    global MEANS_TEST
    MEANS_TEST = np.zeros((NUM_KEY_BYTES, len(TRACES_TEST), len(TRACES[0])))
    for bnum in range(NUM_KEY_BYTES):
        for i, trace in enumerate(TRACES_TEST):
            MEANS_TEST[bnum][i] = MEANS_PROFILE[bnum][VARIABLES_TEST[bnum][i]]

# Estimate the Pearson Correlation Coefficient between the test traces and the
# values predicted by the profile (and also compute the p-value)
def estimate_rf_pf(fold):
    global RF, PF
    try:
        for bnum in range(NUM_KEY_BYTES):
            for i in range(len(TRACES[0])):
                r,p = pearsonr(TRACES_TEST[:, i], MEANS_TEST[bnum][:, i])
                RF[bnum][fold][i] = r
                PF[bnum][fold][i] = p
    except ValueError as e:
        l.LOGGER.error("Cannot compute PCC: {}".format(e))
        raise Exception("Not enough traces to find corelations!")

# Average the results from k different choices of the test set among the k-folds
def average_folds():
    global RS, PS
    RS = np.average(RF, axis=1)
    PS = np.average(PF, axis=1)

# Compute rz as a measure of the significance of the result
def compute_rzs():
    global RZS
    RZS = 0.5*np.log((1+RS)/(1-RS))
    RZS = RZS * math.sqrt(len(TRACES)-3)
 
# Estimate the k-fold r-test
def estimate_r(k_fold):
    global RS, RZS, PS
    global RF, PF
    # RS = np.zeros((NUM_KEY_BYTES, len(TRACES[0])))
    # RZS = np.zeros((NUM_KEY_BYTES, len(TRACES[0])))
    PS = np.zeros((NUM_KEY_BYTES, len(TRACES[0])))

    RF = np.zeros((NUM_KEY_BYTES, k_fold, len(TRACES[0])))
    PF = np.zeros((NUM_KEY_BYTES, k_fold, len(TRACES[0])))
    for fold in range(0,k_fold):
        split(fold, k_fold)
        classify_and_estimate_profile()
        estimate_test()
        estimate_rf_pf(fold)
    average_folds()
    compute_rzs()

# Compute the Sum of Absolute Differences among classes
def soad():
    global SOADS
    SOADS = np.zeros((NUM_KEY_BYTES, len(TRACES[0])))
    for bnum in range(NUM_KEY_BYTES):    
        for i in CLASSES:
            for j in range(i):
                SOADS[bnum] += np.abs(MEANS[bnum][i] - MEANS[bnum][j])

# Estimate the correlation directly between the variables and the traces
# This makes sense, for example, if the leak follows the Hamming Weight model
# and we chose the Hamming Weight model to compute the variables
def estimate_corr():
    global CORRS, PS
    CORRS = np.zeros((NUM_KEY_BYTES, len(TRACES[0])))
    PS = np.zeros((NUM_KEY_BYTES, len(TRACES[0])))
    for bnum in range(NUM_KEY_BYTES):
        for i in range(len(TRACES[0])):
            CORRS[bnum,i], PS[bnum,i] = pearsonr(TRACES[:, i], VARIABLES[bnum])
        print("byte", bnum, "min: ", np.min(CORRS[bnum]),-np.log10(PS[bnum][np.argmin(CORRS[bnum])]))
        print("byte", bnum, "max: ", np.max(CORRS[bnum]),-np.log10(PS[bnum][np.argmax(CORRS[bnum])]))

# Given one among r, t, snr, soad, find the Points of Interest by finding the
# peaks
def find_pois(pois_algo, k_fold, num_pois, poi_spacing, template_dir='.'):
    global POIS
    global SNRS, SOADS
    global RZS, RS
    
    RZS = np.zeros((NUM_KEY_BYTES, len(TRACES[0])))
    RS = np.zeros((NUM_KEY_BYTES, len(TRACES[0])))
    POIS = np.zeros((NUM_KEY_BYTES, num_pois), dtype=int)

    informative = np.zeros((NUM_KEY_BYTES, len(TRACES[0])))
    num_plots = 2
    title = ""
    name = ""
    if pois_algo == "soad":
        soad()
        informative = SOADS
        title = "Sum Of Absolute Differences"
        name = "SOAD"
    elif pois_algo == "snr":
        estimate_snr()
        informative = SNRS
        title = "Signal to Noise Ratio: Var_xk(E(traces))/E_xk(Var(traces))"
        name = "SNR"
    elif pois_algo == "t":
        estimate_ttest()
        informative = TTESTS
        title = "t-test"
        name = "T-TEST"
        num_plots = 3
    elif pois_algo == "r":
        estimate_r(k_fold)
        informative = RS
        num_plots = 4
        title = "%d-folded ro-test: r computed with PCC"%k_fold
        name = "r"
    elif pois_algo == "corr":
        estimate_corr()
        informative = CORRS
        title = "Direct correlation between variables and leaks"
        name = "r(L,Y)"
        num_plots = 3
    else:
        raise Exception("POIs algo type %s is not supported" % pois_algo)

    for bnum in range(NUM_KEY_BYTES):
        temp = np.array(informative[bnum])
        for i in range(num_pois):
            poi = np.argmax(temp)
            POIS[bnum][i] = poi
            
            pmin = max(0, poi - poi_spacing)
            pmax = min(poi + poi_spacing, len(temp))
            for j in range(pmin, pmax):
                temp[j] = 0
    
    if PLOT or SAVE_IMAGES:
        plt.subplots_adjust(hspace = 1) 

        plt.subplot(num_plots, 1, 1)
        #plt.title("average trace (%s, %d traces)"%(DATAPATH,len(TRACES)))
        #plt.title("average trace (%d traces)"%(len(TRACES)))
        plt.xlabel("samples")
        plt.ylabel("normalized\namplitude")
        plt.plot(np.average(TRACES, axis=0))
        
        plt.subplot(num_plots, 1, 2)
        #plt.title(title)
        plt.xlabel("samples")
        plt.ylabel(name)
        for i, snr in enumerate(informative):
            plt.plot(snr, label="subkey %d"%i)
        for bnum in range(NUM_KEY_BYTES):
            plt.plot(POIS[bnum], informative[bnum][POIS[bnum]], '*')

        if pois_algo == "r":
            plt.subplot(num_plots, 1, 3)
            plt.title("%d-folded r-test: r_z = 0.5*ln((1+r)/(1-r)) / (1/sqrt(Ntrace-3))"%k_fold)
            plt.xlabel("samples")
            plt.ylabel("r_z")
            for i, rz in enumerate(RZS):
                plt.plot(rz)
            plt.axhline(y=5, label="5", color='green')
            plt.axhline(y=-5, label="-5", color='green')
            plt.legend(loc='upper right')
 
            plt.subplot(num_plots, 1, 4)
            plt.title("%d-folded ro-test: p computed with PCC"%k_fold)
            plt.xlabel("samples")
            plt.ylabel("-log10(p)")
            for i, p in enumerate(PS):
                plt.plot(-np.log10(p))
            plt.axhline(y=-np.log10(0.05), label="0.05", color='orange')
            plt.axhline(y=-np.log10(0.01), label="0.01", color='green')
            plt.legend(loc='upper right')
        
        elif pois_algo == "t":
            plt.subplot(num_plots, 1, 3)
            plt.title("p")
            plt.xlabel("samples")
            plt.ylabel("-log10(p)")
            for i, p in enumerate(PTTESTS):
                plt.plot(-np.log10(p))
            plt.axhline(y=-np.log10(0.05), label="0.05", color='orange')
            plt.axhline(y=-np.log10(0.01), label="0.01", color='green')
            plt.legend(loc='upper right')
        
        elif pois_algo == "corr":
            plt.subplot(num_plots, 1, 3)
            plt.title("p")
            plt.xlabel("samples")
            plt.ylabel("-log10(p)")
            for i, p in enumerate(PS):
                plt.plot(-np.log10(p))
            plt.axhline(y=-np.log10(0.05), label="0.05", color='orange')
            plt.axhline(y=-np.log10(0.01), label="0.01", color='green')
            plt.legend(loc='upper right')
 
        plt.legend()
        if SAVE_IMAGES:
            # NOTE: Fix savefig() layout.
            figure = plt.gcf() # Get current figure
            figure.set_size_inches(32, 18) # Set figure's size manually to your full screen (32x18).
            plt.savefig(os.path.join(template_dir,'pois.pdf'), bbox_inches='tight', dpi=100)
        if PLOT:
            plt.show()
        plt.clf()

# Once the POIs are known, we can drop all the other points of the traces
# Optionally, instead of taking the peak only, we can take the average of a
# small window areound the peak
def reduce_traces(num_pois, window=0):
    global TRACES_REDUCED
    
    TRACES_REDUCED = np.zeros((NUM_KEY_BYTES, len(TRACES), num_pois))
    for bnum in range(NUM_KEY_BYTES):
        for i, trace in enumerate(TRACES):
            # TRACES_REDUCED[bnum][i] = trace[POIS[bnum,0:num_pois]]
            # find a good reference for the average
            for poi in range(num_pois):
                start = POIS[bnum][poi]-window
                end = POIS[bnum][poi]+window+1   
                TRACES_REDUCED[bnum][i][poi] = np.average(trace[start:end])

# Estimate means, std, and covariance for each possible class
def build_profile(variable, template_dir='.'):
    global PROFILE_MEANS, PROFILE_COVS, PROFILE_STDS

    num_pois = len(POIS[0])
    num_classes = len(CLASSES)

    PROFILE_MEANS = np.zeros((NUM_KEY_BYTES, num_classes, num_pois))
    PROFILE_STDS = np.zeros((NUM_KEY_BYTES, num_classes, num_pois))
    PROFILE_COVS = np.zeros((NUM_KEY_BYTES, num_classes, num_pois, num_pois))

    for bnum in range(NUM_KEY_BYTES):
        for cla in CLASSES:
            for i in range(num_pois):
                PROFILE_MEANS[bnum][cla][i] = MEANS[bnum][cla][POIS[bnum][i]]
                PROFILE_STDS[bnum][cla][i] = STDS[bnum][cla][POIS[bnum][i]]
                for j in range(num_pois):	
                    if(len(SETS[bnum][cla])>0):	
                        PROFILE_COVS[bnum][cla][i][j] = cov(
                                SETS[bnum][cla][:, POIS[bnum][i]],
                                SETS[bnum][cla][:, POIS[bnum][j]])
  
    if PLOT or SAVE_IMAGES:
        for i in range(num_pois):
            for spine in list(plt.gca().spines.values()):
                    spine.set_visible(False)

            #plt.title("Profiles (%s, %d traces, %s variable, %d classes, poi %d)"%(DATAPATH,
            #    len(TRACES), variable, len(CLASSES), i))
            plt.title("Profile")
            plt.xlabel(variable)
            plt.ylabel("normalized amplitude")
 
            #for bnum in range(0,NUM_KEY_BYTES):
            #    for cla in range(0,256):
            #        plt.errorbar(cla, PROFILE_MEANS[bnum][hw[cla], i],
            #                yerr=PROFILE_STDS[bnum][hw[cla], i],
            #                fmt='--o',
            #                label="subkey %d"%bnum)

            for bnum in range(0,NUM_KEY_BYTES):
                plt.errorbar(CLASSES,
                             PROFILE_MEANS[bnum][:, i],
                             yerr=PROFILE_STDS[bnum][:, i],
                             fmt='--o',
                             label="subkey %d"%bnum)
            plt.legend(loc='upper right')
            if SAVE_IMAGES:
                # NOTE: Fix savefig() layout.
                figure = plt.gcf() # Get current figure
                figure.set_size_inches(32, 18) # Set figure's size manually to your full screen (32x18).
                plt.savefig(os.path.join(template_dir,'profile_poi_%d.pdf'%i), bbox_inches='tight', dpi=100)
            if PLOT:
                plt.show()
            plt.clf()

# Find the best (linear) combination of the bits of the leak variable that fits
# the measured traces, compare it with the profile estimated for each possible
# value of the leak variable, and then store it as a profile
def fit(lr_type, variable):
    global PROFILE_BETAS, PROFILE_MEANS_FIT, PROFILE_MEANS
    num_pois = len(POIS[0])

    if lr_type:
        if lr_type == "linear":
            num_betas = 9
            leak_func = lambda x : [(x >> i) & i for i in range(0, num_betas-1)]
        else:
           raise Exception("Linear regression type %s is not supported" %
                    lr_type)
    else:
        return

    PROFILE_BETAS = np.zeros((NUM_KEY_BYTES, num_betas, num_pois))
    for bnum in range(NUM_KEY_BYTES):
        models = list(map(leak_func, VARIABLES[bnum]))
        models = sm.add_constant(models, prepend=False)
        for i in range(num_pois):
            measures = TRACES[:, POIS[bnum][i]]
            params = sm.OLS(measures, models).fit().params
            PROFILE_BETAS[bnum][:, i] = params

    if PLOT:
        for i in range(num_pois):
            for bnum in range(NUM_KEY_BYTES):
                for j in range(0, num_betas - 1):
                    beta = PROFILE_BETAS[bnum][j, i]
                    plt.plot(j, beta, '*')
                    plt.plot([j, j], [0, beta], '-')
            plt.title("Linear regression 8 bits sbox out")
            plt.xlabel("bit")
            plt.ylabel("beta")
            plt.show()

    PROFILE_MEANS_FIT = np.zeros((NUM_KEY_BYTES, len(CLASSES), num_pois))

    for bnum in range(NUM_KEY_BYTES):
        models = list(map(leak_func, CLASSES))
        models = sm.add_constant(models, prepend=False)
        for cla in CLASSES:
            for i in range(num_pois):
                betas = PROFILE_BETAS[bnum][:, i]
                PROFILE_MEANS_FIT[bnum][cla][i] = sum(betas[0:num_betas] *
                        models[cla])
    if PLOT:
        for i in range(num_pois):
            plt.xlabel(variable)
            plt.ylabel("normalized amplitude")
 
            for bnum in range(0,NUM_KEY_BYTES):
                plt.errorbar(CLASSES,
                             PROFILE_MEANS_FIT[bnum][:, i],
                             fmt='--o',
                             label="subkey %d"%bnum)
            plt.legend(loc='upper right')
            plt.show()
 
            plt.xlabel(variable)
            plt.ylabel("normalized amplitude")
            plt.title("profile vs. fit")

            plt.plot(CLASSES,
                         PROFILE_MEANS_FIT[0][:, i],
                         'r-',
                          label="fit")
            plt.errorbar(CLASSES,
                         PROFILE_MEANS[0][:, i],
                         yerr=PROFILE_STDS[bnum][:, i],
                         fmt='g*',
                         label="profile")
            plt.legend(loc='upper right')
            plt.show()

    print("")
    print("Correlation between fit and profile")
    for bnum in range(NUM_KEY_BYTES):
         #print np.corrcoef(PROFILE_MEANS[bnum][:, 0],
         #       PROFILE_MEANS_FIT[bnum][:, 0])[0, 1]
         r,p = pearsonr(PROFILE_MEANS[bnum][:, 0], PROFILE_MEANS_FIT[bnum][:, 0])
         print(r, -10*np.log10(p))

    PROFILE_MEANS = PROFILE_MEANS_FIT
    PROFILE_COVS = None
    
# Store useful information about the profile, to be used for comparing profiles,
# or for profiled correlation and template attacks
def save_profile(template_dir):
    np.save(path.join(template_dir, "POIS.npy"), POIS)
    np.save(path.join(template_dir, "PROFILE_RS.npy"), RS)
    np.save(path.join(template_dir, "PROFILE_RZS.npy"), RZS)
    np.save(path.join(template_dir, "PROFILE_MEANS.npy"), PROFILE_MEANS)
    np.save(path.join(template_dir, "PROFILE_STDS.npy"), PROFILE_STDS)
    np.save(path.join(template_dir, "PROFILE_COVS.npy"), PROFILE_COVS)
    np.save(path.join(template_dir, "PROFILE_MEAN_TRACE.npy"),
            PROFILE_MEAN_TRACE)

# Load the profile, for comparison or for attacks
def load_profile(template_dir):
    global PROFILE_MEANS, PROFILE_COVS, POIS, PROFILE_MEAN_TRACE
    global PROFILE_RS, PROFILE_RZS, PROFILE_STDS
    POIS = np.load(path.join(template_dir, "POIS.npy"))
    PROFILE_RS = np.load(path.join(template_dir, "PROFILE_RS.npy"))
    PROFILE_RZS = np.load(path.join(template_dir, "PROFILE_RZS.npy"))
    PROFILE_MEANS = np.load(path.join(template_dir, "PROFILE_MEANS.npy"))
    PROFILE_COVS = np.load(path.join(template_dir, "PROFILE_COVS.npy"))
    PROFILE_STDS = np.load(path.join(template_dir, "PROFILE_STDS.npy"))
    PROFILE_MEAN_TRACE = np.load(path.join(template_dir, "PROFILE_MEAN_TRACE.npy"))

# Run a template attack or a profiled correlation attack
def run_attack(attack_algo, average_bytes, num_pois, pooled_cov, variable, retmore=False):
    # global PROFILE_MEANS, PROFILE_COVS, POIS
    global LOG_PROBA

    # NOTE: Use np.ndarray to fix memory address misusage.
    # NOTE: Use np.float64 required by HEL (otherwise, segfault).
    LOG_PROBA = np.empty((NUM_KEY_BYTES, 256), dtype=np.float64)

    scores = []
    bestguess = [0]*16
    pge = [256]*16
    
    print("")

    ranking_type = "pearson"
    if attack_algo == "pdf":

        if num_pois > len(PROFILE_COVS[0][0][0]):
            print("Error, there are only %d pois available"%len(PROFILE_COVS[0][0][0]))

        for bnum in range(0, NUM_KEY_BYTES):
            if pooled_cov:
                covs = np.average(PROFILE_COVS[bnum,:,0:num_pois,0:num_pois], axis = 0)
            else:
                covs = PROFILE_COVS[bnum][:,0:num_pois,0:num_pois]

            print("Subkey %2d"%bnum)
            # Running total of log P_k
            P_k = np.zeros(256)
            for j, trace in enumerate(TRACES):
                P_k_tmp = np.zeros(256)
                # Test each key
                for k in range(256):
                    # Find p_{k,j}
                    if FIXED_PLAINTEXT:
                        cla = VARIABLE_FUNC(k, 0)
                    else:
                        cla = VARIABLE_FUNC(PLAINTEXTS[j][bnum], k)
                    if pooled_cov:
                        cov = covs
                    else:
                        cov = covs[cla]
                    
                    rv = multivariate_normal(PROFILE_MEANS[bnum][cla][0:num_pois], cov)
                    p_kj = rv.pdf(TRACES_REDUCED[bnum][j][0:num_pois])

                    # Add it to running total
                    x = np.log(p_kj)
                    if x == -np.inf:
                        # print "inf"
                        continue
                    P_k_tmp[k] += x
                
                P_k += P_k_tmp

                if j % 100 == 0:
                    print(j, "pge ", list(P_k.argsort()[::-1]).index(KEYS[0][bnum]))
            LOG_PROBA[bnum] = P_k
            bestguess[bnum] = P_k.argsort()[-1]
            if FIXED_PLAINTEXT:
                pge[bnum] = list(P_k.argsort()[::-1]).index(PLAINTEXTS[0][bnum])
            else:
                pge[bnum] = list(P_k.argsort()[::-1]).index(KEYS[0][bnum])
            print("PGE ", pge[bnum])
            scores.append(P_k)
    
    elif attack_algo == "pcc":
        cparefs = [None] * NUM_KEY_BYTES
       
        assert len(POIS[0]) >= num_pois, "Requested number of POIs (%d) higher than available (%d)"%(num_pois, len(POIS[0]))

        if average_bytes:
            PROFILE_MEANS_AVG = np.average(PROFILE_MEANS, axis=0)

        # NOTE: Use np.ndarray to fix memory address misusage.
        # NOTE: Use np.float64 required by HEL (otherwise, segfault).
        maxcpa = np.empty((NUM_KEY_BYTES, 256), dtype=np.float64)
        for bnum in range(0, NUM_KEY_BYTES):
            cpaoutput = [0]*256
            print("Subkey %2d"%bnum)
            for kguess in range(256):
                
                clas = [VARIABLE_FUNC(PLAINTEXTS[j][bnum], kguess) for j in
                        range(len(TRACES))]
                if average_bytes:
                    leaks = np.asarray([PROFILE_MEANS_AVG[clas[j]] for j in
                        range(len(TRACES))])
                else:
                    leaks = np.asarray([PROFILE_MEANS[bnum][clas[j]] for j in
                        range(len(TRACES))])
                
                # Combine POIs as proposed in 
                # https://pastel.archives-ouvertes.fr/pastel-00850528/document
                maxcpa[bnum][kguess] = 0
                for i in range(num_pois):
                    r,p = pearsonr(leaks[:, i], TRACES_REDUCED[bnum][:, i])
                    maxcpa[bnum][kguess] += r

                LOG_PROBA[bnum][kguess] = maxcpa[bnum][kguess]
    
            bestguess[bnum] = np.argmax(maxcpa[bnum])
    
            cparefs[bnum] = np.argsort(maxcpa[bnum])[::-1]
    
            #Find PGE
            pge[bnum] = list(cparefs[bnum]).index(KEYS[0][bnum])

    else:
        raise Exception("Attack type not supported: %s"%attack_type)
 
    if FIXED_PLAINTEXT:
        known = PLAINTEXTS[0]
    else:
        known = KEYS[0]

    print_result(bestguess, known, pge)
    if retmore is False:
        return (bestguess == known).all()
    else:
        return maxcpa

# Wrapper to compute AES
def aes(pt, key):
    from Crypto.Cipher import AES

    #_pt = ''.join([chr(c) for c in pt])	
    _pt = b''.join([b.to_bytes(1,"little") for b in pt])	
    #_key = ''.join([chr(c) for c in key])	
    _key = b''.join([b.to_bytes(1,"little") for b in key])	
    cipher = AES.new(_key, AES.MODE_ECB)	
    _ct = cipher.encrypt(_pt)	
    ct = [c for c in _ct]
    
    return ct

# Wrapper to call the Histogram Enumeration Library for key-ranking
def rank():
    # Perform key ranking only if HEL is installed.
    try:
        from python_hel import hel
    except Exception as e:
        l.LOGGER.error("Can't import HEL and perform key ranking!")
        return
    
    print("")
    print("Starting key ranking using HEL")

    import ctypes
    from Crypto.Cipher import AES

    known_key = np.array(KEYS[0], dtype=ctypes.c_ubyte).tolist()

    merge = 2
    bins = 512

    rank_min, rank_rounded, rank_max, time_rank = hel.rank(LOG_PROBA, known_key, merge, bins)

# Wrapper to call the Histogram Enumeration Library for key-enumeration
def bruteforce(bit_bound_end):
    print("")
    print("Starting key enumeration using HEL")
    import ctypes
    from Crypto.Cipher import AES
 
    from python_hel import hel
   
    pt1 = np.array(PLAINTEXTS[0], dtype=ctypes.c_ubyte).tolist()
    pt2 = np.array(PLAINTEXTS[1], dtype=ctypes.c_ubyte).tolist()
 
    print("Assuming that we know two plaintext/ciphertext pairs")
    #_key = ''.join([chr(c) for c in KEYS[0]])	
    __key = KEYS[0].tolist()	
    _key = b''.join([b.to_bytes(1,"little") for b in __key])	
    #_pt1 = ''.join([chr(c) for c in pt1])	
    _pt1 = b''.join([b.to_bytes(1,"little") for b in pt1])	
    #_pt2 = ''.join([chr(c) for c in pt2])	
    _pt2 = b''.join([b.to_bytes(1,"little") for b in pt2])	
 
 
    cipher = AES.new(_key, AES.MODE_ECB)
 
    _ct1 = cipher.encrypt(_pt1)
    _ct2 = cipher.encrypt(_pt2)
    
    #ct1 = [ord(c) for c in _ct1]	
    ct1 = [c for c in _ct1] 
    ct1 = np.array(ct1, dtype=ctypes.c_ubyte)
    #ct2 = [ord(c) for c in _ct2]	
    ct2 = [c for c in _ct2] 
    ct2 = np.array(ct2, dtype=ctypes.c_ubyte)

    merge = 2
    bins = 512
    bit_bound_start = 0
    #bit_bound_end = 30

    found = hel.bruteforce(LOG_PROBA, pt1, pt2, ct1, ct2, merge,
        bins, bit_bound_start, bit_bound_end)


# ** CHES20 Attacks

def profile(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp,
        variable, lr_type, pois_algo, k_fold, num_pois, poi_spacing, pois_dir, align, fs, template_dir):
    attack_global_configure(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp)
    global TRACES

    if pois_dir != "":
        pois = np.load(os.path.join(pois_dir,"POIS.npy"))
        TRACES = TRACES[:,np.sort(pois.flatten())]

    try:
        os.makedirs(template_dir)
    except OSError:
        # Checking the directory before attempting to create it leads to race
        # conditions.
        if not path.isdir(template_dir):
            raise

    if align is True:
        l.LOGGER.info("Align training traces with themselves...")
        TRACES = align_all(TRACES, int(fs), template=TRACES[0], tqdm_log=True)

    compute_variables(variable)
    classify()
    estimate()
    try:
        find_pois(pois_algo, k_fold, num_pois, poi_spacing, template_dir)
    except Exception as e:
        l.LOGGER.error("Cannot find POIs: {}".format(e))
    build_profile(variable, template_dir)
    fit(lr_type, variable)
    save_profile(template_dir)

def attack(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp,
           variable, pois_algo, num_pois, poi_spacing, template_dir, attack_algo, k_fold, average_bytes, pooled_cov, window, align, fs):
    attack_global_configure(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp)
    global TRACES, PROFILE_MEAN_TRACE
    
    if not FIXED_KEY and variable != "hw_p" and variable != "p":
        raise Exception("This set DOES NOT use a FIXED KEY")
 
    load_profile(template_dir)

    if align is True:
        assert fs > 0
        l.LOGGER.info("Align attack traces with themselves...")
        TRACES = align_all(TRACES, int(fs), template=TRACES[0], tqdm_log=True)
        l.LOGGER.info("Align attack traces with the profile...")
        TRACES = align_all(TRACES, int(fs), template=PROFILE_MEAN_TRACE, tqdm_log=True)
    
    if PLOT:
        plt.plot(POIS[:,0], np.average(TRACES, axis=0)[POIS[:,0]], '*')
        plt.plot(np.average(TRACES, axis=0))
        plt.plot(PROFILE_MEAN_TRACE, 'r')
        plt.show()

    compute_variables(variable)
    
    if num_pois == 0:
        num_pois = len(POIS[0])

    if pois_algo != "":
        classify()
        estimate()
        find_pois(pois_algo, num_pois, k_fold, poi_spacing)

    reduce_traces(num_pois, window)
    found = run_attack(attack_algo, average_bytes, num_pois, pooled_cov,
            variable)

    # Always rank if HEL is available.
    rank()

    if BRUTEFORCE and not found:
        bruteforce(BIT_BOUND_END)

# NOTE: Copied from attack() above.
def attack_recombined(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp,
        variable, pois_algo, num_pois, poi_spacing, template_dir, attack_algo, k_fold, average_bytes, pooled_cov, window, align, fs, corr_method):
    attack_global_configure(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp)
    global TRACES, PROFILE_MEAN_TRACE, DATAPATH, COMP, FIXED_KEY, PLAINTEXTS, KEYS, LOG_PROBA

    def attack_comp(comp, template_dir, variable, pois_algo, num_pois,
                    poi_spacing, attack_algo, k_fold,
                    average_bytes, pooled_cov, window, align, fs):
        global TRACES, PROFILE_MEAN_TRACE, DATAPATH, COMP, FIXED_KEY, PLAINTEXTS, KEYS
        print("template_dir={}".format(template_dir))
        print("comp={}".format(comp))
        COMP = comp
        FIXED_KEY, PLAINTEXTS, KEYS, TRACES = generic_load(
            DATAPATH, NUM_TRACES, START_POINT, END_POINT, NORM, NORM2, comp=comp
        )
        CIPHERTEXTS = list(map(aes, PLAINTEXTS, KEYS))
        PLAINTEXTS = np.asarray(PLAINTEXTS)
        KEYS = np.asarray(KEYS)
        CIPHERTEXTS = np.asarray(CIPHERTEXTS)
    
        if not FIXED_KEY and variable != "hw_p" and variable != "p":
            raise Exception("This set DOES NOT use a FIXED KEY")

        load_profile(template_dir)

        if align is True:
            assert fs > 0
            l.LOGGER.info("Align attack traces with themselves...")
            TRACES = align_all(TRACES, int(fs), template=TRACES[0], tqdm_log=True)
            l.LOGGER.info("Align attack traces with the profile...")
            TRACES = align_all(TRACES, int(fs), template=PROFILE_MEAN_TRACE, tqdm_log=True)

        if PLOT:
            plt.plot(POIS[:,0], np.average(TRACES, axis=0)[POIS[:,0]], '*')
            plt.plot(np.average(TRACES, axis=0))
            plt.plot(PROFILE_MEAN_TRACE, 'r')
            plt.show()

        compute_variables(variable)

        if num_pois == 0:
            num_pois = len(POIS[0])

        if pois_algo != "":
            classify()
            estimate()
            find_pois(pois_algo, num_pois, k_fold, poi_spacing)

        reduce_traces(num_pois, window)
        maxcpa = run_attack(attack_algo, average_bytes, num_pois, pooled_cov,
                           variable, retmore=True)

        return maxcpa

    maxcpa = {"amp": None, "phr": None, "recombined": None}

    comp = "amp"
    maxcpa[comp] = attack_comp(comp, template_dir.format(comp), variable, pois_algo, num_pois,
                                poi_spacing, attack_algo, k_fold, average_bytes,
                                pooled_cov, window, align, fs)
    rank()

    comp = "phr"
    maxcpa[comp] = attack_comp(comp, template_dir.format(comp), variable, pois_algo, num_pois,
                                poi_spacing, attack_algo, k_fold, average_bytes,
                                pooled_cov, window, align, fs)
    rank()

    print("comp=recombined_corr ; corr_method={}".format(corr_method))

    bestguess = [0] * 16
    pge = [256] * 16
    cparefs = [None] * NUM_KEY_BYTES
    maxcpa["recombined"] = np.empty_like(maxcpa["amp"])
    for bnum in range(0, NUM_KEY_BYTES):
        for kguess in range(256):
            # NOTE: Combination of correlation coefficient from 2 channels
            # (amplitude and phase rotation) inspired from POI recombination
            # but using addition instead of multiplication.
            if corr_method == "add":
                maxcpa["recombined"][bnum][kguess] = maxcpa["amp"][bnum][kguess] + maxcpa["phr"][bnum][kguess]
            elif corr_method == "mul":
                maxcpa["recombined"][bnum][kguess] = maxcpa["amp"][bnum][kguess] * maxcpa["phr"][bnum][kguess]
            else:
                raise Exception("Invalid corr-method option!")
            LOG_PROBA[bnum][kguess] = np.copy(maxcpa["recombined"][bnum][kguess])
        bestguess[bnum] = np.argmax(maxcpa["recombined"][bnum])
        cparefs[bnum] = np.argsort(maxcpa["recombined"][bnum])[::-1]
        pge[bnum] = list(cparefs[bnum]).index(KEYS[0][bnum])
    known = KEYS[0]

    print_result(bestguess, known, pge)
    rank()

    if BRUTEFORCE and not found:
        bruteforce(BIT_BOUND_END)

def tra_create(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp,
        template_dir, num_pois, poi_spacing):
    attack_global_configure(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp)
    try:
        os.makedirs(template_dir)
    except OSError:
        # Checking the directory before attempting to create it leads to race
        # conditions.
        if not path.isdir(template_dir):
            raise

    if PLOT:
        plt.plot(np.average(TRACES,axis=0),'b')
        plt.show()

    tempKey = KEYS 
    fixed_key = FIXED_KEY 
 
    for knum in range(NUM_KEY_BYTES):
        if(fixed_key):
            tempSbox = [sbox[PLAINTEXTS[i][knum] ^ tempKey[0][knum]] for i in range(len(TRACES))]
        else:
            tempSbox = [sbox[PLAINTEXTS[i][knum] ^ tempKey[i][knum]] for i in range(len(TRACES))]

        tempHW = [hw[s] for s in tempSbox]
        
        # Sort traces by HW
        # Make 9 blank lists - one for each Hamming weight
        tempTracesHW = [[] for _ in range(9)]
        
        # Fill them up
        for i, trace in enumerate(TRACES):
            HW = tempHW[i]
            tempTracesHW[HW].append(trace)

        # Check to have at least a trace for each HW
        for HW in range(9):
            assert len(tempTracesHW[HW]) != 0, "No trace with HW = %d, try increasing the number of traces" % HW

        # Switch to numpy arrays
        tempTracesHW = [np.array(tempTracesHW[HW]) for HW in range(9)]

        # Find averages
        tempMeans = np.zeros((9, len(TRACES[0])))
        for i in range(9):
            tempMeans[i] = np.average(tempTracesHW[i], 0)

        # Find sum of differences
        tempSumDiff = np.zeros(len(TRACES[0]))
        for i in range(9):
            for j in range(i):
                tempSumDiff += np.abs(tempMeans[i] - tempMeans[j])
        
        if PLOT:
            plt.plot(tempSumDiff,label="subkey %d"%knum)
            plt.legend()

        # Find POIs
        POIs = []
        for i in range(num_pois):
            # Find the max
            nextPOI = tempSumDiff.argmax()
            POIs.append(nextPOI)
            
            # Make sure we don't pick a nearby value
            poiMin = max(0, nextPOI - poi_spacing)
            poiMax = min(nextPOI + poi_spacing, len(tempSumDiff))
            for j in range(poiMin, poiMax):
                tempSumDiff[j] = 0

        # Fill up mean and covariance matrix for each HW
        meanMatrix = np.zeros((9, num_pois))
        covMatrix  = np.zeros((9, num_pois, num_pois))
        for HW in range(9):
            for i in range(num_pois):
                # Fill in mean
                meanMatrix[HW][i] = tempMeans[HW][POIs[i]]
                for j in range(num_pois):
                    x = tempTracesHW[HW][:,POIs[i]]
                    y = tempTracesHW[HW][:,POIs[j]]
                    covMatrix[HW,i,j] = cov(x, y)

        with open(path.join(template_dir, 'POIs_%d' % knum), 'wb') as fp:
            pickle.dump(POIs, fp)
        with open(path.join(template_dir, 'covMatrix_%d' % knum), 'wb') as fp:
            pickle.dump(covMatrix, fp)
        with open(path.join(template_dir, 'meanMatrix_%d' % knum), 'wb') as fp:
            pickle.dump(meanMatrix, fp)

    if PLOT:
        plt.show()
    
def tra_attack(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp,
        template_dir):
    attack_global_configure(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp)
    if PLOT:
        plt.plot(np.average(TRACES,axis=0),'b')
        plt.show()
    
    atkKey = KEYS[0] 
    
    scores = []
    bestguess = [0]*16
    pge = [256]*16

    tot = 0
    for knum in range(0,NUM_KEY_BYTES):
        with open(path.join(template_dir, 'POIs_%d' % knum), 'rb') as fp:
            POIs = pickle.load(fp)
        with open(path.join(template_dir, 'covMatrix_%d' % knum), 'rb') as fp:
            covMatrix = pickle.load(fp)
        with open(path.join(template_dir, 'meanMatrix_%d' % knum), 'rb') as fp:
            meanMatrix = pickle.load(fp)

        # Ring buffer for keeping track of the last N best guesses
        window = [None] * 10
        window_index = 0
        
        # Running total of log P_k
        P_k = np.zeros(256)
        for j, trace in enumerate(TRACES):
            # Grab key points and put them in a small matrix
            a = [trace[poi] for poi in POIs]
            
            # Test each key
            for k in range(256):
                # Find HW coming out of sbox
                HW = hw[sbox[PLAINTEXTS[j][knum] ^ k]]
            
                # Find p_{k,j}
                rv = multivariate_normal(meanMatrix[HW], covMatrix[HW])
                p_kj = rv.pdf(a)
           
                # Add it to running total
                P_k[k] += np.log(p_kj)

            guessed = P_k.argsort()[-1]
            window[window_index] = guessed
            window_index = (window_index + 1) % len(window)
            if j % 10 == 1:
                # import os
                # os.system('clear')
                print("PGE ",list(P_k.argsort()[::-1]).index(atkKey[knum]), end=' ')
                # for g in P_k.argsort()[::-1]:
                    # if g == atkKey[knum]:
                        # print '\033[92m%02x\033[0m'%g,
                    # else:
                        # print '%02x'%g,
                print("")
            
            if all(k == atkKey[knum] for k in window) or (j == len(TRACES)-1 and guessed == atkKey[knum]):
                print("subkey %2d found with %4d traces" % (knum, j))
                tot += 1
                break
        else:
            p = list(P_k.argsort()[::-1]).index(atkKey[knum])
            print("subkey %2d NOT found, PGE = %3d" %(knum,p))

        print("")
        bestguess[knum] = P_k.argsort()[-1]
        pge[knum] = list(P_k.argsort()[::-1]).index(atkKey[knum])
        scores.append(P_k)
   
    print_result(bestguess, atkKey, pge)
    # if BRUTEFORCE:
        # brute_force_bitflip(bestguess, atkKey)
    if BRUTEFORCE and not (bestguess == KEYS[0]).all():
        bruteforce(BIT_BOUND_END)

def cra(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp):
    attack_global_configure(data_path, num_traces, start_point, end_point, plot, save_images, bruteforce, norm, norm2, comp)
    global LOG_PROBA
    LOG_PROBA = [[0 for r in range(256)] for bnum in range(NUM_KEY_BYTES)]
    
    if PLOT:
        for t in TRACES:
            plt.plot(t,linewidth=0.5)
        avg = np.average(TRACES, axis=0)
        plt.plot(avg, 'b', linewidth=2, label="average")
        plt.xlabel("samples")
        plt.ylabel("normalized\namplitude")
        plt.legend()
        plt.show()

        # 4: Find sum of differences
        tempSumDiff = np.zeros(np.shape(TRACES)[1])
        for i in range(np.shape(TRACES)[0]-1-5):
            for j in range(i, i+5):
                tempSumDiff += np.abs(TRACES[i] - TRACES[j])
        plt.plot(tempSumDiff)
        plt.show()
    
    knownkey = KEYS[0] 
    numtraces = np.shape(TRACES)[0]-1
    numpoint = np.shape(TRACES)[1]
    
    bestguess = [0]*16
    pge = [256]*16

    stored_cpas = []

    for bnum in range(NUM_KEY_BYTES):
        cpaoutput = [0]*256
        maxcpa = [0]*256
        for kguess in range(256):
            print("Subkey %2d, hyp = %02x: "%(bnum, kguess), end=' ')

            #Initialize arrays and variables to zero
            sumnum = np.zeros(numpoint)
            sumden1 = np.zeros(numpoint)
            sumden2 = np.zeros(numpoint)
    
            hyp = np.zeros(numtraces)
            for tnum in range(numtraces):
                hyp[tnum] = hw[intermediate(PLAINTEXTS[tnum][bnum], kguess)]

            #Mean of hypothesis
            meanh = np.mean(hyp, dtype=np.float64)
    
            #Mean of all points in trace
            meant = np.mean(TRACES, axis=0, dtype=np.float64)
    
            for tnum in range(numtraces):
                hdiff = (hyp[tnum] - meanh)
                tdiff = TRACES[tnum, :] - meant
    
                sumnum = sumnum + (hdiff*tdiff)
                sumden1 = sumden1 + hdiff*hdiff 
                sumden2 = sumden2 + tdiff*tdiff
    
            cpaoutput[kguess] = sumnum / np.sqrt(sumden1 * sumden2)
            maxcpa[kguess] = max(abs(cpaoutput[kguess]))
            LOG_PROBA[bnum][kguess] = maxcpa[kguess]
            print(maxcpa[kguess])
    
        bestguess[bnum] = np.argmax(maxcpa)
    
        cparefs = np.argsort(maxcpa)[::-1]
    
        #Find PGE
        pge[bnum] = list(cparefs).index(knownkey[bnum])
        stored_cpas.append(maxcpa)
    
    print_result(bestguess, knownkey, pge)
    # if BRUTEFORCE:
        # brute_force(stored_cpas, knownkey)
    if BRUTEFORCE and not (bestguess == KEYS[0]).all():
        bruteforce(BIT_BOUND_END)
