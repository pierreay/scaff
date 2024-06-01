import sys
from enum import Enum

import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
from tqdm import tqdm

from scaff import log as l

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
    template_lpf = filters.butter_lowpass_filter(get_amplitude(template), lpf_freq, sr)
    target_lpf   = filters.butter_lowpass_filter(get_amplitude(target), lpf_freq, sr)
    if normalize is True:
        template_lpf = analyze.normalize(template_lpf)
        target_lpf = analyze.normalize(target_lpf)
    shift        = analyze.get_shift_corr(target_lpf, template_lpf)
    if get_shift_only is True:
        return shift
    # Log and check shift value.
    if log:
        l.LOGGER.debug("Shift to maximize cross correlation: {}".format(shift))
    if not ignore:
        assert np.abs(shift) < len(template/10), "shift is too high, inspect"
    # Apply shift on the raw target signal.
    return analyze.shift(target, shift)

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
