import sys
from enum import Enum

import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    t = np.linspace(0,len(data) / config["soapyrx"]["sampling_rate"], len(data))
    plt.plot(t, data)
    plt.title(title)
    plt.xlabel("time [s]")
    plt.ylabel("normalized amplitude")
   
    plt.plot(t, trigger*100)
    plt.axhline(y=trigger_average*100, color='y')
    trace_length = int(config["scaff"]["signal_length"] * config["soapyrx"]["sampling_rate"])
    for start in starts:
        stop = start + trace_length
        plt.axvline(x=start / config["soapyrx"]["sampling_rate"], color='r', linestyle='--')
        plt.axvline(x=stop / config["soapyrx"]["sampling_rate"], color='g', linestyle='--')

    plt.subplot(4, 2, index_base + 2)
    
    plt.specgram(
        data, NFFT=256, Fs=config["soapyrx"]["sampling_rate"], Fc=0, noverlap=127, cmap=None, xextent=None,
        pad_to=None, sides='default', scale_by_freq=None, mode='default',
        scale='default')
    plt.axhline(y=config["scaff"]["bandpass_lower"], color='b', lw=0.2)
    plt.axhline(y=config["scaff"]["bandpass_upper"], color='b', lw=0.2)
    plt.title("Spectrogram")
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

    if(len(traces) == 0):
        l.LOGGER.warn("No encryption was extracted!")
    else:
        t = np.linspace(0,len(traces[0]) / config["soapyrx"]["sampling_rate"], len(traces[0]))
        plt.subplot(4, 2, index_base + 4)
        for trace in traces:
            plt.plot(t, trace / max(trace))
        plt.title("{} aligned traces".format(len(starts)))
        plt.xlabel("time [s]")
        plt.ylabel("normalized amplitude")

        plt.subplot(4, 2, index_base + 6)
        avg = np.average(traces, axis=0)
        plt.plot(t, avg / max(avg))
        plt.title("Average of {} traces".format(len(starts)))
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

def find_starts(config, data, target_path, index):
    """
    Find the starts of interesting activity in the signal.

    The result is a list of indices where interesting activity begins, as well
    as the trigger signal and its average.
    """
    
    trigger = butter_bandpass_filter(
        data, config["scaff"]["bandpass_lower"], config["scaff"]["bandpass_upper"],
        config["soapyrx"]["sampling_rate"], 6)
    
    #TOM ADDITION START
    #plt.clf()
    #plt.plot(trigger)
    #plt.savefig(target_path+"/"+str(index)+"_4-trigger-bandpass.png")
    #TOM ADDITION END
        
    trigger = np.absolute(trigger)
    # Use an SOS filter because the old one raised exception when using small
    # lowpass values:
    lpf = signal.butter(5, config["scaff"]["lowpass_freq"], 'low', fs=config["soapyrx"]["sampling_rate"], output='sos')
    trigger = np.array(signal.sosfilt(lpf, trigger), dtype=trigger.dtype)
    # trigger = butter_lowpass_filter(trigger, config["scaff"]["lowpass_freq"],config["soapyrx"]["sampling_rate"], 6)

    #TOM ADDITION START
    #plt.clf()
    #plt.plot(trigger)
    #plt.savefig(target_path+"/"+str(index)+"_5-trigger-lowpass.png")
    #TOM ADDITION END

    # transient = 0.0005
    # start_idx = int(transient * config["soapyrx"]["sampling_rate"])
    start_idx = 0
    average = np.average(trigger[start_idx:])
    maximum = np.max(trigger[start_idx:])
    minimum = np.min(trigger[start_idx:])
    middle = (np.max(trigger[start_idx:]) - min(trigger[start_idx:])) / 2
    if average < 1.1*middle:
        l.LOGGER.info("Adjusting average to avg + (max - avg) / 2")
        average = average + (maximum - average) / 2
    offset = -int(config["scaff"]["trigger_offset"] * config["soapyrx"]["sampling_rate"])

    if config["scaff"]["trigger_rising"]:
        trigger_fn = lambda x, y: x > y
    else:
        trigger_fn = lambda x, y: x < y

    if config["scaff"]["trigger_threshold"] is not None and config["scaff"]["trigger_threshold"] > 0:
        l.LOGGER.info("Use config trigger treshold instead of average")
        average = config["scaff"]["trigger_threshold"] / 100 # NOTE: / 100 because of *100 in plot_results().

    # The cryptic numpy code below is equivalent to looping over the signal and
    # recording the indices where the trigger crosses the average value in the
    # direction specified by config["scaff"]["trigger_rising"]. It is faster than a Python
    # loop by a factor of ~1000, so we trade readability for speed.
    trigger_signal = trigger_fn(trigger, average)[start_idx:]
    starts = np.where((trigger_signal[1:] != trigger_signal[:-1])
                      * trigger_signal[1:])[0] + start_idx + offset + 1
    # if trigger_signal[0]:
    #     starts = np.insert(starts, 0, start_idx + offset)

    #TOM ADDITION START
    #plt.clf()
    #plt.plot(trigger_signal)
    #plt.savefig(target_path+"/"+str(index)+"_6-triggerstart.png")
    #TOM ADDITION END


    # plt.plot(data)
    # plt.plot(trigger*100)
    # plt.axhline(y=average*100)
    # plt.show()

    return starts, trigger, average

# The part that uses a frequency component as trigger was initially
# inspired by https://github.com/bolek42/rsa-sdr
# The code below contains a few hacks to deal with all possible errors we
# encountered with different radios and setups. It is not very clean but it is
# quite stable.
def extract(data, config, average_file_name=None, plot=False, target_path=None, savePlot=False, index=0):
    """Post-process a radio capture to get a clean and well-aligned trace.

    """
    if len(data) == 0:
        raise Exception("Empty data!")

    template = np.load(config["scaff"]["template_name"]) if config["scaff"]["template_name"] else None

    if template is not None and len(template) != int(
            config["scaff"]["signal_length"] * config["soapyrx"]["sampling_rate"]):
        l.LOGGER.warn("Template length doesn't match collection parameters. "
              "Is this the right template?")

    # cut usless transient
    data = data[int(config["scaff"]["drop_start"] * config["soapyrx"]["sampling_rate"]):]

    # assert len(data) != 0, "ERROR, empty data after drop_start"
    if len(data) == 0:
        raise Exception("Empty data after drop start!")

    # AMPlitude
    data_amp = np.absolute(data)
    # PHase Rotation
    data_phr = get_phase_rot(data)

    #TOM ADDITION START
    #plt.clf()
    #plt.plot(data)
    #plt.savefig(target_path+"/"+str(index)+"_3-data-absolute.png")
    #TOM ADDITION END
    #
    # extract/aling trace with trigger frequency + autocorrelation
    #
    # NOTE: find_starts() will work with the amplitude, but we will use the
    # starts indexes against the raw I/Q.
    trace_starts, trigger, trigger_avg = find_starts(config, data_amp, target_path, index)

    # extract at trigger + autocorrelate with the first to align
    traces_amp = []
    traces_phr = []
    trace_length = int(config["scaff"]["signal_length"] * config["soapyrx"]["sampling_rate"])
    l.LOGGER.info("Number of starts: {}".format(len(trace_starts)))
    for start_idx, start in enumerate(trace_starts):
        if len(traces_amp) >= min(config["fw"]["num_traces_per_point"], config["fw"]["num_traces_per_point_min"]):
            break

        stop = start + trace_length

        if stop > len(data_amp):
            break

        trace_amp = data_amp[start:stop]
        if template is None or len(template) == 0:
            template = trace_amp
            continue

        trace_lpf = butter_lowpass_filter(trace_amp, config["soapyrx"]["sampling_rate"] / 4,
                config["soapyrx"]["sampling_rate"])
        template_lpf = butter_lowpass_filter(template, config["soapyrx"]["sampling_rate"] / 4,
                config["soapyrx"]["sampling_rate"])
        correlation = signal.correlate(trace_lpf**2, template_lpf**2)
        l.LOGGER.debug("corr={}".format(max(correlation)))
        if max(correlation) <= config["scaff"]["min_correlation"]:
            l.LOGGER.warn("WARN: Skip trace start: corr={} <= corr_min={}".format(max(correlation), config["scaff"]["min_correlation"]))
            continue

        shift = np.argmax(correlation) - (len(template)-1)
        traces_amp.append(data_amp[start+shift:stop+shift])
        traces_phr.append(data_phr[start+shift:stop+shift])

    avg_amp = np.average(traces_amp, axis=0)
    avg_phr = np.average(traces_phr, axis=0)

    if plot or savePlot:
        plot_results(config, data_amp, trigger, trigger_avg, trace_starts, traces_amp, target_path, plot, savePlot, "amp", final=False)
        plot_results(config, data_phr, trigger, trigger_avg, trace_starts, traces_phr, target_path, plot, savePlot, "phr", final=True)

    if (np.shape(avg_amp) == () or np.shape(avg_phr) == ()):
        raise Exception("Trigger or correlation configuration excluded all starts!")
    elif average_file_name:
        np.save(average_file_name, avg_amp)

    std = np.std(traces_amp,axis=0)
    l.LOGGER.info("Extraction summary: ")
    l.LOGGER.info("Number = {}".format(len(traces_amp)))
    l.LOGGER.info("avg[Max(std)] = {:.2E}".format(avg_amp[std.argmax()]))
    l.LOGGER.info("Max(u) = Max(std) = {:.2E}".format(max(std)))
    l.LOGGER.info("Max(u_rel) = {:.2E} percentage".format(100*max(std)/avg_amp[std.argmax()]))

    return data, avg_amp, avg_phr
