"""FFT computation utilities for spectrum display.

Pure numpy/scipy — no PyQt or pyqtgraph dependency.
"""

import numpy as np


def compute_fft(audio, sr, max_points=None):
    """Compute the real FFT magnitude spectrum of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        1-D audio samples.
    sr : int or float
        Sample rate in Hz.
    max_points : int, optional
        If set and the spectrum exceeds this many points, decimate for display.

    Returns
    -------
    freqs : np.ndarray
        Frequency bins in Hz (positive half-spectrum only).
    magnitude : np.ndarray
        Absolute FFT magnitude at each frequency bin.
    """
    from scipy.fft import rfft, rfftfreq

    N = len(audio)
    if N == 0 or sr is None or sr == 0:
        return np.array([]), np.array([])

    mag = np.abs(rfft(audio))
    freqs = rfftfreq(N, d=1 / float(sr))

    if max_points is not None and freqs.size > max_points:
        idx = np.linspace(0, freqs.size - 1, max_points).astype(int)
        freqs = freqs[idx]
        mag = mag[idx]

    return freqs, mag
