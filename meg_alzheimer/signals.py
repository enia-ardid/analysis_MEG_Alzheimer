from __future__ import annotations

"""Signal-processing helpers used by the MEG connectivity pipeline.

The raw Brainstorm exports already live in source space and have been broadly
filtered upstream, but the connectivity metrics in this repository are defined
band-by-band. The functions below therefore implement the small set of signal
operations needed to move from trial-wise time series to band-limited analytic
signals:

- define the canonical frequency bands used throughout the project
- design a linear-phase FIR band-pass filter
- apply zero-phase filtering with ``filtfilt``
- compute the analytic signal with a Hilbert transform
- expose phase and envelope, which are the primitives required by PLV, AEC,
  and AEC-orth
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import filtfilt, firwin, hilbert


Band = Tuple[float, float]


def band_defs() -> Dict[str, Band]:
    """Return the canonical frequency bands used across the repository.

    These ranges are intentionally centralized in one function so that the
    cohort pipeline, the subject-level outputs, and the final hypothesis tests
    all operate on the same spectral definitions.
    """
    return {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 48.0),
    }


def design_fir_bandpass(
    fs: float,
    f_lo: float,
    f_hi: float,
    numtaps: int = 801,
    transition: float = 0.15,
) -> np.ndarray:
    """Design a Hamming-window FIR band-pass filter.

    Parameters
    ----------
    fs:
        Sampling rate in Hz.
    f_lo, f_hi:
        Target passband edges in Hz.
    numtaps:
        Number of FIR coefficients. A larger value yields a sharper filter at
        the cost of more computation and longer edge transients.
    transition:
        Fraction of the band width used to extend the passband into a
        transition region on both sides. This softens the response and avoids
        unrealistically sharp cutoffs.
    """
    nyq = fs / 2.0
    bw = f_hi - f_lo
    lo = max(0.001, f_lo - transition * bw)
    hi = min(nyq - 0.001, f_hi + transition * bw)
    return firwin(numtaps, [lo / nyq, hi / nyq], pass_zero=False, window="hamming")


def bandpass_filt(
    X: np.ndarray,
    fs: float,
    band: Band,
    numtaps: int = 801,
    transition: float = 0.15,
    edge_trim: int | None = None,
) -> np.ndarray:
    """Apply zero-phase FIR filtering along the last axis.

    The filter is run with ``filtfilt`` so that the phase response is canceled.
    This is important here because both phase-based and envelope-based metrics
    are sensitive to timing distortions.

    ``edge_trim`` is optional because some workflows already keep extra padding
    around the segment of interest. In this repository the main pipeline uses
    the 2-second margins stored in the dataset and crops the center window
    before this function is called, so explicit trimming is usually not needed.
    """
    X = np.asarray(X)
    taps = design_fir_bandpass(fs, band[0], band[1], numtaps=numtaps, transition=transition)
    Y = filtfilt(taps, [1.0], X, axis=-1, padtype="odd")
    if edge_trim is not None and edge_trim > 0:
        slicer = [slice(None)] * Y.ndim
        slicer[-1] = slice(edge_trim, -edge_trim)
        Y = Y[tuple(slicer)]
    return Y


def analytic_signal(X: np.ndarray) -> np.ndarray:
    """Return the analytic signal along the time axis."""
    return hilbert(X, axis=-1)


def phase_and_envelope(X_band: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a band-limited signal into instantaneous phase and amplitude."""
    Xa = analytic_signal(X_band)
    return np.angle(Xa), np.abs(Xa)


def window_indices(
    n_time: int,
    fs: float,
    win_length_s: float = 4.0,
    step_s: float = 2.0,
) -> List[Tuple[int, int]]:
    """Generate start/stop indices for optional sliding-window analyses."""
    length = int(round(win_length_s * fs))
    step = int(round(step_s * fs))
    idx: List[Tuple[int, int]] = []
    start = 0
    while start + length <= n_time:
        idx.append((start, start + length))
        start += step
    return idx


def demean_detrend(X: np.ndarray) -> np.ndarray:
    """Remove the mean from each row.

    The name is intentionally conservative here: the current implementation
    performs demeaning, not linear detrending. It is kept as a small helper for
    cases where envelope correlations should be computed after mean removal.
    """
    X = np.asarray(X, dtype=float)
    return X - X.mean(axis=-1, keepdims=True)
