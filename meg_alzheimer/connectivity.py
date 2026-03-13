from __future__ import annotations

"""Connectivity metrics used in the project.

The pipeline computes three matrices from band-limited ROI time series:

- PLV: phase-locking value
- AEC: amplitude-envelope correlation
- AEC-orth: orthogonalized amplitude-envelope correlation

All public functions in this module accept either a single ROI time series or
an array shaped ``n_rois x n_time`` and return a symmetric ROI-by-ROI matrix.
"""

from typing import Optional

import numpy as np
from scipy.signal import hilbert

from .signals import bandpass_filt, phase_and_envelope, window_indices


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    """Promote a 1D signal to ``1 x time`` so downstream code stays uniform."""
    X = np.asarray(X)
    if X.ndim == 1:
        return X[None, :]
    return X


def _corrcoef_rows(X: np.ndarray) -> np.ndarray:
    """Compute a row-wise correlation matrix without calling ``np.corrcoef``.

    The explicit implementation makes it easier to guard against nearly constant
    rows and to force any undefined values back to zero.
    """
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=-1, keepdims=True)
    denom = np.linalg.norm(X, axis=-1)
    denom = np.where(denom < 1e-20, np.nan, denom)
    C = (X @ X.T) / np.outer(denom, denom)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(C, -1.0, 1.0)


def _corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Correlation between two one-dimensional signals with safe zero checks."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    norm_x = float(np.linalg.norm(x))
    norm_y = float(np.linalg.norm(y))
    if norm_x < 1e-20 or norm_y < 1e-20:
        return 0.0
    return float(np.dot(x, y) / (norm_x * norm_y))


def plv_from_phase(
    phase: np.ndarray,
    fs: float,
    window_s: Optional[float] = None,
    step_s: Optional[float] = None,
) -> np.ndarray:
    """Compute PLV from instantaneous phase.

    If ``window_s`` is ``None`` the PLV is computed on the full trial. If a
    window is provided, PLV is computed within each window and then averaged,
    which can be useful for time-resolved summaries.
    """
    phase = _ensure_2d(phase)
    n_roi, n_time = phase.shape
    if window_s is None:
        phase_complex = np.exp(1j * phase)
        C = np.abs(phase_complex @ phase_complex.conj().T) / n_time
    else:
        idx = window_indices(n_time, fs, win_length_s=window_s, step_s=step_s or (window_s / 2.0))
        acc = np.zeros((n_roi, n_roi), dtype=float)
        for start, stop in idx:
            phase_complex = np.exp(1j * phase[:, start:stop])
            acc += np.abs(phase_complex @ phase_complex.conj().T) / phase_complex.shape[1]
        C = acc / max(1, len(idx))
    np.fill_diagonal(C, 0.0)
    return 0.5 * (C + C.T)


def aec_from_envelope(
    env: np.ndarray,
    fs: float,
    window_s: Optional[float] = None,
    step_s: Optional[float] = None,
) -> np.ndarray:
    """Compute envelope correlation from amplitude envelopes."""
    env = _ensure_2d(env)
    if window_s is None:
        C = _corrcoef_rows(env)
    else:
        idx = window_indices(env.shape[-1], fs, win_length_s=window_s, step_s=step_s or (window_s / 2.0))
        acc = np.zeros((env.shape[0], env.shape[0]), dtype=float)
        for start, stop in idx:
            acc += _corrcoef_rows(env[:, start:stop])
        C = acc / max(1, len(idx))
    np.fill_diagonal(C, 0.0)
    return np.clip(0.5 * (C + C.T), -1.0, 1.0)


def plv_matrix(
    X: np.ndarray,
    fs: float,
    band: tuple[float, float],
    edge_trim: Optional[int] = None,
    window_s: Optional[float] = None,
    step_s: Optional[float] = None,
) -> np.ndarray:
    """Band-pass a signal, extract phase, and return a PLV matrix."""
    X = _ensure_2d(X)
    Xb = bandpass_filt(X, fs, band, edge_trim=edge_trim)
    phase, _ = phase_and_envelope(Xb)
    return plv_from_phase(phase, fs=fs, window_s=window_s, step_s=step_s)


def aec_matrix(
    X: np.ndarray,
    fs: float,
    band: tuple[float, float],
    edge_trim: Optional[int] = None,
    window_s: Optional[float] = None,
    step_s: Optional[float] = None,
    demean: bool = True,
) -> np.ndarray:
    """Band-pass a signal, extract envelopes, and return an AEC matrix."""
    X = _ensure_2d(X)
    Xb = bandpass_filt(X, fs, band, edge_trim=edge_trim)
    _, env = phase_and_envelope(Xb)
    if demean:
        env = env - env.mean(axis=-1, keepdims=True)
    return aec_from_envelope(env, fs=fs, window_s=window_s, step_s=step_s)


def _orth_component(xa: np.ndarray, ya: np.ndarray) -> np.ndarray:
    """Project one analytic signal onto the imaginary axis of the other."""
    return np.imag(ya * np.exp(-1j * np.angle(xa)))


def aec_orth_matrix(
    X: np.ndarray,
    fs: float,
    band: tuple[float, float],
    edge_trim: Optional[int] = None,
    window_s: Optional[float] = None,
    step_s: Optional[float] = None,
    demean: bool = True,
    symmetric: bool = True,
) -> np.ndarray:
    """Compute orthogonalized envelope correlation directly from raw signals.

    The public API mirrors ``aec_matrix`` for convenience, but the internal
    implementation immediately switches to analytic signals because the
    orthogonalization step operates in the complex plane.
    """
    if window_s is not None:
        raise NotImplementedError("Windowed AEC-orth is not implemented yet.")

    X = _ensure_2d(X)
    Xb = bandpass_filt(X, fs, band, edge_trim=edge_trim)
    Xa = hilbert(Xb, axis=-1)
    return aec_orth_from_analytic(Xa, demean=demean, symmetric=symmetric)


def aec_orth_from_analytic(
    Xa: np.ndarray,
    demean: bool = True,
    symmetric: bool = True,
) -> np.ndarray:
    """Compute symmetric AEC-orth from an analytic signal matrix.

    For each pair of ROIs we orthogonalize each signal with respect to the
    other, correlate the original envelope with the orthogonalized envelope, and
    average both directions. This symmetric version is the quantity used
    throughout the repository.
    """
    Xa = _ensure_2d(Xa)
    env = np.abs(Xa)
    if demean:
        env = env - env.mean(axis=-1, keepdims=True)

    n_roi = Xa.shape[0]
    C = np.zeros((n_roi, n_roi), dtype=float)
    for i in range(n_roi):
        for j in range(i + 1, n_roi):
            # Orthogonalize in both directions so the final matrix is not tied
            # to an arbitrary source/target ordering.
            yi_perp = _orth_component(Xa[i], Xa[j])
            xi_perp = _orth_component(Xa[j], Xa[i])
            rij1 = _corr_1d(env[i], np.abs(yi_perp))
            rij2 = _corr_1d(env[j], np.abs(xi_perp))
            Cij = 0.5 * (rij1 + rij2) if symmetric else rij1
            if not np.isfinite(Cij):
                Cij = 0.0
            C[i, j] = C[j, i] = Cij
    np.fill_diagonal(C, 0.0)
    return np.clip(C, -1.0, 1.0)
