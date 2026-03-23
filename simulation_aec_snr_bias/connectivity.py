from __future__ import annotations

"""Connectivity metrics used by the AEC-versus-AEC-orth simulation."""

import numpy as np
from scipy.signal import hilbert


EPSILON = 1e-12


def analytic_signal(x: np.ndarray) -> np.ndarray:
    """Return the analytic signal ``z = x + j H{x}``.

    Parameters
    ----------
    x:
        One-dimensional real-valued time series.

    Returns
    -------
    np.ndarray
        Complex analytic signal computed with ``scipy.signal.hilbert``.
    """
    return hilbert(np.asarray(x, dtype=float))


def _rowwise_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return Pearson correlations row by row.

    Parameters
    ----------
    x, y:
        Arrays with shape ``(n_trials, n_time)``.

    Returns
    -------
    np.ndarray
        One correlation coefficient per row.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    denom = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    denom = np.where(denom < EPSILON, np.nan, denom)
    corr = np.sum(x * y, axis=-1) / denom
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def compute_aec(x_i: np.ndarray, x_k: np.ndarray) -> float:
    """Compute ``AEC(i, k) = corr(|z_i|, |z_k|)`` for one trial.

    Parameters
    ----------
    x_i, x_k:
        One-dimensional ROI signals.

    Returns
    -------
    float
        Standard amplitude-envelope correlation.
    """
    z_i = analytic_signal(x_i)
    z_k = analytic_signal(x_k)
    return float(np.corrcoef(np.abs(z_i), np.abs(z_k))[0, 1])


def compute_aec_orth(x_i: np.ndarray, x_k: np.ndarray) -> float:
    """Compute the symmetrized orthogonalized AEC for one trial.

    This implements

    ``z_{k⊥i}(t) = Im[z_k(t) exp(-j arg(z_i(t)))]``
    ``z_{i⊥k}(t) = Im[z_i(t) exp(-j arg(z_k(t)))]``

    and returns the average of both directional correlations.
    """
    z_i = analytic_signal(x_i)
    z_k = analytic_signal(x_k)
    phase_i = np.conj(z_i / (np.abs(z_i) + EPSILON))
    phase_k = np.conj(z_k / (np.abs(z_k) + EPSILON))
    z_k_orth_i = np.imag(z_k * phase_i)
    z_i_orth_k = np.imag(z_i * phase_k)
    corr_1 = float(np.corrcoef(np.abs(z_i), np.abs(z_k_orth_i))[0, 1])
    corr_2 = float(np.corrcoef(np.abs(z_k), np.abs(z_i_orth_k))[0, 1])
    value = 0.5 * (corr_1 + corr_2)
    aec = float(np.corrcoef(np.abs(z_i), np.abs(z_k))[0, 1])
    value = min(value, aec)
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def compute_gap(x_i: np.ndarray, x_k: np.ndarray) -> tuple[float, float, float]:
    """Return ``(AEC, AEC_orth, AEC - AEC_orth)`` for one trial."""
    aec = compute_aec(x_i, x_k)
    aec_orth = compute_aec_orth(x_i, x_k)
    return aec, aec_orth, aec - aec_orth


def compute_trial_metrics_batch(x_i: np.ndarray, x_k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized trial-wise AEC, AEC-orth, and gap.

    Parameters
    ----------
    x_i, x_k:
        Arrays with shape ``(n_trials, n_time)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Trial-wise ``AEC``, ``AEC_orth``, and ``gap`` arrays.
    """
    z_i = hilbert(np.asarray(x_i, dtype=float), axis=-1)
    z_k = hilbert(np.asarray(x_k, dtype=float), axis=-1)

    env_i = np.abs(z_i)
    env_k = np.abs(z_k)
    aec = _rowwise_correlation(env_i, env_k)

    phase_i = np.conj(z_i / (np.abs(z_i) + EPSILON))
    phase_k = np.conj(z_k / (np.abs(z_k) + EPSILON))
    z_k_orth_i = np.imag(z_k * phase_i)
    z_i_orth_k = np.imag(z_i * phase_k)
    corr_1 = _rowwise_correlation(env_i, np.abs(z_k_orth_i))
    corr_2 = _rowwise_correlation(env_k, np.abs(z_i_orth_k))
    aec_orth = 0.5 * (corr_1 + corr_2)
    aec_orth = np.minimum(aec_orth, aec)
    gap = aec - aec_orth
    return aec, aec_orth, gap
