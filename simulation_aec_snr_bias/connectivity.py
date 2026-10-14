from __future__ import annotations

"""Connectivity metrics used by the AEC-versus-AEC-orth simulation."""

import numpy as np
from scipy.signal import hilbert


# Used by the runner to reject cached results from the former clipped metric.
METRIC_DEFINITION = "symmetric_signed_aec_orth_v2"
CORRELATION_NORM_EPSILON = 1e-20


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
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    numerator = np.sum(x * y, axis=-1)
    # Match the cohort pipeline's per-signal norm threshold, including for
    # small-amplitude signals. Undefined correlations are represented as zero.
    corr = np.divide(
        numerator,
        norm_x * norm_y,
        out=np.zeros_like(numerator),
        where=(norm_x >= CORRELATION_NORM_EPSILON) & (norm_y >= CORRELATION_NORM_EPSILON),
    )
    return np.clip(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


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
    return float(_rowwise_correlation(np.abs(z_i), np.abs(z_k)))


def compute_aec_orth(x_i: np.ndarray, x_k: np.ndarray) -> float:
    """Compute the symmetrized orthogonalized AEC for one trial.

    This implements

    ``z_{k⊥i}(t) = Im[z_k(t) exp(-j arg(z_i(t)))]``
    ``z_{i⊥k}(t) = Im[z_i(t) exp(-j arg(z_k(t)))]``

    and returns the average of both directional correlations, using the same
    definition as ``meg_alzheimer.connectivity.aec_orth_from_analytic``.
    The result is not constrained to be less than or equal to AEC.
    """
    return compute_gap(x_i, x_k)[1]


def compute_gap(x_i: np.ndarray, x_k: np.ndarray) -> tuple[float, float, float]:
    """Return ``(AEC, AEC_orth, AEC - AEC_orth)`` for one trial."""
    metrics = compute_trial_metrics_batch(
        np.asarray(x_i, dtype=float)[None, :],
        np.asarray(x_k, dtype=float)[None, :],
    )
    return tuple(float(values[0]) for values in metrics)


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

    # Use phase rotations as in the cohort pipeline. Adding an absolute
    # epsilon to the amplitude would change the projection with signal scale.
    phase_i = np.exp(-1j * np.angle(z_i))
    phase_k = np.exp(-1j * np.angle(z_k))
    z_k_orth_i = np.imag(z_k * phase_i)
    z_i_orth_k = np.imag(z_i * phase_k)
    corr_1 = _rowwise_correlation(env_i, np.abs(z_k_orth_i))
    corr_2 = _rowwise_correlation(env_k, np.abs(z_i_orth_k))
    aec_orth = 0.5 * (corr_1 + corr_2)
    # Preserve signed correlations and negative gaps, as in the cohort code.
    gap = aec - aec_orth
    return aec, aec_orth, gap
