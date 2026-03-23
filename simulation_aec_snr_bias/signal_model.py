from __future__ import annotations

"""Signal model used by the AEC/AEC-orth SNR bias simulation.

The original zero-lag shared-source model is useful for illustrating spatial
leakage, but it collapses ``AEC_orth`` toward zero even at very high SNR. That
is not the question we need to answer here. The simulation instead models
*true envelope coupling* between two ROIs carried by the same narrowband alpha
process with a fixed non-zero phase lag. Under high SNR both ``AEC`` and
``AEC_orth`` converge to the same envelope-correlation target, while added
measurement noise selectively perturbs the phase estimate used by
orthogonalization.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import filtfilt, firwin, freqz, hilbert

from .connectivity import compute_trial_metrics_batch


ALPHA_BAND = (8.0, 12.0)
PADDING_SAMPLES = 2000
ENVELOPE_CUTOFF_HZ = 2.0
ENVELOPE_SCALE = 0.30
PHASE_LAG_RAD = 1.00


@dataclass(frozen=True)
class SubjectMetrics:
    """Subject-level averages of the simulated connectivity quantities."""

    aec: float
    aec_orth: float
    gap: float


def make_alpha_filter(fs: float, order: int = 801) -> np.ndarray:
    """Return the FIR alpha-band filter used in the simulation.

    The filter is designed with ``scipy.signal.firwin`` using a Hamming window
    and nominal band edges at 8-12 Hz. The function validates that the -3 dB
    transition points remain within 0.5 Hz of the nominal edges.
    """
    # firwin cutoffs are transition midpoints. Using 7-13 Hz yields -3 dB
    # crossings close to the nominal 8-12 Hz alpha band while keeping a flatter
    # central plateau inside the useful band.
    taps = firwin(order, [7.0, 13.0], fs=fs, pass_zero=False, window="hamming")
    frequencies, response = freqz(taps, worN=32768, fs=fs)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
    lower_idx = np.where(frequencies < ALPHA_BAND[0])[0]
    upper_idx = np.where(frequencies > ALPHA_BAND[1])[0]
    lower_edge = frequencies[lower_idx[np.argmin(np.abs(magnitude_db[lower_idx] + 3.0))]]
    upper_edge = frequencies[upper_idx[np.argmin(np.abs(magnitude_db[upper_idx] + 3.0))]]
    if abs(lower_edge - ALPHA_BAND[0]) > 0.5 or abs(upper_edge - ALPHA_BAND[1]) > 0.5:
        raise ValueError("The alpha FIR filter does not meet the requested -3 dB edge tolerance.")
    return taps


def _make_envelope_filter(fs: float, order: int = 801) -> np.ndarray:
    """Return the low-pass FIR filter used to generate slow envelope dynamics.

    The envelope process is restricted to frequencies below 2 Hz so that the
    analytic-envelope estimate remains close to the underlying amplitude
    modulation, consistent with Bedrosian's regime.
    """
    return firwin(order, ENVELOPE_CUTOFF_HZ, fs=fs, pass_zero="lowpass", window="hamming")


def _standardize_rows(x: np.ndarray) -> np.ndarray:
    """Return zero-mean, unit-variance rows."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=-1, keepdims=True)
    scale = x.std(axis=-1, keepdims=True)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return x / scale


def _band_limited_noise(
    n_trials: int,
    n_samples: int,
    fs: float,
    taps: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate band-limited Gaussian noise trials by filtering wideband noise."""
    wideband = rng.standard_normal((n_trials, n_samples + 2 * PADDING_SAMPLES))
    filtered = filtfilt(taps, [1.0], wideband, axis=-1, padtype="odd")
    cropped = filtered[:, PADDING_SAMPLES : PADDING_SAMPLES + n_samples]
    return _standardize_rows(cropped)


def _make_envelopes(
    rho: float,
    n_trials: int,
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
    taps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return correlated positive envelope processes for the two ROIs.

    The latent Gaussian processes have correlation ``rho`` by construction:

    ``u_i = sqrt(rho) u_s + sqrt(1-rho) u_i'``
    ``u_k = sqrt(rho) u_s + sqrt(1-rho) u_k'``

    and are converted into positive envelopes through an affine transform. The
    scale is chosen so clipping is rare, which keeps the envelope correlation
    close to the target value.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1].")
    shared = _band_limited_noise(n_trials, n_samples, fs, taps, rng)
    noise_i = _band_limited_noise(n_trials, n_samples, fs, taps, rng)
    noise_k = _band_limited_noise(n_trials, n_samples, fs, taps, rng)
    latent_i = np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * noise_i
    latent_k = np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * noise_k
    env_i = np.clip(1.0 + ENVELOPE_SCALE * latent_i, 0.05, None)
    env_k = np.clip(1.0 + ENVELOPE_SCALE * latent_k, 0.05, None)
    return env_i, env_k


def _make_shared_carrier(
    n_trials: int,
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
    taps: np.ndarray,
) -> np.ndarray:
    """Return a unit-modulus analytic alpha carrier shared by both ROIs.

    The carrier is generated from band-pass filtered Gaussian noise and then
    normalized to unit modulus. This preserves a realistic wandering alpha
    phase process without introducing extra envelope correlation from the
    carrier amplitude itself.
    """
    carrier_real = _band_limited_noise(n_trials, n_samples, fs, taps, rng)
    carrier_analytic = hilbert(carrier_real, axis=-1)
    return carrier_analytic / (np.abs(carrier_analytic) + 1e-12)


def sigma_from_snr(snr: float) -> float:
    """Convert an SNR value into measurement-noise standard deviation.

    The simulated signal component is standardized to unit variance before
    measurement noise is added, so ``sigma = snr^{-1/2}`` implements

    ``SNR = var(signal) / var(sigma * w)``.
    """
    if snr <= 0.0:
        raise ValueError("SNR must be strictly positive.")
    return float(np.sqrt(1.0 / snr))


def generate_coupled_pair(
    rho: float,
    n_samples: int,
    fs: float,
    sigma_meas: float,
    rng: np.random.Generator,
    taps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one noisy coupled ROI pair.

    The simulated *true* connectivity lives in the envelope domain. Two slow,
    correlated envelope processes with target correlation ``rho`` modulate a
    shared alpha carrier with a fixed non-zero phase lag between ROIs. This is
    the simplest model that keeps ``AEC`` and ``AEC_orth`` aligned at high SNR
    while still making orthogonalization sensitive to measurement noise in the
    phase estimate.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1].")
    taps = make_alpha_filter(fs) if taps is None else taps
    envelope_taps = _make_envelope_filter(fs)
    env_i, env_k = _make_envelopes(rho, 1, n_samples, fs, rng, envelope_taps)
    carrier = _make_shared_carrier(1, n_samples, fs, rng, taps)
    x_i = np.real(env_i * carrier)[0]
    x_k = np.real(env_k * carrier * np.exp(1j * PHASE_LAG_RAD))[0]
    x_i_noisy = x_i + sigma_meas * rng.standard_normal(n_samples)
    x_k_noisy = x_k + sigma_meas * rng.standard_normal(n_samples)
    return x_i_noisy.astype(float), x_k_noisy.astype(float)


def generate_subject(
    rho: float,
    n_trials: int,
    trial_len: int,
    fs: float,
    sigma_meas: float,
    rng: np.random.Generator,
    taps: np.ndarray | None = None,
) -> SubjectMetrics:
    """Return subject-level AEC, AEC-orth, and gap averaged across trials.

    Each trial uses the same equations as :func:`generate_coupled_pair`, then
    trial-wise ``AEC``, ``AEC_orth``, and ``gap`` are averaged to the subject
    level. This matches the subject-as-unit strategy used in the empirical
    workflow.
    """
    aec, aec_orth, gap = generate_trial_metrics(
        rho,
        n_trials,
        trial_len,
        fs,
        sigma_meas,
        rng,
        taps=taps,
    )
    return SubjectMetrics(
        aec=float(np.mean(aec)),
        aec_orth=float(np.mean(aec_orth)),
        gap=float(np.mean(gap)),
    )


def generate_trial_metrics(
    rho: float,
    n_trials: int,
    trial_len: int,
    fs: float,
    sigma_meas: float,
    rng: np.random.Generator,
    taps: np.ndarray | None = None,
    envelope_taps: np.ndarray | None = None,
    batch_trials: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return trial-level ``AEC``, ``AEC_orth``, and ``gap`` values.

    This is the core simulator used by the manuscript-scale Monte Carlo runs.
    It keeps the per-trial signal model unchanged but separates expensive
    signal generation from the much cheaper subject-level resampling carried
    out later in the experiments.
    """
    taps = make_alpha_filter(fs) if taps is None else taps
    envelope_taps = _make_envelope_filter(fs) if envelope_taps is None else envelope_taps

    aec = np.empty(n_trials, dtype=float)
    aec_orth = np.empty(n_trials, dtype=float)
    gap = np.empty(n_trials, dtype=float)

    if batch_trials <= 0:
        batch_trials = n_trials

    lag_factor = np.exp(1j * PHASE_LAG_RAD)
    for start in range(0, n_trials, batch_trials):
        stop = min(start + batch_trials, n_trials)
        batch_size = stop - start
        env_i, env_k = _make_envelopes(rho, batch_size, trial_len, fs, rng, envelope_taps)
        carrier = _make_shared_carrier(batch_size, trial_len, fs, rng, taps)
        x_i = np.real(env_i * carrier)
        x_k = np.real(env_k * carrier * lag_factor)
        x_i_noisy = x_i + sigma_meas * rng.standard_normal(x_i.shape)
        x_k_noisy = x_k + sigma_meas * rng.standard_normal(x_k.shape)
        aec_batch, aec_orth_batch, gap_batch = compute_trial_metrics_batch(x_i_noisy, x_k_noisy)
        aec[start:stop] = aec_batch
        aec_orth[start:stop] = aec_orth_batch
        gap[start:stop] = gap_batch
    return aec, aec_orth, gap


def generate_group_subject_metrics(
    rho: float,
    trial_counts: np.ndarray,
    trial_len: int,
    fs: float,
    sigma_meas: float,
    rng: np.random.Generator,
    taps: np.ndarray | None = None,
    envelope_taps: np.ndarray | None = None,
    batch_trials: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return subject-level connectivity means for a whole group.

    Parameters
    ----------
    rho:
        Target envelope correlation between the two ROIs.
    trial_counts:
        Number of trials contributed by each subject in the group.
    trial_len:
        Useful trial length in samples.
    fs:
        Sampling frequency in Hz.
    sigma_meas:
        Measurement-noise standard deviation implied by the requested SNR.
    rng:
        Explicit random generator used for every stochastic draw.
    taps, envelope_taps:
        Optional precomputed alpha-band and low-frequency envelope filters.
    batch_trials:
        Maximum number of trials generated at once. Batching keeps memory use
        bounded while still avoiding the large overhead of subject-by-subject
        filtering.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Subject-level mean ``AEC``, mean ``AEC_orth``, and mean ``gap``.
    """
    trial_counts = np.asarray(trial_counts, dtype=int)
    if np.any(trial_counts <= 0):
        raise ValueError("All trial counts must be strictly positive.")

    taps = make_alpha_filter(fs) if taps is None else taps
    envelope_taps = _make_envelope_filter(fs) if envelope_taps is None else envelope_taps

    n_subjects = trial_counts.size
    trial_subject_ids = np.repeat(np.arange(n_subjects, dtype=int), trial_counts)
    total_trials = int(trial_counts.sum())

    aec_sum = np.zeros(n_subjects, dtype=float)
    aec_orth_sum = np.zeros(n_subjects, dtype=float)
    gap_sum = np.zeros(n_subjects, dtype=float)

    lag_factor = np.exp(1j * PHASE_LAG_RAD)
    if batch_trials <= 0:
        batch_trials = total_trials

    for start in range(0, total_trials, batch_trials):
        stop = min(start + batch_trials, total_trials)
        batch_size = stop - start
        subject_ids = trial_subject_ids[start:stop]

        env_i, env_k = _make_envelopes(rho, batch_size, trial_len, fs, rng, envelope_taps)
        carrier = _make_shared_carrier(batch_size, trial_len, fs, rng, taps)
        x_i = np.real(env_i * carrier)
        x_k = np.real(env_k * carrier * lag_factor)
        x_i_noisy = x_i + sigma_meas * rng.standard_normal(x_i.shape)
        x_k_noisy = x_k + sigma_meas * rng.standard_normal(x_k.shape)

        aec, aec_orth, gap = compute_trial_metrics_batch(x_i_noisy, x_k_noisy)
        aec_sum += np.bincount(subject_ids, weights=aec, minlength=n_subjects)
        aec_orth_sum += np.bincount(subject_ids, weights=aec_orth, minlength=n_subjects)
        gap_sum += np.bincount(subject_ids, weights=gap, minlength=n_subjects)

    counts = trial_counts.astype(float)
    return aec_sum / counts, aec_orth_sum / counts, gap_sum / counts
