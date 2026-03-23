from __future__ import annotations

"""Monte Carlo experiments for the AEC versus AEC-orth SNR bias study."""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover - fallback for minimal environments
    Parallel = None
    delayed = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable

from .signal_model import generate_trial_metrics, make_alpha_filter, sigma_from_snr


EMPIRICAL_DELTA_GAP = 0.0060
EMPIRICAL_D = 0.575
TRIAL_LEN = 4000
FS = 1000.0
N_SUBJECTS_PER_GROUP = 35
DEFAULT_SNR_GRID = np.logspace(0.0, np.log10(20.0), 10)
DEFAULT_DELTA_TRIALS = np.array([0, 1, 2, 3, 4, 5, 7, 10], dtype=int)
DEFAULT_SNR_LEVELS = np.array([2.0, 5.0, 10.0, 20.0], dtype=float)
DEFAULT_RHO_LEVELS = np.array([0.04, 0.08, 0.12, 0.16], dtype=float)
DEFAULT_LIBRARY_SIZE = 4096

CONVERTER_TRIALS = {"mean": 42.5, "std": 4.25, "low": 27, "high": 58}
NONCONVERTER_TRIALS = {"mean": 45.7, "std": 8.85, "low": 29, "high": 73}


@dataclass(frozen=True)
class GroupSimulationResult:
    """Container for one group mean and standardized effect."""

    mean_gap_converter: float
    mean_gap_nonconverter: float
    delta_gap: float
    cohen_d: float


def _parallel_map(tasks: list[Any], fn, n_jobs: int, desc: str) -> list[Any]:
    """Run a parameter sweep in parallel when joblib is available."""
    iterator = tqdm(tasks, desc=desc, leave=False)
    if Parallel is None or delayed is None or n_jobs == 1:
        return [fn(task) for task in iterator]
    # Threaded workers avoid the semaphore restrictions of the sandboxed macOS
    # environment while still parallelizing the outer parameter sweep.
    return Parallel(n_jobs=n_jobs, prefer="threads")(delayed(fn)(task) for task in iterator)


def _sample_truncated_normal(
    mean: float,
    std: float,
    low: int,
    high: int,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample rounded integers from a truncated normal distribution."""
    samples = np.empty(size, dtype=int)
    filled = 0
    batch_size = max(size, 64)
    while filled < size:
        draw = np.rint(rng.normal(loc=mean, scale=std, size=batch_size)).astype(int)
        draw = draw[(draw >= low) & (draw <= high)]
        take = min(size - filled, draw.size)
        if take:
            samples[filled : filled + take] = draw[:take]
            filled += take
    return samples


def _sample_paired_truncated_normals(
    mean_c: float,
    std_c: float,
    low_c: int,
    high_c: int,
    mean_nc: float,
    std_nc: float,
    low_nc: int,
    high_nc: int,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample paired trial-count draws for the isolation experiment.

    Experiment 2 is meant to isolate the effect of a trial-count offset, not
    the extra Monte Carlo variance that comes from drawing two unrelated small
    samples from the same distribution. The paired construction uses shared
    quantiles for both truncated normals, so ``delta_trials = 0`` becomes a
    clean no-difference baseline.
    """
    q = rng.uniform(low=1e-6, high=1.0 - 1e-6, size=size)

    def _ppf(mean: float, std: float, low: int, high: int) -> np.ndarray:
        a = (low - mean) / std
        b = (high - mean) / std
        draws = truncnorm.ppf(q, a, b, loc=mean, scale=std)
        return np.rint(draws).astype(int)

    return (
        _ppf(mean_c, std_c, low_c, high_c),
        _ppf(mean_nc, std_nc, low_nc, high_nc),
    )


def _cohen_d_nonconverter_minus_converter(values_c: np.ndarray, values_nc: np.ndarray) -> float:
    """Return Cohen's d oriented so positive values mean smaller converter gaps."""
    var_c = float(np.var(values_c, ddof=1))
    var_nc = float(np.var(values_nc, ddof=1))
    pooled = np.sqrt(((len(values_c) - 1) * var_c + (len(values_nc) - 1) * var_nc) / (len(values_c) + len(values_nc) - 2))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(values_nc) - np.mean(values_c)) / pooled)


def _build_gap_library(
    rho: float,
    snr: float,
    library_size: int,
    fs: float,
    trial_len: int,
    rng: np.random.Generator,
    taps: np.ndarray,
    envelope_taps: np.ndarray,
    batch_trials: int,
) -> np.ndarray:
    """Generate a reusable library of trial-level gap values for one condition."""
    sigma = sigma_from_snr(snr)
    _, _, gap = generate_trial_metrics(
        rho,
        library_size,
        trial_len,
        fs,
        sigma,
        rng,
        taps=taps,
        envelope_taps=envelope_taps,
        batch_trials=batch_trials,
    )
    return gap


def _sample_subject_means_from_library(
    gap_library: np.ndarray,
    trial_counts: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample subject-level mean gaps by resampling precomputed trial values."""
    trial_counts = np.asarray(trial_counts, dtype=int)
    max_trials = int(trial_counts.max())
    sampled = gap_library[rng.integers(0, gap_library.size, size=(trial_counts.size, max_trials))]
    mask = np.arange(max_trials)[None, :] < trial_counts[:, None]
    return (sampled * mask).sum(axis=1) / trial_counts


def _simulate_group_gaps_from_library(
    gap_library: np.ndarray,
    trial_counts_c: np.ndarray,
    trial_counts_nc: np.ndarray,
    rng: np.random.Generator,
) -> GroupSimulationResult:
    """Simulate one Monte Carlo realization using a cached trial-gap library."""
    gaps_c = _sample_subject_means_from_library(gap_library, trial_counts_c, rng)
    gaps_nc = _sample_subject_means_from_library(gap_library, trial_counts_nc, rng)

    mean_gap_c = float(np.mean(gaps_c))
    mean_gap_nc = float(np.mean(gaps_nc))
    return GroupSimulationResult(
        mean_gap_converter=mean_gap_c,
        mean_gap_nonconverter=mean_gap_nc,
        delta_gap=mean_gap_nc - mean_gap_c,
        cohen_d=_cohen_d_nonconverter_minus_converter(gaps_c, gaps_nc),
    )


def _summarize_distribution(parameter_name: str, parameter_value: float | int, deltas: np.ndarray, ds: np.ndarray) -> dict[str, float]:
    """Summarize one parameter condition across Monte Carlo iterations."""
    return {
        parameter_name: float(parameter_value),
        "mean_delta_gap": float(np.mean(deltas)),
        "std_delta_gap": float(np.std(deltas, ddof=1)),
        "delta_gap_q05": float(np.quantile(deltas, 0.05)),
        "delta_gap_q95": float(np.quantile(deltas, 0.95)),
        "mean_cohen_d": float(np.mean(ds)),
        "std_cohen_d": float(np.std(ds, ddof=1)),
        "cohen_d_q05": float(np.quantile(ds, 0.05)),
        "cohen_d_q95": float(np.quantile(ds, 0.95)),
    }


def run_experiment_1_snr_sweep(
    *,
    rho: float = 0.08,
    snr_grid: np.ndarray | None = None,
    n_subjects: int = N_SUBJECTS_PER_GROUP,
    n_iter: int = 2000,
    fs: float = FS,
    trial_len: int = TRIAL_LEN,
    random_seed: int = 42,
    n_jobs: int = -1,
    batch_trials: int = 4096,
    library_size: int = DEFAULT_LIBRARY_SIZE,
) -> pd.DataFrame:
    """Run the main SNR sweep with empirical trial-count distributions."""
    snr_grid = DEFAULT_SNR_GRID if snr_grid is None else np.asarray(snr_grid, dtype=float)
    taps = make_alpha_filter(fs)
    from .signal_model import _make_envelope_filter

    envelope_taps = _make_envelope_filter(fs)
    master_rng = np.random.default_rng(random_seed)
    snr_seeds = {float(snr): int(master_rng.integers(0, 2**32 - 1)) for snr in snr_grid}
    library_seeds = {float(snr): int(master_rng.integers(0, 2**32 - 1)) for snr in snr_grid}
    libraries = {
        float(snr): _build_gap_library(
            rho,
            float(snr),
            library_size,
            fs,
            trial_len,
            np.random.default_rng(library_seeds[float(snr)]),
            taps,
            envelope_taps,
            batch_trials,
        )
        for snr in snr_grid
    }

    def _run_one_snr(snr: float) -> dict[str, float]:
        rng = np.random.default_rng(snr_seeds[float(snr)])
        deltas = np.empty(n_iter, dtype=float)
        ds = np.empty(n_iter, dtype=float)
        for idx in range(n_iter):
            trial_counts_c = _sample_truncated_normal(
                CONVERTER_TRIALS["mean"],
                CONVERTER_TRIALS["std"],
                CONVERTER_TRIALS["low"],
                CONVERTER_TRIALS["high"],
                n_subjects,
                rng,
            )
            trial_counts_nc = _sample_truncated_normal(
                NONCONVERTER_TRIALS["mean"],
                NONCONVERTER_TRIALS["std"],
                NONCONVERTER_TRIALS["low"],
                NONCONVERTER_TRIALS["high"],
                n_subjects,
                rng,
            )
            sim = _simulate_group_gaps_from_library(libraries[float(snr)], trial_counts_c, trial_counts_nc, rng)
            deltas[idx] = sim.delta_gap
            ds[idx] = sim.cohen_d
        summary = _summarize_distribution("snr", float(snr), deltas, ds)
        summary["rho"] = float(rho)
        summary["n_iter"] = int(n_iter)
        return summary

    rows = _parallel_map(list(map(float, snr_grid)), _run_one_snr, n_jobs=n_jobs, desc="Experiment 1: SNR sweep")
    return pd.DataFrame(rows).sort_values("snr").reset_index(drop=True)


def run_experiment_2_trial_count_effect(
    *,
    rho: float = 0.08,
    snr: float = 10.0,
    delta_trials_values: np.ndarray | None = None,
    n_subjects: int = N_SUBJECTS_PER_GROUP,
    n_iter: int = 2000,
    fs: float = FS,
    trial_len: int = TRIAL_LEN,
    random_seed: int = 42,
    n_jobs: int = -1,
    batch_trials: int = 4096,
    library_size: int = DEFAULT_LIBRARY_SIZE,
) -> pd.DataFrame:
    """Run the trial-count isolation experiment at fixed SNR."""
    delta_trials_values = DEFAULT_DELTA_TRIALS if delta_trials_values is None else np.asarray(delta_trials_values, dtype=int)
    taps = make_alpha_filter(fs)
    from .signal_model import _make_envelope_filter

    envelope_taps = _make_envelope_filter(fs)
    master_rng = np.random.default_rng(random_seed)
    seeds = {int(delta): int(master_rng.integers(0, 2**32 - 1)) for delta in delta_trials_values}
    library_rng = np.random.default_rng(int(master_rng.integers(0, 2**32 - 1)))
    gap_library = _build_gap_library(rho, float(snr), library_size, fs, trial_len, library_rng, taps, envelope_taps, batch_trials)

    def _run_one_delta(delta_trials: int) -> dict[str, float]:
        rng = np.random.default_rng(seeds[int(delta_trials)])
        deltas = np.empty(n_iter, dtype=float)
        ds = np.empty(n_iter, dtype=float)
        nc_mean = NONCONVERTER_TRIALS["mean"]
        nc_std = NONCONVERTER_TRIALS["std"]
        conv_mean = nc_mean - float(delta_trials)
        conv_std = nc_std * conv_mean / nc_mean
        for idx in range(n_iter):
            trial_counts_c, trial_counts_nc = _sample_paired_truncated_normals(
                conv_mean,
                conv_std,
                NONCONVERTER_TRIALS["low"],
                NONCONVERTER_TRIALS["high"],
                nc_mean,
                nc_std,
                NONCONVERTER_TRIALS["low"],
                NONCONVERTER_TRIALS["high"],
                n_subjects,
                rng,
            )
            sim = _simulate_group_gaps_from_library(gap_library, trial_counts_c, trial_counts_nc, rng)
            deltas[idx] = sim.delta_gap
            ds[idx] = sim.cohen_d
        summary = _summarize_distribution("delta_trials", int(delta_trials), deltas, ds)
        summary["rho"] = float(rho)
        summary["snr"] = float(snr)
        summary["n_iter"] = int(n_iter)
        return summary

    rows = _parallel_map(list(map(int, delta_trials_values)), _run_one_delta, n_jobs=n_jobs, desc="Experiment 2: trial counts")
    return pd.DataFrame(rows).sort_values("delta_trials").reset_index(drop=True)


def run_experiment_3_snr_rho_interaction(
    *,
    snr_levels: np.ndarray | None = None,
    rho_levels: np.ndarray | None = None,
    n_subjects: int = N_SUBJECTS_PER_GROUP,
    n_iter: int = 1000,
    fs: float = FS,
    trial_len: int = TRIAL_LEN,
    random_seed: int = 42,
    n_jobs: int = -1,
    batch_trials: int = 4096,
    library_size: int = DEFAULT_LIBRARY_SIZE,
) -> pd.DataFrame:
    """Run the SNR-by-rho interaction grid."""
    snr_levels = DEFAULT_SNR_LEVELS if snr_levels is None else np.asarray(snr_levels, dtype=float)
    rho_levels = DEFAULT_RHO_LEVELS if rho_levels is None else np.asarray(rho_levels, dtype=float)
    taps = make_alpha_filter(fs)
    from .signal_model import _make_envelope_filter

    envelope_taps = _make_envelope_filter(fs)
    master_rng = np.random.default_rng(random_seed)
    combos = [(float(snr), float(rho)) for rho in rho_levels for snr in snr_levels]
    combo_seeds = {combo: int(master_rng.integers(0, 2**32 - 1)) for combo in combos}
    combo_library_seeds = {combo: int(master_rng.integers(0, 2**32 - 1)) for combo in combos}
    libraries = {
        combo: _build_gap_library(
            combo[1],
            combo[0],
            library_size,
            fs,
            trial_len,
            np.random.default_rng(combo_library_seeds[combo]),
            taps,
            envelope_taps,
            batch_trials,
        )
        for combo in combos
    }

    def _run_one_combo(combo: tuple[float, float]) -> dict[str, float]:
        snr, rho = combo
        rng = np.random.default_rng(combo_seeds[(snr, rho)])
        ds = np.empty(n_iter, dtype=float)
        deltas = np.empty(n_iter, dtype=float)
        for idx in range(n_iter):
            trial_counts_c = _sample_truncated_normal(
                CONVERTER_TRIALS["mean"],
                CONVERTER_TRIALS["std"],
                CONVERTER_TRIALS["low"],
                CONVERTER_TRIALS["high"],
                n_subjects,
                rng,
            )
            trial_counts_nc = _sample_truncated_normal(
                NONCONVERTER_TRIALS["mean"],
                NONCONVERTER_TRIALS["std"],
                NONCONVERTER_TRIALS["low"],
                NONCONVERTER_TRIALS["high"],
                n_subjects,
                rng,
            )
            sim = _simulate_group_gaps_from_library(libraries[(snr, rho)], trial_counts_c, trial_counts_nc, rng)
            ds[idx] = sim.cohen_d
            deltas[idx] = sim.delta_gap
        summary = _summarize_distribution("snr", snr, deltas, ds)
        summary["rho"] = rho
        summary["n_iter"] = int(n_iter)
        return summary

    rows = _parallel_map(combos, _run_one_combo, n_jobs=n_jobs, desc="Experiment 3: SNR x rho")
    return pd.DataFrame(rows).sort_values(["rho", "snr"]).reset_index(drop=True)


def run_targeted_trial_difference_scenario(
    *,
    snr: float,
    delta_trials: int,
    rho: float = 0.08,
    n_subjects: int = N_SUBJECTS_PER_GROUP,
    n_iter: int = 1000,
    fs: float = FS,
    trial_len: int = TRIAL_LEN,
    random_seed: int = 42,
    batch_trials: int = 4096,
    library_size: int = DEFAULT_LIBRARY_SIZE,
) -> dict[str, float]:
    """Return one targeted scenario used in the final summary table."""
    df = run_experiment_2_trial_count_effect(
        rho=rho,
        snr=snr,
        delta_trials_values=np.array([delta_trials]),
        n_subjects=n_subjects,
        n_iter=n_iter,
        fs=fs,
        trial_len=trial_len,
        random_seed=random_seed,
        n_jobs=1,
        batch_trials=batch_trials,
        library_size=library_size,
    )
    return df.iloc[0].to_dict()
