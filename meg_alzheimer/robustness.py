from __future__ import annotations

"""Robustness analyses for the predefined H1-H3 endpoint family.

This module adds secondary checks around the existing six endpoint definitions
without changing the primary scientific pipeline. The key design choice is to
reuse the same subject-level endpoints as the main analysis and only vary the
questions asked around them:

- what happens when every subject contributes the same number of trials?
- does the group effect survive leave-one-subject-out perturbations?
- do H3 conclusions change under reasonable within-subject aggregation choices?
- does a simple covariate adjustment for trial count materially alter the group effect?

The only expensive step is a one-time trial-level precomputation of the six
endpoint quantities. After that, all robustness analyses reuse the cached
trial-level table.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import hilbert

from .atlas import build_network_masks, mean_inter, mean_intra
from .connectivity import _corr_1d, _orth_component
from .dataset import crop_trials, discover_subjects, load_brainstorm_subject
from .signals import band_defs, bandpass_filt
from .strong_hypotheses import STRONG_ENDPOINTS, _extract_endpoint_frame


GROUP_A = "Converter"
GROUP_B = "Non-converter"
EPS = 1e-6


@dataclass(frozen=True)
class EndpointSummary:
    endpoint_id: str
    hypothesis_id: str
    mean_group_a: float
    mean_group_b: float
    delta_group_a_minus_group_b: float
    t_stat: float
    p_two_sided: float
    p_one_sided: float
    effect_size_d: float
    n_group_a: int
    n_group_b: int


def endpoint_ids() -> list[str]:
    return [spec.endpoint_id for spec in STRONG_ENDPOINTS]


def endpoint_components() -> list[str]:
    return [
        "alpha_full_aec",
        "alpha_full_orth",
        "alpha_inter_aec",
        "alpha_inter_orth",
        "beta_inter_aec",
        "beta_inter_orth",
    ]


def _expected_direction_map() -> dict[str, str]:
    return {spec.endpoint_id: spec.expected_direction for spec in STRONG_ENDPOINTS}


def _one_sided_pvalue(t_stat: float, two_sided_p: float, expected_direction: str) -> float:
    if expected_direction == "group_a_lt_group_b":
        return float(two_sided_p / 2.0) if t_stat < 0.0 else float(1.0 - two_sided_p / 2.0)
    if expected_direction == "group_a_gt_group_b":
        return float(two_sided_p / 2.0) if t_stat > 0.0 else float(1.0 - two_sided_p / 2.0)
    raise ValueError(f"Unsupported expected_direction: {expected_direction}")


def _cohen_d_unpaired(values_a: np.ndarray, values_b: np.ndarray) -> float:
    mean_diff = float(np.mean(values_a) - np.mean(values_b))
    var_a = float(np.var(values_a, ddof=1))
    var_b = float(np.var(values_b, ddof=1))
    pooled = np.sqrt(((len(values_a) - 1) * var_a + (len(values_b) - 1) * var_b) / max(1, len(values_a) + len(values_b) - 2))
    if pooled < 1e-12:
        return float("nan")
    return float(mean_diff / pooled)


def _mean_upper_triangle(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _temppar_composites_from_analytic(analytic_trial: np.ndarray, network_masks: Mapping[str, np.ndarray]) -> tuple[float, float, float, float]:
    """Return TempPar-centered full/inter composites for AEC and AEC-orth.

    Output order:
    - alpha/beta AEC full
    - alpha/beta AEC inter
    - alpha/beta AEC-orth full
    - alpha/beta AEC-orth inter
    """
    env = np.abs(analytic_trial)
    env = env - env.mean(axis=-1, keepdims=True)

    temp_idx = np.where(network_masks["TempPar"])[0]
    other_networks = [name for name in sorted(network_masks) if name not in {"Background", "TempPar"}]
    other_idx = {name: np.where(network_masks[name])[0] for name in other_networks}

    aec_intra_values: list[float] = []
    orth_intra_values: list[float] = []
    for pos_i, i in enumerate(temp_idx[:-1]):
        for j in temp_idx[pos_i + 1 :]:
            aec_intra_values.append(_corr_1d(env[i], env[j]))
            yi_perp = _orth_component(analytic_trial[i], analytic_trial[j])
            xi_perp = _orth_component(analytic_trial[j], analytic_trial[i])
            orth_intra_values.append(0.5 * (_corr_1d(env[i], np.abs(yi_perp)) + _corr_1d(env[j], np.abs(xi_perp))))

    aec_block_means: list[float] = [_mean_upper_triangle(aec_intra_values)]
    orth_block_means: list[float] = [_mean_upper_triangle(orth_intra_values)]
    aec_inter_means: list[float] = []
    orth_inter_means: list[float] = []

    for network_name in other_networks:
        block_aec: list[float] = []
        block_orth: list[float] = []
        for i in temp_idx:
            for j in other_idx[network_name]:
                block_aec.append(_corr_1d(env[i], env[j]))
                yi_perp = _orth_component(analytic_trial[i], analytic_trial[j])
                xi_perp = _orth_component(analytic_trial[j], analytic_trial[i])
                block_orth.append(0.5 * (_corr_1d(env[i], np.abs(yi_perp)) + _corr_1d(env[j], np.abs(xi_perp))))
        aec_mean = float(np.mean(block_aec))
        orth_mean = float(np.mean(block_orth))
        aec_inter_means.append(aec_mean)
        orth_inter_means.append(orth_mean)
        aec_block_means.append(aec_mean)
        orth_block_means.append(orth_mean)

    return (
        float(np.mean(aec_block_means)),
        float(np.mean(aec_inter_means)),
        float(np.mean(orth_block_means)),
        float(np.mean(orth_inter_means)),
    )


def _subject_trial_frame(record, fs: float, crop_start: int, crop_end: int, bands: Mapping[str, tuple[float, float]]) -> pd.DataFrame:
    loaded = load_brainstorm_subject(record)
    cropped = crop_trials(loaded.X_trials, start=crop_start, end=crop_end)
    _, network_masks = build_network_masks(loaded.roi_labels)

    alpha_filtered = bandpass_filt(cropped, fs=fs, band=bands["alpha"])
    beta_filtered = bandpass_filt(cropped, fs=fs, band=bands["beta"])
    alpha_analytic = hilbert(alpha_filtered, axis=-1)
    beta_analytic = hilbert(beta_filtered, axis=-1)

    rows: list[dict[str, Any]] = []
    for trial_idx in range(cropped.shape[0]):
        alpha_full_aec, alpha_inter_aec, alpha_full_orth, alpha_inter_orth = _temppar_composites_from_analytic(
            analytic_trial=alpha_analytic[trial_idx],
            network_masks=network_masks,
        )
        _, beta_inter_aec, _, beta_inter_orth = _temppar_composites_from_analytic(
            analytic_trial=beta_analytic[trial_idx],
            network_masks=network_masks,
        )
        rows.append(
            {
                "subject_id": record.subject_id,
                "group": record.group,
                "trial_index": int(trial_idx),
                "alpha_full_aec": alpha_full_aec,
                "alpha_full_orth": alpha_full_orth,
                "alpha_inter_aec": alpha_inter_aec,
                "alpha_inter_orth": alpha_inter_orth,
                "beta_inter_aec": beta_inter_aec,
                "beta_inter_orth": beta_inter_orth,
                "H1_AEC": alpha_full_aec,
                "H1_AECorth": alpha_full_orth,
                "H2_AEC": beta_inter_aec,
                "H2_AECorth": beta_inter_orth,
                "H3_gap_full": alpha_full_aec - alpha_full_orth,
                "H3_gap_inter": alpha_inter_aec - alpha_inter_orth,
            }
        )
    return pd.DataFrame(rows)


def precompute_trial_endpoint_values(
    data_root: str | Path,
    output_path: str | Path | None = None,
    *,
    fs: float = 1000.0,
    crop_start: int = 2000,
    crop_end: int = 6000,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute or load the trial-level endpoint table for all subjects."""
    cache_path = Path(output_path) if output_path is not None else None
    if cache_path is not None and cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    records = discover_subjects(data_root)
    if not records:
        raise FileNotFoundError(f"No Brainstorm subjects found under {data_root}.")

    bands = band_defs()
    frames: list[pd.DataFrame] = []
    for idx, record in enumerate(records, start=1):
        frames.append(_subject_trial_frame(record, fs=fs, crop_start=crop_start, crop_end=crop_end, bands=bands))
        if verbose and (idx % 10 == 0 or idx == len(records)):
            print(f"Processed {idx}/{len(records)} subjects for trial-level endpoint precomputation.")

    trial_df = pd.concat(frames, ignore_index=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        trial_df.to_csv(cache_path, index=False)
    return trial_df


def validate_against_main_report(
    trial_df: pd.DataFrame,
    output_root: str | Path,
    *,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Check that the trial-derived subject means reproduce the main endpoint table."""
    network_df = pd.read_csv(Path(output_root) / "subject_network_means.csv")
    rows: list[dict[str, Any]] = []
    for spec in STRONG_ENDPOINTS:
        observed = (
            trial_df.groupby(["subject_id", "group"], as_index=False)[spec.endpoint_id]
            .mean()
            .rename(columns={spec.endpoint_id: "observed"})
        )
        reference = _extract_endpoint_frame(spec, network_df)[["subject_id", "group", "value"]].rename(columns={"value": "reference"})
        merged = reference.merge(observed, on=["subject_id", "group"], how="inner")
        merged["abs_diff"] = np.abs(merged["observed"] - merged["reference"])
        max_abs = float(merged["abs_diff"].max())
        rows.append({"endpoint_id": spec.endpoint_id, "max_abs_diff": max_abs})
        if max_abs > tolerance:
            raise ValueError(
                f"Trial-level robustness precomputation does not reproduce the main endpoint values for {spec.endpoint_id}. "
                f"Maximum absolute difference: {max_abs:.6g}"
            )
    return pd.DataFrame(rows)


def subject_trial_counts(trial_df: pd.DataFrame) -> pd.DataFrame:
    return (
        trial_df.groupby(["subject_id", "group"], as_index=False)
        .size()
        .rename(columns={"size": "n_valid_trials"})
        .sort_values(["group", "subject_id"])
        .reset_index(drop=True)
    )


def build_subject_endpoint_table(trial_df: pd.DataFrame, aggregation: str = "mean_r") -> pd.DataFrame:
    """Aggregate trial-level endpoint components into subject-level endpoints.

    Supported aggregation modes
    --------------------------
    mean_r:
        Arithmetic mean across trials on the original correlation scale.
    median_r:
        Median across trials on the original correlation scale.
    fisher_z:
        For component correlations, average in Fisher-z space and transform back
        to r before rebuilding the derived H3 gaps.
    """
    if aggregation not in {"mean_r", "median_r", "fisher_z"}:
        raise ValueError(f"Unsupported aggregation mode: {aggregation}")

    rows: list[dict[str, Any]] = []
    for subject_id, frame in trial_df.groupby("subject_id", sort=True):
        group = str(frame["group"].iloc[0])

        def reduce_component(column: str) -> float:
            values = frame[column].to_numpy(dtype=float)
            if aggregation == "mean_r":
                return float(np.mean(values))
            if aggregation == "median_r":
                return float(np.median(values))
            clipped = np.clip(values, -1.0 + EPS, 1.0 - EPS)
            return float(np.tanh(np.mean(np.arctanh(clipped))))

        alpha_full_aec = reduce_component("alpha_full_aec")
        alpha_full_orth = reduce_component("alpha_full_orth")
        alpha_inter_aec = reduce_component("alpha_inter_aec")
        alpha_inter_orth = reduce_component("alpha_inter_orth")
        beta_inter_aec = reduce_component("beta_inter_aec")
        beta_inter_orth = reduce_component("beta_inter_orth")

        endpoint_map = {
            "H1_AEC": alpha_full_aec,
            "H1_AECorth": alpha_full_orth,
            "H2_AEC": beta_inter_aec,
            "H2_AECorth": beta_inter_orth,
            "H3_gap_full": alpha_full_aec - alpha_full_orth,
            "H3_gap_inter": alpha_inter_aec - alpha_inter_orth,
        }
        for endpoint_id, value in endpoint_map.items():
            rows.append(
                {
                    "subject_id": subject_id,
                    "group": group,
                    "endpoint_id": endpoint_id,
                    "aggregation": aggregation,
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows).sort_values(["aggregation", "endpoint_id", "group", "subject_id"]).reset_index(drop=True)


def _endpoint_stats_from_values(endpoint_id: str, values_a: np.ndarray, values_b: np.ndarray) -> EndpointSummary:
    t_stat, p_two_sided = stats.ttest_ind(values_a, values_b, equal_var=False)
    expected_direction = _expected_direction_map()[endpoint_id]
    return EndpointSummary(
        endpoint_id=endpoint_id,
        hypothesis_id=endpoint_id.split("_")[0],
        mean_group_a=float(np.mean(values_a)),
        mean_group_b=float(np.mean(values_b)),
        delta_group_a_minus_group_b=float(np.mean(values_a) - np.mean(values_b)),
        t_stat=float(t_stat),
        p_two_sided=float(p_two_sided),
        p_one_sided=_one_sided_pvalue(float(t_stat), float(p_two_sided), expected_direction),
        effect_size_d=_cohen_d_unpaired(values_a, values_b),
        n_group_a=len(values_a),
        n_group_b=len(values_b),
    )


def compute_endpoint_stats(subject_endpoint_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for endpoint_id, frame in subject_endpoint_df.groupby("endpoint_id", sort=False):
        values_a = frame.loc[frame["group"] == GROUP_A, "value"].to_numpy(dtype=float)
        values_b = frame.loc[frame["group"] == GROUP_B, "value"].to_numpy(dtype=float)
        summary = _endpoint_stats_from_values(endpoint_id, values_a, values_b)
        rows.append(summary.__dict__)
    return pd.DataFrame(rows)


def matched_trial_subsampling(
    trial_df: pd.DataFrame,
    *,
    n_iter: int,
    seed: int,
    match_n: int | None = None,
    min_trials_threshold: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeatedly equalize trial counts across subjects by subsampling."""
    subject_counts = subject_trial_counts(trial_df)
    if min_trials_threshold is not None:
        keep_subjects = subject_counts.loc[subject_counts["n_valid_trials"] >= min_trials_threshold, "subject_id"]
        trial_df = trial_df.loc[trial_df["subject_id"].isin(set(keep_subjects))].copy()
        subject_counts = subject_trial_counts(trial_df)
        if subject_counts.empty:
            raise ValueError("No subjects remain after applying min_trials_threshold.")

    target_n = int(match_n) if match_n is not None else int(subject_counts["n_valid_trials"].min())
    if target_n < 2:
        raise ValueError(f"Target matched trial count must be at least 2, got {target_n}.")

    subject_trials = {
        subject_id: frame.sort_values("trial_index").reset_index(drop=True)
        for subject_id, frame in trial_df.groupby("subject_id", sort=True)
    }
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for iter_idx in range(n_iter):
        sampled_rows: list[dict[str, Any]] = []
        for subject_id, frame in subject_trials.items():
            choice = rng.choice(len(frame), size=target_n, replace=False)
            sampled = frame.iloc[choice]
            group = str(sampled["group"].iloc[0])
            for endpoint_id in endpoint_ids():
                sampled_rows.append(
                    {
                        "subject_id": subject_id,
                        "group": group,
                        "endpoint_id": endpoint_id,
                        "value": float(sampled[endpoint_id].mean()),
                    }
                )
        sampled_df = pd.DataFrame(sampled_rows)
        stats_df = compute_endpoint_stats(sampled_df)
        for row in stats_df.itertuples(index=False):
            records.append(
                {
                    "iteration": iter_idx,
                    "endpoint_id": row.endpoint_id,
                    "hypothesis_id": row.hypothesis_id,
                    "delta_group_a_minus_group_b": row.delta_group_a_minus_group_b,
                    "t_stat": row.t_stat,
                    "p_two_sided": row.p_two_sided,
                    "p_one_sided": row.p_one_sided,
                    "effect_size_d": row.effect_size_d,
                    "sign_expected": row.delta_group_a_minus_group_b < 0.0,
                    "nominal_p_lt_0_05": row.p_one_sided < 0.05,
                    "target_n_trials": target_n,
                    "n_subjects": sampled_df["subject_id"].nunique(),
                }
            )
    iterations_df = pd.DataFrame(records)
    summary_rows: list[dict[str, Any]] = []
    for endpoint_id, frame in iterations_df.groupby("endpoint_id", sort=False):
        summary_rows.append(
            {
                "endpoint_id": endpoint_id,
                "hypothesis_id": str(frame["hypothesis_id"].iloc[0]),
                "n_iterations": int(len(frame)),
                "target_n_trials": target_n,
                "n_subjects": int(frame["n_subjects"].iloc[0]),
                "delta_mean": float(frame["delta_group_a_minus_group_b"].mean()),
                "delta_sd": float(frame["delta_group_a_minus_group_b"].std(ddof=1)),
                "delta_q05": float(frame["delta_group_a_minus_group_b"].quantile(0.05)),
                "delta_q95": float(frame["delta_group_a_minus_group_b"].quantile(0.95)),
                "t_mean": float(frame["t_stat"].mean()),
                "t_sd": float(frame["t_stat"].std(ddof=1)),
                "d_mean": float(frame["effect_size_d"].mean()),
                "d_sd": float(frame["effect_size_d"].std(ddof=1)),
                "prop_p_lt_0_05": float(frame["nominal_p_lt_0_05"].mean()),
                "prop_expected_sign": float(frame["sign_expected"].mean()),
            }
        )
    settings_df = pd.DataFrame(
        [
            {
                "analysis": "trial_matched_subsampling",
                "n_iterations": n_iter,
                "target_n_trials": target_n,
                "min_trials_threshold": min_trials_threshold if min_trials_threshold is not None else "",
                "n_subjects": int(subject_counts.shape[0]),
            }
        ]
    )
    return iterations_df, pd.DataFrame(summary_rows), settings_df


def _ols_group_plus_trialcount(values: np.ndarray, group_binary: np.ndarray, n_trials: np.ndarray, expected_direction: str) -> dict[str, float]:
    x_trials = n_trials.astype(float) - float(np.mean(n_trials))
    X = np.column_stack([np.ones_like(group_binary, dtype=float), group_binary.astype(float), x_trials])
    y = values.astype(float)
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    fitted = X @ beta
    resid = y - fitted
    dof = X.shape[0] - X.shape[1]
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.diag(cov))
    t_stat = float(beta[1] / se[1])
    p_two_sided = float(2.0 * stats.t.sf(abs(t_stat), df=dof))
    p_one_sided = _one_sided_pvalue(t_stat, p_two_sided, expected_direction)
    return {
        "group_coef": float(beta[1]),
        "group_se": float(se[1]),
        "group_t": t_stat,
        "group_p_two_sided": p_two_sided,
        "group_p_one_sided": p_one_sided,
        "trial_coef": float(beta[2]),
        "trial_se": float(se[2]),
    }


def covariate_adjustment(subject_endpoint_df: pd.DataFrame, trial_counts_df: pd.DataFrame) -> pd.DataFrame:
    merged = subject_endpoint_df.merge(trial_counts_df, on=["subject_id", "group"], how="left")
    rows: list[dict[str, Any]] = []
    direction_map = _expected_direction_map()
    for endpoint_id, frame in merged.groupby("endpoint_id", sort=False):
        values = frame["value"].to_numpy(dtype=float)
        group_binary = (frame["group"] == GROUP_A).to_numpy(dtype=int)
        n_trials = frame["n_valid_trials"].to_numpy(dtype=float)
        fit = _ols_group_plus_trialcount(values, group_binary, n_trials, direction_map[endpoint_id])
        raw_stats = compute_endpoint_stats(frame[["subject_id", "group", "endpoint_id", "value"]]).iloc[0]
        rows.append(
            {
                "endpoint_id": endpoint_id,
                "hypothesis_id": endpoint_id.split("_")[0],
                "raw_delta_group_a_minus_group_b": float(raw_stats["delta_group_a_minus_group_b"]),
                "adjusted_group_coef": fit["group_coef"],
                "adjusted_group_se": fit["group_se"],
                "adjusted_group_t": fit["group_t"],
                "adjusted_group_p_two_sided": fit["group_p_two_sided"],
                "adjusted_group_p_one_sided": fit["group_p_one_sided"],
                "trial_count_coef": fit["trial_coef"],
                "trial_count_se": fit["trial_se"],
            }
        )
    return pd.DataFrame(rows)


def leave_one_out(subject_endpoint_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_ids = sorted(subject_endpoint_df["subject_id"].unique())
    detail_rows: list[dict[str, Any]] = []
    for left_out_subject in subject_ids:
        subset = subject_endpoint_df.loc[subject_endpoint_df["subject_id"] != left_out_subject].copy()
        stats_df = compute_endpoint_stats(subset)
        for row in stats_df.itertuples(index=False):
            detail_rows.append(
                {
                    "left_out_subject_id": left_out_subject,
                    "endpoint_id": row.endpoint_id,
                    "hypothesis_id": row.hypothesis_id,
                    "delta_group_a_minus_group_b": row.delta_group_a_minus_group_b,
                    "effect_size_d": row.effect_size_d,
                    "t_stat": row.t_stat,
                    "p_one_sided": row.p_one_sided,
                    "sign_expected": row.delta_group_a_minus_group_b < 0.0,
                    "nominal_p_lt_0_05": row.p_one_sided < 0.05,
                }
            )
    detail_df = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for endpoint_id, frame in detail_df.groupby("endpoint_id", sort=False):
        summary_rows.append(
            {
                "endpoint_id": endpoint_id,
                "hypothesis_id": str(frame["hypothesis_id"].iloc[0]),
                "n_leave_one_out_runs": int(len(frame)),
                "delta_min": float(frame["delta_group_a_minus_group_b"].min()),
                "delta_max": float(frame["delta_group_a_minus_group_b"].max()),
                "d_min": float(frame["effect_size_d"].min()),
                "d_max": float(frame["effect_size_d"].max()),
                "p_min": float(frame["p_one_sided"].min()),
                "p_max": float(frame["p_one_sided"].max()),
                "n_sign_flips": int((~frame["sign_expected"]).sum()),
                "n_nominal_failures": int((~frame["nominal_p_lt_0_05"]).sum()),
            }
        )
    return detail_df, pd.DataFrame(summary_rows)


def aggregation_sensitivity(trial_df: pd.DataFrame, endpoint_subset: Iterable[str] = ("H3_gap_full", "H3_gap_inter")) -> pd.DataFrame:
    endpoint_subset = list(endpoint_subset)
    rows: list[dict[str, Any]] = []
    for aggregation in ["mean_r", "fisher_z", "median_r"]:
        subject_df = build_subject_endpoint_table(trial_df, aggregation=aggregation)
        subject_df = subject_df.loc[subject_df["endpoint_id"].isin(endpoint_subset)].copy()
        stats_df = compute_endpoint_stats(subject_df)
        for row in stats_df.itertuples(index=False):
            rows.append(
                {
                    "aggregation": aggregation,
                    "endpoint_id": row.endpoint_id,
                    "hypothesis_id": row.hypothesis_id,
                    "mean_group_a": row.mean_group_a,
                    "mean_group_b": row.mean_group_b,
                    "delta_group_a_minus_group_b": row.delta_group_a_minus_group_b,
                    "t_stat": row.t_stat,
                    "p_two_sided": row.p_two_sided,
                    "p_one_sided": row.p_one_sided,
                    "effect_size_d": row.effect_size_d,
                }
            )
    return pd.DataFrame(rows)


def endpoint_trialcount_relationship(subject_endpoint_df: pd.DataFrame, trial_counts_df: pd.DataFrame, endpoint_subset: Iterable[str]) -> pd.DataFrame:
    merged = subject_endpoint_df.merge(trial_counts_df, on=["subject_id", "group"], how="left")
    rows: list[dict[str, Any]] = []
    for endpoint_id in endpoint_subset:
        subset = merged.loc[merged["endpoint_id"] == endpoint_id].copy()
        for scope_name, frame in [("all", subset), (GROUP_A, subset.loc[subset["group"] == GROUP_A]), (GROUP_B, subset.loc[subset["group"] == GROUP_B])]:
            if frame["n_valid_trials"].nunique() < 2:
                corr = np.nan
                p_value = np.nan
            else:
                corr, p_value = stats.pearsonr(frame["n_valid_trials"].to_numpy(dtype=float), frame["value"].to_numpy(dtype=float))
            rows.append(
                {
                    "endpoint_id": endpoint_id,
                    "scope": scope_name,
                    "pearson_r": float(corr) if np.isfinite(corr) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "n_subjects": int(len(frame)),
                }
            )
    return pd.DataFrame(rows)

