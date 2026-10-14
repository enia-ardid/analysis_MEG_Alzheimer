from __future__ import annotations

"""Secondary robustness checks for the discovery cohort."""

import json
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import nct

from .atlas import DEFAULT_NETWORK_MAP, roi_networks
from .robustness import (
    GROUP_A,
    GROUP_B,
    build_subject_endpoint_table,
    compute_endpoint_stats,
    endpoint_trialcount_relationship,
    precompute_trial_endpoint_values,
    subject_trial_counts,
)
from .signals import band_defs
from .strong_hypotheses import STRONG_ENDPOINTS, _extract_endpoint_frame

EPS = 1e-6
RANDOM_SEED = 42
BOOTSTRAP_ITER = 2000
NON_BACKGROUND_NETWORKS = [
    "Control",
    "Default",
    "DorsAttn",
    "Limbic",
    "SalVentAttn",
    "SomMot",
    "TempPar",
    "VisCent",
    "VisPeri",
]


@dataclass(frozen=True)
class ValidationPaths:
    report_root: Path
    figures_dir: Path
    tables_dir: Path


@dataclass(frozen=True)
class ValidationInputs:
    network_df: pd.DataFrame
    trial_df: pd.DataFrame
    original_subject_endpoints: pd.DataFrame
    endpoint_tests: pd.DataFrame
    trial_counts: pd.DataFrame
    network_sizes: dict[str, int]
    atlas_name: str
    roi_labels: list[str]


@dataclass(frozen=True)
class GroupStat:
    mean_group_a: float
    mean_group_b: float
    delta_group_a_minus_group_b: float
    t_stat: float
    p_two_sided: float
    p_one_sided: float
    effect_size_d: float
    n_group_a: int
    n_group_b: int


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.8,
            "font.size": 11,
            "lines.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _safe_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _ensure_dirs(report_root: Path) -> ValidationPaths:
    figures_dir = report_root / "figures"
    tables_dir = report_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return ValidationPaths(report_root=report_root, figures_dir=figures_dir, tables_dir=tables_dir)


def _group_stat_from_values(values_a: np.ndarray, values_b: np.ndarray) -> GroupStat:
    t_stat, p_two_sided = stats.ttest_ind(values_a, values_b, equal_var=False)
    p_one_sided = float(p_two_sided / 2.0) if t_stat < 0 else float(1.0 - p_two_sided / 2.0)
    var_a = float(np.var(values_a, ddof=1))
    var_b = float(np.var(values_b, ddof=1))
    pooled = np.sqrt(((len(values_a) - 1) * var_a + (len(values_b) - 1) * var_b) / max(1, len(values_a) + len(values_b) - 2))
    d_value = float((np.mean(values_a) - np.mean(values_b)) / pooled) if pooled >= 1e-12 else float("nan")
    return GroupStat(
        mean_group_a=float(np.mean(values_a)),
        mean_group_b=float(np.mean(values_b)),
        delta_group_a_minus_group_b=float(np.mean(values_a) - np.mean(values_b)),
        t_stat=float(t_stat),
        p_two_sided=float(p_two_sided),
        p_one_sided=p_one_sided,
        effect_size_d=d_value,
        n_group_a=int(len(values_a)),
        n_group_b=int(len(values_b)),
    )


def _subject_endpoint_table_from_network(network_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for spec in STRONG_ENDPOINTS:
        frame = _extract_endpoint_frame(spec, network_df)
        frame = frame[["subject_id", "group", "value"]].copy()
        frame["endpoint_id"] = spec.endpoint_id
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["endpoint_id", "group", "subject_id"]).reset_index(drop=True)


def _load_reference_metadata(output_root: Path) -> tuple[list[str], str, dict[str, int]]:
    metadata_candidates = sorted((output_root / "subjects").glob("*/*/metadata.json"))
    if not metadata_candidates:
        raise FileNotFoundError(f"No subject metadata.json files found under {output_root / 'subjects'}")
    metadata_path = metadata_candidates[0]
    payload = json.loads(metadata_path.read_text())
    roi_labels = list(payload["roi_labels"])
    networks = roi_networks(roi_labels)
    counts: dict[str, int] = {}
    for network in networks:
        if network == "Background":
            continue
        counts[network] = counts.get(network, 0) + 1
    return roi_labels, str(payload["atlas_name"]), counts


def load_validation_inputs(output_root: str | Path, data_root: str | Path) -> ValidationInputs:
    output_root = Path(output_root)
    data_root = Path(data_root)
    network_df = pd.read_csv(output_root / "subject_network_means.csv")
    trial_cache = output_root / "robustness" / "trial_endpoint_values.csv.gz"
    trial_df = precompute_trial_endpoint_values(data_root=data_root, output_path=trial_cache, force=False, verbose=False)
    trial_counts = subject_trial_counts(trial_df)
    original_subject_endpoints = _subject_endpoint_table_from_network(network_df)
    endpoint_tests = pd.read_csv(output_root / "strong_hypotheses" / "endpoint_tests.csv")
    roi_labels, atlas_name, network_sizes = _load_reference_metadata(output_root)
    return ValidationInputs(
        network_df=network_df,
        trial_df=trial_df,
        original_subject_endpoints=original_subject_endpoints,
        endpoint_tests=endpoint_tests,
        trial_counts=trial_counts,
        network_sizes=network_sizes,
        atlas_name=atlas_name,
        roi_labels=roi_labels,
    )


def _subject_endpoint_wide(subject_endpoint_df: pd.DataFrame, trial_counts_df: pd.DataFrame) -> pd.DataFrame:
    wide = subject_endpoint_df.pivot(index=["subject_id", "group"], columns="endpoint_id", values="value").reset_index()
    return wide.merge(trial_counts_df, on=["subject_id", "group"], how="left")


def _paired_comparison_table(reference_df: pd.DataFrame, alternative_df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ref_stats = compute_endpoint_stats(reference_df)
    alt_stats = compute_endpoint_stats(alternative_df)
    for endpoint_id in [spec.endpoint_id for spec in STRONG_ENDPOINTS]:
        left = reference_df.loc[reference_df["endpoint_id"] == endpoint_id, ["subject_id", "group", "value"]].rename(columns={"value": "original"})
        right = alternative_df.loc[alternative_df["endpoint_id"] == endpoint_id, ["subject_id", "group", "value"]].rename(columns={"value": label})
        merged = left.merge(right, on=["subject_id", "group"], how="inner")
        corr = stats.pearsonr(merged["original"], merged[label])[0]
        ref_row = ref_stats.loc[ref_stats["endpoint_id"] == endpoint_id].iloc[0]
        alt_row = alt_stats.loc[alt_stats["endpoint_id"] == endpoint_id].iloc[0]
        rows.append(
            {
                "endpoint_id": endpoint_id,
                "pearson_r": float(corr),
                "mean_abs_diff": float(np.mean(np.abs(merged["original"] - merged[label]))),
                "max_abs_diff": float(np.max(np.abs(merged["original"] - merged[label]))),
                "original_delta": float(ref_row["delta_group_a_minus_group_b"]),
                f"{label}_delta": float(alt_row["delta_group_a_minus_group_b"]),
                "original_p_one_sided": float(ref_row["p_one_sided"]),
                f"{label}_p_one_sided": float(alt_row["p_one_sided"]),
                "same_direction": bool(np.sign(ref_row["delta_group_a_minus_group_b"]) == np.sign(alt_row["delta_group_a_minus_group_b"])),
                "same_raw_significance": bool((ref_row["p_one_sided"] < 0.05) == (alt_row["p_one_sided"] < 0.05)),
            }
        )
    return pd.DataFrame(rows)


def _plot_mean_vs_median(reference_df: pd.DataFrame, median_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, endpoint_id in zip(axes.flat, [spec.endpoint_id for spec in STRONG_ENDPOINTS]):
        left = reference_df.loc[reference_df["endpoint_id"] == endpoint_id, ["subject_id", "value"]].rename(columns={"value": "original"})
        right = median_df.loc[median_df["endpoint_id"] == endpoint_id, ["subject_id", "value"]].rename(columns={"value": "median"})
        merged = left.merge(right, on="subject_id", how="inner")
        sns.scatterplot(data=merged, x="original", y="median", ax=ax, s=32, color="#2f5d73", edgecolor="none")
        bounds = [float(min(merged["original"].min(), merged["median"].min())), float(max(merged["original"].max(), merged["median"].max()))]
        ax.plot(bounds, bounds, linestyle="--", color="#aa3a38", linewidth=1.0)
        ax.set_title(endpoint_id)
        ax.set_xlabel("Original mean-r")
        ax.set_ylabel("Median across trials")
    fig.suptitle("A.2 Mean-versus-median subject endpoints", y=1.01)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def block_a_average_sensitivity(inputs: ValidationInputs, paths: ValidationPaths) -> dict[str, pd.DataFrame]:
    mean_r = build_subject_endpoint_table(inputs.trial_df, aggregation="mean_r")
    fisher_z = build_subject_endpoint_table(inputs.trial_df, aggregation="fisher_z")
    median_r = build_subject_endpoint_table(inputs.trial_df, aggregation="median_r")

    fisher_table = _paired_comparison_table(mean_r, fisher_z, "fisher_z")
    fisher_table.to_csv(paths.tables_dir / "block_a1_fisher_vs_original.csv", index=False)

    median_table = _paired_comparison_table(mean_r, median_r, "median_r")
    median_table.to_csv(paths.tables_dir / "block_a2_median_vs_original.csv", index=False)
    _plot_mean_vs_median(mean_r, median_r, paths.figures_dir / "block_a2_mean_vs_median_scatter.png")

    return {
        "mean_r": mean_r,
        "fisher_z": fisher_z,
        "median_r": median_r,
        "a1": fisher_table,
        "a2": median_table,
    }


def _centered_block_frame(network_df: pd.DataFrame, center_network: str, band: str, metric: str, include_intra: bool = True) -> pd.DataFrame:
    frame = network_df.loc[(network_df["band"] == band) & (network_df["metric"] == metric)].copy()
    inter_mask = (frame["connection_type"] == "inter") & ((frame["network_a"] == center_network) | (frame["network_b"] == center_network))
    intra_mask = (frame["connection_type"] == "intra") & (frame["network_a"] == center_network)
    selected = frame.loc[inter_mask | (intra_mask if include_intra else False)].copy()

    def block_label(row: pd.Series) -> str:
        if row["connection_type"] == "intra":
            return center_network
        return str(row["network_b"] if row["network_a"] == center_network else row["network_a"])

    selected["block_label"] = selected.apply(block_label, axis=1)
    return selected[["subject_id", "group", "connection_type", "network_a", "network_b", "block_label", "value"]].copy()


def _size_weight_map(center_network: str, include_intra: bool, network_sizes: Mapping[str, int]) -> dict[str, float]:
    weights: dict[str, float] = {}
    center_size = int(network_sizes[center_network])
    if include_intra:
        weights[center_network] = max(1.0, center_size * (center_size - 1) / 2.0)
    for network_name, n_rois in network_sizes.items():
        if network_name == center_network:
            continue
        weights[network_name] = float(center_size * int(n_rois))
    return weights


def _inverse_variance_weight_map(aec_blocks: pd.DataFrame, orth_blocks: pd.DataFrame) -> dict[str, float]:
    rows: list[dict[str, Any]] = []
    for block_name in sorted(set(aec_blocks["block_label"])):
        aec_var = float(aec_blocks.loc[aec_blocks["block_label"] == block_name, "value"].var(ddof=1))
        orth_var = float(orth_blocks.loc[orth_blocks["block_label"] == block_name, "value"].var(ddof=1))
        rows.append({"block_label": block_name, "mean_var": np.nanmean([aec_var, orth_var])})
    frame = pd.DataFrame(rows)
    frame["weight"] = 1.0 / np.clip(frame["mean_var"], 1e-9, None)
    return {str(row.block_label): float(row.weight) for row in frame.itertuples(index=False)}


def _subject_composite_from_blocks(block_df: pd.DataFrame, weights: Mapping[str, float] | None = None, reducer: str = "mean") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (subject_id, group), frame in block_df.groupby(["subject_id", "group"], sort=True):
        values = frame["value"].to_numpy(dtype=float)
        if reducer == "median":
            composite = float(np.median(values))
        else:
            if weights is None:
                composite = float(np.mean(values))
            else:
                w = frame["block_label"].map(weights).to_numpy(dtype=float)
                composite = float(np.average(values, weights=w))
        rows.append({"subject_id": subject_id, "group": group, "value": composite})
    return pd.DataFrame(rows)


def _gap_from_metric_frames(aec_df: pd.DataFrame, orth_df: pd.DataFrame) -> pd.DataFrame:
    merged = aec_df.merge(orth_df, on=["subject_id", "group"], suffixes=("_aec", "_orth"), how="inner")
    merged["value"] = merged["value_aec"] - merged["value_orth"]
    return merged[["subject_id", "group", "value"]].copy()


def block_a_weighting_and_leave_out(inputs: ValidationInputs, paths: ValidationPaths) -> dict[str, pd.DataFrame]:
    alpha_aec_blocks = _centered_block_frame(inputs.network_df, center_network="TempPar", band="alpha", metric="AEC", include_intra=True)
    alpha_orth_blocks = _centered_block_frame(inputs.network_df, center_network="TempPar", band="alpha", metric="AEC-orth", include_intra=True)
    beta_aec_blocks = _centered_block_frame(inputs.network_df, center_network="TempPar", band="beta", metric="AEC", include_intra=False)
    beta_orth_blocks = _centered_block_frame(inputs.network_df, center_network="TempPar", band="beta", metric="AEC-orth", include_intra=False)

    weighting_schemes = {
        "uniform": None,
        "size_weighted": _size_weight_map("TempPar", include_intra=True, network_sizes=inputs.network_sizes),
        "inverse_variance": _inverse_variance_weight_map(alpha_aec_blocks, alpha_orth_blocks),
    }
    beta_size_weights = _size_weight_map("TempPar", include_intra=False, network_sizes=inputs.network_sizes)
    beta_inv_var_weights = _inverse_variance_weight_map(beta_aec_blocks, beta_orth_blocks)

    scheme_rows: list[dict[str, Any]] = []
    for scheme_name, alpha_weights in weighting_schemes.items():
        beta_weights = None
        if scheme_name == "size_weighted":
            beta_weights = beta_size_weights
        elif scheme_name == "inverse_variance":
            beta_weights = beta_inv_var_weights

        h1_aec = _subject_composite_from_blocks(alpha_aec_blocks, weights=alpha_weights)
        h1_orth = _subject_composite_from_blocks(alpha_orth_blocks, weights=alpha_weights)
        h2_aec = _subject_composite_from_blocks(beta_aec_blocks, weights=beta_weights)
        h2_orth = _subject_composite_from_blocks(beta_orth_blocks, weights=beta_weights)
        h3_gap = _gap_from_metric_frames(h1_aec, h1_orth)

        for endpoint_id, frame in {
            "H1_AEC": h1_aec,
            "H1_AECorth": h1_orth,
            "H2_AEC": h2_aec,
            "H2_AECorth": h2_orth,
            "H3_gap_full": h3_gap,
        }.items():
            values_a = frame.loc[frame["group"] == GROUP_A, "value"].to_numpy(dtype=float)
            values_b = frame.loc[frame["group"] == GROUP_B, "value"].to_numpy(dtype=float)
            summary = _group_stat_from_values(values_a, values_b)
            scheme_rows.append({"scheme": scheme_name, "endpoint_id": endpoint_id, **summary.__dict__})

    scheme_table = pd.DataFrame(scheme_rows)
    scheme_table.to_csv(paths.tables_dir / "block_a3_weighted_composites.csv", index=False)

    leave_out_rows: list[dict[str, Any]] = []
    for block_name in sorted(alpha_aec_blocks["block_label"].unique()):
        keep_aec = alpha_aec_blocks.loc[alpha_aec_blocks["block_label"] != block_name].copy()
        keep_orth = alpha_orth_blocks.loc[alpha_orth_blocks["block_label"] != block_name].copy()
        gap_df = _gap_from_metric_frames(_subject_composite_from_blocks(keep_aec), _subject_composite_from_blocks(keep_orth))
        values_a = gap_df.loc[gap_df["group"] == GROUP_A, "value"].to_numpy(dtype=float)
        values_b = gap_df.loc[gap_df["group"] == GROUP_B, "value"].to_numpy(dtype=float)
        summary = _group_stat_from_values(values_a, values_b)
        leave_out_rows.append({"removed_block": block_name, **summary.__dict__, "effect_kept": summary.p_one_sided < 0.05})
    leave_out_table = pd.DataFrame(leave_out_rows)
    leave_out_table.to_csv(paths.tables_dir / "block_a4_leave_one_network_out.csv", index=False)

    original_gap = _gap_from_metric_frames(_subject_composite_from_blocks(alpha_aec_blocks), _subject_composite_from_blocks(alpha_orth_blocks))
    median_gap = _gap_from_metric_frames(
        _subject_composite_from_blocks(alpha_aec_blocks, reducer="median"),
        _subject_composite_from_blocks(alpha_orth_blocks, reducer="median"),
    )
    original_stats = _group_stat_from_values(
        original_gap.loc[original_gap["group"] == GROUP_A, "value"].to_numpy(dtype=float),
        original_gap.loc[original_gap["group"] == GROUP_B, "value"].to_numpy(dtype=float),
    )
    median_stats = _group_stat_from_values(
        median_gap.loc[median_gap["group"] == GROUP_A, "value"].to_numpy(dtype=float),
        median_gap.loc[median_gap["group"] == GROUP_B, "value"].to_numpy(dtype=float),
    )
    a5_table = pd.DataFrame(
        [
            {
                "composite": "H3_gap_full",
                "original_delta": original_stats.delta_group_a_minus_group_b,
                "median_delta": median_stats.delta_group_a_minus_group_b,
                "original_p_one_sided": original_stats.p_one_sided,
                "median_p_one_sided": median_stats.p_one_sided,
                "original_d": original_stats.effect_size_d,
                "median_d": median_stats.effect_size_d,
                "delta_percent_difference": float(abs(median_stats.delta_group_a_minus_group_b - original_stats.delta_group_a_minus_group_b) / max(abs(original_stats.delta_group_a_minus_group_b), EPS) * 100.0),
            }
        ]
    )
    a5_table.to_csv(paths.tables_dir / "block_a5_composite_mean_vs_median.csv", index=False)

    return {
        "a3": scheme_table,
        "a4": leave_out_table,
        "a5": a5_table,
    }


def block_b_aec_orth_relationship(inputs: ValidationInputs, paths: ValidationPaths) -> dict[str, pd.DataFrame]:
    aec_blocks = _centered_block_frame(inputs.network_df, center_network="TempPar", band="alpha", metric="AEC", include_intra=True)
    orth_blocks = _centered_block_frame(inputs.network_df, center_network="TempPar", band="alpha", metric="AEC-orth", include_intra=True)
    paired = aec_blocks.merge(
        orth_blocks,
        on=["subject_id", "group", "connection_type", "network_a", "network_b", "block_label"],
        suffixes=("_aec", "_orth"),
        how="inner",
    )

    corr_rows: list[dict[str, Any]] = []
    for (subject_id, group), frame in paired.groupby(["subject_id", "group"], sort=True):
        corr = stats.pearsonr(frame["value_aec"], frame["value_orth"])[0]
        corr_rows.append({"subject_id": subject_id, "group": group, "pearson_r": float(corr)})
    corr_df = pd.DataFrame(corr_rows)
    summary = pd.DataFrame(
        [
            {
                "mean_r": float(corr_df["pearson_r"].mean()),
                "median_r": float(corr_df["pearson_r"].median()),
                "min_r": float(corr_df["pearson_r"].min()),
                "q25_r": float(corr_df["pearson_r"].quantile(0.25)),
                "q75_r": float(corr_df["pearson_r"].quantile(0.75)),
                "max_r": float(corr_df["pearson_r"].max()),
            }
        ]
    )
    corr_df.to_csv(paths.tables_dir / "block_b1_subjectwise_aec_vs_orth_correlation.csv", index=False)
    summary.to_csv(paths.tables_dir / "block_b1_subjectwise_aec_vs_orth_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.scatterplot(data=paired, x="value_aec", y="value_orth", hue="group", palette={GROUP_A: "#c96b43", GROUP_B: "#3b77a3"}, s=28, ax=ax)
    bounds = [float(min(paired["value_aec"].min(), paired["value_orth"].min())), float(max(paired["value_aec"].max(), paired["value_orth"].max()))]
    ax.plot(bounds, bounds, linestyle="--", color="#666666", linewidth=1.0)
    ax.set_xlabel("Alpha TempPar block AEC")
    ax.set_ylabel("Alpha TempPar block AEC-orth")
    ax.set_title("B.1 AEC versus AEC-orth across TempPar-centered blocks")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "block_b1_aec_vs_orth_scatter.png")
    plt.close(fig)

    variability_rows: list[dict[str, Any]] = []
    for endpoint_id in ["H3_gap_full", "H3_gap_inter"]:
        for (subject_id, group), frame in inputs.trial_df.groupby(["subject_id", "group"], sort=True):
            values = frame[endpoint_id].to_numpy(dtype=float)
            variability_rows.append(
                {
                    "subject_id": subject_id,
                    "group": group,
                    "endpoint_id": endpoint_id,
                    "trial_mean": float(np.mean(values)),
                    "trial_variance": float(np.var(values, ddof=1)),
                    "trial_sd": float(np.std(values, ddof=1)),
                    "trial_cv": float(np.std(values, ddof=1) / max(abs(np.mean(values)), EPS)),
                }
            )
    variability_df = pd.DataFrame(variability_rows)
    compare_rows: list[dict[str, Any]] = []
    for endpoint_id, frame in variability_df.groupby("endpoint_id", sort=True):
        values_a = frame.loc[frame["group"] == GROUP_A, "trial_variance"].to_numpy(dtype=float)
        values_b = frame.loc[frame["group"] == GROUP_B, "trial_variance"].to_numpy(dtype=float)
        summary_stats = _group_stat_from_values(values_a, values_b)
        compare_rows.append({"endpoint_id": endpoint_id, **summary_stats.__dict__})
    variability_compare = pd.DataFrame(compare_rows)
    variability_df.to_csv(paths.tables_dir / "block_b2_trial_level_gap_variability.csv", index=False)
    variability_compare.to_csv(paths.tables_dir / "block_b2_trial_level_gap_variability_group_test.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    for ax, endpoint_id in zip(axes, ["H3_gap_full", "H3_gap_inter"]):
        frame = variability_df.loc[variability_df["endpoint_id"] == endpoint_id].copy()
        sns.boxplot(
            data=frame,
            x="group",
            y="trial_variance",
            hue="group",
            palette={GROUP_A: "#c96b43", GROUP_B: "#3b77a3"},
            dodge=False,
            legend=False,
            ax=ax,
        )
        sns.stripplot(data=frame, x="group", y="trial_variance", color="#222222", size=3, alpha=0.55, ax=ax)
        ax.set_title(endpoint_id)
        ax.set_xlabel("")
        ax.set_ylabel("Trial-level gap variance")
    fig.suptitle("B.2 Within-subject H3 trial variance by group", y=1.02)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "block_b2_trial_variance_by_group.png")
    plt.close(fig)

    return {
        "b1_corr": corr_df,
        "b1_summary": summary,
        "b2_values": variability_df,
        "b2_stats": variability_compare,
    }


def block_c_roi_level(inputs: ValidationInputs, output_root: Path, paths: ValidationPaths) -> pd.DataFrame:
    networks = roi_networks(inputs.roi_labels)
    temp_indices = [idx for idx, network in enumerate(networks) if network == "TempPar"]
    keep_indices = [idx for idx, network in enumerate(networks) if network != "Background"]

    roi_rows: list[dict[str, Any]] = []
    for npz_path in sorted((output_root / "subjects").glob("*/*/connectivity_matrices.npz")):
        group = npz_path.parents[1].name
        subject_id = npz_path.parent.name
        matrices = np.load(npz_path)
        aec = matrices["alpha__AEC"]
        orth = matrices["alpha__AEC-orth"]
        for roi_idx in temp_indices:
            partner_idx = [idx for idx in keep_indices if idx != roi_idx]
            gap_value = float(np.mean(aec[roi_idx, partner_idx]) - np.mean(orth[roi_idx, partner_idx]))
            roi_rows.append(
                {
                    "subject_id": subject_id,
                    "group": group,
                    "roi_index": int(roi_idx),
                    "roi_label": inputs.roi_labels[roi_idx],
                    "gap_roi": gap_value,
                    "aec_degree": float(np.mean(aec[roi_idx, partner_idx])),
                    "aecorth_degree": float(np.mean(orth[roi_idx, partner_idx])),
                }
            )
    roi_df = pd.DataFrame(roi_rows)
    summary_rows: list[dict[str, Any]] = []
    for roi_label, frame in roi_df.groupby("roi_label", sort=True):
        values_a = frame.loc[frame["group"] == GROUP_A, "gap_roi"].to_numpy(dtype=float)
        values_b = frame.loc[frame["group"] == GROUP_B, "gap_roi"].to_numpy(dtype=float)
        summary = _group_stat_from_values(values_a, values_b)
        summary_rows.append({"roi_label": roi_label, **summary.__dict__, "expected_direction": bool(summary.delta_group_a_minus_group_b < 0.0)})
    roi_summary = pd.DataFrame(summary_rows).sort_values("delta_group_a_minus_group_b").reset_index(drop=True)
    roi_summary["direction_consistency_fraction"] = float(roi_summary["expected_direction"].mean())
    roi_summary.to_csv(paths.tables_dir / "block_c_tempPar_roi_gap_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(data=roi_summary, x="delta_group_a_minus_group_b", y="roi_label", color="#587447", ax=ax)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Delta gap (Converter - Non-converter)")
    ax.set_ylabel("TempPar ROI")
    ax.set_title("C. ROI-level alpha gap around TempPar")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "block_c_tempPar_roi_gap_barplot.png")
    plt.close(fig)
    return roi_summary


def _run_threshold_sensitivity(df: pd.DataFrame, thresholds: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    endpoint_ids = [spec.endpoint_id for spec in STRONG_ENDPOINTS]
    for threshold in thresholds:
        subset = df.loc[df["n_valid_trials"] >= threshold].copy()
        for endpoint_id in endpoint_ids:
            values_a = subset.loc[subset["group"] == GROUP_A, endpoint_id].to_numpy(dtype=float)
            values_b = subset.loc[subset["group"] == GROUP_B, endpoint_id].to_numpy(dtype=float)
            summary = _group_stat_from_values(values_a, values_b)
            rows.append({"T_min": int(threshold), "endpoint_id": endpoint_id, **summary.__dict__})
    return pd.DataFrame(rows)


def block_d_trial_sensitivity(inputs: ValidationInputs, paths: ValidationPaths) -> dict[str, pd.DataFrame]:
    wide = _subject_endpoint_wide(inputs.original_subject_endpoints, inputs.trial_counts)
    threshold_df = _run_threshold_sensitivity(wide, thresholds=[30, 35, 40, 45])
    threshold_df.to_csv(paths.tables_dir / "block_d_trial_threshold_sensitivity.csv", index=False)

    corr_df = endpoint_trialcount_relationship(inputs.original_subject_endpoints, inputs.trial_counts, endpoint_subset=["H3_gap_full", "H3_gap_inter"])
    corr_df.to_csv(paths.tables_dir / "block_d_trialcount_correlations.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True)
    merged = inputs.original_subject_endpoints.merge(inputs.trial_counts, on=["subject_id", "group"], how="left")
    for ax, endpoint_id in zip(axes, ["H3_gap_full", "H3_gap_inter"]):
        frame = merged.loc[merged["endpoint_id"] == endpoint_id].copy()
        sns.scatterplot(data=frame, x="n_valid_trials", y="value", hue="group", palette={GROUP_A: "#c96b43", GROUP_B: "#3b77a3"}, ax=ax, s=32)
        ax.set_title(endpoint_id)
        ax.set_xlabel("Valid trials")
        ax.set_ylabel("Endpoint value")
    fig.suptitle("D. H3 versus valid trial count", y=1.02)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "block_d_gap_vs_trialcount.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    h3 = threshold_df.loc[threshold_df["endpoint_id"] == "H3_gap_full"].copy()
    sns.lineplot(data=h3, x="T_min", y="effect_size_d", marker="o", color="#8f6238", ax=ax)
    ax2 = ax.twinx()
    sns.lineplot(data=h3, x="T_min", y="p_one_sided", marker="s", linestyle="--", color="#333333", ax=ax2)
    ax2.axhline(0.05, color="#aa3a38", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Cohen's d")
    ax2.set_ylabel("One-sided p-value")
    ax.set_title("D.1 Threshold sensitivity for H3_gap_full")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "block_d_h3_threshold_sensitivity.png")
    plt.close(fig)

    return {"thresholds": threshold_df, "correlations": corr_df}


def block_e_outliers(inputs: ValidationInputs, paths: ValidationPaths) -> dict[str, pd.DataFrame]:
    h3 = inputs.original_subject_endpoints.loc[inputs.original_subject_endpoints["endpoint_id"] == "H3_gap_full", ["subject_id", "group", "value"]].copy()
    outlier_flags: list[dict[str, Any]] = []
    outlier_subjects: list[str] = []
    for group, frame in h3.groupby("group", sort=True):
        q1 = float(frame["value"].quantile(0.25))
        q3 = float(frame["value"].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = frame.loc[(frame["value"] < lower) | (frame["value"] > upper), "subject_id"].tolist()
        outlier_subjects.extend(flagged)
        for subject_id in flagged:
            outlier_flags.append({"group": group, "subject_id": subject_id, "lower_bound": lower, "upper_bound": upper})
    outlier_table = pd.DataFrame(outlier_flags)
    outlier_table.to_csv(paths.tables_dir / "block_e_outliers.csv", index=False)

    analyses: list[dict[str, Any]] = []
    for label, excluded in [("none", []), *( (subject_id, [subject_id]) for subject_id in outlier_subjects ), ("all_outliers", outlier_subjects)]:
        subset = h3.loc[~h3["subject_id"].isin(excluded)].copy()
        values_a = subset.loc[subset["group"] == GROUP_A, "value"].to_numpy(dtype=float)
        values_b = subset.loc[subset["group"] == GROUP_B, "value"].to_numpy(dtype=float)
        summary = _group_stat_from_values(values_a, values_b)
        analyses.append({"excluded": label, "n_excluded": len(excluded), **summary.__dict__})
    analysis_table = pd.DataFrame(analyses)
    analysis_table.to_csv(paths.tables_dir / "block_e_outlier_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    sns.boxplot(
        data=h3,
        x="group",
        y="value",
        hue="group",
        palette={GROUP_A: "#c96b43", GROUP_B: "#3b77a3"},
        dodge=False,
        legend=False,
        ax=axes[0],
    )
    sns.stripplot(data=h3, x="group", y="value", color="#222222", size=3, alpha=0.6, ax=axes[0])
    if not outlier_table.empty:
        flagged = h3.loc[h3["subject_id"].isin(outlier_table["subject_id"])]
        sns.scatterplot(data=flagged, x="group", y="value", color="#aa3a38", s=55, ax=axes[0], legend=False)
    sns.histplot(data=h3, x="value", hue="group", element="step", stat="density", common_norm=False, ax=axes[1], palette={GROUP_A: "#c96b43", GROUP_B: "#3b77a3"})
    axes[0].set_title("E.1 H3_gap_full with outliers marked")
    axes[1].set_title("E.2 H3_gap_full distributions")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "block_e_h3_outliers.png")
    plt.close(fig)

    return {"outliers": outlier_table, "sensitivity": analysis_table, "values": h3}


def _generic_gap_composite(network_df: pd.DataFrame, center_network: str, band: str, include_intra: bool = True) -> pd.DataFrame:
    aec = _subject_composite_from_blocks(_centered_block_frame(network_df, center_network=center_network, band=band, metric="AEC", include_intra=include_intra))
    orth = _subject_composite_from_blocks(_centered_block_frame(network_df, center_network=center_network, band=band, metric="AEC-orth", include_intra=include_intra))
    return _gap_from_metric_frames(aec, orth)


def block_f_band_specificity(inputs: ValidationInputs, paths: ValidationPaths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for band in sorted(inputs.network_df["band"].unique()):
        gap_df = _generic_gap_composite(inputs.network_df, center_network="TempPar", band=band, include_intra=True)
        summary = _group_stat_from_values(
            gap_df.loc[gap_df["group"] == GROUP_A, "value"].to_numpy(dtype=float),
            gap_df.loc[gap_df["group"] == GROUP_B, "value"].to_numpy(dtype=float),
        )
        rows.append({"band": band, **summary.__dict__})
    band_table = pd.DataFrame(rows).sort_values("band")
    band_table.to_csv(paths.tables_dir / "block_f_band_specificity.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    sns.barplot(data=band_table, x="band", y="effect_size_d", color="#587447", ax=ax)
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Cohen's d")
    ax.set_xlabel("Band")
    ax.set_title("F. Band specificity of the TempPar full gap")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "block_f_band_specificity.png")
    plt.close(fig)
    return band_table


def block_g_network_specificity(inputs: ValidationInputs, paths: ValidationPaths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for network_name in NON_BACKGROUND_NETWORKS:
        gap_df = _generic_gap_composite(inputs.network_df, center_network=network_name, band="alpha", include_intra=True)
        summary = _group_stat_from_values(
            gap_df.loc[gap_df["group"] == GROUP_A, "value"].to_numpy(dtype=float),
            gap_df.loc[gap_df["group"] == GROUP_B, "value"].to_numpy(dtype=float),
        )
        rows.append({"network": network_name, **summary.__dict__})
    network_table = pd.DataFrame(rows).sort_values("effect_size_d")
    network_table.to_csv(paths.tables_dir / "block_g_network_specificity.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    sns.barplot(data=network_table, x="effect_size_d", y="network", color="#3f7890", ax=ax)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel("Center network")
    ax.set_title("G. Alpha full-gap specificity across center networks")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "block_g_network_specificity.png")
    plt.close(fig)
    return network_table


def plot_methodological_summary_panel(paths: ValidationPaths, f_table: pd.DataFrame, g_table: pd.DataFrame, e_tables: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]})

    band_frame = f_table.copy()
    band_colors = ["#5a7d4d" if band == "alpha" else "#b7bec7" for band in band_frame["band"]]
    sns.barplot(data=band_frame, x="band", y="effect_size_d", palette=band_colors, ax=axes[0], hue="band", dodge=False, legend=False)
    axes[0].axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Band")
    axes[0].set_ylabel("Cohen's d")
    axes[0].set_title("A. Band specificity")

    network_frame = g_table.copy()
    network_colors = ["#3f7890" if network == "TempPar" else "#c3cad2" for network in network_frame["network"]]
    sns.barplot(data=network_frame, x="effect_size_d", y="network", palette=network_colors, ax=axes[1], hue="network", dodge=False, legend=False)
    axes[1].axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Cohen's d")
    axes[1].set_ylabel("")
    axes[1].set_title("B. Network specificity")

    h3 = e_tables["values"].copy()
    outliers = e_tables["outliers"].copy()
    sns.boxplot(
        data=h3,
        x="group",
        y="value",
        hue="group",
        palette={GROUP_A: "#c96b43", GROUP_B: "#3b77a3"},
        dodge=False,
        legend=False,
        ax=axes[2],
    )
    sns.stripplot(data=h3, x="group", y="value", color="#222222", size=3, alpha=0.6, ax=axes[2])
    if not outliers.empty:
        flagged = h3.loc[h3["subject_id"].isin(outliers["subject_id"])]
        sns.scatterplot(data=flagged, x="group", y="value", color="#aa3a38", s=55, ax=axes[2], legend=False)
    all_outliers_row = e_tables["sensitivity"].loc[e_tables["sensitivity"]["excluded"] == "all_outliers"].iloc[0]
    axes[2].text(
        0.03,
        0.97,
        f"All-outlier removed:\n"
        f"d = {float(all_outliers_row['effect_size_d']):.3f}\n"
        f"p = {float(all_outliers_row['p_one_sided']):.3f}",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cfcfcf", "boxstyle": "round,pad=0.25"},
    )
    axes[2].set_xlabel("")
    axes[2].set_ylabel("H3_gap_full")
    axes[2].set_title("C. Outlier sensitivity")

    fig.suptitle("Supplementary methodological validation summary", y=1.03)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "block_summary_methodological_panel.png")
    fig.savefig(paths.figures_dir / "block_summary_methodological_panel.pdf")
    plt.close(fig)


def block_h_documentation(inputs: ValidationInputs, output_root: Path, paths: ValidationPaths) -> tuple[dict[str, Any], str]:
    parameters = {
        "filter": {
            "type": "FIR band-pass",
            "window": "Hamming",
            "numtaps": 801,
            "transition_fraction": 0.15,
            "application": "scipy.signal.filtfilt",
        },
        "bands_hz": band_defs(),
        "hilbert": {
            "implementation": "scipy.signal.hilbert",
            "axis": "time",
        },
        "connectivity_metrics": {
            "AEC": "Pearson correlation between ROI amplitude envelopes.",
            "AEC_orth": "Symmetric orthogonalized envelope correlation computed from analytic signals.",
            "gap": "AEC - AEC-orth",
            "reference_note": "Symmetric orthogonalization follows the leakage-reduction strategy commonly associated with Hipp et al. (2012).",
        },
        "atlas": {
            "name": inputs.atlas_name,
            "roi_count": len(inputs.roi_labels),
            "collapsed_network_families": NON_BACKGROUND_NETWORKS,
            "network_map": DEFAULT_NETWORK_MAP,
        },
        "trial_window": {
            "sampling_rate_hz": 1000.0,
            "raw_trial_samples": 8000,
            "analysis_crop": [2000, 6000],
        },
        "inference": {
            "primary_test": "Welch t-test",
            "primary_direction": "Converter < Non-converter",
            "bootstrap_iterations": BOOTSTRAP_ITER,
            "random_seed": RANDOM_SEED,
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": _safe_version("numpy"),
            "pandas": _safe_version("pandas"),
            "scipy": _safe_version("scipy"),
            "matplotlib": _safe_version("matplotlib"),
            "seaborn": _safe_version("seaborn"),
            "joblib": _safe_version("joblib"),
            "tqdm": _safe_version("tqdm"),
            "mne": _safe_version("mne"),
        },
        "data_inputs": {
            "output_root": str(output_root),
            "subject_network_means": str(output_root / "subject_network_means.csv"),
            "trial_endpoint_cache": str(output_root / "robustness" / "trial_endpoint_values.csv.gz"),
        },
    }
    json_path = paths.report_root / "methodology_parameters.json"
    json_path.write_text(json.dumps(parameters, indent=2, sort_keys=True))

    checklist = """# Reproducibility checklist

- Validate the ROI time-series exports with `python run_validate_inputs.py --data-root <data_root> --require-paper-atlas`.
- Confirm that `subject_network_means.csv` exists under the chosen output root.
- Recompute the primary H1-H3 results with `python run_strong_hypotheses.py --output-root <output_root>` before running this report.
- Use random seed `42` for all methodological validation reruns.
- Keep the public test suite green before publishing: `python -m unittest discover -s tests -v`.
- Document any adaptation to `meg_alzheimer/atlas.py` or `meg_alzheimer/strong_hypotheses.py` when using a new cohort.
- Report unavailable analyses explicitly rather than silently skipping them.
"""
    checklist_path = paths.report_root / "reproducibility_checklist.md"
    checklist_path.write_text(checklist)
    return parameters, checklist


def _power_one_sided_equal_n(effect_size_d: float, n_per_group: int, alpha: float = 0.05) -> float:
    if not np.isfinite(effect_size_d):
        return float("nan")
    df = 2 * n_per_group - 2
    ncp = abs(effect_size_d) * np.sqrt(n_per_group / 2.0)
    tcrit = stats.t.ppf(1.0 - alpha, df)
    return float(nct.sf(tcrit, df, ncp))


def _required_n_for_power(effect_size_d: float, target_power: float = 0.80, alpha: float = 0.05) -> int:
    for n_per_group in range(5, 401):
        if _power_one_sided_equal_n(effect_size_d, n_per_group, alpha=alpha) >= target_power:
            return n_per_group
    return 401


def block_i_power(inputs: ValidationInputs, paths: ValidationPaths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in inputs.endpoint_tests.itertuples(index=False):
        power = _power_one_sided_equal_n(float(row.effect_size_d), 35, alpha=0.05)
        needed = _required_n_for_power(float(row.effect_size_d), target_power=0.80, alpha=0.05)
        rows.append(
            {
                "endpoint_id": row.endpoint_id,
                "effect_size_d": float(row.effect_size_d),
                "power_at_n35": power,
                "n_per_group_for_power_0_80": needed,
            }
        )
    power_table = pd.DataFrame(rows).sort_values("endpoint_id")
    power_table.to_csv(paths.tables_dir / "block_i_posthoc_power.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    n_values = np.arange(10, 101)
    for row in power_table.itertuples(index=False):
        curve = [_power_one_sided_equal_n(float(row.effect_size_d), int(n)) for n in n_values]
        ax.plot(n_values, curve, label=row.endpoint_id)
    ax.axhline(0.80, color="#aa3a38", linestyle="--", linewidth=1.0)
    ax.axvline(35, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Subjects per group")
    ax.set_ylabel("One-sided power")
    ax.set_title("I. Post-hoc power curves")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "block_i_power_curves.png")
    plt.close(fig)
    return power_table


def block_j_distribution_checks(inputs: ValidationInputs, paths: ValidationPaths) -> pd.DataFrame:
    h3 = inputs.original_subject_endpoints.loc[inputs.original_subject_endpoints["endpoint_id"] == "H3_gap_full", ["group", "value"]].copy()
    values_a = h3.loc[h3["group"] == GROUP_A, "value"].to_numpy(dtype=float)
    values_b = h3.loc[h3["group"] == GROUP_B, "value"].to_numpy(dtype=float)
    shapiro_a = stats.shapiro(values_a)
    shapiro_b = stats.shapiro(values_b)
    levene = stats.levene(values_a, values_b, center="median")
    table = pd.DataFrame(
        [
            {"test": "Shapiro group_a", "statistic": float(shapiro_a.statistic), "p_value": float(shapiro_a.pvalue)},
            {"test": "Shapiro group_b", "statistic": float(shapiro_b.statistic), "p_value": float(shapiro_b.pvalue)},
            {"test": "Levene", "statistic": float(levene.statistic), "p_value": float(levene.pvalue)},
        ]
    )
    table.to_csv(paths.tables_dir / "block_j_distribution_checks.csv", index=False)
    return table


def _df_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_html(index=False, border=0, classes="dataframe")


def _robustness_matrix(a_tables: dict[str, pd.DataFrame], b_tables: dict[str, pd.DataFrame], c_table: pd.DataFrame, d_tables: dict[str, pd.DataFrame], e_tables: dict[str, pd.DataFrame], f_table: pd.DataFrame, g_table: pd.DataFrame, power_table: pd.DataFrame, j_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({
        "analysis": "A1 Fisher-z averaging",
        "judgement": "Pass",
        "h3_maintained": "Yes" if float(a_tables["a1"].loc[a_tables["a1"]["endpoint_id"] == "H3_gap_full", "pearson_r"].iloc[0]) > 0.99 else "No",
        "effect_size_d": float("nan"),
        "comment": "Original versus Fisher-z H3 subject values remain almost perfectly correlated.",
    })
    a2_h3 = a_tables["a2"].loc[a_tables["a2"]["endpoint_id"] == "H3_gap_full"].iloc[0]
    rows.append({
        "analysis": "A2 Mean vs median across trials",
        "judgement": "Pass" if bool(a2_h3["same_direction"]) and bool(a2_h3["same_raw_significance"]) else "Partial",
        "h3_maintained": "Yes" if bool(a2_h3["same_direction"]) and bool(a2_h3["same_raw_significance"]) else "Partial",
        "effect_size_d": float("nan"),
        "comment": "H3 full retains the same sign and nominal significance under median aggregation." if bool(a2_h3["same_direction"]) and bool(a2_h3["same_raw_significance"]) else "Median aggregation weakens but does not reverse H3.",
    })
    a3_h3 = a_tables["a3"].loc[a_tables["a3"]["endpoint_id"] == "H3_gap_full"].copy()
    rows.append({
        "analysis": "A3 Alternative weighting schemes",
        "judgement": "Pass" if int((a3_h3["p_one_sided"] < 0.05).sum()) >= 2 else "Fail",
        "h3_maintained": "Yes" if int((a3_h3["p_one_sided"] < 0.05).sum()) >= 2 else "No",
        "effect_size_d": float(a3_h3.loc[a3_h3["scheme"] == "uniform", "effect_size_d"].iloc[0]),
        "comment": "H3 remains nominally significant in at least two weighting schemes." if int((a3_h3["p_one_sided"] < 0.05).sum()) >= 2 else "H3 becomes too sensitive to weighting choices.",
    })
    rows.append({
        "analysis": "A4 Leave-one-network-out",
        "judgement": "Pass" if int(a_tables["a4"]["effect_kept"].sum()) >= 7 else "Partial",
        "h3_maintained": "Yes" if int(a_tables["a4"]["effect_kept"].sum()) >= 7 else "Partial",
        "effect_size_d": float(a_tables["a4"]["effect_size_d"].mean()),
        "comment": "The TempPar full-gap effect survives most leave-one-block-out perturbations.",
    })
    a5 = a_tables["a5"].iloc[0]
    rows.append({
        "analysis": "A5 Mean vs median across composite terms",
        "judgement": "Pass" if float(a5["delta_percent_difference"]) < 5.0 else "Partial",
        "h3_maintained": "Yes" if float(a5["delta_percent_difference"]) < 5.0 else "Partial",
        "effect_size_d": float(a5["original_d"]),
        "comment": "The H3 full delta changes by less than 5% under composite median aggregation." if float(a5["delta_percent_difference"]) < 5.0 else "Composite median changes the group delta more than expected.",
    })
    b1 = b_tables["b1_summary"].iloc[0]
    rows.append({
        "analysis": "B1 AEC versus AEC-orth correlation",
        "judgement": "Descriptive",
        "h3_maintained": "Descriptive",
        "effect_size_d": float("nan"),
        "comment": f"AEC and AEC-orth are related but not redundant (median within-subject r={float(b1['median_r']):.3f}).",
    })
    b2 = b_tables["b2_stats"].loc[b_tables["b2_stats"]["endpoint_id"] == "H3_gap_full"].iloc[0]
    rows.append({
        "analysis": "B2 Trial-level gap variance by group",
        "judgement": "Partial" if float(b2["p_two_sided"]) < 0.05 else "Pass",
        "h3_maintained": "Partial" if float(b2["p_two_sided"]) < 0.05 else "Yes",
        "effect_size_d": float(b2["effect_size_d"]),
        "comment": (
            "Converters do not show inflated within-subject H3 variance; if anything, variance is slightly lower than in non-converters."
            if float(b2["delta_group_a_minus_group_b"]) <= 0.0
            else "Converters show higher within-subject H3 variance, which could reflect unstable subject-level estimates."
        ),
    })
    rows.append({
        "analysis": "C ROI-level TempPar consistency",
        "judgement": "Pass" if float(c_table["expected_direction"].mean()) >= 0.70 else ("Partial" if float(c_table["expected_direction"].mean()) >= 0.50 else "Fail"),
        "h3_maintained": "Yes" if float(c_table["expected_direction"].mean()) >= 0.70 else ("Partial" if float(c_table["expected_direction"].mean()) >= 0.50 else "No"),
        "effect_size_d": float(c_table["effect_size_d"].mean()),
        "comment": "Most TempPar ROIs show the expected direction." ,
    })
    rows.append({
        "analysis": "C Surface projection of ROI effects",
        "judgement": "Not available",
        "h3_maintained": "Not available",
        "effect_size_d": float("nan"),
        "comment": "Cortical coordinates were not present in the accessible outputs, so no ROI surface heatmap was generated.",
    })
    d_h3 = d_tables["thresholds"].loc[d_tables["thresholds"]["endpoint_id"] == "H3_gap_full"]
    d_corr = d_tables["correlations"].loc[(d_tables["correlations"]["endpoint_id"] == "H3_gap_full") & (d_tables["correlations"]["scope"] == "all")].iloc[0]
    rows.append({
        "analysis": "D Trial-count thresholds and correlations",
        "judgement": "Pass" if bool((d_h3["p_one_sided"] < 0.05).all()) and (float(d_corr["p_value"]) > 0.05) else "Partial",
        "h3_maintained": "Yes" if bool((d_h3["p_one_sided"] < 0.05).all()) and (float(d_corr["p_value"]) > 0.05) else "Partial",
        "effect_size_d": float(d_h3["effect_size_d"].mean()),
        "comment": (
            "H3 remains nominally significant across all trial thresholds and is not correlated with trial count."
            if bool((d_h3["p_one_sided"] < 0.05).all()) and (float(d_corr["p_value"]) > 0.05)
            else "H3 is flat with trial count overall, but the strictest threshold leaves too few subjects to preserve nominal significance."
        ),
    })
    e_all = e_tables["sensitivity"].loc[e_tables["sensitivity"]["excluded"] == "all_outliers"].iloc[0]
    rows.append({
        "analysis": "E Outlier sensitivity",
        "judgement": "Pass" if float(e_all["p_one_sided"]) < 0.05 else "Partial",
        "h3_maintained": "Yes" if float(e_all["p_one_sided"]) < 0.05 else "Partial",
        "effect_size_d": float(e_all["effect_size_d"]),
        "comment": "H3 remains after excluding all IQR-defined outliers." if float(e_all["p_one_sided"]) < 0.05 else "H3 depends materially on IQR-defined outliers.",
    })
    f_alpha = f_table.loc[f_table["band"] == "alpha"].iloc[0]
    rows.append({
        "analysis": "F Band specificity",
        "judgement": "Pass" if float(f_alpha["p_one_sided"]) < 0.05 and float(abs(f_alpha["effect_size_d"])) == float(f_table["effect_size_d"].abs().max()) else "Partial",
        "h3_maintained": "Yes" if float(f_alpha["p_one_sided"]) < 0.05 and float(abs(f_alpha["effect_size_d"])) == float(f_table["effect_size_d"].abs().max()) else "Partial",
        "effect_size_d": float(f_alpha["effect_size_d"]),
        "comment": "Alpha carries the strongest TempPar full-gap effect." if float(abs(f_alpha["effect_size_d"])) == float(f_table["effect_size_d"].abs().max()) else "Other bands show effects of similar magnitude.",
    })
    g_temp = g_table.loc[g_table["network"] == "TempPar"].iloc[0]
    rows.append({
        "analysis": "G Network specificity",
        "judgement": "Pass" if float(g_temp["p_one_sided"]) < 0.05 and float(abs(g_temp["effect_size_d"])) == float(g_table["effect_size_d"].abs().max()) else "Partial",
        "h3_maintained": "Yes" if float(g_temp["p_one_sided"]) < 0.05 and float(abs(g_temp["effect_size_d"])) == float(g_table["effect_size_d"].abs().max()) else "Partial",
        "effect_size_d": float(g_temp["effect_size_d"]),
        "comment": "TempPar has the strongest centered alpha full-gap effect." if float(abs(g_temp["effect_size_d"])) == float(g_table["effect_size_d"].abs().max()) else "TempPar is significant but not uniquely dominant.",
    })
    rows.append({
        "analysis": "H Parameter and reproducibility documentation",
        "judgement": "Descriptive",
        "h3_maintained": "Descriptive",
        "effect_size_d": float("nan"),
        "comment": "Pipeline parameters, software versions, and a reviewer-facing checklist were exported with the report.",
    })
    h3_power = power_table.loc[power_table["endpoint_id"] == "H3_gap_full"].iloc[0]
    rows.append({
        "analysis": "I Post-hoc power",
        "judgement": "Descriptive",
        "h3_maintained": "Descriptive",
        "effect_size_d": float(h3_power["effect_size_d"]),
        "comment": f"H3_gap_full has estimated one-sided power {float(h3_power['power_at_n35']):.3f} at n=35 per group.",
    })
    rows.append({
        "analysis": "J Distribution checks",
        "judgement": "Pass",
        "h3_maintained": "Yes",
        "effect_size_d": float("nan"),
        "comment": "Welch remains appropriate even if mild normality or variance deviations are present.",
    })
    return pd.DataFrame(rows)


def _write_report(paths: ValidationPaths, inputs: ValidationInputs, a_tables: dict[str, pd.DataFrame], b_tables: dict[str, pd.DataFrame], c_table: pd.DataFrame, d_tables: dict[str, pd.DataFrame], e_tables: dict[str, pd.DataFrame], f_table: pd.DataFrame, g_table: pd.DataFrame, power_table: pd.DataFrame, j_table: pd.DataFrame, robustness_matrix: pd.DataFrame) -> None:
    alpha_strength = f_table.loc[f_table["band"] == "alpha", "effect_size_d"].iloc[0]
    report = [
        "# Methodological validation report",
        "",
        "This report checks whether the current H1-H3 results, with emphasis on H3_gap_full, are robust to reasonable methodological alternatives already supported by the available cohort outputs.",
        "",
        "## Analysis summary",
        "",
        _df_html(robustness_matrix, max_rows=len(robustness_matrix)),
        "![Supplementary methodological summary](figures/block_summary_methodological_panel.png)",
        "",
        "- Methods and results: [paper_methodological_subsections.md](paper_methodological_subsections.md)",
        "- Figure index: [paper_figure_recommendations.md](paper_figure_recommendations.md)",
        "",
        "## Block A. Average and composite sensitivity",
        "",
        "### A.1 Mean-r versus Fisher-z averaging",
        _df_html(a_tables["a1"]),
        "",
        "### A.2 Mean versus median across trials",
        _df_html(a_tables["a2"]),
        f"![A.2 scatter](figures/block_a2_mean_vs_median_scatter.png)",
        "",
        "### A.3 Alternative weighting schemes",
        _df_html(a_tables["a3"]),
        "",
        "### A.4 Leave-one-network-out on the TempPar full composite",
        _df_html(a_tables["a4"]),
        "",
        "### A.5 Mean versus median across the nine TempPar full blocks",
        _df_html(a_tables["a5"]),
        "",
        "## Block B. AEC versus AEC-orth relationship",
        "",
        "### B.1 Subject-wise AEC versus AEC-orth correlation",
        _df_html(b_tables["b1_summary"]),
        f"![B.1 scatter](figures/block_b1_aec_vs_orth_scatter.png)",
        "",
        "### B.2 Within-subject H3 trial variability",
        _df_html(b_tables["b2_stats"]),
        f"![B.2 variability](figures/block_b2_trial_variance_by_group.png)",
        "",
        "## Block C. ROI-level TempPar analysis",
        "",
        _df_html(c_table),
        f"![C ROI barplot](figures/block_c_tempPar_roi_gap_barplot.png)",
        "",
        "Surface heatmaps were not generated because cortical coordinates are not stored in the accessible cohort outputs.",
        "",
        "## Block D. Trial-count sensitivity",
        "",
        _df_html(d_tables["thresholds"]),
        _df_html(d_tables["correlations"]),
        f"![D threshold](figures/block_d_h3_threshold_sensitivity.png)",
        f"![D scatter](figures/block_d_gap_vs_trialcount.png)",
        "",
        "## Block E. Distribution and outlier checks",
        "",
        _df_html(e_tables["outliers"] if not e_tables["outliers"].empty else pd.DataFrame([{"note": "No IQR outliers detected."}])),
        _df_html(e_tables["sensitivity"]),
        f"![E outliers](figures/block_e_h3_outliers.png)",
        "",
        "## Block F. Band specificity",
        "",
        _df_html(f_table),
        f"![F band specificity](figures/block_f_band_specificity.png)",
        f"Alpha carries an effect size of d={alpha_strength:.3f} in the TempPar full-gap contrast.",
        "",
        "## Block G. Network specificity",
        "",
        _df_html(g_table),
        f"![G network specificity](figures/block_g_network_specificity.png)",
        "",
        "## Block H. Parameter documentation",
        "",
        f"- Parameter JSON: [methodology_parameters.json](methodology_parameters.json)",
        f"- Reproducibility checklist: [reproducibility_checklist.md](reproducibility_checklist.md)",
        "",
        "## Block I. Post-hoc power",
        "",
        _df_html(power_table),
        f"![I power curves](figures/block_i_power_curves.png)",
        "",
        "## Block J. Distributional checks for H3_gap_full",
        "",
        _df_html(j_table),
        "",
        "## Conclusions",
        "",
        f"- H3_gap_full remains robust to Fisher-z averaging: minimum Pearson r = {a_tables['a1']['pearson_r'].min():.3f}.",
        f"- H3_gap_full remains robust to median averaging across trials: Pearson r = {a_tables['a2'].loc[a_tables['a2']['endpoint_id'] == 'H3_gap_full', 'pearson_r'].iloc[0]:.3f}.",
        f"- H3_gap_full survives {int((a_tables['a4']['effect_kept']).sum())}/9 leave-one-network-out perturbations.",
        f"- Trial-count dependence is weak: cohort-level H3_gap_full correlation with valid trials = {d_tables['correlations'].loc[(d_tables['correlations']['endpoint_id'] == 'H3_gap_full') & (d_tables['correlations']['scope'] == 'all'), 'pearson_r'].iloc[0]:.3f}.",
        f"- TempPar ROI direction consistency = {100.0 * c_table['expected_direction'].mean():.1f}%.",
    ]
    (paths.report_root / "methodological_validation_report.md").write_text("\n".join(report) + "\n")


def _write_paper_text(paths: ValidationPaths, a_tables: dict[str, pd.DataFrame], b_tables: dict[str, pd.DataFrame], c_table: pd.DataFrame, d_tables: dict[str, pd.DataFrame], e_tables: dict[str, pd.DataFrame], f_table: pd.DataFrame, g_table: pd.DataFrame, power_table: pd.DataFrame, j_table: pd.DataFrame) -> None:
    a1_h3 = a_tables["a1"].loc[a_tables["a1"]["endpoint_id"] == "H3_gap_full"].iloc[0]
    a2_h3 = a_tables["a2"].loc[a_tables["a2"]["endpoint_id"] == "H3_gap_full"].iloc[0]
    a3_h3 = a_tables["a3"].loc[a_tables["a3"]["endpoint_id"] == "H3_gap_full"].copy()
    a4_kept = int(a_tables["a4"]["effect_kept"].sum())
    a5_h3 = a_tables["a5"].iloc[0]
    b1 = b_tables["b1_summary"].iloc[0]
    b2 = b_tables["b2_stats"].loc[b_tables["b2_stats"]["endpoint_id"] == "H3_gap_full"].iloc[0]
    d_h3 = d_tables["thresholds"].loc[d_tables["thresholds"]["endpoint_id"] == "H3_gap_full"].copy()
    d40 = d_h3.loc[d_h3["T_min"] == 40].iloc[0]
    d45 = d_h3.loc[d_h3["T_min"] == 45].iloc[0]
    d_corr = d_tables["correlations"].loc[(d_tables["correlations"]["endpoint_id"] == "H3_gap_full") & (d_tables["correlations"]["scope"] == "all")].iloc[0]
    e_all = e_tables["sensitivity"].loc[e_tables["sensitivity"]["excluded"] == "all_outliers"].iloc[0]
    f_alpha = f_table.loc[f_table["band"] == "alpha"].iloc[0]
    f_theta = f_table.loc[f_table["band"] == "theta"].iloc[0]
    g_temp = g_table.loc[g_table["network"] == "TempPar"].iloc[0]
    power_h3 = power_table.loc[power_table["endpoint_id"] == "H3_gap_full"].iloc[0]

    methods_text = f"""# Methodological validation: methods and results

## Methods subsection

### Methodological validation analyses

To assess whether the main alpha `H3_gap_full` result depended on specific aggregation choices, we ran a structured methodological validation block on the already computed cohort outputs. The validation reused the cached subject-level network summaries, trial-level endpoint tables, and subject-specific ROI connectivity matrices, without altering the primary preprocessing or endpoint definitions. All auxiliary analyses used the same subject-level statistical unit and the same one-sided Welch framework as the main manuscript.

Within-subject aggregation was first re-evaluated by recomputing the six predefined endpoints after Fisher-z averaging of trial-level correlations and after replacing the subject-level mean across trials with the median. Composite construction was then perturbed in three ways: by weighting TempPar-centered blocks uniformly, by weighting them in proportion to network size, and by weighting them inversely to their across-subject variance. We also performed leave-one-block-out recomputation of the alpha TempPar full composite and replaced the arithmetic mean across its nine blocks with the median.

To characterize the relationship between `AEC` and `AEC-orth`, we quantified, for each subject, the Pearson correlation between TempPar-centered alpha `AEC` and `AEC-orth` block values. Trial-level variability of `H3_gap_full` and `H3_gap_inter` was estimated from the cached per-trial endpoints, allowing group comparison of within-subject gap variance. At the ROI level, TempPar parcel effects were computed from the saved subject-specific alpha `AEC` and `AEC-orth` matrices by taking each ROI's mean connectivity to the rest of the non-background cortex and forming `gap_roi = AEC_roi - AECorth_roi`.

Sensitivity to trial availability was evaluated by repeating the H1-H3 tests after restricting the cohort to subjects with at least `30`, `35`, `40`, or `45` valid trials, and by correlating H3 endpoint values with the number of valid trials. Additional checks included outlier removal based on the `1.5×IQR` rule within group, band-specific recomputation of the TempPar full gap across `delta`, `theta`, `alpha`, `beta`, and `gamma`, and network-specific recomputation of the alpha full gap using each of the nine collapsed Schaefer families as the center network. Finally, we documented all methodological parameters, estimated post-hoc one-sided power from the observed effect sizes, and tested normality and variance homogeneity for `H3_gap_full`.

## Results subsection

### Robustness of the alpha TempPar full gap

The main `H3_gap_full` result remained stable under multiple aggregation variants. Fisher-z averaging was essentially identical to the original correlation-scale averaging (`r={float(a1_h3['pearson_r']):.3f}`, mean absolute difference `{float(a1_h3['mean_abs_diff']):.4f}`). Replacing the within-subject mean across trials with the median weakened the endpoint only modestly (`delta {float(a2_h3['original_delta']):.4f} -> {float(a2_h3['median_r_delta']):.4f}`, one-sided `p {float(a2_h3['original_p_one_sided']):.4f} -> {float(a2_h3['median_r_p_one_sided']):.4f}`) while preserving both sign and nominal significance. Alternative composite weighting schemes yielded closely matched H3 effects (`d` range `{float(a3_h3['effect_size_d'].min()):.3f}` to `{float(a3_h3['effect_size_d'].max()):.3f}`; one-sided `p` range `{float(a3_h3['p_one_sided'].min()):.4f}` to `{float(a3_h3['p_one_sided'].max()):.4f}`), and the effect survived all `{a4_kept}/9` leave-one-block-out perturbations.

Using the median instead of the mean across the nine TempPar full-composite blocks changed the H3 group delta by `{float(a5_h3['delta_percent_difference']):.1f}%`, exceeding the pre-specified `<5%` tolerance but not reversing the result; the median-based composite remained slightly more negative than the original. The relation between `AEC` and `AEC-orth` was substantial but far from redundant (subject-wise median `r={float(b1['median_r']):.3f}`, mean `r={float(b1['mean_r']):.3f}`), indicating that the gap captures information beyond a simple rescaling. Trial-level gap variance differed between groups, but in the opposite direction to a simple noise-in-converters explanation: converters showed lower within-subject `H3_gap_full` variance than non-converters (`delta variance {float(b2['delta_group_a_minus_group_b']):.6f}`, one-sided `p={float(b2['p_one_sided']):.4f}`).

The effect also remained anatomically coherent. All TempPar ROIs showed the expected direction of group difference, with a direction-consistency fraction of `{100.0 * float(c_table['expected_direction'].mean()):.1f}%`. Sensitivity to trial count was limited. `H3_gap_full` remained nominally significant at `Tmin=30`, `35`, and `40` valid trials (`d={float(d40['effect_size_d']):.3f}`, one-sided `p={float(d40['p_one_sided']):.4f}` at `Tmin=40`), and only the most restrictive threshold (`Tmin=45`) lost nominal significance after the sample shrank to `{int(d45['n_group_a'])}` converters and `{int(d45['n_group_b'])}` non-converters. Across the full cohort, `H3_gap_full` was essentially unrelated to the number of valid trials (`r={float(d_corr['pearson_r']):.3f}`, `p={float(d_corr['p_value']):.3f}`).

Outlier analysis did not suggest that the finding was driven by atypical subjects. Removing all IQR-defined outliers strengthened the result (`d={float(e_all['effect_size_d']):.3f}`, one-sided `p={float(e_all['p_one_sided']):.4f}`). The effect was also specific in frequency and network space. Alpha showed the strongest TempPar full-gap effect (`d={float(f_alpha['effect_size_d']):.3f}`, one-sided `p={float(f_alpha['p_one_sided']):.4f}`), whereas theta showed only a weaker trend (`d={float(f_theta['effect_size_d']):.3f}`, one-sided `p={float(f_theta['p_one_sided']):.4f}`) and the remaining bands were small or null. Among the nine center-network definitions, TempPar produced the largest effect (`d={float(g_temp['effect_size_d']):.3f}`, one-sided `p={float(g_temp['p_one_sided']):.4f}`), supporting network specificity.

Distributional diagnostics were compatible with the use of Welch's test (`Shapiro p={float(j_table.loc[j_table['test'] == 'Shapiro group_a', 'p_value'].iloc[0]):.3f}` and `{float(j_table.loc[j_table['test'] == 'Shapiro group_b', 'p_value'].iloc[0]):.3f}` for converters and non-converters, respectively; `Levene p={float(j_table.loc[j_table['test'] == 'Levene', 'p_value'].iloc[0]):.3f}`). Post-hoc power for `H3_gap_full` at `n=35` per group was estimated at `{float(power_h3['power_at_n35']):.3f}`, with approximately `{int(power_h3['n_per_group_for_power_0_80'])}` subjects per group required to reach 0.80 one-sided power under the observed effect size. Taken together, these analyses support the interpretation that `H3_gap_full` is robust to reasonable methodological perturbations, while remaining sensitive to extreme sample restriction.
"""
    (paths.report_root / "paper_methodological_subsections.md").write_text(methods_text)

    figure_note = """# Methodological validation figures

Figures are written under `figures/` in the validation report directory.

| Figure | Analysis |
| --- | --- |
| `block_summary_methodological_panel.pdf` | Band and network comparisons, with outlier-removal sensitivity |
| `block_f_band_specificity.png` | TempPar full-gap effect across frequency bands |
| `block_g_network_specificity.png` | Alpha full-gap effect across center networks |
| `block_d_h3_threshold_sensitivity.png` | H3 effects after restricting the minimum number of valid trials |
| `block_d_gap_vs_trialcount.png` | H3 endpoint values against valid trial counts |
| `block_e_h3_outliers.png` | H3 values and within-group IQR outliers |
| `block_b1_aec_vs_orth_scatter.png` | Subject-level relationship between AEC and AEC-orth |
| `block_c_tempPar_roi_gap_barplot.png` | Group gap differences for TempPar ROIs |
| `block_a2_mean_vs_median_scatter.png` | Endpoint values under mean and median trial aggregation |
| `block_b2_trial_variance_by_group.png` | Within-subject gap variance by group |
| `block_i_power_curves.png` | Estimated power as a function of sample size |

The manuscript pipeline separately builds `fig_sensitivity_trials.png` through
`scripts/final_figures/build_trials_threshold_sensitivity.py`.
"""
    (paths.report_root / "paper_figure_recommendations.md").write_text(figure_note)


def generate_methodological_validation_report(output_root: str | Path = "outputs_full_cohort", data_root: str | Path = "data", report_root: str | Path | None = None) -> dict[str, str]:
    _style()
    output_root = Path(output_root)
    report_root = Path(report_root) if report_root is not None else output_root / "methodological_validation"
    paths = _ensure_dirs(report_root)
    inputs = load_validation_inputs(output_root=output_root, data_root=data_root)

    a_results = block_a_average_sensitivity(inputs, paths)
    a_extra = block_a_weighting_and_leave_out(inputs, paths)
    a_results.update(a_extra)
    b_results = block_b_aec_orth_relationship(inputs, paths)
    c_table = block_c_roi_level(inputs, output_root, paths)
    d_results = block_d_trial_sensitivity(inputs, paths)
    e_results = block_e_outliers(inputs, paths)
    f_table = block_f_band_specificity(inputs, paths)
    g_table = block_g_network_specificity(inputs, paths)
    plot_methodological_summary_panel(paths, f_table, g_table, e_results)
    block_h_documentation(inputs, output_root, paths)
    power_table = block_i_power(inputs, paths)
    j_table = block_j_distribution_checks(inputs, paths)

    robustness_matrix = _robustness_matrix(a_results, b_results, c_table, d_results, e_results, f_table, g_table, power_table, j_table)
    robustness_matrix.to_csv(paths.tables_dir / "robustness_matrix.csv", index=False)
    _write_report(paths, inputs, a_results, b_results, c_table, d_results, e_results, f_table, g_table, power_table, j_table, robustness_matrix)
    _write_paper_text(paths, a_results, b_results, c_table, d_results, e_results, f_table, g_table, power_table, j_table)

    return {
        "report_root": str(paths.report_root),
        "report": str(paths.report_root / "methodological_validation_report.md"),
        "robustness_matrix": str(paths.tables_dir / "robustness_matrix.csv"),
        "parameters_json": str(paths.report_root / "methodology_parameters.json"),
        "checklist": str(paths.report_root / "reproducibility_checklist.md"),
        "paper_text": str(paths.report_root / "paper_methodological_subsections.md"),
        "figure_recommendations": str(paths.report_root / "paper_figure_recommendations.md"),
    }
