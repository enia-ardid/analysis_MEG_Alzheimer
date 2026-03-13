from __future__ import annotations

"""Final strong-testing layer for hypotheses H1-H3.

The cohort pipeline exports subject-level network summaries for every band and
metric. This module takes those summaries and constructs the exact endpoint
family used for the final confirmation step:

- H1: alpha TempPar full composite in AEC and AEC-orth
- H2: beta TempPar inter-network composite in AEC and AEC-orth
- H3: alpha TempPar gap in the full and inter-only definitions

The statistical goal here is stronger than the descriptive and exploratory
outputs produced upstream. We therefore evaluate all six endpoints as a closed
family and apply:

- Holm-Bonferroni adjustment on the endpoint p-values
- max-T permutation correction across the whole endpoint family
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

GROUP_COLORS = {
    "Converter": "#c96b43",
    "Non-converter": "#3b77a3",
}


@dataclass(frozen=True)
class StrongEndpointSpec:
    """Definition of one atomic endpoint in the final H1-H3 family."""
    endpoint_id: str
    hypothesis_id: str
    hypothesis_title: str
    analysis_id: str
    source: str
    metric: str
    expected_direction: str
    label: str


STRONG_ENDPOINTS: tuple[StrongEndpointSpec, ...] = (
    StrongEndpointSpec(
        endpoint_id="H1_AEC",
        hypothesis_id="H1",
        hypothesis_title="Alpha TempPar envelope coupling is reduced in converters and survives orthogonalization",
        analysis_id="alpha_tempPar_full_down",
        source="metric",
        metric="AEC",
        expected_direction="group_a_lt_group_b",
        label="H1 alpha TempPar full / AEC",
    ),
    StrongEndpointSpec(
        endpoint_id="H1_AECorth",
        hypothesis_id="H1",
        hypothesis_title="Alpha TempPar envelope coupling is reduced in converters and survives orthogonalization",
        analysis_id="alpha_tempPar_full_down",
        source="metric",
        metric="AEC-orth",
        expected_direction="group_a_lt_group_b",
        label="H1 alpha TempPar full / AEC-orth",
    ),
    StrongEndpointSpec(
        endpoint_id="H2_AEC",
        hypothesis_id="H2",
        hypothesis_title="Beta TempPar envelope coupling is reduced in converters in both AEC and AEC-orth",
        analysis_id="beta_tempPar_inter_down",
        source="metric",
        metric="AEC",
        expected_direction="group_a_lt_group_b",
        label="H2 beta TempPar inter / AEC",
    ),
    StrongEndpointSpec(
        endpoint_id="H2_AECorth",
        hypothesis_id="H2",
        hypothesis_title="Beta TempPar envelope coupling is reduced in converters in both AEC and AEC-orth",
        analysis_id="beta_tempPar_inter_down",
        source="metric",
        metric="AEC-orth",
        expected_direction="group_a_lt_group_b",
        label="H2 beta TempPar inter / AEC-orth",
    ),
    StrongEndpointSpec(
        endpoint_id="H3_gap_full",
        hypothesis_id="H3",
        hypothesis_title="The alpha AEC-to-AEC-orth gap is smaller in converters around TempPar composites",
        analysis_id="alpha_tempPar_full_down",
        source="gap",
        metric="AEC-AECorth-gap",
        expected_direction="group_a_lt_group_b",
        label="H3 alpha TempPar full / gap",
    ),
    StrongEndpointSpec(
        endpoint_id="H3_gap_inter",
        hypothesis_id="H3",
        hypothesis_title="The alpha AEC-to-AEC-orth gap is smaller in converters around TempPar composites",
        analysis_id="alpha_tempPar_inter_down",
        source="gap",
        metric="AEC-AECorth-gap",
        expected_direction="group_a_lt_group_b",
        label="H3 alpha TempPar inter / gap",
    ),
)


def _selector_temp_par_inter(frame: pd.DataFrame) -> pd.Series:
    """Select all inter-network rows in which TempPar participates."""
    return (frame["connection_type"] == "inter") & ((frame["network_a"] == "TempPar") | (frame["network_b"] == "TempPar"))


def _selector_temp_par_full(frame: pd.DataFrame) -> pd.Series:
    """Select the full TempPar-centered family: all inter edges plus TempPar intra."""
    inter_mask = _selector_temp_par_inter(frame)
    intra_mask = (frame["connection_type"] == "intra") & (frame["network_a"] == "TempPar")
    return inter_mask | intra_mask


def _metric_composite_values(network_df: pd.DataFrame, analysis_id: str, band: str, metric: str) -> pd.DataFrame:
    """Build one subject-level composite from the network summary table.

    The output has one row per subject and therefore matches the statistical
    unit used in the final group comparison.
    """
    frame = network_df.loc[(network_df["band"] == band) & (network_df["metric"] == metric)].copy()
    if analysis_id == "alpha_tempPar_full_down":
        frame = frame.loc[_selector_temp_par_full(frame)]
    elif analysis_id == "alpha_tempPar_inter_down":
        frame = frame.loc[_selector_temp_par_inter(frame)]
    elif analysis_id == "beta_tempPar_inter_down":
        frame = frame.loc[_selector_temp_par_inter(frame)]
    else:
        raise ValueError(f"Unsupported analysis_id: {analysis_id}")
    values = (
        frame.groupby(["subject_id", "group"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "value"})
        .sort_values(["group", "subject_id"])
        .reset_index(drop=True)
    )
    return values


def _gap_composite_values(network_df: pd.DataFrame, analysis_id: str, band: str) -> pd.DataFrame:
    """Compute the subject-level gap ``AEC - AEC-orth`` for one composite."""
    aec = _metric_composite_values(network_df, analysis_id=analysis_id, band=band, metric="AEC")
    orth = _metric_composite_values(network_df, analysis_id=analysis_id, band=band, metric="AEC-orth")
    merged = aec.merge(orth, on=["subject_id", "group"], suffixes=("_aec", "_orth"), how="inner")
    merged["value"] = merged["value_aec"] - merged["value_orth"]
    return merged[["subject_id", "group", "value"]].sort_values(["group", "subject_id"]).reset_index(drop=True)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#1d1d1d",
            "axes.labelcolor": "#1d1d1d",
            "xtick.color": "#1d1d1d",
            "ytick.color": "#1d1d1d",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _extract_endpoint_frame(
    spec: StrongEndpointSpec,
    network_df: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve one endpoint specification into a concrete subject-value table."""
    if spec.source == "metric":
        band = "alpha" if spec.analysis_id.startswith("alpha") else "beta"
        frame = _metric_composite_values(network_df, analysis_id=spec.analysis_id, band=band, metric=spec.metric)
    elif spec.source == "gap":
        frame = _gap_composite_values(network_df, analysis_id=spec.analysis_id, band="alpha")
    else:
        raise ValueError(f"Unsupported endpoint source: {spec.source}")
    frame["endpoint_id"] = spec.endpoint_id
    frame["hypothesis_id"] = spec.hypothesis_id
    frame["metric"] = spec.metric
    return frame.sort_values(["group", "subject_id"]).reset_index(drop=True)


def _cohen_d_unpaired(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Unpaired Cohen's d for effect-size reporting."""
    mean_diff = float(np.mean(values_a) - np.mean(values_b))
    var_a = float(np.var(values_a, ddof=1))
    var_b = float(np.var(values_b, ddof=1))
    pooled = np.sqrt(((len(values_a) - 1) * var_a + (len(values_b) - 1) * var_b) / max(1, len(values_a) + len(values_b) - 2))
    if pooled < 1e-12:
        return float("nan")
    return mean_diff / pooled


def _one_sided_pvalue(t_stat: float, two_sided_p: float, expected_direction: str) -> float:
    """Convert a two-sided Welch p-value into the directional p-value used here."""
    if expected_direction == "group_a_lt_group_b":
        return float(two_sided_p / 2.0) if t_stat < 0.0 else float(1.0 - two_sided_p / 2.0)
    if expected_direction == "group_a_gt_group_b":
        return float(two_sided_p / 2.0) if t_stat > 0.0 else float(1.0 - two_sided_p / 2.0)
    raise ValueError(f"Unsupported direction: {expected_direction}")


def _bootstrap_ci_mean_diff(values_a: np.ndarray, values_b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Bootstrap confidence interval for the difference in group means."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample_a = rng.choice(values_a, size=len(values_a), replace=True)
        sample_b = rng.choice(values_b, size=len(values_b), replace=True)
        diffs[idx] = float(np.mean(sample_a) - np.mean(sample_b))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values for one endpoint family."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for idx, value in enumerate(sorted_p):
        adjusted = (m - idx) * value
        running = max(running, adjusted)
        adjusted_sorted[idx] = min(1.0, running)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _welch_t_columns(matrix: np.ndarray, group_a_mask: np.ndarray) -> np.ndarray:
    """Vectorized Welch t-statistics for all endpoint columns at once."""
    group_b_mask = ~group_a_mask
    xa = matrix[group_a_mask]
    xb = matrix[group_b_mask]
    mean_a = xa.mean(axis=0)
    mean_b = xb.mean(axis=0)
    var_a = xa.var(axis=0, ddof=1)
    var_b = xb.var(axis=0, ddof=1)
    denom = np.sqrt(var_a / xa.shape[0] + var_b / xb.shape[0])
    denom = np.where(denom < 1e-12, np.nan, denom)
    t_stat = (mean_a - mean_b) / denom
    return np.nan_to_num(t_stat, nan=0.0, posinf=0.0, neginf=0.0)


def _oriented_t(t_stat: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Flip t-statistics so larger values always mean stronger support."""
    return np.where(directions > 0, t_stat, -t_stat)


def _max_t_correction(
    endpoint_matrix: np.ndarray,
    group_a_mask: np.ndarray,
    directions: np.ndarray,
    n_perm: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Family-wise max-T permutation correction across all endpoints.

    Each permutation shuffles group labels once, recomputes all endpoint
    statistics, and stores only the largest oriented t-statistic. An observed
    endpoint must therefore be extreme not only on its own, but relative to the
    strongest endpoint that could appear by chance anywhere in the family.
    """
    rng = np.random.default_rng(seed)
    observed = _oriented_t(_welch_t_columns(endpoint_matrix, group_a_mask), directions)
    null_max = np.empty(n_perm, dtype=float)
    labels = group_a_mask.astype(int)
    for idx in range(n_perm):
        perm_mask = rng.permutation(labels).astype(bool)
        perm_t = _oriented_t(_welch_t_columns(endpoint_matrix, perm_mask), directions)
        null_max[idx] = float(np.nanmax(perm_t))
    corrected = np.array([(1.0 + np.sum(null_max >= value)) / (n_perm + 1.0) for value in observed], dtype=float)
    return observed, corrected


def _boxplot(ax: plt.Axes, values_by_group: dict[str, np.ndarray], title: str, ylabel: str, annotation: str) -> None:
    groups = ["Converter", "Non-converter"]
    values = [values_by_group[group] for group in groups]
    box = ax.boxplot(values, positions=[1, 2], widths=0.55, patch_artist=True, showfliers=False)
    for patch, group in zip(box["boxes"], groups):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_alpha(0.30)
        patch.set_edgecolor("#1d1d1d")
    for artist_name in ("whiskers", "caps", "medians"):
        for artist in box[artist_name]:
            artist.set_color("#1d1d1d")
    rng = np.random.default_rng(0)
    for pos, group, vals in zip([1, 2], groups, values):
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=16, color=GROUP_COLORS[group], alpha=0.75, linewidths=0)
    ax.set_xticks([1, 2], groups)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.16, linewidth=0.8)
    ax.text(
        0.98,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#b8b8b8", "alpha": 0.95},
    )


def _plot_endpoint_figure(endpoint_values: dict[str, pd.DataFrame], endpoint_tests: pd.DataFrame, report_root: Path) -> None:
    """Create a compact figure with all endpoint-level distributions."""
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 12))
    fig.suptitle("H1-H3 strong endpoint tests", fontsize=16, y=1.01)
    for ax, spec in zip(axes.flat, STRONG_ENDPOINTS):
        frame = endpoint_values[spec.endpoint_id]
        row = endpoint_tests.loc[endpoint_tests["endpoint_id"] == spec.endpoint_id].iloc[0]
        annotation = (
            f"delta={row['delta_group_a_minus_group_b']:.4f}\n"
            f"p1={row['p_one_sided']:.4g}\n"
            f"Holm={row['holm_p_one_sided']:.4g}\n"
            f"maxT={row['max_t_p_one_sided']:.4g}\n"
            f"d={row['effect_size_d']:.3f}"
        )
        _boxplot(
            ax,
            {
                "Converter": frame.loc[frame["group"] == "Converter", "value"].to_numpy(dtype=float),
                "Non-converter": frame.loc[frame["group"] == "Non-converter", "value"].to_numpy(dtype=float),
            },
            title=spec.label,
            ylabel="Composite value",
            annotation=annotation,
        )
    fig.tight_layout()
    fig.savefig(report_root / "strong_endpoints.png")
    plt.close(fig)


def _write_markdown(endpoint_tests: pd.DataFrame, hypothesis_summary: pd.DataFrame, report_root: Path, n_perm: int) -> None:
    """Write a human-readable summary next to the CSV outputs."""
    lines = [
        "# Strong H1-H3 testing",
        "",
        "This report uses a fixed H1-H3 family of six directional endpoints:",
        "- H1: alpha TempPar full in AEC and AEC-orth",
        "- H2: beta TempPar inter in AEC and AEC-orth",
        "- H3: alpha TempPar full gap and alpha TempPar inter gap",
        "",
        f"Family-wise error control was evaluated with Holm-Bonferroni and max-T permutation correction over all six endpoints ({n_perm} permutations).",
        "",
        "## Endpoint tests",
        "",
    ]
    for row in endpoint_tests.itertuples(index=False):
        lines.append(
            f"- {row.endpoint_id}: delta={row.delta_group_a_minus_group_b:.4f}, "
            f"p1={row.p_one_sided:.4g}, Holm={row.holm_p_one_sided:.4g}, "
            f"maxT={row.max_t_p_one_sided:.4g}, d={row.effect_size_d:.3f}"
        )
    lines.extend(["", "## Hypothesis summary", ""])
    for row in hypothesis_summary.itertuples(index=False):
        lines.append(
            f"- {row.hypothesis_id}: strong_supported={bool(row.strong_supported)}, "
            f"all_holm={bool(row.all_holm_significant)}, all_maxT={bool(row.all_max_t_significant)}, "
            f"conjunction_holm={row.conjunction_holm_p_one_sided:.4g}, conjunction_maxT={row.conjunction_max_t_p_one_sided:.4g}"
        )
    (report_root / "strong_summary.md").write_text("\n".join(lines) + "\n")


def generate_strong_hypothesis_report(
    output_root: str | Path,
    report_root: str | Path | None = None,
    group_a: str = "Converter",
    group_b: str = "Non-converter",
    n_perm: int = 50000,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Run the final H1-H3 confirmation step from existing cohort outputs.

    The only required input is ``subject_network_means.csv`` produced by the
    cohort pipeline. From that table we rebuild the exact composites used in the
    hypothesis definitions, evaluate all six endpoints, and then summarize those
    endpoints back into one verdict per hypothesis.
    """
    _style()
    output_root = Path(output_root)
    report_root = Path(report_root) if report_root is not None else output_root / "strong_hypotheses"
    report_root.mkdir(parents=True, exist_ok=True)
    network_df = pd.read_csv(output_root / "subject_network_means.csv")

    endpoint_values: dict[str, pd.DataFrame] = {}
    reference_subjects: pd.DataFrame | None = None
    endpoint_matrix_cols: list[np.ndarray] = []
    endpoint_rows: list[dict] = []

    for idx, spec in enumerate(STRONG_ENDPOINTS):
        frame = _extract_endpoint_frame(spec, network_df=network_df)
        endpoint_values[spec.endpoint_id] = frame
        subject_ref = frame[["subject_id", "group"]].copy()
        if reference_subjects is None:
            reference_subjects = subject_ref
        elif not reference_subjects.equals(subject_ref):
            raise ValueError("Endpoint subject order differs across strong hypothesis tests.")

        values_a = frame.loc[frame["group"] == group_a, "value"].to_numpy(dtype=float)
        values_b = frame.loc[frame["group"] == group_b, "value"].to_numpy(dtype=float)
        # Each endpoint is first tested on its own so we can report the raw
        # effect size, t-statistic, and confidence interval before applying the
        # family-wise corrections.
        t_result = stats.ttest_ind(values_a, values_b, equal_var=False, nan_policy="omit")
        ci_low, ci_high = _bootstrap_ci_mean_diff(values_a, values_b, n_boot=n_boot, seed=seed + idx)
        endpoint_matrix_cols.append(frame["value"].to_numpy(dtype=float))
        endpoint_rows.append(
            {
                "endpoint_id": spec.endpoint_id,
                "hypothesis_id": spec.hypothesis_id,
                "hypothesis_title": spec.hypothesis_title,
                "analysis_id": spec.analysis_id,
                "metric": spec.metric,
                "expected_direction": spec.expected_direction,
                "n_group_a": int(len(values_a)),
                "n_group_b": int(len(values_b)),
                "mean_group_a": float(np.mean(values_a)),
                "mean_group_b": float(np.mean(values_b)),
                "delta_group_a_minus_group_b": float(np.mean(values_a) - np.mean(values_b)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "t_stat": float(t_result.statistic),
                "p_two_sided": float(t_result.pvalue),
                "p_one_sided": _one_sided_pvalue(float(t_result.statistic), float(t_result.pvalue), spec.expected_direction),
                "effect_size_d": _cohen_d_unpaired(values_a, values_b),
            }
        )

    endpoint_tests = pd.DataFrame(endpoint_rows)
    endpoint_tests["holm_p_one_sided"] = _holm_adjust(endpoint_tests["p_one_sided"].to_numpy(dtype=float))
    endpoint_tests["holm_significant"] = endpoint_tests["holm_p_one_sided"] <= 0.05
    endpoint_tests["holm_p_two_sided"] = _holm_adjust(endpoint_tests["p_two_sided"].to_numpy(dtype=float))
    endpoint_tests["holm_two_sided_significant"] = endpoint_tests["holm_p_two_sided"] <= 0.05

    endpoint_matrix = np.column_stack(endpoint_matrix_cols)
    if reference_subjects is None:
        raise ValueError("No strong-hypothesis endpoints were assembled.")
    group_a_mask = reference_subjects["group"].to_numpy() == group_a
    directions = np.array([1.0 if spec.expected_direction == "group_a_gt_group_b" else -1.0 for spec in STRONG_ENDPOINTS], dtype=float)
    oriented_t, max_t_p = _max_t_correction(
        endpoint_matrix=endpoint_matrix,
        group_a_mask=group_a_mask,
        directions=directions,
        n_perm=n_perm,
        seed=seed + 10_000,
    )
    endpoint_tests["oriented_t_stat"] = oriented_t
    endpoint_tests["max_t_p_one_sided"] = max_t_p
    endpoint_tests["max_t_significant"] = endpoint_tests["max_t_p_one_sided"] <= 0.05
    endpoint_tests = endpoint_tests.sort_values(["hypothesis_id", "endpoint_id"]).reset_index(drop=True)
    endpoint_tests.to_csv(report_root / "endpoint_tests.csv", index=False)

    summary_rows: list[dict] = []
    for hypothesis_id, chunk in endpoint_tests.groupby("hypothesis_id", sort=True):
        # A hypothesis is considered strongly supported only if every endpoint
        # that defines it survives both correction schemes.
        summary_rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "hypothesis_title": chunk["hypothesis_title"].iloc[0],
                "n_endpoints": int(len(chunk)),
                "all_holm_significant": bool(chunk["holm_significant"].all()),
                "all_max_t_significant": bool(chunk["max_t_significant"].all()),
                "all_holm_two_sided_significant": bool(chunk["holm_two_sided_significant"].all()),
                "conjunction_holm_p_one_sided": float(chunk["holm_p_one_sided"].max()),
                "conjunction_max_t_p_one_sided": float(chunk["max_t_p_one_sided"].max()),
                "conjunction_holm_p_two_sided": float(chunk["holm_p_two_sided"].max()),
                "strong_supported": bool(chunk["holm_significant"].all() and chunk["max_t_significant"].all()),
            }
        )
    hypothesis_summary = pd.DataFrame(summary_rows).sort_values("hypothesis_id").reset_index(drop=True)
    hypothesis_summary.to_csv(report_root / "hypothesis_summary.csv", index=False)

    _plot_endpoint_figure(endpoint_values=endpoint_values, endpoint_tests=endpoint_tests, report_root=report_root)
    _write_markdown(endpoint_tests=endpoint_tests, hypothesis_summary=hypothesis_summary, report_root=report_root, n_perm=n_perm)

    return {
        "report_root": str(report_root),
        "endpoint_tests": str(report_root / "endpoint_tests.csv"),
        "hypothesis_summary": str(report_root / "hypothesis_summary.csv"),
        "summary": str(report_root / "strong_summary.md"),
    }
