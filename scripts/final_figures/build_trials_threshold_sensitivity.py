#!/usr/bin/env python3
from __future__ import annotations

"""Trial-threshold sensitivity analysis for the six predefined endpoints.

The core function `run_threshold_sensitivity(df, ...)` is self-contained and
only requires a wide subject-level DataFrame with these columns:

    subject_id, group, n_valid_trials,
    H1_AEC, H1_AECorth, H2_AEC, H2_AECorth, H3_gap_full, H3_gap_inter

For convenience inside this repository, the CLI can also reconstruct that wide
DataFrame from the cached robustness outputs if no CSV is provided.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from meg_alzheimer.robustness import build_subject_endpoint_table, precompute_trial_endpoint_values, subject_trial_counts


ENDPOINTS = ["H1_AEC", "H1_AECorth", "H2_AEC", "H2_AECorth", "H3_gap_full", "H3_gap_inter"]
THRESHOLDS = [30, 33, 35, 38, 40]
GROUP_A = "Converter"
GROUP_B = "Non-converter"

EFFECT_COLORS = {
    "H1_AEC": "#2f5d73",
    "H1_AECorth": "#3f7890",
    "H2_AEC": "#587447",
    "H2_AECorth": "#718f5d",
    "H3_gap_full": "#8f6238",
    "H3_gap_inter": "#b07a49",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the valid-trial threshold sensitivity figure.")
    parser.add_argument(
        "--df-csv",
        default=None,
        help="Optional CSV containing the wide subject-level dataframe. If omitted, rebuild from the cached robustness outputs.",
    )
    parser.add_argument("--data-root", default="data", help="Raw data root used only if --df-csv is omitted and no cache exists.")
    parser.add_argument("--output-root", default="outputs_full_cohort", help="Existing cohort output root.")
    parser.add_argument(
        "--figure-path",
        default="figures/final/fig_sensitivity_trials.png",
        help="Destination PNG path for the publication figure.",
    )
    parser.add_argument(
        "--captions-path",
        default="captions_figures.md",
        help="Markdown file where the suggested figure caption will be written.",
    )
    return parser.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#111111",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "grid.color": "#d8d8d8",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _wide_df_from_repo(output_root: Path, data_root: Path) -> pd.DataFrame:
    trial_cache = output_root / "robustness" / "trial_endpoint_values.csv.gz"
    if not trial_cache.exists():
        trial_df = precompute_trial_endpoint_values(
            data_root=data_root,
            output_path=trial_cache,
            force=False,
            verbose=True,
        )
    else:
        trial_df = pd.read_csv(trial_cache)

    subject_wide = (
        build_subject_endpoint_table(trial_df, aggregation="mean_r")
        .pivot(index=["subject_id", "group"], columns="endpoint_id", values="value")
        .reset_index()
    )
    counts = subject_trial_counts(trial_df)
    df = subject_wide.merge(counts, on=["subject_id", "group"], how="left")
    ordered_cols = ["subject_id", "group", "n_valid_trials"] + ENDPOINTS
    return df[ordered_cols].copy()


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
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


def _welch_less(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False, alternative="less")
    return float(t_stat), float(p_value)


def _cohen_d(values_a: np.ndarray, values_b: np.ndarray) -> float:
    mean_diff = float(np.mean(values_a) - np.mean(values_b))
    var_a = float(np.var(values_a, ddof=1))
    var_b = float(np.var(values_b, ddof=1))
    pooled = np.sqrt(((len(values_a) - 1) * var_a + (len(values_b) - 1) * var_b) / max(1, len(values_a) + len(values_b) - 2))
    if pooled < 1e-12:
        return float("nan")
    return mean_diff / pooled


def run_threshold_sensitivity(df: pd.DataFrame, thresholds: list[int] | None = None) -> pd.DataFrame:
    """Run the requested threshold sensitivity analysis on a wide subject table."""
    thresholds = THRESHOLDS if thresholds is None else list(thresholds)
    required_cols = {"subject_id", "group", "n_valid_trials", *ENDPOINTS}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input df is missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for t_min in thresholds:
        subset = df.loc[df["n_valid_trials"] >= t_min].copy()
        pvals: list[float] = []
        threshold_rows: list[dict[str, object]] = []
        for endpoint in ENDPOINTS:
            values_c = subset.loc[subset["group"] == GROUP_A, endpoint].to_numpy(dtype=float)
            values_nc = subset.loc[subset["group"] == GROUP_B, endpoint].to_numpy(dtype=float)
            if len(values_c) < 2 or len(values_nc) < 2:
                t_stat = np.nan
                p_raw = np.nan
                d_val = np.nan
            else:
                t_stat, p_raw = _welch_less(values_c, values_nc)
                d_val = _cohen_d(values_c, values_nc)
            row = {
                "T_min": int(t_min),
                "endpoint": endpoint,
                "n_C": int(len(values_c)),
                "n_NC": int(len(values_nc)),
                "t_stat": t_stat,
                "p_raw": p_raw,
                "d": d_val,
            }
            threshold_rows.append(row)
            pvals.append(p_raw)

        holm = _holm_adjust(np.asarray(pvals, dtype=float))
        for row, holm_p in zip(threshold_rows, holm):
            row["holm_p"] = float(holm_p)
            row["holm_pass"] = bool(np.isfinite(holm_p) and holm_p <= 0.05)
            rows.append(row)
    return pd.DataFrame(rows)


def _plot(results_df: pd.DataFrame, figure_path: Path) -> None:
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), sharex=True)
    axes = axes.ravel()

    for idx, (ax, endpoint) in enumerate(zip(axes, ENDPOINTS)):
        subset = results_df.loc[results_df["endpoint"] == endpoint].copy()
        subset = subset.sort_values("T_min")
        x = subset["T_min"].to_numpy(dtype=float)
        d_vals = subset["d"].to_numpy(dtype=float)
        p_vals = subset["p_raw"].to_numpy(dtype=float)
        holm_pass = subset["holm_pass"].to_numpy(dtype=bool)

        color = EFFECT_COLORS[endpoint]
        ax.plot(x, d_vals, color=color, marker="o", linewidth=2.0, markersize=5.5, zorder=3)
        ax.set_title(endpoint)
        if idx % 3 == 0:
            ax.set_ylabel("Cohen's d")
        ax.grid(axis="y", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(THRESHOLDS)
        ax.set_xlim(min(THRESHOLDS) - 0.8, max(THRESHOLDS) + 0.8)

        y_min, y_max = np.nanmin(d_vals), np.nanmax(d_vals)
        pad = max(0.08, 0.18 * (y_max - y_min if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0))
        ax.set_ylim(y_min - pad, y_max + pad)

        for xi, yi, n_c, n_nc, passed in zip(x, d_vals, subset["n_C"], subset["n_NC"], holm_pass):
            ax.text(xi, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{n_c}/{n_nc}", ha="center", va="bottom", fontsize=8, color="#333333")
            if passed:
                ax.scatter([xi], [yi], marker="*", s=120, facecolor="#111111", edgecolor="white", linewidth=0.6, zorder=5)

        ax_p = ax.twinx()
        ax_p.plot(x, p_vals, color="#555555", linestyle="--", marker="s", linewidth=1.6, markersize=4.5, zorder=2)
        ax_p.axhline(0.05, color="#b22222", linestyle="--", linewidth=1.0)
        if idx % 3 == 2:
            ax_p.set_ylabel("Raw p-value")
        ax_p.set_ylim(0.0, max(0.08, float(np.nanmax(p_vals)) * 1.15))
        ax_p.tick_params(axis="y", labelsize=8)
        if idx % 3 != 2:
            ax_p.set_ylabel("")

    for ax in axes[3:]:
        ax.set_xlabel(r"$T_{\min}$ valid trials")
    fig.suptitle("Sensitivity to minimum valid trials threshold", fontsize=14, y=1.02)
    fig.text(0.5, -0.01, "Solid line: Cohen's d. Dashed line: one-sided Welch p-value (Converter < Non-converter). Filled stars indicate Holm-Bonferroni survival within each threshold.", ha="center", va="top", fontsize=9)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    if figure_path.suffix.lower() == ".png":
        fig.savefig(figure_path.with_suffix(".pdf"))
    plt.close(fig)


def _upsert_caption(path: Path) -> None:
    heading = "## Figure: Sensitivity to minimum valid trials threshold"
    block = (
        f"{heading}\n\n"
        "Suggested caption: Trial-threshold sensitivity analysis for the six predefined endpoints. "
        "At each minimum valid-trial threshold, only subjects meeting the inclusion criterion were retained, "
        "and the Converter versus Non-converter comparison was recomputed with a one-sided Welch test "
        "(`Converter < Non-converter`). Solid lines show Cohen's d, dashed lines show the raw one-sided p-value, "
        "and filled stars mark endpoints that survived Holm-Bonferroni correction within that threshold-specific "
        "six-endpoint family. The labels under each marker report the surviving sample size in the form `n_C/n_NC`.\n"
    )
    text = path.read_text() if path.exists() else ""
    if heading in text:
        start = text.index(heading)
        next_idx = text.find("\n## ", start + 1)
        end = len(text) if next_idx == -1 else next_idx + 1
        text = text[:start] + block + text[end:]
    else:
        text = text.rstrip() + ("\n\n" if text.strip() else "") + block
    path.write_text(text)


def main() -> None:
    args = parse_args()
    if args.df_csv:
        df = pd.read_csv(args.df_csv)
    else:
        df = _wide_df_from_repo(output_root=Path(args.output_root), data_root=Path(args.data_root))

    results = run_threshold_sensitivity(df, thresholds=THRESHOLDS)
    _plot(results, Path(args.figure_path))
    _upsert_caption(Path(args.captions_path))

    summary = results[["T_min", "endpoint", "n_C", "n_NC", "d", "p_raw", "holm_pass"]].copy()
    summary["d"] = summary["d"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    summary["p_raw"] = summary["p_raw"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "nan")
    print(summary.to_string(index=False))
    print(f"\nFigure written to: {args.figure_path}")
    print(f"PDF written to: {Path(args.figure_path).with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
