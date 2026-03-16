#!/usr/bin/env python3
from __future__ import annotations

"""Build a supplementary robustness figure for H1, H2, and H3.

Robustness strategy used here
-----------------------------
The main analysis defines each endpoint as a simple arithmetic mean over a fixed
set of network-summary rows. The robustness check keeps the endpoint
definitions, selected network pairs, and subject-level unit exactly unchanged,
but replaces the within-subject arithmetic mean with the within-subject median.

This is a narrow and defensible robustness check:

- it tests whether the direction of the effect depends on one aggregation choice
- it does not introduce a different dataset, a different endpoint definition, or
  a different inferential family
- it can be implemented directly from `subject_network_means.csv`, which is part
  of the reproducible current workflow
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

from meg_alzheimer.strong_hypotheses import STRONG_ENDPOINTS, _selector_temp_par_full, _selector_temp_par_inter


GROUP_A = "Converter"
GROUP_B = "Non-converter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the supplementary robustness figure for H1-H3.")
    parser.add_argument("--output-root", default="outputs_full_cohort", help="Existing cohort output folder.")
    parser.add_argument("--figure-dir", default="figures/final", help="Destination folder for figure files.")
    parser.add_argument(
        "--captions-path",
        default="captions_figures.md",
        help="Markdown file where the suggested figure caption will be written.",
    )
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap count for the robustness confidence intervals.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
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
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _aggregate_subject_values(frame: pd.DataFrame, agg: str) -> pd.DataFrame:
    return (
        frame.groupby(["subject_id", "group"], as_index=False)["value"]
        .agg(agg)
        .sort_values(["group", "subject_id"])
        .reset_index(drop=True)
    )


def _metric_subject_values(network_df: pd.DataFrame, analysis_id: str, band: str, metric: str, agg: str) -> pd.DataFrame:
    frame = network_df.loc[(network_df["band"] == band) & (network_df["metric"] == metric)].copy()
    if analysis_id == "alpha_tempPar_full_down":
        frame = frame.loc[_selector_temp_par_full(frame)]
    elif analysis_id in ("alpha_tempPar_inter_down", "beta_tempPar_inter_down"):
        frame = frame.loc[_selector_temp_par_inter(frame)]
    else:
        raise ValueError(f"Unsupported analysis_id: {analysis_id}")
    return _aggregate_subject_values(frame, agg=agg)


def _gap_subject_values(network_df: pd.DataFrame, analysis_id: str, agg: str) -> pd.DataFrame:
    aec = _metric_subject_values(network_df, analysis_id=analysis_id, band="alpha", metric="AEC", agg=agg)
    orth = _metric_subject_values(network_df, analysis_id=analysis_id, band="alpha", metric="AEC-orth", agg=agg)
    merged = aec.merge(orth, on=["subject_id", "group"], suffixes=("_aec", "_orth"), how="inner")
    merged["value"] = merged["value_aec"] - merged["value_orth"]
    return merged[["subject_id", "group", "value"]].sort_values(["group", "subject_id"]).reset_index(drop=True)


def _endpoint_values(network_df: pd.DataFrame, analysis_id: str, metric: str, agg: str) -> pd.DataFrame:
    if metric == "AEC-AECorth-gap":
        return _gap_subject_values(network_df, analysis_id=analysis_id, agg=agg)
    band = "alpha" if analysis_id.startswith("alpha") else "beta"
    return _metric_subject_values(network_df, analysis_id=analysis_id, band=band, metric=metric, agg=agg)


def _bootstrap_ci_delta(values_a: np.ndarray, values_b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample_a = rng.choice(values_a, size=len(values_a), replace=True)
        sample_b = rng.choice(values_b, size=len(values_b), replace=True)
        diffs[idx] = float(np.mean(sample_a) - np.mean(sample_b))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def _collect_robustness_rows(
    network_df: pd.DataFrame,
    endpoint_report: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    label_map = {spec.endpoint_id: spec.label for spec in STRONG_ENDPOINTS}
    family_map = {spec.endpoint_id: spec.hypothesis_id for spec in STRONG_ENDPOINTS}

    rows: list[dict] = []
    for idx, spec in enumerate(STRONG_ENDPOINTS):
        primary = endpoint_report.loc[endpoint_report["endpoint_id"] == spec.endpoint_id].iloc[0]
        rows.append(
            {
                "endpoint_id": spec.endpoint_id,
                "hypothesis_id": spec.hypothesis_id,
                "label": spec.label,
                "variant": "Primary mean",
                "delta": float(primary["delta_group_a_minus_group_b"]),
                "ci95_low": float(primary["ci95_low"]),
                "ci95_high": float(primary["ci95_high"]),
                "effect_size_d": float(primary["effect_size_d"]),
            }
        )

        robust = _endpoint_values(network_df, analysis_id=spec.analysis_id, metric=spec.metric, agg="median")
        values_a = robust.loc[robust["group"] == GROUP_A, "value"].to_numpy(dtype=float)
        values_b = robust.loc[robust["group"] == GROUP_B, "value"].to_numpy(dtype=float)
        delta = float(np.mean(values_a) - np.mean(values_b))
        ci_low, ci_high = _bootstrap_ci_delta(values_a, values_b, n_boot=n_boot, seed=seed + idx)
        pooled = np.sqrt(
            (
                (len(values_a) - 1) * np.var(values_a, ddof=1)
                + (len(values_b) - 1) * np.var(values_b, ddof=1)
            )
            / max(1, len(values_a) + len(values_b) - 2)
        )
        d_val = float(delta / pooled) if pooled > 1e-12 else float("nan")
        rows.append(
            {
                "endpoint_id": spec.endpoint_id,
                "hypothesis_id": spec.hypothesis_id,
                "label": spec.label,
                "variant": "Median robustness",
                "delta": delta,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "effect_size_d": d_val,
            }
        )
    frame = pd.DataFrame(rows)
    order = {spec.endpoint_id: idx for idx, spec in enumerate(STRONG_ENDPOINTS)}
    frame["plot_order"] = frame["endpoint_id"].map(order)
    frame = frame.sort_values(["plot_order", "variant"]).reset_index(drop=True)
    return frame


def _upsert_caption(path: Path) -> None:
    heading = "## Figure: Robustness of H1-H3 endpoints"
    block = (
        f"{heading}\n\n"
        "Suggested caption: Supplementary robustness check for the six endpoints defining H1-H3. "
        "The primary analysis uses the arithmetic mean across the network-summary rows included in each endpoint; "
        "the robustness variant keeps the same rows and the same subject-level unit but replaces that within-subject "
        "mean with a within-subject median. Points show the mean difference (Converter minus Non-converter) and "
        "horizontal bars show 95% confidence intervals. The purpose of the figure is to verify that the direction "
        "and approximate magnitude of the endpoint effects do not depend on this single aggregation choice.\n"
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
    output_root = Path(args.output_root)
    figure_dir = Path(args.figure_dir)
    captions_path = Path(args.captions_path)

    endpoint_report_path = output_root / "strong_hypotheses" / "endpoint_tests.csv"
    endpoint_report = pd.read_csv(endpoint_report_path)
    network_df = pd.read_csv(output_root / "subject_network_means.csv")
    robustness = _collect_robustness_rows(
        network_df=network_df,
        endpoint_report=endpoint_report,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    _style()
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    endpoint_order = [spec.endpoint_id for spec in STRONG_ENDPOINTS]
    label_map = {spec.endpoint_id: spec.label for spec in STRONG_ENDPOINTS}
    base_y = np.arange(len(endpoint_order), 0, -1)
    y_map = {endpoint_id: y for endpoint_id, y in zip(endpoint_order, base_y)}
    offsets = {"Primary mean": 0.12, "Median robustness": -0.12}
    styles = {
        "Primary mean": {"marker": "o", "face": "#111111", "edge": "#111111"},
        "Median robustness": {"marker": "^", "face": "white", "edge": "#111111"},
    }

    for variant in ["Primary mean", "Median robustness"]:
        subset = robustness.loc[robustness["variant"] == variant].copy()
        style = styles[variant]
        for row in subset.itertuples(index=False):
            y = y_map[row.endpoint_id] + offsets[variant]
            ax.hlines(y, row.ci95_low, row.ci95_high, color="#555555", linewidth=1.4, zorder=2)
            ax.plot([row.ci95_low, row.ci95_low], [y - 0.06, y + 0.06], color="#555555", linewidth=1.1)
            ax.plot([row.ci95_high, row.ci95_high], [y - 0.06, y + 0.06], color="#555555", linewidth=1.1)
            ax.scatter(
                row.delta,
                y,
                marker=style["marker"],
                s=52,
                facecolor=style["face"],
                edgecolor=style["edge"],
                linewidth=1.0,
                zorder=3,
            )

    ax.axvline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_yticks(base_y)
    ax.set_yticklabels([label_map[eid] for eid in endpoint_order])
    ax.set_xlabel("Mean difference (Converter - Non-converter)")
    ax.set_title("Robustness check: mean vs median composite aggregation")
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.8)
    ax.set_axisbelow(True)

    for boundary in [4.5, 2.5]:
        ax.axhline(boundary, color="#c7c7c7", linewidth=0.9)
    ax.text(ax.get_xlim()[0], 6.55, "H1", ha="left", va="bottom", fontsize=10, fontweight="bold")
    ax.text(ax.get_xlim()[0], 4.55, "H2", ha="left", va="bottom", fontsize=10, fontweight="bold")
    ax.text(ax.get_xlim()[0], 2.55, "H3", ha="left", va="bottom", fontsize=10, fontweight="bold")

    xmin = float(np.min(robustness["ci95_low"])) * 1.18
    xmax = float(np.max(robustness["ci95_high"])) * 1.18
    if xmin == xmax:
        xmin -= 0.01
        xmax += 0.01
    ax.set_xlim(xmin, xmax)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", markersize=7, markerfacecolor="#111111", markeredgecolor="#111111", linewidth=0, label="Primary mean"),
        plt.Line2D([0], [0], marker="^", markersize=7, markerfacecolor="white", markeredgecolor="#111111", linewidth=0, label="Median robustness"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, edgecolor="#b8b8b8", fontsize=9)

    fig.tight_layout()
    png_path = figure_dir / "fig_robustness.png"
    pdf_path = figure_dir / "fig_robustness.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    _upsert_caption(captions_path)

    print("Robustness variant used: within-subject median instead of within-subject mean across the same included network rows")
    print("Why this variant: it uses the current exported subject_network_means.csv and does not alter endpoint definitions")
    print(f"Primary endpoint source: {endpoint_report_path}")
    print(f"Robustness figure PNG: {png_path}")
    print(f"Robustness figure PDF: {pdf_path}")


if __name__ == "__main__":
    main()
