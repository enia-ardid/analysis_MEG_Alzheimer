#!/usr/bin/env python3
from __future__ import annotations

"""Build the main figure and table for the six strong-study endpoints.

The script intentionally reuses the endpoint definitions implemented in
``meg_alzheimer.strong_hypotheses`` and the statistical values already exported
to ``outputs_full_cohort/strong_hypotheses/endpoint_tests.csv``. This avoids
silent methodological drift between the thesis figure/table and the analysis
pipeline that produced the reported endpoint statistics.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from meg_alzheimer.strong_hypotheses import STRONG_ENDPOINTS, _extract_endpoint_frame


GROUP_A = "Converter"
GROUP_B = "Non-converter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the main endpoint figure and table.")
    parser.add_argument("--output-root", default="outputs_full_cohort", help="Existing cohort output folder.")
    parser.add_argument("--figure-dir", default="figures/final", help="Destination folder for final figures.")
    parser.add_argument("--table-dir", default="tables/final", help="Destination folder for final tables.")
    parser.add_argument(
        "--captions-figures",
        default="captions_figures.md",
        help="Markdown file where the suggested figure caption will be written.",
    )
    parser.add_argument(
        "--captions-tables",
        default="captions_tables.md",
        help="Markdown file where the suggested table caption will be written.",
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
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _bootstrap_ci_mean(values: np.ndarray, seed: int = 0, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[idx] = float(np.mean(sample))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _latex_escape(value: Any) -> str:
    text = "--" if value is None or (isinstance(value, float) and not np.isfinite(value)) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _write_latex_table(table_df: pd.DataFrame, path: Path) -> None:
    headers = list(table_df.columns)
    lines = [
        r"\begin{tabular}{p{1.6cm}p{1.2cm}p{4.1cm}p{2.8cm}p{1.9cm}p{1.3cm}p{1.5cm}p{1.4cm}p{1.2cm}p{1.2cm}p{1.2cm}}",
        r"\toprule",
        " & ".join(_latex_escape(col) for col in headers) + r" \\",
        r"\midrule",
    ]
    for row in table_df.itertuples(index=False):
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))


def _upsert_caption(path: Path, heading: str, block: str) -> None:
    text = path.read_text() if path.exists() else ""
    content = f"{heading}\n\n{block}\n"
    if heading in text:
        start = text.index(heading)
        next_idx = text.find("\n## ", start + 1)
        end = len(text) if next_idx == -1 else next_idx + 1
        text = text[:start] + content + text[end:]
    else:
        text = text.rstrip() + ("\n\n" if text.strip() else "") + content
    path.write_text(text)


def _reconstruct_endpoint_values(output_root: Path) -> dict[str, pd.DataFrame]:
    network_df = pd.read_csv(output_root / "subject_network_means.csv")
    endpoint_frames: dict[str, pd.DataFrame] = {}
    for spec in STRONG_ENDPOINTS:
        frame = _extract_endpoint_frame(spec, network_df=network_df).copy()
        endpoint_frames[spec.endpoint_id] = frame
    return endpoint_frames


def _validate_against_report(endpoint_frames: dict[str, pd.DataFrame], endpoint_report: pd.DataFrame) -> None:
    report_indexed = endpoint_report.set_index("endpoint_id")
    discrepancies: list[str] = []
    for spec in STRONG_ENDPOINTS:
        frame = endpoint_frames[spec.endpoint_id]
        observed_a = float(frame.loc[frame["group"] == GROUP_A, "value"].mean())
        observed_b = float(frame.loc[frame["group"] == GROUP_B, "value"].mean())
        observed_delta = observed_a - observed_b
        expected = report_indexed.loc[spec.endpoint_id]
        if not np.isclose(observed_a, expected["mean_group_a"], atol=1e-10):
            discrepancies.append(f"{spec.endpoint_id}: mean_group_a mismatch")
        if not np.isclose(observed_b, expected["mean_group_b"], atol=1e-10):
            discrepancies.append(f"{spec.endpoint_id}: mean_group_b mismatch")
        if not np.isclose(observed_delta, expected["delta_group_a_minus_group_b"], atol=1e-10):
            discrepancies.append(f"{spec.endpoint_id}: delta mismatch")
    if discrepancies:
        raise ValueError(
            "The reconstructed endpoint values do not match the reported endpoint table:\n- "
            + "\n- ".join(discrepancies)
        )


def _endpoint_table(endpoint_report: pd.DataFrame) -> pd.DataFrame:
    label_map = {spec.endpoint_id: spec.label for spec in STRONG_ENDPOINTS}
    analysis_map = {spec.endpoint_id: spec.analysis_id for spec in STRONG_ENDPOINTS}
    metric_map = {spec.endpoint_id: spec.metric for spec in STRONG_ENDPOINTS}
    table = endpoint_report.copy()
    table["endpoint_label"] = table["endpoint_id"].map(label_map)
    table["analysis_origin"] = table["endpoint_id"].map(analysis_map)
    table["metric_origin"] = table["endpoint_id"].map(metric_map)
    table = table[
        [
            "endpoint_id",
            "hypothesis_id",
            "endpoint_label",
            "analysis_origin",
            "metric_origin",
            "mean_group_a",
            "mean_group_b",
            "delta_group_a_minus_group_b",
            "effect_size_d",
            "holm_p_one_sided",
            "max_t_p_one_sided",
        ]
    ].rename(
        columns={
            "endpoint_id": "Endpoint",
            "hypothesis_id": "Hypothesis",
            "endpoint_label": "Definition",
            "analysis_origin": "Composite origin",
            "metric_origin": "Metric",
            "mean_group_a": "Mean Converter",
            "mean_group_b": "Mean Non-converter",
            "delta_group_a_minus_group_b": "Delta (C-NC)",
            "effect_size_d": "Cohen d",
            "holm_p_one_sided": "Holm p",
            "max_t_p_one_sided": "max-T p",
        }
    )
    numeric_cols = ["Mean Converter", "Mean Non-converter", "Delta (C-NC)", "Cohen d", "Holm p", "max-T p"]
    for col in numeric_cols:
        table[col] = table[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "--")
    return table


def _draw_endpoint_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    row: pd.Series,
    title: str,
    seed: int,
) -> None:
    values_a = frame.loc[frame["group"] == GROUP_A, "value"].to_numpy(dtype=float)
    values_b = frame.loc[frame["group"] == GROUP_B, "value"].to_numpy(dtype=float)
    values = [values_a, values_b]
    positions = [1, 2]

    box = ax.boxplot(values, positions=positions, widths=0.52, patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("#111111")
        patch.set_linewidth(1.2)
    for artist_name in ("whiskers", "caps", "medians"):
        for artist in box[artist_name]:
            artist.set_color("#111111")
            artist.set_linewidth(1.1)

    rng = np.random.default_rng(seed)
    point_face = {"Converter": "#5a5a5a", "Non-converter": "#a0a0a0"}
    for xpos, group, vals in zip(positions, [GROUP_A, GROUP_B], values):
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(vals))
        ax.scatter(
            np.full(len(vals), xpos, dtype=float) + jitter,
            vals,
            s=16,
            facecolor=point_face[group],
            edgecolor="#111111",
            linewidth=0.3,
            alpha=0.85,
            zorder=3,
        )
        mean_val = float(np.mean(vals))
        ci_low, ci_high = _bootstrap_ci_mean(vals, seed=seed + xpos)
        ax.errorbar(
            xpos,
            mean_val,
            yerr=[[mean_val - ci_low], [ci_high - mean_val]],
            fmt="o",
            color="#111111",
            markersize=4.8,
            capsize=3,
            linewidth=1.1,
            zorder=4,
        )

    ax.set_xticks(positions, ["Converter", "Non-converter"])
    ax.set_title(title)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8)
    ax.set_axisbelow(True)
    annotation = (
        f"Δ={row['delta_group_a_minus_group_b']:.4f}\n"
        f"d={row['effect_size_d']:.3f}\n"
        f"Holm={row['holm_p_one_sided']:.4g}\n"
        f"max-T={row['max_t_p_one_sided']:.4g}"
    )
    ax.text(
        0.98,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.6,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#b8b8b8", "alpha": 0.95},
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    figure_dir = Path(args.figure_dir)
    table_dir = Path(args.table_dir)
    captions_figures = Path(args.captions_figures)
    captions_tables = Path(args.captions_tables)

    endpoint_report_path = output_root / "strong_hypotheses" / "endpoint_tests.csv"
    if not endpoint_report_path.exists():
        raise SystemExit(f"Missing endpoint report: {endpoint_report_path}")

    _style()
    endpoint_report = pd.read_csv(endpoint_report_path)
    endpoint_frames = _reconstruct_endpoint_values(output_root)
    _validate_against_report(endpoint_frames, endpoint_report)

    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(11.4, 11.8), constrained_layout=True)
    fig.suptitle("Main endpoint distributions", fontsize=15)
    for idx, spec in enumerate(STRONG_ENDPOINTS):
        ax = axes.flat[idx]
        frame = endpoint_frames[spec.endpoint_id]
        row = endpoint_report.loc[endpoint_report["endpoint_id"] == spec.endpoint_id].iloc[0]
        _draw_endpoint_panel(ax, frame=frame, row=row, title=spec.label, seed=idx)
        if idx % 2 == 0:
            ax.set_ylabel("Composite value")
    png_path = figure_dir / "fig_endpoints_distributions.png"
    pdf_path = figure_dir / "fig_endpoints_distributions.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    table_df = _endpoint_table(endpoint_report)
    csv_path = table_dir / "table_endpoints_main.csv"
    tex_path = table_dir / "table_endpoints_main.tex"
    table_df.to_csv(csv_path, index=False)
    _write_latex_table(table_df, tex_path)

    _upsert_caption(
        captions_figures,
        "## Figure: Main endpoint distributions",
        "Suggested caption: Distribution of the six predefined study endpoints across Converter and "
        "Non-converter subjects. Each panel shows the subject-level endpoint values reconstructed from "
        "`subject_network_means.csv` using the same endpoint definitions as the strong H1-H3 analysis. "
        "Individual points are overlaid on boxplots, black markers indicate group means with bootstrap 95% "
        "confidence intervals, and each panel reports the mean difference, Cohen's d, Holm-adjusted one-sided "
        "p-value, and max-T corrected one-sided p-value from the existing strong endpoint report.",
    )
    _upsert_caption(
        captions_tables,
        "## Table: Main endpoint statistics",
        "Suggested caption: Main statistical summary of the six predefined study endpoints. Endpoint definitions, "
        "composite origins, and metrics are identical to those used in the strong H1-H3 confirmation step. "
        "The table reports group means, mean differences (Converter minus Non-converter), Cohen's d, Holm-adjusted "
        "one-sided p-values, and max-T corrected one-sided p-values.",
    )

    print(f"Input endpoint report: {endpoint_report_path}")
    print("Input endpoint values: reconstructed from outputs_full_cohort/subject_network_means.csv")
    print("Discrepancy check: no differences detected between reconstructed means and the reported endpoint table")
    print(f"PNG figure: {png_path}")
    print(f"PDF figure: {pdf_path}")
    print(f"CSV table: {csv_path}")
    print(f"LaTeX table: {tex_path}")


if __name__ == "__main__":
    main()
