#!/usr/bin/env python3
from __future__ import annotations

"""Build a forest plot for the six predefined study endpoints.

The plot is intentionally tied to the same statistical source used by the main
endpoint figure and table:

- `outputs_full_cohort/strong_hypotheses/endpoint_tests.csv`
- `tables/final/table_endpoints_main.csv`

This keeps the visual summary aligned with the final reported endpoint values.
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

from meg_alzheimer.strong_hypotheses import STRONG_ENDPOINTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the forest plot for the six study endpoints.")
    parser.add_argument("--output-root", default="outputs_full_cohort", help="Existing cohort output folder.")
    parser.add_argument("--table-path", default="tables/final/table_endpoints_main.csv", help="Final endpoint table.")
    parser.add_argument("--figure-dir", default="figures/final", help="Destination folder for figure files.")
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
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _ordered_endpoint_report(endpoint_report: pd.DataFrame) -> pd.DataFrame:
    order = {spec.endpoint_id: idx for idx, spec in enumerate(STRONG_ENDPOINTS)}
    labels = {spec.endpoint_id: spec.label for spec in STRONG_ENDPOINTS}
    report = endpoint_report.copy()
    report["plot_order"] = report["endpoint_id"].map(order)
    report["plot_label"] = report["endpoint_id"].map(labels)
    return report.sort_values("plot_order").reset_index(drop=True)


def _validate_consistency(endpoint_report: pd.DataFrame, final_table: pd.DataFrame) -> None:
    table = final_table.rename(
        columns={
            "Endpoint": "endpoint_id",
            "Delta (C-NC)": "delta_table",
            "Cohen d": "d_table",
            "Holm p": "holm_table",
            "max-T p": "max_t_table",
        }
    )
    merged = endpoint_report.merge(
        table[["endpoint_id", "delta_table", "d_table", "holm_table", "max_t_table"]],
        on="endpoint_id",
        how="inner",
    )
    if len(merged) != len(endpoint_report):
        raise ValueError("The final endpoint table does not contain the same endpoint set as endpoint_tests.csv.")
    checks = [
        ("delta_group_a_minus_group_b", "delta_table"),
        ("effect_size_d", "d_table"),
        ("holm_p_one_sided", "holm_table"),
        ("max_t_p_one_sided", "max_t_table"),
    ]
    mismatches: list[str] = []
    for report_col, table_col in checks:
        report_vals = merged[report_col].astype(float).to_numpy()
        table_vals = merged[table_col].astype(float).to_numpy()
        if not np.allclose(report_vals, table_vals, atol=5e-4):
            mismatches.append(f"{report_col} vs {table_col}")
    if mismatches:
        raise ValueError(
            "The forest plot inputs are inconsistent with the final endpoint table: "
            + ", ".join(mismatches)
        )


def _significance_style(row: pd.Series) -> tuple[str, str, float]:
    if bool(row["holm_significant"]) and bool(row["max_t_significant"]):
        return "D", "#111111", 1.0
    if bool(row["holm_significant"]):
        return "s", "#4a4a4a", 1.0
    if float(row["p_one_sided"]) < 0.05:
        return "o", "#7f7f7f", 1.0
    return "o", "white", 1.0


def _upsert_caption(path: Path) -> None:
    heading = "## Figure: Endpoint forest plot"
    block = (
        f"{heading}\n\n"
        "Suggested caption: Forest plot of the six predefined study endpoints, shown as the mean difference "
        "(Converter minus Non-converter) with 95% confidence intervals from the strong endpoint report. "
        "Endpoints are ordered by hypothesis family (H1, H2, H3). Marker shape and fill encode statistical "
        "status: hollow circles indicate no nominal significance, filled circles indicate nominal one-sided "
        "significance only, filled squares indicate survival after Holm adjustment, and filled diamonds indicate "
        "survival after both Holm and max-T correction.\n"
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
    table_path = Path(args.table_path)
    figure_dir = Path(args.figure_dir)
    captions_path = Path(args.captions_path)

    endpoint_report_path = output_root / "strong_hypotheses" / "endpoint_tests.csv"
    endpoint_report = pd.read_csv(endpoint_report_path)
    final_table = pd.read_csv(table_path)
    _validate_consistency(endpoint_report, final_table)
    endpoint_report = _ordered_endpoint_report(endpoint_report)

    _style()
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    y_positions = np.arange(len(endpoint_report), 0, -1)

    for y, row in zip(y_positions, endpoint_report.itertuples(index=False)):
        marker, facecolor, alpha = _significance_style(pd.Series(row._asdict()))
        delta = float(row.delta_group_a_minus_group_b)
        ci_low = float(row.ci95_low)
        ci_high = float(row.ci95_high)
        ax.hlines(y, ci_low, ci_high, color="#444444", linewidth=1.5, zorder=2)
        ax.plot([ci_low, ci_low], [y - 0.08, y + 0.08], color="#444444", linewidth=1.2)
        ax.plot([ci_high, ci_high], [y - 0.08, y + 0.08], color="#444444", linewidth=1.2)
        ax.scatter(
            delta,
            y,
            marker=marker,
            s=64,
            facecolor=facecolor,
            edgecolor="#111111",
            linewidth=1.1,
            alpha=alpha,
            zorder=3,
        )

    ax.axvline(0.0, color="#888888", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(endpoint_report["plot_label"])
    ax.set_xlabel("Mean difference (Converter - Non-converter)")
    ax.set_title("Forest plot of the six study endpoints")
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.8)
    ax.set_axisbelow(True)

    xmin = float(np.min(endpoint_report["ci95_low"])) * 1.18
    xmax = float(np.max(endpoint_report["ci95_high"])) * 1.18
    if xmin == xmax:
        xmin -= 0.01
        xmax += 0.01
    ax.set_xlim(xmin, xmax)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", markersize=7, markerfacecolor="white", markeredgecolor="#111111", linewidth=0, label="Not nominal"),
        plt.Line2D([0], [0], marker="o", markersize=7, markerfacecolor="#7f7f7f", markeredgecolor="#111111", linewidth=0, label="Nominal only"),
        plt.Line2D([0], [0], marker="s", markersize=7, markerfacecolor="#4a4a4a", markeredgecolor="#111111", linewidth=0, label="Holm"),
        plt.Line2D([0], [0], marker="D", markersize=7, markerfacecolor="#111111", markeredgecolor="#111111", linewidth=0, label="Holm + max-T"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, edgecolor="#b8b8b8", fontsize=8.8)

    fig.tight_layout()
    png_path = figure_dir / "fig_forest_endpoints.png"
    pdf_path = figure_dir / "fig_forest_endpoints.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    _upsert_caption(captions_path)

    print(f"Input endpoint report: {endpoint_report_path}")
    print(f"Consistency check against final table: {table_path} passed")
    print(f"PNG figure: {png_path}")
    print(f"PDF figure: {pdf_path}")


if __name__ == "__main__":
    main()
