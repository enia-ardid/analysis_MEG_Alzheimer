#!/usr/bin/env python3
from __future__ import annotations

"""Build an explanatory figure for the H1 and H2 composite definitions.

The goal is interpretability rather than inference: show which network pairs
enter each composite and how those included pairs differ between groups for AEC
and AEC-orth.

The script uses the exact composite selectors from `strong_hypotheses.py`, so
the breakdown stays tied to the methodology used in the final endpoint tests.
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

from meg_alzheimer.strong_hypotheses import _selector_temp_par_full, _selector_temp_par_inter


GROUP_A = "Converter"
GROUP_B = "Non-converter"
NETWORK_ORDER = ["Control", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "TempPar", "VisCent", "VisPeri"]
METRICS = ["AEC", "AEC-orth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H1/H2 composite breakdown figure and table.")
    parser.add_argument("--output-root", default="outputs_full_cohort", help="Existing cohort output folder.")
    parser.add_argument("--figure-dir", default="figures/final", help="Destination folder for the figure.")
    parser.add_argument("--table-dir", default="tables/final", help="Destination folder for the auxiliary table.")
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


def _write_latex(table_df: pd.DataFrame, path: Path) -> None:
    headers = list(table_df.columns)
    lines = [
        r"\begin{tabular}{p{1.2cm}p{2.4cm}p{0.9cm}p{1.0cm}p{1.2cm}p{1.3cm}p{1.3cm}p{2.0cm}p{1.2cm}p{1.3cm}p{1.3cm}p{1.3cm}p{1.4cm}}",
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


def _pair_label(row: pd.Series) -> str:
    if row["connection_type"] == "intra":
        return f"{row['network_a']}-{row['network_a']}"
    return f"{row['network_a']}-{row['network_b']}"


def _analysis_spec() -> list[dict[str, Any]]:
    return [
        {
            "hypothesis_id": "H1",
            "analysis_id": "alpha_tempPar_full_down",
            "band": "alpha",
            "selector": _selector_temp_par_full,
            "title": "H1: alpha TempPar full",
        },
        {
            "hypothesis_id": "H2",
            "analysis_id": "beta_tempPar_inter_down",
            "band": "beta",
            "selector": _selector_temp_par_inter,
            "title": "H2: beta TempPar inter",
        },
    ]


def _included_rows(network_df: pd.DataFrame, band: str, metric: str, selector) -> pd.DataFrame:
    frame = network_df.loc[(network_df["band"] == band) & (network_df["metric"] == metric)].copy()
    return frame.loc[selector(frame)].copy()


def _pair_order(rows: pd.DataFrame) -> list[str]:
    unique_rows = rows[["connection_type", "network_a", "network_b"]].drop_duplicates().copy()
    order_pairs = []
    intra = unique_rows.loc[unique_rows["connection_type"] == "intra"].copy()
    if not intra.empty:
        order_pairs.extend([f"{network}-{network}" for network in intra["network_a"].tolist()])
    inter = unique_rows.loc[unique_rows["connection_type"] == "inter"].copy()
    partner = inter.apply(lambda r: r["network_b"] if r["network_a"] == "TempPar" else r["network_a"], axis=1)
    inter = inter.assign(_partner=partner)
    network_rank = {name: idx for idx, name in enumerate(NETWORK_ORDER) if name != "TempPar"}
    inter = inter.sort_values("_partner", key=lambda s: s.map(network_rank))
    order_pairs.extend(inter.apply(_pair_label, axis=1).tolist())
    return order_pairs


def _build_breakdown_table(network_df: pd.DataFrame) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for spec in _analysis_spec():
        reference = _included_rows(network_df, band=spec["band"], metric="AEC", selector=spec["selector"])
        pair_order = _pair_order(reference)
        weight = 1.0 / len(pair_order)
        for metric in METRICS:
            frame = _included_rows(network_df, band=spec["band"], metric=metric, selector=spec["selector"]).copy()
            summary = (
                frame.groupby(["connection_type", "network_a", "network_b", "group"], as_index=False)["value"]
                .mean()
                .pivot(index=["connection_type", "network_a", "network_b"], columns="group", values="value")
                .reset_index()
            )
            summary["pair_label"] = summary.apply(_pair_label, axis=1)
            summary["pair_order"] = summary["pair_label"].map({label: idx for idx, label in enumerate(pair_order)})
            summary = summary.sort_values("pair_order").reset_index(drop=True)
            summary["delta_group_a_minus_group_b"] = summary[GROUP_A] - summary[GROUP_B]
            summary["weight_in_composite"] = weight
            summary["weighted_delta_contribution"] = summary["delta_group_a_minus_group_b"] * weight
            summary["hypothesis_id"] = spec["hypothesis_id"]
            summary["analysis_id"] = spec["analysis_id"]
            summary["band"] = spec["band"]
            summary["metric"] = metric
            rows_out.extend(summary.to_dict(orient="records"))
    breakdown = pd.DataFrame(rows_out)
    return breakdown[
        [
            "hypothesis_id",
            "analysis_id",
            "band",
            "metric",
            "connection_type",
            "network_a",
            "network_b",
            "pair_label",
            "weight_in_composite",
            GROUP_A,
            GROUP_B,
            "delta_group_a_minus_group_b",
            "weighted_delta_contribution",
        ]
    ].rename(
        columns={
            GROUP_A: "mean_converter",
            GROUP_B: "mean_non_converter",
        }
    )


def _mask_matrix(pair_labels: list[str]) -> np.ndarray:
    mat = np.zeros((len(NETWORK_ORDER), len(NETWORK_ORDER)), dtype=float)
    index = {network: idx for idx, network in enumerate(NETWORK_ORDER)}
    for label in pair_labels:
        a, b = label.split("-")
        i = index[a]
        j = index[b]
        mat[i, j] = 1.0
        mat[j, i] = 1.0
    return mat


def _plot_mask(ax: plt.Axes, pair_labels: list[str], title: str) -> None:
    mask = _mask_matrix(pair_labels)
    temp_idx = NETWORK_ORDER.index("TempPar")
    ax.imshow(mask, cmap="Greys", vmin=0.0, vmax=1.0, origin="upper")
    ax.set_xticks(np.arange(len(NETWORK_ORDER)))
    ax.set_yticks(np.arange(len(NETWORK_ORDER)))
    ax.set_xticklabels(NETWORK_ORDER, rotation=45, ha="right")
    ax.set_yticklabels(NETWORK_ORDER)
    ax.set_title(title)
    ax.tick_params(length=0)
    for idx, tick in enumerate(ax.get_xticklabels()):
        if idx == temp_idx:
            tick.set_fontweight("bold")
    for idx, tick in enumerate(ax.get_yticklabels()):
        if idx == temp_idx:
            tick.set_fontweight("bold")
    lo = temp_idx - 0.5
    hi = temp_idx + 0.5
    ax.axhline(lo, color="#111111", linewidth=1.1)
    ax.axhline(hi, color="#111111", linewidth=1.1)
    ax.axvline(lo, color="#111111", linewidth=1.1)
    ax.axvline(hi, color="#111111", linewidth=1.1)


def _plot_bars(ax: plt.Axes, data: pd.DataFrame, title: str, xlim: float, show_ylabels: bool) -> None:
    ypos = np.arange(len(data))[::-1]
    labels = data["pair_label"].tolist()
    vals = data["delta_group_a_minus_group_b"].to_numpy(dtype=float)
    ax.barh(ypos, vals, color="#6d6d6d", edgecolor="#111111", linewidth=0.7)
    ax.axvline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xlim(-xlim, xlim)
    ax.set_title(title)
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.8)
    ax.set_axisbelow(True)
    if show_ylabels:
        ax.set_yticks(ypos, labels)
    else:
        ax.set_yticks(ypos, [])
    ax.set_xlabel("Mean difference (C - NC)")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    figure_dir = Path(args.figure_dir)
    table_dir = Path(args.table_dir)
    captions_figures = Path(args.captions_figures)
    captions_tables = Path(args.captions_tables)

    network_df = pd.read_csv(output_root / "subject_network_means.csv")
    breakdown = _build_breakdown_table(network_df)

    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    table_csv = table_dir / "table_composite_breakdown.csv"
    table_tex = table_dir / "table_composite_breakdown.tex"
    breakdown_rounded = breakdown.copy()
    for col in ["weight_in_composite", "mean_converter", "mean_non_converter", "delta_group_a_minus_group_b", "weighted_delta_contribution"]:
        breakdown_rounded[col] = breakdown_rounded[col].map(lambda x: f"{x:.6f}")
    breakdown_rounded.to_csv(table_csv, index=False)
    _write_latex(breakdown_rounded, table_tex)

    _style()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.6, 9.0),
        gridspec_kw={"width_ratios": [1.15, 1.25, 1.25]},
        constrained_layout=True,
    )

    for row_idx, spec in enumerate(_analysis_spec()):
        subset = breakdown.loc[breakdown["hypothesis_id"] == spec["hypothesis_id"]].copy()
        pair_labels = (
            subset.loc[subset["metric"] == "AEC", "pair_label"]
            .drop_duplicates()
            .tolist()
        )
        _plot_mask(
            axes[row_idx, 0],
            pair_labels=pair_labels,
            title=f"{spec['title']}\nIncluded network pairs (equal weight = 1/{len(pair_labels)})",
        )

        aec = subset.loc[subset["metric"] == "AEC"].copy()
        orth = subset.loc[subset["metric"] == "AEC-orth"].copy()
        xlim = float(
            np.max(
                np.abs(
                    np.r_[
                        aec["delta_group_a_minus_group_b"].to_numpy(dtype=float),
                        orth["delta_group_a_minus_group_b"].to_numpy(dtype=float),
                    ]
                )
            )
        )
        xlim = max(xlim * 1.15, 1e-3)

        _plot_bars(
            axes[row_idx, 1],
            data=aec,
            title=f"{spec['title']} / AEC",
            xlim=xlim,
            show_ylabels=True,
        )
        _plot_bars(
            axes[row_idx, 2],
            data=orth,
            title=f"{spec['title']} / AEC-orth",
            xlim=xlim,
            show_ylabels=False,
        )

    png_path = figure_dir / "fig_composite_breakdown.png"
    pdf_path = figure_dir / "fig_composite_breakdown.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    _upsert_caption(
        captions_figures,
        "## Figure: Composite breakdown for H1 and H2",
        "Suggested caption: Breakdown of the network pairs included in the H1 and H2 composites. "
        "The left column shows the inclusion mask at the collapsed 9-network level used by the main pipeline, "
        "with TempPar highlighted on both axes. The middle and right columns show the group difference "
        "(Converter minus Non-converter) for each included network pair in AEC and AEC-orth, respectively. "
        "All included pairs have equal weight within each composite: 1/9 for H1 and 1/8 for H2.",
    )
    _upsert_caption(
        captions_tables,
        "## Table: Composite breakdown for H1 and H2",
        "Suggested caption: Network-pair breakdown of the H1 and H2 composites. Each row corresponds to one "
        "included network pair and metric. Because the composites are arithmetic means over the selected "
        "network-summary rows, all included pairs have identical weight within the corresponding composite. "
        "The table reports group means, the group difference (Converter minus Non-converter), and the weighted "
        "contribution of each pair to the composite-level group difference.",
    )

    print(f"Input: {output_root / 'subject_network_means.csv'}")
    print("Composite definitions:")
    print("- H1 = alpha TempPar full = all inter pairs involving TempPar + TempPar intra")
    print("- H2 = beta TempPar inter = all inter pairs involving TempPar")
    print("Weights:")
    print("- H1: equal weight 1/9 per included pair")
    print("- H2: equal weight 1/8 per included pair")
    print(f"PNG figure: {png_path}")
    print(f"PDF figure: {pdf_path}")
    print(f"CSV table: {table_csv}")
    print(f"LaTeX table: {table_tex}")


if __name__ == "__main__":
    main()
