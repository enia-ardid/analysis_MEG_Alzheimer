#!/usr/bin/env python3
from __future__ import annotations

"""Check a post hoc exact Schaefer-17 beta/AEC-orth network block.

The main thesis hypotheses H1-H3 are defined on the collapsed 9-network
summary used by the core pipeline. This script does not modify that family.
Instead, it performs a separate post hoc check motivated by visual inspection
of the exact Schaefer-17 beta AEC-orth heatmap.

Target block
------------
The block of interest is the 2x2 submatrix spanned by:

- SomMotA
- DorsAttnB

At the network level this yields three unique cells:

- SomMotA-SomMotA
- SomMotA-DorsAttnB
- DorsAttnB-DorsAttnB

Primary and robustness summaries
--------------------------------
To avoid turning the block into a black box, the script reports the three cells
individually and defines two simple block summaries over the same three values:

- ``block_mean``: arithmetic mean of the three unique cells
- ``block_median``: median of the same three unique cells

The block mean is treated as the primary post hoc summary and the block median
as a narrow robustness variant. Both tests are two-sided because the direction
was prompted by visual inspection rather than by a predeclared confirmatory
hypothesis.
"""

import argparse
import json
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
from scipy import stats

from meg_alzheimer.atlas import get_network_prefix, mean_inter, mean_intra


GROUP_A = "Converter"
GROUP_B = "Non-converter"
NETWORK_A = "SomMotA"
NETWORK_B = "DorsAttnB"
ANALYSIS_LABEL = "Post hoc exact Schaefer-17 beta / AEC-orth block check"
BLOCK_ORDER = (
    "SomMotA-SomMotA",
    "SomMotA-DorsAttnB",
    "DorsAttnB-DorsAttnB",
    "block_mean",
    "block_median",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the post hoc beta/AEC-orth SomMotA-DorsAttnB block check.")
    parser.add_argument(
        "--subjects-root",
        default="outputs_full_cohort/subjects",
        help="Folder containing per-subject connectivity_matrices.npz and metadata.json files.",
    )
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
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap count for confidence intervals.")
    parser.add_argument("--n-perm", type=int, default=50000, help="Permutation count for block-level p-values.")
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


def _iter_subject_dirs(subjects_root: Path) -> list[Path]:
    return sorted(path for path in subjects_root.glob("*/*") if path.is_dir())


def _load_subject(subject_dir: Path) -> dict[str, Any]:
    with open(subject_dir / "metadata.json") as f:
        meta = json.load(f)
    matrices = np.load(subject_dir / "connectivity_matrices.npz")
    return {"subject_dir": subject_dir, "metadata": meta, "matrices": matrices}


def _raw_network_labels(roi_labels: list[str]) -> list[str]:
    labels = [get_network_prefix(label) for label in roi_labels]
    return ["Background" if label.startswith("Background+FreeSurfer") else label for label in labels]


def _validate_subjects(subject_payloads: list[dict[str, Any]]) -> list[str]:
    if not subject_payloads:
        raise ValueError("No subject folders found.")
    first_labels = subject_payloads[0]["metadata"]["roi_labels"]
    for payload in subject_payloads[1:]:
        if payload["metadata"]["roi_labels"] != first_labels:
            raise ValueError("ROI labels differ across subjects. The exact Schaefer-17 block cannot be compared safely.")
    raw_labels = _raw_network_labels(first_labels)
    present = set(raw_labels)
    missing = [name for name in (NETWORK_A, NETWORK_B) if name not in present]
    if missing:
        raise ValueError(f"Required exact Schaefer-17 networks missing from ROI labels: {missing}")
    return raw_labels


def _cohen_d_unpaired(values_a: np.ndarray, values_b: np.ndarray) -> float:
    mean_diff = float(np.mean(values_a) - np.mean(values_b))
    var_a = float(np.var(values_a, ddof=1))
    var_b = float(np.var(values_b, ddof=1))
    pooled = np.sqrt(((len(values_a) - 1) * var_a + (len(values_b) - 1) * var_b) / max(1, len(values_a) + len(values_b) - 2))
    if pooled < 1e-12:
        return float("nan")
    return mean_diff / pooled


def _bootstrap_ci_delta(values_a: np.ndarray, values_b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample_a = rng.choice(values_a, size=len(values_a), replace=True)
        sample_b = rng.choice(values_b, size=len(values_b), replace=True)
        diffs[idx] = float(np.mean(sample_a) - np.mean(sample_b))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


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


def _permutation_p_two_sided(values_a: np.ndarray, values_b: np.ndarray, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([values_a, values_b])
    n_a = len(values_a)
    observed = abs(float(np.mean(values_a) - np.mean(values_b)))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = abs(float(np.mean(perm[:n_a]) - np.mean(perm[n_a:])))
        if diff >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


def _collect_subject_rows(subject_payloads: list[dict[str, Any]], raw_labels: list[str]) -> pd.DataFrame:
    mask_a = np.array([label == NETWORK_A for label in raw_labels], dtype=bool)
    mask_b = np.array([label == NETWORK_B for label in raw_labels], dtype=bool)

    rows: list[dict[str, Any]] = []
    for payload in subject_payloads:
        meta = payload["metadata"]
        matrix = np.asarray(payload["matrices"]["beta__AEC-orth"], dtype=float)
        cell_aa = mean_intra(matrix, mask_a)
        cell_ab = mean_inter(matrix, mask_a, mask_b)
        cell_bb = mean_intra(matrix, mask_b)
        block_values = np.array([cell_aa, cell_ab, cell_bb], dtype=float)

        rows.extend(
            [
                {
                    "subject_id": meta["subject_id"],
                    "group": meta["group"],
                    "analysis_id": "SomMotA-SomMotA",
                    "value": cell_aa,
                },
                {
                    "subject_id": meta["subject_id"],
                    "group": meta["group"],
                    "analysis_id": "SomMotA-DorsAttnB",
                    "value": cell_ab,
                },
                {
                    "subject_id": meta["subject_id"],
                    "group": meta["group"],
                    "analysis_id": "DorsAttnB-DorsAttnB",
                    "value": cell_bb,
                },
                {
                    "subject_id": meta["subject_id"],
                    "group": meta["group"],
                    "analysis_id": "block_mean",
                    "value": float(np.mean(block_values)),
                },
                {
                    "subject_id": meta["subject_id"],
                    "group": meta["group"],
                    "analysis_id": "block_median",
                    "value": float(np.median(block_values)),
                },
            ]
        )
    frame = pd.DataFrame(rows)
    order = {name: idx for idx, name in enumerate(BLOCK_ORDER)}
    frame["analysis_order"] = frame["analysis_id"].map(order)
    return frame.sort_values(["analysis_order", "group", "subject_id"]).reset_index(drop=True)


def _stats_table(subject_values: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, analysis_id in enumerate(BLOCK_ORDER):
        subset = subject_values.loc[subject_values["analysis_id"] == analysis_id]
        values_a = subset.loc[subset["group"] == GROUP_A, "value"].to_numpy(dtype=float)
        values_b = subset.loc[subset["group"] == GROUP_B, "value"].to_numpy(dtype=float)
        t_stat, p_two = stats.ttest_ind(values_a, values_b, equal_var=False, nan_policy="omit")
        delta = float(np.mean(values_a) - np.mean(values_b))
        ci_low, ci_high = _bootstrap_ci_delta(values_a, values_b, n_boot=n_boot, seed=seed + idx)
        is_block = analysis_id in {"block_mean", "block_median"}
        rows.append(
            {
                "analysis_id": analysis_id,
                "analysis_type": "block" if is_block else "cell",
                "mean_converter": float(np.mean(values_a)),
                "mean_non_converter": float(np.mean(values_b)),
                "delta_converter_minus_non_converter": delta,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "welch_t": float(t_stat),
                "p_two_sided": float(p_two),
                "effect_size_d": _cohen_d_unpaired(values_a, values_b),
                "n_converter": int(len(values_a)),
                "n_non_converter": int(len(values_b)),
            }
        )
    table = pd.DataFrame(rows)

    cell_mask = table["analysis_type"] == "cell"
    table.loc[cell_mask, "holm_p_two_sided_cells"] = _holm_adjust(table.loc[cell_mask, "p_two_sided"].to_numpy(dtype=float))
    table.loc[~cell_mask, "holm_p_two_sided_cells"] = np.nan

    for offset, analysis_id in enumerate(("block_mean", "block_median"), start=len(BLOCK_ORDER)):
        subset = subject_values.loc[subject_values["analysis_id"] == analysis_id]
        values_a = subset.loc[subset["group"] == GROUP_A, "value"].to_numpy(dtype=float)
        values_b = subset.loc[subset["group"] == GROUP_B, "value"].to_numpy(dtype=float)
        p_perm = _permutation_p_two_sided(values_a, values_b, n_perm=n_perm, seed=seed + 1000 + offset)
        table.loc[table["analysis_id"] == analysis_id, "permutation_p_two_sided"] = p_perm

    table.loc[~table["analysis_id"].isin(["block_mean", "block_median"]), "permutation_p_two_sided"] = np.nan

    order = {name: idx for idx, name in enumerate(BLOCK_ORDER)}
    table["analysis_order"] = table["analysis_id"].map(order)
    return table.sort_values("analysis_order").reset_index(drop=True)


def _delta_matrix(stats_table: pd.DataFrame) -> np.ndarray:
    delta_lookup = stats_table.set_index("analysis_id")["delta_converter_minus_non_converter"]
    return np.array(
        [
            [
                delta_lookup["SomMotA-SomMotA"],
                delta_lookup["SomMotA-DorsAttnB"],
            ],
            [
                delta_lookup["SomMotA-DorsAttnB"],
                delta_lookup["DorsAttnB-DorsAttnB"],
            ],
        ],
        dtype=float,
    )


def _marker_style(row: pd.Series) -> tuple[str, str]:
    if row["analysis_type"] == "block" and pd.notna(row["permutation_p_two_sided"]) and row["permutation_p_two_sided"] <= 0.05:
        return "D", "#111111"
    if row["analysis_type"] == "cell" and row["holm_p_two_sided_cells"] <= 0.05:
        return "s", "#111111"
    if row["p_two_sided"] <= 0.05:
        return "o", "#111111"
    return "o", "white"


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
        r"\begin{tabular}{p{2.8cm}p{1.2cm}p{1.5cm}p{1.7cm}p{1.5cm}p{1.5cm}p{1.2cm}p{1.2cm}p{1.2cm}p{1.6cm}}",
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


def main() -> None:
    args = parse_args()
    subjects_root = Path(args.subjects_root)
    figure_dir = Path(args.figure_dir)
    table_dir = Path(args.table_dir)
    captions_figures = Path(args.captions_figures)
    captions_tables = Path(args.captions_tables)

    subject_payloads = [_load_subject(path) for path in _iter_subject_dirs(subjects_root)]
    raw_labels = _validate_subjects(subject_payloads)
    subject_values = _collect_subject_rows(subject_payloads, raw_labels=raw_labels)
    stats_table = _stats_table(subject_values, n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed)

    _style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), constrained_layout=True)

    heatmap = _delta_matrix(stats_table)
    limit = float(np.max(np.abs(heatmap)))
    limit = limit if limit > 0 else 1e-9
    im = axes[0].imshow(heatmap, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    axes[0].set_title("Block difference matrix\nConverter - Non-converter")
    axes[0].set_xticks([0, 1], [NETWORK_A, NETWORK_B], rotation=45, ha="right")
    axes[0].set_yticks([0, 1], [NETWORK_A, NETWORK_B])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{heatmap[i, j]:.4f}", ha="center", va="center", color="#111111", fontsize=9)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("Mean difference")

    plot_rows = stats_table.copy()
    y_positions = np.arange(len(plot_rows), 0, -1)
    for y, row in zip(y_positions, plot_rows.itertuples(index=False)):
        axes[1].hlines(y, row.ci95_low, row.ci95_high, color="#555555", linewidth=1.4, zorder=2)
        axes[1].plot([row.ci95_low, row.ci95_low], [y - 0.06, y + 0.06], color="#555555", linewidth=1.0)
        axes[1].plot([row.ci95_high, row.ci95_high], [y - 0.06, y + 0.06], color="#555555", linewidth=1.0)
        marker, face = _marker_style(pd.Series(row._asdict()))
        axes[1].scatter(
            row.delta_converter_minus_non_converter,
            y,
            marker=marker,
            s=52 if row.analysis_type == "block" else 44,
            facecolor=face,
            edgecolor="#111111",
            linewidth=1.0,
            zorder=3,
        )
    axes[1].axvline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(
        [
            "SomMotA-SomMotA",
            "SomMotA-DorsAttnB",
            "DorsAttnB-DorsAttnB",
            "Block mean",
            "Block median",
        ]
    )
    axes[1].set_xlabel("Mean difference (Converter - Non-converter)")
    axes[1].set_title("Cell-wise and block-wise effect estimates")
    axes[1].grid(axis="x", color="#d8d8d8", linewidth=0.8)
    axes[1].set_axisbelow(True)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", markersize=6, markerfacecolor="white", markeredgecolor="#111111", linewidth=0, label="Not nominal"),
        plt.Line2D([0], [0], marker="o", markersize=6, markerfacecolor="#111111", markeredgecolor="#111111", linewidth=0, label="Nominal p < 0.05"),
        plt.Line2D([0], [0], marker="s", markersize=6, markerfacecolor="#111111", markeredgecolor="#111111", linewidth=0, label="Holm across 3 cells"),
        plt.Line2D([0], [0], marker="D", markersize=6, markerfacecolor="#111111", markeredgecolor="#111111", linewidth=0, label="Block permutation p < 0.05"),
    ]
    axes[1].legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8)

    fig.suptitle(ANALYSIS_LABEL, fontsize=13)

    fig_png = figure_dir / "fig_posthoc_beta_sommota_dorsattnb.png"
    fig_pdf = figure_dir / "fig_posthoc_beta_sommota_dorsattnb.pdf"
    fig.savefig(fig_png)
    fig.savefig(fig_pdf)
    plt.close(fig)

    table = stats_table[
        [
            "analysis_id",
            "analysis_type",
            "mean_converter",
            "mean_non_converter",
            "delta_converter_minus_non_converter",
            "ci95_low",
            "ci95_high",
            "welch_t",
            "p_two_sided",
            "holm_p_two_sided_cells",
            "permutation_p_two_sided",
            "effect_size_d",
        ]
    ].copy()
    numeric_cols = [
        "mean_converter",
        "mean_non_converter",
        "delta_converter_minus_non_converter",
        "ci95_low",
        "ci95_high",
        "welch_t",
        "p_two_sided",
        "holm_p_two_sided_cells",
        "permutation_p_two_sided",
        "effect_size_d",
    ]
    csv_path = table_dir / "table_posthoc_beta_sommota_dorsattnb.csv"
    table.to_csv(csv_path, index=False)

    latex_table = table.rename(
        columns={
            "analysis_id": "Endpoint",
            "analysis_type": "Type",
            "mean_converter": "Mean Converter",
            "mean_non_converter": "Mean Non-converter",
            "delta_converter_minus_non_converter": "Delta (C-NC)",
            "ci95_low": "CI low",
            "ci95_high": "CI high",
            "welch_t": "Welch t",
            "p_two_sided": "p two-sided",
            "holm_p_two_sided_cells": "Holm p (cells)",
            "permutation_p_two_sided": "Permutation p (blocks)",
            "effect_size_d": "Cohen d",
        }
    ).copy()
    for col in latex_table.columns:
        if col in {"Endpoint", "Type"}:
            continue
        latex_table[col] = latex_table[col].map(lambda x: "--" if pd.isna(x) else f"{x:.4f}")
    tex_path = table_dir / "table_posthoc_beta_sommota_dorsattnb.tex"
    _write_latex_table(latex_table, tex_path)

    _upsert_caption(
        captions_figures,
        "## Figure: Post hoc beta AEC-orth SomMotA-DorsAttnB block",
        "Suggested caption: Post hoc exact Schaefer-17 check of the beta-band AEC-orth block defined by SomMotA and DorsAttnB, motivated by visual inspection of the network heatmap. The left panel shows the 2x2 group-difference matrix (Converter minus Non-converter); the right panel reports cell-wise and block-wise effect estimates with 95% confidence intervals. The block mean is the arithmetic mean of the three unique cells in the block, and the block median is a narrow robustness variant over the same three cells. This analysis is descriptive and post hoc and does not belong to the confirmatory H1-H3 family.",
    )
    _upsert_caption(
        captions_tables,
        "## Table: Post hoc beta AEC-orth SomMotA-DorsAttnB block",
        "Suggested caption: Statistical summary of the post hoc exact Schaefer-17 beta-band AEC-orth block defined by SomMotA and DorsAttnB. Three unique network cells are reported individually, together with a block-level arithmetic mean and a block-level median over those same three values. Cell-wise Holm adjustment was applied only across the three individual cells, whereas block-level permutation p-values were computed for the mean and median summaries separately. This table is supplementary and exploratory and is not part of the primary H1-H3 confirmatory family.",
    )


if __name__ == "__main__":
    main()
