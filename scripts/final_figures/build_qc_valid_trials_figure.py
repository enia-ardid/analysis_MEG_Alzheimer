#!/usr/bin/env python3
from __future__ import annotations

"""Build the final QC figure for the number of valid trials per subject.

The figure intentionally reuses the same subject-level QC extraction implemented
for the cohort/QC table so that the definition of "valid trial" is identical in
both outputs. No new trial metric is introduced here.
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
from matplotlib.ticker import MaxNLocator

from scripts.final_tables.build_cohort_qc_table import (
    GROUP_A,
    GROUP_B,
    _subject_qc_frame,
    _verify_subject_manifest,
    discover_subjects,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the valid-trials QC figure.")
    parser.add_argument("--data-root", default="data", help="Folder containing the raw MATLAB files.")
    parser.add_argument(
        "--subjects-csv",
        default="outputs_full_cohort/subjects.csv",
        help="Optional cohort manifest used to verify subject IDs and groups.",
    )
    parser.add_argument("--output-dir", default="figures/final", help="Destination folder for the figure files.")
    parser.add_argument(
        "--captions-path",
        default="captions_figures.md",
        help="Markdown file where the suggested caption will be written.",
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


def _append_caption(path: Path) -> None:
    block = (
        "## Figure: Valid trials per subject\n\n"
        "Suggested caption: Number of valid trials per subject in the Converter and Non-converter groups. "
        "Each point corresponds to one subject, and the box summarizes the group distribution. "
        "Valid trials were defined exactly as in the cohort/QC table, i.e. as the number of clean 8-second source-space "
        "segments available in the Brainstorm `Value` matrix before within-subject averaging.\n"
    )
    if path.exists() and path.read_text().strip():
        existing = path.read_text().rstrip() + "\n\n"
    else:
        existing = ""
    path.write_text(existing + block)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    captions_path = Path(args.captions_path)

    records = discover_subjects(args.data_root)
    if not records:
        raise SystemExit(f"No subjects found under {args.data_root}.")

    subject_df = _subject_qc_frame(records)
    _verify_subject_manifest(subject_df, Path(args.subjects_csv))

    groups = [GROUP_A, GROUP_B]
    values = [
        subject_df.loc[subject_df["group"] == group, "n_valid_trials"].to_numpy(dtype=float)
        for group in groups
    ]
    counts = [int(len(v)) for v in values]

    output_dir.mkdir(parents=True, exist_ok=True)
    _style()

    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    box = ax.boxplot(
        values,
        positions=[1, 2],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
    )
    for patch in box["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("#111111")
        patch.set_linewidth(1.2)
    for artist_name in ("whiskers", "caps", "medians"):
        for artist in box[artist_name]:
            artist.set_color("#111111")
            artist.set_linewidth(1.1)

    rng = np.random.default_rng(0)
    for xpos, vals in zip([1, 2], values):
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(vals))
        ax.scatter(
            np.full(len(vals), xpos, dtype=float) + jitter,
            vals,
            s=18,
            facecolor="#4a4a4a",
            edgecolor="#111111",
            linewidth=0.3,
            alpha=0.80,
            zorder=3,
        )

    ax.set_xticks([1, 2], [f"{GROUP_A}\n(n={counts[0]})", f"{GROUP_B}\n(n={counts[1]})"])
    ax.set_ylabel("Valid trials per subject")
    ax.set_xlabel("Group")
    ax.set_title("Quality control: valid trials per subject")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    png_path = output_dir / "fig_qc_valid_trials.png"
    pdf_path = output_dir / "fig_qc_valid_trials.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    _append_caption(captions_path)

    print(f"Input: raw MATLAB files under {args.data_root} + cohort manifest {args.subjects_csv}")
    print("Definition: n_valid_trials = Value.shape[0] / 102")
    print(f"PNG figure: {png_path}")
    print(f"PDF figure: {pdf_path}")
    print(f"Caption file: {captions_path}")


if __name__ == "__main__":
    main()
