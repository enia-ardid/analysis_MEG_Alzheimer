from __future__ import annotations

"""Run the exact figure-generation path used by the paper.

The goal of this entry point is not to redefine the analysis. It simply
serializes the already established workflow into one command:

1. build the cohort outputs from raw Brainstorm `.mat` files
2. refresh the strong H1-H3 report
3. regenerate the seven figures used in the paper

The figure list is fixed on purpose so the manuscript-facing outputs remain
stable and easy to audit.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


FIGURE_SCRIPTS = [
    ("scripts/final_figures/build_pipeline_overview.py", []),
    ("scripts/final_figures/build_qc_valid_trials_figure.py", []),
    ("scripts/final_figures/build_network_heatmaps.py", []),
    ("scripts/final_figures/build_composite_breakdown.py", []),
    ("scripts/final_figures/build_endpoints_main_figure.py", []),
    ("scripts/final_figures/build_trials_threshold_sensitivity.py", []),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the raw-data-to-paper-figures workflow.")
    parser.add_argument("--data-root", default="data", help="Folder containing the raw Brainstorm MATLAB exports.")
    parser.add_argument(
        "--output-root",
        default="outputs_full_cohort",
        help="Directory where cohort-level outputs and strong-hypothesis outputs are written.",
    )
    parser.add_argument("--figure-dir", default="figures/final", help="Destination folder for paper figures.")
    parser.add_argument("--table-dir", default="tables/final", help="Destination folder for manuscript tables.")
    parser.add_argument(
        "--captions-figures",
        default="captions_figures.md",
        help="Markdown file where figure caption drafts are written.",
    )
    parser.add_argument(
        "--captions-tables",
        default="captions_tables.md",
        help="Markdown file where table caption drafts are written.",
    )
    parser.add_argument(
        "--skip-group-analysis",
        action="store_true",
        help="Reuse an existing output-root instead of recomputing the cohort outputs from raw data.",
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True, env=os.environ.copy())


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    os.environ.setdefault("MPLCONFIGDIR", str(repo_root / ".mplconfig"))
    python_executable = sys.executable
    output_root = Path(args.output_root)

    if not args.skip_group_analysis:
        _run(
            [
                python_executable,
                str(repo_root / "run_group_analysis.py"),
                "--data-root",
                args.data_root,
                "--output-root",
                str(output_root),
            ]
        )
    else:
        _run(
            [
                python_executable,
                str(repo_root / "run_strong_hypotheses.py"),
                "--output-root",
                str(output_root),
            ]
        )

    subjects_csv = output_root / "subjects.csv"
    subjects_root = output_root / "subjects"

    script_args = {
        "scripts/final_figures/build_pipeline_overview.py": [
            "--output-dir",
            args.figure_dir,
            "--captions-path",
            args.captions_figures,
        ],
        "scripts/final_figures/build_qc_valid_trials_figure.py": [
            "--data-root",
            args.data_root,
            "--subjects-csv",
            str(subjects_csv),
            "--output-dir",
            args.figure_dir,
            "--captions-path",
            args.captions_figures,
        ],
        "scripts/final_figures/build_network_heatmaps.py": [
            "--subjects-root",
            str(subjects_root),
            "--output-dir",
            args.figure_dir,
            "--captions-path",
            args.captions_figures,
        ],
        "scripts/final_figures/build_composite_breakdown.py": [
            "--output-root",
            str(output_root),
            "--figure-dir",
            args.figure_dir,
            "--table-dir",
            args.table_dir,
            "--captions-figures",
            args.captions_figures,
            "--captions-tables",
            args.captions_tables,
        ],
        "scripts/final_figures/build_endpoints_main_figure.py": [
            "--output-root",
            str(output_root),
            "--figure-dir",
            args.figure_dir,
            "--table-dir",
            args.table_dir,
            "--captions-figures",
            args.captions_figures,
            "--captions-tables",
            args.captions_tables,
        ],
        "scripts/final_figures/build_trials_threshold_sensitivity.py": [
            "--output-root",
            str(output_root),
            "--figure-path",
            str(Path(args.figure_dir) / "fig_sensitivity_trials.png"),
            "--captions-path",
            args.captions_figures,
        ],
    }

    for script_path, _ in FIGURE_SCRIPTS:
        _run([python_executable, str(repo_root / script_path), *script_args[script_path]])


if __name__ == "__main__":
    main()
