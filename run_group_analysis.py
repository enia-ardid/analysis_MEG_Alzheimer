from __future__ import annotations

"""Command-line entry point for the full raw-data-to-hypothesis workflow."""

import argparse
from pathlib import Path

from meg_alzheimer.pipeline import PipelineConfig, run_group_pipeline
from meg_alzheimer.strong_hypotheses import generate_strong_hypothesis_report


def parse_args() -> argparse.Namespace:
    """Define the CLI for the cohort pipeline.

    The defaults are chosen so that running the command with no extra flags is
    enough for the current project layout:

    - raw files under ``data/``
    - outputs written to ``outputs/`` unless overridden
    - strong H1-H3 testing executed automatically at the end
    """
    parser = argparse.ArgumentParser(
        description="Run multi-subject MEG connectivity analysis for Converter vs Non-converter subjects."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root folder containing the split C_p*.mat and NC_p*.mat files.",
    )
    parser.add_argument("--output-root", default="outputs", help="Directory where results will be written.")
    parser.add_argument("--sampling-rate", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--crop-start", type=int, default=2000, help="First sample kept from each trial.")
    parser.add_argument("--crop-end", type=int, default=6000, help="Last sample kept from each trial.")
    parser.add_argument("--window-s", type=float, default=None, help="Optional sliding-window size in seconds.")
    parser.add_argument("--step-s", type=float, default=None, help="Optional sliding-window step in seconds.")
    parser.add_argument("--edge-trim", type=int, default=None, help="Optional number of filtered edge samples to trim.")
    parser.add_argument("--fdr-q", type=float, default=0.10, help="Benjamini-Hochberg false discovery rate.")
    parser.add_argument("--n-perm", type=int, default=1000, help="Permutation count for max-stat corrected edge tests.")
    parser.add_argument("--perm-seed", type=int, default=0, help="Random seed for permutation testing.")
    parser.add_argument("--group-a", default="Converter", help="First comparison group name used in outputs and contrasts.")
    parser.add_argument(
        "--group-b",
        default="Non-converter",
        help="Second comparison group name used in outputs and contrasts.",
    )
    parser.add_argument(
        "--quicklook-only",
        action="store_true",
        help="Save only the quicklook matrix per subject instead of exporting all band x metric matrix plots.",
    )
    parser.add_argument(
        "--no-subject-graphs",
        action="store_true",
        help="Disable subject-level thresholded graph exports.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-file and per-subject progress logging.",
    )
    parser.add_argument(
        "--skip-hypotheses",
        action="store_true",
        help="Skip the strong H1-H3 report at the end of the cohort analysis.",
    )
    parser.add_argument(
        "--hypothesis-n-perm",
        type=int,
        default=50000,
        help="Permutation count for the strong H1-H3 max-T correction.",
    )
    parser.add_argument(
        "--hypothesis-n-boot",
        type=int,
        default=10000,
        help="Bootstrap count for the strong H1-H3 confidence intervals.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the cohort pipeline and, unless disabled, the final H1-H3 step."""
    args = parse_args()
    config = PipelineConfig(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        sampling_rate=args.sampling_rate,
        crop_start=args.crop_start,
        crop_end=args.crop_end,
        window_s=args.window_s,
        step_s=args.step_s,
        edge_trim=args.edge_trim,
        fdr_q=args.fdr_q,
        n_perm=args.n_perm,
        perm_seed=args.perm_seed,
        save_all_subject_matrix_plots=not args.quicklook_only,
        save_subject_graphs=not args.no_subject_graphs,
        group_a=args.group_a,
        group_b=args.group_b,
        verbose=not args.quiet,
    )
    result = run_group_pipeline(config)
    if not result.get("subjects"):
        print(result["message"])
    else:
        print(f"Processed {result['n_subjects']} subjects into {result['output_root']}.")
        print(f"Groups: {', '.join(result['groups'])}")
        print(f"Counts: {result['group_counts']}")
        if not args.skip_hypotheses:
            # The final strong-testing stage reads the cohort outputs instead of
            # recomputing subject matrices. This keeps the two stages clearly
            # separated while still allowing a one-command workflow.
            strong_result = generate_strong_hypothesis_report(
                output_root=config.output_root,
                group_a=config.group_a,
                group_b=config.group_b,
                n_perm=args.hypothesis_n_perm,
                n_boot=args.hypothesis_n_boot,
                seed=args.perm_seed,
            )
            print(f"Strong H1-H3 report: {strong_result['report_root']}")
            print(f"Hypothesis summary: {strong_result['hypothesis_summary']}")


if __name__ == "__main__":
    main()
