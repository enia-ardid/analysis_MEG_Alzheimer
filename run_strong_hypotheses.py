from __future__ import annotations

"""Command-line entry point for the final H1-H3 confirmation step."""

import argparse
from pathlib import Path

from meg_alzheimer.strong_hypotheses import generate_strong_hypothesis_report


def parse_args() -> argparse.Namespace:
    """Define the CLI for rerunning the strong hypothesis test only."""
    parser = argparse.ArgumentParser(
        description="Run strong family-wise testing for hypotheses H1-H3 from an existing output folder."
    )
    parser.add_argument("--output-root", default="outputs_full_cohort", help="Existing output folder.")
    parser.add_argument(
        "--report-root",
        default=None,
        help="Optional destination folder. Defaults to <output-root>/strong_hypotheses.",
    )
    parser.add_argument("--group-a", default="Converter", help="First comparison group label.")
    parser.add_argument("--group-b", default="Non-converter", help="Second comparison group label.")
    parser.add_argument("--n-perm", type=int, default=50000, help="Permutation count for max-T correction.")
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap count for confidence intervals.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    """Run the strong-testing layer from existing cohort outputs."""
    args = parse_args()
    result = generate_strong_hypothesis_report(
        output_root=Path(args.output_root),
        report_root=Path(args.report_root) if args.report_root else None,
        group_a=args.group_a,
        group_b=args.group_b,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(f"Strong H1-H3 report written to {result['report_root']}.")
    print(f"Endpoint tests: {result['endpoint_tests']}")
    print(f"Hypothesis summary: {result['hypothesis_summary']}")
    print(f"Summary: {result['summary']}")


if __name__ == "__main__":
    main()
