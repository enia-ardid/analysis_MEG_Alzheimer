from __future__ import annotations

"""Run the methodological validation report for the current cohort outputs."""

import argparse
import json

from meg_alzheimer.methodological_validation import generate_methodological_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the methodological validation report for the current "
            "MEG connectivity cohort."
        )
    )
    parser.add_argument(
        "--output-root",
        default="outputs_full_cohort",
        help="Directory containing subject_network_means.csv and subject outputs.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Directory containing the Brainstorm-like ROI time-series exports.",
    )
    parser.add_argument(
        "--report-root",
        default=None,
        help=(
            "Optional destination for the report. Defaults to "
            "<output-root>/methodological_validation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_methodological_validation_report(
        output_root=args.output_root,
        data_root=args.data_root,
        report_root=args.report_root,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
