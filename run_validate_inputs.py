from __future__ import annotations

"""Validate whether a local ROI time-series dataset matches the paper contract.

This command does not run the scientific analysis. Its purpose is narrower:

- discover Brainstorm-style subject structs under ``data/``
- check that each subject can be reshaped into ``trials x ROI x time``
- confirm that ROI labels are consistent across subjects
- optionally verify that the atlas labels are compatible with the paper

The validator is useful both before a full rerun of this repository and when a
different lab wants to reuse the code with source-space ROI exports in the same
format.
"""

import argparse
from collections import Counter
from pathlib import Path

from meg_alzheimer.atlas import DEFAULT_NETWORK_MAP, get_network_prefix
from meg_alzheimer.dataset import discover_subjects, load_brainstorm_subject


def parse_args() -> argparse.Namespace:
    """Define the CLI for dataset validation."""
    parser = argparse.ArgumentParser(
        description="Validate Brainstorm ROI time-series exports before running the pipeline."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Folder containing Brainstorm-style MATLAB exports.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional cap for a quick smoke validation on the first N discovered subjects.",
    )
    parser.add_argument(
        "--require-paper-atlas",
        action="store_true",
        help=(
            "Fail if ROI labels do not use the Schaefer-style prefixes expected by the "
            "paper network mapping."
        ),
    )
    return parser.parse_args()


def _paper_compatible_prefixes() -> set[str]:
    """Return the label prefixes accepted by the paper-level network mapping."""
    return set(DEFAULT_NETWORK_MAP) | {
        "Control",
        "Default",
        "DorsAttn",
        "Limbic",
        "SalVentAttn",
        "SomMot",
        "TempPar",
        "VisCent",
        "VisPeri",
        "Background",
    }


def main() -> None:
    """Validate a dataset and print an audit-style summary."""
    args = parse_args()
    data_root = Path(args.data_root)
    records = discover_subjects(data_root)
    if not records:
        raise SystemExit(f"No supported subject structs were found under {data_root}.")

    if args.max_subjects is not None:
        records = records[: args.max_subjects]

    loaded_subjects = []
    errors: list[str] = []
    file_count = len({record.path for record in records})
    group_counts = Counter(record.group for record in records)

    reference_labels: list[str] | None = None
    reference_n_time: int | None = None
    reference_n_rois: int | None = None
    reference_atlas: str | None = None
    all_prefixes: Counter[str] = Counter()

    for record in records:
        try:
            loaded = load_brainstorm_subject(record)
        except Exception as exc:  # pragma: no cover - surfaced to CLI callers
            errors.append(f"{record.subject_id}: failed to load ({exc})")
            continue

        loaded_subjects.append(loaded)
        if loaded.n_time != len(loaded.t):
            errors.append(
                f"{record.subject_id}: Time length {len(loaded.t)} does not match Value time axis {loaded.n_time}."
            )

        if reference_labels is None:
            reference_labels = loaded.roi_labels
            reference_n_time = loaded.n_time
            reference_n_rois = loaded.n_rois
            reference_atlas = loaded.atlas_name
        else:
            if loaded.roi_labels != reference_labels:
                errors.append(f"{record.subject_id}: ROI label order differs from the first loaded subject.")
            if loaded.n_time != reference_n_time:
                errors.append(
                    f"{record.subject_id}: time axis length {loaded.n_time} differs from reference {reference_n_time}."
                )
            if loaded.n_rois != reference_n_rois:
                errors.append(f"{record.subject_id}: ROI count {loaded.n_rois} differs from reference {reference_n_rois}.")
            if loaded.atlas_name != reference_atlas:
                errors.append(
                    f"{record.subject_id}: atlas name '{loaded.atlas_name}' differs from reference '{reference_atlas}'."
                )

        all_prefixes.update(get_network_prefix(label) for label in loaded.roi_labels)

    print("INPUT VALIDATION SUMMARY")
    print(f"data_root: {data_root}")
    print(f"mat_files: {file_count}")
    print(f"subjects_checked: {len(records)}")
    print(f"group_counts: {dict(group_counts)}")

    if loaded_subjects:
        n_trials = [subject.n_trials for subject in loaded_subjects]
        n_rois = loaded_subjects[0].n_rois
        n_time = loaded_subjects[0].n_time
        atlas_name = loaded_subjects[0].atlas_name
        print(f"atlas_name: {atlas_name}")
        print(f"roi_count: {n_rois}")
        print(f"timepoints_per_trial: {n_time}")
        print(f"trial_count_range: {min(n_trials)}-{max(n_trials)}")
        print(f"roi_prefixes: {', '.join(sorted(all_prefixes))}")

    if args.require_paper_atlas and loaded_subjects:
        unsupported = sorted(prefix for prefix in all_prefixes if prefix not in _paper_compatible_prefixes())
        if unsupported:
            errors.append(
                "Unsupported ROI prefixes for exact paper reproduction: "
                + ", ".join(unsupported)
                + ". Adapt meg_alzheimer/atlas.py and meg_alzheimer/strong_hypotheses.py "
                + "before using a different atlas."
            )

    if errors:
        print("\nVALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDATION PASSED")
    if args.max_subjects is not None:
        print("Note: this was a smoke validation on a subject subset.")
    print("The dataset is structurally compatible with the public pipeline.")


if __name__ == "__main__":
    main()
