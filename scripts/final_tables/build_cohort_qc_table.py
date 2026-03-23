#!/usr/bin/env python3
from __future__ import annotations

"""Build the manuscript QC table from the raw Brainstorm exports.

This table is intentionally conservative. It only reports variables that can
be recovered directly from the raw MATLAB structs or from the cohort manifest
produced by the main pipeline. Anything unavailable in the accessible files is
left explicit in the output instead of being inferred.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meg_alzheimer.dataset import discover_subjects
from meg_alzheimer.qc import (
    build_subject_qc_frame,
    compare_categorical,
    compare_continuous,
    field_names,
    format_categorical,
    format_continuous,
    load_subject_struct,
    verify_subject_manifest,
)


GROUP_A = "Converter"
GROUP_B = "Non-converter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the cohort and quality-control summary table.")
    parser.add_argument("--data-root", default="data", help="Folder containing the raw MATLAB files.")
    parser.add_argument(
        "--subjects-csv",
        default="outputs_full_cohort/subjects.csv",
        help="Optional cohort manifest used to verify subject IDs and groups.",
    )
    parser.add_argument("--output-dir", default="tables/final", help="Destination folder for CSV and LaTeX outputs.")
    parser.add_argument(
        "--captions-path",
        default="captions_tables.md",
        help="Markdown file where the suggested caption will be written.",
    )
    return parser.parse_args()


def _build_summary_table(subject_df: pd.DataFrame) -> pd.DataFrame:
    group_a = subject_df.loc[subject_df["group"] == GROUP_A].copy()
    group_b = subject_df.loc[subject_df["group"] == GROUP_B].copy()

    rows: list[dict[str, str]] = [
        {
            "Variable": "Subjects, n",
            "Converter": str(len(group_a)),
            "Non-converter": str(len(group_b)),
            "Comparison": "Counts only",
            "Statistic": "--",
            "p-value": "--",
            "Availability / source": "Available; exported cohort manifest and raw subject discovery agree",
        }
    ]

    raw_fields_note = ", ".join(subject_df.attrs.get("seen_fields", []))
    variable_specs = [
        ("age_years", "Age (years)", "Not available in the raw MATLAB structs inspected", "continuous"),
        ("sex", "Sex", "Not available in the raw MATLAB structs inspected", "categorical"),
        ("n_valid_trials", "Valid trials per subject", "Available; derived from Value.shape[0] / 102", "continuous"),
        (
            "pct_trials_discarded",
            "Discarded trials (%)",
            "Not available; the raw structs contain clean trials only and do not expose a rejected-trial percentage",
            "continuous",
        ),
        ("n_bad_channels", "Bad MEG channels", "Available; derived from ChannelFlag < 0", "continuous"),
        ("pct_bad_channels", "Bad MEG channels (%)", "Available; derived from ChannelFlag < 0", "continuous"),
        ("leff", "Leff", "Available; Brainstorm field present in every raw struct", "continuous"),
        ("navg", "nAvg", "Available; Brainstorm field present in every raw struct", "continuous"),
        ("snr", "SNR", "Not available; no SNR field was found in the raw structs inspected", "continuous"),
    ]

    for column, label, source_note, kind in variable_specs:
        series_a = group_a[column]
        series_b = group_b[column]

        if kind == "categorical":
            comparison, statistic, pvalue = compare_categorical(series_a, series_b)
            rows.append(
                {
                    "Variable": label,
                    "Converter": format_categorical(series_a),
                    "Non-converter": format_categorical(series_b),
                    "Comparison": comparison,
                    "Statistic": statistic,
                    "p-value": pvalue,
                    "Availability / source": source_note,
                }
            )
            continue

        if pd.to_numeric(subject_df[column], errors="coerce").notna().sum() == 0:
            rows.append(
                {
                    "Variable": label,
                    "Converter": "--",
                    "Non-converter": "--",
                    "Comparison": "Not tested",
                    "Statistic": "--",
                    "p-value": "--",
                    "Availability / source": source_note,
                }
            )
            continue

        comparison, statistic, pvalue = compare_continuous(series_a, series_b)
        rows.append(
            {
                "Variable": label,
                "Converter": format_continuous(series_a),
                "Non-converter": format_continuous(series_b),
                "Comparison": comparison,
                "Statistic": statistic,
                "p-value": pvalue,
                "Availability / source": source_note,
            }
        )

    rows.append(
        {
            "Variable": "Raw struct fields inspected",
            "Converter": "--",
            "Non-converter": "--",
            "Comparison": "Documentation only",
            "Statistic": "--",
            "p-value": "--",
            "Availability / source": raw_fields_note,
        }
    )
    return pd.DataFrame(rows)


def _write_latex(table_df: pd.DataFrame, path: Path) -> None:
    def escape_latex(text: Any) -> str:
        value = "--" if text is None or (isinstance(text, float) and not np.isfinite(text)) else str(text)
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
            value = value.replace(src, dst)
        return value

    headers = list(table_df.columns)
    lines = [
        r"\begin{tabular}{p{3.0cm}p{3.2cm}p{3.2cm}p{2.6cm}p{1.5cm}p{1.5cm}p{5.2cm}}",
        r"\toprule",
        " & ".join(escape_latex(col) for col in headers) + r" \\",
        r"\midrule",
    ]
    for row in table_df.itertuples(index=False):
        lines.append(" & ".join(escape_latex(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))


def _write_caption(path: Path) -> None:
    path.write_text(
        "## Table: Cohort overview and quality-control summary\n\n"
        "Suggested caption: Cohort and quality-control summary for Converter and Non-converter groups. "
        "The table reports the variables that are actually available in the current repository. "
        "Valid trials were derived from the Brainstorm `Value` matrix, and channel-quality metrics were derived "
        "from `ChannelFlag`. Age, sex, discarded-trial percentage, and SNR were not available in the raw MATLAB "
        "structs inspected and are explicitly marked as unavailable. Continuous variables were compared with "
        "two-sided Welch t-tests; categorical variables were compared with Fisher exact or chi-square tests when available.\n"
    )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    captions_path = Path(args.captions_path)
    manifest_path = Path(args.subjects_csv)

    records = discover_subjects(data_root)
    if not records:
        raise SystemExit(f"No subjects found under {data_root}.")

    subject_df = build_subject_qc_frame(records)
    verify_subject_manifest(subject_df, manifest_path)

    if set(subject_df["group"]) != {GROUP_A, GROUP_B}:
        raise SystemExit(
            f"Unexpected group labels found: {sorted(subject_df['group'].unique())}. "
            f"Expected exactly {GROUP_A!r} and {GROUP_B!r}."
        )

    summary_df = _build_summary_table(subject_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "table_cohort_qc.csv"
    tex_path = output_dir / "table_cohort_qc.tex"
    summary_df.to_csv(csv_path, index=False)
    _write_latex(summary_df, tex_path)
    _write_caption(captions_path)

    raw_fields = field_names(load_subject_struct(records[0]))
    print(f"Subject rows inspected: {len(subject_df)}")
    print(f"Group counts: {subject_df['group'].value_counts().sort_index().to_dict()}")
    print(f"Reference raw fields: {raw_fields}")
    print(f"CSV table: {csv_path}")
    print(f"LaTeX table: {tex_path}")
    print(f"Caption file: {captions_path}")


if __name__ == "__main__":
    main()
