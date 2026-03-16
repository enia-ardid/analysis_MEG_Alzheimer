#!/usr/bin/env python3
from __future__ import annotations

"""Build the main cohort table used in the manuscript.

Unlike the broader QC table, this output is intentionally minimal and keeps
only the variables that are both available and useful for the main text:

- group size
- number of valid trials per subject

Age and sex are not included as subject-level rows because those values are not
available in the accessible project files. The cohort design note about
matching is carried in the suggested table caption instead.
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

import numpy as np
import pandas as pd

from scripts.final_tables.build_cohort_qc_table import (
    GROUP_A,
    GROUP_B,
    _compare_continuous,
    _format_continuous,
    _subject_qc_frame,
    _verify_subject_manifest,
    discover_subjects,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the main cohort table for the manuscript.")
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
        r"\begin{tabular}{p{4.2cm}p{3.2cm}p{3.2cm}p{2.4cm}p{1.6cm}p{1.6cm}}",
        r"\toprule",
        " & ".join(escape_latex(col) for col in headers) + r" \\",
        r"\midrule",
    ]
    for row in table_df.itertuples(index=False):
        lines.append(" & ".join(escape_latex(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))


def _upsert_caption(path: Path) -> None:
    heading = "## Table: Main cohort summary"
    block = (
        f"{heading}\n\n"
        "Suggested caption: Main cohort summary for the Converter and Non-converter groups. "
        "The table reports only the variables available and retained for the manuscript-level cohort description: "
        "group size and the number of valid trials per subject. According to the cohort design information provided "
        "by the supervisor, the two groups were matched by age and sex; however, individual age and sex values were "
        "not available in the raw files analyzed in this repository and therefore could not be tabulated here. "
        "Valid trials were derived from the Brainstorm `Value` matrix as `Value.shape[0] / 102`, and groups were "
        "compared with a two-sided Welch t-test.\n"
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


def _build_main_table(subject_df: pd.DataFrame) -> pd.DataFrame:
    group_a = subject_df.loc[subject_df["group"] == GROUP_A].copy()
    group_b = subject_df.loc[subject_df["group"] == GROUP_B].copy()

    comparison, statistic, p_value = _compare_continuous(group_a["n_valid_trials"], group_b["n_valid_trials"])

    rows = [
        {
            "Variable": "Subjects, n",
            "Converter": str(len(group_a)),
            "Non-converter": str(len(group_b)),
            "Comparison": "Counts only",
            "Statistic": "--",
            "p-value": "--",
        },
        {
            "Variable": "Valid trials per subject",
            "Converter": _format_continuous(group_a["n_valid_trials"]),
            "Non-converter": _format_continuous(group_b["n_valid_trials"]),
            "Comparison": comparison,
            "Statistic": statistic,
            "p-value": p_value,
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    captions_path = Path(args.captions_path)

    records = discover_subjects(args.data_root)
    if not records:
        raise SystemExit(f"No subjects found under {args.data_root}.")

    subject_df = _subject_qc_frame(records)
    _verify_subject_manifest(subject_df, Path(args.subjects_csv))

    groups = set(subject_df["group"])
    if groups != {GROUP_A, GROUP_B}:
        raise SystemExit(
            f"Unexpected group labels found: {sorted(groups)}. Expected exactly {GROUP_A!r} and {GROUP_B!r}."
        )

    table_df = _build_main_table(subject_df)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "table_cohort_main.csv"
    tex_path = output_dir / "table_cohort_main.tex"
    table_df.to_csv(csv_path, index=False)
    _write_latex(table_df, tex_path)
    _upsert_caption(captions_path)

    print(f"Rows written: {len(table_df)}")
    print(f"CSV table: {csv_path}")
    print(f"LaTeX table: {tex_path}")
    print(f"Caption file: {captions_path}")


if __name__ == "__main__":
    main()
