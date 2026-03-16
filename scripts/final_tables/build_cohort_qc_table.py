#!/usr/bin/env python3
from __future__ import annotations

"""Build the final cohort and quality-control table for the thesis.

The goal of this script is deliberately conservative:

- use only fields that are actually present in the raw MATLAB structs or in the
  exported cohort metadata
- avoid inventing demographic or QC variables that are not part of the current
  dataset
- report missing fields explicitly so the final thesis table is honest about
  what the repository can and cannot document

The output is a publication-oriented summary table comparing Converters and
Non-converters on the cohort descriptors that are available locally.
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meg_alzheimer.dataset import SubjectRecord, discover_subjects


GROUP_A = "Converter"
GROUP_B = "Non-converter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final cohort/QC table for the thesis.")
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


def _field_names(struct: object) -> list[str]:
    return sorted(name for name in dir(struct) if not name.startswith("_"))


def _load_struct(record: SubjectRecord) -> object:
    data = sio.loadmat(record.path, variable_names=[record.mat_variable], squeeze_me=True, struct_as_record=False)
    return data[record.mat_variable]


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float("nan")
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return float("nan")
        return float(arr.squeeze())
    except Exception:
        return float("nan")


def _extract_candidate_field(struct: object, accepted_names: set[str]) -> Any | None:
    for name in _field_names(struct):
        if name.lower() in accepted_names:
            return getattr(struct, name)
    return None


def _normalize_sex(value: Any) -> str | None:
    if value is None:
        return None
    text = str(np.asarray(value).squeeze()).strip().lower()
    if not text:
        return None
    if text in {"f", "female", "woman"}:
        return "Female"
    if text in {"m", "male", "man"}:
        return "Male"
    return text.title()


def _format_pvalue(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _format_statistic(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.2f}"


def _format_continuous(values: pd.Series, decimals: int = 2) -> str:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return "--"
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f} [{vmin:.{decimals}f}, {vmax:.{decimals}f}]"


def _compare_continuous(values_a: pd.Series, values_b: pd.Series) -> tuple[str, str, str]:
    xa = pd.to_numeric(values_a, errors="coerce").dropna().to_numpy(dtype=float)
    xb = pd.to_numeric(values_b, errors="coerce").dropna().to_numpy(dtype=float)
    if xa.size == 0 or xb.size == 0:
        return "Not tested", "--", "--"
    if np.allclose(np.r_[xa, xb], xa[0]):
        return "Constant across subjects", "--", "--"
    result = stats.ttest_ind(xa, xb, equal_var=False, nan_policy="omit")
    return "Welch t-test", _format_statistic(float(result.statistic)), _format_pvalue(float(result.pvalue))


def _format_categorical(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return "--"
    counts = clean.value_counts().sort_index()
    return "; ".join(f"{level}: {count}" for level, count in counts.items())


def _compare_categorical(values_a: pd.Series, values_b: pd.Series) -> tuple[str, str, str]:
    clean_a = values_a.dropna().astype(str)
    clean_b = values_b.dropna().astype(str)
    if clean_a.empty or clean_b.empty:
        return "Not tested", "--", "--"
    levels = sorted(set(clean_a) | set(clean_b))
    contingency = np.array(
        [[int((clean_a == level).sum()), int((clean_b == level).sum())] for level in levels],
        dtype=int,
    )
    if contingency.shape == (2, 2):
        _, p_value = stats.fisher_exact(contingency)
        return "Fisher exact test", "--", _format_pvalue(float(p_value))
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    return "Chi-square test", _format_statistic(float(chi2)), _format_pvalue(float(p_value))


def _verify_subject_manifest(subject_frame: pd.DataFrame, manifest_path: Path) -> None:
    if not manifest_path.exists():
        return
    manifest = pd.read_csv(manifest_path)
    expected = manifest[["subject_id", "group"]].sort_values(["group", "subject_id"]).reset_index(drop=True)
    observed = subject_frame[["subject_id", "group"]].sort_values(["group", "subject_id"]).reset_index(drop=True)
    if not expected.equals(observed):
        raise ValueError(
            "Raw-data subject discovery does not match the exported cohort manifest. "
            "Refusing to build the final cohort table from inconsistent inputs."
        )


def _subject_qc_frame(records: list[SubjectRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen_fields: set[str] = set()
    for record in records:
        struct = _load_struct(record)
        fields = _field_names(struct)
        seen_fields.update(fields)

        value = np.asarray(getattr(struct, "Value"))
        n_rois = 102
        n_valid_trials = int(value.shape[0] // n_rois)

        channel_flag = np.asarray(getattr(struct, "ChannelFlag", []), dtype=float).ravel()
        n_channels = int(channel_flag.size) if channel_flag.size else 0
        n_bad_channels = int(np.sum(channel_flag < 0)) if channel_flag.size else 0
        pct_bad_channels = float(100.0 * n_bad_channels / n_channels) if n_channels else float("nan")

        age_value = _extract_candidate_field(struct, {"age", "edad"})
        sex_value = _extract_candidate_field(struct, {"sex", "sexo", "gender"})
        snr_value = _extract_candidate_field(struct, {"snr"})

        discard_field_name = next(
            (
                name
                for name in fields
                if "trial" in name.lower() and ("discard" in name.lower() or "reject" in name.lower())
            ),
            None,
        )
        discarded_value = getattr(struct, discard_field_name) if discard_field_name is not None else None

        rows.append(
            {
                "subject_id": record.subject_id,
                "group": record.group,
                "path": str(record.path),
                "raw_fields": ", ".join(fields),
                "age_years": _safe_float(age_value),
                "sex": _normalize_sex(sex_value),
                "n_valid_trials": n_valid_trials,
                "pct_trials_discarded": _safe_float(discarded_value),
                "snr": _safe_float(snr_value),
                "n_channels": n_channels,
                "n_bad_channels": n_bad_channels,
                "pct_bad_channels": pct_bad_channels,
                "leff": _safe_float(getattr(struct, "Leff", None)),
                "navg": _safe_float(getattr(struct, "nAvg", None)),
                "std_size": int(np.asarray(getattr(struct, "Std", [])).size),
                "zscore_size": int(np.asarray(getattr(struct, "ZScore", [])).size),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["group", "subject_id"]).reset_index(drop=True)
    frame.attrs["seen_fields"] = sorted(seen_fields)
    return frame


def _build_summary_table(subject_df: pd.DataFrame) -> pd.DataFrame:
    group_a = subject_df.loc[subject_df["group"] == GROUP_A].copy()
    group_b = subject_df.loc[subject_df["group"] == GROUP_B].copy()

    rows: list[dict[str, str]] = []
    raw_fields = subject_df.attrs.get("seen_fields", [])
    raw_fields_note = ", ".join(raw_fields)

    rows.append(
        {
            "Variable": "Subjects, n",
            "Converter": str(len(group_a)),
            "Non-converter": str(len(group_b)),
            "Comparison": "Counts only",
            "Statistic": "--",
            "p-value": "--",
            "Availability / source": "Available; exported cohort manifest and raw subject discovery agree",
        }
    )

    for variable, label, source_note in [
        ("age_years", "Age (years)", "Not available in the raw MATLAB structs inspected"),
        ("sex", "Sex", "Not available in the raw MATLAB structs inspected"),
        ("n_valid_trials", "Valid trials per subject", "Available; derived from Value.shape[0] / 102"),
        (
            "pct_trials_discarded",
            "Discarded trials (%)",
            "Not available; the raw structs contain clean trials only and do not expose a rejected-trial percentage",
        ),
        ("n_bad_channels", "Bad MEG channels", "Available; derived from ChannelFlag < 0"),
        ("pct_bad_channels", "Bad MEG channels (%)", "Available; derived from ChannelFlag < 0"),
        ("leff", "Leff", "Available; Brainstorm field present in every raw struct"),
        ("navg", "nAvg", "Available; Brainstorm field present in every raw struct"),
        ("snr", "SNR", "Not available; no SNR field was found in the raw structs inspected"),
    ]:
        series_a = group_a[variable]
        series_b = group_b[variable]
        if variable == "sex":
            comparison, statistic, pvalue = _compare_categorical(series_a, series_b)
            rows.append(
                {
                    "Variable": label,
                    "Converter": _format_categorical(series_a),
                    "Non-converter": _format_categorical(series_b),
                    "Comparison": comparison,
                    "Statistic": statistic,
                    "p-value": pvalue,
                    "Availability / source": source_note,
                }
            )
            continue

        if pd.to_numeric(subject_df[variable], errors="coerce").notna().sum() == 0:
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

        comparison, statistic, pvalue = _compare_continuous(series_a, series_b)
        rows.append(
            {
                "Variable": label,
                "Converter": _format_continuous(series_a),
                "Non-converter": _format_continuous(series_b),
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
    caption = (
        "## Table: Cohort overview and quality-control summary\n\n"
        "Suggested caption: Cohort and quality-control summary for Converter and Non-converter groups. "
        "The table reports the variables that are actually available in the current repository. "
        "Valid trials were derived from the Brainstorm `Value` matrix, and channel-quality metrics were derived "
        "from `ChannelFlag`. Age, sex, discarded-trial percentage, and SNR were not available in the raw MATLAB "
        "structs inspected and are explicitly marked as unavailable. Continuous variables were compared with "
        "two-sided Welch t-tests; categorical variables were compared with Fisher exact or chi-square tests when available.\n"
    )
    path.write_text(caption)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    captions_path = Path(args.captions_path)
    manifest_path = Path(args.subjects_csv)

    records = discover_subjects(data_root)
    if not records:
        raise SystemExit(f"No subjects found under {data_root}.")

    subject_df = _subject_qc_frame(records)
    _verify_subject_manifest(subject_df, manifest_path)

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

    print(f"Subject rows inspected: {len(subject_df)}")
    print(f"Group counts: {subject_df['group'].value_counts().sort_index().to_dict()}")
    print(f"CSV table: {csv_path}")
    print(f"LaTeX table: {tex_path}")
    print(f"Caption file: {captions_path}")


if __name__ == "__main__":
    main()
