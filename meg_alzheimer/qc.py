from __future__ import annotations

"""Shared cohort and quality-control utilities.

The paper uses a small set of subject-level descriptors more than once:

- the number of valid trials per subject
- a conservative table of raw-file QC fields
- simple group comparisons for those descriptors

Keeping that logic in one module avoids cross-imports between figure scripts
and table scripts and makes the manuscript outputs easier to audit.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

from .dataset import SubjectRecord


def field_names(struct: object) -> list[str]:
    return sorted(name for name in dir(struct) if not name.startswith("_"))


def load_subject_struct(record: SubjectRecord) -> object:
    data = sio.loadmat(
        record.path,
        variable_names=[record.mat_variable],
        squeeze_me=True,
        struct_as_record=False,
    )
    return data[record.mat_variable]


def safe_float(value: Any) -> float:
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


def extract_candidate_field(struct: object, accepted_names: set[str]) -> Any | None:
    for name in field_names(struct):
        if name.lower() in accepted_names:
            return getattr(struct, name)
    return None


def normalize_sex(value: Any) -> str | None:
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


def format_pvalue(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def format_statistic(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.2f}"


def format_continuous(values: pd.Series, decimals: int = 2) -> str:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return "--"
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    return f"{mean:.{decimals}f} +- {sd:.{decimals}f} [{vmin:.{decimals}f}, {vmax:.{decimals}f}]"


def compare_continuous(values_a: pd.Series, values_b: pd.Series) -> tuple[str, str, str]:
    xa = pd.to_numeric(values_a, errors="coerce").dropna().to_numpy(dtype=float)
    xb = pd.to_numeric(values_b, errors="coerce").dropna().to_numpy(dtype=float)
    if xa.size == 0 or xb.size == 0:
        return "Not tested", "--", "--"
    if np.allclose(np.r_[xa, xb], xa[0]):
        return "Constant across subjects", "--", "--"
    result = stats.ttest_ind(xa, xb, equal_var=False, nan_policy="omit")
    return "Welch t-test", format_statistic(float(result.statistic)), format_pvalue(float(result.pvalue))


def format_categorical(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return "--"
    counts = clean.value_counts().sort_index()
    return "; ".join(f"{level}: {count}" for level, count in counts.items())


def compare_categorical(values_a: pd.Series, values_b: pd.Series) -> tuple[str, str, str]:
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
        return "Fisher exact test", "--", format_pvalue(float(p_value))
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    return "Chi-square test", format_statistic(float(chi2)), format_pvalue(float(p_value))


def verify_subject_manifest(subject_frame: pd.DataFrame, manifest_path: Path) -> None:
    if not manifest_path.exists():
        return
    manifest = pd.read_csv(manifest_path)
    expected = manifest[["subject_id", "group"]].sort_values(["group", "subject_id"]).reset_index(drop=True)
    observed = subject_frame[["subject_id", "group"]].sort_values(["group", "subject_id"]).reset_index(drop=True)
    if not expected.equals(observed):
        raise ValueError(
            "Raw-data subject discovery does not match the exported cohort manifest. "
            "Refusing to build manuscript QC outputs from inconsistent inputs."
        )


def build_subject_qc_frame(records: list[SubjectRecord], *, n_rois: int = 102) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen_fields: set[str] = set()
    for record in records:
        struct = load_subject_struct(record)
        fields = field_names(struct)
        seen_fields.update(fields)

        value = np.asarray(getattr(struct, "Value"))
        n_valid_trials = int(value.shape[0] // n_rois)

        channel_flag = np.asarray(getattr(struct, "ChannelFlag", []), dtype=float).ravel()
        n_channels = int(channel_flag.size) if channel_flag.size else 0
        n_bad_channels = int(np.sum(channel_flag < 0)) if channel_flag.size else 0
        pct_bad_channels = float(100.0 * n_bad_channels / n_channels) if n_channels else float("nan")

        age_value = extract_candidate_field(struct, {"age", "edad"})
        sex_value = extract_candidate_field(struct, {"sex", "sexo", "gender"})
        snr_value = extract_candidate_field(struct, {"snr"})

        discarded_field = next(
            (
                name
                for name in fields
                if "trial" in name.lower() and ("discard" in name.lower() or "reject" in name.lower())
            ),
            None,
        )
        discarded_value = getattr(struct, discarded_field) if discarded_field is not None else None

        rows.append(
            {
                "subject_id": record.subject_id,
                "group": record.group,
                "path": str(record.path),
                "raw_fields": ", ".join(fields),
                "age_years": safe_float(age_value),
                "sex": normalize_sex(sex_value),
                "n_valid_trials": n_valid_trials,
                "pct_trials_discarded": safe_float(discarded_value),
                "snr": safe_float(snr_value),
                "n_channels": n_channels,
                "n_bad_channels": n_bad_channels,
                "pct_bad_channels": pct_bad_channels,
                "leff": safe_float(getattr(struct, "Leff", None)),
                "navg": safe_float(getattr(struct, "nAvg", None)),
                "std_size": int(np.asarray(getattr(struct, "Std", [])).size),
                "zscore_size": int(np.asarray(getattr(struct, "ZScore", [])).size),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["group", "subject_id"]).reset_index(drop=True)
    frame.attrs["seen_fields"] = sorted(seen_fields)
    return frame
