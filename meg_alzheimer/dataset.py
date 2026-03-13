from __future__ import annotations

"""Data-loading helpers for Brainstorm ``.mat`` exports.

The repository works from multi-subject ``.mat`` files where each top-level
struct corresponds to one participant. The functions below solve three concrete
problems:

- discover which subject structs exist inside each file
- infer the study group from the subject identifier
- reshape the Brainstorm ``Value`` matrix into ``trials x rois x time``
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping

import numpy as np
import scipy.io as sio

from .atlas import extract_roi_labels_from_atlas


DEFAULT_GROUP_ALIASES = {
    "C": "Converter",
    "CONVERTER": "Converter",
    "CONVERTERS": "Converter",
    "NC": "Non-converter",
    "NONCONVERTER": "Non-converter",
    "NON-CONVERTER": "Non-converter",
    "NONCONVERTERS": "Non-converter",
    "NON-CONVERTERS": "Non-converter",
    "AD": "AD",
    "ALZHEIMER": "AD",
    "ALZHEIMERS": "AD",
    "CONTROL": "Control",
    "CONTROLS": "Control",
    "CTRL": "Control",
    "HC": "Control",
    "HEALTHY": "Control",
}


@dataclass(frozen=True)
class SubjectRecord:
    """Minimal metadata needed to load one subject from disk."""
    subject_id: str
    group: str
    path: Path
    mat_variable: str


@dataclass
class LoadedSubject:
    """Fully loaded subject data in a pipeline-friendly shape."""
    record: SubjectRecord
    data_key: str
    raw_struct: object
    X: np.ndarray
    t: np.ndarray
    X_trials: np.ndarray
    roi_labels: List[str]
    atlas_name: str
    n_trials: int
    n_rois: int
    n_time: int


def normalize_group_name(name: str, group_aliases: Mapping[str, str] | None = None) -> str | None:
    """Map a subject or path token onto a normalized group label.

    The discovery step looks first at the top-level variable name, because in
    the current dataset subject IDs already encode the group (``C_*`` or
    ``NC_*``). Parent directory names are only used as a fallback.
    """
    aliases = {key.upper(): value for key, value in (group_aliases or DEFAULT_GROUP_ALIASES).items()}
    tokens = [name.upper()]
    tokens.extend(part.upper() for part in Path(name).parts)
    tokens.extend(part.upper() for part in name.replace("-", "_").split("_"))
    for token in tokens:
        if token in aliases:
            return aliases[token]
    return None


def list_mat_struct_names(path: str | Path) -> List[str]:
    """Return the names of top-level MATLAB structs in a file."""
    return [name for name, _, cls in sio.whosmat(path) if cls == "struct"]


def discover_subjects(data_root: str | Path, group_aliases: Mapping[str, str] | None = None) -> List[SubjectRecord]:
    """Discover all subjects under ``data_root``.

    The dataset is split across several large MATLAB files. Each file contains
    many subject structs, so the discovery stage enumerates every struct and
    turns it into a ``SubjectRecord``.
    """
    root = Path(data_root)
    if not root.exists():
        return []
    subjects: List[SubjectRecord] = []
    for path in sorted(root.rglob("*.mat")):
        for variable_name in list_mat_struct_names(path):
            group = normalize_group_name(variable_name, group_aliases=group_aliases)
            if group is None:
                parent_names = [path.parent.name] + [parent.name for parent in path.parents]
                for name in parent_names:
                    group = normalize_group_name(name, group_aliases=group_aliases)
                    if group:
                        break
            if group is None:
                continue
            subjects.append(
                SubjectRecord(
                    subject_id=variable_name,
                    group=group,
                    path=path,
                    mat_variable=variable_name,
                )
            )
    return subjects


def _find_data_struct(data: Mapping[str, object]) -> tuple[str, object]:
    """Find the first MATLAB struct that looks like a Brainstorm subject."""
    for key, value in data.items():
        if key.startswith("__"):
            continue
        if any(hasattr(value, attr) for attr in ("Value", "value")) and any(hasattr(value, attr) for attr in ("Time", "time")):
            return key, value
    raise KeyError("No Brainstorm-like struct with Value and Time fields was found in the .mat file.")


def _get_first_attr(obj: object, candidates: Iterable[str]) -> object | None:
    """Return the first available attribute from a list of candidate names."""
    for candidate in candidates:
        if hasattr(obj, candidate):
            return getattr(obj, candidate)
    return None


def load_subject_structs(path: str | Path, variable_names: Iterable[str]) -> Mapping[str, object]:
    """Load a selected subset of top-level MATLAB variables from disk."""
    return sio.loadmat(path, variable_names=list(variable_names), squeeze_me=True, struct_as_record=False)


def load_brainstorm_subject(record: SubjectRecord, subject_struct: object | None = None) -> LoadedSubject:
    """Load one subject and reshape it into ``trials x rois x time``.

    The Brainstorm ``Value`` field stores ROI rows stacked trial-by-trial. If a
    subject has ``T`` trials and ``R`` ROIs, the matrix has ``T * R`` rows and
    one time axis shared by all trials. This function checks that assumption and
    performs the reshape explicitly.
    """
    if subject_struct is None:
        data = load_subject_structs(record.path, [record.mat_variable])
        data_key, subject_struct = _find_data_struct(data)
    else:
        data_key = record.mat_variable
    X = np.asarray(_get_first_attr(subject_struct, ("Value", "value")))
    t = np.asarray(_get_first_attr(subject_struct, ("Time", "time")))
    atlas = _get_first_attr(subject_struct, ("Atlas", "atlas"))
    roi_labels = extract_roi_labels_from_atlas(atlas) if atlas is not None else []
    n_rois = len(roi_labels) if roi_labels else X.shape[0]
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D Value matrix, got shape {X.shape} for {record.path}.")
    # Some MATLAB exports can arrive transposed after ``loadmat``. If the time
    # axis is clearly on the wrong dimension but the transpose matches the atlas
    # structure, fix it automatically instead of failing later.
    if X.shape[0] % n_rois != 0 and X.shape[1] % n_rois == 0 and X.shape[0] == t.shape[0]:
        X = X.T
    if X.shape[0] % n_rois != 0:
        raise ValueError(
            f"Cannot reshape {record.path} into trials x ROI x time with {n_rois} ROIs; "
            f"Value shape is {X.shape}."
        )
    n_trials = X.shape[0] // n_rois
    X_trials = X.reshape(n_trials, n_rois, X.shape[1])
    atlas_name = str(getattr(atlas, "Name", "unknown"))
    return LoadedSubject(
        record=record,
        data_key=data_key,
        raw_struct=subject_struct,
        X=X,
        t=t,
        X_trials=X_trials,
        roi_labels=roi_labels or [f"ROI{i}" for i in range(n_rois)],
        atlas_name=atlas_name,
        n_trials=n_trials,
        n_rois=n_rois,
        n_time=X.shape[1],
    )


def crop_trials(X_trials: np.ndarray, start: int | None = None, end: int | None = None) -> np.ndarray:
    """Crop the time axis of every trial with Python slice semantics."""
    start_idx = 0 if start is None else start
    end_idx = X_trials.shape[-1] if end is None else end
    return np.asarray(X_trials)[:, :, start_idx:end_idx]
