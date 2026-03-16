#!/usr/bin/env python3
from __future__ import annotations

"""Strong-family check for post hoc exact Schaefer-17 beta/AEC-orth blocks.

This script evaluates all unordered two-network exact Schaefer-17 blocks in the
beta AEC-orth matrix. Each block is summarized, per subject, by the arithmetic
mean of its three unique cells:

- A-A
- A-B
- B-B

The goal is to answer a narrower question than the main H1-H3 analysis:
whether a visually identified exact-network block remains compelling once it is
embedded in a complete, objective family of analogous blocks rather than tested
in isolation.

The output family contains all unordered pairs of distinct exact Schaefer-17
networks. For 17 networks this yields 136 block-mean endpoints. The script
reports:

- two-sided Welch t-tests
- two-sided Holm correction across the 136 block means
- two-sided max-T permutation correction across the same family

The targeted SomMotA-DorsAttnB block can then be interpreted against that full
family-wise benchmark.
"""

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from meg_alzheimer.atlas import get_network_prefix, mean_inter, mean_intra


GROUP_A = "Converter"
GROUP_B = "Non-converter"
TARGET_A = "SomMotA"
TARGET_B = "DorsAttnB"

S17_ORDER = [
    "VisCent",
    "VisPeri",
    "SomMotA",
    "SomMotB",
    "DorsAttnA",
    "DorsAttnB",
    "SalVentAttnA",
    "SalVentAttnB",
    "LimbicA",
    "LimbicB",
    "TempPar",
    "ContA",
    "ContB",
    "ContC",
    "DefaultA",
    "DefaultB",
    "DefaultC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the exact Schaefer-17 beta/AEC-orth block family.")
    parser.add_argument(
        "--subjects-root",
        default="outputs_full_cohort/subjects",
        help="Folder containing per-subject connectivity_matrices.npz and metadata.json files.",
    )
    parser.add_argument("--table-dir", default="tables/final", help="Destination folder for output tables.")
    parser.add_argument("--n-perm", type=int, default=50000, help="Permutation count for max-T correction.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def _iter_subject_dirs(subjects_root: Path) -> list[Path]:
    return sorted(path for path in subjects_root.glob("*/*") if path.is_dir())


def _load_subject(subject_dir: Path) -> dict[str, Any]:
    with open(subject_dir / "metadata.json") as f:
        meta = json.load(f)
    matrices = np.load(subject_dir / "connectivity_matrices.npz")
    return {"subject_dir": subject_dir, "metadata": meta, "matrices": matrices}


def _raw_network_labels(roi_labels: list[str]) -> list[str]:
    labels = [get_network_prefix(label) for label in roi_labels]
    return ["Background" if label.startswith("Background+FreeSurfer") else label for label in labels]


def _validate_subjects(subject_payloads: list[dict[str, Any]]) -> list[str]:
    if not subject_payloads:
        raise ValueError("No subject folders found.")
    first_labels = subject_payloads[0]["metadata"]["roi_labels"]
    for payload in subject_payloads[1:]:
        if payload["metadata"]["roi_labels"] != first_labels:
            raise ValueError("ROI labels differ across subjects.")
    raw_labels = _raw_network_labels(first_labels)
    present = set(raw_labels)
    missing = [label for label in S17_ORDER if label not in present]
    if missing:
        raise ValueError(f"Missing exact Schaefer-17 labels in ROI order: {missing}")
    return raw_labels


def _cohen_d_unpaired(values_a: np.ndarray, values_b: np.ndarray) -> float:
    mean_diff = float(np.mean(values_a) - np.mean(values_b))
    var_a = float(np.var(values_a, ddof=1))
    var_b = float(np.var(values_b, ddof=1))
    pooled = np.sqrt(((len(values_a) - 1) * var_a + (len(values_b) - 1) * var_b) / max(1, len(values_a) + len(values_b) - 2))
    if pooled < 1e-12:
        return float("nan")
    return mean_diff / pooled


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for idx, value in enumerate(sorted_p):
        adjusted = (m - idx) * value
        running = max(running, adjusted)
        adjusted_sorted[idx] = min(1.0, running)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _welch_t_columns(matrix: np.ndarray, group_a_mask: np.ndarray) -> np.ndarray:
    group_b_mask = ~group_a_mask
    xa = matrix[group_a_mask]
    xb = matrix[group_b_mask]
    mean_a = xa.mean(axis=0)
    mean_b = xb.mean(axis=0)
    var_a = xa.var(axis=0, ddof=1)
    var_b = xb.var(axis=0, ddof=1)
    denom = np.sqrt(var_a / xa.shape[0] + var_b / xb.shape[0])
    denom = np.where(denom < 1e-12, np.nan, denom)
    t_stat = (mean_a - mean_b) / denom
    return np.nan_to_num(t_stat, nan=0.0, posinf=0.0, neginf=0.0)


def _max_t_pvals(observed_t: np.ndarray, matrix: np.ndarray, group_a_mask: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray(group_a_mask, dtype=bool)
    observed = np.abs(observed_t)
    null_max = np.empty(n_perm, dtype=float)
    for idx in range(n_perm):
        perm_mask = rng.permutation(labels)
        perm_t = _welch_t_columns(matrix, group_a_mask=perm_mask)
        null_max[idx] = np.nanmax(np.abs(perm_t))
    return np.array([(np.sum(null_max >= value) + 1) / (n_perm + 1) for value in observed], dtype=float)


def _subject_block_rows(subject_payloads: list[dict[str, Any]], raw_labels: list[str]) -> pd.DataFrame:
    masks = {network: np.array([label == network for label in raw_labels], dtype=bool) for network in S17_ORDER}
    rows: list[dict[str, Any]] = []
    for payload in subject_payloads:
        meta = payload["metadata"]
        matrix = np.asarray(payload["matrices"]["beta__AEC-orth"], dtype=float)
        for network_a, network_b in combinations(S17_ORDER, 2):
            aa = mean_intra(matrix, masks[network_a])
            ab = mean_inter(matrix, masks[network_a], masks[network_b])
            bb = mean_intra(matrix, masks[network_b])
            block_mean = float(np.mean([aa, ab, bb]))
            rows.append(
                {
                    "subject_id": meta["subject_id"],
                    "group": meta["group"],
                    "network_a": network_a,
                    "network_b": network_b,
                    "block_id": f"{network_a}__{network_b}",
                    "block_mean": block_mean,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    subjects_root = Path(args.subjects_root)
    table_dir = Path(args.table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    subject_payloads = [_load_subject(path) for path in _iter_subject_dirs(subjects_root)]
    raw_labels = _validate_subjects(subject_payloads)
    block_rows = _subject_block_rows(subject_payloads, raw_labels=raw_labels)

    pivot = (
        block_rows.pivot(index=["subject_id", "group"], columns="block_id", values="block_mean")
        .sort_index()
    )
    matrix = pivot.to_numpy(dtype=float)
    group_labels = pivot.index.get_level_values("group").to_numpy()
    group_a_mask = group_labels == GROUP_A
    columns = list(pivot.columns)

    t_stats = _welch_t_columns(matrix, group_a_mask=group_a_mask)
    p_two = []
    deltas = []
    ds = []
    mean_a = []
    mean_b = []
    for col_idx, block_id in enumerate(columns):
        values_a = matrix[group_a_mask, col_idx]
        values_b = matrix[~group_a_mask, col_idx]
        _, p_val = stats.ttest_ind(values_a, values_b, equal_var=False, nan_policy="omit")
        p_two.append(float(p_val))
        mean_a.append(float(np.mean(values_a)))
        mean_b.append(float(np.mean(values_b)))
        deltas.append(float(np.mean(values_a) - np.mean(values_b)))
        ds.append(_cohen_d_unpaired(values_a, values_b))

    p_two = np.array(p_two, dtype=float)
    holm = _holm_adjust(p_two)
    max_t = _max_t_pvals(t_stats, matrix=matrix, group_a_mask=group_a_mask, n_perm=args.n_perm, seed=args.seed)

    rows = []
    for idx, block_id in enumerate(columns):
        network_a, network_b = block_id.split("__")
        rows.append(
            {
                "block_id": block_id,
                "network_a": network_a,
                "network_b": network_b,
                "mean_converter": mean_a[idx],
                "mean_non_converter": mean_b[idx],
                "delta_converter_minus_non_converter": deltas[idx],
                "welch_t": float(t_stats[idx]),
                "p_two_sided": float(p_two[idx]),
                "holm_p_two_sided": float(holm[idx]),
                "max_t_p_two_sided": float(max_t[idx]),
                "holm_significant": bool(holm[idx] <= 0.05),
                "max_t_significant": bool(max_t[idx] <= 0.05),
                "effect_size_d": ds[idx],
            }
        )

    result = pd.DataFrame(rows).sort_values("p_two_sided").reset_index(drop=True)
    result["rank_by_p"] = np.arange(1, len(result) + 1)
    target_id = f"{TARGET_A}__{TARGET_B}"
    result["is_target_block"] = result["block_id"] == target_id

    csv_path = table_dir / "table_posthoc_beta_block_family.csv"
    result.to_csv(csv_path, index=False)

    target = result.loc[result["is_target_block"]].copy()
    if target.empty:
        raise ValueError(f"Target block {target_id} not found in full family table.")
    target.to_csv(table_dir / "table_posthoc_beta_block_family_target_only.csv", index=False)


if __name__ == "__main__":
    main()
