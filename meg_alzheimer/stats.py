"""Statistical helpers used across the MEG connectivity workflow.

The project uses two broad classes of statistical routines:

1. Vector-level corrections such as Benjamini-Hochberg FDR, which are applied
   to families of p-values generated from the same inferential question.
2. Matrix-level operations for edgewise connectivity analyses, where each
   subject contributes one ROI-by-ROI matrix and group differences are computed
   independently at every edge.

The functions in this module stay intentionally small and explicit. They are
used as building blocks by higher-level modules that define the scientific
question, the family of tests to correct, and how the resulting statistics are
reported.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy import stats


def fdr_bh(pvals: np.ndarray, q: float = 0.1) -> Tuple[np.ndarray, float | None]:
    """Apply the Benjamini-Hochberg FDR procedure to an array of p-values.

    Parameters
    ----------
    pvals:
        Arbitrary-shaped array of p-values.
    q:
        Target false discovery rate.

    Returns
    -------
    mask:
        Boolean array with the same shape as ``pvals`` marking discoveries that
        survive the FDR threshold.
    pcrit:
        The largest p-value still accepted by the procedure, or ``None`` when
        nothing survives.
    """

    p = np.asarray(pvals, dtype=float).ravel()
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * np.arange(1, n + 1) / n
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros_like(np.asarray(pvals), dtype=bool), None
    kmax = np.where(passed)[0].max()
    pcrit = float(ranked[kmax])
    mask = np.asarray(pvals) <= pcrit
    return mask, pcrit


def apply_fdr_to_upper_triangle(Pmat: np.ndarray, q: float = 0.1) -> Tuple[np.ndarray, float | None]:
    """Run FDR on the unique edges of a symmetric p-value matrix.

    Connectivity matrices are symmetric, so only the upper triangle contains
    unique hypothesis tests. This function extracts those entries, applies FDR,
    and mirrors the surviving mask back to the full matrix.
    """

    Pmat = np.asarray(Pmat, dtype=float)
    iu = np.triu_indices_from(Pmat, k=1)
    mask_vec, pcrit = fdr_bh(Pmat[iu], q=q)
    mask = np.zeros_like(Pmat, dtype=bool)
    mask[iu] = mask_vec
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    return mask, pcrit


def edgewise_ttest(group1: np.ndarray, group2: np.ndarray, equal_var: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an independent-samples t-test at every ROI-to-ROI edge.

    Parameters
    ----------
    group1, group2:
        Arrays shaped ``(n_subjects, n_roi, n_roi)`` containing one connectivity
        matrix per subject.
    equal_var:
        Passed to ``scipy.stats.ttest_ind``. The pipeline defaults to Welch's
        test, so this is usually ``False``.

    Returns
    -------
    Tmat, Pmat:
        Symmetric matrices containing the edgewise t-statistics and p-values.
    """

    G1 = np.asarray(group1)
    G2 = np.asarray(group2)
    _, n_roi, _ = G1.shape
    iu = np.triu_indices(n_roi, k=1)
    X1 = G1[:, iu[0], iu[1]]
    X2 = G2[:, iu[0], iu[1]]
    t, p = stats.ttest_ind(X1, X2, axis=0, equal_var=equal_var, nan_policy="omit")
    Tmat = np.zeros((n_roi, n_roi), dtype=float)
    Pmat = np.ones((n_roi, n_roi), dtype=float)
    Tmat[iu] = t
    Pmat[iu] = p
    Tmat = Tmat + Tmat.T
    Pmat = Pmat + Pmat.T - np.eye(n_roi)
    np.fill_diagonal(Tmat, 0.0)
    np.fill_diagonal(Pmat, 1.0)
    return Tmat, Pmat


def cohen_d_edgewise(group1: np.ndarray, group2: np.ndarray) -> np.ndarray:
    """Estimate Cohen's d at every unique edge of two matrix groups."""

    G1 = np.asarray(group1, dtype=float)
    G2 = np.asarray(group2, dtype=float)
    n1, n_roi, _ = G1.shape
    n2 = G2.shape[0]
    iu = np.triu_indices(n_roi, k=1)
    X1 = G1[:, iu[0], iu[1]]
    X2 = G2[:, iu[0], iu[1]]
    mean_diff = np.nanmean(X1, axis=0) - np.nanmean(X2, axis=0)
    var1 = np.nanvar(X1, axis=0, ddof=1)
    var2 = np.nanvar(X2, axis=0, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, (n1 + n2 - 2)))
    pooled = np.where(pooled == 0.0, np.nan, pooled)
    d = mean_diff / pooled
    Dmat = np.zeros((n_roi, n_roi), dtype=float)
    Dmat[iu] = d
    Dmat = Dmat + Dmat.T
    np.fill_diagonal(Dmat, 0.0)
    return Dmat


def permutations_edgewise(group1: np.ndarray, group2: np.ndarray, n_perm: int = 5000, seed: int = 0) -> np.ndarray:
    """Build edgewise permutation p-values using a max-statistic null.

    For every permutation, the function shuffles group labels and stores the
    largest absolute t-statistic observed across all edges. Each empirical edge
    is then compared against this null distribution of maxima, which controls
    family-wise error within the matrix.
    """

    rng = np.random.default_rng(seed)
    G1 = np.asarray(group1)
    G2 = np.asarray(group2)
    n1, n_roi, _ = G1.shape
    n2 = G2.shape[0]
    iu = np.triu_indices(n_roi, 1)
    X = np.concatenate([G1[:, iu[0], iu[1]], G2[:, iu[0], iu[1]]], axis=0)
    labels = np.r_[np.zeros(n1, dtype=int), np.ones(n2, dtype=int)]
    t_obs, _ = edgewise_ttest(G1, G2, equal_var=False)
    tobs = np.abs(t_obs[iu])
    null_max = np.empty(n_perm, dtype=float)
    for idx in range(n_perm):
        permuted = rng.permutation(labels)
        X1 = X[permuted == 0]
        X2 = X[permuted == 1]
        t_perm, _ = stats.ttest_ind(X1, X2, axis=0, equal_var=False, nan_policy="omit")
        # The maximum absolute statistic from each shuffle defines the family-wise
        # null against which every observed edge is tested.
        null_max[idx] = np.nanmax(np.abs(t_perm))
    p_vals = np.array([np.mean(null_max >= value) for value in tobs], dtype=float)
    P = np.ones((n_roi, n_roi), dtype=float)
    P[iu] = p_vals
    P = P + P.T - np.eye(n_roi)
    np.fill_diagonal(P, 1.0)
    return P


def welch_ttest_1d(group1: Iterable[float], group2: Iterable[float]) -> Tuple[float, float]:
    """Convenience wrapper for a one-dimensional Welch t-test."""

    result = stats.ttest_ind(group1, group2, equal_var=False, nan_policy="omit")
    return float(result.statistic), float(result.pvalue)
