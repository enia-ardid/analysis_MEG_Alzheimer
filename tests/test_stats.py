from __future__ import annotations

import unittest

import numpy as np
from scipy import stats

from meg_alzheimer.stats import apply_fdr_to_upper_triangle, edgewise_ttest, permutations_edgewise


def _matrices_from_edges(edges: np.ndarray) -> np.ndarray:
    """Build three-ROI subject matrices from the three unique edge values."""
    matrices = np.zeros((len(edges), 3, 3), dtype=float)
    for column, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        matrices[:, i, j] = matrices[:, j, i] = edges[:, column]
    return matrices


class EdgewiseStatsTests(unittest.TestCase):
    def setUp(self) -> None:
        edges = np.array(
            [
                [0.10, 0.11, 0.30],
                [0.12, 0.16, 0.32],
                [0.14, 0.18, 0.34],
                [0.16, 0.09, 0.36],
                [0.18, 0.14, 0.38],
                [0.20, 0.12, 0.40],
            ]
        )
        self.group_a = _matrices_from_edges(edges)
        # Two edges shift in opposite directions; the middle edge is unchanged.
        self.group_b = _matrices_from_edges(edges + [0.10, 0.0, -0.10])

    def assert_valid_pvalue_matrix(self, matrix: np.ndarray) -> None:
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue(((matrix >= 0.0) & (matrix <= 1.0)).all())
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), np.ones(3))

    def test_edgewise_ttest_matches_scipy(self) -> None:
        # Unequal group sizes also exercise Welch's unequal-variance calculation.
        group_b = self.group_b[:5]
        t_matrix, p_matrix = edgewise_ttest(self.group_a, group_b)

        for i, j in ((0, 1), (0, 2), (1, 2)):
            with self.subTest(edge=(i, j)):
                expected = stats.ttest_ind(
                    self.group_a[:, i, j], group_b[:, i, j], equal_var=False
                )
                self.assertAlmostEqual(t_matrix[i, j], expected.statistic)
                self.assertAlmostEqual(p_matrix[i, j], expected.pvalue)
        self.assert_valid_pvalue_matrix(p_matrix)
        np.testing.assert_array_equal(t_matrix, t_matrix.T)
        np.testing.assert_array_equal(np.diag(t_matrix), np.zeros(3))

    def test_edgewise_pvalues_retain_fdr_discoveries(self) -> None:
        _, p_matrix = edgewise_ttest(self.group_a, self.group_b)
        mask, pcrit = apply_fdr_to_upper_triangle(p_matrix, q=0.05)

        expected = np.array(
            [[False, True, False], [True, False, True], [False, True, False]]
        )
        np.testing.assert_array_equal(mask, expected)
        self.assertIsNotNone(pcrit)
        self.assertLess(pcrit, 0.05)

    def test_permutation_pvalues_for_identical_groups_are_one(self) -> None:
        p_matrix = permutations_edgewise(self.group_a, self.group_a, n_perm=31, seed=0)

        np.testing.assert_array_equal(p_matrix, np.ones((3, 3)))

    def test_permutation_pvalues_distinguish_shifted_and_unchanged_edges(self) -> None:
        p_matrix = permutations_edgewise(self.group_a, self.group_b, n_perm=199, seed=0)

        self.assert_valid_pvalue_matrix(p_matrix)
        self.assertLess(p_matrix[0, 1], 0.05)
        self.assertLess(p_matrix[1, 2], 0.05)
        self.assertEqual(p_matrix[0, 2], 1.0)


if __name__ == "__main__":
    unittest.main()
