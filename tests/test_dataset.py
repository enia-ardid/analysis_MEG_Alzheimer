from __future__ import annotations

import unittest

import numpy as np

from meg_alzheimer.dataset import crop_trials, normalize_group_name


class DatasetTests(unittest.TestCase):
    def test_normalize_group_name_accepts_project_tokens(self) -> None:
        self.assertEqual(normalize_group_name("C_001"), "Converter")
        self.assertEqual(normalize_group_name("NC_subject_12"), "Non-converter")
        self.assertEqual(normalize_group_name("folder/Converters/site_A"), "Converter")

    def test_normalize_group_name_accepts_custom_aliases(self) -> None:
        aliases = {"progressor": "Converter", "stable": "Non-converter"}
        self.assertEqual(normalize_group_name("stable_subject", aliases), "Non-converter")
        self.assertEqual(normalize_group_name("study/progressor", aliases), "Converter")

    def test_crop_trials_uses_python_slice_semantics(self) -> None:
        X_trials = np.arange(2 * 3 * 10).reshape(2, 3, 10)
        cropped = crop_trials(X_trials, start=2, end=5)
        self.assertEqual(cropped.shape, (2, 3, 3))
        self.assertTrue(np.array_equal(cropped, X_trials[:, :, 2:5]))


if __name__ == "__main__":
    unittest.main()
