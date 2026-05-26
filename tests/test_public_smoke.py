from __future__ import annotations

import unittest
from pathlib import Path

from meg_alzheimer.dataset import crop_trials, discover_subjects, load_brainstorm_subject
from meg_alzheimer.pipeline import subject_conn_from_trials


class PublicExampleSmokeTests(unittest.TestCase):
    def test_public_example_runs_loader_and_alpha_connectivity(self) -> None:
        data_root = Path(__file__).resolve().parents[1] / "examples" / "brainstorm_roi_small" / "data"
        records = discover_subjects(data_root)

        self.assertEqual(len(records), 4)

        loaded = load_brainstorm_subject(records[0])
        self.assertEqual(loaded.n_trials, 2)
        self.assertEqual(loaded.n_rois, 102)
        self.assertEqual(loaded.n_time, 8000)

        cropped = crop_trials(loaded.X_trials, start=2000, end=6000)
        matrices = subject_conn_from_trials(
            cropped,
            fs=1000.0,
            bands={"alpha": (8.0, 12.0)},
        )

        self.assertEqual(set(matrices), {"alpha"})
        self.assertEqual(set(matrices["alpha"]), {"PLV", "AEC", "AEC-orth"})
        for matrix in matrices["alpha"].values():
            self.assertEqual(matrix.shape, (102, 102))


if __name__ == "__main__":
    unittest.main()
