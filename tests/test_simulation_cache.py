from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from simulation_aec_snr_bias.run_all import _load_or_build_frame


class SimulationCacheTests(unittest.TestCase):
    def test_legacy_or_invalid_metadata_rebuilds_results(self) -> None:
        for metadata in (None, '{"metric_definition": "clipped_v1"}', "invalid JSON"):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                csv_path = Path(directory) / "experiment.csv"
                pd.DataFrame({"mean_gap": [0.0]}).to_csv(csv_path, index=False)
                if metadata is not None:
                    csv_path.with_suffix(".metadata.json").write_text(metadata)
                corrected = pd.DataFrame({"mean_gap": [-0.03]})
                builder = Mock(return_value=corrected)

                actual = _load_or_build_frame(csv_path, False, builder)

                builder.assert_called_once_with()
                pd.testing.assert_frame_equal(actual, corrected)
                pd.testing.assert_frame_equal(pd.read_csv(csv_path), corrected)
                self.assertIn("metric_definition", json.loads(csv_path.with_suffix(".metadata.json").read_text()))

    def test_current_results_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "experiment.csv"
            expected = pd.DataFrame({"mean_gap": [-0.03]})
            _load_or_build_frame(csv_path, False, lambda: expected)
            builder = Mock(side_effect=AssertionError("A current cache should be reused"))

            actual = _load_or_build_frame(csv_path, False, builder)

            builder.assert_not_called()
            pd.testing.assert_frame_equal(actual, expected)

    def test_force_rebuilds_current_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "experiment.csv"
            _load_or_build_frame(csv_path, False, lambda: pd.DataFrame({"mean_gap": [-0.03]}))
            updated = pd.DataFrame({"mean_gap": [-0.02]})
            builder = Mock(return_value=updated)

            actual = _load_or_build_frame(csv_path, True, builder)

            builder.assert_called_once_with()
            pd.testing.assert_frame_equal(actual, updated)
            pd.testing.assert_frame_equal(pd.read_csv(csv_path), updated)


if __name__ == "__main__":
    unittest.main()
