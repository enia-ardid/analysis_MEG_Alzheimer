from __future__ import annotations

import unittest

import pandas as pd

from scripts.final_figures.build_trials_threshold_sensitivity import ENDPOINTS, run_threshold_sensitivity


class ThresholdSensitivityTests(unittest.TestCase):
    def test_threshold_sensitivity_returns_one_row_per_endpoint_and_threshold(self) -> None:
        records = []
        for idx in range(6):
            records.append(
                {
                    "subject_id": f"C_{idx}",
                    "group": "Converter",
                    "n_valid_trials": 42 + idx,
                    "H1_AEC": 0.05 + 0.001 * idx,
                    "H1_AECorth": 0.02 + 0.001 * idx,
                    "H2_AEC": 0.04 + 0.001 * idx,
                    "H2_AECorth": 0.01 + 0.001 * idx,
                    "H3_gap_full": 0.03 + 0.001 * idx,
                    "H3_gap_inter": 0.03 + 0.001 * idx,
                }
            )
            records.append(
                {
                    "subject_id": f"NC_{idx}",
                    "group": "Non-converter",
                    "n_valid_trials": 42 + idx,
                    "H1_AEC": 0.08 + 0.001 * idx,
                    "H1_AECorth": 0.03 + 0.001 * idx,
                    "H2_AEC": 0.06 + 0.001 * idx,
                    "H2_AECorth": 0.02 + 0.001 * idx,
                    "H3_gap_full": 0.05 + 0.001 * idx,
                    "H3_gap_inter": 0.04 + 0.001 * idx,
                }
            )

        df = pd.DataFrame.from_records(records)
        results = run_threshold_sensitivity(df, thresholds=[40, 45])

        self.assertEqual(len(results), 2 * len(ENDPOINTS))
        self.assertEqual(set(results["endpoint"]), set(ENDPOINTS))
        self.assertEqual(set(results["T_min"]), {40, 45})
        self.assertTrue({"t_stat", "p_raw", "d", "holm_p", "holm_pass"} <= set(results.columns))


if __name__ == "__main__":
    unittest.main()
