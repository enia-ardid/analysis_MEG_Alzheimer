from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from meg_alzheimer.strong_hypotheses import STRONG_ENDPOINTS, _extract_endpoint_frame


def _synthetic_network_df() -> pd.DataFrame:
    rows = []
    specs = [
        ("S1", "Converter", 0.06, 0.02, 0.04, 0.01),
        ("S2", "Converter", 0.07, 0.03, 0.05, 0.02),
        ("S3", "Non-converter", 0.09, 0.03, 0.06, 0.02),
        ("S4", "Non-converter", 0.10, 0.04, 0.07, 0.03),
    ]
    for subject_id, group, alpha_aec, alpha_orth, beta_aec, beta_orth in specs:
        for metric, alpha_value, beta_value in [
            ("AEC", alpha_aec, beta_aec),
            ("AEC-orth", alpha_orth, beta_orth),
        ]:
            rows.extend(
                [
                    {
                        "subject_id": subject_id,
                        "group": group,
                        "band": "alpha",
                        "metric": metric,
                        "connection_type": "intra",
                        "network_a": "TempPar",
                        "network_b": "TempPar",
                        "value": alpha_value,
                    },
                    {
                        "subject_id": subject_id,
                        "group": group,
                        "band": "alpha",
                        "metric": metric,
                        "connection_type": "inter",
                        "network_a": "TempPar",
                        "network_b": "Control",
                        "value": alpha_value,
                    },
                    {
                        "subject_id": subject_id,
                        "group": group,
                        "band": "beta",
                        "metric": metric,
                        "connection_type": "inter",
                        "network_a": "TempPar",
                        "network_b": "Control",
                        "value": beta_value,
                    },
                ]
            )
    return pd.DataFrame(rows)


class StrongHypothesisTests(unittest.TestCase):
    def test_strong_endpoints_can_be_reconstructed_from_network_rows(self) -> None:
        network_df = _synthetic_network_df()
        frames = {spec.endpoint_id: _extract_endpoint_frame(spec, network_df) for spec in STRONG_ENDPOINTS}

        self.assertEqual(set(frames), {spec.endpoint_id for spec in STRONG_ENDPOINTS})
        h3_full = frames["H3_gap_full"]
        h3_inter = frames["H3_gap_inter"]

        self.assertTrue(np.allclose(h3_full.loc[h3_full["subject_id"] == "S1", "value"], [0.04]))
        self.assertTrue(np.allclose(h3_inter.loc[h3_inter["subject_id"] == "S4", "value"], [0.06]))
        self.assertLess(
            h3_full.groupby("group")["value"].mean()["Converter"],
            h3_full.groupby("group")["value"].mean()["Non-converter"],
        )


if __name__ == "__main__":
    unittest.main()
