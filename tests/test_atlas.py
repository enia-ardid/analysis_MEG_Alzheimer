from __future__ import annotations

import unittest

import numpy as np

from meg_alzheimer.atlas import build_network_masks, mean_inter, mean_intra


class AtlasTests(unittest.TestCase):
    def test_build_network_masks_collapses_expected_prefixes(self) -> None:
        roi_labels = [
            "TempPar_1",
            "TempPar_2",
            "ContA_1",
            "DefaultB_1",
            "Background+FreeSurfer_Defined_Medial_Wall L",
        ]
        networks, masks = build_network_masks(roi_labels)

        self.assertEqual(networks, ["TempPar", "TempPar", "Control", "Default", "Background"])
        self.assertEqual(set(masks), {"Background", "Control", "Default", "TempPar"})
        self.assertEqual(masks["TempPar"].tolist(), [True, True, False, False, False])

    def test_mean_intra_and_mean_inter_use_expected_blocks(self) -> None:
        matrix = np.array(
            [
                [0.0, 0.30, 0.10, 0.20],
                [0.30, 0.0, 0.40, 0.50],
                [0.10, 0.40, 0.0, 0.60],
                [0.20, 0.50, 0.60, 0.0],
            ]
        )
        temp_mask = np.array([True, True, False, False])
        control_mask = np.array([False, False, True, True])

        self.assertEqual(mean_intra(matrix, temp_mask), 0.30)
        self.assertAlmostEqual(mean_inter(matrix, temp_mask, control_mask), np.mean([0.10, 0.20, 0.40, 0.50]))


if __name__ == "__main__":
    unittest.main()
