from __future__ import annotations

import unittest

from examples.build_example_dataset import build_roi_labels


class ExampleDatasetTests(unittest.TestCase):
    def test_example_dataset_uses_102_labels(self) -> None:
        labels = build_roi_labels()
        self.assertEqual(len(labels), 102)
        self.assertEqual(labels[-2:], [
            "Background+FreeSurfer_Defined_Medial_Wall L",
            "Background+FreeSurfer_Defined_Medial_Wall R",
        ])


if __name__ == "__main__":
    unittest.main()
