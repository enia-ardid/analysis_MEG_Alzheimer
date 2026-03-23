from __future__ import annotations

import unittest

import numpy as np

from meg_alzheimer.signals import analytic_signal, band_defs, bandpass_filt, phase_and_envelope


class SignalTests(unittest.TestCase):
    def test_band_defs_match_pipeline_bands(self) -> None:
        self.assertEqual(
            band_defs(),
            {
                "delta": (1.0, 4.0),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 12.0),
                "beta": (13.0, 30.0),
                "gamma": (30.0, 48.0),
            },
        )

    def test_bandpass_and_hilbert_keep_signal_shape(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3, 4000))
        filtered = bandpass_filt(x, fs=1000.0, band=(8.0, 12.0))
        analytic = analytic_signal(filtered)
        phase, envelope = phase_and_envelope(filtered)

        self.assertEqual(filtered.shape, x.shape)
        self.assertEqual(analytic.shape, x.shape)
        self.assertEqual(phase.shape, x.shape)
        self.assertEqual(envelope.shape, x.shape)
        self.assertTrue(np.isfinite(envelope).all())


if __name__ == "__main__":
    unittest.main()
