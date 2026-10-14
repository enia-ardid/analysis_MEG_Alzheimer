from __future__ import annotations

import unittest

import numpy as np
from scipy.signal import freqz, hilbert

from meg_alzheimer.connectivity import aec_from_envelope, aec_orth_from_analytic
from simulation_aec_snr_bias.connectivity import compute_aec, compute_aec_orth, compute_gap, compute_trial_metrics_batch
from simulation_aec_snr_bias.signal_model import generate_subject, make_alpha_filter, sigma_from_snr


class SimulationTests(unittest.TestCase):
    def test_high_snr_recovers_true_connectivity(self) -> None:
        rng = np.random.default_rng(0)
        metrics = generate_subject(rho=0.08, n_trials=100, trial_len=4000, fs=1000.0, sigma_meas=sigma_from_snr(1000.0), rng=rng)
        self.assertLess(abs(metrics.aec - 0.08), 0.02)
        self.assertLess(abs(metrics.aec_orth - 0.08), 0.02)

    def test_zero_connectivity_stays_near_zero(self) -> None:
        rng = np.random.default_rng(1)
        metrics = generate_subject(rho=0.0, n_trials=100, trial_len=4000, fs=1000.0, sigma_meas=sigma_from_snr(10.0), rng=rng)
        self.assertLess(abs(metrics.aec), 0.02)

    def test_simulation_metrics_match_cohort_pipeline(self) -> None:
        rng = np.random.default_rng(2)
        x_i = rng.normal(size=(8, 512))
        x_k = 0.2 * x_i + rng.normal(size=x_i.shape)
        expected = []
        for trial_i, trial_k in zip(x_i, x_k):
            analytic = hilbert(np.stack([trial_i, trial_k]), axis=-1)
            aec = aec_from_envelope(np.abs(analytic), fs=1000.0)[0, 1]
            orth = aec_orth_from_analytic(analytic)[0, 1]
            expected.append((aec, orth, aec - orth))
        expected = np.asarray(expected)

        # Include a counterexample to AEC-orth <= AEC, plus signed correlations.
        self.assertLess(expected[:, 2].min(), -0.01)
        self.assertLess(expected[:, 1].min(), -0.01)
        for scale in (1.0, 1e-15, 1e3):
            with self.subTest(scale=scale):
                actual = np.column_stack(compute_trial_metrics_batch(scale * x_i, scale * x_k))
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
                for index, (trial_i, trial_k) in enumerate(zip(scale * x_i, scale * x_k)):
                    np.testing.assert_allclose(compute_gap(trial_i, trial_k), expected[index], rtol=1e-10, atol=1e-12)
                    self.assertAlmostEqual(compute_aec(trial_i, trial_k), expected[index, 0])
                    self.assertAlmostEqual(compute_aec_orth(trial_i, trial_k), expected[index, 1])

    def test_zero_signals_return_finite_zero_metrics(self) -> None:
        signal = np.random.default_rng(3).normal(size=512)
        zero = np.zeros_like(signal)
        x_i = np.stack([zero, signal, zero])
        x_k = np.stack([signal, zero, zero])
        with np.errstate(divide="raise", invalid="raise"):
            np.testing.assert_array_equal(compute_trial_metrics_batch(x_i, x_k), np.zeros((3, 3)))
            for trial_i, trial_k in zip(x_i, x_k):
                self.assertEqual(compute_gap(trial_i, trial_k), (0.0, 0.0, 0.0))
                self.assertEqual(compute_aec(trial_i, trial_k), 0.0)
                self.assertEqual(compute_aec_orth(trial_i, trial_k), 0.0)

    def test_equal_trial_counts_do_not_create_large_gap_effect(self) -> None:
        rng = np.random.default_rng(7)
        gaps = []
        for _ in range(24):
            subject_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
            metrics = generate_subject(
                rho=0.08,
                n_trials=36,
                trial_len=4000,
                fs=1000.0,
                sigma_meas=sigma_from_snr(10.0),
                rng=subject_rng,
            )
            gaps.append(metrics.gap)
        gaps = np.asarray(gaps, dtype=float)

        ds = []
        for _ in range(80):
            perm = rng.permutation(gaps.size)
            group_c = gaps[perm[:12]]
            group_nc = gaps[perm[12:]]
            pooled = np.sqrt(
                ((len(group_c) - 1) * np.var(group_c, ddof=1) + (len(group_nc) - 1) * np.var(group_nc, ddof=1))
                / (len(group_c) + len(group_nc) - 2)
            )
            ds.append(0.0 if pooled < 1e-12 else (group_nc.mean() - group_c.mean()) / pooled)
        self.assertLess(abs(float(np.mean(ds))), 0.15)

    def test_alpha_filter_meets_response_criteria(self) -> None:
        taps = make_alpha_filter(fs=1000.0, order=801)
        frequencies, response = freqz(taps, worN=32768, fs=1000.0)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
        passband = (frequencies >= 9.0) & (frequencies <= 11.0)
        stopband = (frequencies <= 5.0) | (frequencies >= 15.0)
        ripple = float(np.max(magnitude_db[passband]) - np.min(magnitude_db[passband]))
        attenuation = float(-np.max(magnitude_db[stopband]))
        self.assertLess(ripple, 0.1)
        self.assertGreater(attenuation, 40.0)


if __name__ == "__main__":
    unittest.main()
