from __future__ import annotations

import unittest

import numpy as np
from scipy.signal import freqz

from simulation_aec_snr_bias.connectivity import compute_gap
from simulation_aec_snr_bias.signal_model import generate_coupled_pair, generate_subject, make_alpha_filter, sigma_from_snr


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

    def test_aec_orth_does_not_exceed_aec(self) -> None:
        rng = np.random.default_rng(2)
        violations = 0
        for _ in range(120):
            x_i, x_k = generate_coupled_pair(rho=float(rng.uniform(0.0, 0.2)), n_samples=4000, fs=1000.0, sigma_meas=sigma_from_snr(5.0), rng=rng)
            aec, aec_orth, _ = compute_gap(x_i, x_k)
            if aec_orth > aec + 1e-9:
                violations += 1
        self.assertEqual(violations, 0)

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
