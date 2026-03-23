"""Monte Carlo simulation of AEC versus AEC-orth bias under SNR differences."""

import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_PACKAGE_ROOT / ".mplconfig"))

from .connectivity import analytic_signal, compute_aec, compute_aec_orth, compute_gap
from .experiments import (
    run_experiment_1_snr_sweep,
    run_experiment_2_trial_count_effect,
    run_experiment_3_snr_rho_interaction,
)
from .signal_model import generate_coupled_pair, generate_subject, make_alpha_filter

__all__ = [
    "analytic_signal",
    "compute_aec",
    "compute_aec_orth",
    "compute_gap",
    "generate_coupled_pair",
    "generate_subject",
    "make_alpha_filter",
    "run_experiment_1_snr_sweep",
    "run_experiment_2_trial_count_effect",
    "run_experiment_3_snr_rho_interaction",
]
