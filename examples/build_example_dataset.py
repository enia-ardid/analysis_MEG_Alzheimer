from __future__ import annotations

"""Build a small public Brainstorm-like ROI time-series example dataset.

The generated files are intentionally tiny compared with the private cohort.
They are meant to exercise the public loader and validators, not to reproduce
the scientific results of the paper.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATA_ROOT = REPO_ROOT / "examples" / "brainstorm_roi_small" / "data"
FS = 1000.0
N_TIME = 8000
N_TRIALS = 2


def build_roi_labels() -> list[str]:
    """Return a Schaefer-like 102-label set compatible with the paper mapping."""
    labels = [
        *[f"DefaultA_{idx:02d}" for idx in range(1, 7)],
        *[f"DefaultB_{idx:02d}" for idx in range(1, 7)],
        *[f"DefaultC_{idx:02d}" for idx in range(1, 7)],
        *[f"ContA_{idx:02d}" for idx in range(1, 7)],
        *[f"ContB_{idx:02d}" for idx in range(1, 7)],
        *[f"ContC_{idx:02d}" for idx in range(1, 7)],
        *[f"DorsAttnA_{idx:02d}" for idx in range(1, 7)],
        *[f"DorsAttnB_{idx:02d}" for idx in range(1, 7)],
        *[f"SalVentAttnA_{idx:02d}" for idx in range(1, 7)],
        *[f"SalVentAttnB_{idx:02d}" for idx in range(1, 7)],
        *[f"SomMotA_{idx:02d}" for idx in range(1, 7)],
        *[f"SomMotB_{idx:02d}" for idx in range(1, 7)],
        *[f"LimbicA_{idx:02d}" for idx in range(1, 5)],
        *[f"LimbicB_{idx:02d}" for idx in range(1, 5)],
        *[f"TempPar_{idx:02d}" for idx in range(1, 7)],
        *[f"VisCent_{idx:02d}" for idx in range(1, 8)],
        *[f"VisPeri_{idx:02d}" for idx in range(1, 8)],
        "Background+FreeSurfer_Defined_Medial_Wall L",
        "Background+FreeSurfer_Defined_Medial_Wall R",
    ]
    if len(labels) != 102:
        raise ValueError(f"Expected 102 ROI labels, found {len(labels)}.")
    return labels


def build_atlas(labels: list[str]) -> dict[str, object]:
    """Create the minimal Brainstorm-like atlas struct used by the loader."""
    scouts = np.empty(len(labels), dtype=object)
    for idx, label in enumerate(labels):
        scouts[idx] = {"Label": label}
    return {"Name": "Schaefer_100_17net", "Scouts": scouts}


def generate_trial_block(n_rois: int, seed: int) -> np.ndarray:
    """Generate one alpha-dominated ROI-by-time block."""
    rng = np.random.default_rng(seed)
    time_s = np.arange(N_TIME, dtype=float) / FS
    carrier = np.sin(2.0 * np.pi * 10.0 * time_s)
    slow_mod = 1.0 + 0.15 * np.sin(2.0 * np.pi * 0.5 * time_s + rng.uniform(-np.pi, np.pi))
    shared = slow_mod * carrier

    roi_gain = rng.normal(loc=1.0, scale=0.05, size=(n_rois, 1))
    roi_phase = rng.normal(loc=0.0, scale=0.12, size=(n_rois, 1))
    roi_noise = rng.normal(scale=0.35, size=(n_rois, N_TIME))
    signal = roi_gain * np.sin(2.0 * np.pi * 10.0 * time_s + roi_phase)
    return (0.45 * signal + 0.35 * shared + roi_noise).astype(np.float32)


def build_subject_struct(labels: list[str], seed: int) -> dict[str, object]:
    """Create one synthetic subject struct with Brainstorm-like fields."""
    trial_blocks = [generate_trial_block(len(labels), seed + trial_idx) for trial_idx in range(N_TRIALS)]
    value = np.concatenate(trial_blocks, axis=0)
    time = np.linspace(-2.0, 5.999, N_TIME, dtype=np.float32)
    return {
        "Value": value,
        "Time": time,
        "Atlas": build_atlas(labels),
    }


def main() -> None:
    """Write the public example `.mat` files to disk."""
    labels = build_roi_labels()
    EXAMPLE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    sio.savemat(
        EXAMPLE_DATA_ROOT / "C_example.mat",
        {
            "C_EXAMPLE_001": build_subject_struct(labels, seed=1),
            "C_EXAMPLE_002": build_subject_struct(labels, seed=2),
        },
    )
    sio.savemat(
        EXAMPLE_DATA_ROOT / "NC_example.mat",
        {
            "NC_EXAMPLE_001": build_subject_struct(labels, seed=3),
            "NC_EXAMPLE_002": build_subject_struct(labels, seed=4),
        },
    )
    print(f"Example dataset written to {EXAMPLE_DATA_ROOT}")


if __name__ == "__main__":
    main()
