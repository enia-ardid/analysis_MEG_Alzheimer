from __future__ import annotations

"""Plotting helpers for the AEC/AEC-orth SNR bias simulation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from .experiments import EMPIRICAL_D, EMPIRICAL_DELTA_GAP


def apply_publication_style() -> None:
    """Apply a consistent figure style for all simulation outputs."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "font.size": 11,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_experiment_1(exp1_df: pd.DataFrame, output_path: Path) -> None:
    """Plot delta-gap and Cohen's d as a function of SNR."""
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(exp1_df["snr"], exp1_df["mean_delta_gap"], color="#355c7d", marker="o")
    ax.fill_between(exp1_df["snr"], exp1_df["delta_gap_q05"], exp1_df["delta_gap_q95"], color="#355c7d", alpha=0.18)
    ax.axhline(EMPIRICAL_DELTA_GAP, color="#b22222", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Per-trial SNR")
    ax.set_ylabel(r"$\Delta$ gap (NC - C)")
    ax.text(0.03, 0.96, "(A) gap bias", transform=ax.transAxes, ha="left", va="top")

    ax = axes[1]
    ax.plot(exp1_df["snr"], exp1_df["mean_cohen_d"], color="#6c8e3a", marker="o")
    ax.fill_between(exp1_df["snr"], exp1_df["cohen_d_q05"], exp1_df["cohen_d_q95"], color="#6c8e3a", alpha=0.18)
    ax.axhline(EMPIRICAL_D, color="#b22222", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Per-trial SNR")
    ax.set_ylabel("Cohen's d (NC - C)")
    ax.text(0.03, 0.96, "(B) effect size", transform=ax.transAxes, ha="left", va="top")

    crossing = exp1_df.loc[exp1_df["mean_cohen_d"] >= EMPIRICAL_D]
    if not crossing.empty:
        snr_cross = float(crossing.iloc[0]["snr"])
        ax.axvline(snr_cross, color="#444444", linestyle=":", linewidth=1.0)
        ax.annotate(
            f"SNR ≈ {snr_cross:.2f}",
            xy=(snr_cross, float(crossing.iloc[0]["mean_cohen_d"])),
            xytext=(6, 10),
            textcoords="offset points",
            fontsize=9,
        )

    _ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)


def plot_experiment_2(exp2_df: pd.DataFrame, output_path: Path) -> None:
    """Plot Cohen's d as a function of group trial-count difference."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    ax.plot(exp2_df["delta_trials"], exp2_df["mean_cohen_d"], color="#8a5a44", marker="o")
    ax.fill_between(exp2_df["delta_trials"], exp2_df["cohen_d_q05"], exp2_df["cohen_d_q95"], color="#8a5a44", alpha=0.18)
    ax.axhline(EMPIRICAL_D, color="#b22222", linestyle="--", linewidth=1.0)
    ax.axvline(3, color="#444444", linestyle="--", linewidth=1.0)
    ax.annotate("Observed trial difference", xy=(3, ax.get_ylim()[0]), xytext=(5, 8), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Trial-count difference (NC mean - C mean)")
    ax.set_ylabel("Cohen's d (NC - C)")
    _ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)


def plot_experiment_3(exp3_df: pd.DataFrame, output_path: Path) -> None:
    """Plot the SNR-by-rho heatmap of mean Cohen's d."""
    apply_publication_style()
    pivot = exp3_df.pivot(index="rho", columns="snr", values="mean_cohen_d")
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    vmax = float(np.max(np.abs(pivot.to_numpy())))
    image = ax.imshow(
        pivot.to_numpy(),
        origin="lower",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [f"{value:.2f}" for value in pivot.index])
    ax.set_xlabel("Per-trial SNR")
    ax.set_ylabel(r"True connectivity $\rho$")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9, color="#111111")
    fig.colorbar(image, ax=ax, label="Mean Cohen's d (NC - C)")
    _ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)


def plot_summary(exp1_df: pd.DataFrame, output_path: Path) -> None:
    """Plot the SNR curve against empirical effect and plausible MEG SNR bands."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    ax.plot(exp1_df["snr"], exp1_df["mean_cohen_d"], color="#355c7d", marker="o", label="Simulation")
    ax.fill_between(exp1_df["snr"], exp1_df["cohen_d_q05"], exp1_df["cohen_d_q95"], color="#355c7d", alpha=0.15)
    ax.axhspan(EMPIRICAL_D - 0.15, EMPIRICAL_D + 0.15, color="#b22222", alpha=0.12, label="Empirical effect band")
    ax.axvspan(3.0, 8.0, color="#808080", alpha=0.15, label="Plausible MEG SNR")
    ax.set_xscale("log")
    ax.set_xlabel("Per-trial SNR")
    ax.set_ylabel("Cohen's d (NC - C)")
    ax.legend(frameon=False, loc="best")
    _ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)
