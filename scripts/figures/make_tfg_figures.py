#!/usr/bin/env python3
from __future__ import annotations

"""Build thesis-ready figures without altering the analysis definition.

This script only updates figure generation. It reuses the already exported
collapsed 9-network heatmap matrices for alpha and beta, and it builds two
illustrative conceptual schematics using deterministic toy signals.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.signal import hilbert

from simulation_aec_snr_bias.connectivity import compute_gap

NETWORK_ORDER = [
    "VisCent",
    "VisPeri",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "TempPar",
    "Control",
    "Default",
]

DISPLAY_LABELS = {
    "VisCent": "VisCent",
    "VisPeri": "VisPeri",
    "SomMot": "SomMot",
    "DorsAttn": "DorsAttn",
    "SalVentAttn": "SalVentAttn",
    "Limbic": "Limbic",
    "TempPar": "TempPar",
    "Control": "Control",
    "Default": "Default",
}

NOTE_KWARGS = {
    "fontsize": 11.5,
    "color": "#7a3b00",
    "bbox": {"boxstyle": "round,pad=0.28", "facecolor": "#fff7ec", "edgecolor": "#e6c48f", "alpha": 0.98},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis figures under figures/.")
    parser.add_argument(
        "--heatmap-alpha-csv",
        default="figures/final/fig_network_heatmaps_alpha_matrices_collapsed9.csv",
        help="Collapsed 9-network alpha heatmap matrix export produced by the existing repo pipeline.",
    )
    parser.add_argument(
        "--heatmap-beta-csv",
        default="figures/final/fig_network_heatmaps_beta_matrices_collapsed9.csv",
        help="Collapsed 9-network beta heatmap matrix export produced by the existing repo pipeline.",
    )
    parser.add_argument("--output-dir", default="figures", help="Destination directory for thesis figures.")
    return parser.parse_args()


def apply_thesis_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#1b1f23",
            "axes.linewidth": 1.0,
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.titlesize": 21,
            "legend.fontsize": 12,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_existing_file(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing required heatmap input: {path}")
    return path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_pdf_and_png(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    ensure_dir(output_dir)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=400)
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path, png_path


def load_heatmap_export(path: Path, expected_band: str) -> dict[tuple[str, str], np.ndarray]:
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise SystemExit(f"Heatmap input is empty: {path}")

    matrices: dict[tuple[str, str], np.ndarray] = {}
    grouped: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    for row in rows:
        if row["network_resolution"] != "collapsed9":
            raise SystemExit(
                f"Unexpected network resolution in {path}: {row['network_resolution']}. Expected collapsed9."
            )
        if row["band"] != expected_band:
            raise SystemExit(f"Unexpected band in {path}: {row['band']}. Expected {expected_band}.")
        key = (row["metric"], row["panel"])
        grouped.setdefault(key, {})[(row["row_network"], row["col_network"])] = float(row["value"])

    for metric in ("AEC", "AEC-orth"):
        for panel in ("Converter", "Non-converter"):
            key = (metric, panel)
            if key not in grouped:
                raise SystemExit(f"Missing panel {key} in {path}")
            matrix = np.zeros((len(NETWORK_ORDER), len(NETWORK_ORDER)), dtype=float)
            for i, row_network in enumerate(NETWORK_ORDER):
                for j, col_network in enumerate(NETWORK_ORDER):
                    lookup = (row_network, col_network)
                    if lookup not in grouped[key]:
                        raise SystemExit(f"Missing matrix cell {lookup} for panel {key} in {path}")
                    matrix[i, j] = grouped[key][lookup]
            if not np.allclose(matrix, matrix.T, atol=1e-10, equal_nan=True):
                raise SystemExit(f"Matrix {key} in {path} is not symmetric.")
            matrices[key] = matrix

    for metric in ("AEC", "AEC-orth"):
        matrices[(metric, "difference")] = matrices[(metric, "Converter")] - matrices[(metric, "Non-converter")]
    return matrices


def highlight_temp_par(ax: plt.Axes, index: int) -> None:
    accent = "#8b1e3f"
    ax.add_patch(
        patches.Rectangle(
            (-0.5, index - 0.5),
            len(NETWORK_ORDER),
            1.0,
            fill=False,
            linewidth=2.2,
            edgecolor=accent,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (index - 0.5, -0.5),
            1.0,
            len(NETWORK_ORDER),
            fill=False,
            linewidth=2.2,
            edgecolor=accent,
        )
    )
    for tick_index, label in enumerate(ax.get_xticklabels()):
        if tick_index == index:
            label.set_fontweight("bold")
            label.set_color(accent)
    for tick_index, label in enumerate(ax.get_yticklabels()):
        if tick_index == index:
            label.set_fontweight("bold")
            label.set_color(accent)


def draw_heatmap_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    temp_par_index: int,
):
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xticks(np.arange(len(NETWORK_ORDER)))
    ax.set_yticks(np.arange(len(NETWORK_ORDER)))
    ax.set_xticklabels([DISPLAY_LABELS[label] for label in NETWORK_ORDER], rotation=35, ha="right")
    ax.set_yticklabels([DISPLAY_LABELS[label] for label in NETWORK_ORDER])
    ax.tick_params(length=0)
    highlight_temp_par(ax, temp_par_index)
    return image


def build_network_heatmaps(band: str, matrices: dict[tuple[str, str], np.ndarray], output_dir: Path) -> tuple[Path, Path]:
    temp_par_index = NETWORK_ORDER.index("TempPar")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    fig.suptitle(f"{band.title()}-band network connectivity heatmaps", y=1.02, fontweight="bold")

    for row_index, metric in enumerate(("AEC", "AEC-orth")):
        converter = matrices[(metric, "Converter")]
        non_converter = matrices[(metric, "Non-converter")]
        difference = matrices[(metric, "difference")]

        mean_vmin = float(min(np.nanmin(converter), np.nanmin(non_converter)))
        mean_vmax = float(max(np.nanmax(converter), np.nanmax(non_converter)))
        diff_limit = float(np.nanmax(np.abs(difference)))
        diff_limit = diff_limit if diff_limit > 0 else 1e-9

        image_mean = draw_heatmap_panel(
            axes[row_index, 0],
            converter,
            f"{metric} Converter",
            cmap="YlGnBu",
            vmin=mean_vmin,
            vmax=mean_vmax,
            temp_par_index=temp_par_index,
        )
        draw_heatmap_panel(
            axes[row_index, 1],
            non_converter,
            f"{metric} Non-converter",
            cmap="YlGnBu",
            vmin=mean_vmin,
            vmax=mean_vmax,
            temp_par_index=temp_par_index,
        )
        image_diff = draw_heatmap_panel(
            axes[row_index, 2],
            difference,
            f"{metric} difference (Converter - Non-converter)",
            cmap="BrBG_r",
            vmin=-diff_limit,
            vmax=diff_limit,
            temp_par_index=temp_par_index,
        )

        colorbar_mean = fig.colorbar(image_mean, ax=axes[row_index, :2], fraction=0.030, pad=0.03)
        colorbar_mean.set_label(f"{metric} connectivity (unitless)")
        colorbar_diff = fig.colorbar(image_diff, ax=axes[row_index, 2], fraction=0.046, pad=0.04)
        colorbar_diff.set_label("Group difference")

    return save_pdf_and_png(fig, output_dir, f"fig_network_heatmaps_{band}")


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    title_size: float = 13.0,
    body_size: float = 11.0,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color="#182026")
    ax.text(x + w / 2, y + h * 0.25, body, ha="center", va="center", fontsize=body_size, color="#364047")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.7,
            color="#4f5b66",
            shrinkA=4,
            shrinkB=4,
        )
    )


def build_pipeline_overview(output_dir: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(18.5, 10.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(patches.Rectangle((0.03, 0.08), 0.28, 0.82, facecolor="#edf3fb", edgecolor="#b9c9de", linewidth=1.5))
    ax.add_patch(patches.Rectangle((0.36, 0.08), 0.61, 0.82, facecolor="#fcf4e8", edgecolor="#dbc29b", linewidth=1.5))

    ax.text(0.05, 0.885, "A. Upstream lab workflow", fontsize=18.0, fontweight="bold", color="#17324d")
    ax.text(0.40, 0.885, "B. Thesis repository workflow", fontsize=18.0, fontweight="bold", color="#6e3b00")

    left_x = 0.06
    left_w = 0.22
    left_h = 0.105
    left_ys = [0.72, 0.575, 0.43, 0.285, 0.14]
    left_steps = [
        ("Sensor-space MEG", "Recorded data before\nsource analysis"),
        ("tSSS and cleaning", "Noise suppression, artifact handling,\nand trial cleaning"),
        ("Individual MRI +\nco-registration", "MEG sensors aligned to\neach subject anatomy"),
        ("MNE source\nreconstruction", "Source-space activity estimated\non cortical ROIs"),
        ("ROI time-series export", "Source-space ROI trials\ndelivered to this repo"),
    ]

    for idx, (y, (title, body)) in enumerate(zip(left_ys, left_steps, strict=True), start=1):
        box(ax, (left_x, y), (left_w, left_h), title, body, facecolor="#f8fbff", edgecolor="#6d88a8", title_size=14.0, body_size=11.1)
        ax.text(left_x - 0.025, y + left_h / 2, str(idx), ha="center", va="center", fontsize=12, fontweight="bold", color="#17324d")
        if idx < len(left_steps):
            arrow(ax, (left_x + left_w / 2, y), (left_x + left_w / 2, left_ys[idx] + left_h))

    right_w = 0.16
    right_h = 0.105
    right_positions = [
        (0.41, 0.72),
        (0.61, 0.72),
        (0.81, 0.72),
        (0.41, 0.54),
        (0.61, 0.54),
        (0.81, 0.54),
        (0.41, 0.36),
        (0.61, 0.36),
        (0.81, 0.36),
    ]
    right_steps = [
        ("4 s crop", "Central analysis\nwindow"),
        ("Band-pass\nfiltering", "Band-limited ROI\nsignals"),
        ("Hilbert\ntransform", "Analytic signal\n$z_i(t)$"),
        ("Amplitude\nenvelopes", r"$A_i(t)=|z_i(t)|$"),
        ("AEC /\nAEC-orth", "Trial-wise envelope\ncoupling"),
        ("ROI×ROI\nmatrices", "One subject matrix\nper band"),
        ("Network\naggregation", "102 ROIs ->\n9 networks"),
        ("H1 / H2 / H3", "TempPar composites\n+ gap"),
        ("Welch + Holm\n+ max-T", "Final group\ninference"),
    ]

    for idx, ((x, y), (title, body)) in enumerate(zip(right_positions, right_steps, strict=True), start=1):
        box(ax, (x, y), (right_w, right_h), title, body, facecolor="#fffaf2", edgecolor="#b57b2a", title_size=13.2, body_size=11.1)
        ax.text(x - 0.016, y + right_h / 2, f"B{idx}", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#6e3b00")

    flow_pairs = [
        ((0.57, 0.772), (0.61, 0.772)),
        ((0.77, 0.772), (0.81, 0.772)),
        ((0.89, 0.72), (0.89, 0.645)),
        ((0.81, 0.592), (0.77, 0.592)),
        ((0.61, 0.592), (0.57, 0.592)),
        ((0.49, 0.54), (0.49, 0.465)),
        ((0.57, 0.412), (0.61, 0.412)),
        ((0.77, 0.412), (0.81, 0.412)),
    ]
    for start, end in flow_pairs:
        arrow(ax, start, end)

    ax.add_patch(FancyArrowPatch((0.28, 0.19), (0.39, 0.74), arrowstyle="-|>", mutation_scale=18, linewidth=2.0, color="#5c6f82"))
    ax.text(0.312, 0.49, "Input to this repo", rotation=74, ha="center", va="center", fontsize=12, color="#4f5b66", fontweight="bold")

    fig.suptitle("Pipeline from delivered source-space ROI trials to thesis connectivity results", y=0.975, fontweight="bold")
    return save_pdf_and_png(fig, output_dir, "fig_pipeline_overview")


def toy_signal_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 4.0, 2000, endpoint=False)
    slow_1 = 1.0 + 0.35 * np.sin(2 * np.pi * 0.30 * time) + 0.10 * np.cos(2 * np.pi * 0.08 * time)
    slow_2 = 0.92 + 0.30 * np.sin(2 * np.pi * 0.30 * time + 0.55) + 0.08 * np.cos(2 * np.pi * 0.10 * time + 0.20)
    carrier_1 = np.sin(2 * np.pi * 10.0 * time + 0.35 * np.sin(2 * np.pi * 0.45 * time))
    carrier_2 = np.sin(2 * np.pi * 10.0 * time + 0.55 + 0.22 * np.sin(2 * np.pi * 0.35 * time))
    x_i = slow_1 * carrier_1
    x_k = slow_2 * carrier_2
    return time, x_i, x_k


def orthogonalized_component(x_ref: np.ndarray, x_other: np.ndarray) -> np.ndarray:
    z_ref = hilbert(x_ref)
    z_other = hilbert(x_other)
    phase_ref = np.conj(z_ref / np.maximum(np.abs(z_ref), 1e-12))
    return np.imag(z_other * phase_ref)


def add_illustrative_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.02, 0.928, text, ha="left", va="top", **NOTE_KWARGS)


def build_envelope_extraction(output_dir: Path) -> tuple[Path, Path]:
    time, x_i, x_k = toy_signal_pair()
    plot_mask = (time >= 0.12) & (time <= 3.88)
    plot_t = time[plot_mask]
    plot_x_i = x_i[plot_mask]
    plot_x_k = x_k[plot_mask]
    a_i = np.abs(hilbert(x_i))[plot_mask]
    a_k = np.abs(hilbert(x_k))[plot_mask]
    aec_value = float(np.corrcoef(a_i, a_k)[0, 1])

    fig, axes = plt.subplots(3, 1, figsize=(15.5, 10.5), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.88, hspace=0.18)
    fig.suptitle("Illustrative Hilbert envelope extraction for AEC", y=0.97, fontweight="bold")
    add_illustrative_note(fig, "Illustrative toy signals; not empirical data")

    axes[0].plot(plot_t, plot_x_i, color="#355c7d", linewidth=1.4, label="Filtered ROI signal")
    axes[0].plot(plot_t, a_i, color="#c06c84", linewidth=2.0, label=r"Envelope $A_i(t)=|z_i(t)|$")
    axes[0].plot(plot_t, -a_i, color="#c06c84", linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel("ROI i")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_title("1. Filtered oscillatory signal and its Hilbert amplitude envelope", loc="left", fontsize=16, pad=14)

    axes[1].plot(plot_t, plot_x_k, color="#6c8e3a", linewidth=1.4)
    axes[1].plot(plot_t, a_k, color="#f08a24", linewidth=2.0)
    axes[1].plot(plot_t, -a_k, color="#f08a24", linewidth=1.2, alpha=0.85)
    axes[1].set_ylabel("ROI k")
    axes[1].set_title("2. A second ROI produces its own envelope", loc="left", fontsize=16, pad=14)

    axes[2].plot(plot_t, a_i, color="#c06c84", linewidth=2.2, label=r"$A_i(t)$")
    axes[2].plot(plot_t, a_k, color="#f08a24", linewidth=2.2, label=r"$A_k(t)$")
    axes[2].fill_between(plot_t, a_i, a_k, color="#eadcc8", alpha=0.35)
    axes[2].set_ylabel("Envelope")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(frameon=False, loc="upper right")
    axes[2].set_title("3. Pearson correlation between the two envelopes gives AEC", loc="left", fontsize=16, pad=14)
    axes[2].text(
        0.02,
        0.08,
        rf"AEC = corr($A_i$, $A_k$) = {aec_value:.2f}",
        transform=axes[2].transAxes,
        fontsize=15,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.98},
    )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18, linewidth=0.6)

    return save_pdf_and_png(fig, output_dir, "fig_envelope_extraction")


def build_aec_aecorth_concept(output_dir: Path) -> tuple[Path, Path]:
    time = np.linspace(0.0, 4.0, 2000, endpoint=False)
    plot_mask = (time >= 0.12) & (time <= 3.88)

    shared_env = 1.0 + 0.25 * np.sin(2 * np.pi * 0.25 * time) + 0.06 * np.sin(2 * np.pi * 0.07 * time)
    secondary_env = 0.95 + 0.18 * np.sin(2 * np.pi * 0.25 * time + 0.7) + 0.05 * np.sin(2 * np.pi * 0.07 * time + 0.3)
    x_i = shared_env * np.sin(2 * np.pi * 10.0 * time + 0.2 * np.sin(2 * np.pi * 0.3 * time))
    x_k = 0.75 * shared_env * np.sin(2 * np.pi * 10.0 * time + 0.15) + 0.35 * secondary_env * np.sin(2 * np.pi * 10.0 * time - 1.0)

    plot_t = time[plot_mask]
    plot_x_i = x_i[plot_mask]
    plot_x_k = x_k[plot_mask]
    a_i = np.abs(hilbert(x_i))[plot_mask]
    a_k = np.abs(hilbert(x_k))[plot_mask]
    aec_value, aec_orth_value, gap_value = compute_gap(x_i, x_k)
    x_k_orth_i = orthogonalized_component(x_i, x_k)[plot_mask]
    a_k_orth_i = np.abs(hilbert(x_k_orth_i))

    fig = plt.figure(figsize=(16, 10.6), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.88, wspace=0.12, hspace=0.18)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.15, 1.0])
    ax_signals = fig.add_subplot(grid[0, 0])
    ax_aec = fig.add_subplot(grid[0, 1])
    ax_orth = fig.add_subplot(grid[1, 0])
    ax_summary = fig.add_subplot(grid[1, 1])

    fig.suptitle("Illustrative AEC, AEC-orth, and gap schematic", y=0.97, fontweight="bold")
    add_illustrative_note(fig, "Illustrative toy signals; not empirical data")

    ax_signals.plot(plot_t, plot_x_i, color="#355c7d", linewidth=1.5, label="ROI i")
    ax_signals.plot(plot_t, plot_x_k, color="#6c8e3a", linewidth=1.5, alpha=0.88, label="ROI k")
    ax_signals.set_title("1. Filtered ROI signals can share a zero-lag component", loc="left", fontsize=16, pad=14)
    ax_signals.set_ylabel("Signal")
    ax_signals.legend(frameon=False, loc="upper right")
    ax_signals.spines[["top", "right"]].set_visible(False)
    ax_signals.grid(alpha=0.18, linewidth=0.6)

    ax_aec.plot(plot_t, a_i, color="#c06c84", linewidth=2.2, label=r"$A_i(t)$")
    ax_aec.plot(plot_t, a_k, color="#f08a24", linewidth=2.2, label=r"$A_k(t)$")
    ax_aec.fill_between(plot_t, a_i, a_k, color="#eadcc8", alpha=0.35)
    ax_aec.set_title("2. Envelope correlation gives AEC", loc="left", fontsize=16, pad=14)
    ax_aec.set_ylabel("Envelope")
    ax_aec.legend(frameon=False, loc="upper right")
    ax_aec.text(
        0.03,
        0.08,
        rf"AEC = corr($A_i$, $A_k$) = {aec_value:.2f}",
        transform=ax_aec.transAxes,
        fontsize=14,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.98},
    )
    ax_aec.spines[["top", "right"]].set_visible(False)
    ax_aec.grid(alpha=0.18, linewidth=0.6)

    ax_orth.plot(plot_t, a_i, color="#c06c84", linewidth=2.2, label=r"$A_i(t)$")
    ax_orth.plot(plot_t, a_k_orth_i, color="#3f88c5", linewidth=2.0, label=r"$|z_{k\perp i}(t)|$")
    ax_orth.fill_between(plot_t, a_i, a_k_orth_i, color="#dce8f5", alpha=0.45)
    ax_orth.set_title("3. Orthogonalization reduces shared zero-lag dependence", loc="left", fontsize=16, pad=14)
    ax_orth.set_xlabel("Time (s)")
    ax_orth.set_ylabel("Envelope")
    ax_orth.legend(frameon=False, loc="upper right")
    ax_orth.text(
        0.03,
        0.08,
        rf"AEC-orth = {aec_orth_value:.2f}",
        transform=ax_orth.transAxes,
        fontsize=14,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.98},
    )
    ax_orth.spines[["top", "right"]].set_visible(False)
    ax_orth.grid(alpha=0.18, linewidth=0.6)

    bar_colors = ["#f08a24", "#3f88c5", "#8b1e3f"]
    values = [aec_value, aec_orth_value, gap_value]
    labels = ["AEC", "AEC-orth", "gap"]
    ax_summary.bar(labels, values, color=bar_colors, width=0.62)
    ax_summary.axhline(0.0, color="#444444", linewidth=1.0)
    ax_summary.set_title("4. The gap is AEC - AEC-orth", loc="left", fontsize=16, pad=14)
    ax_summary.set_ylabel("Illustrative value")
    ax_summary.set_ylim(0.0, max(values) + 0.22)
    ax_summary.text(
        0.03,
        0.06,
        rf"gap = AEC - AEC-orth = {gap_value:.2f}",
        transform=ax_summary.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.98},
    )
    for xpos, value in enumerate(values):
        ax_summary.text(xpos, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=13)
    ax_summary.spines[["top", "right"]].set_visible(False)
    ax_summary.grid(axis="y", alpha=0.18, linewidth=0.6)

    return save_pdf_and_png(fig, output_dir, "fig_aec_aecorth_concept")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    alpha_csv = require_existing_file(Path(args.heatmap_alpha_csv))
    beta_csv = require_existing_file(Path(args.heatmap_beta_csv))

    apply_thesis_style()
    alpha_matrices = load_heatmap_export(alpha_csv, expected_band="alpha")
    beta_matrices = load_heatmap_export(beta_csv, expected_band="beta")

    outputs: list[Path] = []
    outputs.extend(build_pipeline_overview(output_dir))
    outputs.extend(build_network_heatmaps("alpha", alpha_matrices, output_dir))
    outputs.extend(build_network_heatmaps("beta", beta_matrices, output_dir))
    outputs.extend(build_aec_aecorth_concept(output_dir))
    outputs.extend(build_envelope_extraction(output_dir))

    print("Generated thesis figures:")
    for path in outputs:
        print(path)

    print("\nCommands to regenerate:")
    print(
        "  MPLCONFIGDIR=.mplconfig "
        "/Users/enia.ardid/neural_data_analysis/.venv/bin/python "
        "scripts/figures/make_tfg_figures.py"
    )


if __name__ == "__main__":
    main()
