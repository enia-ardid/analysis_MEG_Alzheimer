"""Build repository presentation figures from versioned aggregate study results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
INK = "#173247"
TEAL = "#007f75"
MUTED = "#607987"
LIGHT = "#edf4f6"
LABELS = {
    "H1_AEC": "H1   Alpha · AEC",
    "H1_AECorth": "H1   Alpha · AEC-orth",
    "H2_AEC": "H2   Beta · AEC",
    "H2_AECorth": "H2   Beta · AEC-orth",
    "H3_gap_full": "H3   Alpha gap · full",
    "H3_gap_inter": "H3   Alpha gap · inter-network",
}


def save_figure(fig: plt.Figure, destination: Path) -> None:
    fig.savefig(destination.with_suffix(".png"), dpi=160, facecolor=fig.get_facecolor())
    svg_path = destination.with_suffix(".svg")
    fig.savefig(svg_path, facecolor=fig.get_facecolor(), metadata={"Date": None})
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def build_endpoint_effects(source: Path, output_dir: Path) -> None:
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [row["endpoint_id"] for row in rows] != list(LABELS):
        raise ValueError("Expected the six thesis endpoints in H1–H3 order")

    fig = plt.figure(figsize=(11.6, 5.5), facecolor="white")
    ax = fig.add_axes((0.31, 0.23, 0.46, 0.58))
    p_ax = fig.add_axes((0.81, 0.23, 0.13, 0.58), sharey=ax)
    fig.text(0.04, 0.925, "Connectivity differences across six endpoints", fontsize=18, weight="bold", color=INK)
    fig.text(0.04, 0.865, "Thesis cohort · 35 converters and 35 non-converters", fontsize=11, color=MUTED)
    for bottom, top in ((3.5, 5.5), (-0.5, 1.5)):
        ax.axhspan(bottom, top, color=LIGHT, zorder=0)
        p_ax.axhspan(bottom, top, color=LIGHT, zorder=0)
    ax.axvline(0, color=MUTED, linewidth=1.1)
    for index, row in enumerate(rows):
        y = 5 - index
        effect = float(row["cohen_d"])
        p_max_t = float(row["p_max_t"])
        survives = row["holm_significant"] == "True" and p_max_t < 0.05
        ax.scatter(effect, y, s=100, facecolor=TEAL if survives else "white", edgecolor=TEAL if survives else MUTED, linewidth=1.7, zorder=3)
        ax.text(effect + 0.024, y, f"{effect:.3f}".replace("-", "−"), va="center", fontsize=11, color=INK)
        p_ax.text(0.5, y, f"{p_max_t:.3f}", ha="center", va="center", color=TEAL if survives else MUTED, fontsize=12, weight="bold" if survives else "normal")
    ax.set_xlim(-0.70, 0.04)
    ax.set_ylim(-0.65, 5.65)
    ax.set_yticks(list(range(5, -1, -1)), list(LABELS.values()))
    ax.set_xticks([-0.6, -0.4, -0.2, 0.0])
    ax.set_xlabel("Cohen's d  (Converter − Non-converter)", color=INK, labelpad=12, fontsize=11)
    ax.tick_params(axis="y", length=0, pad=14, labelsize=11, colors=INK)
    ax.tick_params(axis="x", length=0, labelsize=10, colors=MUTED, pad=8)
    ax.grid(axis="x", color="#d8e3e8", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    p_ax.set_xlim(0, 1)
    p_ax.set_axis_off()
    p_ax.text(0.5, 1.03, "max-T p", transform=p_ax.transAxes, ha="center", fontsize=11, color=INK)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=TEAL, markeredgecolor=TEAL, markersize=7, label="Survives Holm and max-T"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=MUTED, markersize=7, label="Does not survive both corrections"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.035, 0.05), ncol=2, frameon=False, fontsize=10, labelcolor=INK)
    fig.text(0.04, 0.025, "Source: thesis Table 2 · Point estimates; confidence intervals not shown · max-T: 10,000 permutations", fontsize=9, color=MUTED)
    save_figure(fig, output_dir / "thesis-endpoint-effects")


def build_social_preview(output_dir: Path) -> None:
    fig = plt.figure(figsize=(8, 4), facecolor="#102d40")
    fig.text(0.055, 0.87, "NEUROIMAGING  /  PYTHON", color="#8ad6cf", fontsize=10, weight="bold")
    fig.text(0.055, 0.65, "MEG", color="white", fontsize=43, weight="bold")
    fig.text(0.055, 0.52, "connectivity", color="white", fontsize=28, weight="bold")
    fig.text(0.055, 0.37, "Alzheimer's disease\nprogression", color="#d0e3ea", fontsize=17, linespacing=1.45)
    fig.text(0.055, 0.18, "70 participants · 9 networks\n6 statistical endpoints", color="#8ad6cf", fontsize=11, linespacing=1.6)
    fig.text(0.055, 0.065, "Enia Ardid  ·  Physics BSc  ·  UCM", color="#b1c9d4", fontsize=10)
    panel = FancyBboxPatch((0.43, 0.14), 0.545, 0.72, boxstyle="round,pad=0.01,rounding_size=0.018", facecolor="white", edgecolor="none", transform=fig.transFigure, zorder=0)
    fig.add_artist(panel)
    fig.text(0.455, 0.81, "ALPHA-BAND NETWORK CONNECTIVITY", color=INK, fontsize=8, weight="bold")
    ax = fig.add_axes((0.445, 0.19, 0.515, 0.58))
    ax.imshow(plt.imread(ROOT / "docs" / "assets" / "thesis-alpha-networks.png"))
    ax.set_axis_off()
    save_figure(fig, output_dir / "social-preview")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "assets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path", "svg.hashsalt": "meg-study-assets"})
    build_endpoint_effects(ROOT / "supplementary" / "source_tables" / "table_thesis_endpoints.csv", args.output_dir)
    build_social_preview(args.output_dir)


if __name__ == "__main__":
    main()
