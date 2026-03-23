#!/usr/bin/env python3
from __future__ import annotations

"""Build the manuscript pipeline overview figure.

The repository starts from Brainstorm `.mat` exports that already contain
source-space ROI time series. Sensor-space preprocessing and source
reconstruction are shown as upstream context rather than as steps executed by
the current codebase.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


TEXT = {
    "es": {
        "suptitle": "Flujo de análisis desde las exportaciones MEG en espacio fuente hasta los endpoints finales",
        "upstream_header": "Procesado previo externo al repositorio",
        "repo_header": "Flujo reproducible implementado en este repositorio",
        "upstream_note": "Las etapas 1-2 forman parte del preprocesado previo de los datos\nentregados y no se ejecutan en este proyecto.",
        "repo_note": "El análisis reproducible de este trabajo comienza en las series temporales ROI\nen espacio fuente ya exportadas.",
        "caption_default": (
            "Suggested caption (ES): Esquema del flujo de análisis utilizado en el manuscrito. "
            "El preprocesado en espacio sensor y la reconstrucción en espacio fuente se muestran como etapas previas externas al repositorio, "
            "ya que el conjunto de datos entregado ya contiene series temporales ROI exportadas en espacio fuente. "
            "El análisis reproducible implementado en este trabajo comienza en esas series ROI y continúa con el filtrado por bandas, "
            "la transformada de Hilbert, la extracción de envolventes, el cálculo ensayo a ensayo de AEC y AEC-orth, el promedio dentro de sujeto, "
            "el resumen ROI-red, la construcción de composites y la inferencia final de seis endpoints con Welch, Holm y max-T."
        ),
        "steps": [
            ("1", "MEG preprocesado\nen espacio sensor", "Limpieza y segmentación\nprevias al análisis en fuente", "upstream"),
            ("2", "Reconstrucción\nen espacio fuente", "Proyección desde sensores\na la corteza", "upstream"),
            ("3", "Series temporales ROI\nen espacio fuente", "102 ROIs de Schaefer\napiladas por ensayo en `.mat`", "input"),
            ("4", "Filtrado\npor bandas", "Delta, theta,\nalpha, beta y gamma", "repo"),
            ("5", "Señal analítica\nmediante Hilbert", "Representación compleja\npor ROI y ensayo", "repo"),
            ("6", "Envolventes\nde amplitud", "Extracción de la amplitud\ninstantánea", "repo"),
            ("7", "AEC y AEC-orth\npor ensayo", "Una matriz por ensayo,\nbanda y métrica", "repo"),
            ("8", "Promedio dentro\nde sujeto", "Promedio de matrices\nde ensayo a sujeto", "repo"),
            ("9", "Resumen\nROI -> red", "Medias intra-red\ne inter-red", "repo"),
            ("10", "Construcción de\ncomposites", "H1 y H2 centradas en TempPar\ny gap de H3", "repo"),
            ("11", "Contraste final\nde 6 endpoints", "Welch,\nHolm y max-T", "repo"),
        ],
    },
    "en": {
        "suptitle": "Analysis flow from source-space MEG exports to final endpoints",
        "upstream_header": "Upstream preprocessing outside the repository",
        "repo_header": "Reproducible workflow implemented in this repository",
        "upstream_note": "Steps 1-2 belong to the prior preparation pipeline of the delivered dataset\nand are not executed by this project.",
        "repo_note": "The reproducible analysis of this work begins from already exported\nsource-space ROI time series.",
        "caption_default": (
            "Suggested caption (EN): Overview of the analysis workflow used in the manuscript. "
            "Sensor-space preprocessing and source reconstruction are shown as upstream stages external to the repository, "
            "because the delivered dataset already consists of source-space ROI time series. "
            "The reproducible analysis implemented here begins at the ROI time-series level and proceeds through band filtering, "
            "Hilbert-based analytic signals, amplitude envelopes, trial-wise AEC and AEC-orth estimation, within-subject averaging, "
            "ROI-to-network summarization, hypothesis-composite construction, and final six-endpoint inference with Welch tests, Holm correction, and max-T permutation control."
        ),
        "steps": [
            ("1", "Preprocessed MEG\nin sensor space", "Cleaning and segmentation\nbefore source analysis", "upstream"),
            ("2", "Source\nreconstruction", "Projection from sensors\nto cortical source space", "upstream"),
            ("3", "Source-space ROI\ntime series", "102 Schaefer ROIs\nstacked trial-wise in `.mat`", "input"),
            ("4", "Band-pass\nfiltering", "Delta, theta,\nalpha, beta, gamma", "repo"),
            ("5", "Analytic signal\nvia Hilbert", "Complex representation\nper ROI and trial", "repo"),
            ("6", "Amplitude\nenvelopes", "Extraction of instantaneous\nenvelope magnitude", "repo"),
            ("7", "Trial-wise AEC\nand AEC-orth", "One matrix per trial,\nband, and metric", "repo"),
            ("8", "Within-subject\naveraging", "Average trial matrices\nto one matrix per subject", "repo"),
            ("9", "ROI to network\nsummary", "Within-network and\nbetween-network means", "repo"),
            ("10", "Hypothesis\ncomposites", "TempPar-centered H1/H2\nand the H3 gap", "repo"),
            ("11", "Final six-endpoint\ninference", "Welch,\nHolm, and max-T", "repo"),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the manuscript pipeline overview figure.")
    parser.add_argument("--output-dir", default="figures/final", help="Destination folder for figure files.")
    parser.add_argument(
        "--captions-path",
        default="captions_figures.md",
        help="Markdown file where the suggested captions will be written.",
    )
    parser.add_argument(
        "--language",
        choices=sorted(TEXT),
        default="es",
        help="Language used for fig_pipeline_overview.*",
    )
    parser.add_argument(
        "--export-english-copy",
        action="store_true",
        help="Also export fig_pipeline_overview_en.* for supplementary use.",
    )
    return parser.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _draw_box(ax: plt.Axes, x: float, y: float, w: float, h: float, step: tuple[str, str, str, str]) -> None:
    number, title, body, section = step
    face = {
        "upstream": "#ececec",
        "input": "#dcdcdc",
        "repo": "white",
    }[section]
    edge = {
        "upstream": "#4b4b4b",
        "input": "#3a3a3a",
        "repo": "#222222",
    }[section]
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.014",
        linewidth=1.15,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)

    number_box = Rectangle((x + 0.014, y + h + 0.006), 0.044, 0.034, linewidth=0.9, edgecolor=edge, facecolor="#d5d5d5")
    ax.add_patch(number_box)
    ax.text(x + 0.036, y + h + 0.023, number, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#111111")

    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=9.8, fontweight="bold", color="#111111")
    ax.text(x + w / 2, y + h * 0.26, body, ha="center", va="center", fontsize=8.0, color="#333333")


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], connectionstyle: str | None = None) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10.5,
        linewidth=1.0,
        color="#666666",
        shrinkA=5,
        shrinkB=5,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(patch)


def _upsert_captions(path: Path) -> None:
    blocks = {"## Figure: Pipeline overview": TEXT["en"]["caption_default"]}
    text = path.read_text() if path.exists() else ""
    for heading, caption in blocks.items():
        block = f"{heading}\n\n{caption}\n"
        if heading in text:
            start = text.index(heading)
            next_idx = text.find("\n## ", start + 1)
            end = len(text) if next_idx == -1 else next_idx + 1
            text = text[:start] + block + text[end:]
        else:
            text = text.rstrip() + ("\n\n" if text.strip() else "") + block
    path.write_text(text)


def _build_one(language: str, output_dir: Path, stem: str) -> None:
    spec = TEXT[language]

    fig, ax = plt.subplots(figsize=(18.2, 9.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Section backgrounds
    ax.add_patch(Rectangle((0.03, 0.69), 0.42, 0.18, facecolor="#f5f5f5", edgecolor="#cfcfcf", linewidth=1.0))
    ax.add_patch(Rectangle((0.03, 0.07), 0.94, 0.83, facecolor="#fcfcfc", edgecolor="#cfcfcf", linewidth=1.0))
    ax.text(0.04, 0.92, spec["upstream_header"], ha="left", va="center", fontsize=11, fontweight="bold", color="#222222")
    ax.text(0.49, 0.92, spec["repo_header"], ha="left", va="center", fontsize=11, fontweight="bold", color="#222222")

    w, h = 0.19, 0.16
    row1_y, row2_y, row3_y = 0.70, 0.44, 0.18
    row1_x = [0.04, 0.28, 0.52, 0.76]
    row2_x = [0.76, 0.52, 0.28, 0.04]
    row3_x = [0.08, 0.405, 0.73]

    steps = spec["steps"]
    positions: dict[int, tuple[float, float]] = {}

    for idx, x in enumerate(row1_x):
        _draw_box(ax, x, row1_y, w, h, steps[idx])
        positions[idx] = (x, row1_y)

    for idx, x in enumerate(row2_x, start=4):
        _draw_box(ax, x, row2_y, w, h, steps[idx])
        positions[idx] = (x, row2_y)

    for idx, x in enumerate(row3_x, start=8):
        _draw_box(ax, x, row3_y, w, h, steps[idx])
        positions[idx] = (x, row3_y)

    # Row 1 arrows: 1 -> 2 -> 3 -> 4
    for idx in range(3):
        x, y = positions[idx]
        nx, ny = positions[idx + 1]
        _arrow(ax, (x + w, y + h / 2), (nx, ny + h / 2))

    # Vertical down: 4 -> 5
    x4, y4 = positions[3]
    x5, y5 = positions[4]
    _arrow(ax, (x4 + w / 2, y4), (x5 + w / 2, y5 + h))

    # Row 2 arrows: 5 -> 6 -> 7 -> 8 (right to left)
    for idx in range(4, 7):
        x, y = positions[idx]
        nx, ny = positions[idx + 1]
        _arrow(ax, (x, y + h / 2), (nx + w, ny + h / 2))

    # Vertical down: 8 -> 9
    x8, y8 = positions[7]
    x9, y9 = positions[8]
    _arrow(ax, (x8 + w / 2, y8), (x9 + w / 2, y9 + h))

    # Row 3 arrows: 9 -> 10 -> 11
    for idx in range(8, 10):
        x, y = positions[idx]
        nx, ny = positions[idx + 1]
        _arrow(ax, (x + w, y + h / 2), (nx, ny + h / 2))

    ax.text(0.04, 0.682, spec["upstream_note"], ha="left", va="top", fontsize=8.55, color="#444444")
    ax.text(0.04, 0.08, spec["repo_note"], ha="left", va="bottom", fontsize=8.55, color="#444444")

    fig.suptitle(spec["suptitle"], fontsize=14, fontweight="bold", color="#111111", y=0.98)

    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")

    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    captions_path = Path(args.captions_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    _style()
    _build_one(args.language, output_dir=output_dir, stem="fig_pipeline_overview")
    if args.export_english_copy:
        _build_one("en", output_dir=output_dir, stem="fig_pipeline_overview_en")
    _upsert_captions(captions_path)


if __name__ == "__main__":
    main()
