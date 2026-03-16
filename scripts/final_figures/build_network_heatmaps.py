#!/usr/bin/env python3
from __future__ import annotations

"""Build final network-level heatmaps for alpha and beta connectivity.

The script generates two parallel representations from the same subject-level
ROI matrices:

1. ``schaefer17``: the exact 17-network subdivision present in the ROI labels
2. ``collapsed9``: the coarser network family used by the main pipeline

Both are computed directly from the saved subject connectivity matrices so the
figures are fully reproducible from the current repository state.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from meg_alzheimer.atlas import DEFAULT_NETWORK_MAP, get_network_prefix, mean_inter, mean_intra


GROUP_A = "Converter"
GROUP_B = "Non-converter"
METRICS = ("AEC", "AEC-orth")
BANDS = ("alpha", "beta")

S17_ORDER = [
    "VisCent",
    "VisPeri",
    "SomMotA",
    "SomMotB",
    "DorsAttnA",
    "DorsAttnB",
    "SalVentAttnA",
    "SalVentAttnB",
    "LimbicA",
    "LimbicB",
    "TempPar",
    "ContA",
    "ContB",
    "ContC",
    "DefaultA",
    "DefaultB",
    "DefaultC",
]

COLLAPSED9_ORDER = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final network-level alpha/beta heatmaps.")
    parser.add_argument(
        "--subjects-root",
        default="outputs_full_cohort/subjects",
        help="Folder containing per-subject connectivity_matrices.npz and metadata.json files.",
    )
    parser.add_argument("--output-dir", default="figures/final", help="Destination folder for figures and matrix CSVs.")
    parser.add_argument(
        "--captions-path",
        default="captions_figures.md",
        help="Markdown file where suggested figure captions will be written.",
    )
    return parser.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#111111",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _iter_subject_dirs(subjects_root: Path) -> list[Path]:
    return sorted(path for path in subjects_root.glob("*/*") if path.is_dir())


def _load_subject(subject_dir: Path) -> dict[str, Any]:
    with open(subject_dir / "metadata.json") as f:
        meta = json.load(f)
    matrices = np.load(subject_dir / "connectivity_matrices.npz")
    return {"subject_dir": subject_dir, "metadata": meta, "matrices": matrices}


def _raw_network_labels(roi_labels: Iterable[str]) -> list[str]:
    labels = [get_network_prefix(label) for label in roi_labels]
    return ["Background" if label.startswith("Background+FreeSurfer") else label for label in labels]


def _collapsed_network_labels(roi_labels: Iterable[str]) -> list[str]:
    raw_labels = _raw_network_labels(roi_labels)
    collapsed: list[str] = []
    for label in raw_labels:
        if label == "Background":
            collapsed.append("Background")
        else:
            collapsed.append(DEFAULT_NETWORK_MAP.get(label, label))
    return collapsed


def _validate_atlas(subject_payloads: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    first_meta = subject_payloads[0]["metadata"]
    roi_labels = first_meta["roi_labels"]
    raw_labels = _raw_network_labels(roi_labels)
    collapsed_labels = _collapsed_network_labels(roi_labels)
    for payload in subject_payloads[1:]:
        meta = payload["metadata"]
        if meta["roi_labels"] != roi_labels:
            raise ValueError("ROI labels differ across subjects. Network heatmaps require a common ROI order.")
    return roi_labels, raw_labels, collapsed_labels


def _network_matrix(
    roi_matrix: np.ndarray,
    roi_networks: list[str],
    network_order: list[str],
) -> np.ndarray:
    matrix = np.full((len(network_order), len(network_order)), np.nan, dtype=float)
    masks = {network: np.array([label == network for label in roi_networks], dtype=bool) for network in network_order}
    for i, network_a in enumerate(network_order):
        for j, network_b in enumerate(network_order):
            if i == j:
                matrix[i, j] = mean_intra(roi_matrix, masks[network_a])
            elif i < j:
                value = mean_inter(roi_matrix, masks[network_a], masks[network_b])
                matrix[i, j] = value
                matrix[j, i] = value
    return matrix


def _group_mean_matrix(
    subject_payloads: list[dict[str, Any]],
    band: str,
    metric: str,
    group: str,
    roi_networks: list[str],
    network_order: list[str],
) -> np.ndarray:
    matrices = []
    key = f"{band}__{metric}"
    for payload in subject_payloads:
        if payload["metadata"]["group"] != group:
            continue
        roi_matrix = np.asarray(payload["matrices"][key], dtype=float)
        matrices.append(_network_matrix(roi_matrix, roi_networks=roi_networks, network_order=network_order))
    if not matrices:
        raise ValueError(f"No matrices found for group={group}, band={band}, metric={metric}.")
    return np.nanmean(np.stack(matrices, axis=0), axis=0)


def _plot_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    temp_par_index: int | None,
) -> Any:
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)

    for idx, tick in enumerate(ax.get_xticklabels()):
        if temp_par_index is not None and idx == temp_par_index:
            tick.set_fontweight("bold")
    for idx, tick in enumerate(ax.get_yticklabels()):
        if temp_par_index is not None and idx == temp_par_index:
            tick.set_fontweight("bold")

    if temp_par_index is not None:
        lo = temp_par_index - 0.5
        hi = temp_par_index + 0.5
        ax.axhline(lo, color="#111111", linewidth=1.2)
        ax.axhline(hi, color="#111111", linewidth=1.2)
        ax.axvline(lo, color="#111111", linewidth=1.2)
        ax.axvline(hi, color="#111111", linewidth=1.2)
    return im


def _figure_size(n_networks: int) -> tuple[float, float]:
    if n_networks <= 9:
        return (12.2, 7.6)
    return (14.8, 8.8)


def _build_one_figure(
    output_dir: Path,
    resolution_label: str,
    band: str,
    network_order: list[str],
    group_matrices: dict[tuple[str, str], np.ndarray],
) -> tuple[Path, Path, Path]:
    temp_par_index = network_order.index("TempPar") if "TempPar" in network_order else None
    fig, axes = plt.subplots(2, 3, figsize=_figure_size(len(network_order)), constrained_layout=True)
    fig.suptitle(f"{band.title()} network connectivity heatmaps ({resolution_label})", fontsize=14)

    for row_idx, metric in enumerate(METRICS):
        conv = group_matrices[(metric, GROUP_A)]
        nonconv = group_matrices[(metric, GROUP_B)]
        diff = conv - nonconv

        mean_vmin = float(np.nanmin([conv.min(), nonconv.min()]))
        mean_vmax = float(np.nanmax([conv.max(), nonconv.max()]))
        diff_lim = float(np.nanmax(np.abs(diff)))
        diff_lim = diff_lim if diff_lim > 0 else 1e-9

        im_mean_1 = _plot_heatmap(
            axes[row_idx, 0],
            conv,
            network_order,
            f"{metric} {GROUP_A}",
            cmap="cividis",
            vmin=mean_vmin,
            vmax=mean_vmax,
            temp_par_index=temp_par_index,
        )
        _plot_heatmap(
            axes[row_idx, 1],
            nonconv,
            network_order,
            f"{metric} {GROUP_B}",
            cmap="cividis",
            vmin=mean_vmin,
            vmax=mean_vmax,
            temp_par_index=temp_par_index,
        )
        im_diff = _plot_heatmap(
            axes[row_idx, 2],
            diff,
            network_order,
            f"{metric} difference ({GROUP_A} - {GROUP_B})",
            cmap="RdBu_r",
            vmin=-diff_lim,
            vmax=diff_lim,
            temp_par_index=temp_par_index,
        )

        cbar_mean = fig.colorbar(im_mean_1, ax=axes[row_idx, :2], fraction=0.025, pad=0.02)
        cbar_mean.set_label(f"{metric} mean connectivity")
        cbar_diff = fig.colorbar(im_diff, ax=axes[row_idx, 2], fraction=0.046, pad=0.03)
        cbar_diff.set_label("Difference")

    suffix = "" if resolution_label == "schaefer17" else f"_{resolution_label}"
    png_path = output_dir / f"fig_network_heatmaps_{band}{suffix}.png"
    pdf_path = output_dir / f"fig_network_heatmaps_{band}{suffix}.pdf"
    csv_path = output_dir / f"fig_network_heatmaps_{band}_matrices_{resolution_label}.csv"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path, csv_path


def _matrix_export_rows(
    resolution_label: str,
    band: str,
    network_order: list[str],
    group_matrices: dict[tuple[str, str], np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        conv = group_matrices[(metric, GROUP_A)]
        nonconv = group_matrices[(metric, GROUP_B)]
        diff = conv - nonconv
        for panel, matrix in [
            (GROUP_A, conv),
            (GROUP_B, nonconv),
            ("difference", diff),
        ]:
            for i, row_network in enumerate(network_order):
                for j, col_network in enumerate(network_order):
                    rows.append(
                        {
                            "network_resolution": resolution_label,
                            "band": band,
                            "metric": metric,
                            "panel": panel,
                            "row_network": row_network,
                            "col_network": col_network,
                            "value": float(matrix[i, j]),
                            "aggregation_rule": (
                                "intra = mean upper triangle within network; "
                                "inter = mean of all ROI pairs between networks; "
                                "group mean = arithmetic mean across subjects"
                            ),
                        }
                    )
    return rows


def _write_aggregation_note(output_dir: Path, roi_labels: list[str], raw_labels: list[str], collapsed_labels: list[str]) -> Path:
    raw_counts = pd.Series(raw_labels).value_counts().sort_index()
    collapsed_counts = pd.Series(collapsed_labels).value_counts().sort_index()
    lines = [
        "# Network heatmap aggregation",
        "",
        "These figures are generated directly from each subject's saved ROI-by-ROI connectivity matrix.",
        "",
        "Aggregation rules:",
        "- intra-network cell: mean of the upper triangle of the ROI submatrix for that network",
        "- inter-network cell: mean of all ROI pairs connecting the two networks",
        "- group matrix: arithmetic mean of subject-level network matrices within group",
        "",
        "Two network resolutions are exported:",
        "- `schaefer17`: exact ROI prefix groups present in the Schaefer 100/17 atlas export",
        "- `collapsed9`: the coarser family used by the main pipeline (`Control`, `Default`, `DorsAttn`, etc.)",
        "",
        "TempPar is represented by a unique and unambiguous label in both resolutions.",
        "",
        "ROI counts per Schaefer-17 label:",
    ]
    lines.extend([f"- {network}: {int(count)} ROIs" for network, count in raw_counts.items() if network != "Background"])
    lines.extend(["", "ROI counts per collapsed-9 family:"])
    lines.extend([f"- {network}: {int(count)} ROIs" for network, count in collapsed_counts.items() if network != "Background"])
    lines.append("")
    path = output_dir / "network_heatmaps_aggregation.md"
    path.write_text("\n".join(lines))
    return path


def _upsert_captions(path: Path) -> None:
    sections = [
        (
            "## Figure: Alpha network heatmaps (Schaefer-17)",
            "Suggested caption: Group-mean alpha-band network connectivity matrices for Converters and "
            "Non-converters, together with the group difference (Converter minus Non-converter), shown at the "
            "exact Schaefer 17-network level. Panels are shown separately for AEC and AEC-orth. Network cells "
            "were obtained by averaging ROI-level connectivity within and between network blocks, and TempPar is "
            "highlighted on both axes for anatomical localization.",
        ),
        (
            "## Figure: Beta network heatmaps (Schaefer-17)",
            "Suggested caption: Group-mean beta-band network connectivity matrices for Converters and "
            "Non-converters, together with the group difference (Converter minus Non-converter), shown at the "
            "exact Schaefer 17-network level. Panels are shown separately for AEC and AEC-orth. Network cells "
            "were obtained by averaging ROI-level connectivity within and between network blocks, and TempPar is "
            "highlighted on both axes for anatomical localization.",
        ),
        (
            "## Figure: Alpha network heatmaps (collapsed-9)",
            "Suggested caption: Group-mean alpha-band network connectivity matrices using the collapsed 9-network "
            "family employed by the main pipeline. Panels show Converters, Non-converters, and the group "
            "difference for AEC and AEC-orth. These heatmaps provide the direct descriptive counterpart of the "
            "network summaries used downstream in the hypothesis-driven analysis.",
        ),
        (
            "## Figure: Beta network heatmaps (collapsed-9)",
            "Suggested caption: Group-mean beta-band network connectivity matrices using the collapsed 9-network "
            "family employed by the main pipeline. Panels show Converters, Non-converters, and the group "
            "difference for AEC and AEC-orth. These heatmaps provide the direct descriptive counterpart of the "
            "network summaries used downstream in the hypothesis-driven analysis.",
        ),
    ]

    existing = path.read_text() if path.exists() else ""
    blocks: list[str] = []
    text = existing
    for heading, caption in sections:
        block = f"{heading}\n\n{caption}\n"
        if heading in text:
            start = text.index(heading)
            next_idx = text.find("\n## ", start + 1)
            end = len(text) if next_idx == -1 else next_idx + 1
            text = text[:start] + block + text[end:]
        else:
            blocks.append(block)
    if blocks:
        text = text.rstrip() + ("\n\n" if text.strip() else "") + "\n\n".join(blocks) + "\n"
    path.write_text(text)


def main() -> None:
    args = parse_args()
    subjects_root = Path(args.subjects_root)
    output_dir = Path(args.output_dir)
    captions_path = Path(args.captions_path)

    subject_dirs = _iter_subject_dirs(subjects_root)
    if not subject_dirs:
        raise SystemExit(f"No subject folders found under {subjects_root}.")

    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_payloads = [_load_subject(subject_dir) for subject_dir in subject_dirs]
    roi_labels, raw_labels, collapsed_labels = _validate_atlas(subject_payloads)

    if "TempPar" not in raw_labels or "TempPar" not in collapsed_labels:
        raise SystemExit("TempPar was not found in one of the requested network resolutions.")

    if sorted(set(label for label in raw_labels if label != "Background")) != sorted(S17_ORDER):
        raise SystemExit(
            "The raw ROI prefixes do not match the expected Schaefer-17 labels. "
            "Refusing to build misleading exact-atlas heatmaps."
        )

    all_exports: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for resolution_label, roi_networks, network_order in [
        ("schaefer17", raw_labels, S17_ORDER),
        ("collapsed9", collapsed_labels, COLLAPSED9_ORDER),
    ]:
        for band in BANDS:
            group_matrices: dict[tuple[str, str], np.ndarray] = {}
            for metric in METRICS:
                for group in (GROUP_A, GROUP_B):
                    group_matrices[(metric, group)] = _group_mean_matrix(
                        subject_payloads=subject_payloads,
                        band=band,
                        metric=metric,
                        group=group,
                        roi_networks=roi_networks,
                        network_order=network_order,
                    )
            png_path, pdf_path, csv_path = _build_one_figure(
                output_dir=output_dir,
                resolution_label=resolution_label,
                band=band,
                network_order=network_order,
                group_matrices=group_matrices,
            )
            pd.DataFrame(
                _matrix_export_rows(
                    resolution_label=resolution_label,
                    band=band,
                    network_order=network_order,
                    group_matrices=group_matrices,
                )
            ).to_csv(csv_path, index=False)
            outputs.extend([png_path, pdf_path, csv_path])

    note_path = _write_aggregation_note(output_dir, roi_labels=roi_labels, raw_labels=raw_labels, collapsed_labels=collapsed_labels)
    outputs.append(note_path)
    _upsert_captions(captions_path)

    print(f"Subjects loaded: {len(subject_payloads)}")
    print(f"TempPar present in Schaefer-17: {'TempPar' in raw_labels}")
    print(f"TempPar present in collapsed-9: {'TempPar' in collapsed_labels}")
    print("Outputs:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
