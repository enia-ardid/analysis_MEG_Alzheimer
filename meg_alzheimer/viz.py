"""Plotting utilities for subject-level and summary connectivity figures.

The project keeps visualisation helpers in one small module so that analysis
code does not need to manage figure styling, thresholding, or file writing
details inline. The functions here are intentionally simple wrappers around
Matplotlib and NetworkX. They are designed for reproducible reporting rather
than interactive dashboards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_matrix(
    C: np.ndarray,
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    outfile: str | Path | None = None,
) -> None:
    """Render a connectivity matrix as a heatmap and optionally save it."""

    plt.figure(figsize=(6, 5))
    im = plt.imshow(C, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("ROI")
    plt.ylabel("ROI")
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()


def edges_top_percent(C: np.ndarray, top: float = 0.05) -> tuple[np.ndarray, float]:
    """Return a symmetric mask containing only the strongest fraction of edges."""

    iu = np.triu_indices_from(C, 1)
    vals = np.asarray(C, dtype=float)[iu]
    thr = float(np.quantile(vals, 1.0 - top))
    mask = np.zeros_like(C, dtype=bool)
    mask[iu] = vals >= thr
    mask = mask | mask.T
    return mask, thr


def graph_thresholded(
    C: np.ndarray,
    roi_labels: Sequence[str] | None = None,
    top: float = 0.05,
    title: str = "",
    outfile: str | Path | None = None,
    layout: str = "circular",
) -> None:
    """Plot a thresholded graph view of a connectivity matrix.

    The graph keeps only the strongest edges according to ``top`` and scales
    node size by weighted degree so that high-strength regions are easier to
    identify visually.
    """

    C = np.asarray(C, dtype=float)
    mask, thr = edges_top_percent(C, top=top)
    graph = nx.Graph()
    n_roi = C.shape[0]
    for node in range(n_roi):
        label = roi_labels[node] if roi_labels is not None and node < len(roi_labels) else str(node)
        graph.add_node(node, label=label, strength=float(C[node].sum()))
    for i in range(n_roi):
        for j in range(i + 1, n_roi):
            if mask[i, j]:
                graph.add_edge(i, j, weight=float(C[i, j]))

    pos = nx.circular_layout(graph) if layout == "circular" else nx.spring_layout(graph, seed=0)
    strengths = np.array([graph.nodes[node]["strength"] for node in graph.nodes()], dtype=float)
    denom = np.ptp(strengths) if strengths.size else 0.0
    if denom < 1e-12:
        node_sizes = np.full_like(strengths, 150.0, dtype=float)
    else:
        node_sizes = 250.0 * (strengths - strengths.min()) / denom + 80.0

    plt.figure(figsize=(7, 7))
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes)
    widths = [2.0 * edge_data["weight"] for _, _, edge_data in graph.edges(data=True)]
    nx.draw_networkx_edges(graph, pos, width=widths, alpha=0.7)
    if roi_labels is not None:
        nx.draw_networkx_labels(graph, pos, font_size=7)
    plt.title(f"{title}\n(top {int(top * 100)}%, thr={thr:.3f})")
    plt.axis("off")
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()


def violin_global(
    values_by_group: Sequence[np.ndarray],
    labels_groups: Sequence[str],
    title: str = "",
    outfile: str | Path | None = None,
) -> None:
    """Plot a compact violin summary for one scalar value per subject."""

    plt.figure(figsize=(6, 4))
    plt.violinplot(values_by_group, showmeans=True, showmedians=False)
    plt.xticks(np.arange(1, len(labels_groups) + 1), labels_groups)
    plt.title(title)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()
