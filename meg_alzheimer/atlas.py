from __future__ import annotations

"""Atlas and network-level summarization utilities.

The connectivity metrics are first computed at ROI level, yielding 102 x 102
 matrices for each subject, band, and metric. The scientific questions in this
project, however, are formulated at the level of large-scale networks such as
TempPar, DorsAttn, or Control. This module provides the bridge between those
two levels:

- read ROI labels from the Brainstorm atlas object
- map Schaefer label prefixes onto a smaller set of functional networks
- compute within-network and between-network averages from ROI matrices
"""

from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_NETWORK_MAP: Dict[str, str] = {
    "DefaultA": "Default",
    "DefaultB": "Default",
    "DefaultC": "Default",
    "ContA": "Control",
    "ContB": "Control",
    "ContC": "Control",
    "DorsAttnA": "DorsAttn",
    "DorsAttnB": "DorsAttn",
    "SalVentAttnA": "SalVentAttn",
    "SalVentAttnB": "SalVentAttn",
    "SomMotA": "SomMot",
    "SomMotB": "SomMot",
    "LimbicA": "Limbic",
    "LimbicB": "Limbic",
    "TempPar": "TempPar",
    "Background+FreeSurfer": "Background",
}


def extract_roi_labels_from_atlas(atlas: object) -> List[str]:
    """Extract ROI labels from the Brainstorm ``Atlas`` struct.

    Brainstorm structs can arrive with slightly different container shapes
    depending on how ``scipy.io.loadmat`` interprets them. This function keeps
    the parsing logic in one place and falls back to synthetic names if a scout
    label cannot be read.
    """
    scouts = getattr(atlas, "Scouts", [])
    if isinstance(scouts, (list, tuple)):
        scouts_arr = np.array(scouts, dtype=object).ravel()
    else:
        scouts_arr = np.atleast_1d(scouts).ravel()

    roi_labels: List[str] = []
    for scout in scouts_arr:
        if hasattr(scout, "Label"):
            roi_labels.append(str(scout.Label))
        elif isinstance(scout, np.ndarray) and scout.dtype.names and "Label" in scout.dtype.names:
            label = scout["Label"]
            if isinstance(label, np.ndarray):
                label = label.flat[0]
            roi_labels.append(str(label))
        else:
            roi_labels.append(f"ROI_{len(roi_labels)}")
    return roi_labels


def get_network_prefix(label: str) -> str:
    """Extract the Schaefer network prefix from a full ROI label."""
    return label.split("_")[0].split()[0]


def map_prefix_to_network(prefix: str, network_map: Mapping[str, str] | None = None) -> str:
    """Collapse atlas-specific prefixes into the network names used downstream."""
    mapping = network_map or DEFAULT_NETWORK_MAP
    return mapping.get(prefix, prefix)


def roi_networks(roi_labels: Sequence[str], network_map: Mapping[str, str] | None = None) -> List[str]:
    """Map every ROI label to its functional network."""
    return [map_prefix_to_network(get_network_prefix(label), network_map=network_map) for label in roi_labels]


def build_network_masks(
    roi_labels: Sequence[str],
    network_map: Mapping[str, str] | None = None,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Build boolean ROI masks for each network.

    The returned masks let the pipeline select all ROI pairs that belong to a
    given network block in a connectivity matrix.
    """
    networks = roi_networks(roi_labels, network_map=network_map)
    unique_networks = sorted(set(networks))
    masks = {network: np.array([name == network for name in networks], dtype=bool) for network in unique_networks}
    return networks, masks


def mean_intra(C: np.ndarray, mask: np.ndarray) -> float:
    """Average the upper triangle of a within-network block."""
    idx = np.where(mask)[0]
    if len(idx) <= 1:
        return float("nan")
    sub = C[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), 1)
    return float(np.nanmean(sub[iu]))


def mean_inter(C: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Average all ROI pairs between two distinct networks."""
    idx_a = np.where(mask_a)[0]
    idx_b = np.where(mask_b)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        return float("nan")
    return float(np.nanmean(C[np.ix_(idx_a, idx_b)]))


def network_summary_rows(
    subject_id: str,
    group: str,
    band: str,
    metric: str,
    C: np.ndarray,
    network_to_mask: Mapping[str, np.ndarray],
    exclude_networks: Iterable[str] = ("Background",),
) -> List[dict]:
    """Convert one ROI-level matrix into a set of network summary rows.

    For each subject, band, and metric we export:

    - one ``intra`` value per network
    - one ``inter`` value per unordered network pair

    Those rows are later used both for descriptive summaries and for building
    the hypothesis-specific composites in the strong H1-H3 analysis.
    """
    excluded = set(exclude_networks)
    kept_networks = [name for name in sorted(network_to_mask) if name not in excluded]
    rows: List[dict] = []
    for network in kept_networks:
        rows.append(
            {
                "subject_id": subject_id,
                "group": group,
                "band": band,
                "metric": metric,
                "connection_type": "intra",
                "network_a": network,
                "network_b": network,
                "value": mean_intra(C, network_to_mask[network]),
            }
        )
    for network_a, network_b in combinations(kept_networks, 2):
        rows.append(
            {
                "subject_id": subject_id,
                "group": group,
                "band": band,
                "metric": metric,
                "connection_type": "inter",
                "network_a": network_a,
                "network_b": network_b,
                "value": mean_inter(C, network_to_mask[network_a], network_to_mask[network_b]),
            }
        )
    return rows
