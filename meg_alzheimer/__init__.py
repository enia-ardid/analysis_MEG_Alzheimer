"""Public package surface for the MEG Alzheimer connectivity workflow.

Only the stable building blocks used by the streamlined raw-data-to-hypothesis
pipeline are re-exported here. Internal helpers stay inside their modules so
the package namespace remains small and predictable.
"""

from .atlas import (
    DEFAULT_NETWORK_MAP,
    build_network_masks,
    extract_roi_labels_from_atlas,
    get_network_prefix,
    map_prefix_to_network,
    mean_inter,
    mean_intra,
)
from .connectivity import aec_matrix, aec_orth_matrix, plv_matrix
from .dataset import LoadedSubject, SubjectRecord, discover_subjects, load_brainstorm_subject
from .pipeline import PipelineConfig, run_group_pipeline, subject_conn_from_trials
from .strong_hypotheses import generate_strong_hypothesis_report
from .signals import band_defs

__all__ = [
    "DEFAULT_NETWORK_MAP",
    "LoadedSubject",
    "PipelineConfig",
    "SubjectRecord",
    "aec_matrix",
    "aec_orth_matrix",
    "band_defs",
    "build_network_masks",
    "discover_subjects",
    "extract_roi_labels_from_atlas",
    "generate_strong_hypothesis_report",
    "get_network_prefix",
    "load_brainstorm_subject",
    "map_prefix_to_network",
    "mean_inter",
    "mean_intra",
    "plv_matrix",
    "run_group_pipeline",
    "subject_conn_from_trials",
]
