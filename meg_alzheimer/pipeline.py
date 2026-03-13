from __future__ import annotations

"""Main cohort pipeline from raw MATLAB files to group statistics.

This module is the center of the repository. It takes the multi-subject
Brainstorm exports, reconstructs trial-wise source signals for every subject,
computes band-limited connectivity matrices, summarizes those matrices at the
subject and group level, and writes the analysis products used by the final
hypothesis tests.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from .atlas import build_network_masks, network_summary_rows
from .connectivity import aec_from_envelope, aec_matrix, aec_orth_from_analytic, aec_orth_matrix, plv_from_phase, plv_matrix
from .dataset import SubjectRecord, crop_trials, discover_subjects, load_brainstorm_subject, load_subject_structs
from .signals import band_defs, bandpass_filt
from .stats import (
    apply_fdr_to_upper_triangle,
    cohen_d_edgewise,
    edgewise_ttest,
    permutations_edgewise,
    welch_ttest_1d,
)
from .viz import graph_thresholded, plot_matrix


METRICS = ("PLV", "AEC", "AEC-orth")


@dataclass
class PipelineConfig:
    """Configuration for the cohort-level pipeline.

    The defaults reflect the current dataset:

    - source-space MEG data
    - 1000 Hz sampling rate
    - trials padded to 8000 samples
    - central 4-second analysis window stored between samples 2000 and 6000
    """
    data_root: Path = Path("data")
    output_root: Path = Path("outputs")
    sampling_rate: float = 1000.0
    crop_start: int = 2000
    crop_end: int = 6000
    bands: Mapping[str, tuple[float, float]] = field(default_factory=band_defs)
    edge_trim: int | None = None
    window_s: float | None = None
    step_s: float | None = None
    fdr_q: float = 0.10
    n_perm: int = 1000
    perm_seed: int = 0
    graph_top: float = 0.05
    quicklook_band: str = "alpha"
    quicklook_metric: str = "PLV"
    save_all_subject_matrix_plots: bool = True
    save_subject_graphs: bool = True
    group_a: str = "Converter"
    group_b: str = "Non-converter"
    verbose: bool = False


def subject_conn_from_trials(
    X_mid: np.ndarray,
    fs: float,
    bands: Mapping[str, tuple[float, float]],
    edge_trim: int | None = None,
    window_s: float | None = None,
    step_s: float | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute one connectivity matrix per band and metric for a single subject.

    Parameters
    ----------
    X_mid:
        Trial-wise signal array shaped ``n_trials x n_rois x n_time`` after the
        central analysis window has been cropped.

    Notes
    -----
    The aggregation logic deliberately averages trial-level matrices within
    subject instead of treating trials as independent samples. This keeps the
    subject as the statistical unit for all downstream group tests.
    """
    X_mid = np.asarray(X_mid, dtype=float)
    n_trials, n_rois, _ = X_mid.shape
    out: Dict[str, Dict[str, np.ndarray]] = {
        band_name: {"PLV": None, "AEC": None, "AEC-orth": None} for band_name in bands
    }
    for band_name, band_range in bands.items():
        # Filter all trials for the same band in one call. This is cheaper than
        # re-designing and re-running the filter inside the trial loop.
        X_band_all = bandpass_filt(X_mid, fs, band_range, edge_trim=edge_trim)
        analytic_all = hilbert(X_band_all, axis=-1)
        phase_all = np.angle(analytic_all)
        env_all = np.abs(analytic_all)

        plv_acc = np.zeros((n_rois, n_rois), dtype=float)
        aec_acc = np.zeros((n_rois, n_rois), dtype=float)
        aec_orth_acc = np.zeros((n_rois, n_rois), dtype=float)
        for trial_idx in range(n_trials):
            # Each trial contributes one ROI-by-ROI matrix. The final subject
            # matrix is the arithmetic mean across the available clean trials.
            plv_acc += plv_from_phase(phase_all[trial_idx], fs=fs, window_s=window_s, step_s=step_s)
            aec_acc += aec_from_envelope(env_all[trial_idx], fs=fs, window_s=window_s, step_s=step_s)
            aec_orth_acc += aec_orth_from_analytic(analytic_all[trial_idx])

        out[band_name]["PLV"] = plv_acc / n_trials
        out[band_name]["AEC"] = aec_acc / n_trials
        out[band_name]["AEC-orth"] = aec_orth_acc / n_trials
    return out


def _global_mean_upper(C: np.ndarray) -> float:
    """Return the mean of the upper triangle of a symmetric matrix."""
    iu = np.triu_indices_from(C, 1)
    return float(np.nanmean(C[iu]))


def _sanitize_token(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _flatten_matrices(matrices: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Flatten the nested ``band -> metric -> matrix`` structure for ``np.savez``."""
    flat: Dict[str, np.ndarray] = {}
    for band, metric_map in matrices.items():
        for metric, matrix in metric_map.items():
            flat[f"{band}__{metric}"] = np.asarray(matrix)
    return flat


def _edge_table(
    matrix: np.ndarray,
    roi_labels: Sequence[str],
    t_mat: np.ndarray,
    p_mat: np.ndarray,
    extra_mask: np.ndarray,
    effect_mat: np.ndarray,
    extra_name: str,
) -> pd.DataFrame:
    """Convert a boolean edge mask into a tidy edge list table."""
    rows: List[dict] = []
    iu = np.triu_indices_from(matrix, 1)
    for i, j in zip(iu[0], iu[1]):
        if not extra_mask[i, j]:
            continue
        rows.append(
            {
                "roi_i": roi_labels[i],
                "roi_j": roi_labels[j],
                "t_stat": float(t_mat[i, j]),
                "p_value": float(p_mat[i, j]),
                "effect_size_d": float(effect_mat[i, j]),
                extra_name: True,
            }
        )
    return pd.DataFrame(rows)


def _network_group_stats(network_df: pd.DataFrame, fdr_q: float) -> pd.DataFrame:
    raise RuntimeError("Use _network_group_stats_for_groups to keep comparison labels explicit.")


def _network_group_stats_for_groups(
    network_df: pd.DataFrame,
    group_a: str,
    group_b: str,
    fdr_q: float,
) -> pd.DataFrame:
    """Run group comparisons on the network summary table.

    The input table already contains one row per subject, band, metric, and
    network block. This function groups those rows by block and performs a Welch
    t-test between the requested study groups, followed by BH-FDR within each
    ``band x metric`` family.
    """
    rows: List[dict] = []
    grouped = network_df.groupby(["band", "metric", "connection_type", "network_a", "network_b"], sort=True)
    for key, chunk in grouped:
        values_a = chunk.loc[chunk["group"] == group_a, "value"].to_numpy(dtype=float)
        values_b = chunk.loc[chunk["group"] == group_b, "value"].to_numpy(dtype=float)
        if len(values_a) < 2 or len(values_b) < 2:
            continue
        t_stat, p_value = welch_ttest_1d(values_a, values_b)
        rows.append(
            {
                "band": key[0],
                "metric": key[1],
                "connection_type": key[2],
                "network_a": key[3],
                "network_b": key[4],
                "group_a": group_a,
                "group_b": group_b,
                "n_group_a": len(values_a),
                "n_group_b": len(values_b),
                "mean_group_a": float(np.nanmean(values_a)),
                "mean_group_b": float(np.nanmean(values_b)),
                "delta_group_a_minus_group_b": float(np.nanmean(values_a) - np.nanmean(values_b)),
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["fdr_significant"] = False
    result["fdr_p_crit"] = np.nan
    for (band, metric), index in result.groupby(["band", "metric"]).groups.items():
        pvals = result.loc[index, "p_value"].to_numpy(dtype=float)
        order = np.argsort(pvals)
        ranked = pvals[order]
        thresh = fdr_q * np.arange(1, len(ranked) + 1) / len(ranked)
        passed = ranked <= thresh
        if not passed.any():
            continue
        pcrit = float(ranked[np.where(passed)[0].max()])
        mask = pvals <= pcrit
        result.loc[index, "fdr_significant"] = mask
        result.loc[index, "fdr_p_crit"] = pcrit
    return result.sort_values(["band", "metric", "connection_type", "p_value"]).reset_index(drop=True)


def _save_json(path: Path, payload: dict) -> None:
    """Write a small JSON sidecar in a deterministic format."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _config_to_dict(config: PipelineConfig) -> dict:
    return {
        "data_root": str(config.data_root),
        "output_root": str(config.output_root),
        "sampling_rate": config.sampling_rate,
        "crop_start": config.crop_start,
        "crop_end": config.crop_end,
        "bands": {band: list(freqs) for band, freqs in config.bands.items()},
        "edge_trim": config.edge_trim,
        "window_s": config.window_s,
        "step_s": config.step_s,
        "fdr_q": config.fdr_q,
        "n_perm": config.n_perm,
        "perm_seed": config.perm_seed,
        "graph_top": config.graph_top,
        "quicklook_band": config.quicklook_band,
        "quicklook_metric": config.quicklook_metric,
        "save_all_subject_matrix_plots": config.save_all_subject_matrix_plots,
        "save_subject_graphs": config.save_subject_graphs,
        "group_a": config.group_a,
        "group_b": config.group_b,
        "verbose": config.verbose,
    }


def _log(config: PipelineConfig, message: str) -> None:
    """Small logging shim controlled by ``PipelineConfig.verbose``."""
    if config.verbose:
        print(message, flush=True)


def _save_subject_outputs(
    subject_dir: Path,
    record: SubjectRecord,
    roi_labels: Sequence[str],
    networks: Sequence[str],
    atlas_name: str,
    matrices: Mapping[str, Mapping[str, np.ndarray]],
    config: PipelineConfig,
) -> None:
    """Write the per-subject outputs produced by the cohort pipeline.

    Each subject folder contains:

    - the numeric matrices in ``npz`` format
    - a JSON metadata file with atlas labels and runtime parameters
    - optional matrix plots and thresholded graph plots for quick inspection
    """
    subject_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(subject_dir / "connectivity_matrices.npz", **_flatten_matrices(matrices))
    _save_json(
        subject_dir / "metadata.json",
        {
            "subject_id": record.subject_id,
            "group": record.group,
            "path": str(record.path),
            "mat_variable": record.mat_variable,
            "atlas_name": atlas_name,
            "roi_labels": list(roi_labels),
            "networks": list(networks),
            "config": {
                "sampling_rate": config.sampling_rate,
                "crop_start": config.crop_start,
                "crop_end": config.crop_end,
                "edge_trim": config.edge_trim,
                "window_s": config.window_s,
                "step_s": config.step_s,
                "save_all_subject_matrix_plots": config.save_all_subject_matrix_plots,
                "save_subject_graphs": config.save_subject_graphs,
            },
        },
    )
    matrix_plots_dir = subject_dir / "matrix_plots"
    graph_plots_dir = subject_dir / "graph_plots"
    if config.save_all_subject_matrix_plots:
        matrix_plots_dir.mkdir(exist_ok=True)
    if config.save_subject_graphs:
        graph_plots_dir.mkdir(exist_ok=True)
    for band_name, metric_map in matrices.items():
        for metric_name, matrix in metric_map.items():
            stem = f"{band_name}_{_sanitize_token(metric_name)}"
            title = f"{record.subject_id} {metric_name} {band_name}"
            if config.save_all_subject_matrix_plots:
                plot_matrix(matrix, title=title, outfile=matrix_plots_dir / f"{stem}.png")
            if config.save_subject_graphs:
                graph_thresholded(
                    matrix,
                    roi_labels=roi_labels,
                    top=config.graph_top,
                    title=title,
                    outfile=graph_plots_dir / f"{stem}.png",
                )
    band = config.quicklook_band
    metric = config.quicklook_metric
    if band in matrices and metric in matrices[band]:
        matrix = matrices[band][metric]
        plot_matrix(matrix, title=f"{record.subject_id} {metric} {band}", outfile=subject_dir / "quicklook_matrix.png")
        if config.save_subject_graphs:
            graph_thresholded(
                matrix,
                roi_labels=roi_labels,
                top=config.graph_top,
                title=f"{record.subject_id} {metric} {band}",
                outfile=subject_dir / "quicklook_graph.png",
            )


def run_group_pipeline(config: PipelineConfig) -> dict:
    """Run the full cohort analysis from raw MATLAB files to group summaries.

    The pipeline keeps a strict hierarchy:

    1. load each subject from disk
    2. crop the clean central window from every trial
    3. compute subject-level connectivity matrices
    4. summarize matrices globally and by network
    5. compare the requested groups at global, network, and edgewise levels

    All subject-level connectivity is computed before any group-level test is
    run. This ensures that subjects, not trials, are the final statistical
    observations.
    """
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    subjects = discover_subjects(config.data_root)
    _log(config, f"Discovered {len(subjects)} subjects under {config.data_root}.")
    subject_df = pd.DataFrame(
        [
            {
                "subject_id": record.subject_id,
                "group": record.group,
                "path": str(record.path),
                "mat_variable": record.mat_variable,
            }
            for record in subjects
        ],
        columns=["subject_id", "group", "path", "mat_variable"],
    )
    subject_df.to_csv(output_root / "subjects.csv", index=False)
    _save_json(output_root / "run_config.json", _config_to_dict(config))

    if not subjects:
        return {
            "subjects": [],
            "message": f"No .mat subjects were found under {Path(config.data_root)}.",
            "output_root": str(output_root),
        }

    subject_global_rows: List[dict] = []
    subject_network_rows: List[dict] = []
    group_mats: MutableMapping[str, MutableMapping[str, MutableMapping[str, List[np.ndarray]]]] = {}
    reference_roi_labels: Sequence[str] | None = None
    reference_networks: Sequence[str] | None = None

    records_by_file: MutableMapping[Path, List[SubjectRecord]] = {}
    for record in subjects:
        records_by_file.setdefault(record.path, []).append(record)

    for path, file_records in sorted(records_by_file.items()):
        _log(config, f"Loading {path.name} ({len(file_records)} subjects).")
        # Load all requested structs from the same MATLAB file in one pass to
        # avoid repeated disk IO on very large files.
        structs = load_subject_structs(path, [record.mat_variable for record in file_records])
        for record in file_records:
            loaded = load_brainstorm_subject(record, subject_struct=structs[record.mat_variable])
            _log(
                config,
                f"  Processing {record.subject_id} [{record.group}] from {path.name}: "
                f"{loaded.n_trials} trials, {loaded.n_rois} ROIs, {loaded.n_time} samples.",
            )
            cropped = crop_trials(loaded.X_trials, start=config.crop_start, end=config.crop_end)
            matrices = subject_conn_from_trials(
                cropped,
                fs=config.sampling_rate,
                bands=config.bands,
                edge_trim=config.edge_trim,
                window_s=config.window_s,
                step_s=config.step_s,
            )
            networks, network_to_mask = build_network_masks(loaded.roi_labels)
            if reference_roi_labels is None:
                reference_roi_labels = list(loaded.roi_labels)
                reference_networks = list(networks)
            elif list(reference_roi_labels) != list(loaded.roi_labels):
                raise ValueError("ROI labels differ across subjects. Atlas alignment must be fixed before group analysis.")

            subject_dir = output_root / "subjects" / record.group / record.subject_id
            _save_subject_outputs(
                subject_dir=subject_dir,
                record=record,
                roi_labels=loaded.roi_labels,
                networks=networks,
                atlas_name=loaded.atlas_name,
                matrices=matrices,
                config=config,
            )

            for band, metric_map in matrices.items():
                for metric, matrix in metric_map.items():
                    # ``global_mean_upper`` is a coarse descriptive summary. It
                    # is not used for the final H1-H3 tests, but it is useful
                    # to keep a simple one-number-per-subject overview.
                    subject_global_rows.append(
                        {
                            "subject_id": record.subject_id,
                            "group": record.group,
                            "band": band,
                            "metric": metric,
                            "global_mean_upper": _global_mean_upper(matrix),
                        }
                    )
                    subject_network_rows.extend(
                        network_summary_rows(
                            subject_id=record.subject_id,
                            group=record.group,
                            band=band,
                            metric=metric,
                            C=matrix,
                            network_to_mask=network_to_mask,
                        )
                    )
                    # Keep the full subject matrices grouped by band and metric
                    # so the edgewise tests can be run after all subjects have
                    # been processed.
                    group_mats.setdefault(record.group, {}).setdefault(band, {}).setdefault(metric, []).append(matrix)
            _log(config, f"  Finished {record.subject_id}.")

    global_df = pd.DataFrame(subject_global_rows).sort_values(["group", "subject_id", "band", "metric"])
    network_df = pd.DataFrame(subject_network_rows).sort_values(
        ["group", "subject_id", "band", "metric", "connection_type", "network_a", "network_b"]
    )
    global_df.to_csv(output_root / "subject_global_means.csv", index=False)
    network_df.to_csv(output_root / "subject_network_means.csv", index=False)

    global_stats_rows: List[dict] = []
    for (band, metric), chunk in global_df.groupby(["band", "metric"], sort=True):
        values_a = chunk.loc[chunk["group"] == config.group_a, "global_mean_upper"].to_numpy(dtype=float)
        values_b = chunk.loc[chunk["group"] == config.group_b, "global_mean_upper"].to_numpy(dtype=float)
        if len(values_a) < 2 or len(values_b) < 2:
            continue
        t_stat, p_value = welch_ttest_1d(values_a, values_b)
        global_stats_rows.append(
            {
                "band": band,
                "metric": metric,
                "group_a": config.group_a,
                "group_b": config.group_b,
                "n_group_a": len(values_a),
                "n_group_b": len(values_b),
                "mean_group_a": float(np.nanmean(values_a)),
                "mean_group_b": float(np.nanmean(values_b)),
                "delta_group_a_minus_group_b": float(np.nanmean(values_a) - np.nanmean(values_b)),
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )
    global_stats_df = pd.DataFrame(global_stats_rows)
    if not global_stats_df.empty:
        global_stats_df.to_csv(output_root / "global_group_stats.csv", index=False)

    network_stats_df = _network_group_stats_for_groups(
        network_df,
        group_a=config.group_a,
        group_b=config.group_b,
        fdr_q=config.fdr_q,
    )
    if not network_stats_df.empty:
        network_stats_df.to_csv(output_root / "network_group_stats.csv", index=False)
    _log(config, "Finished subject-level summaries. Starting group-level edgewise statistics.")

    edge_stats_dir = output_root / "group_stats"
    edge_stats_dir.mkdir(exist_ok=True)
    edge_results: List[dict] = []
    for band in config.bands:
        for metric in METRICS:
            mats_a = np.asarray(group_mats.get(config.group_a, {}).get(band, {}).get(metric, []), dtype=float)
            mats_b = np.asarray(group_mats.get(config.group_b, {}).get(band, {}).get(metric, []), dtype=float)
            if mats_a.shape[0] < 2 or mats_b.shape[0] < 2:
                continue
            # Edgewise statistics operate on the stack of subject matrices for a
            # given ``band x metric`` family.
            t_mat, p_mat = edgewise_ttest(mats_a, mats_b, equal_var=False)
            fdr_mask, pcrit = apply_fdr_to_upper_triangle(p_mat, q=config.fdr_q)
            perm_p = permutations_edgewise(mats_a, mats_b, n_perm=config.n_perm, seed=config.perm_seed)
            perm_mask = perm_p < 0.05
            effect_mat = cohen_d_edgewise(mats_a, mats_b)

            stem = f"{band}_{_sanitize_token(metric)}"
            np.savez_compressed(
                edge_stats_dir / f"{stem}_edgewise_stats.npz",
                t_stat=t_mat,
                p_value=p_mat,
                fdr_mask=fdr_mask,
                fdr_p_crit=np.array(np.nan if pcrit is None else pcrit),
                perm_p_value=perm_p,
                perm_significant_mask=perm_mask,
                effect_size_d=effect_mat,
                group_a=config.group_a,
                group_b=config.group_b,
                group_a_matrices=mats_a,
                group_b_matrices=mats_b,
            )
            plot_matrix(
                t_mat,
                title=f"{config.group_a} - {config.group_b} t-stat ({metric}, {band})",
                cmap="coolwarm",
                outfile=edge_stats_dir / f"{stem}_t_stat.png",
            )
            plot_matrix(
                fdr_mask.astype(float),
                title=f"FDR significant edges ({metric}, {band})",
                vmin=0.0,
                vmax=1.0,
                cmap="magma",
                outfile=edge_stats_dir / f"{stem}_fdr_mask.png",
            )
            plot_matrix(
                perm_mask.astype(float),
                title=f"Permutation significant edges ({metric}, {band})",
                vmin=0.0,
                vmax=1.0,
                cmap="magma",
                outfile=edge_stats_dir / f"{stem}_perm_mask.png",
            )

            if reference_roi_labels is None:
                continue
            fdr_edges = _edge_table(
                matrix=t_mat,
                roi_labels=reference_roi_labels,
                t_mat=t_mat,
                p_mat=p_mat,
                extra_mask=fdr_mask,
                effect_mat=effect_mat,
                extra_name="fdr_significant",
            )
            perm_edges = _edge_table(
                matrix=t_mat,
                roi_labels=reference_roi_labels,
                t_mat=t_mat,
                p_mat=perm_p,
                extra_mask=perm_mask,
                effect_mat=effect_mat,
                extra_name="perm_significant",
            )
            fdr_edges.to_csv(edge_stats_dir / f"{stem}_fdr_edges.csv", index=False)
            perm_edges.to_csv(edge_stats_dir / f"{stem}_perm_edges.csv", index=False)
            edge_results.append(
                {
                    "band": band,
                    "metric": metric,
                    "group_a": config.group_a,
                    "group_b": config.group_b,
                    "n_group_a": int(mats_a.shape[0]),
                    "n_group_b": int(mats_b.shape[0]),
                    "n_fdr_edges": int(np.count_nonzero(np.triu(fdr_mask, 1))),
                    "n_perm_edges": int(np.count_nonzero(np.triu(perm_mask, 1))),
                    "fdr_p_crit": None if pcrit is None else float(pcrit),
                }
            )
            _log(config, f"Computed edgewise stats for {band} / {metric}.")

    edge_summary_df = pd.DataFrame(edge_results)
    if not edge_summary_df.empty:
        edge_summary_df.to_csv(output_root / "edgewise_group_summary.csv", index=False)

    if reference_roi_labels is not None:
        _save_json(
            output_root / "atlas_reference.json",
            {
                "roi_labels": list(reference_roi_labels),
                "networks": list(reference_networks or []),
            },
        )

    return {
        "subjects": [record.subject_id for record in subjects],
        "n_subjects": len(subjects),
        "output_root": str(output_root),
        "groups": sorted({record.group for record in subjects}),
        "group_counts": subject_df.groupby("group").size().to_dict(),
        "comparison_groups": [config.group_a, config.group_b],
    }
