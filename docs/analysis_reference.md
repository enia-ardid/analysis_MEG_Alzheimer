# Analysis reference

[Back to the project](../README.md) · [Results](results.md) · [Input format](roi_input_contract.md)

## Commands

| Command | Purpose |
| --- | --- |
| [run_validate_inputs.py](../run_validate_inputs.py) | Validate Brainstorm ROI exports |
| [run_group_analysis.py](../run_group_analysis.py) | Compute cohort connectivity and group outputs |
| [run_strong_hypotheses.py](../run_strong_hypotheses.py) | Rebuild the six-endpoint report from existing outputs |
| [run_paper_figures.py](../run_paper_figures.py) | Run the cohort analysis and manuscript figure builders |
| [run_methodological_validation.py](../run_methodological_validation.py) | Run supplementary methodological checks |

To reuse cohort outputs when rebuilding figures:

```bash
python run_paper_figures.py \
  --data-root data \
  --output-root outputs_full_cohort \
  --skip-group-analysis
```

The QC figure still reads the original exports; skipping group analysis does
not make the complete figure set independent of private inputs.

## Scientific pipeline

### 1. Subject loading and trial reconstruction

Code:

- [meg_alzheimer/dataset.py](../meg_alzheimer/dataset.py)

Brainstorm stores trials as one long stack of ROI blocks. The loader restores
the original `trial x ROI x time` layout and keeps the ROI labels and time axis
explicit.

### 2. Central window selection

Code:

- [meg_alzheimer/dataset.py](../meg_alzheimer/dataset.py)

The study exports contain 8 seconds per trial at 1000 Hz. The current
pipeline selects the central 4-second segment (`2000:6000`) before filtering.

### 3. Band-limited analytic signals

Code:

- [meg_alzheimer/signals.py](../meg_alzheimer/signals.py)

Each cropped trial is filtered into delta, theta, alpha, beta, and gamma. The
filtering uses `scipy.signal.firwin` (`numtaps = 801`, Hamming window) and is
applied with `filtfilt` to preserve a symmetric phase response. The analytic
signal is then obtained with a Hilbert transform.

### 4. Trial-wise connectivity

Code:

- [meg_alzheimer/connectivity.py](../meg_alzheimer/connectivity.py)

For every trial, band, and ROI pair, the pipeline computes `PLV`, `AEC`, and
`AEC-orth`. Negative values are kept. The main analysis averages on the raw
correlation scale; Fisher-z sensitivity is handled separately in the robustness
layer.

### 5. Within-subject averaging

Code:

- [meg_alzheimer/pipeline.py](../meg_alzheimer/pipeline.py)

Trial-level matrices are averaged within subject. Subjects, not trials, are the
statistical unit in the group analysis.

### 6. ROI to network summary

Code:

- [meg_alzheimer/atlas.py](../meg_alzheimer/atlas.py)

The source export contains 102 ROI labels:

- 100 Schaefer parcels
- 2 medial-wall/background labels

The two background labels are excluded from network summaries. The paper uses a
collapsed 9-network family:

- `Control`
- `Default`
- `DorsAttn`
- `Limbic`
- `SalVentAttn`
- `SomMot`
- `TempPar`
- `VisCent`
- `VisPeri`

This step is paper-specific. If another lab uses a different atlas, the
connectivity code still applies, but exact `H1-H3` reproduction requires
adapting:

- [meg_alzheimer/atlas.py](../meg_alzheimer/atlas.py)
- [meg_alzheimer/strong_hypotheses.py](../meg_alzheimer/strong_hypotheses.py)

### 7. Endpoint construction

Code:

- [meg_alzheimer/strong_hypotheses.py](../meg_alzheimer/strong_hypotheses.py)

The six-endpoint family is fixed: `H1_AEC`, `H1_AECorth`, `H2_AEC`,
`H2_AECorth`, `H3_gap_full`, and `H3_gap_inter`. `H3` is defined as
`AEC - AEC-orth` in alpha-band TempPar-centered composites.

### 8. Final inference

Code:

- [meg_alzheimer/strong_hypotheses.py](../meg_alzheimer/strong_hypotheses.py)

The final report uses one-sided Welch tests, effect size `d`, bootstrap
intervals, Holm correction, and max-T permutation control across the six
endpoints.


## Input contract and portability

The pipeline expects Brainstorm-style MATLAB exports under `data/`:

```text
data/
├── C_p1.mat
├── C_p2.mat
├── C_p3.mat
├── NC_p1.mat
├── NC_p2.mat
└── NC_p3.mat
```

Current group discovery:

- `C_* -> Converter`
- `NC_* -> Non-converter`

Each subject struct is expected to contain:

- `Value`: ROI time series stacked trial by trial
- `Time`: sample axis
- `Atlas`: ROI labels

For a subject with `T` clean trials, `Value` has shape `(T * 102, 8000)`.

This code starts from source-space ROI time series that were exported upstream.
For the full structural contract, see:

- [docs/roi_input_contract.md](../docs/roi_input_contract.md)

In practice:

- another lab can reuse the low-level pipeline if it can export ROI time series
  in the same Brainstorm-like struct format
- exact reproduction of the paper endpoints also requires the same
  Schaefer-style ROI naming used by `meg_alzheimer/atlas.py`

Validate a new dataset before running the full workflow:

```bash
python run_validate_inputs.py --data-root data --require-paper-atlas
```

For a public smoke example that follows the same struct contract, see:

- [examples/README.md](../examples/README.md)

## Paper figure set

The main figure set is:

- `fig_pipeline_overview`
- `fig_qc_valid_trials`
- `fig_network_heatmaps_alpha`
- `fig_network_heatmaps_beta`
- `fig_composite_breakdown`
- `fig_endpoints_distributions`
- `fig_sensitivity_trials`

These are generated by:

- [scripts/final_figures/build_pipeline_overview.py](../scripts/final_figures/build_pipeline_overview.py)
- [scripts/final_figures/build_qc_valid_trials_figure.py](../scripts/final_figures/build_qc_valid_trials_figure.py)
- [scripts/final_figures/build_network_heatmaps.py](../scripts/final_figures/build_network_heatmaps.py)
- [scripts/final_figures/build_composite_breakdown.py](../scripts/final_figures/build_composite_breakdown.py)
- [scripts/final_figures/build_endpoints_main_figure.py](../scripts/final_figures/build_endpoints_main_figure.py)
- [scripts/final_figures/build_trials_threshold_sensitivity.py](../scripts/final_figures/build_trials_threshold_sensitivity.py)

`build_endpoints_main_figure.py` also writes the paper-ready endpoint table.

Figure-to-script map:

| Figure | Script | Main inputs |
| --- | --- | --- |
| `fig_pipeline_overview` | `scripts/final_figures/build_pipeline_overview.py` | documented workflow only |
| `fig_qc_valid_trials` | `scripts/final_figures/build_qc_valid_trials_figure.py` | raw `.mat` files + `subjects.csv` |
| `fig_network_heatmaps_alpha` | `scripts/final_figures/build_network_heatmaps.py` | saved subject matrices under `outputs_full_cohort/subjects/` |
| `fig_network_heatmaps_beta` | `scripts/final_figures/build_network_heatmaps.py` | saved subject matrices under `outputs_full_cohort/subjects/` |
| `fig_composite_breakdown` | `scripts/final_figures/build_composite_breakdown.py` | `subject_network_means.csv` |
| `fig_endpoints_distributions` | `scripts/final_figures/build_endpoints_main_figure.py` | `subject_network_means.csv` + `endpoint_tests.csv` |
| `fig_sensitivity_trials` | `scripts/final_figures/build_trials_threshold_sensitivity.py` | cached trial-level endpoint table or rebuilt endpoint DataFrame |
