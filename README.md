# MEG Connectivity Pipeline for ROI Time-Series Analysis

This is the maintained code path behind the manuscript. It starts from private
Brainstorm source-space ROI exports and rebuilds the analysis from subject
trials to the final `H1-H3` endpoints and paper figures.

The repository is intentionally narrow. It does not try to be a general MEG
toolbox, and it does not include the upstream sensor-space preprocessing or
source reconstruction steps. What it does include is the full analysis that can
be reproduced from ROI time series onward: data loading, band-limited
connectivity, within-subject averaging, network summaries, endpoint
construction, and the figure set used in the paper.

The public repository covers the part that can actually be rerun here:

- load ROI time-series `.mat` exports
- rebuild `trial x ROI x time` tensors
- compute trial-wise connectivity
- average within subject
- collapse ROI matrices to network summaries
- build the predefined endpoint family `H1-H3`
- export the paper figures

Files written under `figures/`, `tables/`, and `outputs_full_cohort/` are local
artifacts. The code is versioned; the private data and derived outputs are not.

The public supplementary Fisher-z package is versioned under `supplementary/`.
It contains the canonical source CSVs, the LaTeX tables, and optional rendered
PDF/PNG table figures used for manuscript assembly.

## Canonical entry points

- [run_group_analysis.py](run_group_analysis.py)
  Raw Brainstorm `.mat` files to cohort outputs and strong `H1-H3` testing.
- [run_strong_hypotheses.py](run_strong_hypotheses.py)
  Refresh only the final `H1-H3` report from an existing output folder.
- [run_paper_figures.py](run_paper_figures.py)
  Regenerate the paper figure set from raw data or from existing outputs.
- [run_validate_inputs.py](run_validate_inputs.py)
  Check whether a local ROI time-series dataset matches the structural contract
  expected by the public pipeline.
- [run_methodological_validation.py](run_methodological_validation.py)
  Rebuild the methodological sensitivity checks that accompany the main
  confirmatory analysis.

## Repository layout

```text
.
├── README.md
├── CITATION.cff
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
│   └── roi_input_contract.md
├── environment.yml
├── examples/
│   ├── README.md
│   ├── build_example_dataset.py
│   └── brainstorm_roi_small/
│       └── data/
├── requirements.txt
├── tests/
├── run_validate_inputs.py
├── run_group_analysis.py
├── run_strong_hypotheses.py
├── run_paper_figures.py
├── run_methodological_validation.py
├── simulation_aec_snr_bias/
├── supplementary/
│   ├── README.md
│   ├── source_tables/
│   ├── tables/
│   └── figures/
├── meg_alzheimer/
│   ├── __init__.py
│   ├── atlas.py
│   ├── connectivity.py
│   ├── dataset.py
│   ├── methodological_validation.py
│   ├── pipeline.py
│   ├── qc.py
│   ├── robustness.py
│   ├── signals.py
│   ├── stats.py
│   ├── strong_hypotheses.py
│   └── viz.py
└── scripts/
    ├── README.md
    ├── figures/
    ├── final_figures/
    ├── final_tables/
    └── supplementary/
```

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

This repository starts from source-space ROI time series that were already
exported upstream. Sensor-space preprocessing and source reconstruction happen
before this codebase.

For a detailed contract aimed at other labs, see:

- [docs/roi_input_contract.md](docs/roi_input_contract.md)

In practice:

- another lab can reuse the low-level pipeline if it can export ROI time series
  in the same Brainstorm-like struct format
- exact reproduction of the paper endpoints also requires the same
  Schaefer-style ROI naming used by `meg_alzheimer/atlas.py`

Validate a new dataset before running the full workflow:

```bash
python run_validate_inputs.py --data-root data --require-paper-atlas
```

For a public smoke example that matches the same struct contract, see:

- [examples/README.md](examples/README.md)

## Scientific pipeline

### 1. Subject loading and trial reconstruction

Code:

- [meg_alzheimer/dataset.py](meg_alzheimer/dataset.py)

Brainstorm stores trials as one long stack of ROI blocks. The loader restores
the original `trial x ROI x time` layout and keeps track of ROI labels and time
axis metadata while doing so. That reshape is one of the main things that can
silently go wrong in these exports, so the repository keeps it explicit and
easy to audit.

### 2. Central window selection

Code:

- [meg_alzheimer/dataset.py](meg_alzheimer/dataset.py)

Each trial contains 8 seconds of data at 1000 Hz. The pipeline keeps the
central 4-second segment (`2000:6000`). This mirrors the discovery analysis and
avoids leaning on the filtered edges of the trial.

### 3. Band-limited analytic signals

Code:

- [meg_alzheimer/signals.py](meg_alzheimer/signals.py)

Each cropped trial is filtered into delta, theta, alpha, beta, and gamma. The
implementation uses an FIR filter built with `scipy.signal.firwin`
(`numtaps = 801`, Hamming window) and applies it with `filtfilt` so the phase
response stays symmetric. The analytic signal is then obtained with a Hilbert
transform, which provides the phase and amplitude-envelope quantities used by
the connectivity metrics.

### 4. Trial-wise connectivity

Code:

- [meg_alzheimer/connectivity.py](meg_alzheimer/connectivity.py)

For every trial, band, and ROI pair, the pipeline computes `PLV`, `AEC`, and
`AEC-orth`. `AEC` and `AEC-orth` are unitless correlation-based measures.
Negative values are kept, and the main analysis does not apply a Fisher
transform before averaging. That choice was deliberate and is one of the
decisions revisited in the robustness block.

### 5. Within-subject averaging

Code:

- [meg_alzheimer/pipeline.py](meg_alzheimer/pipeline.py)

Trial-level matrices are averaged within subject. The repository is built
around the subject as the statistical unit, so trials are never treated as
independent observations in the group tests.

### 6. ROI to network summary

Code:

- [meg_alzheimer/atlas.py](meg_alzheimer/atlas.py)

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

This collapsing rule is paper-specific. If another lab uses a different atlas,
the core connectivity pipeline still applies, but exact `H1-H3` reproduction
requires adapting:

- [meg_alzheimer/atlas.py](meg_alzheimer/atlas.py)
- [meg_alzheimer/strong_hypotheses.py](meg_alzheimer/strong_hypotheses.py)

### 7. Endpoint construction

Code:

- [meg_alzheimer/strong_hypotheses.py](meg_alzheimer/strong_hypotheses.py)

The confirmatory endpoint family is fixed: `H1_AEC`, `H1_AECorth`, `H2_AEC`,
`H2_AECorth`, `H3_gap_full`, and `H3_gap_inter`. `H3` is defined as
`AEC - AEC-orth` in alpha-band TempPar-centered composites. The endpoint logic
is kept in one place so that the paper figures, tables, and strong-testing
report all point back to the same definitions.

### 8. Final inference

Code:

- [meg_alzheimer/strong_hypotheses.py](meg_alzheimer/strong_hypotheses.py)

The final report uses one-sided Welch tests, effect size `d`, bootstrap
intervals, Holm correction, and max-T permutation control across the six
endpoints. The idea is to keep the descriptive pipeline and the confirmatory
layer separate: network summaries are exported first, and the strong-testing
step reads those outputs rather than rebuilding the subject matrices.

## Paper figure set

The maintained paper figure set is:

- `fig_pipeline_overview`
- `fig_qc_valid_trials`
- `fig_network_heatmaps_alpha`
- `fig_network_heatmaps_beta`
- `fig_composite_breakdown`
- `fig_endpoints_distributions`
- `fig_sensitivity_trials`

These are generated by:

- [scripts/final_figures/build_pipeline_overview.py](scripts/final_figures/build_pipeline_overview.py)
- [scripts/final_figures/build_qc_valid_trials_figure.py](scripts/final_figures/build_qc_valid_trials_figure.py)
- [scripts/final_figures/build_network_heatmaps.py](scripts/final_figures/build_network_heatmaps.py)
- [scripts/final_figures/build_composite_breakdown.py](scripts/final_figures/build_composite_breakdown.py)
- [scripts/final_figures/build_endpoints_main_figure.py](scripts/final_figures/build_endpoints_main_figure.py)
- [scripts/final_figures/build_trials_threshold_sensitivity.py](scripts/final_figures/build_trials_threshold_sensitivity.py)

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

## Supplementary simulation

The repository also includes a standalone supplementary simulation package:

- [simulation_aec_snr_bias](simulation_aec_snr_bias)

This package does not change the empirical pipeline. It addresses one specific
question: whether realistic SNR and trial-count differences could bias
`AEC-orth` enough to explain the observed `H3` gap effect.

## Installation

The files below pin the environment used to rerun the paper outputs:

- [requirements.txt](requirements.txt)
- [environment.yml](environment.yml)

Using `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate meg-alzheimer-paper
```

## Citation

The repository includes machine-readable citation metadata:

- [CITATION.cff](CITATION.cff)

## Tests

The public test suite avoids private data and checks only the stable logic that
can be exercised from synthetic inputs.

```bash
python -m unittest discover -s tests -v
```

The tests cover:

- dataset loading helpers
- atlas/network collapsing
- public example-dataset smoke execution
- signal-processing helpers
- endpoint reconstruction logic
- trial-threshold sensitivity bookkeeping

## Typical usage

### Validate a new ROI time-series dataset

```bash
python run_validate_inputs.py --data-root data --require-paper-atlas
```

### Validate the public example dataset

```bash
python run_validate_inputs.py \
  --data-root examples/brainstorm_roi_small/data \
  --require-paper-atlas
```

### Run the public smoke test

```bash
python -m unittest tests.test_public_smoke -v
```

### Full batch: raw data to paper figures

```bash
python run_paper_figures.py --data-root data --output-root outputs_full_cohort
```

This command runs the cohort pipeline, refreshes the strong `H1-H3` outputs,
and regenerates the seven paper figures.

### Rebuild figures only from existing outputs

```bash
python run_paper_figures.py \
  --data-root data \
  --output-root outputs_full_cohort \
  --skip-group-analysis
```

### Run the scientific pipeline only

```bash
python run_group_analysis.py --data-root data --output-root outputs_full_cohort
```

### Refresh only the strong endpoint report

```bash
python run_strong_hypotheses.py --output-root outputs_full_cohort
```

## Reproducibility limits

- The repository does not document baseline diagnosis, follow-up duration, eyes
  open versus closed, or MEG vendor because those fields are not available in
  the accessible files.
- Age and sex were not available as subject-level values in the raw structs
  accessible to this codebase.
- Sensor-space preprocessing and source reconstruction are upstream steps. This
  repository starts from already exported source-space ROI time series.
- Exact reproduction of the paper-level endpoint family requires the same
  Schaefer-style network naming convention encoded in
  [meg_alzheimer/atlas.py](meg_alzheimer/atlas.py). Other ROI atlases are still
  compatible with the low-level signal-processing code, but not with the paper
  endpoint definitions without explicit adaptation.

Those limits are real. The repository is meant to make the part of the workflow
that is available fully reproducible, not to imply that every upstream or
clinical detail is present in the public codebase.
