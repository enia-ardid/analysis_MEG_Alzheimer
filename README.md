# MEG Connectivity Pipeline for ROI Time-Series Connectivity Analysis

This repository is the maintained raw-data-to-paper code path used in the
manuscript. It starts from private Brainstorm source-space ROI exports and
rebuilds the complete empirical workflow used in the paper figures.

The public code is intentionally narrow:

- load ROI time-series `.mat` exports from `data/`
- rebuild `trial x ROI x time` tensors
- compute trial-wise connectivity
- average within subject
- summarize ROI matrices at the network level
- build the predefined endpoint family `H1-H3`
- export the seven paper figures

Generated outputs under `figures/`, `tables/`, and `outputs_full_cohort/` are
local artifacts. The code is versioned; the private data and the derived
manuscript outputs are not.

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
├── simulation_aec_snr_bias/
├── meg_alzheimer/
│   ├── __init__.py
│   ├── atlas.py
│   ├── connectivity.py
│   ├── dataset.py
│   ├── pipeline.py
│   ├── qc.py
│   ├── robustness.py
│   ├── signals.py
│   ├── stats.py
│   ├── strong_hypotheses.py
│   └── viz.py
└── scripts/
    ├── README.md
    ├── final_figures/
    └── final_tables/
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

The repository starts from source-space ROI time series that were already
exported upstream. Sensor-space preprocessing and source reconstruction are not
executed here; they are external preparation steps.

For a detailed contract aimed at other labs, see:

- [docs/roi_input_contract.md](docs/roi_input_contract.md)

In short:

- if another lab can export ROI time series in the same Brainstorm-like struct
  format, the low-level pipeline can be reused without changing the signal
  processing
- exact reproduction of the paper endpoints and figures also requires the same
  Schaefer-style ROI naming conventions used by `meg_alzheimer/atlas.py`

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

The Brainstorm export stores trials as one long stack of ROI blocks. The loader
reconstructs the original organization:

- subject
- trial
- ROI
- time

This keeps the subject, not the trial, as the later statistical unit.

### 2. Central window selection

Code:

- [meg_alzheimer/dataset.py](meg_alzheimer/dataset.py)

Each trial contains 8 seconds of data at 1000 Hz. The pipeline keeps the
central 4-second segment (`2000:6000`) to avoid filter-edge contamination.

### 3. Band-limited analytic signals

Code:

- [meg_alzheimer/signals.py](meg_alzheimer/signals.py)

Each cropped trial is filtered into:

- delta
- theta
- alpha
- beta
- gamma

The implementation uses an FIR filter built with `scipy.signal.firwin`:

- Hamming window
- `numtaps = 801`
- zero-phase application via `filtfilt`

The Hilbert transform is then used to obtain the analytic signal and amplitude
envelope.

### 4. Trial-wise connectivity

Code:

- [meg_alzheimer/connectivity.py](meg_alzheimer/connectivity.py)

For every trial, band, and ROI pair, the pipeline computes:

- `PLV`
- `AEC`
- `AEC-orth`

`AEC` and `AEC-orth` are unitless correlation-based connectivity measures.
Negative values are kept; no Fisher transform is applied in the main analysis.

### 5. Within-subject averaging

Code:

- [meg_alzheimer/pipeline.py](meg_alzheimer/pipeline.py)

Trial-level matrices are averaged within subject. This avoids
pseudoreplication and preserves the subject as the unit that enters the group
comparisons.

### 6. ROI to network summary

Code:

- [meg_alzheimer/atlas.py](meg_alzheimer/atlas.py)

The source export contains 102 ROI labels:

- 100 Schaefer parcels
- 2 medial-wall/background labels

The two background labels are excluded from network summaries. The main paper
uses a collapsed 9-network family:

- `Control`
- `Default`
- `DorsAttn`
- `Limbic`
- `SalVentAttn`
- `SomMot`
- `TempPar`
- `VisCent`
- `VisPeri`

This network collapsing rule is paper-specific. If another lab uses a
different atlas, the core connectivity pipeline still applies, but exact `H1-H3`
reproduction requires adapting:

- [meg_alzheimer/atlas.py](meg_alzheimer/atlas.py)
- [meg_alzheimer/strong_hypotheses.py](meg_alzheimer/strong_hypotheses.py)

### 7. Endpoint construction

Code:

- [meg_alzheimer/strong_hypotheses.py](meg_alzheimer/strong_hypotheses.py)

The confirmatory endpoint family is fixed:

- `H1_AEC`
- `H1_AECorth`
- `H2_AEC`
- `H2_AECorth`
- `H3_gap_full`
- `H3_gap_inter`

`H3` is defined as `AEC - AEC-orth` in alpha-band TempPar-centered composites.

### 8. Final inference

Code:

- [meg_alzheimer/strong_hypotheses.py](meg_alzheimer/strong_hypotheses.py)

The final report uses:

- one-sided Welch tests
- effect size `d`
- bootstrap confidence intervals
- Holm correction
- max-T permutation control across the six-endpoint family

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

This package does not alter the empirical pipeline above. It addresses one
specific methodological question: whether realistic SNR and trial-count
differences could bias `AEC-orth` enough to explain the observed `H3` gap
effect. The simulation uses the same alpha-band filtering choices as the main
analysis and mirrors the empirical converter/non-converter trial-count
distributions.

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
can be validated from synthetic inputs.

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

This command:

1. runs the cohort pipeline
2. refreshes the strong `H1-H3` outputs
3. regenerates the seven paper figures

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
