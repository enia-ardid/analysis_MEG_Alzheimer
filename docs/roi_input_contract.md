# ROI Time-Series Input Contract

This repository starts from already exported source-space ROI time series. It
does not perform sensor-space preprocessing or source reconstruction.

The code is reusable by other labs if they can provide ROI data that match the
same structural contract. Exact reproduction of the paper results requires the
same atlas conventions as the original dataset.

## Minimal on-disk format

The validator and the main pipeline expect MATLAB `.mat` files under `data/`.
Each file may contain one or many subject structs. The current project uses
three split files per group, but the code discovers subjects recursively and
does not require that exact split.

Example:

```text
data/
├── C_p1.mat
├── C_p2.mat
├── C_p3.mat
├── NC_p1.mat
├── NC_p2.mat
└── NC_p3.mat
```

## Required fields inside each subject struct

Each top-level MATLAB struct must contain:

- `Value`
- `Time`
- `Atlas`

### `Value`

`Value` stores trials stacked ROI block by ROI block. For `T` trials, `R` ROIs,
and `N` time samples per trial, the expected shape is:

- `(T * R, N)`

The loader also accepts the transposed case if it can detect that the time axis
is on the wrong dimension and transpose safely.

### `Time`

`Time` is a one-dimensional sample axis shared by every trial. Its length must
match the time dimension of `Value`.

### `Atlas`

`Atlas` must contain a `Scouts` field with readable ROI labels. The loader uses
these labels to:

- recover the ROI count
- check consistency across subjects
- collapse parcels into large-scale networks for the paper analyses

## Cross-subject invariants

The public pipeline assumes that all subjects share:

- the same ROI count
- the same ROI label order
- the same trial time axis length
- the same atlas definition

If any of these differ across subjects, the validator should fail and the
dataset should be harmonized before running the paper workflow.

## Group discovery

Subjects are assigned to groups by tokens found either in the top-level MATLAB
variable name or in parent directory names.

Supported tokens are:

- `C`, `Converter`, `Converters`
- `NC`, `Non-converter`, `Non-converters`

For exact reproduction of this paper, converters and non-converters must be
identifiable through those tokens or by renaming the files or structs to match
them.

## Exact paper reproduction requirements

Running the code on another lab's ROI time-series exports is possible, but the
exact manuscript figures and endpoints are only portable if the following are
also true:

- the sampling rate is `1000 Hz`
- each trial contains `8000` samples
- the central window `2000:6000` is the intended analysis segment
- ROI labels use the Schaefer-style network prefixes expected by
  `meg_alzheimer/atlas.py`
- the network collapsing rule in `DEFAULT_NETWORK_MAP` remains appropriate

The exact paper endpoints `H1-H3` depend on the collapsed networks:

- `Control`
- `Default`
- `DorsAttn`
- `Limbic`
- `SalVentAttn`
- `SomMot`
- `TempPar`
- `VisCent`
- `VisPeri`

If another lab uses a different atlas, the low-level signal-processing code can
still be reused, but two files must be adapted before the paper-level network
summaries make sense:

- `meg_alzheimer/atlas.py`
- `meg_alzheimer/strong_hypotheses.py`

## Validation command

Validate a new dataset before running the full pipeline:

```bash
python run_validate_inputs.py --data-root data --require-paper-atlas
```

For a quick smoke check on only a few subjects:

```bash
python run_validate_inputs.py --data-root data --max-subjects 4 --require-paper-atlas
```

## What is generic and what is paper-specific

Reusable without changing the scientific definitions:

- FIR band-pass filtering
- Hilbert analytic signal
- `AEC`
- `AEC-orth`
- within-subject averaging

Paper-specific:

- Schaefer-derived ROI label parsing
- ROI-to-network collapsing
- `H1-H3` endpoint family
- manuscript figure builders

That separation is deliberate. It allows other labs to reuse the reliable
signal-processing core while keeping the manuscript claims tied to the exact
network definitions that motivated this study.
