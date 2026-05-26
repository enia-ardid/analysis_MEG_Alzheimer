# ROI Time-Series Input Contract

This code starts from source-space ROI time series exported upstream. It does
not perform sensor-space preprocessing or source reconstruction.

Another lab can usually reuse the low-level pipeline if it can provide ROI data
with the same structure. Exact reproduction of the paper outputs also requires
the same atlas conventions.

The practical rule is simple: if a new dataset passes the checks below, the
filtering and connectivity code can usually be reused as-is. If it does not,
it is better to adapt the loader or the atlas mapping explicitly than to let the
pipeline fail later in a less obvious way.

## Minimal on-disk format

The validator and main pipeline expect MATLAB `.mat` files under `data/`. Each
file may contain one or more subject structs. The current project uses three
split files per group, but the loader discovers subjects recursively and does
not depend on that exact split.

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

The loader also accepts the transposed case when it can safely detect that the
time axis is on the wrong dimension.

### `Time`

`Time` is a one-dimensional sample axis shared by every trial. Its length must
match the time dimension of `Value`.

### `Atlas`

`Atlas` must contain a `Scouts` field with readable ROI labels. These labels
are used to:

- recover the ROI count
- check consistency across subjects
- collapse parcels into large-scale networks for the paper analyses

## Cross-subject invariants

The public pipeline assumes that all subjects share:

- the same ROI count
- the same ROI label order
- the same trial time axis length
- the same atlas definition

If any of these differ across subjects, validation should fail and the dataset
should be harmonized before running the paper workflow. Mixed atlas definitions
and inconsistent ROI ordering are the two cases most likely to spoil downstream
summaries without an obvious crash.

## Group discovery

Subjects are assigned to groups from tokens found either in the top-level
MATLAB variable name or in parent directory names.

Supported tokens are:

- `C`, `Converter`, `Converters`
- `NC`, `Non-converter`, `Non-converters`

For exact reproduction of this paper, converters and non-converters must be
identifiable through those tokens or by renaming files or structs accordingly.

## Exact paper reproduction requirements

Running the code on another lab's ROI exports is possible, but the exact paper
figures and endpoints only carry over if the following are also true:

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
still be reused, but these two files need to be adapted before the paper-level
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

The signal-processing core is reusable. The manuscript claims are tied to the
network definitions used here. A different atlas can still support the same
style of analysis, but at that point it becomes a related pipeline rather than
a literal rerun of this paper.
