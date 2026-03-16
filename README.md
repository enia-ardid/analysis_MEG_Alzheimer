# MEG Connectivity Analysis for Converters vs Non-converters

This repository contains a streamlined analysis pipeline for taking raw
Brainstorm source-space `.mat` files and turning them into subject-level
connectivity matrices, network-level summaries, and strongly tested hypotheses.

The code is organized around one main idea: every processing step should move
the data from a lower-level representation to a representation that matches the
scientific question more directly.

In practice, the workflow is:

1. load raw subject structs from `data/`
2. reconstruct `trial x ROI x time` arrays
3. keep the clean 4-second segment from each 8-second padded trial
4. compute band-limited connectivity per trial
5. average trials within each subject
6. summarize ROI-by-ROI matrices at the network level
7. build hypothesis-specific composites
8. test hypotheses `H1-H3` with strong family-wise correction

## Repository layout

- `meg_alzheimer/`: core package
- `run_group_analysis.py`: full raw-data-to-results entry point
- `run_strong_hypotheses.py`: reruns only the final `H1-H3` confirmation step
- `scripts/final_figures/`: scripts that regenerate manuscript figures
- `scripts/final_tables/`: scripts that regenerate manuscript tables
- `analysis_alzheimer.ipynb`: notebook for inspecting outputs after the batch run
- `requirements.txt`: Python dependencies

Generated figure files, table files, and caption drafts are treated as local
artifacts and are not meant to be versioned in the public repository. They can
be regenerated from the code in `scripts/` once the private dataset is
available locally.

Public-facing tree:

```text
.
├── LICENSE
├── README.md
├── requirements.txt
├── analysis_alzheimer.ipynb
├── run_group_analysis.py
├── run_strong_hypotheses.py
├── meg_alzheimer/
│   ├── __init__.py
│   ├── atlas.py
│   ├── connectivity.py
│   ├── dataset.py
│   ├── pipeline.py
│   ├── signals.py
│   ├── stats.py
│   ├── strong_hypotheses.py
│   └── viz.py
└── scripts/
    ├── README.md
    ├── final_figures/
    └── final_tables/
```

## Expected data layout

The pipeline expects Brainstorm-style `.mat` files under `data/`:

```text
data/
├── C_p1.mat
├── C_p2.mat
├── C_p3.mat
├── NC_p1.mat
├── NC_p2.mat
└── NC_p3.mat
```

The current group mapping is:

- `C_*` -> `Converter`
- `NC_*` -> `Non-converter`

Each top-level subject struct is expected to contain:

- `Value`: ROI time series stacked trial by trial
- `Time`: sample axis
- `Atlas`: ROI labels

For a subject with `T` trials, `Value` is expected to have shape:

\[
(T \cdot 102) \times 8000
\]

where:

- `102` is the number of Schaefer ROIs in the source export
- `8000` samples correspond to 8 seconds at 1000 Hz
- the central `2000:6000` samples contain the clean 4-second segment of
  interest

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## End-to-end workflow

### Stage 1. Raw `.mat` files to subject tensors

Module:

- [`meg_alzheimer/dataset.py`](meg_alzheimer/dataset.py)

The loader inspects each `.mat` file, finds all top-level subject structs, and
reconstructs the original organization of the data:

\[
X_s(\tau, i, n)
\]

with:

- \(s\): subject
- \(\tau\): trial
- \(i\): ROI
- \(n\): time sample

Why this step exists:

- the Brainstorm export stores all trials as one long stack of ROI blocks
- connectivity must be computed trial by trial, not on the flattened matrix
- the subject, not the trial, is the unit that will later enter the group
  statistics

### Stage 2. Keep the central clean segment

Module:

- [`meg_alzheimer/dataset.py`](meg_alzheimer/dataset.py)

Only the central samples are analyzed:

\[
X^{mid}_s(\tau, i, n) = X_s(\tau, i, n), \quad n \in [2000, 6000)
\]

Why this step exists:

- the raw file keeps 2 seconds before and after the segment of interest
- this padding is useful during filtering
- the central 4-second window avoids edge artifacts created by band-pass
  filtering

### Stage 3. Band-limited analytic signals

Module:

- [`meg_alzheimer/signals.py`](meg_alzheimer/signals.py)

Each cropped trial is filtered into the canonical bands used in the project:

- delta
- theta
- alpha
- beta
- gamma

If \(x^{(b)}(t)\) is the filtered signal in band \(b\), the analytic signal is:

\[
z^{(b)}(t) = x^{(b)}(t) + j \mathcal{H}\{x^{(b)}(t)\}
\]

and the envelope is:

\[
a^{(b)}(t) = |z^{(b)}(t)|
\]

Why this step exists:

- the hypotheses are frequency-specific
- `AEC` and `AEC-orth` operate on band-limited envelopes
- the Hilbert transform is the standard way to extract those envelopes from
  oscillatory signals

### Stage 4. Trial-wise connectivity matrices

Module:

- [`meg_alzheimer/connectivity.py`](meg_alzheimer/connectivity.py)

For every trial, band, and ROI pair, the pipeline computes:

- `PLV`
- `AEC`
- `AEC-orth`

`AEC` is the Pearson correlation between ROI envelopes:

\[
AEC_{ij}^{(b)} = corr(a_i^{(b)}, a_j^{(b)})
\]

`AEC-orth` first orthogonalizes one analytic signal with respect to the other to
reduce zero-lag mixing, then correlates envelopes of the orthogonalized
components.

Why this step exists:

- trial-wise estimation preserves the original segmentation of the recording
- `AEC` captures amplitude co-fluctuations
- `AEC-orth` provides a more conservative envelope measure that is less
  sensitive to instantaneous source leakage

### Stage 5. Subject-level connectivity matrices

Module:

- [`meg_alzheimer/pipeline.py`](meg_alzheimer/pipeline.py)

Each subject has multiple trials. The pipeline averages trial-level matrices:

\[
\bar{C}_{s,b,m}(i,j) = \frac{1}{T_s}\sum_{\tau=1}^{T_s} C_{s,\tau,b,m}(i,j)
\]

where:

- \(b\): frequency band
- \(m\): connectivity metric
- \(T_s\): number of trials for subject \(s\)

Why this step exists:

- trials from the same subject are not independent observations
- averaging within subject reduces noise and avoids pseudo-replication
- group comparisons should operate on one matrix per subject

Main subject output:

- `outputs_full_cohort/subjects/<group>/<subject_id>/connectivity_matrices.npz`

### Stage 6. Network-level summaries

Modules:

- [`meg_alzheimer/atlas.py`](meg_alzheimer/atlas.py)
- [`meg_alzheimer/pipeline.py`](meg_alzheimer/pipeline.py)

The ROI-by-ROI matrix is then summarized using the Schaefer network labels.

If \(\mathcal{R}_A\) is the set of ROIs belonging to network \(A\), the mean
intra-network connectivity is:

\[
M^{intra}_{s,b,m}(A) =
\frac{2}{|\mathcal{R}_A|(|\mathcal{R}_A|-1)}
\sum_{i<j,\; i,j \in \mathcal{R}_A}\bar{C}_{s,b,m}(i,j)
\]

The mean inter-network connectivity between networks \(A\) and \(B\) is:

\[
M^{inter}_{s,b,m}(A,B) =
\frac{1}{|\mathcal{R}_A||\mathcal{R}_B|}
\sum_{i \in \mathcal{R}_A}\sum_{j \in \mathcal{R}_B}\bar{C}_{s,b,m}(i,j)
\]

Why this step exists:

- hypotheses are formulated at the functional-network level, not at single ROI
  pairs
- summarizing by network greatly reduces dimensionality
- network summaries are easier to interpret biologically than thousands of raw
  edges

Main outputs:

- `outputs_full_cohort/subject_network_means.csv`
- `outputs_full_cohort/network_group_stats.csv`

### Stage 7. Hypothesis-specific composites

Module:

- [`meg_alzheimer/strong_hypotheses.py`](meg_alzheimer/strong_hypotheses.py)

The final hypotheses are not tested on every possible edge. They are tested on
targeted composites derived from the network summaries.

A composite is a subject-level mean over a predefined family of network
connections that represent one scientific hypothesis.

For example, the alpha temporo-parietal composite used in `H1` is built by
combining:

- all inter-network rows involving `TempPar`
- the intra-network `TempPar-TempPar` row

If \(Q_h\) is the set of selected network summaries for hypothesis \(h\), the
composite is:

\[
Comp^{(m)}_{s,h} = \frac{1}{|Q_h|}\sum_{q \in Q_h} M^{(m)}_{s,q}
\]

The `H3` gap is then defined as:

\[
Gap_{s,h} = Comp^{AEC}_{s,h} - Comp^{AECorth}_{s,h}
\]

Why this step exists:

- the hypotheses are about distributed network patterns, not isolated edges
- a composite turns a multi-connection hypothesis into one scalar per subject
- this reduces noise and keeps the statistical family small and interpretable

### Stage 8. Strong confirmation of `H1-H3`

Module:

- [`meg_alzheimer/strong_hypotheses.py`](meg_alzheimer/strong_hypotheses.py)

The final strong report tests six fixed endpoints:

- `H1_AEC`
- `H1_AECorth`
- `H2_AEC`
- `H2_AECorth`
- `H3_gap_full`
- `H3_gap_inter`

Each endpoint is tested with a Welch t-test:

\[
t = \frac{\bar{x}_A - \bar{x}_B}
{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}
\]

The report also computes:

- one-sided p-values
- Cohen's `d`
- bootstrap 95% confidence intervals
- Holm-Bonferroni family-wise correction
- max-T permutation family-wise correction

Why this step exists:

- the hypotheses have a directional prediction
- Welch's test is appropriate for two independent groups without forcing equal
  variance
- Holm and max-T provide strong family-wise control across the fixed set of
  hypothesis endpoints

Main outputs:

- `outputs_full_cohort/strong_hypotheses/endpoint_tests.csv`
- `outputs_full_cohort/strong_hypotheses/hypothesis_summary.csv`
- `outputs_full_cohort/strong_hypotheses/strong_summary.md`
- `outputs_full_cohort/strong_hypotheses/strong_endpoints.png`

## How to run the pipeline

### Option A. One command from raw data to final hypothesis report

```bash
python run_group_analysis.py --data-root data --output-root outputs_full_cohort
```

This command runs:

1. raw subject discovery
2. subject-level connectivity estimation
3. group-level summaries
4. strong `H1-H3` confirmation

Use this when you want the complete analysis from scratch.

### Option B. Run the cohort processing first, then the final hypothesis report

Step 1:

```bash
python run_group_analysis.py \
  --data-root data \
  --output-root outputs_full_cohort \
  --skip-hypotheses
```

Step 2:

```bash
python run_strong_hypotheses.py --output-root outputs_full_cohort
```

Use this when:

- subject-level processing has already finished
- you want to rerun the final inferential layer only
- you change the strong-testing permutation or bootstrap settings

### Useful runtime flags

Full cohort processing can take hours. These options are often useful:

```bash
python run_group_analysis.py \
  --data-root data \
  --output-root outputs_full_cohort \
  --quicklook-only \
  --no-subject-graphs
```

Key flags:

- `--quicklook-only`: save only one quick matrix plot per subject
- `--no-subject-graphs`: skip thresholded network graph images
- `--skip-hypotheses`: stop after cohort-level outputs
- `--hypothesis-n-perm`: permutation count for the final strong report
- `--hypothesis-n-boot`: bootstrap count for the final strong report

To inspect all available options:

```bash
python run_group_analysis.py --help
python run_strong_hypotheses.py --help
```

## Output structure

The most important outputs are:

- `outputs_full_cohort/subjects.csv`
- `outputs_full_cohort/run_config.json`
- `outputs_full_cohort/subjects/<group>/<subject_id>/connectivity_matrices.npz`
- `outputs_full_cohort/subject_global_means.csv`
- `outputs_full_cohort/subject_network_means.csv`
- `outputs_full_cohort/global_group_stats.csv`
- `outputs_full_cohort/network_group_stats.csv`
- `outputs_full_cohort/edgewise_group_summary.csv`
- `outputs_full_cohort/strong_hypotheses/endpoint_tests.csv`
- `outputs_full_cohort/strong_hypotheses/hypothesis_summary.csv`
- `outputs_full_cohort/strong_hypotheses/strong_summary.md`

These outputs are generated locally and are ignored by Git in the public
repository configuration.

## Notebook usage

After the batch run:

```bash
jupyter lab
```

Then open `analysis_alzheimer.ipynb` to inspect saved outputs rather than
recomputing the cohort interactively.

## Runtime expectations

- full subject-level processing takes hours for the complete cohort
- the final strong `H1-H3` report is much faster than recomputing all subjects

## Interpretation notes

- `group_stats/*_mask.png` are edgewise ROI-by-ROI significance masks
- a black mask means no individual ROI pair survived correction in that family
- this does not rule out effects at the network or composite level

## Minimal reproducible path

If you want the shortest reproducible route from raw data to final inference,
use exactly these two commands:

```bash
python run_group_analysis.py --data-root data --output-root outputs_full_cohort --skip-hypotheses
python run_strong_hypotheses.py --output-root outputs_full_cohort
```

That path is sufficient to reproduce:

- subject-level matrices
- network summaries
- the strong confirmation report for `H1-H3`

## Manuscript artifact scripts

Once the cohort outputs exist locally, the scripts under `scripts/` can be used
to regenerate manuscript-facing figures and tables. The repository keeps those
generation scripts, but not the generated `.png`, `.pdf`, `.csv`, or `.tex`
artifacts.

- `scripts/final_tables/build_cohort_main_table.py`
- `scripts/final_tables/build_cohort_qc_table.py`
- `scripts/final_figures/build_qc_valid_trials_figure.py`
- `scripts/final_figures/build_network_heatmaps.py`
- `scripts/final_figures/build_endpoints_main_figure.py`
- `scripts/final_figures/build_forest_endpoints.py`
- `scripts/final_figures/build_composite_breakdown.py`
- `scripts/final_figures/build_robustness_figure.py`

The post hoc exact-network checks live alongside those scripts but remain
separate from the confirmatory `H1-H3` workflow.
