# AEC versus AEC-orth SNR Bias Simulation

This supplementary package evaluates whether differential signal-to-noise
conditions could generate the observed reduction in the alpha
`AEC - AEC-orth` gap (`H3`) between converters and non-converters.

## Contents

- `signal_model.py`
  Synthetic alpha-band signal generation using filtered Gaussian noise.
- `connectivity.py`
  Trial-wise AEC, AEC-orth, and gap definitions.
- `experiments.py`
  Three Monte Carlo experiments:
  1. SNR sweep
  2. Trial-count effect
  3. SNR × true-connectivity interaction
- `plotting.py`
  Publication-ready PDF figures.
- `run_all.py`
  End-to-end entry point.

## Design choices

- Sampling rate: `1000 Hz`
- Useful trial length: `4000 samples`
- Alpha band: `8-12 Hz`
- FIR filter: Hamming window, `order = 801`, `filtfilt`
- Default true connectivity: `rho = 0.08`
- AEC-orth uses the same symmetric, signed correlation definition as the cohort
  pipeline. It is not capped at AEC; `AEC - AEC-orth` can be negative for an
  individual trial. Scalar and batch calculations are tested against the cohort
  implementation, including small-amplitude and zero signals.
- True connectivity is implemented as correlated low-frequency envelopes carried
  by a shared alpha process with a fixed non-zero phase lag. This choice is
  deliberate: it preserves genuine envelope coupling under orthogonalization at
  high SNR, which lets the simulation isolate the measurement-noise bias that
  matters for the empirical `H3` interpretation.
- The full Monte Carlo runner builds a reusable library of trial-level gap
  values for each `(rho, SNR)` condition and then resamples subject-level means
  from that library. This keeps the per-trial signal model unchanged while
  making manuscript-scale runs tractable.
- Empirical trial-count distributions are matched to the dataset:
  - converters: mean `42.5`, SD `4.25`, range `[27, 58]`
  - non-converters: mean `45.7`, SD `8.85`, range `[29, 73]`

## Running

Quick smoke test:

```bash
python -m simulation_aec_snr_bias.run_all --quick --force
```

Manuscript-scale run:

```bash
python -m simulation_aec_snr_bias.run_all --force
```

Optional controls:

- `--library-size 4096`
  Override the number of precomputed trial metrics per condition.
- `--force`
  Recompute CSV results even if existing outputs are already present.

Outputs:

- PDFs under `simulation_aec_snr_bias/figures/`
- CSV and NPZ files under `simulation_aec_snr_bias/results/`

### Metric correction and cached results

Earlier versions capped AEC-orth at AEC in the simulation, forcing every trial's
gap to be non-negative. That cap was absent from the cohort pipeline and has
been removed. Simulation results produced with the earlier definition must be
regenerated before interpreting the SNR and trial-count experiments.

The runner now records the metric definition in a `.metadata.json` file beside
each result CSV. Missing metadata, invalid JSON or an older metric definition
triggers recomputation, including the targeted trial-difference summary. These
metadata check the metric definition only: use `--force` when changing run parameters or
switching from a quick run to a manuscript-scale run. Quick runs verify execution
and are not sufficient for scientific interpretation.

## Interpretation

The key quantity is the simulated group effect on the gap:

- `delta_gap = mean_gap_NC - mean_gap_C`
- positive values mean converters have a smaller gap

Effect sizes are reported as positive `Cohen's d` when the non-converter mean
is larger than the converter mean, so the empirical reference value
`d = 0.575` can be compared directly to the simulation.
