# Script Layout

This directory contains the source scripts used to regenerate manuscript-facing
artifacts from locally available cohort outputs.

## Subdirectories

- `final_figures/`: builds publication-ready figure files from
  `outputs_full_cohort/`
- `final_tables/`: builds manuscript tables from the same local outputs

## Canonical entry points

The main scientific pipeline is not launched from this directory. The canonical
top-level entry points remain:

- `run_group_analysis.py`: raw Brainstorm `.mat` files to cohort outputs
- `run_strong_hypotheses.py`: final strong testing of `H1-H3`

The scripts in this folder sit downstream of those entry points. They assume
that local outputs already exist and are intended to regenerate figures and
tables without redefining the underlying methodology.

## Publication policy

The public repository is configured to version the code in `scripts/`, but not
the generated files under `figures/`, `tables/`, or the caption drafts written
to the repository root. Those artifacts are reproducible from the scripts once
the private dataset is available locally.
