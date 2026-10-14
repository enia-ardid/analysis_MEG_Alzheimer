# Manuscript Scripts

This directory contains the small scripts that turn pipeline outputs into paper
figures and tables. They sit downstream of the analysis code and should not
change the scientific definitions.

## Figure scripts

- [figures/build_project_assets.py](figures/build_project_assets.py)
  Builds the README endpoint summary and social preview from versioned
  aggregate thesis results and the published alpha network figure.
- [final_figures/build_pipeline_overview.py](final_figures/build_pipeline_overview.py)
- [final_figures/build_qc_valid_trials_figure.py](final_figures/build_qc_valid_trials_figure.py)
- [final_figures/build_network_heatmaps.py](final_figures/build_network_heatmaps.py)
- [final_figures/build_composite_breakdown.py](final_figures/build_composite_breakdown.py)
- [final_figures/build_endpoints_main_figure.py](final_figures/build_endpoints_main_figure.py)
- [final_figures/build_trials_threshold_sensitivity.py](final_figures/build_trials_threshold_sensitivity.py)

The `final_figures/` scripts are called by [run_paper_figures.py](../run_paper_figures.py).

## Table scripts

- [final_tables/build_cohort_main_table.py](final_tables/build_cohort_main_table.py)
- [final_tables/build_cohort_qc_table.py](final_tables/build_cohort_qc_table.py)

## Supplementary scripts

- [supplementary/make_fisher_z_tables.py](supplementary/make_fisher_z_tables.py)

This script reads the Fisher-z source CSVs in `supplementary/source_tables/`
and writes the LaTeX tables and optional PDF/PNG table renders used in the
supplementary material.

## Rule of thumb

If a script in this directory starts redefining the analysis itself, it is in
the wrong place. These scripts are for formatting, exporting, and assembling
paper-facing outputs from results that already exist.
