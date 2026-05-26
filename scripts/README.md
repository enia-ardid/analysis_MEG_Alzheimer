# Manuscript scripts

This directory contains the downstream scripts that turn pipeline outputs into
paper tables and figures.

## Figure scripts kept in the maintained path

- [final_figures/build_pipeline_overview.py](final_figures/build_pipeline_overview.py)
- [final_figures/build_qc_valid_trials_figure.py](final_figures/build_qc_valid_trials_figure.py)
- [final_figures/build_network_heatmaps.py](final_figures/build_network_heatmaps.py)
- [final_figures/build_composite_breakdown.py](final_figures/build_composite_breakdown.py)
- [final_figures/build_endpoints_main_figure.py](final_figures/build_endpoints_main_figure.py)
- [final_figures/build_trials_threshold_sensitivity.py](final_figures/build_trials_threshold_sensitivity.py)

These are the scripts called by [run_paper_figures.py](../run_paper_figures.py).

## Table scripts kept in the maintained path

- [final_tables/build_cohort_main_table.py](final_tables/build_cohort_main_table.py)
- [final_tables/build_cohort_qc_table.py](final_tables/build_cohort_qc_table.py)


## Supplementary scripts kept in the maintained path

- [supplementary/make_fisher_z_tables.py](supplementary/make_fisher_z_tables.py)

This script turns the canonical CSVs in `supplementary/source_tables/` into
LaTeX-ready supplementary tables and optional PDF/PNG table figures.

## Design rule

The scripts in this directory should not redefine the scientific pipeline. They
sit downstream of:

- [run_group_analysis.py](../run_group_analysis.py)
- [run_strong_hypotheses.py](../run_strong_hypotheses.py)

Their job is to format and export paper-facing outputs from those results.
