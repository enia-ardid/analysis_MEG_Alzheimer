# Manuscript scripts

This directory contains the downstream scripts that regenerate manuscript
artifacts from the private local dataset and the outputs produced by the core
pipeline.

## Figure scripts kept in the maintained path

- [final_figures/build_pipeline_overview.py](final_figures/build_pipeline_overview.py)
- [final_figures/build_qc_valid_trials_figure.py](final_figures/build_qc_valid_trials_figure.py)
- [final_figures/build_network_heatmaps.py](final_figures/build_network_heatmaps.py)
- [final_figures/build_composite_breakdown.py](final_figures/build_composite_breakdown.py)
- [final_figures/build_endpoints_main_figure.py](final_figures/build_endpoints_main_figure.py)
- [final_figures/build_trials_threshold_sensitivity.py](final_figures/build_trials_threshold_sensitivity.py)

These scripts are the ones called by [run_paper_figures.py](../run_paper_figures.py).

## Table scripts kept in the maintained path

- [final_tables/build_cohort_main_table.py](final_tables/build_cohort_main_table.py)
- [final_tables/build_cohort_qc_table.py](final_tables/build_cohort_qc_table.py)

## Design rule

The scripts in this directory should not redefine the scientific pipeline.
They sit downstream of:

- [run_group_analysis.py](../run_group_analysis.py)
- [run_strong_hypotheses.py](../run_strong_hypotheses.py)

Their job is to format and export the paper-facing outputs from those results.
