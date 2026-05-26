# Supplementary Assets

This directory contains the Fisher-z supplementary material in the form used by
the manuscript.

Contents:
- `source_tables/`: source CSVs used to build the supplementary tables
- `tables/`: LaTeX tables ready for `\input{...}`
- `figures/`: optional PDF/PNG renders of those same tables

Regenerate everything from the repository root with:

```bash
python scripts/supplementary/make_fisher_z_tables.py
```

Explicit full command:

```bash
python scripts/supplementary/make_fisher_z_tables.py \
  --input-dir supplementary/source_tables \
  --supplementary-dir supplementary/tables \
  --figure-dir supplementary/figures
```
