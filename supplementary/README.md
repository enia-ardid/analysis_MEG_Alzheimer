# Supplementary Assets

This directory contains supplementary assets intended for direct inclusion in the thesis manuscript.

Contents:
- `source_tables/`: canonical CSV sources for the Fisher-z supplementary tables.
- `tables/`: LaTeX tables ready for `\input{...}`.
- `figures/`: optional rendered PDF/PNG table figures.

Regenerate the Fisher-z supplementary assets from the repository root with:

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
