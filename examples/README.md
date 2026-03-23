# Public Example Dataset

This directory contains a tiny Brainstorm-like ROI time-series dataset that can
be distributed publicly. Its purpose is practical:

- validate the expected `.mat` structure
- let external users smoke-test the loader and the main pipeline
- provide a concrete example of the `Value` / `Time` / `Atlas` contract

It is not intended to reproduce the manuscript results. The example subjects
are synthetic and deliberately small:

- `2` converters
- `2` non-converters
- `2` trials per subject
- `102` ROI labels
- `8000` samples per trial

## Layout

```text
examples/
├── README.md
├── build_example_dataset.py
└── brainstorm_roi_small/
    └── data/
        ├── C_example.mat
        └── NC_example.mat
```

## Regenerate the example files

```bash
python examples/build_example_dataset.py
```

## Validate the example format

```bash
python run_validate_inputs.py \
  --data-root examples/brainstorm_roi_small/data \
  --require-paper-atlas
```

## Smoke-run the pipeline

```bash
python run_group_analysis.py \
  --data-root examples/brainstorm_roi_small/data \
  --output-root examples/brainstorm_roi_small/outputs \
  --hypothesis-n-perm 100 \
  --hypothesis-n-boot 200
```

The run above is only a structural smoke test. With such a small synthetic
cohort, the resulting group statistics are not scientifically interpretable.
