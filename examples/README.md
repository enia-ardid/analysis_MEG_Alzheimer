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

## Automated public smoke test

```bash
python -m unittest tests.test_public_smoke -v
```

This smoke test goes beyond format validation. It loads the public example
dataset, reconstructs the `trial x ROI x time` tensor for one subject, crops
the central analysis window, and computes the alpha-band subject-level
connectivity matrices. That makes it suitable for CI while keeping runtime
short and deterministic.

The example dataset is not intended for scientific interpretation.
