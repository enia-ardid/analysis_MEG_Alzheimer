# Visual assets and provenance

[Back to the project](../README.md)

These assets include original saved notebook figures, a workflow diagram and a
visualization of versioned supplementary tables. They do not change analysis
definitions or add new scientific results.

| Asset | Use | Source |
| --- | --- | --- |
| [social-preview.png](assets/social-preview.png) · [SVG](assets/social-preview.svg) | Original figures placed side by side for the README and GitHub; 1280 × 640 px | Saved alpha PLV matrix and graph from the first notebook revision |
| [original-alpha-plv-matrix.png](assets/original-alpha-plv-matrix.png) | Unmodified original matrix figure; 444 × 390 px | First PNG output in notebook cell 8 |
| [original-alpha-plv-graph.png](assets/original-alpha-plv-graph.png) | Unmodified original graph figure; 590 × 590 px | First PNG output in notebook cell 9 |
| [fisher-z-sensitivity.png](assets/fisher-z-sensitivity.png) · [SVG](assets/fisher-z-sensitivity.svg) | Detailed comparison; 1600 × 920 px | The endpoint comparison CSV, with absolute differences calculated from its rounded values |
| [pipeline-overview.svg](assets/pipeline-overview.svg) | Diagram of the existing analysis | Dataset, signals, connectivity, pipeline, atlas and strong-hypothesis modules |

The social preview arranges the two original PNGs without redrawing their
measurements, labels or connections. Its SVG embeds those PNG bytes. The
Fisher-z figure and workflow diagram were rendered separately with Matplotlib;
their SVGs contain vector shapes and outlined text.

## Original notebook figures

The matrix and graph were extracted from saved outputs in
[analysis_alzheimer.ipynb at commit 3a67f65](https://github.com/enia-ardid/analysis_MEG_Alzheimer/blob/3a67f653e0f0a9027375cd711dd7ac0c6970486c/analysis_alzheimer.ipynb).
Cell indices below are zero-based:

1. Cell 6 loads a study MATLAB export with `scipy.io.loadmat`. Its saved output
   reports `Value` shape `(3570, 8000)`, reconstructed shape `(35, 102, 8000)`
   and cropped shape `(35, 102, 4000)`.
2. Cell 7 computes connectivity per trial and averages the matrices within that
   subject.
3. Cell 8's first PNG shows the alpha-band PLV matrix.
4. Cell 9's first PNG shows the corresponding graph, retaining connections at
   or above the upper-triangle 95th percentile. Its saved title reports
   `top ~5%, thr≈0.742`.

These are historical **single-subject exploratory figures**. Their recorded
computation uses the study MATLAB input rather than the later public synthetic
example. The MATLAB file is not available in this checkout, so the figures have
not been recomputed or independently validated against the raw signals. They
must not be described as final cohort results, network-level group differences
or a rerun of the current code.

The extracted originals match the notebook's decoded PNG outputs byte for byte:

| Original PNG | SHA-256 |
| --- | --- |
| `original-alpha-plv-matrix.png` | `546cc3f13c50588bf09d652c3a58afa067bdd6a8843f349777d6fa0efcd4cf98` |
| `original-alpha-plv-graph.png` | `f2084a9174b5bdacf9ccef222b305408640c3bf0bb4e8ba1b113c74f436c6187` |

The preview preserves the complete images, including their original titles,
axes and color scale, on a white canvas. Resizing them for the preview does not
recover resolution beyond that of the saved notebook exports.

## Effect-size comparison

Source:
[table_fisher_z_endpoint_comparison.csv](../supplementary/source_tables/table_fisher_z_endpoint_comparison.csv).

The separate Fisher-z figure uses `Endpoint`, `Cohen d mean-r` and
`Cohen d Fisher-z`. Effect sizes retain the **Converter − Non-converter** sign
convention; they have not been flipped or normalized for presentation.

| Endpoint | Mean-r d | Fisher-z d | Absolute change in d |
| --- | ---: | ---: | ---: |
| H1 AEC | -0.5870 | -0.5870 | 0.0000 |
| H1 AEC-orth | -0.5026 | -0.5028 | 0.0002 |
| H2 AEC | -0.4356 | -0.4366 | 0.0010 |
| H2 AEC-orth | -0.5627 | -0.5630 | 0.0003 |
| H3 gap full | -0.5749 | -0.5777 | 0.0028 |
| H3 gap inter | -0.5446 | -0.5482 | 0.0036 |

The detailed figure's summary annotations are calculated as follows:

- **6 / 6 directions preserved:** the signs of the two effect sizes agree in
  every row.
- **0.0036 maximum absolute change:** `max(abs(d_fisher_z − d_mean_r))`, using
  the displayed precision of the CSV. A displayed zero does not imply exact
  equality before rounding.
- **Pearson r ≥ 0.999854:** the minimum of the CSV's
  `Pearson r(mean-r, Fisher-z)` column. These are correlations between the two
  sets of subject-level endpoint values, not brain-region connectivity values.

Filled teal points represent mean-r and open amber points represent Fisher-z.
Vertical offsets keep near-identical values distinguishable and have no
quantitative meaning. The right panel uses lines and points to show absolute
changes. **This figure does not show confidence intervals.**

The figure does not show p-values. The CSV's `p1` columns are nominal one-sided
tests, and agreement between averaging methods does not imply family-wise
significance or clinical prediction performance. The main endpoint inference
uses the separate Holm and max-T procedures in
[strong_hypotheses.py](../meg_alzheimer/strong_hypotheses.py).

Check the displayed numbers from the repository root with Python's standard
library:

```python
import csv
from decimal import Decimal
from pathlib import Path

source = Path("supplementary/source_tables/table_fisher_z_endpoint_comparison.csv")
with source.open(newline="") as stream:
    rows = list(csv.DictReader(stream))

changes = []
same_direction = 0
for row in rows:
    raw = Decimal(row["Cohen d mean-r"])
    fisher = Decimal(row["Cohen d Fisher-z"])
    change = abs(fisher - raw)
    changes.append(change)
    same_direction += (raw > 0) == (fisher > 0) and (raw < 0) == (fisher < 0)
    print(row["Endpoint"], f"{change:.4f}")

print("Same direction:", same_direction, "/", len(rows))
print("Maximum absolute change in d:", max(changes))
print("Minimum Pearson r:", min(
    Decimal(row["Pearson r(mean-r, Fisher-z)"]) for row in rows
))
```

Fisher-z aggregation follows
[build_subject_endpoint_table](../meg_alzheimer/robustness.py): average component
correlations in Fisher-z space, transform them back to r, then reconstruct the
derived H3 gaps. The supplementary CSV is already an aggregate result; the
underlying cohort cannot be recomputed from this table alone.

## Workflow diagram

The six stages summarize the existing code in processing order:

1. [dataset.py](../meg_alzheimer/dataset.py): reconstruct the trial tensor and
   crop samples `2000:6000` from each 8-second, 1000 Hz trial.
2. [signals.py](../meg_alzheimer/signals.py): delta `1–4`, theta `4–8`, alpha
   `8–12`, beta `13–30` and gamma `30–48` Hz; FIR filtering and Hilbert signals.
3. [connectivity.py](../meg_alzheimer/connectivity.py): PLV, AEC and symmetric
   AEC-orth per trial and band.
4. [pipeline.py](../meg_alzheimer/pipeline.py): average trial matrices within
   subject before comparing groups.
5. [atlas.py](../meg_alzheimer/atlas.py): summarize the 100 parcels into nine
   network families, excluding two background labels.
6. [strong_hypotheses.py](../meg_alzheimer/strong_hypotheses.py): construct the
   six H1–H3 endpoints and run the family-wise inference.

The diagram is a method illustration, not a visualization of measured brain
activity. Sensor preprocessing, MRI handling and source reconstruction occur
upstream of this repository.
