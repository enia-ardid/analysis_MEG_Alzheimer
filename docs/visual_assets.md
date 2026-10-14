# Figures and provenance

[Back to the project](../README.md) · [Results gallery](results.md)

| Asset | Source | Role |
| --- | --- | --- |
| [Alpha network connectivity](assets/thesis-alpha-networks.png) | Thesis Figure 3a | Cohort means and group differences across nine networks |
| [Composite breakdown](assets/thesis-composite-breakdown.png) | Thesis Figure 5 | Contributions of network pairs to H1 and H2 |
| [Endpoint effect sizes](assets/thesis-endpoint-effects.svg) · [PNG](assets/thesis-endpoint-effects.png) | Thesis Table 2 | Six effect-size estimates and corrected p-values |
| [Fisher-z sensitivity](assets/fisher-z-sensitivity.svg) · [PNG](assets/fisher-z-sensitivity.png) | Versioned supplementary CSV | Comparison of aggregation methods |
| [Pipeline overview](assets/pipeline-overview.svg) | Analysis modules | Processing steps from ROI signals to endpoints |
| [Social preview](assets/social-preview.png) · [SVG](assets/social-preview.svg) | Alpha network figure and project description | 1280 × 640 repository preview |

## Thesis sources

Enia Ardid Prats, *Caracterización de la progresión del Alzheimer mediante
conectividad funcional en MEG*. Physics BSc thesis, Universidad Complutense de
Madrid, 2026. Supervisor: Gianluca Susi.

The alpha network figure comes from printed page 16 (PDF page 17), and the
composite breakdown from printed page 19 (PDF page 20). Both are rendered from
figure regions in the original PDF, preserving plotted values, labels and color
scales. The PNG exports omit text and EXIF metadata. Exporting at a larger pixel
size does not restore detail in elements already rasterized in the source.

The source PDF's SHA-256 is
`3d99ccfd786071f5f75eff6c1138ef271ef251a546b795046bae11c86ab1dde2`.
The [figure manifest](assets/thesis-figures.json) records page numbers, extraction
bounds, image dimensions and output hashes. The full PDF and participant-level
records are not part of this release.

The endpoint plot is built from
[table_thesis_endpoints.csv](../supplementary/source_tables/table_thesis_endpoints.csv),
transcribed from thesis Table 2 on printed page 18 (PDF page 19). The table
retains the displayed precision of the source, including rounded p-values.
Cohen's d uses the Converter − Non-converter sign convention. Filled points
identify endpoints reported to survive Holm correction with max-T p < 0.05.
The source table does not provide effect-size confidence intervals, so the
plot shows point estimates only.

These are results of the thesis analysis. They have not been recalculated from
the private cohort with the current repository revision.

## Analysis settings

| Setting | Thesis report | Current code default |
| --- | --- | --- |
| Alpha band | 8–12 Hz | 8–12 Hz |
| Beta band | 12–30 Hz | 13–30 Hz in `band_defs()` |
| Endpoint max-T permutations | 10,000 | 50,000 in `run_strong_hypotheses.py` |
| Group comparison | Converter − Non-converter | Converter − Non-converter |

The beta-band difference requires the original execution configuration to
resolve. The figures are retained as reported, with no relabeling or inferred
recalculation. The permutation count is configurable in the current code.

The supplementary simulation now uses the same uncapped AEC-orth definition as
the cohort pipeline. Simulation outputs from the former capped implementation
need regeneration; the runner detects their older cache format. See the
[simulation documentation](../simulation_aec_snr_bias/README.md#metric-correction-and-cached-results).

## Effect-size comparison

The Fisher-z figure uses `Endpoint`, `Cohen d mean-r` and `Cohen d Fisher-z` from
[the supplementary endpoint comparison](../supplementary/source_tables/table_fisher_z_endpoint_comparison.csv).

| Endpoint | Mean-r d | Fisher-z d | Absolute change in d |
| --- | ---: | ---: | ---: |
| H1 AEC | −0.5870 | −0.5870 | 0.0000 |
| H1 AEC-orth | −0.5026 | −0.5028 | 0.0002 |
| H2 AEC | −0.4356 | −0.4366 | 0.0010 |
| H2 AEC-orth | −0.5627 | −0.5630 | 0.0003 |
| H3 gap full | −0.5749 | −0.5777 | 0.0028 |
| H3 gap inter | −0.5446 | −0.5482 | 0.0036 |

All six signs agree. The maximum absolute change, 0.0036, is calculated from the
rounded CSV values. The minimum Pearson correlation between the two sets of
subject-level endpoint values is 0.999854; this is not a brain-region
connectivity value. Small vertical offsets separate overlapping markers and
have no quantitative meaning. Confidence intervals are not shown.

Fisher-z aggregation follows
[build_subject_endpoint_table](../meg_alzheimer/robustness.py): average component
correlations in Fisher-z space, transform them back to r, then reconstruct the
H3 gaps. Agreement between methods addresses aggregation sensitivity and does
not replace the Holm and max-T analysis.

## Workflow diagram

The diagram summarizes [dataset loading](../meg_alzheimer/dataset.py),
[signal processing](../meg_alzheimer/signals.py),
[connectivity](../meg_alzheimer/connectivity.py),
[within-subject averaging](../meg_alzheimer/pipeline.py),
[network aggregation](../meg_alzheimer/atlas.py) and
[endpoint inference](../meg_alzheimer/strong_hypotheses.py).
Sensor preprocessing and source reconstruction occur upstream.

## Original notebook figures

The earlier [PLV matrix](assets/original-alpha-plv-matrix.png) and
[PLV graph](assets/original-alpha-plv-graph.png) remain available as exploratory
single-subject examples. They were extracted from saved outputs in
[the original notebook at commit 3a67f65](https://github.com/enia-ardid/analysis_MEG_Alzheimer/blob/3a67f653e0f0a9027375cd711dd7ac0c6970486c/analysis_alzheimer.ipynb).

The notebook reports 35 trials, 102 ROIs and a central four-second window. The
graph retains connections at or above the upper-triangle 95th percentile,
with a displayed threshold of approximately 0.742. These figures predate the
current pipeline and represent one subject, rather than cohort differences.

| Original PNG | SHA-256 |
| --- | --- |
| `original-alpha-plv-matrix.png` | `546cc3f13c50588bf09d652c3a58afa067bdd6a8843f349777d6fa0efcd4cf98` |
| `original-alpha-plv-graph.png` | `f2084a9174b5bdacf9ccef222b305408640c3bf0bb4e8ba1b113c74f436c6187` |

## Regeneration

Rebuild the endpoint plot and repository preview from the versioned assets:

```bash
python scripts/figures/build_project_assets.py
```

This command uses the aggregate endpoint CSV and the alpha figure already in
`docs/assets/`. It does not load participant data or rerun the cohort analysis.
The source exports are needed to recompute empirical connectivity figures;
see the [analysis reference](analysis_reference.md#paper-figure-set).
