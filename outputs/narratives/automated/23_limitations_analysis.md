# Step 23 — Limitations Analysis

This report quantifies key threats to validity and fairness:
1) **Data quality** — missingness in core fields and protected attributes.
2) **Representation** — coverage & balance across groups (entropy/Gini).
3) **Temporal drift** — changes in Black-women share over time; Simpson’s effect.
4) **Category sparsity** — reliance on a long-tail taxonomy and its implications.
5) **Lexicon dependency** — reliance on HurtLex categories (proxy).

## Checklist (Flags & Metrics)
```
                                                                 issue  flag  metric_detail
                                 High missingness in key fields (>20%) False       0.000000
                 Strong representation imbalance (entropy_norm < 0.80)  True       0.629300
                       Temporal drift in BW share (|slope| > 0.005/yr) False       0.000938
                      Severe long-tail (Top-10 categories cover < 50%) False       0.720732
                 HurtLex dependency high (median max-prevalence ≥ 10%) False            NaN
Simpson’s flip detected (overall vs per-year correlation sign differs) False      -0.015600
```

## Notes & Mitigations (literature-informed)
- **High missingness** → impute with uncertainty; report explicitly (Datasheets; Model Cards).
- **Representation imbalance** → stratified evaluation; reweighing; careful augmentation (Suresh & Guttag).
- **Temporal drift** → time-aware splits; report year-conditioned metrics; monitor drift regularly.
- **Long-tail** → validate head vs tail; avoid category-only features driving bias.
- **Lexicon dependency** → combine lexicons with contextual models; audit with manual samples.

*Seed=95. Figures in `outputs/figures/(dark|light)`; tables in `dissertation/auto_tables`. Self-check mode writes `_selfcheck` artefacts only.*

_Reminder: category totals can exceed N due to multi-label assignment. Non-English titles are common; tags/categories remain interpretable (MPU)._
## Quick interpretation notes
- **Missingness**: bars near 0 mean those fields are complete; ≥0.2 flags data quality risk.
- **BW Share Over Time**: upward slope = increasing representation; we also report Simpson’s flip check.
