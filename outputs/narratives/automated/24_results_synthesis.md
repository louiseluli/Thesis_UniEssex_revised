# Step 24 — Results Synthesis

This synthesis collates ablation sensitivity (Step 22) and limitations (Step 23).

## Checklist
| issue                                      | flag   |     metric |
|:-------------------------------------------|:-------|-----------:|
| Representation entropy low (race)          | True   |   0.7186   |
| Representation entropy low (gender)        | True   |   0.6293   |
| Long-tail severe (Top-10 < 50%)            | False  |   0.720732 |
| High lexicon dependency (median max ≥ 10%) | False  | nan        |
| Bootstrap CI width improved (200→1000)     | False  | nan        |

## Highlights & Interpretation
- **Representation**: entropy/gini summarize balance; outlier max shares suggest dominant groups.
- **Long-tail**: Top-K mass shows category head coverage; diminishing returns beyond ~30 are typical.
- **Lexicon dependency**: high median max-prevalence suggests results sensitive to HurtLex; corroborate with context models.
- **Uncertainty**: bootstrap CI widths shrink with larger N_BOOT; stability improves accordingly.

_Totals can exceed N due to multi-label assignment. Titles may be in other languages, but tags/categories (MPU) preserve interpretable semantics. Years printed as integers; ratings shown with one decimal._

*Seed=95. Figures saved under `outputs/figures/(dark|light)`; tables in `dissertation/auto_tables`. Self-check mode writes `_selfcheck` artefacts only.*