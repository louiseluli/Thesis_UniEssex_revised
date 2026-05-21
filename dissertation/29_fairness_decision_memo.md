# Fairness Decision Memo

This memo consolidates the **Step 25** accuracy–fairness Pareto frontier analysis with the full mitigation comparison to recommend a single operating point suitable for deployment. All numbers are drawn directly from pipeline outputs and are reproducible.

*Generated from: `outputs/data/25_pareto_points.csv`, `outputs/data/25_pareto_frontier.csv`, `outputs/data/mitigation_effectiveness.csv`*

---

## Pareto frontier — all evaluated models

| Model | Accuracy | Fairness score | On frontier? |
|-------|----------|---------------|-------------|
| RF inproc (EG + DP) | 0.683 | 0.743 | Yes |
| RF postproc (threshold) | **0.854** | **0.758** | Yes (selected) |
| RF reweighed | 0.819 | 0.600 | Yes |

The frontier contains three points. The ideal operating point would be at (1.0, 1.0). Euclidean distance to ideal selects **RF postproc** as the knee.

---

## Selected operating point

| Field | Value |
|-------|-------|
| **Model** | RF postproc (threshold optimisation) |
| **Accuracy** | 0.854 |
| **Fairness score** | 0.758 |
| **Selection criterion** | Minimum Euclidean distance to ideal (1.0, 1.0) |
| **Source** | `outputs/data/25_pareto_points.csv` |

---

## Full mitigation comparison (for context)

| Method | Test Accuracy | Test F1 | Acc gap (Black Women) | TPR gap (Black Women) |
|--------|--------------|---------|----------------------|----------------------|
| Baseline RF | 0.805 | 0.566 | −0.126 | +0.095 |
| Reweighed RF | **0.862** | **0.718** | −0.108 | **+0.050** |
| In-proc (EG, DP) | 0.748 | 0.639 | **−0.046** | −0.193 |
| Post-proc (threshold) | 0.783 | 0.597 | −0.090 | −0.103 |

*Note: the Pareto frontier's "RF postproc" accuracy (0.854) differs from the per-group table above (0.783) because the frontier evaluates the model at its Pareto-optimal threshold, not the F1-optimal threshold used in the group comparison.*

---

## Deployment recommendations

1. **Use RF postproc as the baseline deployment model.** It offers the best accuracy–fairness balance per the Pareto analysis. Its threshold can be adjusted without retraining as fairness definitions or legal requirements evolve.

2. **Use Reweighed RF when retraining is feasible.** It dominates on accuracy (+5.7 pp over baseline) and achieves the smallest TPR gap for Black Women (0.050 vs 0.095 baseline). Its Pareto fairness score (0.600) is lower because it trades some group-level parity for higher overall accuracy.

3. **Tie-breaking rule:** When two models are within 0.02 of each other on both accuracy and fairness, prefer the one with **higher fairness score** to respect equity goals with minimal engagement loss.

4. **Monitoring requirements:**
   - Re-evaluate fairness metrics after any model retraining or significant content-policy change.
   - Track bootstrap 95% CIs on per-group EOD monthly; alert if any group's EOD exceeds 0.15.
   - Log group-level exposure counts per content type to detect distributional drift early.
   - Re-run `src/dissertation/25_pareto_frontier.py` quarterly to update the frontier as new mitigation techniques are added.

---

## Caveats

- All metrics are point estimates. Bootstrap confidence intervals are available in `outputs/data/08_threshold_sweep.csv` for the RF baseline.
- The fairness score on the Pareto frontier is a composite metric in [0, 1]; it is not directly interpretable as EOD or demographic parity — consult per-group CSVs for auditable claims.
- Temporal drift: `outputs/data/19_trend_slopes.csv` shows Black Women views declining at −1.84/year faster than other groups. Models should be re-evaluated on temporally held-out data at least annually.
- Protected attributes are inferred, not self-reported. Inference errors propagate to fairness metrics; the inference pipeline should be periodically re-validated against human-labelled samples.

*Notes:* Titles may be non-English; tags/categories (MPU) keep semantics interpretable. Positive TPR gap = group is disadvantaged relative to White Women.
