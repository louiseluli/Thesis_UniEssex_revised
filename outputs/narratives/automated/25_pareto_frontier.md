# Step 25 — Pareto Frontier

- Total models considered: **5**
- Pareto-optimal models: **2**
- Frontier set: RF reweighed, RF postproc

Interpretation:
- Points further to the upper-right jointly improve accuracy and fairness.
- Frontier models are **non-dominated** (no other model strictly better on both axes).
- Large gaps between neighbors on the frontier suggest headroom (algorithm/mitigation).

Notes:
- Totals in other steps can exceed *N* due to multi-label assignment; here we work per model.
- Titles can be non-English; tags/categories (MPU) keep semantics interpretable.

*Seed=95. Figures in `outputs/figures/(dark|light)`; tables in `dissertation/auto_tables`. Self-check mode writes `_selfcheck` artefacts only.*