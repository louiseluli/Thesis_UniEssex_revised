# 22 — Ablation Studies
- Baseline KPI (mean |log2 RR| over top-30 categories): **1.4709**.
- Scenarios: Lexicon OFF, random category noise (10/25/50%), and Top-K coverage (10/20/30/50).
- Deltas (scenario − baseline) are saved under outputs/ablation/ as one-row CSVs, plus a summary table.
- Multi-label categories imply totals can exceed N. Non-English titles are common; tags/categories (MPU) anchor semantics.
- Seed=95 ensures reproducibility; self-check writes *_selfcheck.csv only.
