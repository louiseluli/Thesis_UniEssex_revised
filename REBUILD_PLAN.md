# Thesis Pipeline — Rebuild Plan

**Canonical run date:** 2026-05-20  
**Purpose:** Step-by-step instructions to reproduce all results from scratch on a new machine, and to extend the analysis for PhD applications.

---

## 0. Before You Start

### What you need

| Requirement | Notes |
|:---|:---|
| `redtube_videos.db` | Raw SQLite database — **not in the repo**. Copy from your personal drive to `data/`. |
| Python 3.12+ | Use the pinned lock file for exact reproduction. |
| 32 GB RAM | Required for full corpus runs (535,236 records). |
| GPU (optional) | Significantly accelerates steps 09 (DistilBERT). |

### What is already committed

All `outputs/data/*.csv` files from the canonical run are committed. You only need to re-run the pipeline if:
- You want to regenerate **figures** (`.png`) — excluded from git
- You want to re-run **DistilBERT** (step 09) on the corrected corpus — pending
- You are working on **new experiments** (PhD extension)

### Critical fixes applied in the canonical run (2026-05-20)

These bugs are already fixed in `src/`. Do **not** revert them:

1. **`clean_text()` comma-fix** — a missing comma caused incorrect field parsing in `src/data/01_corpus_builder.py`
2. **`is_animated` flag** — animated content was not filtered; now excluded before modelling
3. **`rating_clean` column** — incoherent rating records (e.g. `ratings=0` but `rating > 0`) set to `NaN` rather than carried forward
4. **`is_duplicate` flag** — near-duplicate videos detected and excluded from the ML corpus
5. **`video_categories.video_id` index** — database index added; reduces join time from ~4 min to ~8 sec

---

## 1. Environment Setup

```bash
# Clone the repo (personal account only — repo is private)
git clone https://github.com/louiseluli/thesis.git
cd thesis

# Copy the raw database (from personal drive / encrypted backup)
cp /path/to/backup/redtube_videos.db data/

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies (exact reproduction)
pip install -r requirements-lock.txt
```

---

## 2. Phase 1 — Data & Bias Discovery (Steps 01–06)

Run in order. Each step writes to `outputs/data/` and is safe to re-run without affecting later steps (they read from the database, not from each other's outputs — except step 06 which depends on step 01).

```bash
# 01 — Build the canonical ML corpus (535,236 videos, 52 features, 111 categories)
#      Applies: clean_text fix, is_animated filter, rating_clean, is_duplicate flag
#      Writes: outputs/data/01_ml_corpus.parquet, 01_corpus_stats.json
python src/data/01_corpus_builder.py

# 02 — Comprehensive EDA: representation, engagement, quality, language, seasonality
#      Writes: ~25 CSV files prefixed outputs/data/02_*
python src/analysis/02_comprehensive_eda.py

# 03 — PMI intersectional profiling: tag co-occurrence bias for Black Women
#      Writes: outputs/data/03_pmi_intersectional_black_women*.csv
python src/analysis/03_intersectional_profiling.py

# 04 — Multilayer harm analysis: HurtLex mapping to video metadata
#      Writes: outputs/data/04_harm_category_by_group*.csv
python src/analysis/04_multilayer_harm_analysis.py

# 05 — Statistical bias tests: Kruskal-Wallis, Mann-Whitney for views and ratings
#      Writes: outputs/data/05_bias_tests_*.csv
python src/analysis/05_statistical_bias_tests.py

# 06 — Stratified train/val/test split (seed 95, deterministic)
#      Depends on: outputs/data/01_ml_corpus.parquet
#      Writes: outputs/data/06_train_ids.csv, 06_val_ids.csv, 06_test_ids.csv
python src/data/06_stratified_splitting.py
```

**Selfcheck mode** (fast validation, ~5 min, does not overwrite canonical outputs):
```bash
python src/data/01_corpus_builder.py --selfcheck
python src/data/06_stratified_splitting.py --selfcheck
```

---

## 3. Phase 2 — Modelling & Fairness Interventions (Steps 07–13)

Each step depends on the split IDs from step 06. Run in order.

```bash
# 07 — Random Forest baseline (seed 95)
#      Canonical result: acc 0.878, F1 0.758; Black Women EOD 0.191
#      Writes: outputs/data/07_*.csv, outputs/models/rf_baseline.pkl
python src/models/07_rf_baseline.py

# 08 — Comprehensive fairness evaluation at operating threshold
#      Writes: outputs/data/08_*.csv (group metrics, confusion matrix, CADP curve)
python src/fairness/08_comprehensive_evaluation.py

# 09 — DistilBERT baseline  ⚠️ PENDING RE-RUN on corrected corpus
#      Last run pre-fix; canonical acc 0.926 is an upper bound, not verified
#      Estimated time: 2–4 hours on GPU, 12–24 hours on CPU
#      Writes: outputs/data/09_*.csv, outputs/models/bert_baseline/
python src/models/09_bert_baseline.py

# 10 — Pre-processing: Reweighing
#      Canonical result: acc 0.953, F1 0.916; Black Women EOD 0.076 (−60%)
#      Writes: outputs/data/10_reweigh_*.csv
python src/fairness/10_preprocessing_mitigation.py

# 11 — In-processing: Exponentiated Gradient (Demographic Parity constraint)
#      Writes: outputs/data/11_inproc_*.csv
python src/fairness/11_inprocessing_mitigation.py

# 11b — Adversarial debiasing  ⚠️ DOCUMENTED FAILURE
#       Asian Women EOD worsens 0.105 → 0.283 due to feature representation problem
#       Writes: outputs/data/11b_adversarial_*.csv
#       (already committed; only re-run if you modify the adversarial architecture)

# 12 — Post-processing: ThresholdOptimizer (Equalized Odds)
#      Canonical result: acc 0.837; EOD range −0.018 to 0.000 (near-exact parity)
#      Writes: outputs/data/12_postproc_*.csv
python src/fairness/12_postprocessing_mitigation.py

# 13 — Mitigation effectiveness synthesis + Pareto frontier
#      Writes: outputs/data/25_pareto_points.csv, 25_pareto_frontier.csv
python src/fairness/13_mitigation_effectiveness.py
```

---

## 4. Phase 3 — Advanced Analyses (Steps 15–23)

These are independent of Phase 2 and can run in parallel once Phase 1 is complete.

```bash
# 15 — Qualitative error analysis: top-k confident mistakes
python src/analysis/15_qualitative_deep_dive.py

# 16 — Temporal analysis: representation and rating/view trends over time
#      Key output: slope gap rating = −3.38/year (Black Women faster decline)
python src/analysis/16_deep_data_analysis.py

# 17 — Engagement bias: head/tail composition, yearly gaps, bootstrap CIs
python src/analysis/17_engagement_bias_analysis.py

# 18 — Category group dynamics: stereotyping, over/under-representation
python src/analysis/18_category_group_dynamics.py

# 19 — Advanced statistics: KL divergence, effect sizes, trend slopes
#      Key output: 19_trend_slopes.csv (verified: slope_gap_rating = −3.38)
python src/analysis/19_advanced_statistics.py

# 20 — Network analysis: category co-occurrence graph, hub centrality
python src/analysis/20_network_analysis.py

# 22 — Ablation studies: sensitivity to seed, feature removal, threshold
python src/experiments/22_ablation_studies.py

# 23 — Limitations analysis: long-tail, missingness, temporal drift
python src/analysis/23_limitations_analysis.py
```

---

## 5. Phase 4 — Synthesis & Final Outputs (Steps 24–30)

Run after all prior phases complete.

```bash
# 24 — Results synthesis: answers all five RQs with evidence
python src/dissertation/24_results_synthesis.py

# 25 — Pareto frontier plot (requires Phase 2 complete)
python src/dissertation/25_pareto_frontier.py

# 26 — Executive summary document (auto-generated)
python src/dissertation/26_executive_summary.py

# 28 — Q&A backup slides
python src/presentation/28_qa_backup_slides.py

# 30 — Causal analysis: PSM + IPW for treatment effect estimation
python src/fairness/causal/30_psm_ipw.py
```

---

## 6. Pending Tasks (Before PhD Applications)

These are the outstanding items that need attention before this work can be cited in a PhD application or paper submission:

| Priority | Task | Effort | Notes |
|:---:|:---|:---:|:---|
| 🔴 High | Re-run DistilBERT (step 09) on corrected corpus | ~4 hrs GPU | Current BERT numbers are pre-fix; cannot cite as verified |
| 🔴 High | Generate all figures (steps 25, 17, 20) | ~30 min | `.png` files excluded from git; regenerate locally |
| 🟡 Medium | Re-run matched-N reweighing sensitivity (10b) | ~1 hr | Validates that reweighing gains are not sample-size artefacts |
| 🟡 Medium | Update `RESULTS_LEDGER.md` after BERT re-run | — | Move BERT from `pending_check` to `verified` |
| 🟢 Low | Add confidence intervals to Pareto frontier plot | ~2 hrs | Strengthens the adversarial failure narrative |
| 🟢 Low | `27_ground_truth.py` — complete annotation template | — | For future supervised fairness evaluation |

---

## 7. Working on a New Machine (Work Computer)

Since the database cannot be in the repo, use this transfer checklist:

```
□ Copy redtube_videos.db from personal encrypted drive → data/
□ Verify: python -c "import sqlite3; c=sqlite3.connect('data/redtube_videos.db'); print(c.execute('SELECT COUNT(*) FROM videos').fetchone())"
  Expected: (535236,)  [or close — depends on dedup run]
□ git pull origin main  (to get latest canonical CSVs)
□ source .venv/bin/activate && pip install -r requirements-lock.txt
□ Run selfcheck: python src/data/01_corpus_builder.py --selfcheck
  (Should complete in ~5 min and match outputs/data/01_corpus_selfcheck.csv)
```

---

## 8. Pushing to Private GitHub (Personal Account)

The repo remote points to `https://github.com/louiseluli/thesis.git`.  
The `gh` CLI on this machine is authenticated as `louiseferreira` (work). To push:

**Option A — Temporary personal token (recommended for work machine):**
```bash
# Generate a PAT at https://github.com/settings/tokens (louiseluli account)
# Scopes needed: repo
git remote set-url origin https://louiseluli:<YOUR_PAT>@github.com/louiseluli/thesis.git
git push origin main

# After pushing, remove the token from the URL for safety:
git remote set-url origin https://github.com/louiseluli/thesis.git
```

**Option B — Add personal account to gh CLI:**
```bash
gh auth login --hostname github.com
# Follow prompts, log in as louiseluli
gh repo edit louiseluli/thesis --visibility private --accept-visibility-change-consequences
git push origin main
```

**Make repo private (do this before or immediately after first push):**
```bash
# Via gh CLI (as louiseluli):
gh repo edit louiseluli/thesis --visibility private --accept-visibility-change-consequences

# Or via browser:
# https://github.com/louiseluli/thesis/settings → Danger Zone → Change visibility → Private
```
