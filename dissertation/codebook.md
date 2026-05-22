# Codebook — Source File Reference

**Project:** AlgoFairness Pornometrics — MSc Dissertation, University of Essex 2025  
**Total source files:** 49 Python files across 8 modules  
**Seed:** 95 (fixed globally in `config/settings.yaml`)

> Every script that writes outputs respects the `--selfcheck` flag: it samples a random subset and writes only `*_selfcheck.*` artefacts, never overwriting canonical results.

---

## How to Read This Document

Each entry follows this structure:

```
### step_number — filename.py
Phase | Runtime
Purpose (what problem it solves)
Inputs → Outputs
Key functions
CLI usage
Dependencies (project modules)
Notes (caveats, known issues, extension points)
```

---

## Module: `src/data/` — Data Layer

---

### collector.py
**Phase:** Data collection (pre-pipeline) | **Runtime:** Up to 29,999 API calls/day (~8–14 hours for full corpus)

**Purpose.** Incrementally crawls the Redtube public API (`redtube.Videos.searchVideos`) by category, stores raw video metadata in SQLite, and resumes interrupted runs from saved state. It is the data source for the entire pipeline.

**Inputs:**
- Redtube public API (`https://api.redtube.com/`) — no authentication required
- `data/collector_state.json` — resume state (current category, page)
- `data/redtube_videos.db` — target SQLite database (created if absent)

**Outputs:**
- Rows written to `videos`, `video_tags`, `video_categories` tables in `data/redtube_videos.db`
- `data/collector_state.json` — updated after every page
- `collection_state` DB table — daily API request counter

**Key functions:**

| Function | What it does |
|:---------|:-------------|
| `make_session()` | Creates `requests.Session` with exponential backoff retry (5 attempts, status 429/5xx) |
| `load_or_init_rate_state(conn)` | Reads/creates daily rate-limit row; returns requests_used for today |
| `can_consume_requests(conn, n)` | Returns `(allowed, sleep_seconds)` — blocks until midnight UTC if daily cap hit |
| `_treatment_from_categories()` | Treatment indicator used by step 30 (Amateur presence) |
| `save_videos_to_db(conn, videos, cat)` | UPSERT into `videos`; INSERT-OR-IGNORE into `video_tags`, `video_categories` |
| `main()` | Outer loop: fetch categories → paginate → save → persist state |

**CLI:**
```bash
python src/data/collector.py                          # resume from state
python src/data/collector.py --ordering newest --new-only   # daily incremental (recommended)
python src/data/collector.py --category "Amateur" --start-page 60
python src/data/collector.py --reset                  # restart from page 1
```

**Key config (hardcoded at top of file):**
- `API_DAILY_LIMIT = 29999` — safety margin below hard 30k cap
- `REQUEST_DELAY = 0.35` — seconds between requests (polite pacing)
- `EXISTING_CAT_DUP_LIMIT = 9999` — consecutive duplicate pages before marking category complete

**Dependencies:** `src/utils/database.py`

**Notes:**
- The `.db-shm` and `.db-wal` files alongside the DB are SQLite WAL-mode journal files — normal, never delete them while the DB is in use.
- `--new-only` flag skips video IDs already in the `video_categories` table for the current category, making daily runs fast.
- Rate limit state is stored both in JSON (fine-grained resume) and in the DB `collection_state` table (authoritative daily counter).

---

### 01_corpus_builder.py
**Phase:** Phase 1 — Data | **Runtime:** ~4 minutes on full 535k corpus

**Purpose.** Transforms the raw SQLite database into a single clean Parquet file used by every downstream step. Applies all four canonical fixes: comma-separated token normalisation, animated content exclusion, incoherent rating nullification, and duplicate detection.

**Inputs:**
- `data/redtube_videos.db` — raw SQLite (tables: `videos`, `video_tags`, `video_categories`)
- `config/protected_terms.json` — demographic inference lexicon
- `config/settings.yaml` — paths, seed, feature generation keys

**Outputs:**
- `outputs/data/01_ml_corpus.parquet` — canonical ML corpus (535,236 rows, 52+ columns)
- `outputs/data/01_corpus_stats.json` — column list + protected group counts
- `outputs/data/01_corpus_selfcheck.csv` — 10-row random sample (always written)

**Key functions:**

| Function | What it does |
|:---------|:-------------|
| `clean_text(text)` | Lowercase → replace commas with spaces → strip non-alphanumeric → collapse whitespace. The comma replacement is the critical fix: prevents "white guyebony" concatenation artefacts. |
| `_compile_glob_terms_to_regex(terms, settings)` | Converts lexicon terms (with `*` wildcards) to a single alternation regex. Supports `casefold`, `word_boundaries`, `allow_hyphen_variants`. |
| `create_protected_group_features(df, text_col)` | Iterates lexicon groups → applies compiled regex → creates one-hot columns (`race_ethnicity_black`, `gender_female`, etc.) |
| `create_intersectional_features(df)` | Creates `intersectional_black_female = race_ethnicity_black AND gender_female` |
| `fetch_data_from_db(conn)` | Single correlated-subquery SQL to aggregate tags and categories per video without cross-multiplication |
| `quick_sanity_report(df)` | Prints top-10 categories and tags to stdout (no file write) |

**Columns produced (key ones):**

| Column | Type | Description |
|:-------|:-----|:------------|
| `combined_text_clean` | str | Normalised `title + tags + categories` (for lexicon matching) |
| `model_input_text` | str | Normalised `title + tags` (for ML model input) |
| `race_ethnicity_black` | int | 1 if any Black/Ebony lexicon term present |
| `gender_female` | int | 1 if any female-identifying term present |
| `intersectional_black_female` | int | 1 if both above |
| `is_animated` | int | 1 if animated tags present AND no real-person tags |
| `rating_clean` | float | NaN where `ratings > 0` but `rating == 0` (API artefact) |
| `is_duplicate` | int | 1 for second+ occurrence of `(title_lower, duration)` pair |

**CLI:**
```bash
python src/data/01_corpus_builder.py              # full run (always runs on full DB)
python src/data/01_corpus_builder.py --selfcheck  # no-op (no argparse; selfcheck CSV always written)
```

**Dependencies:** `src/utils/theme_manager`, `src/utils/database`

**Notes:**
- `--selfcheck` has no effect (script has no argparse). The 10-row selfcheck CSV is always written at the end of `main()`.
- Regex patterns for `is_animated` use non-capturing groups `(?:...)` — the capturing-group version caused pandas warnings in earlier runs (fixed 2026-05-21).
- Runtime is dominated by `create_protected_group_features` (~3.5 min) due to regex application over 535k rows.

---

### 06_stratified_splitting.py
**Phase:** Phase 1 — Data | **Runtime:** ~3 seconds

**Purpose.** Creates reproducible 60/20/20 train/validation/test splits with stratification on intersectional group labels. Saves only video_id lists — downstream scripts subset the parquet directly.

**Inputs:**
- `outputs/data/01_ml_corpus.parquet` — canonical corpus from step 01

**Outputs:**
- `outputs/data/06_train_ids.csv` — 321,141 video IDs
- `outputs/data/06_val_ids.csv` — 107,047 video IDs
- `outputs/data/06_test_ids.csv` — 107,048 video IDs
- `outputs/data/06_stratify_key_distribution.csv` — class counts used for stratification
- `outputs/data/06_split_selfcheck.csv` — 30-row sample with disjointness verification

**Key functions:**

| Function | What it does |
|:---------|:-------------|
| `make_stratify_key(df)` | Priority-ordered group labels: Black_Female > White_Female > Asian_Female > Latina_Female > Other. Each group is mutually exclusive. |
| `coalesce_rare_classes(key, min_count)` | Merges classes with < 50 members into "Other" to prevent stratification errors |
| `stratified_two_stage_split(df, key, ...)` | Stage 1: 80/20 train+val vs test. Stage 2: 75/25 train vs val (from the 80%). |
| `write_selfcheck_splits(...)` | Verifies train∩val=0, train∩test=0, val∩test=0 and writes a small sample |

**CLI:**
```bash
python src/data/06_stratified_splitting.py                              # full run
python src/data/06_stratified_splitting.py --selfcheck --sample 50000  # quick validation
python src/data/06_stratified_splitting.py --seed 42 --test-size 0.15  # custom params
```

**Dependencies:** `src/utils/theme_manager`

**Notes:**
- The priority ordering (Black first) matches `group_labels_intersectional()` in `fairness_evaluation_utils.py` — these must always agree.
- Split IDs are stable: re-running with the same seed and corpus produces identical splits.
- Multi-label counts elsewhere in the project can exceed N; here each video appears exactly once.

---

### `src/data_processing/feature_engineering.py`
**Phase:** Used by integration tests and model training | **Runtime:** Milliseconds per call

**Purpose.** Public API for converting a raw corpus DataFrame into ML-ready `(X, y, groups, fitted_scaler)`. Used primarily by the integration test suite; the numbered pipeline scripts implement their own feature engineering inline.

**Inputs:** DataFrame with `is_amateur` label column, optional text columns and numeric columns.

**Outputs:** Returns `(X: np.ndarray, y: np.ndarray, groups: np.ndarray, scaler: StandardScaler)`.

**Key functions:**

| Function | What it does |
|:---------|:-------------|
| `create_features(df, scaler=None)` | Full pipeline: text vectorisation → numeric standardisation → group label derivation. Pass a fitted scaler on val/test to avoid leakage. |
| `_build_group_labels(df)` | Priority-ordered intersectional group labels (same logic as `fairness_evaluation_utils.py`) |

**Notes:**
- The 4-tuple return `(X, y, groups, scaler)` is the current API. Tests that expect 3 values need `X, y, groups, _ = create_features(df)`.
- The scaler fitted on training data should be passed to val/test calls: `X_val, y_val, g_val, _ = create_features(df_val, scaler=fitted_scaler)`.

---

## Module: `src/analysis/` — Analysis Layer

---

### 02_comprehensive_eda.py
**Phase:** Phase 1 — EDA | **Runtime:** ~3–5 minutes

**Purpose.** Comprehensive exploratory data analysis covering representation, engagement, content quality, language, seasonality, and label structure. Produces ~25 CSVs, all dissertation LaTeX tables for EDA, and ~30 figures.

**Inputs:** `outputs/data/01_ml_corpus.parquet`

**Outputs (selected):**

| File | Content |
|:-----|:--------|
| `02_eda_views_per_day_by_group.csv` | Median views/day per intersectional group |
| `02_eda_views_disparities_stats.csv` | Full descriptive stats (mean, std, quartiles) by group |
| `02_eda_quality_proxies_by_group.csv` | HD%, 4K%, Verified Amateur% by group |
| `02_eda_top_protected_tags.csv` | Most frequent protected-group-associated tags |
| `02_eda_intersectional_representation.csv` | Group × year × orientation cross-tabulation |
| `dissertation/auto_tables/02_eda_*.tex` | LaTeX table for each CSV |

**CLI:**
```bash
python src/analysis/02_comprehensive_eda.py                              # full run
python src/analysis/02_comprehensive_eda.py --selfcheck --sample 80000  # fast validation
```

**Dependencies:** `src/utils/theme_manager`, `src/utils/academic_tables`

**Notes:**
- Requires `jinja2` (for `df.to_latex()`) — install with `pip install Jinja2`.
- The function `_intersectional_group_labels(df)` in this file uses the priority-ordered logic (fixed 2026-05-21) with `pd.NA` as default for unclassified videos.

---

### 03_intersectional_profiling.py
**Phase:** Phase 1 — Analysis | **Runtime:** ~2 minutes

**Purpose.** Computes Pointwise Mutual Information (PMI) between 1–2-gram text terms and each intersectional group. Identifies stereotypical tag associations and their statistical strength.

**Inputs:** `outputs/data/01_ml_corpus.parquet`

**Outputs:**
- `outputs/data/03_pmi_intersectional_black_women.csv` — top-N PMI terms for Black Women
- `outputs/data/03_pmi_intersectional_black_women_full.csv` — all terms with PMI > threshold
- `outputs/data/03_pmi_intersectional_black_women_outliers.csv` — unusually high-frequency terms
- `outputs/figures/pmi/03_pmi_associations_bar_{light,dark}.png`
- `outputs/narratives/automated/03_pmi_summary.md`

**PMI formula:**
```
PMI(term, group) = log2( P(term, group) / (P(term) × P(group)) )
```
A PMI of 5.07 (the ceiling observed for "black girl") means the term is 2⁵·⁰⁷ ≈ 33× more likely to appear in Black Women content than chance would predict.

**CLI:**
```bash
python src/analysis/03_intersectional_profiling.py
python src/analysis/03_intersectional_profiling.py --selfcheck --sample 80000
```

---

### 04_multilayer_harm_analysis.py
**Phase:** Phase 1 — Analysis | **Runtime:** ~2 minutes

**Purpose.** Maps HurtLex harm categories (personal stigmatisation, aggressive sexual material, etc.) to video metadata across intersectional groups. Produces a group × harm-category matrix.

**Inputs:**
- `outputs/data/01_ml_corpus.parquet`
- `config/abusive_lexica/hurtlex_EN.tsv` — HurtLex English harm lexicon (Bassignana et al. 2018)

**Outputs:**
- `outputs/data/04_harm_category_by_group.csv` — prevalence matrix (groups × harm categories)
- `outputs/data/04_harm_category_by_group_counts.csv` — raw counts
- `outputs/data/04_harm_category_outliers.csv` — group-category cells above 2 SD
- `outputs/figures/harm/04_harm_category_heatmap_{light,dark}.png`

**CLI:**
```bash
python src/analysis/04_multilayer_harm_analysis.py
python src/analysis/04_multilayer_harm_analysis.py --selfcheck --sample 80000
```

**Notes:**
- `hurtlex_EN.tsv` maps English terms to categories like `ps` (personal stigmatisation), `asf` (aggressive sexual material). The analysis checks whether these terms appear in video tags/titles by group.
- `baseLexicon.txt` and `expandedLexicon.txt` in `config/abusive_lexica/` are supplementary word lists used for additional harm scoring.

---

### 05_statistical_bias_tests.py
**Phase:** Phase 1 — Analysis | **Runtime:** ~1 minute

**Purpose.** Formal statistical testing of engagement disparities (views, ratings) across intersectional groups using Mann-Whitney U tests, with effect sizes (Cohen's d, Cliff's delta) and FDR correction.

**Inputs:** `outputs/data/01_ml_corpus.parquet`

**Outputs:**
- `outputs/data/05_bias_tests_views.csv` — pairwise test results for views
- `outputs/data/05_bias_tests_ratings.csv` — pairwise test results for ratings
- `outputs/figures/bias/05_bias_forest_views_{light,dark}.png` — forest plot
- `outputs/figures/bias/05_bias_forest_ratings_{light,dark}.png`

**Tests applied:**
- **Mann-Whitney U**: non-parametric comparison of two group distributions
- **Cohen's d**: standardised mean difference (small=0.2, medium=0.5, large=0.8)
- **Cliff's delta**: probability that a random value from group A exceeds a random value from group B
- **Benjamini-Hochberg FDR correction**: controls false discovery rate across all pairwise comparisons

**CLI:**
```bash
python src/analysis/05_statistical_bias_tests.py
python src/analysis/05_statistical_bias_tests.py --selfcheck --sample 80000
```

---

### 15_qualitative_deep_dive.py
**Phase:** Phase 3 — Advanced Analysis | **Runtime:** ~30 seconds

**Purpose.** Collects the highest-confidence model mistakes — videos where the model was most confident in the wrong direction — for qualitative inspection. Supports filtering by intersectional group to focus on specific harm patterns.

**Inputs:**
- `outputs/data/01_ml_corpus.parquet` — for metadata enrichment
- Prediction CSVs (auto-discovered: `07_rf_test_predictions.csv` preferred)

**Outputs:**
- `outputs/data/15_qualitative_error_samples_topk.csv` — top-K errors with title, group, probability, margin
- `outputs/narratives/automated/15_qualitative_summary.md`

**CLI:**
```bash
python src/analysis/15_qualitative_deep_dive.py
python src/analysis/15_qualitative_deep_dive.py --k 50 --focus "Black Women"
python src/analysis/15_qualitative_deep_dive.py --selfcheck --sample 200
```

**Notes:**
- Confident mistakes are ranked by `|probability − 0.5|` (margin from the decision boundary). A probability of 0.99 for a negative case is more informative than a 0.52 error.
- Non-English titles appear in the outlier lists; the `tags` column provides semantic context.

---

### 16_deep_data_analysis.py
**Phase:** Phase 3 — Advanced Analysis | **Runtime:** ~3 minutes

**Purpose.** Temporal analysis of engagement trends: yearly representation by group, views-per-day trends, ratings trends. Normalises by video age to enable fair comparison across publication years.

**Inputs:** `outputs/data/01_ml_corpus.parquet`

**Outputs:**
- `outputs/data/16_temporal_group_representation.csv` — % of corpus by group × year
- `outputs/data/16_temporal_rating_view_trends.csv` — mean views/day, ratings/day by group × year
- `outputs/data/16_engagement_bias_by_group.csv` — engagement disparities by group
- `outputs/figures/16_temporal_*.png` — 6 dual-theme figures

**Key output: `16_temporal_rating_view_trends.csv`**
```
_publish_year, race_ethnicity_asian, mean_rating, mean_views, mean_views_per_day, mean_ratings_per_day, n
```
This is the source for the slope gap calculation in step 19.

**CLI:**
```bash
python src/analysis/16_deep_data_analysis.py
python src/analysis/16_deep_data_analysis.py --selfcheck --sample 150000
```

**Notes:**
- Uses `pd.Timestamp.now('UTC')` (fixed from deprecated `.utcnow()`).
- Falls back to a synthetic `__all__` group if protected attributes are missing.

---

### 17_engagement_bias_analysis.py
**Phase:** Phase 3 — Advanced Analysis | **Runtime:** ~2 minutes

**Purpose.** Black Women vs Others engagement gap analysis with bootstrap 95% confidence intervals. Computes the year-by-year gap in age-normalised views and ratings, and analyses head/tail content composition.

**Inputs:** `outputs/data/01_ml_corpus.parquet`

**Outputs:**
- `outputs/data/17_gap_views_per_day.csv` — yearly gap (BW − Others) with bootstrap CI
- `outputs/data/17_gap_rating.csv` — yearly rating gap with CI
- `outputs/data/17_head_tail_composition.csv` — group share of top 1% vs bottom 99%
- `outputs/data/17_yearly_bw_gaps.csv` — summary gap table
- `outputs/data/17_quantiles_race_ethnicity.csv` — views distribution quantiles
- `outputs/data/17_bw_correlations.csv` — point-biserial correlations
- `outputs/figures/17_gap_rating_{light,dark}.png` — the key "gap with CI" figure
- `outputs/figures/17_head_tail_stack_{light,dark}.png`

**Bootstrap procedure:**
- 1000 resamples per year
- 95% CI computed as 2.5th and 97.5th percentile of bootstrap distribution
- Applied to `mean_views_per_day` and `mean_rating` separately

**CLI:**
```bash
python src/analysis/17_engagement_bias_analysis.py
python src/analysis/17_engagement_bias_analysis.py --selfcheck --sample 120000
```

---

### 18_category_group_dynamics.py
**Phase:** Phase 3 — Advanced Analysis | **Runtime:** ~2 minutes

**Purpose.** Category × group dynamics: which categories are over- or under-represented for Black Women relative to the corpus baseline. Uses Laplace-smoothed representation ratios (log₂ scale).

**Inputs:** `outputs/data/01_ml_corpus.parquet`

**Outputs:**
- `outputs/data/18_category_group_matrix.csv` — group × category presence matrix
- `outputs/data/18_bw_under_over.csv` — top-K under/over-represented categories for Black Women
- `outputs/data/18_category_cooccurrence.csv` — category co-occurrence matrix (used by step 20)
- `outputs/figures/18_heatmap_log2rr_{light,dark}.png` — representation ratio heatmap
- `outputs/figures/18_bw_overrep_bar_{light,dark}.png`
- `outputs/figures/18_bw_underrep_bar_{light,dark}.png`

**CLI:**
```bash
python src/analysis/18_category_group_dynamics.py
python src/analysis/18_category_group_dynamics.py --selfcheck --sample 120000 --top-k 30
```

---

### 19_advanced_statistics.py
**Phase:** Phase 3 — Advanced Analysis | **Runtime:** ~5 minutes (bootstrap CI computation)

**Purpose.** Advanced statistical analysis centred on Black Women: Cliff's delta effect sizes, temporal slope estimation with bootstrap CIs, KL divergence of category distributions from the global distribution.

**Inputs:** `outputs/data/01_ml_corpus.parquet`, optionally `outputs/data/04_harm_category_by_group.csv`

**Outputs:**
- `outputs/data/19_trend_slopes.csv` — **the key output**: slope_bw, slope_others, slope_gap, CI for views/day and rating
- `outputs/data/19_bw_effect_sizes.csv` — Cliff's delta for engagement metrics
- `outputs/data/19_category_divergence.csv` — KL divergence per group from global category distribution
- `outputs/figures/19_slope_bw_vs_others_{light,dark}.png`
- `outputs/figures/19_category_kl_bar_{light,dark}.png`

**`19_trend_slopes.csv` schema:**

| Column | Description |
|:-------|:------------|
| `metric` | `views_per_day` or `rating` |
| `slope_bw` | OLS slope for Black Women (rating-points or views per year) |
| `slope_bw_lo/hi` | 95% bootstrap CI for BW slope |
| `slope_others` | OLS slope for all Other videos |
| `slope_gap_bw_minus_others` | **The key finding**: BW slope − Others slope |
| `slope_gap_lo/hi` | 95% bootstrap CI for the gap |
| `n_bw`, `n_others` | Sample sizes |

**Verified canonical values:**
- `slope_gap_rating = −3.38` (95% CI [−3.75, −2.98]) — entirely below zero

**CLI:**
```bash
python src/analysis/19_advanced_statistics.py
python src/analysis/19_advanced_statistics.py --selfcheck --sample 120000
```

---

### 20_network_analysis.py
**Phase:** Phase 3 — Advanced Analysis | **Runtime:** ~1 minute

**Purpose.** Models the category co-occurrence structure as a weighted graph. Computes degree, strength, betweenness centrality, PageRank, and clustering coefficient to identify "hub" categories.

**Inputs:** `outputs/data/18_category_cooccurrence.csv` (preferred) or rebuilds from corpus

**Outputs:**
- `outputs/data/20_category_centrality.csv` — centrality metrics per category node
- `outputs/figures/20_top_strength_{light,dark}.png` — top-20 categories by connection strength

**Graph construction:**
- Nodes: categories (111 total)
- Edges: weighted by number of videos containing both categories simultaneously
- Strength: sum of all edge weights for a node (total co-occurrence mass)

**Interpretation:** A high-strength category is a hub — it co-occurs with many other categories. Fairness interventions targeting these hubs have disproportionate downstream impact because correcting bias in a hub propagates through the network.

**CLI:**
```bash
python src/analysis/20_network_analysis.py
python src/analysis/20_network_analysis.py --selfcheck --sample-k 60
```

---

### 23_limitations_analysis.py
**Phase:** Phase 3 — Advanced Analysis | **Runtime:** ~2 minutes

**Purpose.** Systematic limitations audit per Datasheets for Datasets (Gebru et al.) and Model Cards (Mitchell et al.): data quality, representation skew, temporal drift, category long-tail, lexicon dependency, missingness.

**Inputs:** `outputs/data/01_ml_corpus.parquet`, optionally `outputs/data/04_harm_category_by_group.csv`

**Outputs:**
- `outputs/data/23_data_quality.csv` — missingness rates, outlier burden, duplicate rate
- `outputs/data/23_representation_skew.csv` — group × corpus-share table
- `outputs/data/23_temporal_drift.csv` — content age distribution
- `outputs/data/23_category_longtail.csv` — cumulative coverage by top-K categories
- `outputs/data/23_summary_checklist.csv` — binary checklist (PASS/WARN/FAIL per criterion)
- `outputs/figures/light/23_*.png` and `dark/23_*.png` — 5 dual-theme figures

**CLI:**
```bash
python src/analysis/23_limitations_analysis.py
python src/analysis/23_limitations_analysis.py --selfcheck --sample 80000
```

---

## Module: `src/models/` — Model Layer

---

### 07_rf_baseline.py
**Phase:** Phase 2 — Modelling | **Runtime:** ~12–15 minutes (500 trees on 321k rows)

**Purpose.** Trains the canonical Random Forest baseline and evaluates it on val/test with per-group fairness metrics. Also writes compatibility artefacts for step 08 (`08_rf_*` prefix files).

**Inputs:**
- `outputs/data/01_ml_corpus.parquet`
- `outputs/data/06_train_ids.csv`, `06_val_ids.csv`, `06_test_ids.csv`

**Outputs:**
- `outputs/data/07_overall_metrics.csv` — val + test accuracy/precision/recall/F1
- `outputs/data/07_fairness_group_metrics.csv` — per-group metrics at threshold 0.5
- `outputs/data/07_fairness_disparities.csv` — disparities vs White Women
- `outputs/data/07_rf_val_predictions.csv`, `07_rf_test_predictions.csv` — probability scores
- `outputs/data/07_rf_feature_importance.csv` — SVD component + numeric feature importance
- `outputs/models/07_rf.joblib`, `outputs/models/rf_baseline.joblib` — saved model
- `outputs/figures/models/07_margins_baseline_{light,dark}.png`

**Pipeline:**
```
model_input_text → HashingVectorizer(2-gram, 2¹⁸, L2) → TruncatedSVD(256) ─┐
                                                                              ├─ RandomForestClassifier(500, seed=95)
[duration, ratings] ───────────────────────────────────────────────────────┘
```

**CLI:**
```bash
python src/models/07_rf_baseline.py                                    # full run
python src/models/07_rf_baseline.py --selfcheck --sample 120000        # fast check
python src/models/07_rf_baseline.py --n-estimators 300 --n-svd-components 128  # lighter
```

**Dependencies:** `src/fairness/fairness_evaluation_utils`, `src/utils/theme_manager`

---

### 07a_category_sweep.py
**Phase:** Phase 2 — Modelling | **Runtime:** ~7 hours (12 RF models × 500 trees × 321k rows)

**Purpose.** Systematic comparison of 12 candidate positive-class definitions. For each category, trains a full RF pipeline, tunes the decision threshold on val (maximises F1), and reports test metrics. Validates the choice of "Amateur" as the positive class.

**Inputs:** Same as step 07

**Outputs:**
- `outputs/data/07a_category_sweep_results.csv` — per-category: TrainPos, TestF1, TestAUROC, DecisionThreshold, Prevalence
- `outputs/figures/models/07a_category_sweep_{light,dark}.png`
- `outputs/narratives/automated/07a_category_sweep_summary.md`

**Default categories:** Amateur, Big Tits, Anal, Asian, Blonde, Blowjob, Masturbation, Creampie, Fetish, Cumshot, Feet, Brunette

**CLI:**
```bash
python src/models/07a_category_sweep.py                                          # all defaults
python src/models/07a_category_sweep.py --categories "Amateur,Big Tits,Anal"    # subset
python src/models/07a_category_sweep.py --n-estimators 300 --max-depth 28       # faster
```

**Notes:** The threshold sweep uses 81 points on [0.1, 0.9]. The optimal threshold per category is stored in `DecisionThreshold` column — Amateur's optimal threshold (0.39) closely matches the F1-optimal threshold found in step 08 (0.40).

---

### 09_bert_baseline.py
**Phase:** Phase 2 — Modelling | **Runtime:** ~2.5 hours (MPS/Apple Silicon), ~12 hours (CPU)

**Purpose.** Fine-tunes DistilBERT for the Amateur binary classification task and evaluates with full per-group fairness metrics. Uses MPS (Metal Performance Shaders) on Apple Silicon automatically.

**Inputs:**
- `outputs/data/01_ml_corpus.parquet`
- `outputs/data/06_train_ids.csv`, `06_val_ids.csv`, `06_test_ids.csv`
- HuggingFace model: `distilbert-base-uncased` (downloaded on first run)

**Outputs:**
- `outputs/data/09_overall_metrics.csv` — test acc 0.924, F1 0.868
- `outputs/data/09_fairness_group_metrics.csv` — per-group metrics
- `outputs/data/09_fairness_disparities.csv` — EOD vs White Women
- `outputs/data/09_bert_val_metrics.csv`, `09_bert_test_metrics.csv`
- `outputs/data/09_bert_test_predictions.csv`, `09_bert_val_predictions.csv`
- `outputs/models/09_bert/` — HuggingFace checkpoint directory
- `outputs/figures/models/09_margins_baseline_{light,dark}.png`

**Training config (defaults):**
- Model: `distilbert-base-uncased`
- Epochs: 2
- Batch size: 16 per device
- Device: MPS (Apple Silicon) → CUDA → CPU (auto-detected)
- Seed: 95

**CLI:**
```bash
python src/models/09_bert_baseline.py                                    # full run (~2.5h MPS)
python src/models/09_bert_baseline.py --selfcheck --sample 16000 --epochs 1 --model prajjwal1/bert-tiny  # fast check (~5min)
python src/models/09_bert_baseline.py --epochs 3 --batch 32             # tune
```

**Dependencies:** `torch`, `transformers`, `accelerate>=1.1.0`

---

### `src/models/model_training.py`
**Phase:** Utility | **Runtime:** Seconds (pure function)

**Purpose.** Importable API wrapping RandomForestClassifier training. Used by integration tests. Does not implement the full pipeline (no text vectorisation) — for that, see `07_rf_baseline.py`.

```python
from src.models.model_training import train_baseline_model
model = train_baseline_model(X, y, random_state=95)
predictions = model.predict(X_test)
```

---

## Module: `src/fairness/` — Fairness Layer

---

### fairness_evaluation_utils.py
**Phase:** Utility (imported by all fairness steps) | **Runtime:** Milliseconds per call

**Purpose.** Central fairness evaluation library. Defines the canonical group labelling, all metric computation, and GOLD annotation helpers. After the 2026-05-21 priority-fix session, this is the single source of truth for group assignment.

**Key functions:**

| Function | Signature | Returns |
|:---------|:----------|:--------|
| `group_labels_intersectional(df)` | `df: DataFrame` → `Series` | Group labels with priority: Black > White > Asian > Latina > Other |
| `overall_metrics(y_true, y_pred)` | arrays → `ClassifMetrics` | Dataclass: acc, prec, rec, f1 (all rounded to 3 dp) |
| `calculate_group_metrics(df_meta)` | `df` with y_true, y_pred → `DataFrame` | N, Accuracy, Precision, Recall, F1 per group |
| `calculate_fairness_disparities(df_group, privileged)` | group metrics df → `DataFrame` | Accuracy Disparity, EOD, Precision Disparity vs privileged group |
| `top_confident_outliers(df_meta, probs, k)` | — → `DataFrame` | Top-k errors ranked by \|p − 0.5\| |
| `demographic_parity_difference(y_pred, groups)` | arrays → float | max(rate) − min(rate) across groups |
| `equal_opportunity_difference(y_true, y_pred, groups)` | arrays → float | max(TPR) − min(TPR) across groups |
| `expected_calibration_error(y_true, y_scores, n_bins)` | arrays → float | Weighted mean calibration gap |
| `load_gold_table(path)` | path → `DataFrame | None` | Loads GOLD annotations; normalises label column |
| `package_fairness_summary(gm, disp)` | — → dict | JSON-safe dict for dashboards |

**The priority fix (2026-05-21).** The original code applied group labels sequentially (last assignment wins → Latina always beats Black for multi-flagged videos). Fixed by applying exclusive masks: `ww = race_white & gf & ~bw`. This change was applied to this file AND to steps 02, 07, 09, 10, 11, 12.

---

### 08_comprehensive_evaluation.py
**Phase:** Phase 2 — Fairness | **Runtime:** ~10 seconds

**Purpose.** Full fairness diagnostic on any prediction file: global ROC/PR curves, threshold sweep, F1-optimal threshold selection, per-group metrics at operating point, fairness curve (EOD vs threshold), confusion matrix, CADP analysis.

**Inputs:**
- `outputs/data/07_rf_val_predictions.csv` — for threshold selection
- `outputs/data/07_rf_test_predictions.csv` — for evaluation
- `outputs/data/01_ml_corpus.parquet` — for group repair if predictions missing group labels

**Key outputs:**

| File | Content |
|:-----|:--------|
| `08_group_metrics_at_opt.csv` | Per-group metrics at F1-optimal threshold (0.40) |
| `08_disparities_at_opt.csv` | EOD, accuracy disparity, precision disparity at threshold 0.40 |
| `08_threshold_sweep.csv` | Accuracy/precision/recall/F1 at 201 thresholds [0.0, 1.0] |
| `08_fairness_curve.csv` | max\|EOD\| at each of 201 thresholds |
| `08_confusion_matrix_at_opt.csv` | 2×2 confusion matrix at operating threshold |
| `08_roc_{light,dark}.png` | ROC curve (AUROC 0.957) |
| `08_pr_{light,dark}.png` | Precision-recall curve |
| `08_fairness_curve_{light,dark}.png` | EOD vs threshold |

**CLI:**
```bash
python src/fairness/08_comprehensive_evaluation.py                       # uses step 07 defaults
python src/fairness/08_comprehensive_evaluation.py --selfcheck --sample 80000
python src/fairness/08_comprehensive_evaluation.py --val-csv path/to/val.csv --test-csv path/to/test.csv
```

---

### 10_preprocessing_mitigation.py
**Phase:** Phase 2 — Fairness | **Runtime:** ~12 minutes

**Purpose.** Pre-processing bias mitigation using Kamiran-Calders reweighing. Trains a new RF on the reweighted training set and evaluates per-group fairness.

**The reweighing formula:**
```
w(a,y) = P(A=a) · P(Y=y) / P(A=a, Y=y)
normalized so that sum(w) = N
```

**Inputs:** Same as step 07 (parquet + split IDs)

**Key outputs:**
- `outputs/data/10_reweigh_overall_metrics.csv` — acc 0.953, F1 0.915
- `outputs/data/10_reweigh_group_metrics.csv` — per-group metrics
- `outputs/data/10_reweigh_disparities.csv` — Black Women EOD: 0.092 (−53%)
- `outputs/models/rf_reweighed.joblib` — saved model for step 12

**Feature columns:** Uses `combined_text_clean` (not `model_input_text`) and `rating`, `views` (not `duration`, `ratings`) — different from step 07. This is deliberate: the reweighing pipeline was designed to match the original thesis feature set.

**CLI:**
```bash
python src/fairness/10_preprocessing_mitigation.py
python src/fairness/10_preprocessing_mitigation.py --selfcheck --sample 80000
python src/fairness/10_preprocessing_mitigation.py --use-gold  # override labels with gold annotations
```

---

### 11_inprocessing_mitigation.py
**Phase:** Phase 2 — Fairness | **Runtime:** ~5 minutes

**Purpose.** In-processing mitigation using Fairlearn's `ExponentiatedGradient` with `DemographicParity` constraint. Wraps a `LogisticRegression` classifier.

**Why EG + DP fails for this corpus.** DP equalises positive prediction rates `P(Ŷ=1|A=a)` regardless of the true label. In this corpus, where category vocabulary is semantically entangled with group membership, satisfying DP forces the model into a high-recall/low-precision regime that worsens EOD for Latina Women (0.200→0.297). DP is the wrong constraint when the positive class correlates with group.

**Outputs:**
- `outputs/data/11_inproc_overall_metrics.csv` — acc 0.878, high recall (0.914), low precision (0.726)
- `outputs/data/11_inproc_group_metrics.csv` — Latina Women EOD 0.297 (worse than baseline)
- `outputs/data/11_inproc_disparities.csv`

**CLI:**
```bash
python src/fairness/11_inprocessing_mitigation.py
python src/fairness/11_inprocessing_mitigation.py --selfcheck --sample 80000
python src/fairness/11_inprocessing_mitigation.py --eps 0.02 --max-iter 1500
```

---

### 12_postprocessing_mitigation.py
**Phase:** Phase 2 — Fairness | **Runtime:** ~30 seconds

**Purpose.** Post-processing mitigation using Fairlearn's `ThresholdOptimizer` with `EqualizedOdds` constraint. Learns per-group decision thresholds from the validation set; no retraining required.

**How it works.** Solves a linear programme on validation set predictions to find thresholds θ_a for each group a such that:
- TPR_a ≈ TPR_b and FPR_a ≈ FPR_b for all pairs (a,b)
- Accuracy loss is minimised subject to the parity constraint

**Inputs:**
- `outputs/models/rf_baseline.joblib` — fitted baseline (preferred) or `rf_reweighed.joblib`
- Split IDs and corpus parquet

**Key outputs:**
- `outputs/data/12_postproc_overall_metrics.csv` — acc 0.839 (−3.9pp from baseline)
- `outputs/data/12_postproc_group_metrics.csv` — Black Women EOD: 0.008 (−96%)
- `outputs/data/12_postproc_disparities.csv` — all EODs < 0.03

**CLI:**
```bash
python src/fairness/12_postprocessing_mitigation.py
python src/fairness/12_postprocessing_mitigation.py --selfcheck --sample 80000
```

---

### 13_mitigation_effectiveness.py
**Phase:** Phase 2 — Fairness | **Runtime:** ~5 seconds

**Purpose.** Synthesises results from steps 07, 10, 11, 12 into a single comparison table and Pareto-ready metrics. Auto-discovers output files from prior steps.

**Inputs:** Auto-discovered CSVs matching patterns `07_*`, `10_*`, `11_*`, `12_*`

**Outputs:**
- `outputs/data/mitigation_effectiveness.csv` — accuracy and EOD per strategy
- `dissertation/auto_tables/mitigation_effectiveness.tex`

**CLI:**
```bash
python src/fairness/13_mitigation_effectiveness.py
python src/fairness/13_mitigation_effectiveness.py --focus-group "Black Women"
```

---

### 27_ground_truth.py
**Phase:** Optional — Gold annotation | **Runtime:** Seconds per operation

**Purpose.** Creates and manages a gold-standard annotation dataset from a stratified test sample. Supports template generation, annotation merging, inter-annotator agreement computation, and finalisation.

**Outputs:**
- `outputs/data/gold/gold_template.csv` — blank annotation template
- `outputs/data/gold/gold_annotator_*.csv` — per-annotator fills
- `outputs/data/gold/gold_final.csv` — merged + adjudicated gold labels

**Metrics computed:** Cohen's κ (pairwise), Fleiss' κ (multi-annotator), percent agreement.

**CLI:**
```bash
python src/fairness/27_ground_truth.py --make-template --sample 1200   # create template
python src/fairness/27_ground_truth.py --merge                         # merge annotator files
python src/fairness/27_ground_truth.py --finalize                      # adjudicate + save gold_final.csv
```

---

### 30_psm_ipw.py (`src/fairness/causal/`)
**Phase:** Phase 4 — Synthesis | **Runtime:** ~8 seconds

**Purpose.** Estimates the causal Average Treatment Effect (ATE) of Amateur category assignment on video rating using Inverse Probability of Treatment Weighting (IPTW / Horvitz-Thompson estimator).

**Causal question:** What is the effect of a video being classified as "Amateur" on its rating, after controlling for what other categories it belongs to and its engagement history?

**Causal diagram:**
```
Covariates X (top-20 categories, log-duration, log-ratings-count)
        ↓
Propensity p(T=1|X) ← Logistic Regression
        ↓
Stabilized IPTW weights w_i
        ↓
ATE = Σ(w_i · T_i · Y_i) / Σ(w_i · T_i) − Σ(w_i · (1−T_i) · Y_i) / Σ(w_i · (1−T_i))
```

**Outputs:**
- `outputs/data/30_ate_rating.csv` — ATE +7.216 (95% CI [7.215, 7.216]), overlap 94.7%
- `outputs/data/30_sensitivity.csv` — ATE stability across clipping thresholds 0.01, 0.02, 0.05
- `outputs/figures/causal/30_propensity_by_treatment_{light,dark}.png`
- `outputs/figures/causal/30_weights_hist_{light,dark}.png`

**CLI:**
```bash
python src/fairness/causal/30_psm_ipw.py               # full run
python src/fairness/causal/30_psm_ipw.py --selfcheck --sample 80000
python src/fairness/causal/30_psm_ipw.py --topk 30 --clip 0.02
```

**Limitations:** Overlap (94.7%) is good but not perfect — 5.3% of propensities fall outside [0.1, 0.9]. The narrow CI ([7.215, 7.216]) reflects the large N but does not capture structural confounding (e.g., platform recommendation feedback loops).

---

### `src/fairness/preprocessing_mitigation.py`
**Phase:** Utility | **Runtime:** Milliseconds

**Purpose.** Importable reweighing API used by integration tests. Pure functions; no I/O.

```python
from src.fairness.preprocessing_mitigation import compute_reweighing_weights, apply_reweighing
weights = compute_reweighing_weights(y, groups)  # → np.ndarray
model = apply_reweighing(X, y, groups)           # → fitted RF
```

---

### `src/fairness/postprocessing_mitigation.py`
**Phase:** Utility | **Runtime:** Milliseconds

**Purpose.** Importable post-processing API. Provides threshold optimisation per group given a metric (TPR, FPR, precision). Used by integration tests.

---

### `src/fairness/merge_annotations.py`
**Phase:** Utility — Gold annotation | **Runtime:** Seconds

**Purpose.** Merges original and supplementary gold annotation files, deduplicates on `video_id`, reports any conflicts. Outputs a single complete annotation file.

---

## Module: `src/experiments/`

---

### 10b_matchedN_reweighing.py
**Phase:** Phase 2 — Experiments | **Runtime:** ~6 minutes

**Purpose.** Sensitivity test for the reweighing result: caps White Women training instances to N_BW (the number of Black Women training samples) to verify that the accuracy improvement is not an artefact of sample-size imbalance.

**Experimental design:**
1. Subsample White Women training instances to match Black Women count
2. Run Kamiran-Calders reweighing on the matched sample
3. Compare results to full-corpus reweighing

**Result:** Test acc 0.951 (vs 0.953 full) — essentially identical. The reweighing gain is real.

**Outputs:**
- `outputs/data/10b_matchedN_overall_metrics.csv`
- `outputs/data/10b_matchedN_group_metrics.csv`
- `outputs/data/10b_matchedN_disparities.csv`

---

### 22_ablation_studies.py
**Phase:** Phase 3 — Experiments | **Runtime:** ~10 minutes

**Purpose.** Sensitivity analysis on Black Women category representation. Tests how the representation ratio (BW over/under-representation by category) changes when: (a) lexicon terms are disabled, (b) category assignments have random noise injected, (c) top-K category coverage varies.

**Outputs (under `outputs/ablation/`):**
- `22_ablation_all.csv` — overall summary
- `22_rr_lexicon_off.csv` — baseline KPI without lexicon
- `22_topcats_noise_{10,25,50}.csv` — KPI under 10/25/50% category noise
- `22_topk_mass_{10,20,30,50}.csv` — KPI at different top-K thresholds

**CLI:**
```bash
python src/experiments/22_ablation_studies.py
python src/experiments/22_ablation_studies.py --selfcheck --sample 120000 --top-k-base 30
```

---

## Module: `src/dissertation/` — Synthesis Layer

---

### 14_rq_synthesis.py
**Phase:** Phase 4 — Synthesis | **Runtime:** ~5 seconds

**Purpose.** Collates per-step fairness outputs into a structured answer table for each of the 5 research questions. Best-effort: gracefully skips missing files.

**Outputs:** `14_rq_synthesis.csv`, `outputs/narratives/14_rq_synthesis.md`

---

### 24_results_synthesis.py
**Phase:** Phase 4 — Synthesis | **Runtime:** ~30 seconds

**Purpose.** Combines ablation (step 22) and limitations (step 23) into dissertation-ready KPI tables, dashboards, and a structured narrative. Requires `tabulate` package.

**Outputs:**
- `outputs/data/24_ablation_kpis.csv`, `24_limitations_kpis.csv`, `24_checklist.csv`
- `outputs/figures/light/24_ablation_dashboard_light.png`
- `outputs/narratives/automated/24_results_synthesis.md`

---

### 25_pareto_frontier.py
**Phase:** Phase 4 — Synthesis | **Runtime:** ~1 second

**Purpose.** Collects accuracy and fairness scores from all strategy evaluation CSVs, computes the Pareto frontier, generates the key accuracy-fairness trade-off figure.

**How Pareto dominance is defined:**
- Strategy A dominates B if: `accuracy_A ≥ accuracy_B AND fairness_A ≥ fairness_B` (with at least one strict)
- `fairness = 1 − max|EOD|` across all intersectional groups

**Outputs:**
- `outputs/data/25_pareto_points.csv` — all strategies with accuracy + fairness score
- `outputs/data/25_pareto_frontier.csv` — non-dominated strategies only
- `outputs/figures/light/25_pareto_frontier_light.png`

**CLI:**
```bash
python src/dissertation/25_pareto_frontier.py
python src/dissertation/25_pareto_frontier.py --selfcheck
```

---

### 26_executive_summary.py
**Phase:** Phase 4 — Synthesis | **Runtime:** ~5 seconds

**Purpose.** Auto-generates a ≤500-word executive summary from pipeline artefacts (corpus size, engagement gaps, effect sizes, category extremes). Output: `dissertation/executive_summary.md`.

---

## Module: `src/presentation/`

---

### 28_qa_backup_slides.py
**Phase:** Phase 4 — Presentation | **Runtime:** ~5 seconds

**Purpose.** Generates a self-contained, keyboard-navigable HTML deck (no external dependencies) covering 10 high-priority Q&A topics for a dissertation defence or conference presentation.

**Topics covered:** Intersectionality rationale, group inference methodology, fairness metric choice, adversarial debiasing failure, regulatory gap, causal claims limitations, BERT vs RF tradeoffs, temporal harm mechanism, matched-N validation, future directions.

**Output:** `outputs/interactive/qa_backup_slides.html` — open in any browser, navigate with arrow keys.

---

### 29_single_video_explainer.py
**Phase:** Phase 4 — Presentation | **Runtime:** ~30 seconds

**Purpose.** Interactive HTML explainer for a single video: shows engagement metrics, predictions from all models, harm taxonomy matches, PMI stereotypes, fairness metrics, and comparator videos matched by year and category.

**CLI:**
```bash
python src/presentation/29_single_video_explainer.py --random            # random video
python src/presentation/29_single_video_explainer.py --only-bw           # random Black Women video
python src/presentation/29_single_video_explainer.py --search "amateur"  # search by term
python src/presentation/29_single_video_explainer.py --n-compare 4       # 4 comparators
```

---

## Module: `src/visualization/`

---

### 21_interactive_dashboard.py
**Phase:** Phase 4 — Visualization | **Runtime:** ~2 minutes

**Purpose.** Professional Plotly-based interactive dashboard integrating all major analyses: fairness metrics, engagement disparities, representation, ablation studies, network centrality, temporal trends.

**Output:** `outputs/interactive/21_interactive_dashboard.html` — self-contained HTML (~15 MB). Open in browser; no server required.

**Tabs/sections:** Corpus overview, group representation, engagement bias, model fairness, mitigation comparison, temporal trends, network analysis, ablation sensitivity.

---

### dashboard_stats.py
**Purpose.** Pre-calculates and caches key metrics for the dashboard to avoid recomputing on load. Writes `outputs/data/dashboard_metrics.json`.

### launch_dashboard.py
**Purpose.** Starts a local HTTP server (`localhost:8000`) serving `outputs/interactive/`. Simpler than double-clicking the HTML file when CORS restrictions apply.
```bash
python src/visualization/launch_dashboard.py
# then open http://localhost:8000/21_interactive_dashboard.html
```

---

## Module: `src/utils/` — Utilities

---

### theme_manager.py
**Purpose.** Central configuration loader and dual-theme plotting decorator.

**Key functions:**

| Function | What it does |
|:---------|:-------------|
| `load_config()` | Reads `config/settings.yaml` with `${variable}` interpolation, returns dict |
| `@plot_dual_theme(section)` | Decorator that runs the wrapped function twice — once with light theme, once dark — saving `*_light.png` and `*_dark.png`. `section` selects the colour palette (`eda`, `fairness`, `models`). |

**Usage:**
```python
@plot_dual_theme(section="fairness")
def my_plot(ax=None, palette=None, **kwargs):
    ax.plot(...)  # ax is injected by the decorator
```

---

### database.py
**Purpose.** SQLite connection management with WAL mode, foreign key enforcement, and the `video_categories.video_id` index (the change that reduced join time from 4 minutes to 8 seconds).

**Key functions:**

| Function | What it does |
|:---------|:-------------|
| `create_connection()` | Opens DB at path from config; sets WAL mode + FK pragma; returns connection |
| `get_conn()` | Context manager version: `with get_conn() as conn:` |
| `_apply_pragmas(conn)` | Sets `PRAGMA journal_mode=WAL`, `foreign_keys=ON`, `synchronous=NORMAL` |

---

### academic_tables.py
**Purpose.** Converts DataFrames to publication-quality LaTeX `\begin{table}` environments with consistent styling, caption, label, and optional note. Used by every step that writes dissertation tables.

```python
dataframe_to_latex_table(
    df=my_df.set_index("Group"),
    save_path="dissertation/auto_tables/my_table.tex",
    caption="Group-wise performance under reweighing.",
    label="tab:reweigh-group",
    note="EOD = White Women TPR − Group TPR.",
    precision=3,
)
```

---

## Scripts

### `scripts/collect_daily.sh`
Daily incremental API collection. Logs before/after row counts. Rotates logs > 30 days old. Designed for launchd 3am daily trigger.

### `scripts/pipeline_selfcheck.sh`
Weekly pipeline health check. Runs steps 01, 06, 07, 08, 10 in `--selfcheck` mode. Commits updated `*_selfcheck.csv` files and pushes to GitHub. Designed for launchd Sunday 4am trigger.

### `scripts/com.louisesfer.thesis.collect.plist`
macOS LaunchAgent: runs `collect_daily.sh` daily at 03:00. Load with:
```bash
cp scripts/com.louisesfer.thesis.collect.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.louisesfer.thesis.collect.plist
```

### `scripts/com.louisesfer.thesis.selfcheck.plist`
macOS LaunchAgent: runs `pipeline_selfcheck.sh` every Sunday at 04:00.

---

## Quick Reference: Script → Output Mapping

| Script | Primary output | Runtime |
|:-------|:--------------|:--------|
| `collector.py` | `data/redtube_videos.db` | Up to 14h |
| `01_corpus_builder.py` | `01_ml_corpus.parquet` | 4 min |
| `02_comprehensive_eda.py` | `02_eda_*.csv` (25 files) | 3–5 min |
| `03_intersectional_profiling.py` | `03_pmi_*.csv` | 2 min |
| `04_multilayer_harm_analysis.py` | `04_harm_*.csv` | 2 min |
| `05_statistical_bias_tests.py` | `05_bias_tests_*.csv` | 1 min |
| `06_stratified_splitting.py` | `06_*_ids.csv` | 3 sec |
| `07_rf_baseline.py` | `07_*.csv`, `rf_baseline.joblib` | 12–15 min |
| `07a_category_sweep.py` | `07a_category_sweep_results.csv` | ~7 hrs |
| `08_comprehensive_evaluation.py` | `08_*.csv`, ROC/PR/fairness figures | 10 sec |
| `09_bert_baseline.py` | `09_*.csv`, `09_bert/` checkpoint | 2.5h (MPS) |
| `10_preprocessing_mitigation.py` | `10_reweigh_*.csv` | 12 min |
| `10b_matchedN_reweighing.py` | `10b_matchedN_*.csv` | 6 min |
| `11_inprocessing_mitigation.py` | `11_inproc_*.csv` | 5 min |
| `12_postprocessing_mitigation.py` | `12_postproc_*.csv` | 30 sec |
| `13_mitigation_effectiveness.py` | `mitigation_effectiveness.csv` | 5 sec |
| `15_qualitative_deep_dive.py` | `15_qualitative_error_samples_topk.csv` | 30 sec |
| `16_deep_data_analysis.py` | `16_temporal_*.csv` | 3 min |
| `17_engagement_bias_analysis.py` | `17_gap_*.csv`, `17_head_tail_*.csv` | 2 min |
| `18_category_group_dynamics.py` | `18_*.csv` | 2 min |
| `19_advanced_statistics.py` | `19_trend_slopes.csv`, `19_bw_effect_sizes.csv` | 5 min |
| `20_network_analysis.py` | `20_category_centrality.csv` | 1 min |
| `22_ablation_studies.py` | `outputs/ablation/22_*.csv` | 10 min |
| `23_limitations_analysis.py` | `23_*.csv` | 2 min |
| `24_results_synthesis.py` | `24_*.csv`, dashboards | 30 sec |
| `25_pareto_frontier.py` | `25_pareto_points.csv`, Pareto figure | 1 sec |
| `26_executive_summary.py` | `dissertation/executive_summary.md` | 5 sec |
| `27_ground_truth.py` | `gold/gold_final.csv` | Seconds |
| `28_qa_backup_slides.py` | `interactive/qa_backup_slides.html` | 5 sec |
| `29_single_video_explainer.py` | `interactive/29_enhanced_video_*.html` | 30 sec |
| `30_psm_ipw.py` | `30_ate_rating.csv`, causal figures | 8 sec |

---

*For reproduction instructions, see `REBUILD_PLAN.md`. For results interpretation, see `dissertation/analysis.md`.*
