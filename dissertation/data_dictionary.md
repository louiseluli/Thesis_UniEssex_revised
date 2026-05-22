# Data Dictionary — Output Files Reference

**Project:** AlgoFairness Pornometrics | **Total CSVs:** 137  
**Location:** `outputs/data/` unless noted  
**Convention:** `*_selfcheck.csv` files are identical in schema to their canonical counterpart but computed on a random subset (~50k–120k rows). Never overwrite canonical files.

> **How to use this document:** Find the step that generated the file you need. Each entry gives the schema, what each column means, and how to interpret key values.

---

## Step 01 — Corpus Builder

### `01_corpus_selfcheck.csv`
**What it is:** 10 randomly sampled rows from the full corpus, written at the end of every step 01 run to confirm the pipeline completed successfully.

| Column | Type | Description |
|:-------|:-----|:------------|
| `video_id` | int | Unique video identifier from the Redtube API |
| `title` | str | Video title (may be non-English) |
| `rating` | float | Platform rating score (0–100 scale, 1 decimal) |
| `views` | int | Total view count at collection time |
| `publish_date` | str | ISO 8601 datetime string (`YYYY-MM-DD HH:MM:SS`) |
| `intersectional_black_female` | int | 1 if video is classified as Black Women content; 0 otherwise |
| `year` | int | Publication year extracted from `publish_date` |

**How to use:** Check that `intersectional_black_female` contains some 1s, and that `rating` and `views` are non-zero for most rows. A selfcheck with all zeros suggests the lexicon or corpus join failed.

---

## Step 02 — Comprehensive EDA

### `02_eda_age_days_by_group.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group name |
| `Median Age (days)` | Median number of days since publish date at collection time |

**Interpretation:** White Women content (2,553 days median) is significantly older than Latina (613 days), reflecting the older studio-production back-catalogue vs newer amateur content.

---

### `02_eda_age_tag_by_group.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group name |
| `18-25 %` | % of videos in each group that carry the "18-25" age tag |

---

### `02_eda_duplication_summary.csv`
| Column | Description |
|:-------|:------------|
| `Type` | Deduplication method: `Exact(title,duration)` or `Title-only` |
| `Duplicate Rows` | Number of rows matching the duplication criterion |

---

### `02_eda_entropy_by_group.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group name |
| `Entropy (Categories)` | Shannon entropy of category distribution within the group |
| `Entropy (Tags)` | Shannon entropy of tag distribution within the group |

**Interpretation:** Higher entropy = more diverse tag/category vocabulary. Black Women (entropy 9.41 tags) and Latina Women (9.23) have more diverse tag vocabularies than White Women (7.88), partly because racial-identity tags add vocabulary mass.

---

### `02_eda_intersectional_representation.csv`
| Column | Description |
|:-------|:------------|
| `Intersection` | Race × gender label (e.g., "White x Female") |
| `Count` | Number of videos in this intersection |
| `Percentage` | Share of total corpus (%) |

---

### `02_eda_intersections_rgo.csv`
| Column | Description |
|:-------|:------------|
| `Race` | Race/ethnicity label |
| `Gender` | Gender label |
| `Orientation` | Sexual orientation category (Lesbian, Gay, Bisexual, Other/Unspecified) |
| `Count` | Number of videos |
| `Percentage` | Share of total corpus (%) |

---

### `02_eda_label_cardinality.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group name |
| `#Tags` | Median number of tags per video in the group |
| `#Categories` | Median number of categories per video in the group |

---

### `02_eda_language_by_intersection.csv`
| Column | Description |
|:-------|:------------|
| `language` | Detected language of title (mostly "unknown" — API titles are often English or short) |
| `Group` | Intersectional group name |
| `N` | Number of videos |
| `Share` | % of intersectional group in this language bucket |
| `MedianRating` | Median rating for this language × group cell |
| `MedianViewsPerDay` | Median age-normalised views/day |

---

### `02_eda_month_seasonality_by_group.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group name |
| `Month` | Calendar month (1–12) |
| `N` | Number of videos published in this month |
| `Share %` | % of group's corpus published in this month |

---

### `02_eda_orientation_by_group.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group name |
| `Orientation` | Lesbian, Gay, Bisexual (derived from category/tag matching) |
| `Count` | Number of videos with this orientation label |

---

### `02_eda_outlier_burden.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group name |
| `Above99% duration %` | % of group's videos above the 99th percentile for duration |
| `Above99% views %` | % above 99th percentile for total views |
| `Above99% ratings %` | % above 99th percentile for number of ratings |

**Interpretation:** White Women have much higher outlier burden on views and ratings (2.1% and 2.8%) vs other groups (~0.4%), reflecting the studio-production content with extreme engagement counts.

---

### `02_eda_protected_rating_medians.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Group name (includes "Overall") |
| `Median Rating (Protected Tags)` | Median rating for videos with protected-group tags |
| `Median Rating (No Protected Tags)` | Median rating for videos without such tags |
| `N Protected` | Videos with protected tags |
| `N No-Protected` | Videos without protected tags |

---

### `02_eda_quality_proxies_by_group.csv`
**The key quality table.** Shows the occupational pipeline split.

| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group name |
| `HD %` | % of group's videos with the HD tag |
| `4K %` | % with 4K tag |
| `Verified %` | % with "Verified Amateurs" tag (platform credential for independent creators) |

**Key values:** Latina Women 77.1% Verified, White Women 19.2% — the largest gap in the dataset.

---

### `02_eda_quality_by_language.csv`
| Column | Description |
|:-------|:------------|
| `language` | Detected language |
| `HD %`, `4K %`, `Verified %` | Quality proxies for this language bucket |

---

### `02_eda_rating_count_by_group.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group name |
| `Median` | Median number of ratings (votes) received |
| `IQR` | Interquartile range of ratings count |

**Interpretation:** White Women receive 65 ratings median vs 9–15 for other groups — a 4–7× engagement gap in absolute vote count. This means White Women's rating scores are more statistically stable.

---

### `02_eda_rating_disparities_stats.csv`
Full descriptive statistics for rating scores (0–100) by group.

| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group name |
| `count` | Number of videos |
| `mean` | Mean rating score |
| `std` | Standard deviation |
| `min`, `25%`, `50%`, `75%`, `max` | Percentiles |

**Key insight:** White Women mean rating 72.7 vs Black Women 64.4. But White Women std is only 23.4 vs Black Women 38.5 — Black Women ratings are more volatile, partly because of incoherent records (see `rating_clean` column).

---

### `02_eda_title_quality_by_group.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group name |
| `MedianTokens` | Median word count of title |
| `NonASCIIShare` | % of titles containing non-ASCII characters |
| `DigitsShare` | % of titles containing digits |
| `MissingTitle` | % with missing title |

**Key insight:** Asian Women have 8.8% non-ASCII titles (Japanese, Chinese characters) vs ~0.5–1.3% for other groups. This impairs bag-of-words text features for this group.

---

### `02_eda_top_categories.csv`
| Column | Description |
|:-------|:------------|
| `Category` | Category name (lowercase) |
| `Video Count` | Number of videos with this category |
| `Percentage` | Share of total corpus (%) |

Top category: Blowjob (38.1%), followed by Big Tits (32.2%), Amateur (28.4%).

---

### `02_eda_top_protected_tags.csv`
| Column | Description |
|:-------|:------------|
| `Protected Tag` | Tag from the protected-group lexicon |
| `Videos With Tag` | Number of videos carrying this tag |
| `Percentage` | Share of total corpus (%) |

Top protected tag: fetish (13.0%), interracial (6.8%), bbw (4.5%).

---

### `02_eda_top_tags.csv`
| Column | Description |
|:-------|:------------|
| `Tag` | Tag name |
| `Videos With Tag` | Count |
| `Percentage` | Share of corpus |

Top tags: hd (92.3%), verified amateurs (48.0%), blowjob (43.1%).

---

### `02_eda_views_disparities_stats.csv`
Full descriptive statistics for age-normalised views per day by group.

| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group name |
| `count` | Number of videos |
| `mean` | Mean views/day (age-normalised: views ÷ days since publish) |
| `std` | Standard deviation |
| `min`, `25%`, `50%`, `75%`, `max` | Percentiles |

**Key values:** White Women mean 4.261, Black Women 3.464, Latina 3.444, Asian 3.426.

---

### `02_eda_views_per_day_by_group.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group name |
| `Median` | Median views/day (integer, rounded) |

**Key values:** White Women 11, Latina 9, Black Women 7, Asian 6.

---

## Step 03 — PMI Intersectional Profiling

### `03_pmi_intersectional_black_women.csv`
Top-25 PMI terms for the Black Women group.

| Column | Description |
|:-------|:------------|
| `Term` | Text term (1–2 grams) |
| `PMI` | Pointwise Mutual Information score. Higher = stronger group association. Ceiling is log₂(1/P(group)) ≈ 5.07. |
| `DF` | Document frequency (videos containing this term) |
| `DF_in_group` | Document frequency within Black Women videos |
| `P(term)` | P(term appears in a random video) |
| `P(group)` | P(video is in Black Women group) = 0.02983 |
| `P(term,group)` | Joint probability |
| `P(term\|group)` | Conditional: P(term appears \| video is Black Women) |

**Ceiling value:** PMI 5.07 means the term appears *only* in Black Women content — perfect group predictability. "black girl", "ebony female", "ghetto hd" all sit at this ceiling.

---

### `03_pmi_intersectional_black_women_full.csv`
Same schema as above, but containing all 30,000 terms analysed (not just top-25).

---

### `03_pmi_intersectional_black_women_outliers.csv`
Terms with unusually high document frequency relative to their PMI — terms that appear in many videos but with weaker group association than expected. Useful for identifying terms that are widely used but not group-specific.

---

## Step 04 — Multilayer Harm Analysis

### `04_harm_category_by_group.csv`
Group × HurtLex category prevalence matrix (percentages).

| Column | Description |
|:-------|:------------|
| `Group` | Protected group label |
| `an` | Animosity — general hostile language |
| `asf` | Aggressive sexual fantasy |
| `asm` | Aggressive sexual material |
| `cds` | Derogatory or demeaning sexual language |
| `ddf` | Defamatory or discriminatory content — female-targeted |
| `ddp` | Defamatory or discriminatory content — general |
| `dmc` | Derogatory terms for minority/community |
| `is` | Identity-based stigmatisation |
| `om` | Objectifying material |
| `or` | Objectifying references |
| `pa` | Physical assault language |
| `pr` | Prostitution references |
| `ps` | Personal stigmatisation |
| `qas` | Quasi-abusive speech |
| `rci` | Racial/cultural insult |
| `re` | Racist or ethnic-group derogatory language |
| `svp` | Severe violent/predatory language |

Values are **% of videos in that group containing at least one term from each HurtLex category**.

### `04_harm_category_by_group_counts.csv`
Same as above but raw counts instead of percentages.

### `04_harm_category_by_group_long.csv`
Long-format version of the same data with columns: `Group`, `HurtLex Category`, `Count`, `N`, `Prevalence (%)`.

### `04_harm_category_outliers.csv`
Group-category cells more than 2 standard deviations above the cross-group mean for that harm category. Identifies exceptionally high harm concentrations.

---

## Step 05 — Statistical Bias Tests

### `05_bias_tests_ratings.csv` and `05_bias_tests_views.csv`
One row per comparison (group vs White Women).

| Column | Description |
|:-------|:------------|
| `Group` | Comparison group |
| `Privileged` | Always "White Women" (the reference group) |
| `N_group`, `N_priv` | Sample sizes |
| `Mean_group`, `Mean_priv` | Mean values for the outcome |
| `Median_group`, `Median_priv` | Median values |
| `MWU_U` | Mann-Whitney U statistic |
| `p_value` | Uncorrected p-value |
| `Cliffs_delta` | Effect size: P(group > privileged) − P(privileged > group). Range [−1, +1]. Negative = group has lower values. |
| `Cohens_d` | Standardised mean difference. Negative = group has lower mean. |
| `MedianDiff(group-priv)` | Median difference (group − privileged). Negative = group lower. |
| `CI_low`, `CI_high` | 95% bootstrap CI for the median difference |
| `p_fdr` | FDR-corrected p-value (Benjamini-Hochberg) |
| `reject@0.05FDR` | Whether the null hypothesis is rejected at α=0.05 FDR |
| `sig` | Significance symbol (\*\*: p<0.01, \*\*\*\*: p<0.0001) |

**Key findings:** All groups significantly below White Women on views (p<0.0001). Ratings are also significantly lower (Black Women Cliff's delta −0.073, p<1e-38). All results survive FDR correction.

---

## Step 06 — Stratified Splitting

### `06_train_ids.csv`, `06_val_ids.csv`, `06_test_ids.csv`
Single column: `video_id` (int). The canonical train/val/test split IDs. Every modelling step loads these and subsets the parquet.

- Train: 321,141 rows (60.0%)
- Val: 107,047 rows (20.0%)
- Test: 107,048 rows (20.0%)
- Disjointness verified: all three sets have zero overlap

### `06_stratify_key_distribution.csv`
| Column | Description |
|:-------|:------------|
| `class` | Stratification class (Other, White_Female, Black_Female, Asian_Female, Latina_Female) |
| `count` | Number of videos in this class across the full corpus |

### `06_split_selfcheck.csv`
30-row sample (10 per split) with `video_id` and `split` columns. Confirms disjointness is maintained.

---

## Steps 07 / 07a — RF Baseline & Category Sweep

### `07_overall_metrics.csv`
| Column | Description |
|:-------|:------------|
| `Split` | Val or Test |
| `Accuracy` | Overall classification accuracy |
| `Precision` | Positive-class precision |
| `Recall` | Positive-class recall (TPR) |
| `F1` | Harmonic mean of precision and recall |

**Canonical values:** Test acc 0.878, F1 0.758.

### `07_fairness_group_metrics.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group (Black Women, White Women, Asian Women, Latina Women, Other) |
| `N` | Number of videos in this group in the test set |
| `Accuracy` | Group-specific accuracy |
| `Precision` | Group-specific precision |
| `Recall` | Group-specific recall (TPR) — the key metric for the fairness audit |
| `F1` | Group-specific F1 |

### `07_fairness_disparities.csv`
| Column | Description |
|:-------|:------------|
| `Comparison Group` | The disadvantaged group being compared |
| `Accuracy Disparity` | White Women accuracy − Group accuracy. Positive = group disadvantaged. |
| `Equal Opportunity Difference (EOD)` | White Women recall − Group recall. Positive = group has lower recall. **The primary fairness metric.** |
| `Precision Disparity` | White Women precision − Group precision |

**Key values:** Black Women EOD 0.196, Latina Women EOD 0.200.

### `07_rf_val_predictions.csv` and `07_rf_test_predictions.csv`
One row per video in the val/test set.

| Column | Description |
|:-------|:------------|
| `video_id` | Video ID |
| `title` | Title (for qualitative inspection) |
| `Group` | Intersectional group label |
| `y_true` | True label (1=Amateur, 0=not) |
| `y_pred` | Predicted label at threshold 0.5 |
| `prob` | Predicted probability for the positive class (rounded to 3 dp) |
| `margin` | `prob − 0.5` (positive = predicted positive) |

### `07_rf_feature_importance.csv`
| Column | Description |
|:-------|:------------|
| `feature` | Feature name (`svd_comp_N` for SVD components, column name for numerics) |
| `importance` | RF Gini importance (sums to 1 across all features) |

Top feature is `svd_comp_3` (0.088) — the third SVD component captures the most discriminative variance in the tag vocabulary.

### `07_rf_outliers_top10.csv`
Top-10 most confident mistakes (highest `|prob − 0.5|` among errors).

| Column | Description |
|:-------|:------------|
| `video_id` | Video ID |
| `title` | Title |
| `Group` | Intersectional group |
| `y_true` | True label |
| `y_pred` | Predicted label |
| `prob` | Predicted probability |
| `margin_abs` | `|prob − 0.5|` — distance from decision boundary |

### `07_rf_val_metrics.csv`
Single-row summary of val performance.

---

### `07a_category_sweep_results.csv`
One row per candidate positive-class category.

| Column | Description |
|:-------|:------------|
| `Category` | Category name (positive class definition) |
| `TrainPos` | Number of positive training examples |
| `TrainN` | Total training examples |
| `ValPos`, `ValN` | Validation set counts |
| `TestPos`, `TestN` | Test set counts |
| `ValAcc`, `ValPrec`, `ValRec`, `ValF1`, `ValAUROC` | Val metrics at optimal threshold |
| `TestAcc`, `TestPrec`, `TestRec`, `TestF1`, `TestAUROC` | Test metrics at optimal threshold |
| `PrevalenceTrain` | Positive class prevalence in training set |
| `DecisionThreshold` | Val-F1-optimal threshold (applied to test set) |

**Key finding:** Amateur has the best balance of F1 (0.824) and interpretability. Big Tits has slightly higher F1 (0.840) but no causal mechanism linking it to fairness.

---

## Step 08 — Comprehensive Evaluation

### `08_group_metrics_at_opt.csv`
Per-group metrics at the F1-optimal operating threshold (0.40). Same schema as `07_fairness_group_metrics.csv`.

### `08_disparities_at_opt.csv`
Disparities at threshold 0.40. Same schema as `07_fairness_disparities.csv`. Black Women EOD drops from 0.196 (threshold 0.5) to 0.124 (threshold 0.40).

### `08_threshold_sweep.csv`
| Column | Description |
|:-------|:------------|
| `threshold` | Decision threshold (0.000 to 1.000 in steps of 0.005) |
| `accuracy` | Overall accuracy at this threshold |
| `precision` | Precision at this threshold |
| `recall` | Recall at this threshold |
| `f1` | F1 at this threshold |

### `08_fairness_curve.csv`
| Column | Description |
|:-------|:------------|
| `threshold` | Decision threshold |
| `max_abs_eod` | Maximum |EOD| across all groups at this threshold. Lower is fairer. |

**Key finding:** max|EOD| reaches its minimum around threshold 0.35–0.40, not 0.50.

### `08_confusion_matrix_at_opt.csv`
2×2 confusion matrix at operating threshold. Rows: `True_0`, `True_1`. Columns: `Pred_0`, `Pred_1`.

### `08_cadp_at_opt.csv`
Correlation-Adjusted Demographic Parity at operating threshold.

| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group |
| `AdjustedRate` | CADP-adjusted positive prediction rate |
| `Gap_vs_Priv` | Gap relative to White Women |
| `threshold` | Operating threshold used |

### `08_cadp_curve.csv`
CADP at all 201 thresholds (same structure as `08_cadp_at_opt.csv` but 1,005 rows = 201 thresholds × 5 groups).

### `08_rf_*.csv` (compatibility files)
`08_rf_overall_metrics.csv`, `08_rf_group_metrics.csv`, `08_rf_disparities.csv`, `08_rf_predictions_test.csv`, `08_rf_outliers_top10.csv` — these are copies of step 07 outputs written with `08_` prefix for downstream step compatibility. Identical in content.

---

## Step 09 — DistilBERT Baseline

### `09_overall_metrics.csv`
Same schema as `07_overall_metrics.csv`. Values: Test acc 0.924, F1 0.868.

### `09_fairness_group_metrics.csv`
Same schema as `07_fairness_group_metrics.csv`. Key: Black Women recall 0.861 (vs 0.562 RF baseline).

### `09_fairness_disparities.csv`
Same schema as `07_fairness_disparities.csv`. Black Women EOD 0.080 (vs 0.196 RF baseline).

### `09_bert_val_metrics.csv`, `09_bert_test_metrics.csv`
Single-row summaries of val/test performance.

### `09_bert_val_predictions.csv`, `09_bert_test_predictions.csv`
Same schema as `07_rf_*_predictions.csv` — video_id, title, Group, y_true, y_pred, prob, margin.

### `09_bert_outliers_top10.csv`
Same schema as `07_rf_outliers_top10.csv` — top-10 confident mistakes.

---

## Steps 10 / 10b — Reweighing

### `10_reweigh_overall_metrics.csv`
Same schema as `07_overall_metrics.csv`. Test acc 0.953, F1 0.915.

### `10_reweigh_group_metrics.csv`
Same schema as `07_fairness_group_metrics.csv`. Key: Black Women recall 0.838, EOD 0.092.

### `10_reweigh_disparities.csv`
Same schema as `07_fairness_disparities.csv`. Black Women EOD 0.092 (−53% vs baseline).

### `10_reweigh_predictions_test.csv`
Same schema as prediction files above.

### `10_reweigh_val_predictions.csv`
Val predictions — used by step 12 (ThresholdOptimizer) to fit per-group thresholds.

### `10_reweigh_outliers_top10.csv`
Top-10 confident mistakes under reweighing.

### `rf_reweighed_val_predictions.csv`
Legacy name for `10_reweigh_val_predictions.csv` — kept for backward compatibility with downstream steps.

### `10b_matchedN_overall_metrics.csv`
Same schema as overall metrics. Test acc 0.951 — confirms reweighing gain is not a sample-size artefact.

### `10b_matchedN_group_metrics.csv`, `10b_matchedN_disparities.csv`
Per-group and disparity tables for the matched-N experiment.

---

## Step 11 — In-processing (EG + DP)

### `11_inproc_overall_metrics.csv`
Test acc 0.878 (same as baseline), but recall 0.914 / precision 0.726 — the high-recall/low-precision signature of the DP constraint.

### `11_inproc_group_metrics.csv`
**Key finding:** Latina Women EOD 0.297 — *worse* than the baseline (0.200).

### `11_inproc_disparities.csv`
Latina Women accuracy disparity 0.162, EOD 0.297 — the DP constraint fails for this group.

### `11_inproc_predictions_test.csv`, `11_inproc_outliers_top10.csv`
Predictions and top errors.

### `11_inproc_importances.csv`
Feature importances from the underlying LogisticRegression (coefficient magnitudes).

### `egdp_*.csv` (legacy names)
`egdp_overall_metrics_inproc.csv`, `egdp_group_metrics_inproc.csv`, `egdp_disparities_inproc.csv`, `egdp_outliers_topk_inproc.csv`, `egdp_predictions_test_inproc.csv` — copies of step 11 outputs with legacy naming for downstream compatibility.

---

## Step 12 — Post-processing (ThresholdOptimizer)

### `12_postproc_overall_metrics.csv`
Test acc 0.839 (−3.9pp from baseline). The accuracy cost of near-exact parity.

### `12_postproc_group_metrics.csv`
All group recalls cluster between 0.623 and 0.650 — near-equal opportunity achieved.

### `12_postproc_disparities.csv`
**Key values:** Black Women EOD 0.008 (−96%), Latina Women EOD 0.025, Asian 0.027. Latina precision disparity −0.032 (negative = slight over-correction).

### `12_postproc_predictions_test.csv`, `12_postproc_outliers_top10.csv`
Predictions and top errors.

### `egpp_*.csv` (legacy names)
`egpp_overall_metrics_postproc.csv`, `egpp_group_metrics_postproc.csv`, `egpp_disparities_postproc.csv`, `egpp_outliers_top10_postproc.csv`, `egpp_predictions_test_postproc.csv` — copies of step 12 outputs with legacy naming.

---

## Step 13 — Mitigation Effectiveness

### `mitigation_effectiveness.csv`
| Column | Description |
|:-------|:------------|
| `Model` | Strategy name |
| `Test Accuracy` | Overall test accuracy |
| `Accuracy Δ (vs White) [Black Women]` | White Women accuracy − Black Women accuracy |
| `TPR Δ (vs White) [Black Women]` | White Women recall − Black Women recall (EOD) |

---

## Step 15 — Qualitative Deep Dive

### `15_qualitative_error_samples_topk.csv`
| Column | Description |
|:-------|:------------|
| `video_id` | Video ID |
| `title` | Video title |
| `Group` | Intersectional group |
| `y_true` | True label |
| `y_pred` | Predicted label |
| `prob` | Predicted probability |
| `margin_abs` | Distance from decision boundary |

---

## Step 16 — Temporal Analysis

### `16_temporal_group_representation.csv`
Group × year representation table: `Group`, `Year`, `N`, `Share %`.

### `16_temporal_rating_view_trends.csv`
Yearly aggregate trends by group flag (binary: 1 = Asian, etc.):

| Column | Description |
|:-------|:------------|
| `_publish_year` | Calendar year |
| `race_ethnicity_asian` | Binary flag (0=Others, 1=Asian Women) |
| `mean_rating` | Mean rating score for this group × year cell |
| `mean_views` | Mean total views |
| `mean_views_per_day` | Mean age-normalised views/day |
| `mean_ratings_per_day` | Mean ratings/day |
| `n` | Number of videos |

This is the source for the step 19 slope estimation.

### `16_engagement_bias_by_group.csv`
Engagement summary table by group.

### `16_engagement_outliers_top10.csv`
Top-10 videos with extreme engagement relative to group norms.

---

## Step 17 — Engagement Bias Analysis

### `17_gap_rating.csv` and `17_gap_views_per_day.csv`
Yearly gap (Black Women − Others) with bootstrap confidence intervals.

| Column | Description |
|:-------|:------------|
| `Year` | Calendar year |
| `gap_mean` | Mean(BW) − Mean(Others) for the metric |
| `ci_lo`, `ci_hi` | 95% bootstrap CI for the gap |
| `bw_n`, `others_n` | Sample sizes |

**Key interpretation:** When `ci_hi < 0`, the gap is statistically significantly negative (BW below Others) at 95% confidence.

### `17_head_tail_composition.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group |
| `Head_Share %` | Group's share of top-1% videos by views/day |
| `Tail_Share %` | Group's share of bottom-99% videos |

### `17_yearly_bw_gaps.csv`
Yearly summary of Black Women vs Others gaps across multiple metrics.

### `17_quantiles_race_ethnicity.csv`
Views-per-day quantile distribution by race/ethnicity flag.

### `17_bw_correlations.csv`
Point-biserial correlations between the Black Women binary indicator and continuous outcome variables (views, ratings, duration).

---

## Step 18 — Category Group Dynamics

### `18_category_group_matrix.csv`
Binary presence matrix: rows = categories, columns = intersectional groups. Value = % of group's videos with this category.

### `18_bw_under_over.csv`
| Column | Description |
|:-------|:------------|
| `Category` | Category name |
| `BW_share` | % of Black Women videos with this category |
| `Global_share` | % of all videos with this category |
| `log2_RR` | log₂(BW_share / Global_share) — representation ratio. Positive = over-represented, negative = under-represented. |

### `18_category_cooccurrence.csv`
Category × category co-occurrence matrix (used as input by step 20 network analysis). Values are counts of videos containing both categories simultaneously.

---

## Step 19 — Advanced Statistics

### `19_trend_slopes.csv`
**The key temporal harm evidence.**

| Column | Description |
|:-------|:------------|
| `metric` | `views_per_day` or `rating` |
| `slope_bw` | OLS annual slope for Black Women |
| `slope_bw_lo`, `slope_bw_hi` | 95% bootstrap CI for BW slope |
| `slope_others` | OLS annual slope for all others |
| `slope_others_lo`, `slope_others_hi` | 95% CI for Others slope |
| `slope_gap_bw_minus_others` | **Key finding**: BW slope − Others slope |
| `slope_gap_lo`, `slope_gap_hi` | 95% CI for the gap. Both bounds negative for `rating` → statistically significant disadvantage. |
| `n_bw`, `n_others` | Sample sizes (n_bw = 7,136; n_others = 528,100) |

**Canonical values:** `slope_gap_rating = −3.38`, CI [−3.75, −2.98].

### `19_bw_effect_sizes.csv`
| Column | Description |
|:-------|:------------|
| `metric` | Outcome variable |
| `cliffs_delta` | Effect size |
| `ci_lo`, `ci_hi` | 95% CI for Cliff's delta |
| `n_bw`, `n_others` | Sample sizes |

### `19_category_divergence.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group |
| `KL_divergence` | KL divergence of group's category distribution from the global distribution |

Higher KL = more distinctive category vocabulary. Asian Women have the highest KL (0.555), explaining why EG+DP fails so severely for them.

---

## Step 20 — Network Analysis

### `20_category_centrality.csv`
| Column | Description |
|:-------|:------------|
| `Category` | Category name |
| `degree` | Number of distinct categories it co-occurs with |
| `strength` | Sum of all edge weights (total co-occurrence mass) |
| `betweenness` | Fraction of shortest paths passing through this node |
| `pagerank` | PageRank centrality score |
| `clustering` | Local clustering coefficient |

---

## Step 22 — Ablation Studies (`outputs/ablation/`)

### `22_ablation_all.csv`
Summary of all ablation experiments with the baseline KPI (mean absolute log₂ representation ratio for Black Women across top categories).

### `22_rr_lexicon_off.csv`
KPI when lexicon-based group assignment is disabled (random baseline).

### `22_topcats_noise_{10,25,50}.csv`
KPI under 10%, 25%, 50% random noise in category assignments.

### `22_topk_mass_{10,20,30,50}.csv`
KPI computed using only the top-10, top-20, top-30, or top-50 categories by coverage.

---

## Step 23 — Limitations Analysis

### `23_data_quality.csv`
| Column | Description |
|:-------|:------------|
| `metric` | Quality metric name |
| `value` | Measured value |
| `status` | PASS / WARN / FAIL |

Covers: missingness rates, duplicate rate, incoherent rating rate, animated content rate.

### `23_representation_skew.csv`
| Column | Description |
|:-------|:------------|
| `Group` | Intersectional group |
| `corpus_share` | % of corpus |
| `ideal_share` | Equal-share baseline (100/5 = 20%) |
| `skew` | `corpus_share / ideal_share` |

### `23_temporal_drift.csv`
Video age distribution by group × year.

### `23_category_longtail.csv`
| Column | Description |
|:-------|:------------|
| `top_k` | Number of top categories included |
| `cumulative_coverage` | % of all category assignments covered by top-k categories |

### `23_summary_checklist.csv`
Binary checklist (PASS/WARN/FAIL) for each Datasheets and Model Cards criterion.

### `23_missingness.csv`
Per-column missing value rates across the corpus.

---

## Steps 24–26 — Synthesis

### `24_ablation_kpis.csv`
Key performance indicators from the ablation study, formatted for the dissertation.

### `24_checklist.csv`
Dissertation-ready checklist of what has and has not been validated.

### `24_limitations_kpis.csv`
Limitations metrics formatted for the dissertation.

### `25_pareto_points.csv`
**The core Pareto table.**

| Column | Description |
|:-------|:------------|
| `model` | Strategy name |
| `accuracy` | Test accuracy |
| `fairness` | Fairness score = `1 − max|EOD|` across all groups |

**Canonical values:**

| model | accuracy | fairness |
|:------|--------:|---------:|
| RF reweighed | 0.954 | 0.901 |
| BERT baseline | 0.924 | 0.817 |
| RF baseline | 0.881 | 0.800 |
| RF inproc | 0.880 | 0.703 |
| RF postproc | 0.840 | 0.968 |

### `25_pareto_frontier.csv`
Subset of `25_pareto_points.csv` containing only non-dominated strategies (RF reweighed + RF postproc).

---

## Step 30 — Causal Analysis

### `30_ate_rating.csv`
Single-row result of the IPTW estimation.

| Column | Description |
|:-------|:------------|
| `N` | Full corpus size (535,236) |
| `ATE_rating` | Average Treatment Effect: rating points gained by being classified as Amateur (after adjustment) |
| `lo95`, `hi95` | 95% confidence interval for ATE |
| `clip` | Propensity clipping threshold used (0.01) |
| `overlap_share_0.1_0.9` | Proportion of propensity scores in [0.1, 0.9] — positivity check. Above 0.80 is acceptable. |
| `w99` | 99th percentile of IPTW weights (extreme weights indicate overlap issues) |
| `n_above_w99` | Number of videos with weight above w99 |
| `n_clip_lo`, `n_clip_hi` | Videos with propensity clipped at lower/upper bound |

**Canonical value:** ATE_rating = 7.216. Interpretation: Amateur-classified content earns ~7.2 rating points more than equivalent non-Amateur content.

### `30_sensitivity.csv`
| Column | Description |
|:-------|:------------|
| `clip` | Clipping threshold tested (0.01, 0.02, 0.05) |
| `ATE`, `lo95`, `hi95` | ATE and CI at this clipping level |
| `n_above_w99`, `n_clip_lo`, `n_clip_hi` | Diagnostics |

All three clipping levels give ATE ≈ 7.216, confirming robustness.

---

## Supporting Files

### `cgd_category_cooccurrence.csv`
Alternative name for `18_category_cooccurrence.csv` (written by an earlier naming scheme in step 18). Same content.

### `selfcheck_fairness_metrics.csv`, `selfcheck_fairness_disparities.csv`, `selfcheck_outliers.csv`
Written by `fairness_evaluation_utils.py --selfcheck`. Quick validation files with a toy coin-flip predictor. Not part of the main results.

### `gold/` directory
Contains gold annotation files (`gold_template.csv`, annotator fills, `gold_final.csv`). See step 27 documentation in `codebook.md`.

---

## Column Name Conventions

| Pattern | Meaning |
|:--------|:--------|
| `y_true` | Binary ground truth (1=positive/Amateur, 0=negative) |
| `y_pred` | Binary prediction at operating threshold |
| `prob` | Predicted probability for the positive class, rounded to 3 dp |
| `margin` | `prob − 0.5`. Positive = classified positive, negative = classified negative |
| `margin_abs` | `|prob − 0.5|`. Larger = higher confidence |
| `EOD` | Equal Opportunity Difference = privileged recall − group recall. Positive = group disadvantaged. |
| `_selfcheck` suffix | Result computed on a random subset; schema identical to canonical version |
| `Split` | "Val" or "Test" |
| `N` | Count of videos in this cell |

---

*For the code that generates each file, see `dissertation/codebook.md`. For interpretation in the context of research questions, see `dissertation/analysis.md`.*
