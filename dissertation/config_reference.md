# Configuration Reference

**Location:** `config/`  
**Files:** `settings.yaml`, `protected_terms.json`, `abusive_lexica/hurtlex_EN.tsv`, `abusive_lexica/baseLexicon.txt`, `abusive_lexica/expandedLexicon.txt`

All configuration is centralised here. Scripts read these files via `src/utils/theme_manager.load_config()` and never hardcode paths or parameters.

---

## `settings.yaml` — Master Configuration

The single source of truth for all pipeline parameters. Loaded at module import time by every script. Variables are interpolated using `${key.subkey}` syntax.

### Section 1: Project identity

```yaml
project:
  name: AlgoFairnessPornometrics
  root: "."       # Resolved to the absolute path of the repo at runtime
```

All path resolution uses `project.root` as the base. When running from any directory, paths resolve correctly as long as the working directory is the repo root.

### Section 2: Paths

```yaml
paths:
  data: "${project.root}/outputs/data"        # All CSV outputs
  config: "${project.root}/config"            # Lexica and settings
  outputs: "${project.root}/outputs"          # Root of all generated artefacts
  figures: "${project.root}/outputs/figures"  # PNG files
  narratives: "${project.root}/outputs/narratives"  # Markdown summaries
  db_file: "${project.root}/data/redtube_videos.db" # Raw SQLite (not in git)
```

**Important:** `data/redtube_videos.db` is excluded from git via `.gitignore`. When cloning on a new machine, copy the database file manually before running the pipeline.

### Section 3: Reproducibility

```yaml
reproducibility:
  seed: 95
```

This seed is used for:
- All `random_state` arguments to scikit-learn models
- `numpy.random.default_rng(seed)` in bootstrap procedures
- `train_test_split` in step 06
- HuggingFace `TrainingArguments(seed=95)`
- All `df.sample(random_state=seed)` calls

**Never change this value** after the canonical run. Changing it would invalidate all committed results.

### Section 4: Visualisation

```yaml
viz:
  dpi: 300          # PNG resolution (publication quality)
  save_png: true
  save_pdf: false   # PDF excluded to reduce artefact size

  palettes:
    sections:
      eda:          # Exploratory plots: balanced, approachable hues
        light: [...]
        dark: [...]
      fairness:     # Fairness plots: emphasise interpretability
        light: [...]
        dark: [...]
      models:       # Model diagnostics: strong contrast
        light: [...]
        dark: [...]
```

All palettes are **colorblind-friendly** (verified against WCAG 2.1 colour contrast guidelines). The `@plot_dual_theme(section="fairness")` decorator selects the palette by section name.

### Section 5: Database table names

```yaml
db:
  tables:
    videos: "videos"
    video_tags: "video_tags"
    video_categories: "video_categories"
```

These match the actual SQLite table names in `redtube_videos.db`. If you ever need to rename tables in the database, change here first.

### Section 6: Project specifics

```yaml
project_specifics:
  intersection:
    primary_race_col: "race_ethnicity_black"
    primary_gender_col: "gender_female"
    output_col_name: "intersectional_black_female"

  feature_generation_keys:
    - "stereotype_terms"
    - "race_ethnicity"
    - "nationality"
    - "hair_color"
    - "gender"
    - "sexuality"
    - "age"
```

`feature_generation_keys` lists which top-level groups from `protected_terms.json` are used to generate one-hot columns. The order matters: groups are processed in this sequence, and column names follow the pattern `{group}_{subgroup}` (e.g., `race_ethnicity_black`, `gender_female`).

The `intersection` section defines which columns are combined for the primary intersectional indicator (`intersectional_black_female = race_ethnicity_black AND gender_female`).

---

## `protected_terms.json` — Demographic Inference Lexicon

**Size:** 1,659 lines | **Top-level groups:** 10

This file drives all demographic group inference. It maps text terms (from video titles, tags, and categories) to protected-attribute dimensions. It is the most sensitive configuration file — changes here affect every corpus statistic, every group metric, and every fairness result.

### Structure

```json
{
  "_source_notes": "...",
  "regex_settings": { ... },
  "race_ethnicity": { "asian": [...], "black": [...], ... },
  "gender": { "female": [...], "gender_male": [...], "gender_trans": [...] },
  "nationality": { ... },
  "hair_color": { ... },
  "sexuality": { ... },
  "age": { ... },
  "stereotype_terms": { ... },
  "bias_markers": { ... },
  "vocabulary_rules": { ... }
}
```

### `regex_settings`

Controls how terms are compiled into regular expressions by `_compile_glob_terms_to_regex()`:

| Setting | Value | Effect |
|:--------|:------|:-------|
| `casefold` | `true` | Case-insensitive matching (`re.IGNORECASE`) |
| `word_boundaries` | `true` | Wraps each term in `\b...\b` to prevent partial matches ("asian" won't match "caucasian") |
| `allow_hyphen_variants` | `true` | Treats `-`, space, and `_` as interchangeable |
| `ascii_fallback` | `true` | Falls back to ASCII-normalised matching for accented characters |
| `exceptions` | `[{"pattern": "asian", "negative_contexts": ["caucasian"]}]` | Explicitly blocks "asian" from matching inside "caucasian" (belt-and-suspenders beyond word boundaries) |

### `race_ethnicity` group

| Subgroup | Key terms (first 5) | One-hot column created |
|:---------|:---------------------|:-----------------------|
| `asian` | asian, oriental, japanese, chinese, korean | `race_ethnicity_asian` |
| `black` | black, ebony, blk, african, afro | `race_ethnicity_black` |
| `latina` | latina, latino, latinx, hispanic, chicana | `race_ethnicity_latina` |
| `white` | white, caucasian, european | `race_ethnicity_white` |
| `middle_eastern_north_african` | arab, middle eastern, mena | `race_ethnicity_middle_eastern_north_african` |
| `mixed_or_other` | interracial, mixed, indian, south asian, native american | `race_ethnicity_mixed_or_other` |

**Note on inference limitations.** Group assignment is based on user-submitted text, not creator self-identification. A video tagged "asian" may or may not be produced by an Asian creator — the lexicon captures the *platform's categorisation*, not creator identity. All findings should be interpreted as "how the platform categorises this content," not "properties of the creators themselves."

### `gender` group

| Subgroup | Key terms | One-hot column |
|:---------|:----------|:---------------|
| `female` | female, woman, women, girl, she | `gender_female` |
| `gender_male` | male, man, men, boy, he | `gender_gender_male` |
| `gender_trans` | transgender, trans, transsexual, ftm, mtf | `gender_gender_trans` |

### Group priority and mutual exclusivity

The one-hot columns are **not mutually exclusive** — a video can have both `race_ethnicity_black=1` and `gender_female=1`. The priority-ordered intersectional labels (`group_labels_intersectional()` in `fairness_evaluation_utils.py`) enforce exclusivity:

```
Black Women > White Women > Asian Women > Latina Women > Other
```

A video with both `race_ethnicity_black=1` and `race_ethnicity_asian=1` is assigned to "Black Women." This priority was chosen to ensure the most disadvantaged group is not diluted by multi-assignment.

### How to extend the lexicon

To add a new term to the `race_ethnicity_black` group:
1. Open `config/protected_terms.json`
2. Add the term to the `race_ethnicity.black` array: `"race_ethnicity": {"black": ["black", "ebony", ..., "your_new_term"]}`
3. Re-run step 01 (`python src/data/01_corpus_builder.py`) to rebuild the parquet
4. Re-run all downstream steps

To add an entirely new protected group:
1. Add a new top-level key under `race_ethnicity` or another section
2. Add that group name to `feature_generation_keys` in `settings.yaml`
3. Re-run the full pipeline

---

## `abusive_lexica/hurtlex_EN.tsv` — HurtLex English Harm Lexicon

**Source:** Bassignana et al. (2020). HurtLex: A Multilingual Lexicon of Words to Hurt. Proceedings of LREC 2020.  
**Size:** 8,229 terms | **Format:** TSV with header

### Schema

| Column | Description |
|:-------|:------------|
| `id` | Unique term ID (e.g., `EN1382`) |
| `pos` | Part of speech: `n` (noun), `a` (adjective), `v` (verb) |
| `category` | HurtLex harm category code (see table below) |
| `stereotype` | `yes` if the term encodes a stereotype, `no` otherwise |
| `lemma` | Canonical form of the term |
| `level` | `inclusive` (widely used, including in reclaimed contexts) or `conservative` (narrower, more reliably offensive) |

### HurtLex Category Codes

| Code | Full name | Example terms |
|:-----|:----------|:--------------|
| `an` | Animosity | hate, enemy |
| `asf` | Aggressive sexual fantasy | rape, assault |
| `asm` | Aggressive sexual material | (sexual violence terms) |
| `cds` | Derogatory or demeaning sexual language | slut, whore |
| `ddf` | Defamatory/discriminatory — female-targeted | bitch, hag |
| `ddp` | Defamatory/discriminatory — general | freak, weirdo |
| `dmc` | Derogatory terms for minority/community | (ethnic slurs) |
| `is` | Identity-based stigmatisation | (disability slurs) |
| `om` | Objectifying material | (body part objectification) |
| `or` | Objectifying references | (dehumanising terms) |
| `pa` | Physical assault language | hit, punch |
| `pr` | Prostitution references | hooker, escort |
| `ps` | Personal stigmatisation | loser, freak |
| `qas` | Quasi-abusive speech | idiot, stupid |
| `rci` | Racial/cultural insult | (racial slurs) |
| `re` | Racist or ethnic derogatory language | (ethnic slurs) |
| `svp` | Severe violent/predatory language | murder, kill |

Step 04 checks whether HurtLex terms (filtered to `inclusive` level for broader coverage) appear in video metadata (title + tags) grouped by intersectional group.

### Usage in step 04

```python
# Simplified logic from 04_multilayer_harm_analysis.py
for term, category in hurtlex_terms:
    col = f"hurtlex_{category}"
    df[col] = df['combined_text_clean'].str.contains(term, regex=False)
```

Results are aggregated as `% of videos in group containing ≥1 term from category`.

---

## `abusive_lexica/baseLexicon.txt` — Base Abusive Terms

**Size:** 1,650 terms | **Format:** Plain text, one term per line

A manually curated list of terms identified as potentially harmful in adult platform metadata. More conservative than HurtLex — each term has been reviewed for relevance to the platform context.

Used as a supplementary signal in step 04 alongside HurtLex.

---

## `abusive_lexica/expandedLexicon.txt` — Expanded Abusive Terms

**Size:** 8,478 terms | **Format:** Plain text, one term per line

An expanded version of the base lexicon generated through semantic similarity expansion (word embeddings + manual review). Includes variants, misspellings, and platform-specific slang.

**Caution:** The expanded lexicon has lower precision than the base lexicon — some terms may be used non-offensively in context (e.g., reclaimed language). Results from analyses using the expanded lexicon should be interpreted with this caveat.

---

## Configuration Dependency Graph

```
settings.yaml
├── paths.data        → all outputs/data/ writes
├── paths.db_file     → collector.py, 01_corpus_builder.py
├── reproducibility.seed → all stochastic operations
├── viz.palettes      → plot_dual_theme decorator
├── db.tables         → collector.py, 01_corpus_builder.py
└── project_specifics
    ├── feature_generation_keys → 01_corpus_builder.py
    └── intersection            → 01_corpus_builder.py, 06_stratified_splitting.py

protected_terms.json
└── read by 01_corpus_builder.py → creates one-hot columns in parquet
    → used by all downstream steps for group labelling

hurtlex_EN.tsv + baseLexicon.txt + expandedLexicon.txt
└── read by 04_multilayer_harm_analysis.py → harm category matrix
```

---

## Changing Configuration Safely

### Changing the seed
**Impact:** All stochastic results change. All committed CSVs become stale.  
**Required re-runs:** Full pipeline (steps 01–30).  
**Recommended approach:** Do not change for reproduction. Change only for a new PhD-extension experiment, branching the repo first.

### Adding lexicon terms
**Impact:** Group counts in step 01 change. All downstream fairness metrics change.  
**Required re-runs:** Steps 01, 02, 03, 05, 06, 07, 08, 09, 10, 11, 12, 13, 16, 17, 18, 19, 22, 24, 25.  
**Validation:** After adding terms, run `python src/data/01_corpus_builder.py --selfcheck` and check that `01_corpus_stats.json` shows increased counts for the modified group.

### Adding a new protected group
**Impact:** As above, plus group priority ordering in `fairness_evaluation_utils.py` must be updated manually.  
**Required changes:** `protected_terms.json`, `settings.yaml` (`feature_generation_keys`), `fairness_evaluation_utils.py` (`group_labels_intersectional()`), `06_stratified_splitting.py` (`make_stratify_key()`).

### Changing paths
**Impact:** All scripts that use `load_config()` will read the new path. No code changes needed.  
**Caution:** If moving `outputs/data/` to a different location, ensure the new path has write permissions and enough disk space (~2 GB for the full pipeline outputs).

---

*For usage in context, see `dissertation/codebook.md` (which scripts use each config key) and `dissertation/analysis.md` (research implications of configuration choices).*
