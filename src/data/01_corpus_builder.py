# -*- coding: utf-8 -*-
"""
01_corpus_builder.py
====================

Purpose
-------
Build the **master machine-learning corpus** from the raw SQLite database.
This is Step 01 and the foundation for all downstream analyses.

Inputs
------
- SQLite DB path from `config/settings.yaml` (see `src/utils/database.py`)
    Tables:
        - `videos` (title, views, ratings, publish_date, etc.)
        - `video_tags` (video_id, tag)
        - `video_categories` (video_id, category)
        - Protected terms lexicon: `config/protected_terms.json`

Outputs
-------
- `outputs/data/ml_corpus.parquet` (canonical corpus)
- `outputs/data/corpus_stats.json` (schema & group coverage stats)

Key Columns Produced
--------------------
- `combined_text_clean` (normalized text for lexicon matching)
- Binary protected-attribute columns (e.g., `race_ethnicity_black`, `gender_female`)
- `intersectional_black_female` (1/0)

Runtime
-------
Prints total elapsed seconds:
    [TIME] Step 01 runtime: XX.XXs

Notes
-----
- Does **not** write to the DB; reads only.
"""

import re
import os
import json
import time
import sqlite3
import warnings
from pathlib import Path
from typing import Tuple, Dict, List

import pandas as pd

# --- 1. Configuration and Setup ---
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config
from src.utils.database import create_connection

CONFIG = load_config()

# Define paths from the config
CORPUS_PATH = Path(CONFIG['paths']['data']) / '01_ml_corpus.parquet'
STATS_PATH  = Path(CONFIG['paths']['data']) / '01_corpus_stats.json'
SELF_PATH   = Path(CONFIG['paths']['data']) / '01_corpus_selfcheck.csv'

LEXICON_PATH = Path(CONFIG['paths']['config']) / 'protected_terms.json'

# --- 2. Lexicon helpers ------------------------------------------------------

def _compile_glob_terms_to_regex(terms: List[str], settings: Dict) -> Tuple[str, int]:
    """
    Compile a list of 'glob-like' terms (supporting '*' wildcards) into a single regex.

    Parameters
    ----------
    terms : list of str
        Terms from the lexicon, may include '*' as wildcard pieces.
    settings : dict
        `regex_settings` from the lexicon. Keys:
            - casefold: bool -> use re.IGNORECASE
            - word_boundaries: bool -> wrap pattern with \\b...\\b
            - allow_hyphen_variants: bool -> treat '-', space, '_' as interchangeable

    Returns
    -------
    pattern : str
        Combined safe regex using non-greedy wildcard pieces.
    flags : int
        Flags for pandas .str.contains(..., regex=True, flags=...).

    Notes
    -----
    - '*' becomes '.*?' to avoid over-greedy matches.
    - All other characters are escaped.
    """
    flags = 0
    if isinstance(settings, dict) and settings.get("casefold", True):
        flags |= re.IGNORECASE

    safe_pieces: List[str] = []
    for raw in terms:
        if not isinstance(raw, str) or not raw.strip():
            continue
        t = raw.strip()
        parts = [re.escape(p) for p in t.split('*')]
        piece = ".*?".join(parts)
        if settings.get("allow_hyphen_variants", True):
            piece = re.sub(r"(\\ |-)+", r"[-\\s_]+", piece)
        if settings.get("word_boundaries", True):
            piece = r"\b" + piece + r"\b"
        safe_pieces.append(f"(?:{piece})")

    if not safe_pieces:
        return r"(?!x)x", flags  # match nothing
    return "|".join(safe_pieces), flags


def _load_protected_terms() -> dict:
    """
    Load the protected-terms lexicon JSON.

    Returns
    -------
    dict
        Parsed lexicon including optional "regex_settings".

    Timing
    ------
    Prints a [TIME] line with elapsed seconds.
    """
    t0 = time.perf_counter()
    try:
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[TIME] _load_protected_terms: {time.perf_counter() - t0:.2f}s")
        return data
    except Exception as e:
        print(f"[WARN] Could not load protected_terms.json at {LEXICON_PATH}: {e}")
        print(f"[TIME] _load_protected_terms: {time.perf_counter() - t0:.2f}s")
        return {}


PROTECTED_TERMS = _load_protected_terms()


# --- 3. Data fetching --------------------------------------------------------

def fetch_data_from_db(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Fetch and join core data from SQLite (lossless; every video is kept).

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to the SQLite database.

    Returns
    -------
    pd.DataFrame
        One row per video with aggregated, de-duplicated tags and categories.

    Notes
    -----
    - Uses two correlated subqueries to avoid cross-multiplication from joins.
    - COALESCE → empty strings for missing tags/categories (easier downstream).
    - Prints [TIME] with elapsed seconds.
    """
    t0 = time.perf_counter()
    print("Fetching data from database...")

    q = f"""
    SELECT
        v.video_id,
        v.title,
        v.duration,
        v.views,
        v.rating,
        v.ratings,
        v.publish_date,
        COALESCE((
            SELECT GROUP_CONCAT(DISTINCT t.tag)
            FROM {CONFIG['db']['tables']['video_tags']} t
            WHERE t.video_id = v.video_id
        ), '') AS tags,
        COALESCE((
            SELECT GROUP_CONCAT(DISTINCT c.category)
            FROM {CONFIG['db']['tables']['video_categories']} c
            WHERE c.video_id = v.video_id
        ), '') AS categories
    FROM {CONFIG['db']['tables']['videos']} v;
    """

    try:
        df = pd.read_sql_query(q, conn)
        print(f"✓ Fetched {len(df):,} videos.")
    except Exception as e:
        print(f"✗ ERROR: Failed to fetch data. Reason: {e}")
        df = pd.DataFrame()

    print(f"[TIME] fetch_data_from_db: {time.perf_counter() - t0:.2f}s")
    return df


# --- 4. Feature engineering --------------------------------------------------

def clean_text(text: str) -> str:
    """
    Basic text normalization for lexicon matching and PMI analysis.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Lowercased, alphanumeric+space, squashed whitespace.

    Notes
    -----
    Commas are replaced with spaces BEFORE stripping punctuation so that
    comma-separated tokens (tags, categories stored as "tag1,tag2,tag3")
    are separated rather than concatenated.  Without this step, "white
    guy,ebony" would become "white guyebony" — a spurious PMI artefact.
    """
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = s.replace(",", " ")          # separate comma-joined tokens first
    s = re.sub(r"[^a-z0-9\s]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def create_protected_group_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Generate one-hot protected-attribute indicators using the lexicon.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing `text_col`.
    text_col : str
        Normalized text column to scan.

    Returns
    -------
    pd.DataFrame
        Copy with additional one-hot columns based on
        CONFIG['project_specifics']['feature_generation_keys'].

    Behavior
    --------
    - Supports '*' wildcards from the lexicon via `_compile_glob_terms_to_regex`.
    - Suppresses benign pandas regex warnings for large alternations.
    """
    t0 = time.perf_counter()
    print("Creating protected group features...")
    out = df.copy()
    feature_keys = CONFIG['project_specifics']['feature_generation_keys']
    regex_settings = PROTECTED_TERMS.get("regex_settings",
    {"casefold": True, "word_boundaries": True, "allow_hyphen_variants": True})

    for group in feature_keys:
        categories = PROTECTED_TERMS.get(group, {})
        if not isinstance(categories, dict):
            continue

        for category, terms in categories.items():
            if not isinstance(terms, list):
                continue
            valid_terms = [t for t in terms if isinstance(t, str) and t.strip()]
            if not valid_terms:
                continue

            col = f"{group}_{category}"
            pattern, flags = _compile_glob_terms_to_regex(valid_terms, regex_settings)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    out[col] = out[text_col].str.contains(pattern, regex=True, flags=flags, na=False).astype(int)
                except re.error as e:
                    print(f"[WARN] Regex compile failed for {col}: {e}; defaulting to zeros.")
                    out[col] = 0

    print("✓ Protected group features created.")
    print(f"[TIME] create_protected_group_features: {time.perf_counter() - t0:.2f}s")
    return out


def create_intersectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create intersectional indicators (e.g., Black × Female) per config.

    Returns
    -------
    pd.DataFrame
        Copy with intersectional column if inputs exist.
    """
    print("Creating intersectional features...")
    out = df.copy()
    isect = CONFIG['project_specifics']['intersection']
    race_col = isect['primary_race_col']
    gender_col = isect['primary_gender_col']
    output_col = isect['output_col_name']

    if race_col in out.columns and gender_col in out.columns:
        out[output_col] = ((out[race_col] == 1) & (out[gender_col] == 1)).astype(int)
        print(f"✓ Created '{output_col}' feature.")
    else:
        print(f"✗ WARNING: Could not create '{output_col}' (missing columns).")
    return out


# --- 5. Quick sanity prints --------------------------------------------------

def _explode(series: pd.Series) -> pd.Series:
    """
    Explode comma-separated strings to a clean 1-D series.
    """
    s = series.fillna('').astype(str).str.lower().str.split(',')
    s = s.explode().str.strip()
    return s[s != '']


def quick_sanity_report(df: pd.DataFrame) -> None:
    """
    Print real numbers to reassure the corpus is complete.

    Prints
    ------
    - Total videos
    - Top 10 categories (by videos)
    - Top 10 tags (by videos containing the tag)
    """
    print("\n=== Step 01 Quick Sanity Report (real numbers) ===")
    print(f"Total videos: {len(df):,}")

    cats = _explode(df['categories'])
    top_cats = cats.value_counts().head(10)
    if not top_cats.empty:
        pct = (top_cats / max(len(df), 1) * 100).round(1)
        print("\nTop 10 categories (by video coverage):")
        for name, cnt in top_cats.items():
            print(f"  - {name}: {cnt:,} videos ({pct[name]:.1f}%)")
    else:
        print("\nNo categories available.")

    tags = _explode(df['tags'])
    top_tags = tags.value_counts().head(10)
    if not top_tags.empty:
        # % of videos that have the tag at least once
        tag_sets = df['tags'].fillna('').str.lower()
        print("\nTop 10 tags (by video coverage):")
        for tag, _ in top_tags.items():
            # use non-capturing groups to avoid pandas warning
            mask = tag_sets.str.contains(rf"(?:^|,)\s*{re.escape(tag)}\s*(?:,|$)", regex=True)
            vids = int(mask.sum())
            pct = vids / max(len(df), 1) * 100
            print(f"  - {tag}: {vids:,} videos ({pct:.1f}%)")
    else:
        print("\nNo tags available.")
    print("=== End sanity report ===\n")


def write_selfcheck(df: pd.DataFrame, out_path: Path, n: int = 10) -> None:
    """
    Write a tiny random-sample self-check that never overwrites full artefacts.

    Parameters
    ----------
    df : pd.DataFrame
        Full corpus dataframe (not modified).
    out_path : Path
        outputs/data/01_corpus_selfcheck.csv
    n : int
        Number of random rows to sample for the check.

    Behavior
    --------
    - Uses global seed from config for reproducibility (seed=95).
    - Keeps years as integers, ratings with 1 decimal.
    - Includes key columns to quickly eyeball correctness.
    """
    t0 = time.perf_counter()
    rng = pd.Series(range(len(df))).sample(n=min(n, len(df)), random_state=int(CONFIG["reproducibility"]["seed"]))
    sample = df.iloc[rng.index][[
        "video_id", "title", "rating", "views", "publish_date",
        CONFIG["project_specifics"]["intersection"]["output_col_name"]
    ]].copy()
    # format types
    sample["rating"] = sample["rating"].astype(float).round(1)
    sample["views"] = sample["views"].astype(float).round(0).astype(int)
    sample["year"] = pd.to_datetime(sample["publish_date"], errors="coerce").dt.year.astype("Int64")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    print(f"✓ Self-check saved: {out_path.resolve()}")
    print(f"[TIME] write_selfcheck: {time.perf_counter() - t0:.2f}s")

# --- 6. Main pipeline --------------------------------------------------------

def main() -> None:
    """
    Build and save the canonical ML corpus (always loads the real DB).
    """
    t0_total = time.perf_counter()
    print("--- Starting Step 01: Corpus Builder ---")

    conn = create_connection()
    if conn is None:
        print("✗ Halting execution due to database connection failure.")
        return

    df = fetch_data_from_db(conn)
    conn.close()
    if df.empty:
        print("✗ No rows returned from DB; aborting.")
        return

    print("Cleaning and combining text fields...")
    df['tags'] = df['tags'].fillna('')
    df['categories'] = df['categories'].fillna('')
    df['combined_text'] = df['title'].fillna('') + ' ' + df['tags'] + ' ' + df['categories']
    df['combined_text_clean'] = df['combined_text'].apply(clean_text)
    df['model_input_text'] = (df['title'].fillna('') + ' ' + df['tags']).apply(clean_text)

    # protected + intersectional
    df = create_protected_group_features(df, 'combined_text_clean')
    df = create_intersectional_features(df)

    # --- quality flags (added 2026-05-20) ---
    # is_animated: 1 if the video has animated/non-human tags AND no real-person tags.
    # Purely animated content should be excluded from human-performer fairness analysis.
    _anim_pat = r'\b(hentai|cartoon|anime|3d|animated|animation|toon|cgi)\b'
    _real_pat  = r'\b(amateur|milf|teen|pornstar|lesbian|black|ebony|asian|caucasian|latina|interracial|gay|bisexual|trans)\b'
    _tags_lower = df['tags'].str.lower().fillna('')
    has_anim = _tags_lower.str.contains(_anim_pat, regex=True, na=False)
    has_real = _tags_lower.str.contains(_real_pat, regex=True, na=False)
    df['is_animated'] = (has_anim & ~has_real).astype(int)
    n_anim = int(df['is_animated'].sum())
    print(f"✓ is_animated flag: {n_anim:,} purely animated videos flagged ({n_anim/len(df)*100:.2f}%)")

    # rating_clean: NaN for incoherent records where ratings_count>0 but score=0
    # (API artefact — votes were recorded but score was not returned correctly).
    # All downstream rating analyses should use rating_clean, not raw rating.
    import numpy as np
    df['rating_clean'] = df['rating'].copy().astype(float)
    incoherent_mask = (df['ratings'] > 0) & (df['rating'] == 0)
    df.loc[incoherent_mask, 'rating_clean'] = np.nan
    n_incoherent = int(incoherent_mask.sum())
    print(f"✓ rating_clean: {n_incoherent:,} incoherent records set to NaN (ratings_count>0 but score=0)")

    # is_duplicate: same (title_lower, duration) pair — re-upload detection.
    # First occurrence is kept (is_duplicate=0); subsequent are flagged (is_duplicate=1).
    # Represents platform re-upload behaviour; used in sensitivity analysis, not filtered.
    dup_key = df['title'].fillna('').str.lower().str.strip() + '__' + df['duration'].astype(str)
    df['is_duplicate'] = dup_key.duplicated(keep='first').astype(int)
    n_dup = int(df['is_duplicate'].sum())
    print(f"✓ is_duplicate flag: {n_dup:,} re-upload duplicates flagged ({n_dup/len(df)*100:.2f}%)")

    # quick real-number sanity prints (no DB upload, just stdout)
    quick_sanity_report(df)

    print("Saving artefacts...")
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CORPUS_PATH, index=False)
    print(f"✓ Artefact saved: {CORPUS_PATH.resolve()}")

    stats = {
        'num_records': int(len(df)),
        'num_columns': int(len(df.columns)),
        'columns': list(map(str, df.columns)),
        'protected_group_counts': {
            col: int(df[col].sum()) for col in df.columns
            if any(k in col for k in ['race_', 'gender_', 'intersectional_'])
        }
    }
    with open(STATS_PATH, 'w', encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"✓ Artefact saved: {STATS_PATH.resolve()}")
    
    write_selfcheck(df, SELF_PATH, n=10)
    dt = time.perf_counter() - t0_total
    print(f"[TIME] Step 01 runtime: {dt:.2f}s")
    print("--- Step 01: Corpus Builder Completed Successfully ---")


if __name__ == '__main__':
    main()
