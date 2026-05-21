# -*- coding: utf-8 -*-
"""
02_comprehensive_eda.py
=======================

Purpose
-------
Exploratory Data Analysis (EDA) on the canonical corpus (Step 01 parquet only).
Quantifies representation, intersectionality, engagement, and outcome disparities
with publication-ready figures/tables. Intersectionality-first, non-overlapping.

What it does
------------
- Representation: categories, tags, race×gender, race×gender×orientation
- Language-stratified intersectionality
- Uploader concentration (HHI, Top-10 share) by intersection
- Rating-count reliability by intersection
- Label cardinality (tags/categories per video) by intersection
- Title/token quality by intersection
- Outlier burden (99th pct) by intersection
- Seasonality & recency (month-of-year; age-in-days) by intersection
- Quality proxies prevalence (hd/4k/verified) by intersection
- Views-per-day normalization snapshot by intersection
- Diversity: Shannon entropy of categories/tags by intersection
- Orientation coverage by intersection (descriptive; no PMI)
- “Age tag” (“18-25”) prevalence by intersection
- Language × quality proxies heatmap
- Pre-split risk: duplicates & uploader overlap matrix (leakage heads-up)

Important notes
---------------
- Titles may be non-English; tags/categories help anchor semantics.
- Categories/tags are multi-label, so totals across labels can exceed N.
- Years are integers; months are integers; ratings use 1 decimal place; percentages
use 1 decimal place; entropies use 2 decimals; other integers are rounded.

CLI
---
# Full run (canonical artefacts under outputs/data & outputs/figures/eda)
python3 src/analysis/02_comprehensive_eda.py

# Self-check (random sample; writes *_selfcheck artefacts only)
python3 src/analysis/02_comprehensive_eda.py --selfcheck --sample 80000
"""

from __future__ import annotations

# --- Imports (keep at top) ----------------------------------------------------
import sys
import re
import json
import time
import math
import argparse
import itertools
from pathlib import Path
from typing import Tuple, List, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- 1) Config & paths --------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.utils.academic_tables import dataframe_to_latex_table

CONFIG = load_config()
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))

DATA_DIR       = Path(CONFIG['paths']['data'])
FIGURES_DIR    = Path(CONFIG['paths']['figures']) / 'eda'
TABLES_DIR     = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'
NARRATIVE_PATH = Path(CONFIG['paths']['narratives']) / 'automated' / '02_comprehensive_eda_summary.md'

CORPUS_PATH    = DATA_DIR / '01_ml_corpus.parquet'
LEXICON_PATH   = Path(CONFIG['paths']['config']) / 'protected_terms.json'

# --- 2) Small helpers ---------------------------------------------------------
def _timer(msg: str) -> float:
    """Start a perf counter and print a standardized header."""
    t0 = time.perf_counter(); print(msg); return t0

def _finish(label: str, t0: float) -> None:
    """Print a standardized [TIME] line with elapsed seconds."""
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")

def _explode_csv_col(series: pd.Series) -> pd.Series:
    """
    Cleanly explode a comma-separated Series to a 1-D lowercase string Series.
    Empty values are removed.
    """
    s = series.fillna('').astype(str).str.lower().str.split(',')
    s = s.explode().str.strip()
    return s[s != '']

def _load_lexicon() -> dict:
    """
    Load protected/bias lexicon JSON once for tag matching. Returns {} on failure.
    """
    t0 = _timer(f"Loading lexicon: {LEXICON_PATH}")
    try:
        with open(LEXICON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _finish("eda.load_lexicon", t0); return data
    except Exception as e:
        print(f"[WARN] Could not load lexicon at {LEXICON_PATH}: {e}")
        _finish("eda.load_lexicon", t0); return {}

def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name from candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _group_labels_intersectional(df: pd.DataFrame) -> pd.Series:
    """
    Non-overlapping 4-group female intersection labels:
      Black Women / White Women / Asian Women / Latina Women
    Everyone else → drop (handled by masking when grouping).
    """
    gf = (df.get('gender_female', 0) == 1)
    bw = (df.get('race_ethnicity_black', 0) == 1) & gf
    ww = (df.get('race_ethnicity_white', 0) == 1) & gf & ~bw
    aw = (df.get('race_ethnicity_asian', 0) == 1) & gf & ~bw & ~ww
    lw = (df.get('race_ethnicity_latina', 0) == 1) & gf & ~bw & ~ww & ~aw
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out[bw] = "Black Women"; out[ww] = "White Women"; out[aw] = "Asian Women"; out[lw] = "Latina Women"
    return out

def _orientation_from_text(df: pd.DataFrame) -> pd.Series:
    """
    Derive a single orientation label per video from categories/tags (priority):
      Lesbian > Gay > Bisexual > Other/Unspecified
    Exact token matches on comma-separated categories/tags, case-insensitive.
    """
    cats = df.get('categories', '').fillna('').astype(str).str.lower()
    tags = df.get('tags', '').fillna('').astype(str).str.lower()
    def has_token(s: pd.Series, token: str) -> pd.Series:
        pat = rf"(?:^|,)\s*{re.escape(token)}\s*(?:,|$)"; return s.str.contains(pat, regex=True)
    lesbian = has_token(cats, "lesbian") | has_token(tags, "lesbian")
    gay     = has_token(cats, "gay") | has_token(tags, "gay")
    bi      = has_token(cats, "bisexual") | has_token(tags, "bisexual") | has_token(tags, "bi")
    orient = np.where(lesbian, "Lesbian",
              np.where(gay, "Gay",
              np.where(bi, "Bisexual", "Other/Unspecified")))
    return pd.Series(orient, index=df.index, dtype="object", name="Orientation")

def _ensure_minimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure minimal columns exist with sensible types/defaults:
    - video_id (fallback index)
    - categories, tags
    - rating (float, 1 d.p. for prints), ratings (int)
    - views (int), duration (int seconds)
    - publish_date (datetime if present)
    - title, language, uploader if present
    """
    d = df.copy()
    if 'video_id' not in d.columns: d['video_id'] = np.arange(len(d))
    for c in ['categories','tags','title']:
        if c not in d.columns: d[c] = ''
        d[c] = d[c].fillna('').astype(str)
    for c in ['rating','views','ratings','duration']:
        if c not in d.columns: d[c] = 0
    d['rating']  = pd.to_numeric(d['rating'], errors='coerce').fillna(0.0).astype(float)
    d['views']   = pd.to_numeric(d['views'], errors='coerce').fillna(0).astype(int)
    d['ratings'] = pd.to_numeric(d['ratings'], errors='coerce').fillna(0).astype(int)
    d['duration']= pd.to_numeric(d['duration'], errors='coerce').fillna(0).astype(int)
    # publish date (robust)
    pub_col = _first_existing_column(d, ['publish_date','published_at','upload_date','date'])
    if pub_col:
        d[pub_col] = pd.to_datetime(d[pub_col], errors='coerce', utc=True)
    d['_publish_col'] = pub_col  # remember which one we used (or None)
    # language/uploader if available
    if 'language' not in d.columns: d['language'] = 'unknown'
    up_col = _first_existing_column(d, ['uploader','uploader_id','channel','author','creator'])
    d['_uploader_col'] = up_col
    return d

def _safe_percent(n: int, d: int) -> float:
    return (100.0 * n / d) if max(d, 1) > 0 else 0.0

def _entropy_from_counts(counts: pd.Series) -> float:
    """Shannon entropy in bits (base 2)."""
    p = counts[counts > 0].astype(float); p = p / p.sum()
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

# --- 3) Tag index (FAST) ------------------------------------------------------
def build_tag_index(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Build a fast video-count index per normalized tag and a tidy tag frame.

    Returns
    -------
    (tag_index, tag_df)
      tag_index: pd.Series tag_norm -> #unique videos with tag
      tag_df   : pd.DataFrame columns ['video_id','tag_norm'] unique per pair

    Notes
    -----
    - One pass explosion; dedupe by (video_id, tag_norm).
    - Normalization: lower, hyphen/underscore→space, collapse spaces.
    - Complexity ~ O(total tags).
    """
    t0 = time.perf_counter()
    tags_split = df['tags'].fillna('').str.lower().str.split(',')
    total_tokens = int(tags_split.map(len).sum())
    if total_tokens == 0:
        print("[INFO] No tags present in corpus.")
        return pd.Series(dtype='int64'), pd.DataFrame(columns=['video_id','tag_norm'])
    lens   = tags_split.map(len).to_numpy()
    vid_id = np.repeat(df['video_id'].to_numpy(), lens)
    flat   = np.concatenate(tags_split.to_numpy())
    tags   = pd.Series(flat, dtype='object').str.strip()
    mask   = tags != ''
    tag_df = pd.DataFrame({'video_id': vid_id[mask], 'tag': tags[mask]})
    tag_df['tag_norm'] = (tag_df['tag']
                          .str.replace(r'[-_]+', ' ', regex=True)
                          .str.replace(r'\s+', ' ', regex=True)
                          .str.strip())
    tag_df = tag_df.drop_duplicates(['video_id', 'tag_norm'])
    tag_index = (tag_df.groupby('tag_norm')['video_id']
                 .nunique()
                 .sort_values(ascending=False))
    print(f"[TIME] eda.build_tag_index: {time.perf_counter() - t0:.2f}s "
          f"(unique tags: {tag_index.size:,}; tokens: {total_tokens:,})")
    return tag_index, tag_df

# --- 4) Core analyses (original) ---------------------------------------------
def analyze_category_distribution(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top-N categories and share of corpus."""
    t0 = _timer("Analyzing top video category distribution...")
    cats = _explode_csv_col(df['categories'])
    top = cats.value_counts().nlargest(top_n).reset_index()
    top.columns = ['Category', 'Video Count']
    top['Percentage'] = (top['Video Count'] / max(len(df), 1)) * 100
    _finish("eda.analyze_category_distribution", t0); return top

def analyze_top_tags(df: pd.DataFrame, tag_index: pd.Series, top_n: int = 25) -> pd.DataFrame:
    """Fast top-N tags using a precomputed video-count index."""
    t0 = time.perf_counter(); print("Analyzing top tags (overall)...")
    top = (tag_index.head(top_n).rename_axis('Tag').reset_index(name='Videos With Tag'))
    top['Percentage'] = (top['Videos With Tag'] / max(len(df), 1)) * 100
    print(f"[TIME] eda.analyze_top_tags: {time.perf_counter() - t0:.2f}s"); return top

def analyze_top_protected_tags(
    df: pd.DataFrame, tag_index: pd.Series, tag_df: pd.DataFrame, lex: dict, top_n: int = 25
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Top-N protected/bias tags and a per-video boolean mask 'has_protected'.
    Returns (prot_table, has_protected).
    """
    t0 = time.perf_counter(); print("Analyzing top protected/bias tags...")
    protected_vocab = set()
    st = lex.get('stereotype_terms', {})
    for vals in st.values():
        if isinstance(vals, list):
            protected_vocab.update([t for t in vals if isinstance(t, str)])
    bm = lex.get('bias_markers', {})
    for obj in bm.values():
        if isinstance(obj, dict):
            protected_vocab.update([t for t in obj.get('terms', []) if isinstance(t, str)])
    if not protected_vocab:
        print("[WARN] No protected/bias vocabulary found in lexicon.")
        return (pd.DataFrame(columns=['Protected Tag','Videos With Tag','Percentage']),
                pd.Series(False, index=df.index))
    def _norm(s: str) -> str:
        s = s.lower(); s = re.sub(r'[-_]+', ' ', s); s = re.sub(r'\s+', ' ', s).strip(); return s
    vocab_norm = {_norm(t): t for t in protected_vocab}
    counts = {orig: int(tag_index.get(norm, 0)) for norm, orig in vocab_norm.items()}
    ser = pd.Series(counts, dtype='int64'); ser = ser[ser > 0].sort_values(ascending=False).head(top_n)
    out = ser.rename_axis('Protected Tag').reset_index(name='Videos With Tag')
    out['Percentage'] = (out['Videos With Tag'] / max(len(df), 1)) * 100
    prot_norm_set = set(vocab_norm.keys())
    vids_with_prot = tag_df.loc[tag_df['tag_norm'].isin(prot_norm_set), 'video_id'].unique()
    has_protected = pd.Series(df['video_id'].isin(vids_with_prot), index=df.index)
    print(f"[TIME] eda.analyze_top_protected_tags: {time.perf_counter() - t0:.2f}s"); return out, has_protected

def analyze_full_intersections(df: pd.DataFrame) -> pd.DataFrame:
    """Prevalence of all race × gender intersections in the corpus."""
    t0 = _timer("Analyzing all race × gender intersections...")
    race_cols = sorted([c for c in df.columns if c.startswith('race_ethnicity_')])
    gender_cols = sorted([c for c in df.columns if c.startswith('gender_')])
    N = max(len(df), 1)
    rows = []
    for r, g in itertools.product(race_cols, gender_cols):
        inter_name = f"{r.split('_')[-1].capitalize()} x {g.split('_')[-1].capitalize()}"
        mask = (df.get(r, 0) == 1) & (df.get(g, 0) == 1)
        n = int(mask.sum())
        if n > 0:
            rows.append({'Intersection': inter_name, 'Count': n, 'Percentage': (n / N) * 100})
    out = pd.DataFrame(rows).sort_values('Count', ascending=False).reset_index(drop=True)
    _finish("eda.analyze_full_intersections", t0); return out

def analyze_rating_disparities(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare rating distributions across key female intersections."""
    t0 = _timer("Analyzing rating disparities across key female intersections...")
    gl = _group_labels_intersectional(df)
    dfp = df.loc[gl.notna(), ['rating']].copy(); dfp['Group'] = gl.dropna().values
    _finish("eda.analyze_rating_disparities", t0)
    return dfp, (dfp.groupby('Group')['rating'].describe() if not dfp.empty else pd.DataFrame())

def analyze_views_disparities(df: pd.DataFrame, log10: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare (log) views across key female intersections."""
    t0 = _timer(f"Analyzing views disparities (log10={log10}) across key female intersections...")
    gl = _group_labels_intersectional(df)
    col_raw = 'views'; col_out = 'views_log10' if log10 else 'views'
    dfp = df.loc[gl.notna(), [col_raw]].copy()
    if dfp.empty:
        _finish("eda.analyze_views_disparities", t0); return pd.DataFrame(columns=[col_out,'Group']), pd.DataFrame()
    dfp[col_out] = np.log10(dfp[col_raw].clip(lower=0).astype(float) + 1.0) if log10 else dfp[col_raw].astype(float)
    dfp['Group'] = gl.dropna().values
    _finish("eda.analyze_views_disparities", t0)
    return dfp[[col_out,'Group']], (dfp.groupby('Group')[col_out].describe() if not dfp.empty else pd.DataFrame())

def protected_rating_medians(df: pd.DataFrame, has_protected: pd.Series) -> pd.DataFrame:
    """Median rating with/without protected-term tags overall and by intersection."""
    t0 = _timer("Computing median ratings conditional on protected tags (overall + intersections)...")
    gl = _group_labels_intersectional(df)
    rows = []
    def med_pair(mask: pd.Series) -> Dict[str, float]:
        hp = has_protected[mask.fillna(False)]
        r = df.loc[mask.fillna(False), 'rating']
        r_prot = float(r.loc[hp].median()) if hp.any() else np.nan
        r_nop  = float(r.loc[~hp].median()) if (~hp).any() else np.nan
        return {'Median Rating (Protected Tags)': r_prot, 'Median Rating (No Protected Tags)': r_nop,
                'N Protected': int(hp.sum()), 'N No-Protected': int((~hp).sum())}
    rows.append({'Group': 'Overall', **med_pair(pd.Series(True, index=df.index))})
    for g in ['Black Women','White Women','Asian Women','Latina Women']:
        rows.append({'Group': g, **med_pair(gl == g)})
    out = pd.DataFrame(rows).set_index('Group').round(3)
    _finish("eda.protected_rating_medians", t0); return out

# --- 5) New analyses (enhancements) ------------------------------------------
def triple_intersection_rgo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Triple intersection: race × gender × orientation.
    Orientation derived from categories/tags with priority: Lesbian > Gay > Bisexual > Other/Unspecified.
    """
    t0 = _timer("Computing triple-intersection coverage (race × gender × orientation)...")
    orient = _orientation_from_text(df)
    race_cols = sorted([c for c in df.columns if c.startswith('race_ethnicity_')])
    gender_cols = sorted([c for c in df.columns if c.startswith('gender_')])
    rows = []
    N = len(df)
    for r, g in itertools.product(race_cols, gender_cols):
        base_mask = (df.get(r, 0) == 1) & (df.get(g, 0) == 1)
        if base_mask.sum() == 0: continue
        for o in ['Lesbian','Gay','Bisexual','Other/Unspecified']:
            m = base_mask & (orient == o)
            n = int(m.sum())
            if n > 0:
                rows.append({
                    'Race': r.replace('race_ethnicity_', '').capitalize(),
                    'Gender': g.replace('gender_', '').capitalize(),
                    'Orientation': o,
                    'Count': n,
                    'Percentage': _safe_percent(n, N)
                })
    out = pd.DataFrame(rows).sort_values('Count', ascending=False).reset_index(drop=True)
    _finish("eda.triple_intersection_rgo", t0)
    return out

def language_by_intersection(df: pd.DataFrame, top_k_lang: int = 10) -> pd.DataFrame:
    """
    Language × (race×gender) share and outcomes (median rating 1 d.p., median views_per_day).
    Assumes df['language'] if available; otherwise 'unknown' and prints a warning.
    views_per_day computed against a fixed anchor (max publish date) for reproducibility.
    """
    t0 = _timer("Computing language-stratified intersectionality (top languages)...")
    gl = _group_labels_intersectional(df)
    if 'language' not in df.columns or df['language'].nunique() <= 1:
        print("[WARN] 'language' column missing or single-valued; using 'unknown'.")
    lang = df.get('language', pd.Series('unknown', index=df.index)).fillna('unknown').astype(str)
    pub_col = df.get('_publish_col', None)
    if isinstance(pub_col, pd.Series):  # from _ensure_minimal_columns
        pub_col = pub_col.iloc[0] if (pub_col.notna().any()) else None
    if pub_col:
        anchor = df[pub_col].max()
        days = (anchor - df[pub_col]).dt.days.clip(lower=1)
    else:
        days = pd.Series(1, index=df.index)
    vpd = df['views'].astype(float) / days.astype(float)
    base = pd.DataFrame({'Group': gl, 'language': lang, 'rating': df['rating'].astype(float), 'vpd': vpd})
    base = base.dropna(subset=['Group'])
    top_langs = (base['language'].value_counts().head(top_k_lang).index.tolist())
    sub = base[base['language'].isin(top_langs)]
    agg = (sub.groupby(['language','Group'])
             .agg(N=('rating','size'),
                  Share=('rating', lambda s: _safe_percent(len(s), len(sub))),
                  MedianRating=('rating', lambda s: round(float(s.median()), 1)),
                  MedianViewsPerDay=('vpd', lambda s: int(round(float(s.median())))))
             .reset_index())
    _finish("eda.language_by_intersection", t0)
    return agg

def uploader_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uploader concentration per intersection: N uploaders, HHI (3 d.p.), Top-10 share (%).
    Requires an uploader-like column; if missing, returns empty with a warning.
    """
    t0 = _timer("Computing uploader concentration by intersection...")
    gl = _group_labels_intersectional(df)
    up_col = df.get('_uploader_col', None)
    if isinstance(up_col, pd.Series): up_col = up_col.iloc[0]
    if not up_col:
        print("[WARN] No uploader column found; skipping concentration analysis.")
        _finish("eda.uploader_concentration", t0)
        return pd.DataFrame(columns=['Group','N Uploaders','HHI','Top10 Share %'])
    d = df.loc[gl.notna(), [up_col]].copy(); d['Group'] = gl.dropna().values
    rows = []
    for g, sub in d.groupby('Group'):
        vc = sub[up_col].fillna('unknown').astype(str).value_counts()
        n_up = int(vc.size)
        shares = (vc / vc.sum()).to_numpy()
        hhi = float((shares ** 2).sum())
        top10 = float(100.0 * (vc.head(10).sum() / max(vc.sum(), 1)))
        rows.append({'Group': g, 'N Uploaders': n_up, 'HHI': round(hhi, 3), 'Top10 Share %': round(top10, 1)})
    out = pd.DataFrame(rows).sort_values('HHI', ascending=False)
    _finish("eda.uploader_concentration", t0)
    return out

def rating_count_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Rating-count distribution per group: median & IQR."""
    t0 = _timer("Summarising rating-count reliability by intersection...")
    gl = _group_labels_intersectional(df)
    d = df.loc[gl.notna(), ['ratings']].copy(); d['Group'] = gl.dropna().values
    def iqr(s): q1, q3 = np.percentile(s, 25), np.percentile(s, 75); return int(round(q3 - q1))
    out = (d.groupby('Group')['ratings']
             .agg(Median='median', IQR=iqr)
             .astype(int)
             .reset_index())
    _finish("eda.rating_count_by_group", t0); return out

def label_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """Median #tags and #categories per video by intersection."""
    t0 = _timer("Computing label cardinality by intersection...")
    gl = _group_labels_intersectional(df)
    def _count_tokens(s):
        return s.fillna('').astype(str).str.strip().str.split(',').map(lambda x: 0 if x == [''] else len(x))
    tags_n = _count_tokens(df['tags']); cats_n = _count_tokens(df['categories'])
    d = pd.DataFrame({'Group': gl, '#Tags': tags_n, '#Categories': cats_n}).dropna(subset=['Group'])
    out = (d.groupby('Group')[['#Tags','#Categories']].median().round(1).reset_index())
    _finish("eda.label_cardinality", t0); return out

def title_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Title length (tokens), non-ASCII share, digits share, missing rate by group."""
    t0 = _timer("Assessing title/token quality by intersection...")
    gl = _group_labels_intersectional(df)
    t = df['title'].fillna('').astype(str)
    tokens = t.str.split().map(len)
    non_ascii_share = t.map(lambda s: 0.0 if len(s) == 0 else sum(ord(ch) > 127 for ch in s) / len(s))
    digits_share = t.map(lambda s: 0.0 if len(s) == 0 else sum(ch.isdigit() for ch in s) / len(s))
    missing = t.eq('')
    d = pd.DataFrame({
        'Group': gl, 'TitleTokens': tokens, 'NonASCIIShare': non_ascii_share,
        'DigitsShare': digits_share, 'MissingTitle': missing
    }).dropna(subset=['Group'])
    out = (d.groupby('Group')
             .agg(MedianTokens=('TitleTokens', lambda s: int(round(float(s.median())))),
                  NonASCIIShare=('NonASCIIShare', lambda s: round(float(s.mean())*100, 1)),
                  DigitsShare=('DigitsShare', lambda s: round(float(s.mean())*100, 1)),
                  MissingTitle=('MissingTitle', lambda s: round(float(s.mean())*100, 1)))
             .reset_index())
    _finish("eda.title_quality", t0); return out

def outlier_burden(df: pd.DataFrame) -> pd.DataFrame:
    """
    Share (%) of videos above global 99th pct for duration, views, ratings per intersection.
    """
    t0 = _timer("Computing outlier burden (global 99th pct) by intersection...")
    gl = _group_labels_intersectional(df)
    thr = {
        'duration': float(pd.Series(df['duration'].astype(float)).quantile(0.99)),
        'views':    float(pd.Series(df['views'].astype(float)).quantile(0.99)),
        'ratings':  float(pd.Series(df['ratings'].astype(float)).quantile(0.99)),
    }
    d = df.loc[gl.notna(), ['duration','views','ratings']].copy(); d['Group'] = gl.dropna().values
    rows = []
    for g, sub in d.groupby('Group'):
        rows.append({
            'Group': g,
            'Above99% duration %': round(100.0 * (sub['duration'] > thr['duration']).mean(), 1),
            'Above99% views %':    round(100.0 * (sub['views'] > thr['views']).mean(), 1),
            'Above99% ratings %':  round(100.0 * (sub['ratings'] > thr['ratings']).mean(), 1),
        })
    out = pd.DataFrame(rows)
    _finish("eda.outlier_burden", t0); return out

def seasonality_and_recency(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Month-of-year distribution by group + age-in-days per group.
    Anchor for age: max publish date in corpus (reproducible).
    """
    t0 = _timer("Summarising seasonality & recency (month-of-year; age-in-days)...")
    gl = _group_labels_intersectional(df)
    pub_col = df.get('_publish_col', None)
    if isinstance(pub_col, pd.Series): pub_col = pub_col.iloc[0] if (pub_col.notna().any()) else None
    if not pub_col:
        print("[WARN] No publish date column; skipping seasonality/recency.")
        _finish("eda.seasonality_and_recency", t0)
        return pd.DataFrame(columns=['Group','Month','N','Share %']), pd.DataFrame(columns=['Group','Median Age (days)'])
    dtime = df[pub_col].copy()
    month = dtime.dt.month.astype('Int64')
    anchor = dtime.max()
    age_days = (anchor - dtime).dt.days.astype('Int64')
    base = pd.DataFrame({'Group': gl, 'Month': month, 'AgeDays': age_days}).dropna(subset=['Group','Month'])
    by_month = (base.groupby(['Group','Month'])
                    .size().rename('N').reset_index())
    by_month['Share %'] = by_month.groupby('Group')['N'].transform(lambda s: (s / s.sum()) * 100).round(1)
    age = (base.groupby('Group')['AgeDays']
              .median().astype(int).rename('Median Age (days)').reset_index())
    _finish("eda.seasonality_and_recency", t0)
    return by_month, age

def quality_proxies_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage of quality proxies (hd, 4k, verified amateurs) by intersection."""
    t0 = _timer("Measuring quality proxies prevalence by intersection...")
    gl = _group_labels_intersectional(df)
    tags = df['tags'].fillna('').astype(str).str.lower()
    def has_token(token: str) -> pd.Series:
        pat = rf"(?:^|,)\s*{re.escape(token)}\s*(?:,|$)"; return tags.str.contains(pat, regex=True)
    proxies = pd.DataFrame({
        'hd': has_token('hd'),
        '4k': has_token('4k'),
        'verified': has_token('verified amateurs') | has_token('verified')
    })
    base = pd.concat([gl.rename('Group'), proxies], axis=1).dropna(subset=['Group'])
    out = (base.groupby('Group')[['hd','4k','verified']].mean().mul(100).round(1).reset_index()
           .rename(columns={'hd':'HD %','4k':'4K %','verified':'Verified %'}))
    _finish("eda.quality_proxies_by_group", t0); return out

def views_per_day_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Views_per_day distribution summary by group (raw median & log10 boxplot-ready)."""
    t0 = _timer("Computing views_per_day snapshot by intersection...")
    gl = _group_labels_intersectional(df)
    pub_col = df.get('_publish_col', None)
    if isinstance(pub_col, pd.Series): pub_col = pub_col.iloc[0] if (pub_col.notna().any()) else None
    if pub_col:
        anchor = df[pub_col].max()
        days = (anchor - df[pub_col]).dt.days.clip(lower=1)
    else:
        days = pd.Series(1, index=df.index)
    vpd = df['views'].astype(float) / days.astype(float)
    base = pd.DataFrame({'Group': gl, 'vpd': vpd}).dropna(subset=['Group'])
    summary = (base.groupby('Group')['vpd']
                 .agg(Median=lambda s: int(round(float(s.median()))))
                 .reset_index())
    _finish("eda.views_per_day_snapshot", t0)
    return summary, base  # base for plotting

def entropy_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Shannon entropy of categories and tags within each intersection group."""
    t0 = _timer("Computing category/tag entropy by intersection...")
    gl = _group_labels_intersectional(df)
    rows = []
    for g in ['Black Women','White Women','Asian Women','Latina Women']:
        mask = gl == g
        if mask.sum() == 0: continue
        cats = _explode_csv_col(df.loc[mask, 'categories']); tags = _explode_csv_col(df.loc[mask, 'tags'])
        ent_c = round(_entropy_from_counts(cats.value_counts()), 2)
        ent_t = round(_entropy_from_counts(tags.value_counts()), 2)
        rows.append({'Group': g, 'Entropy (Categories)': ent_c, 'Entropy (Tags)': ent_t})
    out = pd.DataFrame(rows)
    _finish("eda.entropy_by_group", t0); return out

def orientation_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Orientation label counts within each race×gender group (descriptive only)."""
    t0 = _timer("Summarising orientation coverage by intersection...")
    gl = _group_labels_intersectional(df); orient = _orientation_from_text(df)
    base = pd.DataFrame({'Group': gl, 'Orientation': orient})
    base = base.dropna(subset=['Group'])
    out = (base.groupby(['Group','Orientation']).size().rename('Count').reset_index())
    _finish("eda.orientation_by_group", t0); return out

def age_tag_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Prevalence of the '18-25' age tag by intersection (descriptive)."""
    t0 = _timer("Measuring '18-25' age tag prevalence by intersection...")
    gl = _group_labels_intersectional(df)
    tags = df['tags'].fillna('').astype(str).str.lower()
    age_tag = tags.str.contains(r"(?:^|,)\s*18-25\s*(?:,|$)", regex=True)
    base = pd.DataFrame({'Group': gl, 'AgeTag': age_tag}).dropna(subset=['Group'])
    out = (base.groupby('Group')['AgeTag'].mean().mul(100).round(1).rename("18-25 %").reset_index())
    _finish("eda.age_tag_by_group", t0); return out

def quality_by_language(df: pd.DataFrame, top_k_lang: int = 10) -> pd.DataFrame:
    """
    Heatmap-ready table for quality proxies coverage across top languages.
    Returns columns: language, HD %, 4K %, Verified % (percentages rounded 1 d.p.).
    """
    t0 = _timer("Computing language × quality proxies coverage...")
    lang = df.get('language', pd.Series('unknown', index=df.index)).fillna('unknown').astype(str)
    tags = df['tags'].fillna('').astype(str).str.lower()

    def has_token(token: str) -> pd.Series:
        """Match exact comma-separated token (case-insensitive) in tags."""
        pat = rf"(?:^|,)\s*{re.escape(token)}\s*(?:,|$)"
        return tags.str.contains(pat, regex=True)

    proxies = pd.DataFrame({
        'hd': has_token('hd'),
        '4k': has_token('4k'),
        'verified': has_token('verified amateurs') | has_token('verified')
    })

    base = pd.concat([lang.rename('language'), proxies], axis=1)
    top_langs = base['language'].value_counts().head(top_k_lang).index.tolist()
    sub = base[base['language'].isin(top_langs)]

    out = (
        sub.groupby('language')[['hd', '4k', 'verified']]
           .mean()
           .mul(100)
           .round(1)
           .rename(columns={'hd': 'HD %', '4k': '4K %', 'verified': 'Verified %'})
           .reset_index()
    )
    _finish("eda.quality_by_language", t0)
    return out


def duplication_and_overlap(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pre-split leakage checks:
      - Duplication summary by (title,duration) exact and by normalized title only.
      - Uploader overlap matrix across intersection groups (count of shared uploaders).
    """
    t0 = _timer("Running duplication & uploader-overlap pre-checks...")
    gl = _group_labels_intersectional(df)
    title_norm = df['title'].fillna('').str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    # Duplicates
    dup_exact = df.duplicated(subset=['title','duration'], keep=False).sum()
    dup_title = title_norm.duplicated(keep=False).sum()
    dup_df = pd.DataFrame({
        'Type': ['Exact(title,duration)','Title-only'],
        'Duplicate Rows': [int(dup_exact), int(dup_title)]
    })
    # Uploader overlap matrix
    up_col = df.get('_uploader_col', None)
    if isinstance(up_col, pd.Series): up_col = up_col.iloc[0]
    groups = ['Black Women','White Women','Asian Women','Latina Women']
    if not up_col:
        print("[WARN] No uploader column; skipping overlap matrix.")
        overlap = pd.DataFrame(0, index=groups, columns=groups)
    else:
        buckets = {g: set(df.loc[gl == g, up_col].astype(str)) for g in groups}
        idx = groups; cols = groups
        mat = np.zeros((len(idx), len(cols)), dtype=int)
        for i, gi in enumerate(idx):
            for j, gj in enumerate(cols):
                mat[i, j] = len(buckets[gi].intersection(buckets[gj]))
        overlap = pd.DataFrame(mat, index=idx, columns=cols)
    _finish("eda.duplication_and_overlap", t0)
    return dup_df, overlap

# --- 6) Visualizations (dual theme) ------------------------------------------
def _smart_palette(palette: list, n: int) -> list:
    if palette is None: return None
    return (sns.color_palette("plasma", n_colors=n) if n > len(palette) else palette[:n])

@plot_dual_theme(section='eda')
def plot_categories(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Top categories with on-bar labels + explainer box."""
    active = _smart_palette(palette, len(data['Category'].unique()))
    sns.barplot(x='Video Count', y='Category', hue='Category', data=data, palette=active, legend=False, ax=ax)
    ax.set_title(f'Top {len(data)} Most Frequent Video Categories'); ax.set_xlabel('Total Video Count'); ax.set_ylabel(None)
    for patch, (_, row) in zip(ax.patches, data.iterrows()):
        count = int(row['Video Count']); pct = float(row['Percentage'])
        ax.text(patch.get_width(), patch.get_y()+patch.get_height()/2,
                f"{count} ({pct:.1f}%)", va='center', ha='left', fontsize=10)
    ax.margins(x=0.02, y=0.01)

@plot_dual_theme(section='eda')
def plot_intersections(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Intersectional representation with % labels + explainer box."""
    active = _smart_palette(palette, len(data['Intersection'].unique()))
    sns.barplot(x='Percentage', y='Intersection', hue='Intersection', data=data, palette=active, legend=False, ax=ax)
    ax.set_title('Representation of Race × Gender Intersections'); ax.set_xlabel('Share of Corpus'); ax.set_ylabel(None)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))

@plot_dual_theme(section='eda')
def plot_ratings(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Boxplot of ratings with N in labels, median markers, explainer box."""
    counts = data.groupby('Group')['rating'].size().to_dict()
    dd = data.copy(); dd['Group_N'] = dd['Group'].map(lambda g: f"{g} (N={counts.get(g,0)})")
    active = _smart_palette(palette, dd['Group_N'].nunique())
    sns.boxplot(x='rating', y='Group_N', hue='Group_N', data=dd, palette=active, ax=ax, orient='h')
    # hide the redundant legend (y==hue)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    ax.set_title('Comparison of Video Rating Distributions by Group'); ax.set_xlabel('Video Rating'); ax.set_ylabel(None)

@plot_dual_theme(section='eda')
def plot_views(data: pd.DataFrame, value_col: str, ax=None, palette=None, **kwargs):
    """Boxplot of (log) views with N in labels, medians, and log explainer."""
    counts = data.groupby('Group')[value_col].size().to_dict()
    dd = data.copy(); dd['Group_N'] = dd['Group'].map(lambda g: f"{g} (N={counts.get(g,0)})")
    active = _smart_palette(palette, dd['Group_N'].nunique())
    sns.boxplot(x=value_col, y='Group_N', hue='Group_N', data=dd, palette=active, ax=ax, orient='h')
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    title_suffix = "log10(views + 1)" if value_col.endswith("log10") else "views"
    ax.set_title(f'Comparison of {title_suffix} by Group'); ax.set_xlabel(title_suffix); ax.set_ylabel(None)

@plot_dual_theme(section='eda')
def plot_rgo_stack(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Stacked bar: orientation mix within race×gender (top lines)."""
    piv = (data.pivot_table(index=['Race','Gender'], columns='Orientation', values='Count', aggfunc='sum').fillna(0))
    share = piv.div(piv.sum(axis=1), axis=0).mul(100)
    # consistent orientation ordering helps color association
    order = [c for c in ['Lesbian','Gay','Bisexual','Other/Unspecified'] if c in share.columns]
    share = share[order]
    share.plot(kind='barh', stacked=True, ax=ax, legend=True)
    ax.set_xlim(0, 100)
    ax.set_title('Orientation Coverage within Race×Gender')
    ax.set_xlabel('Share (%)')
    ax.set_ylabel('Race × Gender')
    # move legend to top, no overlap with bars
    ax.legend(title='Orientation', loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=min(4, len(order)), frameon=False)
    # draw % labels for segments that are visible enough
    for bars in ax.containers:
        ax.bar_label(bars, fmt=lambda v: f"{v:.1f}%" if v >= 3 else "", label_type='center', fontsize=9)


@plot_dual_theme(section='eda')
def plot_lang_intersection_heatmap(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Heatmap of language share across intersections (top languages)."""
    piv = data.pivot_table(index='language', columns='Group', values='Share', aggfunc='sum').fillna(0.0)
    sns.heatmap(piv, annot=False, fmt=".1f", cmap='magma', cbar_kws={'label': 'Share (%)'}, ax=ax)
    ax.set_title('Language × Intersection Share (%)')
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    # ensure nothing is clipped when saving
    for label in ax.get_xticklabels(): label.set_rotation(30); label.set_ha('right')


@plot_dual_theme(section='eda')
def plot_uploader_concentration(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Bar chart of HHI & Top-10 share (secondary axis)."""
    if data.empty: ax.set_visible(False); return
    ax2 = ax.twinx()
    sns.barplot(x='Group', y='HHI', data=data, ax=ax)
    sns.scatterplot(x='Group', y='Top10 Share %', data=data, ax=ax2, s=60)
    ax.set_title('Uploader Concentration by Intersection'); ax2.set_ylabel('Top-10 Share (%)'); ax.set_ylabel('HHI')

@plot_dual_theme(section='eda')
def plot_rating_count_box(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Boxplot of ratings count by group (reliability)."""
    sns.boxplot(
        x='ratings', y='Group', data=data, ax=ax, orient='h',
        linewidth=1.2, width=0.5,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(alpha=0.6)
    )
    ax.set_title('Ratings Count by Group')
    ax.set_xlabel('Ratings')
    ax.grid(True, axis='x', alpha=0.25)


@plot_dual_theme(section='eda')
def plot_cardinality_hist(data: pd.Series, title: str, ax=None, palette=None, **kwargs):
    """Histogram of label cardinality (tags/categories)."""
    ax.hist(data, bins=40); ax.set_title(title); ax.set_xlabel('Count per video'); ax.set_ylabel('Frequency')

@plot_dual_theme(section='eda')
def plot_title_len_box(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Box/violin alternative for title tokens by group."""
    sns.boxplot(x='TitleTokens', y='Group', data=data, ax=ax, orient='h'); ax.set_title('Title Token Length by Group')

@plot_dual_theme(section='eda')
def plot_outlier_burden(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Dot chart for outlier burden by group (share above global 99th pct)."""
    m = data.set_index('Group')[['Above99% duration %','Above99% views %','Above99% ratings %']]
    y = np.arange(len(m))
    offsets = np.linspace(-0.18, 0.18, m.shape[1])
    for off, col in zip(offsets, m.columns):
        ax.scatter(m[col], y + off, s=50, label=col)
    ax.set_yticks(y)
    ax.set_yticklabels(m.index)
    ax.set_xlabel('Share (%)')
    ax.set_title('Outlier Burden (99th pct)')
    ax.grid(True, axis='x', alpha=0.3)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=m.shape[1], frameon=False)


@plot_dual_theme(section='eda')
def plot_month_seasonality(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Line plot of month-of-year share by group."""
    for g, sub in data.groupby('Group'):
        sub = sub.sort_values('Month'); ax.plot(sub['Month'].astype(int), sub['Share %'], marker='o', label=g)
    ax.set_xticks(range(1,13)); ax.legend(); ax.set_title('Seasonality by Month'); ax.set_xlabel('Month'); ax.set_ylabel('Share (%)')

@plot_dual_theme(section='eda')
def plot_quality_proxies(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Bar chart of quality proxies % by group."""
    dd = data.set_index('Group')[['HD %','4K %','Verified %']]
    dd.plot(kind='bar', ax=ax); ax.set_title('Quality Proxies by Intersection'); ax.set_ylabel('Share (%)'); ax.legend(loc='upper right')

@plot_dual_theme(section='eda')
def plot_vpd_box(base_vpd: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Boxplot of log10(views_per_day + 1) by group."""
    d = base_vpd.copy(); d['log10_vpd'] = np.log10(d['vpd'].clip(lower=0)+1.0)
    sns.boxplot(x='log10_vpd', y='Group', data=d, ax=ax, orient='h')
    ax.set_title('Views per Day (log10) by Group'); ax.set_xlabel('log10(vpd + 1)')

@plot_dual_theme(section='eda')
def plot_entropy(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Bar chart of entropy by group."""
    dd = data.set_index('Group')[['Entropy (Categories)','Entropy (Tags)']]
    dd.plot(kind='bar', ax=ax)
    ax.set_title('Category/Tag Diversity (Entropy)')
    ax.set_ylabel('Bits')
    # move legend above plot so it never hides bars
    leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    # add a little top margin for the legend
    ax.margins(y=0.05)

@plot_dual_theme(section='eda')
def plot_orientation_stack(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Stacked bar of orientation shares within each group."""
    piv = data.pivot_table(index='Group', columns='Orientation', values='Count', aggfunc='sum').fillna(0)
    share = piv.div(piv.sum(axis=1), axis=0).mul(100)
    order = [c for c in ['Lesbian','Gay','Bisexual','Other/Unspecified'] if c in share.columns]
    share = share[order]
    share.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Orientation Coverage by Group')
    ax.set_ylabel('Share (%)')
    # legend outside to avoid overlap
    ax.legend(title='Orientation', loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=min(4, len(order)), frameon=False)
    # label only visible segments
    for bars in ax.containers:
        ax.bar_label(bars, fmt=lambda v: f"{v:.1f}%" if v >= 3 else "", label_type='center', fontsize=9)


@plot_dual_theme(section='eda')
def plot_quality_lang_heatmap(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Heatmap of quality proxies % across languages."""
    df_ = data.copy()
    df_ = df_.set_index('language') if 'language' in df_.columns else df_
    cols = [c for c in ['HD %', '4K %', 'Verified %'] if c in df_.columns]
    sns.heatmap(df_[cols], annot=False, cmap='magma', cbar_kws={'label':'% of videos'}, ax=ax)
    ax.set_title('Quality Proxies by Language')
    ax.set_xlabel(None); ax.set_ylabel(None)
    for label in ax.get_xticklabels(): label.set_rotation(30); label.set_ha('right')


@plot_dual_theme(section='eda')
def plot_overlap_matrix(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Matrix plot of uploader overlap across groups."""
    sns.heatmap(data, annot=True, fmt='d', cmap='mako', ax=ax); ax.set_title('Uploader Overlap Matrix'); ax.set_xlabel('Group'); ax.set_ylabel('Group')

# --- 7) Main ------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """Run Step 02 EDA on the REAL corpus by default (no DB access)."""
    t_all = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true", help="Random sample from parquet; writes *_selfcheck artefacts.")
    p.add_argument("--sample", type=int, default=None, help="Sample size for self-check (default: min(80k, N)).")
    args = p.parse_args(argv)

    print("--- Starting Step 02: Comprehensive EDA ---")
    print("[NOTE] Titles may be non-English; tags/categories help anchor semantics.")
    print(f"[READ] Parquet: {CORPUS_PATH}")

    df = pd.read_parquet(CORPUS_PATH)
    df = _ensure_minimal_columns(df)

    if args.selfcheck:
        n = args.sample or min(80_000, len(df))
        df = df.sample(n=n, random_state=SEED, replace=False).reset_index(drop=True)
        print(f"[SELF-CHECK] Random sample drawn: {len(df):,} rows (seed={SEED}).")
        suffix = "_selfcheck"
    else:
        suffix = ""

    total_videos = len(df)
    print(f"\n[STATS] Total videos considered: {total_videos:,}")

    # Tag index (single pass)
    tag_index, tag_df = build_tag_index(df)

    # Core analyses
    top_cats                      = analyze_category_distribution(df, top_n=15)
    top_tags                      = analyze_top_tags(df, tag_index, top_n=20)
    lex                           = _load_lexicon()
    prot_tags, has_protected      = analyze_top_protected_tags(df, tag_index, tag_df, lex, top_n=20)
    intersections                 = analyze_full_intersections(df)
    ratings_plot, ratings_stats   = analyze_rating_disparities(df)
    views_plot,   views_stats     = analyze_views_disparities(df, log10=True)
    med_shift_tbl                 = protected_rating_medians(df, has_protected)

    # New enhancements
    rgo                           = triple_intersection_rgo(df)
    lang_inter                    = language_by_intersection(df)
    up_conc                       = uploader_concentration(df)
    rating_cnt                    = rating_count_by_group(df)
    cardinality_tbl               = label_cardinality(df)
    title_q                       = title_quality(df)
    outlier_tbl                   = outlier_burden(df)
    month_by_g, age_days          = seasonality_and_recency(df)
    quality_g                     = quality_proxies_by_group(df)
    vpd_summary, vpd_base         = views_per_day_snapshot(df)
    entropy_tbl                   = entropy_by_group(df)
    orient_tbl                    = orientation_by_group(df)
    age_tag_tbl                   = age_tag_by_group(df)
    quality_lang_tbl              = quality_by_language(df)
    dup_summary, overlap_mat      = duplication_and_overlap(df)

    # Console summary (selected)
    print("\n=== REAL NUMBERS SUMMARY ===")
    print(f"Total videos: {total_videos:,}")
    show_n = min(10, len(top_cats))
    print(f"\nTop {show_n} categories (by videos):")
    print(top_cats.head(show_n).to_string(index=False))
    show_n = min(10, len(top_tags))
    print(f"\nTop {show_n} tags (by videos with tag):")
    print(top_tags[['Tag','Videos With Tag','Percentage']].head(show_n).to_string(index=False))
    show_n = min(10, len(rgo))
    print(f"\nTop {show_n} triple intersections (race×gender×orientation):")
    if show_n: print(rgo.head(show_n).to_string(index=False))
    print("\nUploader concentration (HHI, Top-10 share %):")
    print((up_conc.head(4) if not up_conc.empty else up_conc).to_string(index=False))
    print("\nOutlier burden (99th pct thresholds; % within each group):")
    print(outlier_tbl.to_string(index=False))
    print("\nViews per day (median) by group:")
    print(vpd_summary.to_string(index=False))
    print("\nDuplicates summary (pre-split leakage heads-up):")
    print(dup_summary.to_string(index=False))

    # Save data artefacts (prefix 02_)
    print("\nSaving data artefacts...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    def save_df(df_, name: str):
        path = DATA_DIR / f"02_{name}{suffix}.csv"
        df_.to_csv(path, index=False)
        print(f"✓ Artefact saved: {path}")

    # Core
    save_df(top_cats, "eda_top_categories")
    save_df(intersections, "eda_intersectional_representation")
    save_df(top_tags, "eda_top_tags")
    save_df(prot_tags, "eda_top_protected_tags")
    ratings_stats.round(3).to_csv(DATA_DIR / f"02_eda_rating_disparities_stats{suffix}.csv")
    print(f"✓ Artefact saved: {DATA_DIR / f'02_eda_rating_disparities_stats{suffix}.csv'}")
    views_stats.round(3).to_csv(DATA_DIR / f"02_eda_views_disparities_stats{suffix}.csv")
    print(f"✓ Artefact saved: {DATA_DIR / f'02_eda_views_disparities_stats{suffix}.csv'}")
    save_df(med_shift_tbl.reset_index(), "eda_protected_rating_medians")

    # New
    save_df(rgo, "eda_intersections_rgo")
    save_df(lang_inter, "eda_language_by_intersection")
    save_df(up_conc, "eda_uploader_concentration")
    save_df(rating_cnt, "eda_rating_count_by_group")
    save_df(cardinality_tbl, "eda_label_cardinality")
    save_df(title_q, "eda_title_quality_by_group")
    save_df(outlier_tbl, "eda_outlier_burden")
    save_df(month_by_g, "eda_month_seasonality_by_group")
    save_df(age_days, "eda_age_days_by_group")
    save_df(quality_g, "eda_quality_proxies_by_group")
    save_df(vpd_summary, "eda_views_per_day_by_group")
    save_df(entropy_tbl, "eda_entropy_by_group")
    save_df(orient_tbl, "eda_orientation_by_group")
    save_df(age_tag_tbl, "eda_age_tag_by_group")
    save_df(quality_lang_tbl, "eda_quality_by_language")
    save_df(dup_summary, "eda_duplication_summary")
    overlap_mat.to_csv(DATA_DIR / f"02_eda_uploader_overlap_matrix{suffix}.csv")
    print(f"✓ Artefact saved: {DATA_DIR / f'02_eda_uploader_overlap_matrix{suffix}.csv'}")

    # Figures (prefix 02_*)
    print("\nGenerating visualizations...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_categories(data=top_cats, save_path=str(FIGURES_DIR / f'02_eda_top_categories_bar{suffix}'), figsize=(10, 8))
    plot_intersections(data=intersections, save_path=str(FIGURES_DIR / f'02_eda_intersections_bar{suffix}'), figsize=(10, 8))
    if not ratings_plot.empty:
        plot_ratings(data=ratings_plot, save_path=str(FIGURES_DIR / f'02_eda_ratings_boxplot{suffix}'), figsize=(10, 6))
    if not views_plot.empty:
        vcol = 'views_log10' if 'views_log10' in views_plot.columns else 'views'
        plot_views(data=views_plot, value_col=vcol, save_path=str(FIGURES_DIR / f'02_eda_views_boxplot{suffix}'), figsize=(10, 6))
    if not rgo.empty:
        plot_rgo_stack(data=rgo, save_path=str(FIGURES_DIR / f'02_eda_intersections_rgo{suffix}'), figsize=(10, 8))
    if not lang_inter.empty:
        plot_lang_intersection_heatmap(
    data=lang_inter,
    save_path=str(FIGURES_DIR / f'02_eda_lang_intersection_heatmap{suffix}'),
    figsize=(10, 8),
)

    if not up_conc.empty:
        plot_uploader_concentration(data=up_conc, save_path=str(FIGURES_DIR / f'02_eda_uploader_concentration{suffix}'), figsize=(10, 6))
    if not df.empty:
        # ratings count box uses per-video rows
        tmp = pd.DataFrame({'Group': _group_labels_intersectional(df), 'ratings': df['ratings']}).dropna(subset=['Group'])
        plot_rating_count_box(data=tmp, save_path=str(FIGURES_DIR / f'02_eda_rating_count_box{suffix}'), figsize=(10, 6))
    # cardinality hist
    def _count_tokens(s):
        return s.fillna('').astype(str).str.strip().str.split(',').map(lambda x: 0 if x == [''] else len(x))
    plot_cardinality_hist(data=_count_tokens(df['tags']), title='Tag Cardinality per Video',
                          save_path=str(FIGURES_DIR / f'02_eda_cardinality_tags{suffix}'), figsize=(9, 6))
    plot_cardinality_hist(data=_count_tokens(df['categories']), title='Category Cardinality per Video',
                          save_path=str(FIGURES_DIR / f'02_eda_cardinality_cats{suffix}'), figsize=(9, 6))
    if not title_q.empty:
        # use per-video distribution for box
        base_title = pd.DataFrame({'Group': _group_labels_intersectional(df), 'TitleTokens': df['title'].fillna('').str.split().map(len)})
        base_title = base_title.dropna(subset=['Group'])
        if not base_title.empty:
            plot_title_len_box(data=base_title, save_path=str(FIGURES_DIR / f'02_eda_title_len_box{suffix}'), figsize=(10, 6))
    if not outlier_tbl.empty:
        plot_outlier_burden(data=outlier_tbl, save_path=str(FIGURES_DIR / f'02_eda_outlier_burden{suffix}'), figsize=(10, 6))
    if not month_by_g.empty:
        plot_month_seasonality(data=month_by_g, save_path=str(FIGURES_DIR / f'02_eda_month_seasonality{suffix}'), figsize=(10, 6))
    if not quality_lang_tbl.empty:
        plot_quality_lang_heatmap(
    data=quality_lang_tbl,
    save_path=str(FIGURES_DIR / f'02_eda_quality_lang_heatmap{suffix}'),
    figsize=(9, 7),
)

    if not vpd_base.empty:
        plot_vpd_box(
    base_vpd=vpd_base,
    save_path=str(FIGURES_DIR / f'02_eda_vpd_box{suffix}'),
    figsize=(10, 6),
)
    if not entropy_tbl.empty:
        plot_entropy(data=entropy_tbl, save_path=str(FIGURES_DIR / f'02_eda_entropy{suffix}'), figsize=(10, 6))
    if not orient_tbl.empty:
        plot_orientation_stack(data=orient_tbl, save_path=str(FIGURES_DIR / f'02_eda_orientation_stack{suffix}'), figsize=(10, 6))
    if not overlap_mat.empty and (overlap_mat.to_numpy().sum() - np.trace(overlap_mat.to_numpy())) > 10:
        plot_overlap_matrix(data=overlap_mat, save_path=str(FIGURES_DIR / f'02_eda_uploader_overlap{suffix}'), figsize=(7, 6))
    else:
        print("[INFO] Uploader overlap is negligible; figure suppressed.")

    # Narrative + LaTeX
    print("\nGenerating narrative & LaTeX tables...")
    NARRATIVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _first_safe(df_, col, default="N/A"):
        return df_.iloc[0][col] if not df_.empty else default

    top_cat   = _first_safe(top_cats, 'Category')
    top_cat_n = int(_first_safe(top_cats, 'Video Count', 0))
    top_cat_p = float(_first_safe(top_cats, 'Percentage', 0.0))
    rgo_head  = rgo.head(1) if not rgo.empty else pd.DataFrame([{'Race':'N/A','Gender':'N/A','Orientation':'N/A','Percentage':0.0}])

    summary = f"""
# Automated Summary: Comprehensive EDA

**Corpus size:** {total_videos:,} videos. (Note: category/tag totals elsewhere can exceed N due to multi-label.)

**Top category:** “{top_cat}” — {top_cat_n:,} videos ({top_cat_p:.1f}% of corpus).

**Top triple-intersection (race×gender×orientation):** {rgo_head.iloc[0]['Race']} × {rgo_head.iloc[0]['Gender']} × {rgo_head.iloc[0]['Orientation']} at {float(rgo_head.iloc[0]['Percentage']):.1f}% share.

**Views/day (median) by group:**  
{vpd_summary.to_string(index=False)}

*Ethics note:* “18–25” tag prevalence by intersection is reported descriptively only.
"""

    with open(NARRATIVE_PATH if not args.selfcheck else NARRATIVE_PATH.with_name(NARRATIVE_PATH.stem + "_selfcheck.md"), 'w') as f:
        f.write(summary)
    print(f"✓ Narrative saved: {NARRATIVE_PATH if not args.selfcheck else NARRATIVE_PATH.with_name(NARRATIVE_PATH.stem + '_selfcheck.md')}")

    # LaTeX tables (02_* names)
    def save_tex(df_, name: str, caption: str, label: str, index_col: Optional[str] = None):
        path = str(TABLES_DIR / f"02_{name}{suffix}.tex")
        if df_.empty: return
        dataframe_to_latex_table(df_ if index_col is None else df_.set_index(index_col), path, caption, f"tab:{name}")

    save_tex(top_cats, "eda_top_categories", "Top 15 Most Frequent Video Categories.", "top-categories", index_col='Category')
    save_tex(intersections, "eda_intersections", "Representation of Race × Gender Intersections.", "intersections", index_col='Intersection')
    if not ratings_stats.empty:
        save_tex(ratings_stats, "eda_rating_stats", "Summary Statistics of Video Ratings by Group.", "rating-stats")
    if not views_stats.empty:
        save_tex(views_stats, "eda_views_stats", "Summary Statistics of log10(views+1) by Group.", "views-stats")
    save_tex(top_tags[['Tag','Videos With Tag','Percentage']], "eda_top_tags", "Top Tags by Video Coverage.", "top-tags", index_col='Tag')
    save_tex(prot_tags, "eda_top_protected_tags", "Top Protected/Bias Tags by Video Coverage.", "top-protected-tags", index_col='Protected Tag')
    save_tex(med_shift_tbl, "eda_protected_rating_medians", "Median Ratings with vs. without Protected/Bias Tags.", "protected-rating-medians")
    save_tex(rgo, "eda_intersections_rgo", "Triple Intersection Coverage (Race × Gender × Orientation).", "intersections-rgo")
    save_tex(lang_inter, "eda_language_by_intersection", "Language × Intersection Share & Outcomes (Top Languages).", "lang-inter")
    save_tex(up_conc, "eda_uploader_concentration", "Uploader Concentration by Intersection (HHI, Top-10 Share).", "uploader-concentration")
    save_tex(rating_cnt, "eda_rating_count_by_group", "Rating Count Distribution by Intersection (Median & IQR).", "rating-count")
    save_tex(cardinality_tbl, "eda_label_cardinality", "Label Cardinality by Intersection (Median #tags/#categories).", "label-cardinality")
    save_tex(title_q, "eda_title_quality_by_group", "Title/Token Quality by Intersection.", "title-quality")
    save_tex(outlier_tbl, "eda_outlier_burden", "Outlier Burden by Intersection (Share above 99th pct).", "outlier-burden")
    save_tex(month_by_g, "eda_month_seasonality_by_group", "Month-of-Year Seasonality by Intersection (Share %).", "month-seasonality")
    save_tex(age_days, "eda_age_days_by_group", "Median Age-in-Days by Intersection.", "age-days")
    save_tex(quality_g, "eda_quality_proxies_by_group", "Quality Proxies by Intersection.", "quality-proxies")
    save_tex(vpd_summary, "eda_views_per_day_by_group", "Median Views per Day by Intersection.", "vpd")
    save_tex(entropy_tbl, "eda_entropy_by_group", "Category/Tag Diversity (Entropy) by Intersection.", "entropy")
    save_tex(orient_tbl, "eda_orientation_by_group", "Orientation Coverage by Intersection (Counts).", "orientation")
    save_tex(age_tag_tbl, "eda_age_tag_by_group", "“18–25” Tag Prevalence by Intersection.", "age-tag")
    save_tex(quality_lang_tbl.set_index('language'), "eda_quality_by_language", "Quality Proxies Coverage by Language (Top Languages).", "quality-by-language")
    save_tex(overlap_mat, "eda_uploader_overlap_matrix", "Uploader Overlap Matrix across Intersections.", "uploader-overlap")

    _finish("eda.step02_total", t_all)
    print("\n--- Step 02: Comprehensive EDA Completed Successfully ---")

if __name__ == '__main__':
    main()
