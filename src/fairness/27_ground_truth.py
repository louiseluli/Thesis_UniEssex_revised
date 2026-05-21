# /Users/louisesfer/Documents/University of Essex/Dissertation/AlgoFairness-Pornometrics1/AlgoFairness-FinalCode copy/src/fairness/27_ground_truth.py
# -*- coding: utf-8 -*-
"""
Step 27 — Ground Truth Management
=================================

Purpose
-------
Create, quality-check, and finalize a GOLD label set for evaluation:
  1) Stratified sample from Test split by intersectional group × class
  2) Build annotation template (metadata only; no content viewing)
  3) Compute IAA (Cohen's κ for 2 raters, Fleiss' κ for >2)
  4) Adjudicate and save a single '27_gold_final.csv'
  5) Add supplementary samples to existing gold standard

What it does
------------
- Reads canonical corpus and Step-06 splits from your repo.
- Writes/reads from outputs/data/gold/.
- Uses number-first artefact names: 27_gold_*.csv
- Self-check mode appends `_selfcheck` before extensions without changing logic.
- Timed blocks and a final total time line.

CLI
---
# Create template (sample ~1200, stratified; use seed from config)
python -m src.fairness.27_ground_truth --make-template --sample 1200

# Add more samples to existing gold standard (excludes already annotated)
python -m src.fairness.27_ground_truth --add-samples --sample 500 --exclude "outputs/data/gold/gold_labels_annotator_louise.csv"

# Merge annotator files (pattern) and compute IAA preview (pairwise if 2 raters)
python -m src.fairness.27_ground_truth --merge "outputs/data/gold/gold_labels_annotator_*.csv"

# Finalize (majority vote + simple adjudication rules)
python -m src.fairness.27_ground_truth --finalize "outputs/data/gold/gold_labels_annotator_*.csv"

# Optional: run in self-check mode to avoid touching canonical artefacts
python -m src.fairness.27_ground_truth --make-template --selfcheck
"""

from __future__ import annotations

# --- Imports (keep at top) ---------------------------------------------------
import sys
import glob
import time
from pathlib import Path
from typing import Optional, List, Tuple, Set
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Project utils
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config

# --- Config & paths ----------------------------------------------------------
_t0_cfg = time.perf_counter()
CONFIG = load_config()
print(f"[TIME] theme_manager.load_config: {time.perf_counter() - _t0_cfg:.2f}s")

DATA_DIR = Path(CONFIG['paths']['data'])
OUT_DIR  = Path(CONFIG['paths']['outputs'])
GOLD_DIR = OUT_DIR / "data" / "gold"
GOLD_DIR.mkdir(parents=True, exist_ok=True)

SEED = int(CONFIG.get('reproducibility', {}).get('seed', 95))

CORPUS_PATH = DATA_DIR / '01_ml_corpus.parquet'
TRAIN_IDS   = DATA_DIR / '06_train_ids.csv'
VAL_IDS     = DATA_DIR / '06_val_ids.csv'
TEST_IDS    = DATA_DIR / '06_test_ids.csv'

TEXT_COL    = "model_input_text"
NUM_COLS    = ["duration", "ratings"]
CATS_COL    = "categories"

GROUPS = ["Black Women", "White Women", "Asian Women", "Latina Women", "Other"]

# --- Timers ------------------------------------------------------------------
def _t0(msg: str) -> float:
    """
    Start a high-resolution timer and print a heading.

    Parameters
    ----------
    msg : str
        Heading to print before timing.

    Returns
    -------
    float
        perf_counter start time.
    """
    t = time.perf_counter()
    print(msg)
    return t

def _tend(label: str, t0: float) -> None:
    """
    Stop timer and print standardized [TIME] message.

    Parameters
    ----------
    label : str
        Short label for the timed block.
    t0 : float
        Start time to compute elapsed duration.
    """
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")

# --- Suffixing / filenames ---------------------------------------------------
def _suffix(name: str, selfcheck: bool) -> str:
    """
    Append `_selfcheck` before the extension when requested.

    Examples
    --------
    _suffix('27_gold_final.csv', True) -> '27_gold_final_selfcheck.csv'
    """
    if not selfcheck:
        return name
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        return f"{stem}_selfcheck.{ext}"
    return f"{name}_selfcheck"

# --- Helpers -----------------------------------------------------------------
def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist for downstream template building.

    - Derives TEXT_COL if missing using title + tags
    - Ensures numeric columns exist (filled with zeros)
    - Ensures categories column exists
    """
    d = df.copy()
    if TEXT_COL not in d.columns:
        d[TEXT_COL] = (d.get('title', '').fillna('').astype(str) + ' ' +
                       d.get('tags', '').fillna('').astype(str))
    for c in NUM_COLS:
        if c not in d.columns:
            d[c] = 0
    if CATS_COL not in d.columns:
        d[CATS_COL] = ''
    return d

def _group_label_row(r: pd.Series) -> str:
    """
    Compute intersectional group label from one-hot columns (female + race).

    Returns
    -------
    str
        One of GROUPS (default 'Other' if no match).
    """
    bw = (r.get('race_ethnicity_black', 0) == 1) and (r.get('gender_female', 0) == 1)
    ww = (r.get('race_ethnicity_white', 0) == 1) and (r.get('gender_female', 0) == 1)
    aw = (r.get('race_ethnicity_asian', 0) == 1) and (r.get('gender_female', 0) == 1)
    lw = (r.get('race_ethnicity_latina', 0) == 1) and (r.get('gender_female', 0) == 1)
    if bw: return "Black Women"
    if ww: return "White Women"
    if aw: return "Asian Women"
    if lw: return "Latina Women"
    return "Other"

def _derive_is_amateur(d: pd.Series) -> int:
    """
    Heuristic: derive 'is_amateur' from first category string.

    Returns
    -------
    int
        1 if first category equals 'Amateur', else 0.
    """
    primary = d.get(CATS_COL, "")
    if pd.isna(primary): primary = ""
    primary = str(primary).split(",")[0].strip()
    return int(primary == "Amateur")

def _load_corpus_and_test() -> Tuple[pd.DataFrame, pd.Index]:
    """
    Load the main parquet corpus and Test IDs from Step 06.

    Returns
    -------
    (df, test_index)
        DataFrame of corpus (with ensured columns) and Index of test video_ids.
    """
    t0 = _t0(f"[READ] Parquet: {CORPUS_PATH}")
    df = pd.read_parquet(CORPUS_PATH)
    _tend("gt.load_corpus", t0)
    df = _ensure_cols(df)
    if not TEST_IDS.exists():
        raise FileNotFoundError(f"Missing {TEST_IDS} (Step 06).")
    te_ids = pd.read_csv(TEST_IDS)['video_id']
    return df, pd.Index(te_ids)

def _build_template(n: int, seed: int) -> pd.DataFrame:
    """
    Stratified sample by (Group × is_amateur_derived) from Test IDs.
    Ensures minimum per-group coverage; uses deterministic RNG.

    Parameters
    ----------
    n : int
        Approximate desired sample size across all strata.
    seed : int
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame
        Annotation template with AUTO hints and empty gold columns.
    """
    df, test_ids = _load_corpus_and_test()
    d = df[df['video_id'].isin(test_ids)].copy().reset_index(drop=True)
    d['Group_auto'] = d.apply(_group_label_row, axis=1)
    d['is_amateur_auto'] = d.apply(_derive_is_amateur, axis=1)

    # target per-cell (Group x class) quotas ~ balanced
    rng = np.random.default_rng(seed)
    per_group = max(150, n // len(GROUPS))  # e.g., 1200 -> ~240 per group
    per_cell = max(60, per_group // 2)      # ~ half per class

    rows = []
    for g in GROUPS:
        for c in [0, 1]:
            pool = d[(d['Group_auto'] == g) & (d['is_amateur_auto'] == c)]
            k = min(per_cell, len(pool))
            if k > 0:
                take = pool.sample(n=k, random_state=int(rng.integers(0, 1e9)))
                rows.append(take)
    if not rows:
        raise RuntimeError("No samples available under the requested stratification.")
    S = pd.concat(rows, axis=0, ignore_index=True).drop_duplicates('video_id')

    # Template shown to annotators; include metadata but not model-only columns
    tmpl = S.loc[:, ['video_id', 'title', 'tags', 'categories']].copy()
    tmpl.loc[:, 'is_amateur_gold'] = ""   # annotators fill 0/1
    tmpl.loc[:, 'group_gold'] = ""        # annotators pick from GROUPS
    tmpl.loc[:, 'AUTO_is_amateur'] = S['is_amateur_auto'].to_numpy()
    tmpl.loc[:, 'AUTO_group'] = S['Group_auto'].astype(str).to_numpy()
    return tmpl

def _add_supplementary_samples(n: int, seed: int, exclude_path: str) -> pd.DataFrame:
    """
    Add more samples to gold standard, excluding already annotated videos.

    Parameters
    ----------
    n : int
        Number of additional samples to add.
    seed : int
        Random seed for reproducibility.
    exclude_path : str
        Path to CSV file containing already annotated video_ids.

    Returns
    -------
    pd.DataFrame
        Template with new videos for annotation.
    """
    # Load existing annotated videos to exclude
    t0 = _t0(f"[READ] Existing annotations: {exclude_path}")
    existing_df = pd.read_csv(exclude_path)
    existing_ids = set(existing_df['video_id'].values)
    _tend("gt.load_existing", t0)
    print(f"  → Found {len(existing_ids)} already annotated videos to exclude")

    # Load corpus and test IDs
    df, test_ids = _load_corpus_and_test()

    # Filter to test set and exclude already annotated
    d = df[df['video_id'].isin(test_ids)].copy()
    d = d[~d['video_id'].isin(existing_ids)].reset_index(drop=True)
    print(f"  → Available pool after exclusions: {len(d)} videos")

    if len(d) == 0:
        raise RuntimeError("No additional videos available in test set after exclusions")

    # Add derived columns
    d['Group_auto'] = d.apply(_group_label_row, axis=1)
    d['is_amateur_auto'] = d.apply(_derive_is_amateur, axis=1)

    # Stratified sampling with balanced quotas
    rng = np.random.default_rng(seed)
    per_group = max(50, n // len(GROUPS))  # Target samples per group
    per_cell = max(25, per_group // 2)     # Target per class within group

    rows = []
    sampled_count = 0

    # Try to maintain balance across groups and classes
    for g in GROUPS:
        for c in [0, 1]:
            pool = d[(d['Group_auto'] == g) & (d['is_amateur_auto'] == c)]
            available = len(pool)
            k = min(per_cell, available)

            if k > 0:
                take = pool.sample(n=k, random_state=int(rng.integers(0, 1e9)))
                rows.append(take)
                sampled_count += k
                print(f"    - {g} (amateur={c}): sampled {k}/{available} available")
            else:
                print(f"    - {g} (amateur={c}): no videos available")

    if not rows:
        raise RuntimeError("Could not sample any additional videos with the requested stratification")

    # Combine and remove any duplicates
    S = pd.concat(rows, axis=0, ignore_index=True).drop_duplicates('video_id')

    # If we couldn't reach target n due to limited pool, sample more from available groups
    if len(S) < n:
        remaining_needed = n - len(S)
        remaining_pool = d[~d['video_id'].isin(S['video_id'])]

        if len(remaining_pool) > 0:
            additional = remaining_pool.sample(
                n=min(remaining_needed, len(remaining_pool)),
                random_state=int(rng.integers(0, 1e9))
            )
            S = pd.concat([S, additional], axis=0, ignore_index=True).drop_duplicates('video_id')
            print(f"  → Added {len(additional)} additional samples to reach closer to target")

    print(f"\n  Total new samples: {len(S)} videos")

    # Create template format
    tmpl = S[['video_id', 'title', 'tags', 'categories']].copy()
    tmpl['is_amateur_gold'] = ""  # To be filled by annotator
    tmpl['group_gold'] = ""       # To be filled by annotator
    tmpl['AUTO_is_amateur'] = S['is_amateur_auto']
    tmpl['AUTO_group'] = S['Group_auto']

    return tmpl

def _fleiss_kappa(table: np.ndarray) -> float:
    """
    Compute Fleiss' kappa for >2 raters.

    Parameters
    ----------
    table : np.ndarray
        Shape (N items, K categories) count matrix.

    Returns
    -------
    float
        Fleiss' kappa value.
    """
    N, k = table.shape
    n = table.sum(axis=1)[0]
    p = table.sum(axis=0) / (N * n)
    P = ((table**2).sum(axis=1) - n) / (n*(n-1))
    Pbar = P.mean()
    PbarE = (p**2).sum()
    return (Pbar - PbarE) / (1 - PbarE) if (1 - PbarE) != 0 else 0.0

def _read_annotators(glob_pat: str) -> List[pd.DataFrame]:
    """
    Read all annotator CSVs matching a glob pattern and validate columns.

    Parameters
    ----------
    glob_pat : str
        Glob pattern (e.g., 'outputs/data/gold/gold_labels_annotator_*.csv').

    Returns
    -------
    list[pd.DataFrame]
        List of per-annotator dataframes with required columns.
    """
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise FileNotFoundError(f"No annotator files match: {glob_pat}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        needed = {'video_id', 'is_amateur_gold', 'group_gold'}
        if not needed.issubset(df.columns):
            raise ValueError(f"{p} missing required columns {needed}")
        dfs.append(df[['video_id','is_amateur_gold','group_gold']].copy())
    return dfs

def _majority_vote(values: List) -> Optional:
    """
    Majority vote helper: returns the modal value among non-null entries.

    Parameters
    ----------
    values : list
        Raw values across annotators.

    Returns
    -------
    Optional
        The mode if present, else None.
    """
    s = pd.Series(values).dropna()
    if s.empty: return None
    return s.mode().iloc[0] if not s.mode().empty else None

def _finalize(glob_pat: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge annotator files, perform majority voting, and compute IAA.

    Parameters
    ----------
    glob_pat : str
        Glob for annotator CSVs.

    Returns
    -------
    (gold_df, iaa_df)
        Final GOLD labels and inter-annotator agreement summary.
    """
    dfs = _read_annotators(glob_pat)
    # Merge on video_id
    merged = dfs[0]
    for i, d in enumerate(dfs[1:], start=2):
        merged = merged.merge(d, on='video_id', how='outer', suffixes=(None, f"_{i}"))
    # Majority vote per field
    vote_rows = []
    for vid, row in merged.groupby('video_id'):
        amat_votes = []
        grp_votes  = []
        for c in merged.columns:
            if c.startswith('is_amateur_gold'):
                v = row[c].iloc[0]
                try: v = int(v)
                except: v = np.nan
                amat_votes.append(v)
            if c.startswith('group_gold'):
                grp_votes.append(row[c].iloc[0] if isinstance(row[c].iloc[0], str) else np.nan)
        mv_amat = _majority_vote(amat_votes)
        mv_grp  = _majority_vote(grp_votes)
        vote_rows.append({'video_id': vid, 'is_amateur_gold': mv_amat, 'group_gold': mv_grp})
    gold = pd.DataFrame(vote_rows)

    # IAA
    iaa_rows = []
    if len(dfs) == 2:
        a1 = dfs[0].set_index('video_id').loc[gold['video_id']]
        a2 = dfs[1].set_index('video_id').loc[gold['video_id']]
        kappa_amat = cohen_kappa_score(a1['is_amateur_gold'], a2['is_amateur_gold'])
        kappa_grp  = cohen_kappa_score(a1['group_gold'], a2['group_gold'])
        iaa_rows.append({'metric': "Cohen_kappa_is_amateur", 'value': round(kappa_amat, 3)})
        iaa_rows.append({'metric': "Cohen_kappa_group", 'value': round(kappa_grp, 3)})
    else:
        # Build count tables for Fleiss
        # Amateur
        vid_to_counts = []
        for vid in gold['video_id']:
            vals = []
            for d in dfs:
                s = d.loc[d['video_id']==vid, 'is_amateur_gold']
                if len(s): 
                    try:
                        vals.append(int(s.iloc[0]))
                    except Exception:
                        pass
            counts = [vals.count(0), vals.count(1)]
            vid_to_counts.append(counts)
        fk_amat = _fleiss_kappa(np.array(vid_to_counts))
        # Group (K=5)
        catG = GROUPS
        vid_to_counts = []
        for vid in gold['video_id']:
            vals = []
            for d in dfs:
                s = d.loc[d['video_id']==vid, 'group_gold']
                if len(s): vals.append(str(s.iloc[0]))
            counts = [vals.count(c) for c in catG]
            vid_to_counts.append(counts)
        fk_grp = _fleiss_kappa(np.array(vid_to_counts))
        iaa_rows.append({'metric': "Fleiss_kappa_is_amateur", 'value': round(fk_amat, 3)})
        iaa_rows.append({'metric': "Fleiss_kappa_group", 'value': round(fk_grp, 3)})

    iaa = pd.DataFrame(iaa_rows)
    return gold, iaa

# --- Main --------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Orchestrate Step 27: template creation, supplementary sampling, IAA preview,
    and finalization with majority vote — with self-check-safe filenames.

    CLI examples
    ------------
    python -m src.fairness.27_ground_truth --make-template --sample 1200
    python -m src.fairness.27_ground_truth --add-samples --sample 500 --exclude path.csv
    python -m src.fairness.27_ground_truth --merge "outputs/data/gold/gold_labels_annotator_*.csv"
    python -m src.fairness.27_ground_truth --finalize "outputs/data/gold/gold_labels_annotator_*.csv"
    """
    p = argparse.ArgumentParser()
    p.add_argument("--make-template", action="store_true", help="Create annotation CSV template.")
    p.add_argument("--add-samples", action="store_true", help="Add more samples excluding already annotated.")
    p.add_argument("--merge", type=str, default=None, help="Glob pattern of annotator CSVs to compute IAA (preview).")
    p.add_argument("--finalize", type=str, default=None, help="Glob pattern of annotator CSVs to finalize GOLD.")
    p.add_argument("--sample", type=int, default=1200, help="Target sample size for template or additional samples.")
    p.add_argument("--exclude", type=str, default=None, help="Path to CSV with already annotated videos to exclude.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--selfcheck", action="store_true", help="Write *_selfcheck artefacts (never overwrite canonicals).")
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    print("--- Step 27: Ground Truth Management ---")

    if args.make_template:
        t0 = _t0(f"Building annotation template (n≈{args.sample}, seed={args.seed}) ...")
        tmpl = _build_template(n=args.sample, seed=args.seed)
        out_name = _suffix(f"27_gold_template_{pd.Timestamp.now():%Y%m%d}.csv", args.selfcheck)
        out_path = GOLD_DIR / out_name
        tmpl.to_csv(out_path, index=False)
        _tend("gt.make_template", t0)
        print(f"[WRITE] {out_path.resolve()}")

    if args.add_samples:
        if not args.exclude:
            raise ValueError("--exclude path required when using --add-samples")

        t0 = _t0(f"Adding supplementary samples (n≈{args.sample}, seed={args.seed}) ...")
        tmpl = _add_supplementary_samples(n=args.sample, seed=args.seed, exclude_path=args.exclude)
        out_name = _suffix(f"27_gold_supplement_{pd.Timestamp.now():%Y%m%d}.csv", args.selfcheck)
        out_path = GOLD_DIR / out_name
        tmpl.to_csv(out_path, index=False)
        _tend("gt.add_samples", t0)
        print(f"[WRITE] {out_path.resolve()}")
        print(f"\nNext steps:")
        print(f"1. Annotate the new videos in: {out_path.name}")
        print(f"2. Append to your existing annotations")
        print(f"3. Run finalize with both files to create the expanded gold standard")

    if args.merge:
        t0 = _t0("Computing IAA (merge annotators) ...")
        dfs = _read_annotators(args.merge)
        # Quick IAA without adjudication (pairwise Cohen if len==2)
        if len(dfs) == 2:
            M = dfs[0].merge(dfs[1], on='video_id', suffixes=('_1','_2'))
            k1 = cohen_kappa_score(M['is_amateur_gold_1'], M['is_amateur_gold_2'])
            k2 = cohen_kappa_score(M['group_gold_1'], M['group_gold_2'])
            iaa = pd.DataFrame([
                {'metric':'Cohen_kappa_is_amateur','value':round(k1,3)},
                {'metric':'Cohen_kappa_group','value':round(k2,3)}
            ])
        else:
            # For >2 raters, advise finalize to compute Fleiss properly
            iaa = pd.DataFrame([{'metric':'note','value':'Run --finalize to compute Fleiss κ'}])
        iaa_name = _suffix("27_gold_iaa_preview.csv", args.selfcheck)
        iaa_path = GOLD_DIR / iaa_name
        iaa.to_csv(iaa_path, index=False)
        _tend("gt.merge_preview", t0)
        print(f"[WRITE] {iaa_path.resolve()}")

    if args.finalize:
        t0 = _t0("Finalizing GOLD via majority vote ...")
        gold, iaa = _finalize(args.finalize)
        gold_name = _suffix("27_gold_final.csv", args.selfcheck)
        iaa_name  = _suffix("27_gold_iaa.csv", args.selfcheck)
        gold_path = GOLD_DIR / gold_name
        iaa_path  = GOLD_DIR / iaa_name
        gold.to_csv(gold_path, index=False)
        iaa.to_csv(iaa_path, index=False)
        _tend("gt.finalize", t0)
        print(f"[WRITE] {gold_path.resolve()}")
        print(f"[WRITE] {iaa_path.resolve()}")

    _tend("gt.step27_total", t_all)
    print("\n--- Step 27 completed ---")

if __name__ == "__main__":
    main()
