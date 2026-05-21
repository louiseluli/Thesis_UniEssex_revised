# -*- coding: utf-8 -*-
"""
06_stratified_splitting.py
==========================

Purpose
-------
Create reproducible train/validation/test splits using a robust stratification
signal derived from intersectional groups (priority-based). Ensures that rare
classes are coalesced to avoid stratification errors.

What it does
------------
1) Loads the canonical parquet (no DB).
2) Builds a single 'stratify_key' with a clear priority order across
   intersectional groups; coalesces rare classes into 'Other' to make
   stratification stable.
3) Two-stage split:
     - Stage 1: Train+Val vs Test (e.g., 80/20)
     - Stage 2: Train vs Val   (e.g., 75/25 of the 80 → 60/20 overall)
4) Saves only video_id lists (CSV) for each split.
5) Prints friendly, timed logs with a total runtime line.

Notes on counts
---------------
- Splitting uses a **single label per video** (`stratify_key`), so split totals
  sum exactly to N. Elsewhere in the project (EDA, PMI, harm profiles) videos
  can have **multiple tags/categories simultaneously**; in those contexts, sums
  across labels can exceed N — this is expected in multi-label corpora.

Artefacts
---------
- outputs/data/06_train_ids.csv
- outputs/data/06_val_ids.csv
- outputs/data/06_test_ids.csv
- outputs/data/06_stratify_key_distribution.csv
- outputs/data/06_split_selfcheck.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- 1. Configuration and Setup ----------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config

CONFIG = load_config()

# Paths
CORPUS_PATH = Path(CONFIG['paths']['data']) / '01_ml_corpus.parquet'
OUTPUT_DIR  = Path(CONFIG['paths']['data'])

# Defaults (can be overridden via CLI)
DEFAULT_TEST_SIZE = 0.20              # fraction of total
DEFAULT_VAL_FRAC_OF_REMAINDER = 0.25  # -> 0.25 * 0.80 = 0.20 of total
DEFAULT_MIN_PER_CLASS = 50            # coalesce classes below this count
DEFAULT_SEED = int(CONFIG.get('reproducibility', {}).get('seed', 95))


# --- 2. Small timing helpers --------------------------------------------------
def _t0(msg: str) -> float:
    """Start a monotonic timer and print a standardized header."""
    t = time.perf_counter()
    print(msg)
    return t

def _tend(label: str, t0: float) -> None:
    """Finish a timer with a standardized [TIME] line."""
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")


# --- 3. Stratification helpers ------------------------------------------------
def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the required one-hot group columns exist (filled with zeros if missing).

    Returns
    -------
    pd.DataFrame
        A copy of the input with missing columns added as zeroes.
    """
    cols = [
        'intersectional_black_female',
        'race_ethnicity_white', 'race_ethnicity_black',
        'race_ethnicity_asian', 'race_ethnicity_latina',
        'gender_female'
    ]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0

    # Derive intersectional_black_female if not present or all zeros
    try:
        needs_ibf = ('intersectional_black_female' not in out.columns) or (int(out['intersectional_black_female'].sum()) == 0)
    except Exception:
        needs_ibf = True
    if needs_ibf and ('race_ethnicity_black' in out.columns) and ('gender_female' in out.columns):
        out['intersectional_black_female'] = ((out['race_ethnicity_black'] == 1) & (out['gender_female'] == 1)).astype(int)

    return out


def make_stratify_key(df: pd.DataFrame) -> pd.Series:
    """
    Build a single, **priority-based** stratification key.

    Priority (first match wins)
    ---------------------------
    1) Black_Female          -> intersectional_black_female == 1
    2) White_Female          -> race_ethnicity_white  & gender_female
    3) Asian_Female          -> race_ethnicity_asian  & gender_female
    4) Latina_Female         -> race_ethnicity_latina & gender_female
    5) Other                 -> everyone else

    Parameters
    ----------
    df : pd.DataFrame
        Corpus with one-hot protected columns.

    Returns
    -------
    pd.Series of dtype 'object'
        One label per video.
    """
    t0 = _t0("Creating priority-based 'stratify_key'...")
    d = _ensure_columns(df)

    keys = np.full(len(d), "Other", dtype=object)

    mask_bf = (d['intersectional_black_female'] == 1)
    keys[mask_bf] = "Black_Female"

    mask_wf = (d['race_ethnicity_white'] == 1) & (d['gender_female'] == 1) & (~mask_bf)
    keys[mask_wf] = "White_Female"

    mask_af = (d['race_ethnicity_asian'] == 1) & (d['gender_female'] == 1) & (~mask_bf) & (~mask_wf)
    keys[mask_af] = "Asian_Female"

    mask_lf = (d['race_ethnicity_latina'] == 1) & (d['gender_female'] == 1) & (~mask_bf) & (~mask_wf) & (~mask_af)
    keys[mask_lf] = "Latina_Female"

    ser = pd.Series(keys, index=d.index, name="stratify_key")
    vc = ser.value_counts(dropna=False).sort_values(ascending=False)
    print("Value counts (pre-coalesce):")
    print(vc.to_string())
    _tend("split.make_stratify_key", t0)
    return ser


def coalesce_rare_classes(key: pd.Series, min_count: int) -> pd.Series:
    """
    Coalesce classes with < min_count observations into 'Other'.

    Parameters
    ----------
    key : pd.Series
        Class labels (one per row), typically from `make_stratify_key`.
    min_count : int
        Minimum count to keep a class intact for stratification.

    Returns
    -------
    pd.Series
        Possibly modified labels with rare classes mapped to 'Other'.

    Notes
    -----
    - Ensures scikit-learn's stratifier has enough examples per class for both
      splits (train/val/test).
    """
    t0 = _t0(f"Coalescing rare classes (< {min_count}) into 'Other'...")
    vc = key.value_counts()
    rare = vc[vc < min_count].index
    out = key.where(~key.isin(rare), "Other")
    vc2 = out.value_counts()
    print("Value counts (post-coalesce):")
    print(vc2.to_string())
    _tend("split.coalesce_rare", t0)
    return out


def stratified_two_stage_split(
    df: pd.DataFrame,
    stratify_key: pd.Series,
    *,
    test_size: float,
    val_frac_of_remainder: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a two-stage stratified split:
      1) (Train+Val) vs Test
      2) Train vs Val (from Train+Val)

    Parameters
    ----------
    df : pd.DataFrame
        The full corpus (must align with `stratify_key` index).
    stratify_key : pd.Series
        One label per video, used as the stratification signal.
    test_size : float
        Fraction of the total reserved for the Test split (e.g., 0.20).
    val_frac_of_remainder : float
        Fraction of the (Train+Val) portion that becomes Validation (e.g., 0.25).
    seed : int
        Random state for reproducibility.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    t0 = _t0(f"Two-stage stratified split (test={test_size:.2%}, val_of_rem={val_frac_of_remainder:.2%})...")

    # Stage 1: Train+Val vs Test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_key
    )

    # Stage 2: Train vs Val from Train+Val
    val_size = val_frac_of_remainder
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_val_df['stratify_key']
    )

    _tend("split.two_stage", t0)
    return train_df, val_df, test_df


def save_split_ids(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    outdir: Path,
    *,
    suffix: str = ""
) -> None:
    """
    Save the split ID lists to CSV.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        DataFrames containing at least a 'video_id' column.
    outdir : Path
        Output directory.
    suffix : str, default=""
        Optional suffix like "_selfcheck" to ensure self-check runs do not
        overwrite full-run artefacts.

    Side-effects
    ------------
    - Writes files:
        06_train_ids{suffix}.csv
        06_val_ids{suffix}.csv
        06_test_ids{suffix}.csv
    - Emits a [TIME] line.

    Notes
    -----
    - Keeps imports at top of file.
    - No mid-file imports.
    """
    t0 = _t0("Saving split IDs to CSV...")
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure required column exists
    for _name, _df in (('train', train_df), ('val', val_df), ('test', test_df)):
        if 'video_id' not in _df.columns:
            raise KeyError(f"Missing 'video_id' column in {_name}_df")

    train_path = outdir / f"06_train_ids{suffix}.csv"
    val_path   = outdir / f"06_val_ids{suffix}.csv"
    test_path  = outdir / f"06_test_ids{suffix}.csv"

    train_df[['video_id']].to_csv(train_path, index=False)
    val_df[['video_id']].to_csv(val_path, index=False)
    test_df[['video_id']].to_csv(test_path, index=False)

    print(f"✓ Artefact saved: {train_path.resolve()}")
    print(f"✓ Artefact saved: {val_path.resolve()}")
    print(f"✓ Artefact saved: {test_path.resolve()}")
    _tend("split.save_ids", t0)


def save_stratify_distribution(stratify_key: pd.Series, outdir: Path, *, suffix: str = "") -> None:
    """
    Save the final stratification key distribution for auditability.

    Parameters
    ----------
    stratify_key : pd.Series
        Final, post-coalesce class labels (one per row).
    outdir : Path
        Output directory.
    suffix : str, default=""
        Optional suffix like "_selfcheck" so self-check runs do not overwrite
        full-run artefacts.
    """
    t0 = _t0("Saving stratify_key distribution...")
    path = outdir / f"06_stratify_key_distribution{suffix}.csv"
    (stratify_key.value_counts()
     .rename_axis("class")
     .reset_index(name="count")
     .to_csv(path, index=False))
    print(f"✓ Artefact saved: {path.resolve()}")
    _tend("split.save_distribution", t0)


def print_split_sizes(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, total: int) -> None:
    """
    Pretty-print split sizes and shares.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        The three splits.
    total : int
        Total number of rows in the original corpus.
    """
    print("\nSplit sizes:")
    print(f"  Training set:   {len(train_df):,} records ({len(train_df)/total:.2%})")
    print(f"  Validation set: {len(val_df):,} records ({len(val_df)/total:.2%})")
    print(f"  Test set:       {len(test_df):,} records ({len(test_df)/total:.2%})")

def write_selfcheck_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, outdir: Path, *, seed: int) -> None:
    """
    Seeded, non-destructive self-check: verifies disjointness and coverage on a random sample.
    Writes outputs/data/06_split_selfcheck.csv.
    """
    t0 = _t0("Writing self-check for splits...")
    rng = np.random.default_rng(int(seed))
    # sample up to 10 ids from each (without assuming sizes)
    parts = []
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        if len(d):
            k = min(10, len(d))
            idx = rng.choice(len(d), size=k, replace=False)
            tmp = d.iloc[idx][['video_id']].copy()
            tmp['split'] = name
            parts.append(tmp)
    sample = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame(columns=['video_id','split'])

    # basic checks
    u_train = set(train_df['video_id'])
    u_val   = set(val_df['video_id'])
    u_test  = set(test_df['video_id'])
    inter_tv = len(u_train & u_val)
    inter_tt = len(u_train & u_test)
    inter_vt = len(u_val & u_test)
    print(f"[CHECK] Intersections (should be 0): train∩val={inter_tv}, train∩test={inter_tt}, val∩test={inter_vt}")

    out_path = outdir / "06_split_selfcheck.csv"
    sample.to_csv(out_path, index=False)
    print(f"✓ Self-check saved: {out_path.resolve()}")
    _tend("split.selfcheck", t0)

# --- 4. Main ------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    """
    Perform stratified train/val/test splitting on the REAL corpus by default.

    CLI
    ---
    # Full run
    python -m src.data.06_stratified_splitting

    # Self-check: quick run on a random sample from the parquet (no hardcoded rows)
    python -m src.data.06_stratified_splitting --selfcheck --sample 50000 --seed 123

    Options
    -------
    --test-size FLOAT                 (default: 0.20)
    --val-frac-of-remainder FLOAT     (default: 0.25)
    --min-per-class INT               (default: 50)
    --seed INT                        (default: CONFIG.reproducibility.seed or 95)
    --selfcheck                       (use a random subset)
    --sample INT                      (rows to sample when self-checking)
    """
    t_all = time.perf_counter()
    print("--- Starting Step 06: Stratified Data Splitting ---")
    print("[NOTE] Titles may be non-English; tags/categories help anchor semantics (multi-label upstream; single-label here for splitting).")


    # argparse imported at top
    p = argparse.ArgumentParser()
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--val-frac-of-remainder", type=float, default=DEFAULT_VAL_FRAC_OF_REMAINDER)
    p.add_argument("--min-per-class", type=int, default=DEFAULT_MIN_PER_CLASS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--selfcheck", action="store_true", help="Use a random subset from the real parquet.")
    p.add_argument("--sample", type=int, default=None, help="Number of rows for self-check sampling.")
    args = p.parse_args(argv)

    # Load corpus (full, then optional sampling)
    print(f"Loading corpus from {CORPUS_PATH}...")
    df = pd.read_parquet(CORPUS_PATH)
    if args.selfcheck and args.sample:
        df = df.sample(n=args.sample, random_state=args.seed, replace=False).reset_index(drop=True)
        print(f"[SELF-CHECK] Random sample drawn: {len(df):,} videos (seed={args.seed}).")

    total = len(df)
    print(f"[STATS] Total videos considered: {total:,}")

    # Build & coalesce stratification key
    stratify_key = make_stratify_key(df)
    stratify_key = coalesce_rare_classes(stratify_key, min_count=args.min_per_class)
    df = df.assign(stratify_key=stratify_key)

    # Do the two-stage split
    train_df, val_df, test_df = stratified_two_stage_split(
        df, stratify_key, test_size=args.test_size,
        val_frac_of_remainder=args.val_frac_of_remainder, seed=args.seed
    )

    # Report & save
    print_split_sizes(train_df, val_df, test_df, total)

    # Never overwrite full-run artefacts during self-checks:
    suffix = "_selfcheck" if args.selfcheck else ""
    save_split_ids(train_df, val_df, test_df, OUTPUT_DIR, suffix=suffix)
    save_stratify_distribution(stratify_key, OUTPUT_DIR, suffix=suffix)

    # Tiny self-check sample (separate file; never overwrites):
    write_selfcheck_splits(train_df, val_df, test_df, OUTPUT_DIR, seed=args.seed)

    _tend("split.step06_total", t_all)
    print("\n--- Step 06: Stratified Data Splitting Completed Successfully ---")


if __name__ == '__main__':
    main()
