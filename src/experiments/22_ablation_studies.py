#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_ablation_studies.py
======================

Purpose
-------
Sensitivity (ablation) analysis for category-based representation metrics:
we recompute Black-women (BW) category over/under-representation (log2 RR,
Laplace-smoothed) under controlled perturbations and report deltas vs a
baseline KPI.

Scenarios
---------
A) Lexicon OFF: remove race-coded tokens (e.g., "ebony", "black", etc.)
B) Random category noise: drop {10%, 25%, 50%} of (item, category) assignments
C) Top-K coverage: recompute KPI with K ∈ {10, 20, 30, 50}

Baseline KPI
------------
Mean absolute log2 RR (BW) across top-K_base categories (default 30).

Interpretability & language notes
---------------------------------
- Some titles are non-English; tags/categories (MPU) anchor semantics.
- Categories are multi-label; totals can exceed N by design.
- Years are integers; if ratings appear elsewhere, show 1 decimal (not used here).

CLI
---
# Full (canonical artefacts under outputs/ablation/)
python -m src.experiments.22_ablation_studies

# Self-check (seeded row subsample; writes *_selfcheck.csv)
python -m src.experiments.22_ablation_studies --selfcheck --sample 120000 --top-k-base 30
"""

from __future__ import annotations

# ----------------------------- Imports (top only) ----------------------------
import argparse
import sys
import time
from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure 'src' is importable when running as a plain script
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Project utils (prints its own [TIME] line)
from src.utils.theme_manager import load_config  # noqa: E402

# ----------------------------- Timers ----------------------------------------
def _t0(msg: str) -> float:
    """Start a timer and print a standardized message."""
    print(msg)
    return time.perf_counter()

def _tend(label: str, t_start: float) -> None:
    """End a timer and print a standardized [TIME] line."""
    print(f"[TIME] {label}: {time.perf_counter() - t_start:.2f}s")

# ----------------------------- Config & paths --------------------------------
CONFIG = load_config()
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
rng = np.random.default_rng(SEED)

ROOT = Path(CONFIG.get("project", {}).get("root", Path(__file__).resolve().parents[2]))
PATHS = CONFIG.get("paths", {}) or {}
DATA_DIR = Path(PATHS.get("data", ROOT / "outputs" / "data"))
ABL_DIR = Path(PATHS.get("ablation", ROOT / "outputs" / "ablation"))
NARR_DIR = Path(PATHS.get("narratives", ROOT / "outputs" / "narratives")) / "automated"
ABL_DIR.mkdir(parents=True, exist_ok=True)
NARR_DIR.mkdir(parents=True, exist_ok=True)

# Corpus candidates (schema may vary)
CORPUS_CANDIDATES = [
    DATA_DIR / "01_ml_corpus.parquet",
    DATA_DIR / "ml_corpus.parquet",
    DATA_DIR / "01_ml_corpus.snappy.parquet",
]

# Smoothing & defaults
LAPLACE = 1.0
TOPK_DEFAULT = int(CONFIG.get("analysis", {}).get("top_k_categories", 30))

# Race-coded tokens to remove in "Lexicon OFF" (lowercased; extend as needed)
BW_RACE_TOKENS = {
    "ebony", "black", "african", "african american", "afro", "blk",
    "ebony teen", "ebony milf", "black teen", "black milf"
}

# ----------------------------- Small utils -----------------------------------
def _first_existing(paths: List[Path]) -> Optional[Path]:
    """Return the first existing path from a candidates list, or None."""
    for p in paths:
        if p.exists():
            return p
    return None

def _parse_listish(x) -> List[str]:
    """
    Parse list-like strings/structures into a list of cleaned, lowercase tokens.
    Handles: python-list-in-string, comma/pipe-separated, true lists/tuples/sets.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        vals = list(x)
    else:
        s = str(x).strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                obj = literal_eval(s)
                vals = list(obj) if isinstance(obj, (list, tuple, set)) else [s]
            except Exception:
                vals = [s]
        else:
            sep = "," if ("," in s and "|" not in s) else ("|" if "|" in s else None)
            vals = s.split(sep) if sep else s.split()
    out = []
    for v in vals:
        t = str(v).strip().strip("'\"`").lower()
        if t and t not in {"nan", "none", "null"}:
            out.append(t)
    return out

def _round(x: float, k: int = 3) -> float:
    """Safe rounding to k decimals."""
    try:
        return round(float(x), k)
    except Exception:
        return x

# ----------------------------- IO & preparation ------------------------------
def _load_corpus() -> pd.DataFrame:
    """
    Load the canonical corpus parquet using a robust path search.
    """
    t = _t0("Loading corpus parquet ...")
    p = _first_existing(CORPUS_CANDIDATES)
    if p is None:
        raise FileNotFoundError("No corpus parquet found under outputs/data/. "
                                "Expected one of: 01_ml_corpus.parquet, ml_corpus.parquet")
    df = pd.read_parquet(p)
    _tend("step22.load_corpus", t)
    return df

def _prepare_protected(df: pd.DataFrame) -> None:
    """
    Ensure tidy 'race_ethnicity' and 'gender' columns exist.
    Derive from one-hot if needed. Normalize gender labels.
    """
    t = _t0("Preparing protected columns ...")

    def _derive_from_onehot(prefix: str, out_col: str, fallback: str = "unknown") -> Optional[str]:
        pref = f"{prefix}_"
        onehot = [c for c in df.columns if c.lower().startswith(pref)]
        if not onehot:
            return None
        labels = [c[len(pref):].lower() for c in onehot]
        mixed_token = "mixed_or_other" if any(l == "mixed_or_other" for l in labels) else "mixed"
        oh = df[onehot].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        oh = (oh > 0.5).astype(int)
        vals = []
        for i in range(len(df)):
            row = oh.iloc[i].values
            hits = np.where(row == 1)[0]
            if len(hits) == 1:
                vals.append(labels[hits[0]])
            elif len(hits) > 1:
                vals.append(mixed_token)
            else:
                vals.append(fallback)
        df[out_col] = pd.Series(vals, index=df.index, dtype="object")
        return out_col

    if "race_ethnicity" not in df.columns:
        _derive_from_onehot("race_ethnicity", "race_ethnicity")
    if "gender" not in df.columns:
        _derive_from_onehot("gender", "gender")

    if "gender" in df.columns:
        def _norm_gender(x: str) -> str:
            s = (x or "").lower()
            if s in {"female", "woman", "women", "cis_female", "cis_woman"}:
                return "female"
            if s in {"male", "man", "men", "cis_male", "cis_man"}:
                return "male"
            return s or "unknown"
        df["gender"] = df["gender"].map(_norm_gender)

    print(f"[INFO] Protected columns present: {[c for c in ['race_ethnicity','gender'] if c in df.columns]}")
    _tend("step22.prepare_protected", t)

def _extract_categories(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Extract categories/tags column as a Series of list[str].
    Tries config-preferred names first, then common fallbacks.
    """
    t = _t0("Parsing categories/tags ...")
    cols_cfg: Dict[str, str] = CONFIG.get("columns", {}) or {}
    prefer = [cols_cfg.get(k) for k in ("categories", "tags", "labels") if cols_cfg.get(k)]
    fallbacks = ["categories", "tags", "category", "tag_list", "labels"]
    for c in [*prefer, *fallbacks]:
        if c and c in df.columns:
            s = df[c].apply(_parse_listish)
            _tend("step22.parse_categories", t)
            return s
    print("[INFO] No categories/tags columns found.")
    _tend("step22.parse_categories", t)
    return None

# ----------------------------- Core computations -----------------------------
def _explode(df: pd.DataFrame, cat_lists: pd.Series, keep: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Explode list-like categories per row to long format, optionally filtering to 'keep'.
    """
    tmp = df.copy()
    tmp["_categories"] = cat_lists
    tmp = tmp.loc[tmp["_categories"].map(bool)]
    if tmp.empty:
        return tmp
    tmp = tmp.explode("_categories")
    if keep:
        tmp = tmp.loc[tmp["_categories"].isin(keep)]
    return tmp

def _topk_vocab(cat_lists: pd.Series, k: int) -> List[str]:
    """
    Build the top-K category vocabulary by frequency.
    """
    cnt: Dict[str, int] = {}
    for lst in cat_lists:
        for tkn in lst:
            cnt[tkn] = cnt.get(tkn, 0) + 1
    return [w for w, _ in sorted(cnt.items(), key=lambda kv: kv[1], reverse=True)[:k]]

def _bw_log2rr_by_category(df: pd.DataFrame, cat_lists: pd.Series, top: List[str]) -> pd.DataFrame:
    """
    Compute BW vs global share per category with Laplace smoothing,
    returning log2 RR and supporting counts (like Step-18 'under/over' table).
    """
    ex = _explode(df, cat_lists, keep=top)
    if ex.empty:
        return pd.DataFrame(columns=["category", "n_cat", "n_bw_cat", "share_bw_in_cat",
                                     "global_share_bw", "repr_ratio_bw", "log2_rr_bw"])

    is_bw = (ex["race_ethnicity"].astype(str).str.lower() == "black") & \
            (ex["gender"].astype(str).str.lower() == "female")

    cat_totals = ex.groupby("_categories").size().rename("n_cat")
    bw_counts = ex.loc[is_bw].groupby("_categories").size().reindex(cat_totals.index, fill_value=0).rename("n_bw_cat")

    n_total = int(len(ex))
    n_bw_global = int(is_bw.sum())
    p_bw_global = (n_bw_global + LAPLACE) / (n_total + 2 * LAPLACE)

    p_bw_cat = (bw_counts + LAPLACE) / (cat_totals + 2 * LAPLACE)
    rr = p_bw_cat / p_bw_global
    log2_rr = np.log2(rr)

    out = pd.DataFrame({
        "category": cat_totals.index,
        "n_cat": cat_totals.values.astype(int),
        "n_bw_cat": bw_counts.values.astype(int),
        "share_bw_in_cat": p_bw_cat.values.astype(float),
        "global_share_bw": float(p_bw_global),
        "repr_ratio_bw": rr.values.astype(float),
        "log2_rr_bw": log2_rr.values.astype(float),
    }).sort_values("log2_rr_bw", ascending=True)
    return out

def _kpi_mean_abs_log2rr(bw_table: pd.DataFrame) -> float:
    """Baseline KPI: mean absolute log2 RR across the provided table."""
    if bw_table.empty:
        return float("nan")
    vals = pd.to_numeric(bw_table["log2_rr_bw"], errors="coerce").to_numpy()
    if vals.size == 0:
        return float("nan")
    return float(np.nanmean(np.abs(vals)))

# ----------------------------- Perturbations ---------------------------------
def _apply_lexicon_off(cat_lists: pd.Series) -> pd.Series:
    """Remove race-coded tokens (lowercase) used in the BW-focused lexicon."""
    def filt(lst: List[str]) -> List[str]:
        return [t for t in lst if t not in BW_RACE_TOKENS]
    return cat_lists.apply(filt)

def _apply_noise_drop(cat_lists: pd.Series, p_drop: float) -> pd.Series:
    """Randomly drop each (row, token) with probability p_drop ∈ [0,1)."""
    def drop(lst: List[str]) -> List[str]:
        if not lst:
            return lst
        mask = rng.random(len(lst)) >= p_drop
        return [t for t, keep in zip(lst, mask) if keep]
    return cat_lists.apply(drop)

# ----------------------------- Save helpers ----------------------------------
def _write_ablation_csv(path: Path, scenario: str, delta: float, baseline: float, value: float, note: str, *, selfcheck: bool) -> Path:
    """Write a one-row CSV with columns: scenario, delta, baseline, value, note."""
    df = pd.DataFrame([{
        "scenario": scenario,
        "delta": _round(delta, 4),
        "baseline": _round(baseline, 4),
        "value": _round(value, 4),
        "note": note
    }])
    out = path.with_name(path.stem + "_selfcheck.csv") if selfcheck else path
    df.to_csv(out, index=False)
    print(f"✓ Artefact saved: {out}")
    return out

# ----------------------------- Orchestrator ----------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Run ablation experiments end-to-end.

    Options
    -------
    --selfcheck            Random subsample of rows; outputs *_selfcheck.csv only.
    --sample INT           Subsample size for self-check (default: min(150k, N)).
    --top-k-base INT       Baseline top-K for KPI (default from config or 30).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--top-k-base", type=int, default=TOPK_DEFAULT)
    args = p.parse_args(argv)

    t_all = _t0("--- Starting Step 22: Ablation Studies ---")

    # 1) Load & optional subsample
    df = _load_corpus()
    if args.selfcheck:
        n = args.sample or min(150_000, len(df))
        df = df.sample(n=n, replace=False, random_state=SEED).reset_index(drop=True)
        print(f"[SELF-CHECK] Random sample drawn: {len(df):,} rows (seed={SEED}).")

    # 2) Prepare protected + categories
    _prepare_protected(df)
    cat_series = _extract_categories(df)
    if cat_series is None or not cat_series.map(bool).any():
        print("--- Step 22: Skipped (no categories/tags found or all empty) ---")
        _tend("step22.total_runtime", t_all)
        return

    # 3) Baseline vocab + KPI
    t = _t0("Computing baseline KPI ...")
    top_base = _topk_vocab(cat_series, k=args.top_k_base)
    if not top_base:
        print("[INFO] Top-K vocabulary is empty; cannot compute KPI. Aborting.")
        _tend("step22.baseline", t)
        _tend("step22.total_runtime", t_all)
        return
    base_table = _bw_log2rr_by_category(df, cat_series, top=top_base)
    kpi_base = _kpi_mean_abs_log2rr(base_table)
    print(f"[BASE] top-K={args.top_k_base} | mean |log2RR| = {kpi_base:.4f}")
    _tend("step22.baseline", t)

    selfcheck = args.selfcheck
    results: List[Dict[str, object]] = []

    # 4) A) Lexicon OFF
    t = _t0("Ablation A: Lexicon OFF ...")
    cat_lexoff = _apply_lexicon_off(cat_series)
    table_lexoff = _bw_log2rr_by_category(df, cat_lexoff, top=top_base)
    kpi_lexoff = _kpi_mean_abs_log2rr(table_lexoff)
    delta_lexoff = kpi_lexoff - kpi_base
    outA = _write_ablation_csv(
        ABL_DIR / "22_rr_lexicon_off.csv",
        scenario="Lexicon OFF",
        delta=delta_lexoff,
        baseline=kpi_base,
        value=kpi_lexoff,
        note="Race-coded tokens removed before computing BW log2 RR.",
        selfcheck=selfcheck
    )
    results.append({"scenario": "Lexicon OFF", "delta": float(delta_lexoff), "baseline": float(kpi_base),
                    "value": float(kpi_lexoff), "path": str(outA)})
    _tend("step22.ablation_lexicon_off", t)

    # 5) B) Random noise (drop)
    for p_drop, tag in [(0.10, "10"), (0.25, "25"), (0.50, "50")]:
        t = _t0(f"Ablation B: Random category noise drop {int(p_drop*100)}% ...")
        noisy = _apply_noise_drop(cat_series, p_drop=p_drop)
        table_noisy = _bw_log2rr_by_category(df, noisy, top=top_base)
        kpi_noisy = _kpi_mean_abs_log2rr(table_noisy)
        delta_noisy = kpi_noisy - kpi_base
        outB = _write_ablation_csv(
            ABL_DIR / f"22_topcats_noise_{tag}.csv",
            scenario=f"Noise drop {int(p_drop*100)}%",
            delta=delta_noisy, baseline=kpi_base, value=kpi_noisy,
            note=f"Randomly dropped {int(p_drop*100)}% of (row,category) assignments.",
            selfcheck=selfcheck
        )
        results.append({"scenario": f"Noise {int(p_drop*100)}%", "delta": float(delta_noisy),
                        "baseline": float(kpi_base), "value": float(kpi_noisy), "path": str(outB)})
        _tend(f"step22.ablation_noise_{tag}", t)

    # 6) C) Top-K coverage variants
    for K in [10, 20, 30, 50]:
        t = _t0(f"Ablation C: Top-K coverage K={K} ...")
        topK = _topk_vocab(cat_series, k=K)
        tableK = _bw_log2rr_by_category(df, cat_series, top=topK)
        kpiK = _kpi_mean_abs_log2rr(tableK)
        deltaK = kpiK - kpi_base
        outC = _write_ablation_csv(
            ABL_DIR / f"22_topk_mass_{K}.csv",
            scenario=f"TopK {K}",
            delta=deltaK, baseline=kpi_base, value=kpiK,
            note=f"KPI computed over top-{K} categories by frequency.",
            selfcheck=selfcheck
        )
        results.append({"scenario": f"TopK {K}", "delta": float(deltaK),
                        "baseline": float(kpi_base), "value": float(kpiK), "path": str(outC)})
        _tend(f"step22.ablation_topk_{K}", t)

    # 7) Summary table (optional helper for dashboards/papers)
    summary_path = ABL_DIR / ("22_ablation_all_selfcheck.csv" if selfcheck else "22_ablation_all.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"✓ Artefact saved: {summary_path}")

    # 8) Qualitative readout (use the saved/recorded results — deterministic)
    print("\n--- Quick qualitative readout ---")
    if results:
        for row in sorted(results, key=lambda r: abs(r["delta"]), reverse=True)[:5]:
            print(f"• Largest |Δ| scenario: {row['scenario']:<12} Δ={row['delta']:+.4f}")
    print("*Note:* category totals can exceed N because categories are multi-label per item. "
          "Some titles are not in English; tags/categories (MPU) anchor interpretation.")

    # 9) Narrative
    t = _t0("Writing narrative ...")
    md = NARR_DIR / ("22_ablation_summary_selfcheck.md" if selfcheck else "22_ablation_summary.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join([
            "# 22 — Ablation Studies",
            f"- Baseline KPI (mean |log2 RR| over top-{args.top_k_base} categories): **{kpi_base:.4f}**.",
            "- Scenarios: Lexicon OFF, random category noise (10/25/50%), and Top-K coverage (10/20/30/50).",
            "- Deltas (scenario − baseline) are saved under outputs/ablation/ as one-row CSVs, plus a summary table.",
            "- Multi-label categories imply totals can exceed N. Non-English titles are common; tags/categories (MPU) anchor semantics.",
            f"- Seed={SEED} ensures reproducibility; self-check writes *_selfcheck.csv only.",
        ]) + "\n")
    print(f"✓ Narrative saved: {md}")
    _tend("step22.write_narrative", t)

    _tend("step22.total_runtime", t_all)
    print("--- Step 22: Ablation Studies Completed Successfully ---")

if __name__ == "__main__":
    main()
