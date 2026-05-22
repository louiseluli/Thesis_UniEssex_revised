# -*- coding: utf-8 -*-
"""
Step 30 — IPTW Causal Lens (Amateur → Rating)
=============================================

Purpose
-------
Estimate the causal effect of being in the 'Amateur' category (treatment) on
the outcome 'rating' using Inverse Probability of Treatment Weighting (IPTW).

What it does
------------
1) Loads corpus parquet and enforces required columns (robust path resolver).
2) Defines treatment T = 1 if categories include 'Amateur', else 0.
3) Builds covariates X from top-K categories (binary indicators) + light numerics.
4) Fits a logistic regression for propensity scores p(T=1|X).
5) Forms stabilized IPTW weights and estimates ATE with a 95% CI.
6) Runs a short sensitivity sweep over propensity clipping thresholds.
7) Saves CSV artefacts and dual-theme figures; prints overlap/outlier diagnostics.
8) Self-check uses a random sample and writes *_selfcheck artefacts only.

Interpretability note
---------------------
Some titles are not English; tags/categories often anchor semantics. We rely on
categories to build covariates and report overlap/weights to reason about bias.

CLI
---
# Full run (canonical artefacts):
python3 src/fairness/causal/30_psm_ipw.py

# Self-check (non-destructive):
python3 src/fairness/causal/30_psm_ipw.py --selfcheck --sample 80000

Artefacts (all start with 30_)
-----------------------------
- outputs/data/30_ate_rating.csv
- outputs/data/30_sensitivity.csv
- outputs/data/30_ipw_causal_lens_selfcheck.csv   (self-check only)
- outputs/figures/causal/30_propensity_by_treatment_{light,dark}.png
- outputs/figures/causal/30_weights_hist_{light,dark}.png
"""

from __future__ import annotations

# --- Imports ----------------------------------------------------
import argparse
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Project utils (config + plotting theme)
_ROOT = Path(__file__).resolve()
for _ in range(6):
    if (_ROOT / "src").is_dir():
        sys.path.append(str(_ROOT))
        break
    _ROOT = _ROOT.parent

from src.utils.theme_manager import load_config, plot_dual_theme


# --- 1) Config & paths --------------------------------------------------------
CONFIG: Dict[str, Any] = load_config() or {}
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))

PATHS = CONFIG.get("paths", {}) or {}
ROOT = Path(CONFIG.get("project", {}).get("root", Path(__file__).resolve().parents[3]))
DATA_DIR = Path(PATHS.get("data", ROOT / "outputs" / "data"))
FIG_DIR = Path(PATHS.get("figures", ROOT / "outputs" / "figures")) / "causal"

# Inputs (parquet is resolved robustly at runtime)
# Outputs (all 30_*)
ATE_CSV         = DATA_DIR / "30_ate_rating.csv"
SENS_CSV        = DATA_DIR / "30_sensitivity.csv"
SELF_CHECK_CSV  = DATA_DIR / "30_ipw_causal_lens_selfcheck.csv"

# self-check artefacts expected by the dashboard JSON writer
ATE_CSV_SC      = DATA_DIR / "30_ate_rating_selfcheck.csv"
SENS_CSV_SC     = DATA_DIR / "30_sensitivity_selfcheck.csv"

FIG_PROP_LIGHT  = FIG_DIR / "30_propensity_by_treatment_light.png"
FIG_PROP_DARK   = FIG_DIR / "30_propensity_by_treatment_dark.png"
FIG_WTS_LIGHT   = FIG_DIR / "30_weights_hist_light.png"
FIG_WTS_DARK    = FIG_DIR / "30_weights_hist_dark.png"

# Columns & task
CATS_COL  = "categories"
RATING    = "rating"    # outcome (float); will be displayed rounded to 1 d.p.
RATINGS_N = "ratings"   # count of ratings (int)
DURATION  = "duration"  # seconds (int)
POS_CLASS = "Amateur"   # treatment = Amateur category present


# --- 2) Lightweight timers ----------------------------------------------------
def _t0(msg: str) -> float:
    """Start a timer and print a standardized header."""
    t = time.perf_counter()
    print(msg)
    return t

def _tend(label: str, t0: float) -> None:
    """Finish a timer with a standardized [TIME] line."""
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")


# --- 3) Data helpers ----------------------------------------------------------
def _resolve_corpus_path() -> Path:
    """
    Resolve the corpus parquet path using config and robust fallbacks.

    Order:
      1) CONFIG['paths']['ml_corpus'] or CONFIG['paths']['corpus']
      2) outputs/data/ml_corpus.parquet
      3) outputs/data/01_ml_corpus.parquet
      4) glob: outputs/data/*ml_corpus*.parquet (latest by mtime)

    Raises
    ------
    FileNotFoundError if nothing is found.
    """
    cfg_paths = CONFIG.get("paths", {}) or {}
    for k in ("ml_corpus", "corpus"):
        p = cfg_paths.get(k)
        if p:
            pth = Path(p)
            if pth.exists():
                print(f"[INFO] Using corpus from CONFIG['paths']['{k}']: {pth}")
                return pth
    p2 = DATA_DIR / "ml_corpus.parquet"
    if p2.exists():
        print(f"[INFO] Using corpus at {p2}")
        return p2
    p3 = DATA_DIR / "01_ml_corpus.parquet"
    if p3.exists():
        print(f"[INFO] Using corpus at {p3}")
        return p3
    candidates = sorted(DATA_DIR.glob("*ml_corpus*.parquet"),
                        key=lambda x: x.stat().st_mtime if x.exists() else 0,
                        reverse=True)
    if candidates:
        print(f"[INFO] Using corpus via glob: {candidates[0]}")
        return candidates[0]
    tried = [str(cfg_paths.get("ml_corpus", "")), str(cfg_paths.get("corpus", "")),
             str(p2), str(p3), str(DATA_DIR / "*ml_corpus*.parquet")]
    raise FileNotFoundError("Could not resolve corpus parquet. Tried: " + " | ".join([t for t in tried if t]))


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist and types are reasonable.

    - rating: numeric (rounded to 1 decimal for printouts)
    - ratings: integer count fallback 0
    - duration: integer seconds fallback 0
    - categories: non-null string (comma-separated; falls back to tags/labels if present)
    """
    d = df.copy()

    # categories: use alternates if needed, then normalize to comma-separated strings
    if CATS_COL not in d.columns:
        for alt in ("tags", "labels", "category", "tag_list"):
            if alt in d.columns:
                s = d[alt]
                if s.dtype == "O":
                    d[CATS_COL] = s
                else:
                    d[CATS_COL] = s.astype(str)
                break
        else:
            d[CATS_COL] = ""

    # normalize categories to comma-separated string
    def _to_csv(x) -> str:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return ""
        if isinstance(x, (list, tuple, set)):
            return ",".join(str(t) for t in x if str(t).strip())
        return str(x)

    d[CATS_COL] = d[CATS_COL].apply(_to_csv).fillna("").astype(str)

    for c in (RATING, RATINGS_N, DURATION):
        if c not in d.columns:
            d[c] = 0
    d[RATING]    = pd.to_numeric(d[RATING], errors="coerce").fillna(0.0)
    d[RATINGS_N] = pd.to_numeric(d[RATINGS_N], errors="coerce").fillna(0).astype(int)
    d[DURATION]  = pd.to_numeric(d[DURATION], errors="coerce").fillna(0).astype(int)
    return d


def _load_corpus() -> pd.DataFrame:
    """
    Load corpus parquet (robust path resolution) and enforce required columns.

    Returns
    -------
    pd.DataFrame
        Canonical corpus frame.
    """
    corpus_path = _resolve_corpus_path()
    t0 = _t0(f"[READ] Parquet: {corpus_path}")
    df = pd.read_parquet(corpus_path)
    _tend("step30.load_corpus", t0)
    return _ensure_columns(df)


def _treatment_from_categories(cats_series: pd.Series) -> pd.Series:
    """
    Define treatment T=1 if 'Amateur' appears in comma-separated categories.
    Case-insensitive exact token match (comma boundaries).

    Returns
    -------
    pd.Series of {0,1}
    """
    cats = cats_series.fillna("").str.lower()
    mask = cats.str.contains(r"(?:^|,)\s*amateur\s*(?:,|$)", regex=True)
    return mask.astype(int)


def _top_categories(df: pd.DataFrame, k: int = 20) -> List[str]:
    """
    Get top-k categories by coverage.

    Returns
    -------
    list[str]
        Category names (lowercased tokens).
    """
    s = df[CATS_COL].fillna("").str.lower().str.split(",")
    flat = s.explode().str.strip()
    flat = flat[flat != ""]
    vc = flat.value_counts()
    tops = [c for c in vc.index[:k].tolist() if c != ""]
    return tops


# --- 4) Covariates & propensity ----------------------------------------------
def _make_covariates(df: pd.DataFrame, top_cats: List[str]) -> pd.DataFrame:
    """
    Build compact covariates: top-K category dummies (EXCLUDING the treatment label)
    + log1p(duration, ratings).

    Note
    ----
    We deliberately exclude the treatment-defining category (Amateur) to prevent
    perfect separation in the propensity model and to improve overlap diagnostics.
    """
    t0 = _t0("Building covariates X ...")
    out = pd.DataFrame(index=df.index)
    cats = df[CATS_COL].fillna("").str.lower()

    for c in top_cats:
        if c.strip().lower() == POS_CLASS.lower():
            continue  # avoid treatment leakage
        col = f"cat__{c.replace(' ', '_')}"
        pat = rf"(?:^|,)\s*{re.escape(c)}\s*(?:,|$)"  # exact token match
        out[col] = cats.str.contains(pat, regex=True)

    out["log1p_duration"] = np.log1p(df[DURATION].astype(float))
    out["log1p_ratings"]  = np.log1p(df[RATINGS_N].astype(float))

    out = out.astype(int)
    _tend("step30.covariates.build", t0)
    return out


def _fit_propensity(X: pd.DataFrame, T: np.ndarray) -> np.ndarray:
    """
    Fit logistic regression p = P(T=1|X).

    Returns
    -------
    np.ndarray
        Propensity scores in (0,1).
    """
    t0 = _t0("Fitting propensity model (logistic regression) ...")
    # Reproducible, simple baseline
    lr = LogisticRegression(solver="lbfgs", max_iter=300, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(X, T)
    p = lr.predict_proba(X)[:, 1]
    _tend("step30.propensity.fit", t0)
    return p


def _stabilized_weights(p: np.ndarray, T: np.ndarray, clip: float = 0.01) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute stabilized IPTW with clipping.

    w_i = [T_i * s1 / p_i] + [(1-T_i) * s0 / (1-p_i)]
    where s1 = P(T=1), s0 = P(T=0).

    Returns
    -------
    (weights, stats)
        weights : np.ndarray
        stats   : dict with diagnostics
    """
    t0 = _t0(f"Computing stabilized IPTW (clip={clip:.3f}) ...")
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=int)
    s1 = float(T.mean())
    s0 = 1.0 - s1

    p_clipped = p.clip(clip, 1 - clip)
    n_clip_lo = int((p < clip).sum())
    n_clip_hi = int((p > 1 - clip).sum())

    w = np.where(T == 1, s1 / p_clipped, s0 / (1 - p_clipped))
    w = np.asarray(w, dtype=float)

    # Outlier weights diagnostics
    w99 = float(np.quantile(w, 0.99))
    n_hi = int((w > w99).sum())

    stats = dict(
        s1=round(s1, 4),
        s0=round(s0, 4),
        clip=clip,
        n_clip_lo=n_clip_lo,
        n_clip_hi=n_clip_hi,
        w99=round(w99, 3),
        n_above_w99=n_hi,
    )
    _tend("step30.weights.compute", t0)
    return w, stats


def _ate_ipw(y: np.ndarray, T: np.ndarray, w: np.ndarray) -> Tuple[float, float, float]:
    """
    Horvitz–Thompson ATE with a simple large-sample SE and 95% CI.

    Returns
    -------
    (ate, lo95, hi95)
    """
    t0 = _t0("Estimating ATE with IPTW ...")
    y = np.asarray(y, dtype=float)
    T = np.asarray(T, dtype=int)
    w = np.asarray(w, dtype=float)

    # Weighted means for each arm
    wt1 = w * (T == 1)
    wt0 = w * (T == 0)
    mu1 = (wt1 * y).sum() / max(wt1.sum(), 1e-12)
    mu0 = (wt0 * y).sum() / max(wt0.sum(), 1e-12)
    ate = mu1 - mu0

    # crude SE via weighted variance / n_eff
    def _wvar(y_, w_):
        ybar = (w_ * y_).sum() / max(w_.sum(), 1e-12)
        return ((w_ * (y_ - ybar) ** 2).sum() / max(w_.sum(), 1e-12))

    n_eff = (w.sum() ** 2) / max((w ** 2).sum(), 1e-12)
    var = (_wvar(y[T == 1], wt1[T == 1]) / max(wt1[T == 1].sum(), 1e-12)) + \
          (_wvar(y[T == 0], wt0[T == 0]) / max(wt0[T == 0].sum(), 1e-12))
    se  = float(np.sqrt(max(var, 0.0) / max(n_eff, 1.0)))
    lo, hi = ate - 1.96 * se, ate + 1.96 * se

    _tend("step30.ate.estimate", t0)
    return float(ate), float(lo), float(hi)


# --- 5) Plots -----------------------------------------------------------------
@plot_dual_theme(section="fairness")
def _plot_propensity_by_T(p: np.ndarray, T: np.ndarray, title: str, ax=None, **kwargs):
    """Propensity score histogram split by treatment (overlap check)."""
    ax.hist(p[T == 1], bins=40, alpha=0.7, label="Treated (Amateur)")
    ax.hist(p[T == 0], bins=40, alpha=0.7, label="Control")
    ax.set_title(title)
    ax.set_xlabel("Propensity p(T=1|X)")
    ax.set_ylabel("Count")
    ax.legend(loc="best")

@plot_dual_theme(section="fairness")
def _plot_weights_hist(w: np.ndarray, title: str, ax=None, **kwargs):
    """Histogram of IPTW weights (extremes indicate limited overlap)."""
    ax.hist(w, bins=40)
    ax.set_title(title)
    ax.set_xlabel("Stabilized weight")
    ax.set_ylabel("Count")


# --- 6) Sensitivity -----------------------------------------------------------
def _sensitivity(df: pd.DataFrame, X: pd.DataFrame, T: np.ndarray, y: np.ndarray,
                 clip_grid: List[float]) -> pd.DataFrame:
    """
    Simple sensitivity sweep over clipping thresholds.

    Returns
    -------
    pd.DataFrame
        Columns: clip, ATE, lo95, hi95, n_above_w99, n_clip_lo, n_clip_hi
    """
    rows = []
    for clip in clip_grid:
        p = _fit_propensity(X, T)
        w, stats = _stabilized_weights(p, T, clip=clip)
        ate, lo, hi = _ate_ipw(y, T, w)
        rows.append({
            "clip": clip,
            "ATE": round(ate, 3),
            "lo95": round(lo, 3),
            "hi95": round(hi, 3),
            "n_above_w99": int(stats["n_above_w99"]),
            "n_clip_lo": int(stats["n_clip_lo"]),
            "n_clip_hi": int(stats["n_clip_hi"]),
        })
    sens = pd.DataFrame(rows)
    return sens


# --- 7) Self-check ------------------------------------------------------------
def _self_check(df: pd.DataFrame, *, sample: int, seed: int) -> pd.DataFrame:
    """
    Tiny self-check on a random sample from the real parquet (non-destructive).
    Writes `30_ipw_causal_lens_selfcheck.csv`.

    Returns
    -------
    pd.DataFrame with a single row summarising the self-check.
    """
    t0 = _t0(f"[SELF-CHECK] Sampling {sample:,} rows (seed={seed}) ...")
    d = df.sample(n=sample, random_state=seed, replace=False).reset_index(drop=True)
    _tend("step30.selfcheck.sample", t0)

    # pipeline
    top = _top_categories(d, k=20)
    X   = _make_covariates(d, top)
    T   = _treatment_from_categories(d[CATS_COL]).to_numpy()
    y   = d[RATING].to_numpy()

    p   = _fit_propensity(X, T)
    w, stats = _stabilized_weights(p, T, clip=0.01)
    ate, lo, hi = _ate_ipw(y, T, w)

    # simple overlap PASS if >= 80% of propensities in [0.1, 0.9]
    overlap = float(((p >= 0.1) & (p <= 0.9)).mean())
    passed = overlap >= 0.80

    row = pd.DataFrame([{
        "N": int(len(d)),
        "ATE": round(ate, 3),
        "lo95": round(lo, 3),
        "hi95": round(hi, 3),
        "overlap_share_0.1_0.9": round(overlap, 3),
        "pass": bool(passed),
        "w99": stats["w99"],
        "n_above_w99": stats["n_above_w99"],
        "n_clip_lo": stats["n_clip_lo"],
        "n_clip_hi": stats["n_clip_hi"],
    }])
    SELF_CHECK_CSV.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(SELF_CHECK_CSV, index=False)

    # also write the artefacts the dashboard expects in self-check mode
    sens_sc = _sensitivity(d, X, T, y, clip_grid=[0.01, 0.02, 0.05])
    ate_sc  = pd.DataFrame([{
        "N": int(len(d)),
        "ATE_rating": round(ate, 3),
        "lo95": round(lo, 3),
        "hi95": round(hi, 3),
        "clip": 0.01,
        "overlap_share_0.1_0.9": round(overlap, 3),
        "w99": stats["w99"],
        "n_above_w99": stats["n_above_w99"],
        "n_clip_lo": stats["n_clip_lo"],
        "n_clip_hi": stats["n_clip_hi"],
    }])

    ate_sc.to_csv(ATE_CSV_SC, index=False)
    sens_sc.to_csv(SENS_CSV_SC, index=False)

    print(f"✓ Self-check saved: {SELF_CHECK_CSV.resolve()}")
    print(f"✓ Self-check saved: {ATE_CSV_SC.resolve()}")
    print(f"✓ Self-check saved: {SENS_CSV_SC.resolve()}")

    _tend("step30.selfcheck.write", t0)
    return row


# --- 8) Main ------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Run the IPTW causal lens for the rating outcome.

    Options
    -------
    --selfcheck         Run on a random sample and write *_selfcheck only.
    --sample INT        Number of rows for self-check (default: 80,000 if available).
    --topk INT          Number of top categories to include as covariates (default: 20).
    --clip FLOAT        Clipping threshold for main ATE (default: 0.01).
    """
    t_all = time.perf_counter()
    print("--- Starting 30: IPTW Causal Lens (Rating) ---")

    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--clip", type=float, default=0.01)
    args = p.parse_args(argv)

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load corpus
    df = _load_corpus()
    total = len(df)
    print(f"[STATS] Total videos available: {total:,}")
    print("[NOTE] Titles may be non-English; categories anchor semantics here.")

    if args.selfcheck:
        n = args.sample or min(80_000, total)
        _self_check(df, sample=n, seed=SEED)
        _tend("step30.total", t_all)
        print("\n--- Step 30: IPTW Causal Lens (self-check) Completed ---")
        return

    # ===== Full run =====
    # 1) Treatment & covariates
    top_cats = _top_categories(df, k=args.topk)
    X        = _make_covariates(df, top_cats)
    T        = _treatment_from_categories(df[CATS_COL]).to_numpy()
    y        = df[RATING].to_numpy()

    # 2) Propensity, weights, ATE
    p_hat    = _fit_propensity(X, T)
    w, wstats = _stabilized_weights(p_hat, T, clip=args.clip)
    ate, lo, hi = _ate_ipw(y, T, w)

    # 3) Overlap diagnostics
    overlap_share = float(((p_hat >= 0.1) & (p_hat <= 0.9)).mean())
    print(f"[OVERLAP] Share of propensities in [0.1, 0.9]: {overlap_share:.3f}")
    print(f"[WEIGHTS] 99th percentile ≈ {wstats['w99']}, "
        f"above 99th: {wstats['n_above_w99']:,}; clipped low/high: "
        f"{wstats['n_clip_lo']:,}/{wstats['n_clip_hi']:,}")

    # 4) Sensitivity sweep
    sens = _sensitivity(df, X, T, y, clip_grid=[0.01, 0.02, 0.05])

    # 5) Save artefacts
    ate_row = pd.DataFrame([{
        "N": int(len(df)),
        "ATE_rating": round(ate, 3),
        "lo95": round(lo, 3),
        "hi95": round(hi, 3),
        "clip": args.clip,
        "overlap_share_0.1_0.9": round(overlap_share, 3),
        "w99": wstats["w99"],
        "n_above_w99": wstats["n_above_w99"],
        "n_clip_lo": wstats["n_clip_lo"],
        "n_clip_hi": wstats["n_clip_hi"],
    }])
    ate_row.to_csv(ATE_CSV, index=False)
    sens.to_csv(SENS_CSV, index=False)
    print(f"✓ Artefact saved: {ATE_CSV.resolve()}")
    print(f"✓ Artefact saved: {SENS_CSV.resolve()}")

    # 6) Plots (dual theme via theme_manager)
    _plot_propensity_by_T(
        p=p_hat, T=T,
        title="Propensity by Treatment (Amateur vs Control)",
        save_path=str(FIG_DIR / "30_propensity_by_treatment"),
        figsize=(9, 6),
    )
    _plot_weights_hist(
        w=w,
        title="Distribution of Stabilized IPTW Weights",
        save_path=str(FIG_DIR / "30_weights_hist"),
        figsize=(9, 6),
    )
    print(f"✓ Figures saved: {FIG_PROP_LIGHT.name}, {FIG_PROP_DARK.name}, "
          f"{FIG_WTS_LIGHT.name}, {FIG_WTS_DARK.name}")

    _tend("step30.total", t_all)
    print("\n--- Step 30: IPTW Causal Lens Completed Successfully ---")


if __name__ == "__main__":
    main()
