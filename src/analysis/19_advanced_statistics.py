#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19_advanced_statistics.py
=========================

Purpose
-------
Black-women–centric, uncertainty-aware analytics with interpretable outputs:
A) Cliff’s delta (rank-based) + bootstrap CIs for engagement metrics
B) Temporal slopes (views/day, rating) for BW vs Others + bootstrap CIs and slope-gap CI
C) Category divergence by group via KL (top-K categories)
D) Optional: Harm relative risk if harm-by-group artefact exists

What it does
------------
1) Loads config + corpus (fast/robust path) and derives common fields.
2) Computes effect sizes and uncertainty via bootstrapping.
3) Fits temporal slopes per group and reports CI + gap CI.
4) Compares category distributions vs global using KL divergence (top-K).
5) If available, computes harm relative risks for BW vs Others.
6) Saves CSV/TEX/PNG + narrative, prints compact qualitative readouts.
7) Adds timers, self-check (random subsample, non-destructive *_selfcheck suffix).

Interpretability & language notes
---------------------------------
- Some titles are non-English; tags/categories help anchor semantics (MPU).
- Category analyses are multi-label → totals can exceed N by design.
- Rounding: years as integers; ratings shown with one decimal place in readouts;
  CSV/TEX rounded sensibly for interpretability.

CLI
---
# Full run (canonical artefacts):
python -m src.analysis.19_advanced_statistics

# Self-check (random subsample; safe, non-destructive):
python -m src.analysis.19_advanced_statistics --selfcheck --sample 120000 --top-k 30
"""

from __future__ import annotations

# --- Imports (keep at top) ---------------------------------------------------
import sys
import time
from ast import literal_eval
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project utils (config); theme_manager.load_config prints its own [TIME]
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config  # noqa: E402

# --- Config & paths ----------------------------------------------------------
_t0_cfg = time.perf_counter()
CONFIG = load_config()
print(f"[TIME] theme_manager.load_config: {_t0_cfg and (time.perf_counter() - _t0_cfg):.2f}s")

SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
np.random.seed(SEED)

ROOT      = Path(CONFIG["project"]["root"])
DATA_DIR  = Path(CONFIG["paths"]["data"])         # outputs/data
FIG_DIR   = Path(CONFIG["paths"]["figures"])      # outputs/figures
TABLE_DIR = ROOT / "dissertation" / "auto_tables"
NARR_DIR  = Path(CONFIG["paths"]["narratives"]) / "automated"
for d in (DATA_DIR, FIG_DIR, TABLE_DIR, NARR_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Accept several canonical names (first win)
CORPUS_CANDIDATES = [
    DATA_DIR / "01_ml_corpus.parquet",
    DATA_DIR / "ml_corpus.parquet",
    DATA_DIR / "01_ml_corpus.snappy.parquet",
]

HARM_BY_GROUP  = DATA_DIR / "harm_category_by_group.csv"
TOP_K_DEFAULT  = 30
N_BOOT         = 1000
ALPHA          = 0.95
EPS            = 1e-12

# --- Timers ------------------------------------------------------------------
def _t0(msg: str) -> float:
    """
    Start a timer and print a standardized header.

    Parameters
    ----------
    msg : str
        Message printed before timing starts.

    Returns
    -------
    float
        perf_counter() start time.
    """
    print(msg)
    return time.perf_counter()

def _tend(label: str, t0: float) -> None:
    """
    Stop a timer and print a standardized [TIME] line.

    Parameters
    ----------
    label : str
        Short label describing the timed block.
    t0 : float
        Start time from _t0.
    """
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")

# --- Theme (Matplotlib only) -------------------------------------------------
def _set_mpl_theme(dark: bool) -> None:
    """
    Apply a minimal Matplotlib theme.
    """
    plt.rcParams.update({
        "figure.facecolor": "black" if dark else "white",
        "axes.facecolor": "black" if dark else "white",
        "axes.edgecolor": "white" if dark else "black",
        "axes.labelcolor": "white" if dark else "black",
        "xtick.color": "white" if dark else "black",
        "ytick.color": "white" if dark else "black",
        "text.color": "white" if dark else "black",
        "savefig.facecolor": "black" if dark else "white",
        "grid.color": "gray",
        "grid.alpha": 0.25,
    })

# --- Small utils -------------------------------------------------------------
def _round(x: float, k: int = 3) -> float:
    """
    Round floats safely; returns x unchanged on failure.

    Parameters
    ----------
    x : float
    k : int

    Returns
    -------
    float
    """
    try:
        return round(float(x), k)
    except Exception:
        return x

def _round1(x: float) -> float:
    """
    Round floats to 1 decimal place (for ratings shown in readouts).

    Parameters
    ----------
    x : float

    Returns
    -------
    float
    """
    try:
        return round(float(x), 1)
    except Exception:
        return x

def _first_existing(paths: List[Path]) -> Optional[Path]:
    """
    Return the first existing path among candidates.

    Parameters
    ----------
    paths : List[Path]

    Returns
    -------
    Optional[Path]
    """
    for p in paths:
        if p.exists():
            return p
    return None

# --- IO + preparation --------------------------------------------------------
def _load_corpus_fast() -> pd.DataFrame:
    """
    Load the corpus parquet with an attempt to read minimal columns.
    Falls back to a full read if the subset fails (schema variance).
    Accepts several canonical names; picks the first that exists.
    """
    path = _first_existing(CORPUS_CANDIDATES)
    if path is None:
        exp = ", ".join(p.name for p in CORPUS_CANDIDATES)
        raise FileNotFoundError(f"Corpus not found under {DATA_DIR}. Expected one of: {exp}")
    t0 = _t0(f"[READ] {path} (fast)")
    minimal = [
        # dates
        "publish_date","upload_date","published_at","date",
        # engagement
        "views","view_count","views_count","play_count",
        "rating_clean","rating","rating_mean","average_rating","avg_rating","score",
        "ratings","rating_count","num_ratings","n_ratings","votes",
        # protected + categories
        "race_ethnicity","gender","categories","tags","category","tag_list","labels",
        # helpful misc (optional)
        "title"
    ]
    try:
        df = pd.read_parquet(path, columns=minimal)
    except Exception:
        df = pd.read_parquet(path)
    _tend("step19.load_corpus", t0)
    return df

def _ensure_datetime(series: pd.Series) -> pd.Series:
    """
    Convert a series to timezone-naive pandas datetime; coerce invalids to NaT.

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    pd.Series
    """
    s = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return s.dt.tz_convert(None)
    except Exception:
        return s

def _prepare_columns(df: pd.DataFrame) -> None:
    """
    Derive unified time/engagement/protected fields in-place:
      - _publish_dt (datetime), _publish_year (int), _age_days (int)
      - _views, _rating, _rating_n
      - _views_per_day, _ratings_per_day
      - tidy 'race_ethnicity'/'gender' from one-hot if needed

    Parameters
    ----------
    df : pd.DataFrame
    """
    t0 = _t0("Preparing derived columns ...")
    # publish date
    publish_col = next((c for c in ["publish_date","upload_date","published_at","date"] if c in df.columns), None)
    if publish_col is not None:
        dt = _ensure_datetime(df[publish_col])
        df["_publish_dt"] = dt
        df["_publish_year"] = dt.dt.year.astype("Int64")
        today = pd.Timestamp.utcnow().tz_convert(None)
        df["_age_days"] = (today - dt).dt.days.astype("Int64")
    else:
        df["_publish_dt"] = pd.NaT
        df["_publish_year"] = pd.Series([pd.NA]*len(df), dtype="Int64")
        df["_age_days"] = pd.Series([pd.NA]*len(df), dtype="Int64")

    # numeric fields (robust to schema variants)
    def _num(colnames, default=np.nan):
        c = next((c for c in colnames if c in df.columns), None)
        return pd.to_numeric(df[c], errors="coerce") if c else default
    df["_views"]    = _num(["views","view_count","views_count","play_count"])
    df["_rating"]   = _num(["rating","rating_mean","average_rating","avg_rating","score"])
    df["_rating_n"] = _num(["ratings","rating_count","num_ratings","n_ratings","votes"])

    # protected (derive from one-hot if needed)
    def _derive_categorical_from_onehot(df: pd.DataFrame, prefix: str, out_col: str, fallback: str = "unknown") -> Optional[str]:
        pref = f"{prefix}_"
        onehot_cols = [c for c in df.columns if c.lower().startswith(pref)]
        if not onehot_cols:
            return None
        labels = [c[len(pref):].lower() for c in onehot_cols]
        mixed_token = "mixed_or_other" if any(l == "mixed_or_other" for l in labels) else "mixed"
        oh = df[onehot_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
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
        _derive_categorical_from_onehot(df, "race_ethnicity", "race_ethnicity")
    if "gender" not in df.columns:
        _derive_categorical_from_onehot(df, "gender", "gender")
    if "gender" in df.columns:
        def _norm_gender_label(x: str) -> str:
            x = (x or "").lower()
            if x in {"female","woman","women","cis_female","cis_woman"}: return "female"
            if x in {"male","man","men","cis_male","cis_man"}: return "male"
            return x or "unknown"
        df["gender"] = df["gender"].map(_norm_gender_label)

    # engagement per day
    age = pd.to_numeric(df["_age_days"], errors="coerce")
    age_clamped = age.clip(lower=1)
    df["_views_per_day"]   = pd.to_numeric(df["_views"], errors="coerce") / age_clamped
    df["_ratings_per_day"] = pd.to_numeric(df["_rating_n"], errors="coerce") / age_clamped
    _tend("step19.prepare_columns", t0)

# --- Statistics helpers ------------------------------------------------------
def cliffs_delta_fast(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cliff's delta via ranks (Mann–Whitney U equivalence).
    Returns δ in [-1, 1], positive if 'a' tends to be larger than 'b'.

    Parameters
    ----------
    a, b : np.ndarray

    Returns
    -------
    float
    """
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    m, n = a.size, b.size
    if m == 0 or n == 0:
        return float("nan")
    x = np.concatenate([a, b])
    ranks = pd.Series(x).rank(method="average").to_numpy()
    Ra = ranks[:m].sum()  # 'a' placed first in concat
    U = Ra - m*(m+1)/2.0
    delta = 2.0*U/(m*n) - 1.0
    return float(delta)

def bootstrap_ci_delta(a: np.ndarray, b: np.ndarray, n: int = N_BOOT, alpha: float = ALPHA) -> Tuple[float, float, float]:
    """
    Bootstrap CI for Cliff's delta between arrays a and b.
    Returns (point, lo, hi).
    """
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    ia = np.arange(a.size); ib = np.arange(b.size)
    vals = []
    for _ in range(n):
        sa = a[np.random.choice(ia, size=ia.size, replace=True)]
        sb = b[np.random.choice(ib, size=ib.size, replace=True)]
        vals.append(cliffs_delta_fast(sa, sb))
    vals = np.array(vals, dtype=float)
    point = cliffs_delta_fast(a, b)
    lo, hi = np.quantile(vals, [(1-alpha)/2.0, 1-(1-alpha)/2.0])
    return float(point), float(lo), float(hi)

def slopes_bootstrap(x: np.ndarray, y: np.ndarray, n: int = N_BOOT) -> np.ndarray:
    """
    Bootstrap slopes for simple linear regression y ~ x (degree=1).
    Returns an array of slope samples (length n). Empty if insufficient data.
    """
    m = (~np.isnan(x)) & (~np.isnan(y))
    x = x[m].astype(float); y = y[m].astype(float)
    if x.size < 3:
        return np.array([], dtype=float)
    idx = np.arange(x.size)
    out = np.empty(n, dtype=float)
    for i in range(n):
        s = np.random.choice(idx, size=idx.size, replace=True)
        out[i] = float(np.polyfit(x[s], y[s], 1)[0])
    return out

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL(p || q) with epsilon-smoothing and normalized inputs.
    """
    p = np.clip(p, EPS, None); q = np.clip(q, EPS, None)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

# --- Category parsing --------------------------------------------------------
def _parse_listish(x) -> List[str]:
    """
    Parse list-like strings/structures to a cleaned list of lowercase tokens.
    Handles python-list-in-string, comma/pipe-separated, and true sequences.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        vals = list(x)
    else:
        s = str(x).strip()
        if not s: return []
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
        if t and t not in {"nan","none","null"}:
            out.append(t)
    return out

def _extract_categories(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Return a Series of list[str] categories per row if a category/tags field exists.
    Preferred order: config-declared names first, then common fallbacks.
    """
    cols_cfg: Dict[str, str] = CONFIG.get("columns", {})
    prefer = [cols_cfg.get(k) for k in ("categories","tags","labels") if cols_cfg.get(k)]
    fallbacks = ["categories","tags","category","tag_list","labels"]
    for c in [*prefer, *fallbacks]:
        if c and c in df.columns:
            return df[c].apply(_parse_listish)
    return None

# --- Analyses ----------------------------------------------------------------
def effect_sizes_bw_vs_others(df: pd.DataFrame, *, suffix: str = "") -> pd.DataFrame:
    """
    Compute Cliff's δ (BW minus Others) with bootstrap CIs for:
      - views/day, ratings/day, rating
    Save CSV/TEX and a CI bar chart (dual theme).
    """
    t0 = _t0("Effect sizes (Cliff's δ) for BW vs Others ...")
    if "race_ethnicity" not in df.columns or "gender" not in df.columns:
        print("[INFO] Missing race_ethnicity or gender; skipping BW effect sizes.")
        return pd.DataFrame()

    is_bw = (df["race_ethnicity"].astype(str).str.lower()=="black") & (df["gender"].astype(str).str.lower()=="female")
    metrics = {
        "views_per_day": df.loc[is_bw, "_views_per_day"].to_numpy(float),
        "ratings_per_day": df.loc[is_bw, "_ratings_per_day"].to_numpy(float),
        "rating": df.loc[is_bw, "_rating"].to_numpy(float),
    }
    others = {
        "views_per_day": df.loc[~is_bw, "_views_per_day"].to_numpy(float),
        "ratings_per_day": df.loc[~is_bw, "_ratings_per_day"].to_numpy(float),
        "rating": df.loc[~is_bw, "_rating"].to_numpy(float),
    }

    rows = []
    for name in ["views_per_day","ratings_per_day","rating"]:
        point, lo, hi = bootstrap_ci_delta(metrics[name], others[name], n=N_BOOT, alpha=ALPHA)
        rows.append({
            "metric": name,
            "cliffs_delta": _round(point, 4),
            "ci_lo": _round(lo, 4),
            "ci_hi": _round(hi, 4),
            "n_bw": int(np.isfinite(metrics[name]).sum()),
            "n_others": int(np.isfinite(others[name]).sum()),
        })
    out = pd.DataFrame(rows)

    csv = DATA_DIR / f"19_bw_effect_sizes{suffix}.csv"
    tex = TABLE_DIR / f"19_bw_effect_sizes{suffix}.tex"
    out.to_csv(csv, index=False); print(f"✓ Artefact saved: {csv}")
    try:
        with open(tex, "w", encoding="utf-8") as f:
            f.write(out.to_latex(index=False))
        print(f"✓ Artefact saved: {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # CI bar figure
    for dark in (True, False):
        _set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(8.2, 5))
        xs = np.arange(len(out))
        y = out["cliffs_delta"].astype(float).to_numpy()
        ylo = y - out["ci_lo"].astype(float).to_numpy()
        yhi = out["ci_hi"].astype(float).to_numpy() - y
        ax.bar(xs, y, yerr=[ylo, yhi], capsize=5)
        ax.axhline(0.0, ls="--", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(out["metric"].tolist())
        ax.set_ylabel("Cliff's δ (BW − Others)")
        ax.set_title("Effect sizes with 95% bootstrap CI")
        ax.grid(True, axis="y", alpha=0.3)
        f = FIG_DIR / f"19_bw_effectsizes_{'dark' if dark else 'light'}{suffix}.png"
        fig.tight_layout(); fig.savefig(f, dpi=200); plt.close(fig)
        print(f"✓ Artefact saved: {f}")
    _tend("step19.effect_sizes", t0)
    return out

def temporal_slopes_bw_vs_others(df: pd.DataFrame, *, suffix: str = "") -> pd.DataFrame:
    """
    Fit yearly slopes for BW vs Others for:
      - views/day (per-year trend)
      - rating (per-year trend)
    Report bootstrap CIs for each slope and a CI for the slope gap (BW−Others).
    Save CSV/TEX and compare bars plot (dual theme).
    """
    t0 = _t0("Temporal slopes (BW vs Others) ...")
    if "race_ethnicity" not in df.columns or "gender" not in df.columns:
        print("[INFO] Missing race_ethnicity or gender; skipping temporal slopes.")
        return pd.DataFrame()
    m = df["_publish_year"].notna()
    if not m.any():
        print("[INFO] No publish years; skipping temporal slopes.")
        return pd.DataFrame()

    is_bw = (df["race_ethnicity"].astype(str).str.lower()=="black") & (df["gender"].astype(str).str.lower()=="female")
    rows = []
    for metric, label in [("_views_per_day","views_per_day"), ("_rating","rating")]:
        a_y = df.loc[is_bw & m, "_publish_year"].astype(float).to_numpy()
        a_v = df.loc[is_bw & m, metric].astype(float).to_numpy()
        b_y = df.loc[~is_bw & m, "_publish_year"].astype(float).to_numpy()
        b_v = df.loc[~is_bw & m, metric].astype(float).to_numpy()

        boot_a = slopes_bootstrap(a_y, a_v, n=N_BOOT)
        boot_b = slopes_bootstrap(b_y, b_v, n=N_BOOT)
        s_a = float(np.polyfit(a_y[~np.isnan(a_v)], a_v[~np.isnan(a_v)], 1)[0]) if a_y.size>2 else np.nan
        s_b = float(np.polyfit(b_y[~np.isnan(b_v)], b_v[~np.isnan(b_v)], 1)[0]) if b_y.size>2 else np.nan

        lo_a, hi_a = (np.quantile(boot_a, [(1-ALPHA)/2.0, 1-(1-ALPHA)/2.0]) if boot_a.size else (np.nan, np.nan))
        lo_b, hi_b = (np.quantile(boot_b, [(1-ALPHA)/2.0, 1-(1-ALPHA)/2.0]) if boot_b.size else (np.nan, np.nan))

        k = int(min(len(boot_a), len(boot_b)))
        gap = s_a - s_b
        gap_samples = (np.random.choice(boot_a, size=k, replace=True) - np.random.choice(boot_b, size=k, replace=True)) if k>0 else np.array([])
        lo_g, hi_g = (np.quantile(gap_samples, [(1-ALPHA)/2.0, 1-(1-ALPHA)/2.0]) if gap_samples.size else (np.nan, np.nan))

        rows.append({
            "metric": label,
            "slope_bw": _round(s_a, 4), "slope_bw_lo": _round(lo_a, 4), "slope_bw_hi": _round(hi_a, 4),
            "slope_others": _round(s_b, 4), "slope_others_lo": _round(lo_b, 4), "slope_others_hi": _round(hi_b, 4),
            "slope_gap_bw_minus_others": _round(gap, 4),
            "slope_gap_lo": _round(float(lo_g), 4), "slope_gap_hi": _round(float(hi_g), 4),
            "n_bw": int(np.isfinite(a_v).sum()), "n_others": int(np.isfinite(b_v).sum()),
        })

    out = pd.DataFrame(rows)

    csv = DATA_DIR / f"19_trend_slopes{suffix}.csv"
    tex = TABLE_DIR / f"19_trend_slopes{suffix}.tex"
    out.to_csv(csv, index=False); print(f"✓ Artefact saved: {csv}")
    try:
        with open(tex, "w", encoding="utf-8") as f:
            f.write(out.to_latex(index=False))
        print(f"✓ Artefact saved: {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # BW vs Others slope bars
    for dark in (True, False):
        _set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(8.6, 5))
        xs = np.arange(len(out))
        width = 0.35
        s_bw = out["slope_bw"].astype(float).to_numpy()
        s_ot = out["slope_others"].astype(float).to_numpy()
        eb_bw = [s_bw - out["slope_bw_lo"].astype(float), out["slope_bw_hi"].astype(float) - s_bw]
        eb_ot = [s_ot - out["slope_others_lo"].astype(float), out["slope_others_hi"].astype(float) - s_ot]
        ax.bar(xs - width/2, s_bw, width=width, yerr=eb_bw, capsize=5, label="Black women")
        ax.bar(xs + width/2, s_ot, width=width, yerr=eb_ot, capsize=5, label="Others")
        ax.axhline(0.0, ls="--", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(out["metric"].tolist())
        ax.set_ylabel("Yearly slope")
        ax.set_title("Temporal slopes (95% bootstrap CI)")
        ax.legend(frameon=False); ax.grid(True, axis="y", alpha=0.3)
        f = FIG_DIR / f"19_slope_bw_vs_others_{'dark' if dark else 'light'}{suffix}.png"
        fig.tight_layout(); fig.savefig(f, dpi=200); plt.close(fig)
        print(f"✓ Artefact saved: {f}")
    _tend("step19.temporal_slopes", t0)
    return out

def category_divergence_by_group(df: pd.DataFrame, *, top_k: int, suffix: str = "") -> pd.DataFrame:
    """
    Compute KL divergence between each group's category distribution and the global distribution,
    over the top-K categories. Save CSV/TEX and a bar plot (dual theme).
    """
    t0 = _t0("Category divergence (KL vs global) by group ...")
    cat_series = _extract_categories(df)
    if cat_series is None:
        print("[INFO] No categories/tags; skipping category divergence.")
        _tend("step19.category_divergence", t0)
        return pd.DataFrame()

    # vocab (top-K)
    counts = {}
    for lst in cat_series:
        for t in lst:
            counts[t] = counts.get(t, 0) + 1
    vocab = [w for w,_ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
    if not vocab:
        print("[INFO] Empty vocab; skipping category divergence.")
        _tend("step19.category_divergence", t0)
        return pd.DataFrame()

    # global distribution
    g = np.zeros(len(vocab), float)
    for lst in cat_series:
        for t in lst:
            if t in vocab:
                g[vocab.index(t)] += 1
    gp = (g + EPS) / (g.sum() + EPS*len(vocab))

    # choose grouping column
    group_col = "race_ethnicity" if "race_ethnicity" in df.columns else ("gender" if "gender" in df.columns else None)
    if group_col is None:
        print("[INFO] No group column; skipping category divergence.")
        _tend("step19.category_divergence", t0)
        return pd.DataFrame()

    rows = []
    for gname, sub in df.groupby(group_col):
        s = np.zeros(len(vocab), float)
        subs = cat_series.loc[sub.index]
        for lst in subs:
            for t in lst:
                if t in vocab:
                    s[vocab.index(t)] += 1
        sp = (s + EPS) / (s.sum() + EPS*len(vocab))
        D = kl_divergence(sp, gp)
        rows.append({"group": str(gname), "kl_vs_global": _round(D, 4), "tokens": int(s.sum())})

    out = pd.DataFrame(rows).sort_values("kl_vs_global", ascending=False)

    csv = DATA_DIR / f"19_category_divergence{suffix}.csv"
    tex = TABLE_DIR / f"19_category_divergence{suffix}.tex"
    out.to_csv(csv, index=False); print(f"✓ Artefact saved: {csv}")
    try:
        with open(tex, "w", encoding="utf-8") as f:
            f.write(out.to_latex(index=False))
        print(f"✓ Artefact saved: {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    for dark in (True, False):
        _set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(out["group"], out["kl_vs_global"].astype(float))
        ax.set_ylabel("KL divergence vs global")
        ax.set_title(f"Category divergence by {group_col} (top-{top_k})")
        ax.grid(True, axis="y", alpha=0.3)
        f = FIG_DIR / f"19_category_kl_bar_{'dark' if dark else 'light'}{suffix}.png"
        fig.tight_layout(); fig.savefig(f, dpi=200); plt.close(fig)
        print(f"✓ Artefact saved: {f}")
    _tend("step19.category_divergence", t0)
    return out

def harm_relative_risk(*, suffix: str = "") -> pd.DataFrame:
    """
    Compute harm-category Relative Risks for BW vs Others if the harm-by-group file exists.
    Save CSV/TEX.

    Returns
    -------
    pd.DataFrame
    """
    t0 = _t0("Harm relative risks (BW vs Others) ...")
    if not HARM_BY_GROUP.exists():
        print("[INFO] harm_category_by_group.csv not found; skipping harm RR.")
        _tend("step19.harm_rr", t0)
        return pd.DataFrame()
    df = pd.read_csv(HARM_BY_GROUP)
    cols = {c.lower(): c for c in df.columns}
    if not {"group","harm_category"}.issubset(set(cols.keys())):
        print("[INFO] harm file missing required columns; skipping harm RR.")
        _tend("step19.harm_rr", t0)
        return pd.DataFrame()
    group_col = cols["group"]; harm_col = cols["harm_category"]
    count_col = cols.get("count") or cols.get("n") or cols.get("freq")
    if count_col is None:
        print("[INFO] harm file missing count column; skipping harm RR.")
        _tend("step19.harm_rr", t0)
        return pd.DataFrame()

    # Identify BW rows (robust matching)
    bw_mask = (df[group_col].astype(str).str.contains("black", case=False) &
               df[group_col].astype(str).str.contains("female|woman|women", case=False))

    totals = df.groupby(group_col)[count_col].sum().rename("total").reset_index()
    merged = df.merge(totals, on=group_col)
    merged["_is_bw"] = merged[group_col].astype(str).isin(df.loc[bw_mask, group_col].astype(str).unique())

    rows = []
    for h, sub in merged.groupby(harm_col):
        a = sub.loc[sub["_is_bw"], [count_col, "total"]].sum()
        b = sub.loc[~sub["_is_bw"], [count_col, "total"]].sum()
        a1, a0 = float(a[count_col]), float(a["total"])
        b1, b0 = float(b[count_col]), float(b["total"])
        p1 = (a1 + EPS) / (a0 + EPS); p0 = (b1 + EPS) / (b0 + EPS)
        rr = p1 / p0
        log_rr = np.log(rr)
        se = np.sqrt((1/(a1+EPS)) - (1/(a0+EPS)) + (1/(b1+EPS)) - (1/(b0+EPS)))
        lo = np.exp(log_rr - 1.96*se); hi = np.exp(log_rr + 1.96*se)
        rows.append({
            "harm_category": h,
            "RR_bw_vs_others": _round(rr, 4),
            "RR_lo": _round(lo, 4), "RR_hi": _round(hi, 4),
            "bw_count": int(a1), "bw_total": int(a0),
            "others_count": int(b1), "others_total": int(b0),
        })
    out = pd.DataFrame(rows).sort_values("RR_bw_vs_others")

    csv = DATA_DIR / f"19_harm_relative_risks{suffix}.csv"
    tex = TABLE_DIR / f"19_harm_relative_risks{suffix}.tex"
    out.to_csv(csv, index=False); print(f"✓ Artefact saved: {csv}")
    try:
        with open(tex, "w", encoding="utf-8") as f:
            f.write(out.to_latex(index=False))
        print(f"✓ Artefact saved: {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")
    _tend("step19.harm_rr", t0)
    return out

# --- Orchestrator ------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Run Step 19 end-to-end.

    Options
    -------
    --selfcheck       Randomly subsample the corpus (non-destructive; *_selfcheck outputs).
    --sample INT      Subsample size for self-check (default: min(150k, N)).
    --top-k INT       How many categories to analyse for KL divergence (default from config or 30).
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--top-k", type=int, default=int(CONFIG.get("analysis", {}).get("top_k_categories", TOP_K_DEFAULT)))
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    print("--- Starting Step 19: Advanced Statistics ---")

    # Load & maybe subsample
    df = _load_corpus_fast()
    if args.selfcheck:
        n = args.sample or min(150_000, len(df))
        df = df.sample(n=n, random_state=SEED, replace=False).reset_index(drop=True)
        print(f"[SELF-CHECK] Random sample drawn: {len(df):,} rows (seed={SEED}).")

    # Prepare columns
    _prepare_columns(df)

    # Suffix to protect canonical artefacts during self-check
    suffix = "_selfcheck" if args.selfcheck else ""

    # Analyses
    eff    = effect_sizes_bw_vs_others(df, suffix=suffix)
    slopes = temporal_slopes_bw_vs_others(df, suffix=suffix)
    catdiv = category_divergence_by_group(df, top_k=args.top_k, suffix=suffix)
    harr   = harm_relative_risk(suffix=suffix)

    # Compact qualitative readout (extremes/outliers & interpretability)
    print("\n--- Quick qualitative readout ---")
    if not eff.empty:
        top_eff = eff.iloc[eff["cliffs_delta"].abs().astype(float).idxmax()]
        print(f"• Largest |Cliff’s δ|: {top_eff['metric']} = {float(top_eff['cliffs_delta']):.4f} "
              f"[{float(top_eff['ci_lo']):.4f}, {float(top_eff['ci_hi']):.4f}] "
              f"(n_BW={int(top_eff['n_bw'])}, n_Others={int(top_eff['n_others'])})")
    if not slopes.empty:
        r = slopes.set_index("metric")
        if "views_per_day" in r.index:
            print(f"• Slope gap (BW−Others) — views/day: {float(r.loc['views_per_day','slope_gap_bw_minus_others']):.4f}")
        if "rating" in r.index:
            print(f"• Slope gap (BW−Others) — rating: {float(r.loc['rating','slope_gap_bw_minus_others']):.4f}")
    if not catdiv.empty:
        gmax = catdiv.iloc[0]
        print(f"• Highest category KL divergence vs global: {gmax['group']} = {float(gmax['kl_vs_global']):.4f} (tokens={int(gmax['tokens'])})")
        print("*Note:* category totals can exceed N due to multi-label assignment. Some titles are non-English; tags/categories (MPU) anchor interpretation.")
    if not harr.empty:
        worst = harr.iloc[harr['RR_bw_vs_others'].astype(float).idxmax()]
        print(f"• Max harm RR (BW vs Others): {worst['harm_category']} = {float(worst['RR_bw_vs_others']):.3f} "
              f"[{float(worst['RR_lo']):.3f}, {float(worst['RR_hi']):.3f}]")

    # Narrative
    t0 = _t0("Writing narrative ...")
    lines = [
        "# 19 — Advanced Statistics (Uncertainty-aware, BW-centric)",
        "- Cliff’s δ with bootstrap CIs for age-normalised engagement (views/day, ratings/day) and ratings.",
        "- Temporal slopes per year with bootstrap CIs, incl. BW−Others slope gap.",
        f"- KL divergence vs global over top-{args.top_k} categories (multi-label; totals > N are expected).",
        "- If present, harm relative risks (BW vs Others) with log-normal CIs.",
        "- Some titles are not English; tags/categories (MPU) anchor semantics during interpretation.",
    ]
    md = NARR_DIR / f"19_advanced_statistics_summary{suffix}.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✓ Narrative saved: {md}")
    _tend("step19.write_narrative", t0)

    _tend("step19.total_runtime", t_all)
    print("\n--- Step 19: Advanced Statistics Completed Successfully ---")


if __name__ == "__main__":
    main()
