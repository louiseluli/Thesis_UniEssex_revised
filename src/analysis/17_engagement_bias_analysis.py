#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_engagement_bias_analysis.py
==============================

Purpose
-------
Deep engagement-bias analysis centred on Black women vs Others using
age-normalised metrics and bootstrap uncertainty.

What it does
------------
1) Loads config (seed=95) + canonical corpus; standardises time/engagement fields.
2) Derives tidy protected columns from one-hot where needed.
3) Computes:
   - Yearly gaps (Black women − Others) for views/day and rating with bootstrap 95% CIs.
   - Race-ethnicity quantiles for views/day & rating (p25/50/75/90/99) + BW vs Others plot.
   - Head vs Tail composition at the top 1% of views/day.
   - Point-biserial correlations between BW-flag and log-engagement metrics with CIs.
4) Saves CSV, LaTeX, dual-theme PNGs, and a brief narrative. Lightweight timers for each block.

CLI
---
# Full run
python -m src.analysis.17_engagement_bias_analysis

# Self-check (safe subsample; artefacts suffixed *_selfcheck)
python -m src.analysis.17_engagement_bias_analysis --selfcheck --sample 120000
"""
from __future__ import annotations

# -------- stdlib --------
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -------- third-party --------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- local --------
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config  # prints its own [TIME] line


# ---------------- Configuration & Paths ----------------
_t0_cfg = time.perf_counter()
CONFIG = load_config()
print(f"[TIME] theme_manager.load_config: {time.perf_counter() - _t0_cfg:.2f}s")

SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
np.random.seed(SEED)

ROOT = Path(CONFIG["project"]["root"])
DATA_DIR = Path(CONFIG["paths"]["data"])        # outputs/data
FIG_DIR  = Path(CONFIG["paths"]["figures"])     # outputs/figures
TABLE_DIR = ROOT / "dissertation" / "auto_tables"
NARR_DIR = Path(CONFIG["paths"]["narratives"]) / "automated"
for d in (DATA_DIR, FIG_DIR, TABLE_DIR, NARR_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Prefer step-01 parquet but allow fallbacks
CORPUS_CANDIDATES = [
    DATA_DIR / "01_ml_corpus.parquet",
    DATA_DIR / "ml_corpus.parquet",
    DATA_DIR / "01_ml_corpus.snappy.parquet",
]

# Bootstrap parameters
N_BOOT = 1000
ALPHA = 0.95


# ---------------- Timers ----------------
def _t0(msg: str) -> float:
    """Start timer with a standard header."""
    print(msg)
    return time.perf_counter()


def _tend(label: str, t0: float) -> None:
    """Stop timer and print standardized [TIME] line."""
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")


# ---------------- Theme ----------------
def _set_mpl_theme(dark: bool) -> None:
    """Minimal Matplotlib theme (pure Matplotlib)."""
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


# ---------------- Utils ----------------
def _first_existing(paths: List[Path]) -> Optional[Path]:
    """Return the first existing path among candidates, or None."""
    for p in paths:
        if p.exists():
            return p
    return None


def _round(x: float, k: int = 3) -> float:
    """Round floats safely; return x unchanged on failure."""
    try:
        return round(float(x), k)
    except Exception:
        return x


def _round_rating(x: float) -> float:
    """Ratings printed with one decimal place."""
    try:
        return round(float(x), 1)
    except Exception:
        return x


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Parse to datetime (UTC-naive for plotting) with coercion."""
    s = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return s.dt.tz_convert(None)
    except Exception:
        return s


def _read_corpus() -> pd.DataFrame:
    """Load the canonical corpus parquet (first available candidate)."""
    path = _first_existing(CORPUS_CANDIDATES)
    if path is None:
        exp = ", ".join(p.name for p in CORPUS_CANDIDATES)
        raise FileNotFoundError(f"Corpus not found under {DATA_DIR}. Expected one of: {exp}")
    t0 = _t0(f"[READ] {path}")
    df = pd.read_parquet(path)
    _tend("step17.load_corpus", t0)
    return df


def _standardise_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Standardise key columns and derive:
      _publish_dt, _publish_year (Int64), _age_days,
      _views, _rating, _rating_n.
    Resolution order: CONFIG['columns'] mapping, then robust name heuristics.
    """
    cols = {c.lower(): c for c in df.columns}
    cols_cfg = CONFIG.get("columns", {})

    def pick(name_candidates: List[str], cfg_key: Optional[str]) -> Optional[str]:
        if cfg_key and cfg_key in cols_cfg:
            c = cols_cfg[cfg_key]
            return c if c in df.columns else None
        for cand in name_candidates:
            if cand in cols:
                return cols[cand]
        return None

    publish_col  = pick(["publish_date","upload_date","published_at","date"], "publish_date")
    views_col    = pick(["views","view_count","views_count","play_count"],    "views")
    rating_col   = pick(["rating_clean","rating","rating_mean","average_rating","avg_rating","score"], "rating_mean")
    rating_n_col = pick(["ratings","rating_count","num_ratings","n_ratings","votes"],   "rating_count")

    if publish_col is not None:
        dt = _ensure_datetime(df[publish_col])
        df["_publish_dt"] = dt
        df["_publish_year"] = dt.dt.year.astype("Int64")
        today = pd.Timestamp.utcnow().tz_convert(None)
        df["_age_days"] = (today - dt).dt.days
    else:
        df["_publish_dt"] = pd.NaT
        df["_publish_year"] = pd.Series([pd.NA]*len(df), dtype="Int64")
        df["_age_days"] = np.nan

    df["_views"]    = pd.to_numeric(df[views_col], errors="coerce") if views_col else np.nan
    df["_rating"]   = pd.to_numeric(df[rating_col], errors="coerce") if rating_col else np.nan
    df["_rating_n"] = pd.to_numeric(df[rating_n_col], errors="coerce") if rating_n_col else np.nan

    return {
        "publish_date": publish_col,
        "views": views_col,
        "rating_mean": rating_col,
        "rating_count": rating_n_col,
    }


def _derive_categorical_from_onehot(
    df: pd.DataFrame, prefix: str, out_col: str, fallback: str = "unknown"
) -> Optional[str]:
    """
    Derive a tidy categorical from one-hot columns like '{prefix}_asian'.
    If multiple active bits → 'mixed_or_other'; if none → fallback.
    """
    pref = f"{prefix}_"
    onehot_cols = [c for c in df.columns if c.lower().startswith(pref)]
    if not onehot_cols:
        return None

    labels = [c[len(pref):].lower() for c in onehot_cols]
    mixed_token = "mixed_or_other" if "mixed_or_other" in labels else "mixed"

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


def _norm_gender_label(x: str) -> str:
    """Normalise gender labels for consistency."""
    x = (x or "").lower()
    if x in {"female", "woman", "women", "cis_female", "cis_woman"}: return "female"
    if x in {"male", "man", "men", "cis_male", "cis_man"}: return "male"
    return x or "unknown"


def _prepare_protected_columns(df: pd.DataFrame) -> List[str]:
    """
    Create tidy 'race_ethnicity'/'gender' if only one-hot exists.
    Return protected columns present/created (first is primary if needed).
    """
    created = []
    re_col = _derive_categorical_from_onehot(df, "race_ethnicity", "race_ethnicity")
    if re_col: created.append(re_col)
    g_col = _derive_categorical_from_onehot(df, "gender", "gender")
    if g_col: created.append(g_col)
    if "gender" in df.columns:
        df["gender"] = df["gender"].map(_norm_gender_label)

    candidates = [c for c in ["race_ethnicity", "gender", "sexual_orientation"] if c in df.columns]
    if not candidates:
        df["__all__"] = "__all__"
        candidates = ["__all__"]
        print("[WARN] No protected columns found. Using synthetic '__all__'.")
    else:
        print(f"[INFO] Protected columns: {candidates}")
    return candidates


# ---------------- Stats helpers ----------------
def _bootstrap_ci(values: np.ndarray, func, n: int = N_BOOT, alpha: float = ALPHA) -> Tuple[float, float, float]:
    """Generic bootstrap CI for a statistic; returns (point, lo, hi)."""
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.arange(vals.size)
    res = []
    for _ in range(n):
        samp = np.random.choice(idx, size=idx.size, replace=True)
        res.append(func(vals[samp]))
    arr = np.array(res, dtype=float)
    point = func(vals)
    lo, hi = np.quantile(arr, [(1 - alpha) / 2.0, 1 - (1 - alpha) / 2.0])
    return float(point), float(lo), float(hi)


def _mean_diff_ci(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Bootstrap CI for mean(a) − mean(b); returns (point, lo, hi)."""
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    na, nb = a.size, b.size
    ia = np.arange(na); ib = np.arange(nb)
    diffs = []
    for _ in range(N_BOOT):
        sa = np.random.choice(ia, size=na, replace=True)
        sb = np.random.choice(ib, size=nb, replace=True)
        diffs.append(a[sa].mean() - b[sb].mean())
    diffs = np.array(diffs, dtype=float)
    point = a.mean() - b.mean()
    lo, hi = np.quantile(diffs, [(1 - ALPHA) / 2.0, 1 - (1 - ALPHA) / 2.0])
    return float(point), float(lo), float(hi)


def _corr_ci(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Bootstrap CI for Pearson r (point-biserial when x is binary)."""
    m = (~np.isnan(x)) & (~np.isnan(y))
    x = x[m]; y = y[m]
    if x.size < 3:
        return float("nan"), float("nan"), float("nan")
    idx = np.arange(x.size)
    vals = []
    for _ in range(N_BOOT):
        s = np.random.choice(idx, size=idx.size, replace=True)
        xv = x[s]; yv = y[s]
        vx = xv.std(ddof=1); vy = yv.std(ddof=1)
        vals.append(float(np.corrcoef(xv, yv)[0, 1]) if vx > 0 and vy > 0 else 0.0)
    arr = np.array(vals, dtype=float)
    point = float(np.corrcoef(x, y)[0, 1]) if x.std(ddof=1) > 0 and y.std(ddof=1) > 0 else 0.0
    lo, hi = np.quantile(arr, [(1 - ALPHA) / 2.0, 1 - (1 - ALPHA) / 2.0])
    return point, float(lo), float(hi)


# ---------------- Feature engineering ----------------
def _compute_engagement_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add age-normalised and log metrics:
      _views_per_day, _ratings_per_day, _log_views, _log_vpd, _log_rpd.
    """
    df = df.copy()
    df["_age_days_clamped"] = pd.to_numeric(df["_age_days"], errors="coerce").clip(lower=1)
    df["_views_per_day"]    = df["_views"] / df["_age_days_clamped"]
    df["_ratings_per_day"]  = df["_rating_n"] / df["_age_days_clamped"]
    df["_log_views"]        = np.log1p(df["_views"].astype(float))
    df["_log_vpd"]          = np.log1p(df["_views_per_day"].astype(float))
    df["_log_rpd"]          = np.log1p(df["_ratings_per_day"].astype(float))
    return df


# ---------------- Analyses ----------------
def yearly_black_women_gaps(df: pd.DataFrame, *, suffix: str = "") -> pd.DataFrame:
    """
    Yearly mean gaps (Black women − Others) for views/day and rating with bootstrap 95% CIs.
    Returns a table with year, gaps, CIs, and group sample sizes.
    """
    if "race_ethnicity" not in df.columns or "gender" not in df.columns:
        print("[INFO] Missing race_ethnicity or gender; skipping yearly BW gaps.")
        return pd.DataFrame()

    tmp = df.dropna(subset=["_publish_year"]).copy()
    tmp["_publish_year"] = tmp["_publish_year"].astype(int)
    is_bw = (tmp["race_ethnicity"].str.lower() == "black") & (tmp["gender"].str.lower() == "female")

    rows = []
    for y, sub in tmp.groupby("_publish_year"):
        a = sub.loc[is_bw, "_views_per_day"].to_numpy(dtype=float)
        b = sub.loc[~is_bw, "_views_per_day"].to_numpy(dtype=float)
        d_vpd, lo_vpd, hi_vpd = _mean_diff_ci(a, b)

        a2 = sub.loc[is_bw, "_rating"].to_numpy(dtype=float)
        b2 = sub.loc[~is_bw, "_rating"].to_numpy(dtype=float)
        d_r, lo_r, hi_r = _mean_diff_ci(a2, b2)

        rows.append({
            "_publish_year": int(y),
            "gap_views_per_day": _round(d_vpd, 3),
            "gap_views_per_day_lo": _round(lo_vpd, 3),
            "gap_views_per_day_hi": _round(hi_vpd, 3),
            "gap_rating": _round_rating(d_r),
            "gap_rating_lo": _round_rating(lo_r),
            "gap_rating_hi": _round_rating(hi_r),
            "n_bw": int(np.isfinite(a).sum()),
            "n_others": int(np.isfinite(b).sum()),
        })
    out = pd.DataFrame(rows).sort_values("_publish_year")

    csv = DATA_DIR / f"17_yearly_bw_gaps{suffix}.csv"
    out.to_csv(csv, index=False); print(f"[WRITE] {csv}")

    tex = TABLE_DIR / f"17_yearly_bw_gaps{suffix}.tex"
    try:
        with open(tex, "w", encoding="utf-8") as f:
            f.write(out.to_latex(index=False))
        print(f"[TEX]   {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # Plots (dual theme)
    for metric, title, fname in [
        ("gap_views_per_day", "Yearly Gap — Views per day (BW − Others)", "17_gap_views_per_day"),
        ("gap_rating",       "Yearly Gap — Rating (BW − Others)",         "17_gap_rating"),
    ]:
        for dark in (True, False):
            _set_mpl_theme(dark=dark)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(out["_publish_year"], out[metric], marker="o", linewidth=2)
            lo = out.get(f"{metric}_lo"); hi = out.get(f"{metric}_hi")
            if lo is not None and hi is not None:
                ax.fill_between(out["_publish_year"], lo, hi, alpha=0.2)
            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("Year")
            ax.set_ylabel("Difference in means")
            ax.grid(True, alpha=0.3)
            f = FIG_DIR / f"{fname}_{'dark' if dark else 'light'}{suffix}.png"
            fig.tight_layout(); fig.savefig(f, dpi=200); plt.close(fig)
            print(f"[PLOT] {f}")

    return out


def quantile_profiles_by_race(df: pd.DataFrame, *, suffix: str = "") -> pd.DataFrame:
    """
    Race-ethnicity quantiles for views/day and rating at p25/50/75/90/99.
    Ratings rounded to 1 dp; views/day to 3 dp. Also emits a BW vs Others
    quantile plot for views/day.
    """
    if "race_ethnicity" not in df.columns:
        print("[INFO] No race_ethnicity; skipping quantile profiles.")
        return pd.DataFrame()

    qs = [0.25, 0.5, 0.75, 0.9, 0.99]
    rows = []
    for g, sub in df.groupby("race_ethnicity"):
        vpd = sub["_views_per_day"].to_numpy(dtype=float)
        rt  = sub["_rating"].to_numpy(dtype=float)
        qv = np.quantile(vpd[~np.isnan(vpd)], qs) if np.isfinite(vpd).any() else [np.nan]*len(qs)
        qr = np.quantile(rt[~np.isnan(rt)],  qs) if np.isfinite(rt).any()  else [np.nan]*len(qs)
        row = {"race_ethnicity": g}
        for q, val in zip(qs, qv):
            row[f"vpd_q{int(q*100)}"] = _round(val, 3)
        for q, val in zip(qs, qr):
            row[f"rating_q{int(q*100)}"] = _round_rating(val)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("race_ethnicity")

    csv = DATA_DIR / f"17_quantiles_race_ethnicity{suffix}.csv"
    out.to_csv(csv, index=False); print(f"[WRITE] {csv}")

    tex = TABLE_DIR / f"17_quantiles_race_ethnicity{suffix}.tex"
    try:
        with open(tex, "w", encoding="utf-8") as f:
            f.write(out.to_latex(index=False))
        print(f"[TEX]   {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # BW vs Others quantile lines for views/day
    if "gender" in df.columns:
        is_bw = (df["race_ethnicity"].str.lower() == "black") & (df["gender"].str.lower() == "female")
        a = df.loc[is_bw, "_views_per_day"].to_numpy(dtype=float)
        b = df.loc[~is_bw, "_views_per_day"].to_numpy(dtype=float)
        qa = np.quantile(a[~np.isnan(a)], qs) if np.isfinite(a).any() else [np.nan]*len(qs)
        qb = np.quantile(b[~np.isnan(b)], qs) if np.isfinite(b).any() else [np.nan]*len(qs)

        for dark in (True, False):
            _set_mpl_theme(dark=dark)
            fig, ax = plt.subplots(figsize=(8.5, 5))
            x = np.array([25, 50, 75, 90, 99], dtype=int)
            ax.plot(x, qa, marker="o", linewidth=2, label="Black women")
            ax.plot(x, qb, marker="o", linewidth=2, label="Others")
            ax.set_title("Quantile Profile — Views per day")
            ax.set_xlabel("Quantile (%)"); ax.set_ylabel("Views per day")
            ax.grid(True, alpha=0.3); ax.legend(frameon=False)
            f = FIG_DIR / f"17_quantiles_bw_vs_others_{'dark' if dark else 'light'}{suffix}.png"
            fig.tight_layout(); fig.savefig(f, dpi=200); plt.close(fig)
            print(f"[PLOT] {f}")

    return out


def head_tail_composition(df: pd.DataFrame, top_pct: float = 0.01, *, suffix: str = "") -> pd.DataFrame:
    """
    Composition of the head (top_pct by views/day) vs tail by race_ethnicity.
    Returns segment counts and shares per group.
    """
    if "_views_per_day" not in df.columns:
        return pd.DataFrame()

    thr = df["_views_per_day"].quantile(1 - top_pct)
    df = df.copy()
    df["_is_head"] = df["_views_per_day"] >= thr

    rows = []
    if "race_ethnicity" not in df.columns:
        totals = df.groupby("_is_head").size().rename("count").reset_index()
        for _, r in totals.iterrows():
            rows.append({
                "segment": "head" if r["_is_head"] else "tail",
                "race_ethnicity": "all",
                "count": int(r["count"]),
                "share": 1.0
            })
    else:
        for head_flag, sub in df.groupby("_is_head"):
            total = len(sub)
            if total == 0:
                continue
            for g, sub2 in sub.groupby("race_ethnicity"):
                rows.append({
                    "segment": "head" if head_flag else "tail",
                    "race_ethnicity": g,
                    "count": int(len(sub2)),
                    "share": _round(len(sub2) / total, 3),
                })

    out = pd.DataFrame(rows).sort_values(["segment", "share"], ascending=[True, False])

    csv = DATA_DIR / f"17_head_tail_composition{suffix}.csv"
    out.to_csv(csv, index=False); print(f"[WRITE] {csv}")

    tex = TABLE_DIR / f"17_head_tail_composition{suffix}.tex"
    try:
        with open(tex, "w", encoding="utf-8") as f:
            f.write(out.to_latex(index=False))
        print(f"[TEX]   {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # BW vs Others stacked bars if both columns exist
    if "race_ethnicity" in df.columns and "gender" in df.columns:
        is_bw = (df["race_ethnicity"].str.lower() == "black") & (df["gender"].str.lower() == "female")
        shares = (df.assign(_is_bw=is_bw)
                    .groupby(["_is_head", "_is_bw"]).size().reset_index(name="count"))
        shares["total_seg"] = shares.groupby("_is_head")["count"].transform("sum")
        shares["share"] = shares["count"] / shares["total_seg"]
        for dark in (True, False):
            _set_mpl_theme(dark=dark)
            fig, ax = plt.subplots(figsize=(7.5, 5))
            for i, seg in enumerate([False, True]):  # tail, head
                sub = shares.loc[shares["_is_head"] == seg]
                bw_share = sub.loc[sub["_is_bw"], "share"].astype(float).iloc[0] if sub["_is_bw"].any() else 0.0
                oth_share = 1.0 - bw_share
                ax.bar(i, bw_share, width=0.8, label="Black women" if i == 0 else None)
                ax.bar(i, oth_share, width=0.8, bottom=bw_share, label="Others" if i == 0 else None)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Tail (99%)", "Head (1%)"])
            ax.set_ylim(0, 1); ax.set_ylabel("Share")
            ax.set_title("Head vs Tail Composition — Black women share")
            ax.grid(True, axis="y", alpha=0.3); ax.legend(frameon=False)
            f = FIG_DIR / f"17_head_tail_stack_{'dark' if dark else 'light'}{suffix}.png"
            fig.tight_layout(); fig.savefig(f, dpi=200); plt.close(fig)
            print(f"[PLOT] {f}")

    return out


def bw_correlations(df: pd.DataFrame, *, suffix: str = "") -> pd.DataFrame:
    """
    Point-biserial correlations (Pearson with binary) between is_black_woman
    and engagement proxies (log-views, log-views/day, rating) with bootstrap CIs.
    """
    if "race_ethnicity" not in df.columns or "gender" not in df.columns:
        print("[INFO] Missing race_ethnicity or gender; skipping BW correlations.")
        return pd.DataFrame()

    is_bw = ((df["race_ethnicity"].str.lower() == "black") &
             (df["gender"].str.lower() == "female")).astype(float)

    metrics = {
        "log_views": df["_log_views"].to_numpy(dtype=float),
        "log_vpd":   df["_log_vpd"].to_numpy(dtype=float),
        "rating":    df["_rating"].to_numpy(dtype=float),
    }
    rows = []
    for name, arr in metrics.items():
        r, lo, hi = _corr_ci(is_bw.to_numpy(dtype=float), arr)
        rows.append({
            "metric": name,
            "corr": round(float(r), 4),
            "corr_lo": round(float(lo), 4),
            "corr_hi": round(float(hi), 4),
            "n": int(np.isfinite(arr).sum()),
        })
    out = pd.DataFrame(rows)

    csv = DATA_DIR / f"17_bw_correlations{suffix}.csv"
    out.to_csv(csv, index=False); print(f"[WRITE] {csv}")

    tex = TABLE_DIR / f"17_bw_correlations{suffix}.tex"
    try:
        with open(tex, "w", encoding="utf-8") as f:
            f.write(out.to_latex(index=False))
        print(f"[TEX]   {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    return out


# ---------------- Orchestrator ----------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Run Step 17 end-to-end.

    Options
    -------
    --selfcheck     Randomly subsample the corpus (non-destructive).
    --sample INT    Subsample size for self-check (default: min(150k, N)).
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=None)
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    print("--- Starting Step 17: Engagement Bias Analysis ---")

    # Load & optionally subsample
    df = _read_corpus()
    if args.selfcheck:
        n = args.sample or min(150_000, len(df))
        df = df.sample(n=n, random_state=SEED, replace=False).reset_index(drop=True)
        print(f"[SELF-CHECK] Random sample drawn: {len(df):,} rows (seed={SEED}).")

    # Standardise, prepare protected cols, engineer features
    _ = _standardise_columns(df)
    _ = _prepare_protected_columns(df)
    df = _compute_engagement_fields(df)

    suffix = "_selfcheck" if args.selfcheck else ""

    # Analyses with timers
    t0 = _t0("Yearly BW gaps ...")
    gaps = yearly_black_women_gaps(df, suffix=suffix)
    _tend("step17.yearly_gaps", t0)

    t0 = _t0("Quantile profiles by race_ethnicity ...")
    quants = quantile_profiles_by_race(df, suffix=suffix)
    _tend("step17.quantiles", t0)

    t0 = _t0("Head vs Tail composition ...")
    headtail = head_tail_composition(df, top_pct=0.01, suffix=suffix)
    _tend("step17.head_tail", t0)

    t0 = _t0("BW correlations with log-engagement ...")
    corr = bw_correlations(df, suffix=suffix)
    _tend("step17.correlations", t0)

    # Quick qualitative readout
    print("\n--- Quick qualitative readout ---")
    if not gaps.empty:
        gv = gaps["gap_views_per_day"].replace([np.inf, -np.inf], np.nan).dropna()
        gr = gaps["gap_rating"].replace([np.inf, -np.inf], np.nan).dropna()
        mean_gv = _round(gv.mean(), 3) if not gv.empty else np.nan
        mean_gr = _round_rating(gr.mean()) if not gr.empty else np.nan
        y_ext = gaps.loc[gaps["gap_views_per_day"].abs().idxmax(), "_publish_year"] if not gv.empty else "—"
        g_ext = _round(float(gaps["gap_views_per_day"].abs().max()), 3) if not gv.empty else np.nan
        print(f"• Mean yearly gap (views/day, BW−Others): {mean_gv} (largest |gap| {g_ext} in {y_ext})")
        print(f"• Mean yearly gap (rating, BW−Others): {mean_gr}")
    if not headtail.empty and "segment" in headtail.columns:
        head_seg = headtail[headtail["segment"] == "head"]
        if not head_seg.empty:
            top_share = head_seg["share"].max()
            print(f"• Max race share in the 1% head: {_round(float(top_share), 3)}")
    if not corr.empty:
        r_vpd = corr.loc[corr["metric"] == "log_vpd", "corr"].values
        if r_vpd.size:
            print(f"• Corr(BW, log-views/day): {float(r_vpd[0]):.4f}")

    # Narrative
    t0 = _t0("Writing narrative ...")
    title_col = CONFIG.get("columns", {}).get("title", "title")

    def _non_ascii_ratio(s: str) -> float:
        if not isinstance(s, str) or not s:
            return 0.0
        return sum(ord(ch) > 127 for ch in s) / len(s)

    non_ascii_share = float(np.round(df.get(title_col, pd.Series([""] * len(df))).fillna("")
                                     .map(_non_ascii_ratio).mean(), 3))
    notes = [
        "# 17 — Engagement Bias Deep Dive",
        f"- Non-ASCII title share (language heuristic): **{non_ascii_share:.3f}**.",
        "- Age-normalised metrics (views/day, ratings/day) mitigate recency effects.",
        "- Yearly BW−Others differences reported with bootstrap 95% CIs.",
        "- Quantile profiles expose tail vs head disparities; head/tail composition shows BW presence at the very top (1%).",
        "- Point-biserial correlations quantify association between the BW flag and engagement proxies.",
    ]
    md_name = f"17_engagement_bias_summary{suffix}.md"
    md_path = NARR_DIR / md_name
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(notes) + "\n")
    print(f"✓ Narrative saved: {md_path}")
    _tend("step17.write_narrative", t0)

    _tend("step17.total_runtime", t_all)
    print("\n--- Step 17: Completed ---")


if __name__ == "__main__":
    main()
