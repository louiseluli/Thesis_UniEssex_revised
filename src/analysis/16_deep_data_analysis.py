#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_deep_data_analysis.py
========================

Purpose
-------
Temporal evolution + engagement-bias diagnostics with robust fallbacks.
If protected-attribute columns are missing, a synthetic single group "__all__"
is used so the analysis still runs. Uses only Matplotlib.

What it does
------------
1) Loads config (seed=95) and discovers the canonical corpus parquet.
2) Standardises date/engagement fields and derives:
   - _publish_dt, _publish_year (int), _age_days
   - _views, _rating, _rating_n
   - _views_per_day, _ratings_per_day
3) Detects the primary protected attribute column (or falls back to "__all__").
4) Builds & saves:
   - Yearly group representation (CSV/TEX + dual-theme PNGs)
   - Yearly rating/view trends (with age normalisation)
   - Engagement inequality summary (Gini + max |Cliff's δ| on views)
5) Detects outliers (robust IQR) on engagement rates and prints a peek.
6) Writes a short narrative plus lightweight timers and a total runtime line.

Self-check
----------
--selfcheck randomly subsamples the corpus (reproducible; seed from config)
and writes *_selfcheck artefacts only (non-destructive).

Totals vs N
-----------
Earlier steps can be multi-label (totals > N). Here we aggregate per video/year,
so counts sum to N at this step.

CLI
---
# Full run (recommended)
python -m src.analysis.16_deep_data_analysis

# Self-check (safe, random sample)
python -m src.analysis.16_deep_data_analysis --selfcheck --sample 150000
"""
from __future__ import annotations

# ---------- stdlib ----------
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------- third-party ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------- local ----------
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config  # prints its own [TIME] line


# ----------------------- Configuration -----------------------
_t0_cfg = time.perf_counter()
CONFIG = load_config()
print(f"[TIME] theme_manager.load_config: {time.perf_counter() - _t0_cfg:.2f}s")

SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
np.random.seed(SEED)

ROOT = Path(CONFIG["project"]["root"])
DATA_DIR = Path(CONFIG["paths"]["data"])             # outputs/data
FIG_DIR = Path(CONFIG["paths"]["figures"])           # outputs/figures
TABLE_DIR = ROOT / "dissertation" / "auto_tables"
NARR_DIR = Path(CONFIG["paths"]["narratives"]) / "automated"
for d in (DATA_DIR, FIG_DIR, TABLE_DIR, NARR_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Canonical corpus candidates (prefer step-01 parquet; tolerate fallbacks)
CORPUS_CANDIDATES = [
    DATA_DIR / "01_ml_corpus.parquet",
    DATA_DIR / "ml_corpus.parquet",
    DATA_DIR / "01_ml_corpus.snappy.parquet",
]


# ----------------------- Timers -----------------------
def _t0(msg: str) -> float:
    print(msg)
    return time.perf_counter()


def _tend(label: str, t0: float) -> None:
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")


# ----------------------- Theme -----------------------
def set_mpl_theme(dark: bool) -> None:
    """Apply a minimal Matplotlib theme."""
    plt.rcParams.update({
        "figure.facecolor": "black" if dark else "white",
        "axes.facecolor": "black" if dark else "white",
        "axes.edgecolor": "white" if dark else "black",
        "axes.labelcolor": "white" if dark else "black",
        "xtick.color": "white" if dark else "black",
        "ytick.color": "white" if dark else "black",
        "text.color": "white" if dark else "black",
        "savefig.facecolor": "black" if dark else "white",
        "savefig.edgecolor": "black" if dark else "white",
        "grid.color": "gray",
        "grid.alpha": 0.25,
        "axes.prop_cycle": plt.cycler("color", ["#1f77b4", "#ff7f0e", "#2ca02c",
                                                "#d62728", "#9467bd", "#8c564b",
                                                "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]),
    })


# ----------------------- Utils -----------------------
def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _round(x: float, k: int = 3) -> float:
    try:
        return round(float(x), k)
    except Exception:
        return x


def _ensure_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return s.dt.tz_convert(None)
    except Exception:
        return s


def _read_corpus() -> pd.DataFrame:
    path = _first_existing(CORPUS_CANDIDATES)
    if path is None:
        exp = ", ".join(p.name for p in CORPUS_CANDIDATES)
        raise FileNotFoundError(f"Corpus not found under {DATA_DIR}. Expected one of: {exp}")
    t0 = _t0(f"[READ] {path}")
    df = pd.read_parquet(path)
    _tend("step16.load_corpus", t0)
    return df


def _standardise_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Standardise key columns (publish date, views, rating mean, rating count).
    Derives:
      _publish_dt, _publish_year (int), _age_days,
      _views, _rating, _rating_n, _views_per_day, _ratings_per_day.
    """
    cols = {c.lower(): c for c in df.columns}

    publish_col  = next((cols[c] for c in ("publish_date","upload_date","published_at","date") if c in cols), None)
    views_col    = next((cols[c] for c in ("views","view_count","views_count","play_count") if c in cols), None)
    rating_col   = next((cols[c] for c in ("rating","rating_mean","average_rating","avg_rating","score") if c in cols), None)
    rating_n_col = next((cols[c] for c in ("ratings","rating_count","num_ratings","n_ratings","votes") if c in cols), None)

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

    df["_views"]    = pd.to_numeric(df[views_col],    errors="coerce") if views_col    else np.nan
    df["_rating"]   = pd.to_numeric(df[rating_col],   errors="coerce") if rating_col   else np.nan
    df["_rating_n"] = pd.to_numeric(df[rating_n_col], errors="coerce") if rating_n_col else np.nan

    # Age-normalised rates
    df["_age_days_clamped"] = pd.to_numeric(df["_age_days"], errors="coerce").clip(lower=1)
    df["_views_per_day"]    = df["_views"] / df["_age_days_clamped"]
    df["_ratings_per_day"]  = df["_rating_n"] / df["_age_days_clamped"]

    return {
        "publish_date": publish_col,
        "views": views_col,
        "rating_mean": rating_col,
        "rating_count": rating_n_col,
    }


def _detect_protected_columns(df: pd.DataFrame) -> List[str]:
    """
    Return a list of protected-attribute columns (robust heuristics).
    Fallback: synthetic '__all__' single-group column.
    """
    candidates: List[str] = []
    for c in ["Group", "intersection_group", "race_ethnicity", "gender", "sexual_orientation"]:
        if c in df.columns:
            candidates.append(c)
    if not candidates:
        for c in df.columns:
            lc = c.lower()
            if ("race" in lc) or ("ethnic" in lc) or (lc == "gender") or ("sexual_orientation" in lc):
                candidates.append(c)
        if candidates:
            print(f"[WARN] Using heuristic protected columns: {candidates}")
    if not candidates:
        df["__all__"] = "__all__"
        candidates = ["__all__"]
        print("[WARN] No protected-attribute columns found. Using synthetic group '__all__'.")
    return candidates


def _detect_outliers_iqr(df: pd.DataFrame, group_col: str, k: int = 10) -> pd.DataFrame:
    """
    Robust IQR-based outlier detection on engagement rates per group/year.
    Returns top-k records by absolute IQR deviation score across:
    - _views_per_day, _ratings_per_day
    """
    records: List[Dict[str, object]] = []
    for metric in ["_views_per_day", "_ratings_per_day"]:
        sub = df.dropna(subset=[group_col, "_publish_year", metric]).copy()
        for (yr, grp), s in sub.groupby([sub["_publish_year"].astype(int), group_col])[metric]:
            q1 = np.nanpercentile(s, 25)
            q3 = np.nanpercentile(s, 75)
            iqr = max(q3 - q1, 1e-9)
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            vals = s.to_numpy()
            dist = np.maximum(low - vals, 0) + np.maximum(vals - high, 0)
            if dist.size:
                idx = int(np.argmax(dist))
                if dist[idx] > 0:
                    records.append({
                        group_col: grp,
                        "_publish_year": int(yr),
                        "metric": metric,
                        "value": float(vals[idx]),
                        "iqr_score": float(dist[idx] / (iqr + 1e-9)),
                    })
    if not records:
        return pd.DataFrame(columns=[group_col, "_publish_year", "metric", "value", "iqr_score"])
    out = pd.DataFrame.from_records(records).sort_values("iqr_score", ascending=False).head(k).reset_index(drop=True)
    out["_publish_year"] = out["_publish_year"].astype(int)
    out["value"] = out["value"].map(lambda x: _round(x, 3))
    out["iqr_score"] = out["iqr_score"].map(lambda x: _round(x, 3))
    return out


# ----------------------- Analyses -----------------------
def build_temporal_group_representation(df: pd.DataFrame, group_col: str, *, suffix: str = "") -> pd.DataFrame:
    """Share of each group per year (lines), saved to CSV/TEX + dual-theme PNGs."""
    out = (
        df.dropna(subset=["_publish_year"])
          .assign(_publish_year=lambda x: x["_publish_year"].astype(int))
          .groupby(["_publish_year", group_col])
          .size().reset_index(name="count")
    )
    total_year = out.groupby("_publish_year")["count"].transform("sum")
    out["share"] = (out["count"] / total_year).fillna(0.0)
    out["_publish_year"] = out["_publish_year"].astype(int)
    out["count"] = out["count"].astype(int)
    out["share"] = out["share"].map(lambda x: _round(x, 3))

    csv_path = DATA_DIR / f"16_temporal_group_representation{suffix}.csv"
    out.sort_values(["_publish_year", "share"], ascending=[True, False]).to_csv(csv_path, index=False)
    print(f"[WRITE] {csv_path}")

    tex_path = TABLE_DIR / f"16_temporal_group_representation{suffix}.tex"
    try:
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(out.to_latex(index=False))
        print(f"[TEX]   {tex_path}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    for dark in (True, False):
        set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        for g, sub in out.groupby(group_col):
            ax.plot(sub["_publish_year"], sub["share"], label=str(g), marker="o", linewidth=2)

        ax.set_title(f"Temporal Representation by Year — {group_col}")
        ax.set_xlabel("Year"); ax.set_ylabel("Share")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncols=2, frameon=False)
        f = FIG_DIR / f"16_temporal_representation_line_{'dark' if dark else 'light'}{suffix}.png"
        fig.tight_layout(); fig.savefig(f, dpi=200); plt.close(fig)
        print(f"[PLOT] {f}")

    print("\n[PREVIEW] temporal_group_representation (head):")
    print(out.head(8).to_string(index=False))
    return out


def build_temporal_rating_view_trends(df: pd.DataFrame, group_col: str, *, suffix: str = "") -> pd.DataFrame:
    """
    Yearly means for ratings/views and age-normalised rates; rounded per convention.

    Rounding:
      - mean_rating: 1 decimal
      - mean_views: 0 decimals (nearest integer)
      - mean_views_per_day, mean_ratings_per_day: 3 decimals
      - n: int
      - _publish_year: int
    """
    agg = (
        df.dropna(subset=["_publish_year"])
          .assign(_publish_year=lambda x: x["_publish_year"].astype(int))
          .groupby(["_publish_year", group_col])
          .agg(
              mean_rating=("_rating", "mean"),
              mean_views=("_views", "mean"),
              mean_views_per_day=("_views_per_day", "mean"),
              mean_ratings_per_day=("_ratings_per_day", "mean"),
              n=(group_col, "count"),
          ).reset_index()
    )

    r = agg.copy()
    r["_publish_year"] = r["_publish_year"].astype(int)
    r["n"] = r["n"].astype(int)
    r["mean_rating"] = r["mean_rating"].map(lambda x: _round(x, 1))
    r["mean_views"] = r["mean_views"].map(lambda x: float(int(round(x))))
    r["mean_views_per_day"] = r["mean_views_per_day"].map(lambda x: _round(x, 3))
    r["mean_ratings_per_day"] = r["mean_ratings_per_day"].map(lambda x: _round(x, 3))

    csv_path = DATA_DIR / f"16_temporal_rating_view_trends{suffix}.csv"
    r.to_csv(csv_path, index=False); print(f"[WRITE] {csv_path}")

    tex_path = TABLE_DIR / f"16_temporal_rating_view_trends{suffix}.tex"
    try:
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(r.to_latex(index=False))
        print(f"[TEX]   {tex_path}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # Plot four trend panels (ratings, views, and both age-normalised rates)
    plots = [
        ("mean_rating", "Temporal Rating Trends", "16_temporal_rating_trends"),
        ("mean_views", "Temporal Views Trends", "16_temporal_views_trends"),
        ("mean_views_per_day", "Temporal Views-per-day Trends", "16_temporal_views_per_day_trends"),
        ("mean_ratings_per_day", "Temporal Ratings-per-day Trends", "16_temporal_ratings_per_day_trends"),
    ]
    for metric, title, fname in plots:
        for dark in (True, False):
            set_mpl_theme(dark=dark)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            for g, sub in agg.groupby(group_col):
                ax.plot(sub["_publish_year"], sub[metric], label=str(g), marker="o", linewidth=2)
            ax.set_title(f"{title} — {group_col}")
            ax.set_xlabel("Year"); ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", ncols=2, frameon=False)
            f = FIG_DIR / f"{fname}_{'dark' if dark else 'light'}{suffix}.png"
            fig.tight_layout(); fig.savefig(f, dpi=200); plt.close(fig)
            print(f"[PLOT] {f}")

    print("\n[PREVIEW] temporal_rating_view_trends (head):")
    print(r.head(8).to_string(index=False))
    return r


def gini(values: np.ndarray) -> float:
    """
    Gini coefficient for non-negative arrays (ignores NaNs).
    Returns NaN if there are no valid values or the sum is zero.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = arr[arr >= 0]
    if arr.size == 0:
        return float("nan")
    arr.sort()
    n = arr.size
    cum = np.cumsum(arr, dtype=float)
    total = cum[-1]
    if total <= 0:
        return float("nan")
    return float((n + 1 - 2 * (cum.sum() / total)) / n)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's delta (approximate for very large inputs via sub-sampling).
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    MAX = 20000
    if a.size * b.size > MAX * MAX:
        rng = np.random.default_rng(SEED)
        a = rng.choice(a, size=min(a.size, MAX), replace=False)
        b = rng.choice(b, size=min(b.size, MAX), replace=False)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return float((gt - lt) / (a.size * b.size))


def build_engagement_bias_summary(df: pd.DataFrame, group_col: str, *, suffix: str = "") -> pd.DataFrame:
    """
    Summary by group: central tendency + inequality (Gini) + max |Cliff's δ| on views.

    Rounding:
      - *_mean/*_median: 3 decimals (except rating_mean/median shown at 1 decimal in narrative)
      - n: int
    """
    summary = (
        df.groupby(group_col)
          .agg(
              n=(group_col, "count"),
              views_mean=("_views", "mean"),
              views_median=("_views", "median"),
              rating_mean=("_rating", "mean"),
              rating_median=("_rating", "median"),
              ratings_count_mean=("_rating_n", "mean"),
              ratings_count_median=("_rating_n", "median"),
          ).reset_index()
    )

    # Inequality (Gini)
    gv, gr = [], []
    for g, sub in df.groupby(group_col):
        gv.append(gini(sub["_views"].values))
        gr.append(gini(sub["_rating_n"].values))
    summary["gini_views"] = [ _round(x, 4) for x in gv ]
    summary["gini_rating_count"] = [ _round(x, 4) for x in gr ]

    # Max |Cliff's δ| on views across pairs
    groups = summary[group_col].tolist()
    max_abs_delta = 0.0
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            a = df.loc[df[group_col] == groups[i], "_views"].values
            b = df.loc[df[group_col] == groups[j], "_views"].values
            max_abs_delta = max(max_abs_delta, abs(cliffs_delta(a, b)))
    summary["max_abs_cliffs_delta_views"] = _round(max_abs_delta, 4)

    # Rounding & types for table
    for c in ("views_mean","views_median","ratings_count_mean","ratings_count_median"):
        summary[c] = summary[c].map(lambda x: _round(x, 3))
    summary["rating_mean"] = summary["rating_mean"].map(lambda x: _round(x, 3))
    summary["rating_median"] = summary["rating_median"].map(lambda x: _round(x, 3))
    summary["n"] = summary["n"].astype(int)

    csv_path = DATA_DIR / f"16_engagement_bias_by_group{suffix}.csv"
    summary.to_csv(csv_path, index=False); print(f"[WRITE] {csv_path}")

    tex_path = TABLE_DIR / f"16_engagement_bias_by_group{suffix}.tex"
    try:
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(summary.to_latex(index=False))
        print(f"[TEX]   {tex_path}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    print("\n[PREVIEW] engagement_bias_by_group (head):")
    print(summary.head(8).to_string(index=False))
    return summary


# ----------------------- Orchestrator -----------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Orchestrate Step-16 with timers, outlier peek, and optional self-check.

    Options
    -------
    --selfcheck       Randomly subsample the corpus (non-destructive).
    --sample INT      Sample size (default: min(150k, N)).
    --peek-outliers   Print top outliers table to console.
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--peek-outliers", action="store_true")
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    print("--- Starting Step 16: Deep Data Analysis ---")

    # Load & (optionally) sample
    df = _read_corpus()
    if args.selfcheck:
        n = args.sample or min(150_000, len(df))
        df = df.sample(n=n, random_state=SEED, replace=False).reset_index(drop=True)
        print(f"[SELF-CHECK] Random sample drawn: {len(df):,} rows (seed={SEED}).")

    # Prepare fields
    _ = _standardise_columns(df)

    # Protected columns
    t0 = _t0("Detecting protected-attribute columns ...")
    prot_cols = _detect_protected_columns(df)
    primary = prot_cols[0]
    print(f"[INFO] Protected columns: {prot_cols}. Primary for plots: {primary}")
    _tend("step16.detect_protected", t0)

    suffix = "_selfcheck" if args.selfcheck else ""

    # Representation
    t0 = _t0("Analyzing temporal representation trends ...")
    rep = build_temporal_group_representation(df, primary, suffix=suffix)
    _tend("step16.representation", t0)

    # Engagement summary
    t0 = _t0("Analyzing engagement disparities ...")
    summ = build_engagement_bias_summary(df, primary, suffix=suffix)
    _tend("step16.engagement_summary", t0)

    # Rating/View trends
    t0 = _t0("Analyzing rating/views temporal trends ...")
    trends = build_temporal_rating_view_trends(df, primary, suffix=suffix)
    _tend("step16.rating_views_trends", t0)

    # Outliers (IQR) on engagement rates
    t0 = _t0("Detecting outliers on engagement rates ...")
    outliers = _detect_outliers_iqr(df, primary, k=10)
    out_csv = DATA_DIR / f"16_engagement_outliers_top10{suffix}.csv"
    outliers.to_csv(out_csv, index=False)
    print(f"[WRITE] {out_csv}")
    if args.peek_outliers and not outliers.empty:
        print("\n[PEEK] Top engagement outliers (|IQR score|):")
        print(outliers.to_string(index=False))
    _tend("step16.outliers", t0)

    # Narrative
    t0 = _t0("Writing temporal & engagement narrative ...")
    def _non_ascii_ratio(s: str) -> float:
        if not isinstance(s, str) or not s:
            return 0.0
        return sum(ord(ch) > 127 for ch in s) / len(s)
    title_col = CONFIG.get("columns", {}).get("title", "title")
    non_ascii_share = float(np.round(df.get(title_col, pd.Series([""]*len(df))).fillna("").map(_non_ascii_ratio).mean(), 3))

    notes = [
        "# 16 — Temporal & Engagement Bias Summary",
        f"- Primary attribute analysed: **{primary}**.",
        f"- Non-ASCII title share (language heuristic): **{non_ascii_share:.3f}**.",
        "- Outputs: temporal representation, rating/views trends (age-normalised),",
        "  engagement inequality (Gini) with max |Cliff’s δ| on views, and outlier table.",
        "*Note:* Titles in other languages can be harder to interpret; **tags** and **categories** help anchor meaning.",
    ]
    md_path = NARR_DIR / (f"16_temporal_engagement_summary{suffix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(notes) + "\n")
    print(f"[WRITE] {md_path}")
    _tend("step16.write_narrative", t0)

    _tend("step16.total_runtime", t_all)
    print("\n--- Step 16: Deep Data Analysis Completed ---")


if __name__ == "__main__":
    main()
