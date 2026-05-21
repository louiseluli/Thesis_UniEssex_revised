#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 23 — Limitations Analysis
==============================

Literature anchors for thesis narrative:
- Datasheets for Datasets — Gebru et al., 2018 (documentation & missingness)
- Model Cards — Mitchell et al., 2019 (reporting limitations)
- Suresh & Guttag, 2019 (taxonomy of ML harm: representation/measurement/allocation)
- Barocas & Selbst, 2016 (disparate impact)

This module computes:
A. Data Quality & Missingness
B. Representation Skew (coverage/balance)
C. Temporal Drift (BW share & Simpson’s flip check)
D. Category Long-Tail (Top-K mass & curve)
E. Lexicon Dependency (if harm_category_by_group.csv exists)
F. Checklist synthesis (flags + metrics)

Outputs (canonical; self-check writes *_selfcheck.*):
  CSVs:
    outputs/data/23_missingness[ _selfcheck].csv
    outputs/data/23_representation_skew[ _selfcheck].csv
    outputs/data/23_temporal_drift[ _selfcheck].csv
    outputs/data/23_category_longtail[ _selfcheck].csv
    outputs/data/23_hurtlex_dependency[ _selfcheck].csv  (if available)
    outputs/data/23_summary_checklist[ _selfcheck].csv
  LaTeX (for dissertation):
    dissertation/auto_tables/23_*.tex  (mirrors CSVs; _selfcheck suffix when applicable)
  Figures (dark|light):
    outputs/figures/dark|light/23_missingness_bar_*.png
    outputs/figures/dark|light/23_representation_bar_*.png
    outputs/figures/dark|light/23_temporal_share_*.png
    outputs/figures/dark|light/23_longtail_curve_*.png
  Narrative:
    outputs/narratives/automated/23_limitations_analysis[ _selfcheck].md

Notes:
• Category totals can exceed N due to multi-label assignment.
• Titles can be non-English; tags/categories preserve interpretable semantics (MPU).
• Years are integers; ratings are rounded to 1 decimal; other numerics rounded sensibly.
"""

from __future__ import annotations

# ----------------------------- Imports (top only) -----------------------------
import argparse
import math
import sys
import time
from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

# ----------------------------- Config & theming ------------------------------
# Ensure 'src' is importable when running file directly (not just -m)
sys.path.append(str(Path(__file__).resolve().parents[2]))
try:
    from src.utils.theme_manager import ThemeManager, load_config
    THEME = ThemeManager()  # theme_manager prints its own [TIME] lines
    CONFIG = load_config() or {}
except Exception:
    THEME = None
    CONFIG = {}

# project roots & paths (from config when present)
ROOT = Path(CONFIG.get("project", {}).get("root", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(CONFIG.get("paths", {}).get("data", ROOT / "outputs" / "data"))
FIG_DARK = ROOT / "outputs" / "figures" / "dark"
FIG_LIGHT = ROOT / "outputs" / "figures" / "light"
NARR_DIR = ROOT / "outputs" / "narratives" / "automated"
AUTO_TEX = ROOT / "dissertation" / "auto_tables"

for d in (DATA_DIR, FIG_DARK, FIG_LIGHT, NARR_DIR, AUTO_TEX):
    d.mkdir(parents=True, exist_ok=True)

HARM_WIDE = DATA_DIR / "harm_category_by_group.csv"

# reproducibility seed from config (NOT 42)
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
np.random.seed(SEED)

EPS = 1e-12

# ----------------------------- Lightweight timers ----------------------------
def _t0(msg: str) -> float:
    """Start a high-resolution timer and print a heading."""
    print(msg)
    return time.perf_counter()

def _tend(label: str, t_start: float) -> None:
    """Stop timer and print standardized [TIME] message."""
    print(f"[TIME] {label}: {time.perf_counter() - t_start:.2f}s")

# ----------------------------- IO helpers ------------------------------------
def _suffix(name: str, selfcheck: bool) -> str:
    """Append _selfcheck to a base filename (before extension) when requested."""
    if not selfcheck:
        return name
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        return f"{stem}_selfcheck.{ext}"
    return f"{name}_selfcheck"

def _out_csv(name: str, selfcheck: bool = False) -> Path:
    """Build a CSV output path in DATA_DIR with optional self-check suffix."""
    return DATA_DIR / _suffix(name, selfcheck)

def _out_tex(name: str, selfcheck: bool = False) -> Path:
    """Build a LaTeX output path in AUTO_TEX with optional self-check suffix."""
    return AUTO_TEX / _suffix(name, selfcheck)

def _out_md(name: str, selfcheck: bool = False) -> Path:
    """Build a Markdown output path in NARR_DIR with optional self-check suffix."""
    return NARR_DIR / _suffix(name, selfcheck)

def _plot_path(base: str, dark: bool, selfcheck: bool = False) -> Path:
    """Build a plot path (dark/light) with optional self-check suffix."""
    side = "dark" if dark else "light"
    name = _suffix(f"{base}_{side}.png", selfcheck)
    return (FIG_DARK if dark else FIG_LIGHT) / name

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV with a friendly confirmation line."""
    df.to_csv(path, index=False)
    print(f"[WRITE] {path}")

def _write_tex_table(df: pd.DataFrame, path: Path) -> None:
    """Write a minimal LaTeX table; avoids heavy dependencies."""
    try:
        tex = df.to_latex(index=False)
    except Exception:
        tex = df.to_string(index=False)
    path.write_text(tex, encoding="utf-8")
    print(f"[TEX]   {path}")

def _delete_stale(paths: List[Path]) -> None:
    """Delete stale artefacts; report deletions."""
    for p in paths:
        try:
            if p.exists():
                p.unlink()
                print(f"[DELETE] {p}")
        except Exception:
            pass

def _first_existing(paths: List[Path]) -> Optional[Path]:
    """Return the first existing path from a candidates list (or None)."""
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

def _resolve_corpus_path() -> Tuple[Optional[Path], List[Path]]:
    """
    Resolve the corpus parquet path using config and common fallbacks.

    Returns
    -------
    (found_path_or_None, searched_candidates)
    """
    searched: List[Path] = []
    pconf = CONFIG.get("paths", {}) if isinstance(CONFIG, dict) else {}

    # 1) Config-driven candidates
    for key in ("ml_corpus", "corpus", "corpus_parquet"):
        val = pconf.get(key)
        if not val:
            continue
        cand = [Path(val), ROOT / val, DATA_DIR / val]
        searched.extend(cand)
        hit = _first_existing(cand)
        if hit:
            return hit.resolve(), searched

    # 2) Common filenames
    common = [
        DATA_DIR / "01_ml_corpus.parquet",
        DATA_DIR / "ml_corpus.parquet",
        DATA_DIR / "corpus.parquet",
    ]
    searched.extend(common)
    hit = _first_existing(common)
    if hit:
        return hit.resolve(), searched

    # 3) Glob fallbacks (choose most-recent modified if multiple)
    globbed: List[Path] = []
    for pat in ("*ml_corpus*.parquet", "*corpus*.parquet"):
        globbed.extend(sorted(DATA_DIR.glob(pat)))
    searched.extend(globbed)
    if globbed:
        hit = max(globbed, key=lambda p: p.stat().st_mtime)
        return hit.resolve(), searched

    return None, searched

# ----------------------------- Plot theming ----------------------------------
def _set_theme(dark: bool) -> None:
    """
    Apply dark/light plotting theme. Uses ThemeManager if available; else rcParams.
    """
    if THEME is not None:
        THEME.apply(dark=dark)
        return
    plt.rcParams.update(
        {
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
        }
    )

# ----------------------------- Parsing helpers -------------------------------
def _try_cols(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first existing column from candidates; else None."""
    return next((c for c in candidates if c in df.columns), None)

def _ensure_datetime(s: pd.Series) -> pd.Series:
    """Coerce to timezone-naive pandas datetime (UTC→naive)."""
    s = pd.to_datetime(s, errors="coerce", utc=True)
    try:
        return s.dt.tz_convert(None)
    except Exception:
        return s

def _parse_listish(x) -> List[str]:
    """
    Parse a stringified list or delimited categories into clean lowercase tokens.
    Accepts JSON-like lists, comma/pipe separated strings, or whitespace tokens.
    """
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(t).strip().lower() for t in x if str(t).strip()]
    s = str(x).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            obj = literal_eval(s)
            if isinstance(obj, (list, tuple, set)):
                return [str(t).strip().lower() for t in obj if str(t).strip()]
        except Exception:
            return [s.lower()]
    sep = "," if ("," in s and "|" not in s) else ("|" if "|" in s else None)
    return [p.strip().lower() for p in s.split(sep)] if sep else [p.strip().lower() for p in s.split()]

# ----------------------------- Data loading & prep ---------------------------
def load_minimal_corpus(max_rows: Optional[int] = None, *, random_sample: bool = False) -> pd.DataFrame:
    """
    Load the corpus with minimally required columns. Optionally random-subsample rows.
    """
    t = _t0("Loading corpus ...")

    found, searched = _resolve_corpus_path()
    if not found:
        lines = "\n  - ".join(str(p) for p in searched) if searched else "(no candidates)"
        raise FileNotFoundError(
            "Could not locate a corpus parquet.\nSearched:\n"
            f"  - {lines}\n"
            "Tip: add one of these to CONFIG['paths']: 'ml_corpus', 'corpus', or 'corpus_parquet'."
        )

    try:
        df = pd.read_parquet(found)
        print(f"[READ] {found} (fast)")
    except Exception as e:
        raise RuntimeError(f"Failed to read corpus parquet at {found}: {e}")

    if max_rows is not None and max_rows < len(df):
        if random_sample:
            df = df.sample(n=max_rows, random_state=SEED, replace=False).reset_index(drop=True)
            print(f"[SELF-CHECK] Random sample drawn: {len(df):,} rows (seed={SEED}).")
        else:
            df = df.head(max_rows).copy()

    _tend("step23.load_corpus", t)
    return df

def prep_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare derived fields: publish year, age, numeric metrics, categories/tags, views/day.

    Notes
    -----
    • Keeps years as integers (nullable).
    • Ratings rounded to one decimal only when reported; raw kept numeric.
    """
    t = _t0("Preparing derived fields ...")

    # publish date/year + age
    pcol = _try_cols(df, ["publish_date", "upload_date", "published_at", "date"])
    if pcol:
        dt = _ensure_datetime(df[pcol])
        df["_publish_year"] = dt.dt.year.astype("Int64")
        today = pd.Timestamp.utcnow().tz_convert(None)
        df["_age_days"] = (today - dt).dt.days.astype("Int64")
    else:
        df["_publish_year"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        df["_age_days"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # numeric metrics
    def _num(cols):
        c = _try_cols(df, cols)
        return pd.to_numeric(df[c], errors="coerce") if c else pd.Series(np.nan, index=df.index)

    df["_views"] = _num(["views", "view_count", "views_count", "play_count"])
    df["_rating"] = _num(["rating", "rating_mean", "average_rating", "avg_rating", "score"])
    df["_rating_n"] = _num(["ratings", "rating_count", "num_ratings", "n_ratings", "votes"])

    # categories/tags column (best-effort)
    df["_categories_col"] = _try_cols(df, ["categories", "tags", "category", "tag_list", "labels"])

    # protected attributes — attempt to derive categorical from one-hot if needed
    if "race_ethnicity" not in df.columns:
        pref = "race_ethnicity_"
        onehot = [c for c in df.columns if c.lower().startswith(pref)]
        if onehot:
            labels = [c[len(pref):].lower() for c in onehot]
            oh = df[onehot].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            oh = (oh > 0.5).astype(int)
            vals = []
            for i in range(len(df)):
                row = oh.iloc[i].values
                hits = np.where(row == 1)[0]
                vals.append(labels[hits[0]] if len(hits) == 1 else ("mixed_or_other" if len(hits) > 1 else "unknown"))
            df["race_ethnicity"] = pd.Series(vals, index=df.index, dtype="object")

    if "gender" not in df.columns:
        pref = "gender_"
        onehot = [c for c in df.columns if c.lower().startswith(pref)]
        if onehot:
            labels = [c[len(pref):].lower() for c in onehot]
            oh = df[onehot].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            oh = (oh > 0.5).astype(int)
            vals = []
            for i in range(len(df)):
                row = oh.iloc[i].values
                hits = np.where(row == 1)[0]
                vals.append(labels[hits[0]] if len(hits) == 1 else ("mixed_or_other" if len(hits) > 1 else "unknown"))
            df["gender"] = pd.Series(vals, index=df.index, dtype="object")

    # engagement per day
    df["_age_days_clamped"] = pd.to_numeric(df["_age_days"], errors="coerce").clip(lower=1)
    df["_views_per_day"] = pd.to_numeric(df["_views"], errors="coerce") / df["_age_days_clamped"]

    _tend("step23.prepare", t)
    return df

# ----------------------------- Metrics primitives ----------------------------
def _gini(arr: np.ndarray) -> float:
    """Compute the Gini coefficient (non-negative inputs preferred)."""
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.any(x < 0):
        x = x - x.min()
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))

def _norm_entropy(proportions: np.ndarray) -> float:
    """Normalized entropy ∈ [0,1] for a distribution of shares."""
    p = proportions.astype(float) + EPS
    p = p / p.sum()
    H = -np.sum(p * np.log(p))
    Hmax = np.log(len(p))
    return float(H / (Hmax + EPS))

# ----------------------------- A. Missingness --------------------------------
def analyze_missingness(df: pd.DataFrame, selfcheck: bool) -> pd.DataFrame:
    """
    Compute missingness fraction for key fields and save CSV/TeX + bar charts.
    """
    t = _t0("A. Data Quality & Missingness ...")
    cols = {
        "publish_date": _try_cols(df, ["publish_date", "upload_date", "published_at", "date"]),
        "views": _try_cols(df, ["views", "view_count", "views_count", "play_count"]),
        "rating": _try_cols(df, ["rating", "rating_mean", "average_rating", "avg_rating", "score"]),
        "ratings_count": _try_cols(df, ["ratings", "rating_count", "num_ratings", "n_ratings", "votes"]),
        "race_ethnicity": "race_ethnicity" if "race_ethnicity" in df.columns else None,
        "gender": "gender" if "gender" in df.columns else None,
    }
    rows = []
    for name, col in cols.items():
        if col is None:
            frac = 1.0
            n = 0
        else:
            miss = df[col].isna().sum()
            frac = float(miss / len(df)) if len(df) else 1.0
            n = int(len(df) - miss)
        rows.append({"field": name, "non_null_n": int(n), "missing_fraction": round(frac, 4)})
    out = pd.DataFrame(rows).sort_values("missing_fraction", ascending=False)

    _write_csv(out, _out_csv("23_missingness.csv", selfcheck))
    _write_tex_table(out, _out_tex("23_data_quality.tex", selfcheck))

    # plot
    for dark in (True, False):
        _set_theme(dark)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(out["field"], out["missing_fraction"])
        ax.set_ylabel("Missing fraction")
        ax.set_title("Data Missingness by Field")
        ax.grid(True, axis="y", alpha=0.3)

        # Keep axis anchored at 0; if all zeros, show a small headroom so bars/labels are visible
        y_top = float(out["missing_fraction"].max())
        ax.set_ylim(0.0, max(0.05, y_top + 0.02))

        # Value labels on bars (works on Matplotlib ≥3.4)
        try:
            ax.bar_label(bars, fmt="%.2f", padding=3)
        except Exception:
            pass  # older Matplotlib—skip labels

        # A tiny interpretation cue (non-intrusive)
        ax.annotate(
            "Lower is better; 0 means no missing values for that field.",
            xy=(0.01, 0.98), xycoords="axes fraction",
            va="top", ha="left", fontsize=8, alpha=0.8
        )

        fig.tight_layout()
        fn = _plot_path("23_missingness_bar", dark, selfcheck)
        fig.savefig(fn, dpi=200)
        plt.close(fig)
        print(f"[PLOT] {fn}")


    _tend("step23.missingness", t)
    return out

# ----------------------------- B. Representation -----------------------------
def analyze_representation(df: pd.DataFrame, selfcheck: bool) -> pd.DataFrame:
    """
    Measure representation coverage/balance via shares, normalized entropy, Gini.
    Also plots bar charts of group shares for race_ethnicity and gender.
    """
    t = _t0("B. Representation Skew (coverage/balance) ...")
    rows = []
    for col in ["race_ethnicity", "gender"]:
        if col not in df.columns:
            continue
        vc = df[col].astype(str).str.lower().value_counts()
        total = max(1, vc.sum())
        p = (vc / total).to_numpy(float)
        rows.append(
            {
                "dimension": col,
                "n_distinct": int(vc.size),
                "min_share": float(round(vc.min() / total, 4)),
                "max_share": float(round(vc.max() / total, 4)),
                "entropy_norm": round(_norm_entropy(p), 4),
                "gini": round(_gini(vc.to_numpy(float)), 4),
                "underrep_groups": ", ".join(vc[vc <= 0.02 * vc.sum()].index.tolist()[:10]) if vc.size else "",
            }
        )

        # plot shares (fixed ticks to avoid set_ticklabels warnings)
        x = np.arange(len(vc))
        y = (vc / total).to_numpy()
        for dark in (True, False):
            _set_theme(dark)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(x, y)
            ax.set_title(f"Group Shares — {col}")
            ax.set_ylabel("Share")
            ax.set_xticks(x)
            ax.set_xticklabels(vc.index, rotation=45, ha="right")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fn = _plot_path(f"23_representation_bar_{col}", dark, selfcheck)
            fig.savefig(fn, dpi=200)
            plt.close(fig)
            print(f"[PLOT] {fn}")

    out = pd.DataFrame(rows)
    _write_csv(out, _out_csv("23_representation_skew.csv", selfcheck))
    _write_tex_table(out, _out_tex("23_representation_skew.tex", selfcheck))
    _tend("step23.representation", t)
    return out

# ----------------------------- C. Temporal Drift -----------------------------
def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Correlation with guards against small samples and near-constant arrays.
    Returns NaN if unstable.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if a.size < 3 or np.isclose(a.std(), 0.0) or np.isclose(b.std(), 0.0):
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def analyze_temporal_drift(df: pd.DataFrame, selfcheck: bool) -> pd.DataFrame:
    """
    Compute BW share per year, slope, overall vs per-year correlations (Simpson’s check).
    """
    t = _t0("C. Temporal Drift & Simpson’s check ...")

    if "_publish_year" not in df.columns:
        _tend("step23.temporal", t)
        return pd.DataFrame()

    m = df["_publish_year"].notna()
    if not m.any():
        _tend("step23.temporal", t)
        return pd.DataFrame()

    # robust protected series (avoid .astype on str default)
    re_series = df["race_ethnicity"] if "race_ethnicity" in df.columns else pd.Series([""] * len(df), index=df.index)
    g_series = df["gender"] if "gender" in df.columns else pd.Series([""] * len(df), index=df.index)

    # BW indicator (intersection: Black & Female)
    is_bw = re_series.astype(str).str.lower().eq("black") & g_series.astype(str).str.lower().isin(
        {"female", "woman", "women", "cis_female", "cis_woman"}
    )

    # yearly shares
    yearly = (
        pd.DataFrame({"year": df.loc[m, "_publish_year"].astype("Int64"), "is_bw": is_bw.loc[m].astype(bool)})
        .groupby("year", dropna=True)
        .agg(n=("is_bw", "size"), bw=("is_bw", "sum"))
        .reset_index()
    )
    if not yearly.empty:
        yearly["year"] = yearly["year"].astype(int)
        yearly["bw_share"] = yearly["bw"] / yearly["n"].clip(lower=1)

    # slope of bw_share
    slope = float(np.polyfit(yearly["year"], yearly["bw_share"], 1)[0]) if len(yearly) >= 2 else np.nan

    # per-year corr of BW vs views/day
    cors = []
    views_per_day = pd.to_numeric(df["_views_per_day"], errors="coerce")
    for y, sub in df.loc[m].groupby("_publish_year"):
        if sub.shape[0] < 10:
            continue
        v = pd.to_numeric(sub["_views_per_day"], errors="coerce").to_numpy()
        sub_re = sub["race_ethnicity"] if "race_ethnicity" in sub.columns else pd.Series([""] * len(sub), index=sub.index)
        sub_g = sub["gender"] if "gender" in sub.columns else pd.Series([""] * len(sub), index=sub.index)
        z = (sub_re.astype(str).str.lower().eq("black") & sub_g.astype(str).str.lower().isin(
            {"female", "woman", "women", "cis_female", "cis_woman"}
        )).astype(int).to_numpy()
        cors.append(_safe_corr(z, v))
    per_year_median_corr = float(np.nanmedian(np.array(cors))) if cors else np.nan
    overall_corr = (
        _safe_corr(is_bw.astype(int).to_numpy(), views_per_day.to_numpy()) if views_per_day.notna().sum() > 5 else np.nan
    )
    simpson_flip = (
        np.sign(overall_corr) != np.sign(per_year_median_corr)
        if np.isfinite(overall_corr) and np.isfinite(per_year_median_corr)
        else False
    )

    out = pd.DataFrame(
        [
            {
                "years_covered": f"{int(yearly['year'].min())}–{int(yearly['year'].max())}" if not yearly.empty else "",
                "bw_share_slope_per_year": round(slope, 6),
                "overall_corr_bw_vs_views_per_day": round(overall_corr, 4) if np.isfinite(overall_corr) else np.nan,
                "median_per_year_corr": round(per_year_median_corr, 4) if np.isfinite(per_year_median_corr) else np.nan,
                "simpson_flip_detected": bool(simpson_flip),
                "n_years": int(len(yearly)),
            }
        ]
    )

    _write_csv(out, _out_csv("23_temporal_drift.csv", selfcheck))
    _write_tex_table(out, _out_tex("23_temporal_drift.tex", selfcheck))

    # plot BW share over time
    if not yearly.empty:
        for dark in (True, False):
            _set_theme(dark)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(yearly["year"], yearly["bw_share"], marker="o")
            ax.set_title("BW Share Over Time")
            ax.set_ylabel("Share")
            ax.set_xlabel("Year")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fn = _plot_path("23_temporal_share", dark, selfcheck)
            fig.savefig(fn, dpi=200)
            plt.close(fig)
            print(f"[PLOT] {fn}")

    _tend("step23.temporal", t)
    return out

# ----------------------------- D. Long-tail ----------------------------------
def _topk_mass(cat_lists: List[List[str]], ks=(10, 20, 30, 50)) -> pd.DataFrame:
    """Compute mass captured by Top-K categories for a list of token lists."""
    counts: Dict[str, int] = {}
    for lst in cat_lists:
        for tok in lst:
            counts[tok] = counts.get(tok, 0) + 1
    if not counts:
        return pd.DataFrame([{"K": k, "mass_captured": np.nan} for k in ks])
    ser = pd.Series(counts).sort_values(ascending=False)
    total = ser.sum()
    rows = [{"K": int(k), "mass_captured": float(ser.head(k).sum() / max(1, total))} for k in ks]
    return pd.DataFrame(rows)

def analyze_longtail(df: pd.DataFrame, selfcheck: bool) -> pd.DataFrame:
    """
    Long-tail analysis: Top-K mass (K={10,20,30,50}) and cumulative coverage curve.
    """
    t_start = _t0("D. Category Long-Tail ...")
    ccol = df.get("_categories_col", None)
    if isinstance(ccol, pd.Series):
        ccol = ccol.iloc[0] if not ccol.empty else None
    cat_lists = df[ccol].map(_parse_listish).tolist() if ccol else []
    out = _topk_mass(cat_lists, ks=(10, 20, 30, 50))
    _write_csv(out, _out_csv("23_category_longtail.csv", selfcheck))
    _write_tex_table(out, _out_tex("23_category_longtail.tex", selfcheck))

    # curve
    if cat_lists:
        counts: Dict[str, int] = {}
        for lst in cat_lists:
            for tok in lst:
                counts[tok] = counts.get(tok, 0) + 1
        ser = pd.Series(counts).sort_values(ascending=False)
        cum = ser.cumsum() / max(1, ser.sum())
        xs = np.arange(1, len(ser) + 1)
        for dark in (True, False):
            _set_theme(dark)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(xs, cum.values)
            ax.set_title("Category Long-Tail Coverage Curve")
            ax.set_xlabel("Top-K categories")
            ax.set_ylabel("Mass captured")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fn = _plot_path("23_longtail_curve", dark, selfcheck)
            fig.savefig(fn, dpi=200)
            plt.close(fig)
            print(f"[PLOT] {fn}")

    _tend("step23.longtail", t_start)
    return out

# ----------------------------- E. Lexicon dependency -------------------------
def analyze_lexicon_dependency(selfcheck: bool) -> Optional[pd.DataFrame]:
    """
    If harm_category_by_group.csv exists: summarize mean/max harm prevalence (pct)
    across groups and report BW prevalence if row present. If missing, delete stale
    lim* legacy artefacts and return None.
    """
    t = _t0("E. Lexicon Dependency (HurtLex) ...")
    if not HARM_WIDE.exists():
        print("[INFO] harm_category_by_group.csv not found; deleting stale outputs for E.")
        _delete_stale(
            [
                _out_csv("lim23_hurtlex_dependency.csv", False),
                _out_csv("lim23_hurtlex_dependency.csv", True),
                _out_tex("lim23_hurtlex_dependency.tex", False),
                _out_tex("lim23_hurtlex_dependency.tex", True),
            ]
        )
        _tend("step23.hurtlex", t)
        return None

    wide = pd.read_csv(HARM_WIDE)
    gcol = "Group" if "Group" in wide.columns else ("group" if "group" in wide.columns else None)
    if gcol is None:
        print("[INFO] group column not found in harm wide; skipping E.")
        _tend("step23.hurtlex", t)
        return None

    harm_cols = [c for c in wide.columns if c != gcol]
    df = wide.copy()
    df["mean_prevalence_pct"] = df[harm_cols].astype(float).mean(axis=1)
    df["max_prevalence_pct"] = df[harm_cols].astype(float).max(axis=1)

    key = None
    gl = df[gcol].astype(str).str.lower()
    for cand in ["intersectional_black_female", "black_women", "black_woman", "black_female"]:
        if cand in set(gl):
            key = cand
            break

    if key is not None:
        bw_row = df[gl == key]
        bw_mean = float(bw_row["mean_prevalence_pct"].iloc[0])
        bw_max = float(bw_row["max_prevalence_pct"].iloc[0])
    else:
        bw_mean = np.nan
        bw_max = np.nan

    out = pd.DataFrame(
        [
            {
                "n_groups": int(df.shape[0]),
                "mean_prevalence_pct_median": float(df["mean_prevalence_pct"].median()),
                "max_prevalence_pct_median": float(df["max_prevalence_pct"].median()),
                "bw_mean_prevalence_pct": bw_mean,
                "bw_max_prevalence_pct": bw_max,
            }
        ]
    )
    _write_csv(out, _out_csv("23_hurtlex_dependency.csv", selfcheck))
    _write_tex_table(out, _out_tex("23_hurtlex_dependency.tex", selfcheck))
    _tend("step23.hurtlex", t)
    return out

# ----------------------------- F. Checklist synthesis ------------------------
def synthesize_checklist(
    miss: pd.DataFrame,
    rep: pd.DataFrame,
    drift: pd.DataFrame,
    longtail: pd.DataFrame,
    lex: Optional[pd.DataFrame],
    selfcheck: bool,
) -> pd.DataFrame:
    """
    Combine signals into a checklist with heuristic flags to guide discussion.
    """
    t = _t0("F. Checklist synthesis ...")
    missing_flag = miss["missing_fraction"].max() > 0.20 if not miss.empty else True

    entropy_low = False
    if not rep.empty and "entropy_norm" in rep.columns:
        entropy_low = rep["entropy_norm"].min() < 0.80

    drift_flag = False
    if not drift.empty and "bw_share_slope_per_year" in drift.columns:
        drift_flag = abs(float(drift["bw_share_slope_per_year"].iloc[0])) > 0.005

    longtail_flag = False
    if not longtail.empty and (longtail["K"] == 10).any():
        k10 = float(longtail.loc[longtail["K"] == 10, "mass_captured"].iloc[0])
        longtail_flag = k10 < 0.50

    lex_flag = False
    if lex is not None and not lex.empty:
        lex_flag = float(lex["max_prevalence_pct_median"].iloc[0]) >= 10.0

    simpson_flip = bool(drift["simpson_flip_detected"].iloc[0]) if ("simpson_flip_detected" in drift.columns and not drift.empty) else False

    rows = [
        {"issue": "High missingness in key fields (>20%)", "flag": bool(missing_flag), "metric_detail": float(miss["missing_fraction"].max()) if not miss.empty else np.nan},
        {"issue": "Strong representation imbalance (entropy_norm < 0.80)", "flag": bool(entropy_low), "metric_detail": float(rep["entropy_norm"].min()) if not rep.empty else np.nan},
        {"issue": "Temporal drift in BW share (|slope| > 0.005/yr)", "flag": bool(drift_flag), "metric_detail": float(drift["bw_share_slope_per_year"].iloc[0]) if not drift.empty else np.nan},
        {"issue": "Severe long-tail (Top-10 categories cover < 50%)", "flag": bool(longtail_flag), "metric_detail": float(longtail.loc[longtail["K"] == 10, "mass_captured"].iloc[0]) if ((not longtail.empty) and (longtail["K"] == 10).any()) else np.nan},
        {"issue": "HurtLex dependency high (median max-prevalence ≥ 10%)", "flag": bool(lex_flag), "metric_detail": float(lex["max_prevalence_pct_median"].iloc[0]) if (lex is not None and not lex.empty) else np.nan},
        {"issue": "Simpson’s flip detected (overall vs per-year correlation sign differs)", "flag": bool(simpson_flip), "metric_detail": float(drift["overall_corr_bw_vs_views_per_day"].iloc[0]) if ("overall_corr_bw_vs_views_per_day" in drift.columns and not drift.empty) else np.nan},
    ]
    out = pd.DataFrame(rows)
    _write_csv(out, _out_csv("23_summary_checklist.csv", selfcheck))
    _write_tex_table(out, _out_tex("23_summary_checklist.tex", selfcheck))
    _tend("step23.checklist", t)
    return out

# ----------------------------- Narrative -------------------------------------
def _df_to_markdown_safe(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        # fallback if markdown dependency isn't available
        return "```\n" + df.to_string(index=False) + "\n```"

def write_narrative(
    miss: pd.DataFrame,
    rep: pd.DataFrame,
    drift: pd.DataFrame,
    longtail: pd.DataFrame,
    lex: Optional[pd.DataFrame],
    checklist: pd.DataFrame,
    selfcheck: bool,
) -> None:
    """
    Write a short, literature-cued markdown narrative with the checklist embedded.
    """
    t = _t0("Narrative write-up ...")
    md = [
        "# Step 23 — Limitations Analysis",
        "",
        "This report quantifies key threats to validity and fairness:",
        "1) **Data quality** — missingness in core fields and protected attributes.",
        "2) **Representation** — coverage & balance across groups (entropy/Gini).",
        "3) **Temporal drift** — changes in Black-women share over time; Simpson’s effect.",
        "4) **Category sparsity** — reliance on a long-tail taxonomy and its implications.",
        "5) **Lexicon dependency** — reliance on HurtLex categories (proxy).",
        "",
        "## Checklist (Flags & Metrics)",
        _df_to_markdown_safe(checklist) if not checklist.empty else "_No checklist available._",
        "",
        "## Notes & Mitigations (literature-informed)",
        "- **High missingness** → impute with uncertainty; report explicitly (Datasheets; Model Cards).",
        "- **Representation imbalance** → stratified evaluation; reweighing; careful augmentation (Suresh & Guttag).",
        "- **Temporal drift** → time-aware splits; report year-conditioned metrics; monitor drift regularly.",
        "- **Long-tail** → validate head vs tail; avoid category-only features driving bias.",
        "- **Lexicon dependency** → combine lexicons with contextual models; audit with manual samples.",
        "",
        f"*Seed={SEED}. Figures in `outputs/figures/(dark|light)`; tables in `dissertation/auto_tables`. "
        f"Self-check mode writes `_selfcheck` artefacts only.*",
        "",
        "_Reminder: category totals can exceed N due to multi-label assignment. "
        "Non-English titles are common; tags/categories remain interpretable (MPU)._",
                "## Quick interpretation notes",
        "- **Missingness**: bars near 0 mean those fields are complete; ≥0.2 flags data quality risk.",
        "- **BW Share Over Time**: upward slope = increasing representation; we also report Simpson’s flip check.",
        "",

    ]
    outp = _out_md("23_limitations_analysis.md", selfcheck)
    outp.write_text("\n".join(md), encoding="utf-8")
    print(f"[WRITE] {outp}")
    _tend("step23.narrative", t)

# ----------------------------- Qualitative readout ---------------------------
def qualitative_readout(
    miss: pd.DataFrame,
    rep: pd.DataFrame,
    drift: pd.DataFrame,
    longtail: pd.DataFrame,
    lex: Optional[pd.DataFrame],
) -> None:
    """
    Print compact qualitative analysis, highlighting outliers for quick inspection.
    """
    print("\n--- Quick qualitative readout ---")

    if not miss.empty:
        worst = miss.sort_values("missing_fraction", ascending=False).iloc[0]
        print(f"• Most missing: {worst['field']:<16}  missing={worst['missing_fraction']*100:.0f}% (non-null n={int(worst['non_null_n'])})")

    if not rep.empty:
        for dim in ["race_ethnicity", "gender"]:
            row = rep[rep["dimension"] == dim]
            if not row.empty:
                r = row.iloc[0]
                print(
                    f"• {dim}: entropy_norm={float(r['entropy_norm']):.2f}, gini={float(r['gini']):.2f}, "
                    f"min={float(r['min_share'])*100:.0f}%, max={float(r['max_share'])*100:.0f}%"
                )

    if not drift.empty:
        print(
            f"• BW share slope: {float(drift['bw_share_slope_per_year'].iloc[0]):+0.4f}/yr; "
            f"Simpson’s flip: {bool(drift['simpson_flip_detected'].iloc[0])}"
        )

    if not longtail.empty and (longtail["K"] == 10).any() and (longtail["K"] == 50).any():
        k10 = longtail.loc[longtail["K"] == 10, "mass_captured"].iloc[0]
        k50 = longtail.loc[longtail["K"] == 50, "mass_captured"].iloc[0]
        if np.isfinite(k10) and np.isfinite(k50):
            print(f"• Top-K coverage rises from {k10*100:.0f}% (K=10) to {k50*100:.0f}% (K=50); diminishing returns beyond ~30 are typical.")

    if lex is not None and not lex.empty:
        print(
            f"• HurtLex dependency (median across groups): mean={float(lex['mean_prevalence_pct_median'].iloc[0]):.1f}%, "
            f"max={float(lex['max_prevalence_pct_median'].iloc[0]):.1f}%."
        )

    print(
        "*Note:* category totals can exceed N because categories are multi-label assignment. "
        "Titles may be non-English; categories/tags keep semantics interpretable (MPU)."
    )

# ----------------------------- Legacy cleanup --------------------------------
def _delete_legacy_lim23(selfcheck: bool) -> None:
    """Remove old lim23_* artefacts to avoid duplication after renaming to 23_*."""
    # CSV + TEX
    legacy_csv = [
        "lim23_missingness.csv",
        "lim23_representation_skew.csv",
        "lim23_temporal_drift.csv",
        "lim23_category_longtail.csv",
        "lim23_hurtlex_dependency.csv",
        "lim23_summary_checklist.csv",
    ]
    for base in legacy_csv:
        for sc in (False, True):
            _delete_stale([_out_csv(base, sc), _out_tex(base.replace(".csv", ".tex"), sc)])
    # FIGS
    fig_bases = [
        "lim23_missingness_bar",
        "lim23_representation_bar_race_ethnicity",
        "lim23_representation_bar_gender",
        "lim23_temporal_share",
        "lim23_longtail_curve",
    ]
    for b in fig_bases:
        for sc_suf in (["", "_selfcheck"]):
            _delete_stale([FIG_DARK / f"{b}_dark{sc_suf}.png", FIG_LIGHT / f"{b}_light{sc_suf}.png"])

# ----------------------------- Orchestrator ----------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Orchestrate Step 23 with timers, self-check, robust optional artefacts, and narrative.

    CLI:
      python -m src.analysis.23_limitations_analysis
      python -m src.analysis.23_limitations_analysis --selfcheck --sample-rows 150000
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true", help="Write *_selfcheck artefacts; do not overwrite canonical files.")
    ap.add_argument("--sample-rows", type=int, default=None, help="Optional row cap for self-check (random sample; seeded).")
    args = ap.parse_args(argv)

    t_all = _t0("--- Starting Step 23: Limitations Analysis ---")

    # load + prep
    max_rows = args.sample_rows if args.sample_rows is not None else (150_000 if args.selfcheck else None)
    df = load_minimal_corpus(max_rows, random_sample=bool(args.selfcheck))
    df = prep_corpus(df)

    # delete legacy lim23_* artefacts (one-time clean as we switch to number-first)
    _delete_legacy_lim23(selfcheck=args.selfcheck)

    # analyses
    miss = analyze_missingness(df, selfcheck=args.selfcheck)
    rep = analyze_representation(df, selfcheck=args.selfcheck)
    drift = analyze_temporal_drift(df, selfcheck=args.selfcheck)
    longtail = analyze_longtail(df, selfcheck=args.selfcheck)
    lex = analyze_lexicon_dependency(selfcheck=args.selfcheck)

    # checklist + narrative
    checklist = synthesize_checklist(miss, rep, drift, longtail, lex, selfcheck=args.selfcheck)
    write_narrative(miss, rep, drift, longtail, lex, checklist, selfcheck=args.selfcheck)

    # console readout
    qualitative_readout(miss, rep, drift, longtail, lex)

    _tend("step23.total_runtime", t_all)
    print("--- Step 23: Limitations Analysis Completed Successfully ---")

if __name__ == "__main__":
    main()
