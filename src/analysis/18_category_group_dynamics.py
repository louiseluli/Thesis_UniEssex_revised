#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_category_group_dynamics.py
=============================

Purpose
-------
Category × group dynamics with intersectional focus on Black women.
Robust category parsing, representation ratios with Laplace smoothing,
co-occurrence among top-K categories, interpretable outputs (CSV/TEX/PNG),
timers, self-check mode (non-destructive), and a concise qualitative readout.

CLI
---
# Full run
python -m src.analysis.18_category_group_dynamics

# Self-check (random subsample; writes *_selfcheck artefacts only)
python -m src.analysis.18_category_group_dynamics --selfcheck --sample 120000 --top-k 30
"""
from __future__ import annotations

import sys
import time
from ast import literal_eval
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project utils
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config  # prints its own [TIME]

# --- Config & paths ----------------------------------------------------------
_t0_cfg = time.perf_counter()
CONFIG = load_config()
print(f"[TIME] theme_manager.load_config: {time.perf_counter() - _t0_cfg:.2f}s")

SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
np.random.seed(SEED)

ROOT      = Path(CONFIG["project"]["root"])
DATA_DIR  = Path(CONFIG["paths"]["data"])         # outputs/data
FIG_DIR   = Path(CONFIG["paths"]["figures"])      # outputs/figures
TABLE_DIR = ROOT / "dissertation" / "auto_tables"
NARR_DIR  = Path(CONFIG["paths"]["narratives"]) / "automated"
for d in (DATA_DIR, FIG_DIR, TABLE_DIR, NARR_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Accept both step-01 and canonical names
CORPUS_CANDIDATES = [
    DATA_DIR / "01_ml_corpus.parquet",
    DATA_DIR / "ml_corpus.parquet",
    DATA_DIR / "01_ml_corpus.snappy.parquet",
]

LAPLACE = 1.0  # smoothing for proportions/ratios

# --- Timers ------------------------------------------------------------------
def _t0(msg: str) -> float:
    """Start a timer and print a standardized header."""
    print(msg)
    return time.perf_counter()

def _tend(label: str, t0: float) -> None:
    """Stop a timer and print a standardized [TIME] line."""
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")

# --- Theme (Matplotlib only) -------------------------------------------------
def _set_mpl_theme(dark: bool) -> None:
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
        "grid.color": "gray",
        "grid.alpha": 0.25,
    })

# --- Small utils -------------------------------------------------------------
def _round(x: float, k: int = 3) -> float:
    """Round floats safely; returns x unchanged on failure."""
    try:
        return round(float(x), k)
    except Exception:
        return x

def _first_existing(paths: List[Path]) -> Optional[Path]:
    """Return the first existing path among candidates."""
    for p in paths:
        if p.exists():
            return p
    return None

# --- IO helpers --------------------------------------------------------------
def _load_corpus() -> pd.DataFrame:
    """Load corpus parquet with a timing line; tolerate several canonical names."""
    path = _first_existing(CORPUS_CANDIDATES)
    if path is None:
        exp = ", ".join(p.name for p in CORPUS_CANDIDATES)
        raise FileNotFoundError(f"Corpus not found under {DATA_DIR}. Expected one of: {exp}")
    t0 = _t0(f"[READ] {path}")
    df = pd.read_parquet(path)
    _tend("step18.load_corpus", t0)
    return df

def _prepare_protected_columns(df: pd.DataFrame) -> List[str]:
    """Create tidy 'race_ethnicity'/'gender' if only one-hot is present. Return protected columns."""
    def _derive_categorical_from_onehot(d: pd.DataFrame, prefix: str, out_col: str, fallback: str = "unknown") -> Optional[str]:
        pref = f"{prefix}_"
        onehot_cols = [c for c in d.columns if c.lower().startswith(pref)]
        if not onehot_cols:
            return None
        labels = [c[len(pref):].lower() for c in onehot_cols]
        mixed_token = "mixed_or_other" if any(l == "mixed_or_other" for l in labels) else "mixed"
        oh = d[onehot_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        oh = (oh > 0.5).astype(int)
        vals = []
        for i in range(len(d)):
            row = oh.iloc[i].values
            hits = np.where(row == 1)[0]
            if len(hits) == 1:
                vals.append(labels[hits[0]])
            elif len(hits) > 1:
                vals.append(mixed_token)
            else:
                vals.append(fallback)
        d[out_col] = pd.Series(vals, index=d.index, dtype="object")
        return out_col

    def _norm_gender_label(x: str) -> str:
        x = (x or "").lower()
        if x in {"female", "woman", "women", "cis_female", "cis_woman"}: return "female"
        if x in {"male", "man", "men", "cis_male", "cis_man"}:          return "male"
        return x or "unknown"

    if "race_ethnicity" not in df.columns:
        _derive_categorical_from_onehot(df, "race_ethnicity", "race_ethnicity")
    if "gender" not in df.columns:
        _derive_categorical_from_onehot(df, "gender", "gender")
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

# --- Category parsing --------------------------------------------------------
def _parse_listish(x) -> List[str]:
    """Parse list-like strings or sequences into a list of cleaned category tokens."""
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

def _extract_categories(df: pd.DataFrame) -> Optional[pd.Series]:
    """Try config-preferred then likely fields for categories/tags. Return Series[list[str]] or None."""
    cols_cfg = CONFIG.get("columns", {})
    prefer = [cols_cfg.get(k) for k in ("categories", "tags", "labels") if cols_cfg.get(k)]
    fallbacks = ["categories", "tags", "category", "tag_list", "labels"]
    for c in [*prefer, *fallbacks]:
        if c and c in df.columns:
            return df[c].apply(_parse_listish)
    print("[INFO] No category/tags fields found; skipping category-group dynamics.")
    return None

# --- Core computations -------------------------------------------------------
def _top_k_categories(cat_lists: pd.Series, k: int) -> List[str]:
    """Return top-K category tokens by frequency."""
    cnt = Counter()
    for lst in cat_lists:
        cnt.update(lst)
    return [w for w, _ in cnt.most_common(k)]

def _explode_categories(df: pd.DataFrame, cat_lists: pd.Series, keep_only: Optional[List[str]] = None) -> pd.DataFrame:
    """Explode list-like categories to long format; optionally filter to top-K."""
    tmp = df.copy()
    tmp["_categories"] = cat_lists
    tmp = tmp.loc[tmp["_categories"].map(bool)]
    tmp = tmp.explode("_categories")
    if keep_only:
        tmp = tmp.loc[tmp["_categories"].isin(keep_only)]
    return tmp

def category_group_matrix(
    df: pd.DataFrame,
    cat_lists: pd.Series,
    group_col: str,
    *,
    top_k: int,
    suffix: str = ""
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build Category × Group matrix with counts, shares, smoothed representation ratios and log2 rr.
    Saves CSV/TEX and heatmap (dual theme). Returns the raw matrix and the top-K list.
    """
    t0 = _t0("Building Category × Group matrix ...")
    top = _top_k_categories(cat_lists, k=top_k)
    ex  = _explode_categories(df, cat_lists, keep_only=top)

    grp = ex.groupby(["_categories", group_col]).size().reset_index(name="count")
    total_cat   = grp.groupby("_categories")["count"].transform("sum")
    group_total = grp.groupby(group_col)["count"].transform("sum")
    grand_total = float(grp["count"].sum())
    G = grp[group_col].nunique()

    p_g_given_c = (grp["count"] + LAPLACE) / (total_cat + LAPLACE * G)
    p_g         = (group_total + LAPLACE) / (grand_total + LAPLACE * G)
    rr = p_g_given_c / p_g
    log2_rr = np.log2(rr)

    out = grp.copy()
    out["share_in_category"]   = (grp["count"] / total_cat).astype(float)
    out["global_group_share"]  = (group_total / grand_total).astype(float)
    out["repr_ratio"]          = rr.astype(float)
    out["log2_rr"]             = log2_rr.astype(float)
    out = out.sort_values(["_categories", "log2_rr"], ascending=[True, False])

    exp = out.copy()
    for c in ["share_in_category","global_group_share","repr_ratio","log2_rr"]:
        exp[c] = exp[c].map(lambda x: _round(x, 3))
    exp.rename(columns={"_categories": "category"}, inplace=True)

    csv = DATA_DIR / f"18_category_group_matrix{suffix}.csv"
    tex = TABLE_DIR / f"18_category_group_matrix{suffix}.tex"
    exp.to_csv(csv, index=False); print(f"✓ Artefact saved: {csv}")
    try:
        with open(tex, "w", encoding="utf-8") as f:
            f.write(exp.to_latex(index=False))
        print(f"✓ Artefact saved: {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    pv = out.pivot(index=group_col, columns="_categories", values="log2_rr").reindex(columns=top)
    for dark in (True, False):
        _set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(min(18, 1.0 + 0.5 * len(top)), 6))
        im = ax.imshow(pv.to_numpy(dtype=float), aspect="auto", origin="upper", cmap="coolwarm", vmin=-2, vmax=2)
        ax.set_xticks(range(len(top))); ax.set_xticklabels([t[:22] for t in top], rotation=45, ha="right")
        ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
        ax.set_title(f"Category × {group_col} — log2 over/under-representation (smoothed)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log2 ratio")
        fig.tight_layout()
        fpath = FIG_DIR / f"18_heatmap_log2rr_{'dark' if dark else 'light'}{suffix}.png"
        fig.savefig(fpath, dpi=200); plt.close(fig)
        print(f"✓ Artefact saved: {fpath}")
    _tend("step18.cgd_matrix", t0)
    return out, top

def black_women_under_over(
    df: pd.DataFrame,
    cat_lists: pd.Series,
    top: List[str],
    *,
    suffix: str = ""
) -> pd.DataFrame:
    """
    Under/over-representation of Black women by category vs global baseline.
    Saves CSV/TEX and bar plots (dual theme).
    """
    t0 = _t0("Analyzing Black-women under/over-representation ...")
    if "race_ethnicity" not in df.columns or "gender" not in df.columns:
        print("[INFO] Missing race_ethnicity or gender; skipping BW under/over analysis.")
        return pd.DataFrame()

    ex = _explode_categories(df, cat_lists, keep_only=top)
    is_bw = (ex["race_ethnicity"].str.lower() == "black") & (ex["gender"].str.lower() == "female")

    cat_totals = ex.groupby("_categories").size().rename("n_cat")
    bw_counts  = ex.loc[is_bw].groupby("_categories").size().reindex(cat_totals.index, fill_value=0).rename("n_bw_cat")

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

    for c in ["share_bw_in_cat","global_share_bw","repr_ratio_bw","log2_rr_bw"]:
        out[c] = out[c].map(lambda x: _round(x, 3))

    csv = DATA_DIR / f"18_bw_under_over{suffix}.csv"
    tex = TABLE_DIR / f"18_bw_under_over{suffix}.tex"
    out.to_csv(csv, index=False); print(f"✓ Artefact saved: {csv}")
    try:
        with open(tex, "w", encoding="utf-8") as f:
            f.write(out.to_latex(index=False))
        print(f"✓ Artefact saved: {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    under = out.nsmallest(15, "log2_rr_bw")
    over  = out.nlargest(15, "log2_rr_bw")

    def _bar(df_sub: pd.DataFrame, title: str, fname: str):
        xs = df_sub["category"].tolist()
        ys = df_sub["log2_rr_bw"].astype(float).tolist()
        for dark in (True, False):
            _set_mpl_theme(dark=dark)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(xs)), ys)
            ax.set_yticks(range(len(xs))); ax.set_yticklabels([x[:28] for x in xs])
            ax.set_xlabel("log2 over/under-representation")
            ax.set_title(title)
            ax.axvline(0.0, linestyle="--", linewidth=1)
            ax.grid(True, axis="x", alpha=0.25)
            fig.tight_layout()
            f = FIG_DIR / f"{fname}_{'dark' if dark else 'light'}{suffix}.png"
            fig.savefig(f, dpi=200); plt.close(fig)
            print(f"✓ Artefact saved: {f}")

    _bar(under, "Most Under-represented Categories — Black Women", "18_bw_underrep_bar")
    _bar(over,  "Most Over-represented Categories — Black Women",  "18_bw_overrep_bar")
    _tend("step18.bw_under_over", t0)
    return out

def category_cooccurrence(
    cat_lists: pd.Series,
    top: List[str],
    *,
    suffix: str = ""
) -> pd.DataFrame:
    """
    Symmetric co-occurrence counts among top categories. Saves CSV only.
    """
    t0 = _t0("Computing category co-occurrence ...")
    idx = {c: i for i, c in enumerate(top)}
    mat = np.zeros((len(top), len(top)), dtype=int)
    for lst in cat_lists:
        row = [c for c in lst if c in idx]
        if len(set(row)) < 2:
            continue
        for a, b in combinations(sorted(set(row)), 2):
            ia, ib = idx[a], idx[b]
            mat[ia, ib] += 1
            mat[ib, ia] += 1
    out = pd.DataFrame(mat, index=top, columns=top)
    csv = DATA_DIR / f"18_category_cooccurrence{suffix}.csv"
    out.to_csv(csv); print(f"✓ Artefact saved: {csv}")
    _tend("step18.cooccurrence", t0)
    return out

# --- Orchestrator ------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Run Step 18 end-to-end.

    Options
    -------
    --selfcheck   Randomly subsample the corpus (non-destructive; *_selfcheck outputs).
    --sample INT  Subsample size for self-check (default: min(150k, N)).
    --top-k INT   How many categories to analyse (default: 30).
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--top-k", type=int, default=30)
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    print("--- Starting Step 18: Category–Group Dynamics ---")

    # Load
    df = _load_corpus()
    if args.selfcheck:
        n = args.sample or min(150_000, len(df))
        df = df.sample(n=n, random_state=SEED, replace=False).reset_index(drop=True)
        print(f"[SELF-CHECK] Random sample drawn: {len(df):,} rows (seed={SEED}).")

    # Protected attributes
    _ = _prepare_protected_columns(df)

    # Categories/tags
    t0 = _t0("Parsing categories/tags ...")
    cat_series = _extract_categories(df)
    _tend("step18.parse_categories", t0)
    if cat_series is None:
        print("--- Step 18: Skipped (no categories/tags found) ---")
        return

    suffix = "_selfcheck" if args.selfcheck else ""

    group_col = "race_ethnicity" if "race_ethnicity" in df.columns else ("gender" if "gender" in df.columns else "__all__")
    cgm, top = category_group_matrix(df, cat_lists=cat_series, group_col=group_col, top_k=args.top_k, suffix=suffix)
    bw = black_women_under_over(df, cat_lists=cat_series, top=top, suffix=suffix)
    _ = category_cooccurrence(cat_lists=cat_series, top=top, suffix=suffix)

    print("\n--- Qualitative analysis (compact) ---")
    if not bw.empty:
        print("• Top 5 over-represented categories for Black women (log2_rr_bw):")
        print(bw.nlargest(5, "log2_rr_bw")[["category","n_bw_cat","n_cat","log2_rr_bw"]].to_string(index=False))
        print("• Top 5 under-represented categories for Black women (log2_rr_bw):")
        print(bw.nsmallest(5, "log2_rr_bw")[["category","n_bw_cat","n_cat","log2_rr_bw"]].to_string(index=False))
        print("*Note:* totals can exceed N because categories are multi-label per item.")

    # Narrative
    t0 = _t0("Writing narrative ...")
    notes = [
        "# 18 — Category × Group Dynamics",
        f"- Grouping attribute: **{group_col}**; top-{args.top_k} categories considered.",
        "- We compute smoothed representation ratios log2(P(group|category)/P(group)). 0≈parity; <0 under-representation; >0 over-representation.",
        "- Black-women focus: most under/over-represented categories vs global baseline.",
        "- Co-occurrence matrix saved for downstream network visualisation.",
    ]
    md = NARR_DIR / f"18_category_group_dynamics_summary{suffix}.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(notes) + "\n")
    print(f"✓ Narrative saved: {md}")
    _tend("step18.write_narrative", t0)

    _tend("step18.total_runtime", t_all)
    print("\n--- Step 18: Completed ---")


if __name__ == "__main__":
    main()
