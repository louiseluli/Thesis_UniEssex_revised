#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_network_analysis.py
======================

Purpose
-------
Category co-occurrence network analytics (top-K categories) with interpretable outputs:
- Degree, strength, eigenvector centrality, PageRank, clustering coefficient
- Correlate network strength with Black-women log2 over/under-representation (from Step 18)

Design
------
- Reads cgd_category_cooccurrence.csv (Step 18). If missing, rebuilds from corpus.
- Self-check mode: random subgraph with *_selfcheck artefacts (non-destructive).
- Matplotlib only; dual-theme figures (dark/light).
- Timers per block and total runtime; config-seeded randomness.

CLI
---
# Full run (canonical artefacts)
python -m src.analysis.20_network_analysis

# Self-check (random subgraph; safe suffixing)
python -m src.analysis.20_network_analysis --selfcheck --sample-k 60
"""

from __future__ import annotations

# ----------------------------- Imports (top) ---------------------------------
import sys
import time
from ast import literal_eval
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project utils (config/theme); theme_manager.load_config prints its own [TIME]
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config  # noqa: E402

# ----------------------------- Config & paths --------------------------------
CONFIG = load_config()
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
np.random.seed(SEED)

ROOT = Path(CONFIG["project"]["root"])
DATA_DIR = Path(CONFIG["paths"]["data"])
FIG_DIR = Path(CONFIG["paths"]["figures"])
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = ROOT / "dissertation" / "auto_tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
NARR_DIR = Path(CONFIG["paths"]["narratives"]) / "automated"
NARR_DIR.mkdir(parents=True, exist_ok=True)

COOC_PATH = DATA_DIR / "cgd_category_cooccurrence.csv"
BW_PATH = DATA_DIR / "cgd_black_women_under_over.csv"
CORPUS_CANDIDATES = [
    DATA_DIR / "01_ml_corpus.parquet",
    DATA_DIR / "ml_corpus.parquet",
    DATA_DIR / "01_ml_corpus.snappy.parquet",
]
TOP_K_FALLBACK = int(CONFIG.get("analysis", {}).get("top_k_categories", 30))
EPS = 1e-12

# ----------------------------- Timers ----------------------------------------
def _t0(msg: str) -> float:
    """Start a timer with a standard message."""
    print(msg)
    return time.perf_counter()

def _tend(label: str, t_start: float) -> None:
    """End a timer and print standardized [TIME] line."""
    print(f"[TIME] {label}: {time.perf_counter() - t_start:.2f}s")

# ----------------------------- Plot theme ------------------------------------
def set_mpl_theme(dark: bool) -> None:
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
        "savefig.edgecolor": "black" if dark else "white",
        "grid.color": "gray",
        "grid.alpha": 0.25,
    })

# ----------------------------- Small utils -----------------------------------
def _round(x: float, k: int = 3) -> float:
    """
    Safe rounding to k decimals.
    """
    try:
        return round(float(x), k)
    except Exception:
        return x

def _first_existing(cands: List[Path]) -> Optional[Path]:
    """
    Return the first path that exists among candidates.
    """
    for p in cands:
        if p.exists():
            return p
    return None

def _parse_listish(x) -> List[str]:
    """
    Parse list-like strings/structures to a cleaned list of lowercase tokens.
    Handles python-list-in-string, comma/pipe-separated, and true sequences.
    (Uses ast.literal_eval safely; never eval.)
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

def _category_series_from_df(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Return a Series of list[str] categories/tags per row, using config-preferred
    column names if available, then fallbacks.
    """
    prefer = []
    cols_cfg = CONFIG.get("columns", {})
    for k in ("categories", "tags", "labels"):
        c = cols_cfg.get(k)
        if c:
            prefer.append(c)
    fallbacks = ["categories", "tags", "category", "tag_list", "labels"]
    for c in [*prefer, *fallbacks]:
        if c and c in df.columns:
            return df[c].apply(_parse_listish)
    return None

# ----------------------- Co-occurrence load / rebuild ------------------------
def rebuild_cooccurrence_from_corpus(top_k: int = TOP_K_FALLBACK) -> pd.DataFrame:
    """
    Rebuild category co-occurrence from the corpus when the Step-18 artefact is missing.
    Uses the most common top_k tokens from categories/tags and counts symmetric co-occurrences.
    """
    t = _t0("[INFO] Rebuilding co-occurrence from corpus (fallback).")
    corpus_path = _first_existing(CORPUS_CANDIDATES)
    if corpus_path is None:
        raise FileNotFoundError("No corpus parquet found; cannot rebuild co-occurrence.")
    df = pd.read_parquet(corpus_path)

    lists = _category_series_from_df(df)
    if lists is None:
        raise RuntimeError("No categories/tags columns found to rebuild co-occurrence.")

    # vocab
    counts = {}
    for lst in lists:
        for tkn in lst:
            counts[tkn] = counts.get(tkn, 0) + 1
    vocab = [w for w, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
    idx = {c: i for i, c in enumerate(vocab)}

    # matrix
    W = np.zeros((len(vocab), len(vocab)), dtype=int)
    for lst in lists:
        row = [c for c in set(lst) if c in idx]
        for a, b in combinations(sorted(row), 2):
            ia, ib = idx[a], idx[b]
            W[ia, ib] += 1
            W[ib, ia] += 1
    mat = pd.DataFrame(W, index=vocab, columns=vocab)
    mat.to_csv(COOC_PATH)
    print(f"✓ Artefact saved: {COOC_PATH}")
    _tend("step20.rebuild_cooccurrence", t)
    return mat

def load_cooccurrence() -> pd.DataFrame:
    """
    Load the co-occurrence matrix produced by Step 18; rebuild if missing.
    Ensures the matrix is symmetric with zero diagonal and numeric.
    """
    t = _t0("Loading category co-occurrence matrix ...")
    if COOC_PATH.exists():
        try:
            M = pd.read_csv(COOC_PATH, index_col=0)
        except Exception:
            M = pd.read_csv(COOC_PATH)
            M = M.set_index(M.columns[0])
    else:
        M = rebuild_cooccurrence_from_corpus(top_k=TOP_K_FALLBACK)

    M = M.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # enforce symmetry + zero diag
    if not (M.values.T == M.values).all():
        M = (M + M.T) / 2.0
    M_arr = M.values.copy()
    np.fill_diagonal(M_arr, 0.0)
    M = pd.DataFrame(M_arr, index=M.index, columns=M.columns)
    _tend("step20.load_cooccurrence", t)
    return M

# ----------------------------- Network metrics --------------------------------
def eigenvector_centrality(W: np.ndarray, tol: float = 1e-8, max_iter: int = 500) -> np.ndarray:
    """
    Power-iteration eigenvector centrality on weighted adjacency W.
    Returns a vector normalised to unit max.
    """
    n = W.shape[0]
    v = np.ones(n, dtype=float) / np.sqrt(n)
    for _ in range(max_iter):
        v_new = W @ v
        norm = np.linalg.norm(v_new)
        if norm == 0:
            return v
        v_new /= norm
        if np.linalg.norm(v_new - v) < tol:
            v = v_new
            break
        v = v_new
    v = v / (v.max() + EPS)
    return v

def pagerank(W: np.ndarray, d: float = 0.85, tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
    """
    Weighted PageRank with damping d on weighted adjacency W (row-stochastic).
    """
    n = W.shape[0]
    S = W.sum(axis=1, keepdims=True)
    P = np.divide(W, np.where(S == 0.0, 1.0, S), where=~(S == 0.0))
    r = np.ones(n, dtype=float) / n
    teleport = (1.0 - d) / n
    for _ in range(max_iter):
        r_new = teleport + d * (P.T @ r)
        if np.linalg.norm(r_new - r, 1) < tol:
            r = r_new
            break
        r = r_new
    r = r / (r.max() + EPS)
    return r

def clustering_coefficients(A: np.ndarray) -> np.ndarray:
    """
    Unweighted local clustering coefficient C_i = triangles_i / choose(k_i, 2),
    where triangles_i equals the number of edges among neighbors of node i.
    """
    n = A.shape[0]
    C = np.zeros(n, dtype=float)
    for i in range(n):
        nbrs = np.where(A[i] > 0)[0]
        k = nbrs.size
        if k < 2:
            C[i] = 0.0
            continue
        sub = A[np.ix_(nbrs, nbrs)]
        # number of edges among neighbors (each forms a triangle with i)
        tri = (np.triu(sub, 1) > 0).sum()
        C[i] = (2.0 * tri) / (k * (k - 1))
    return C

# ----------------------------- Analysis helpers ------------------------------
def compute_centrality_table(M: pd.DataFrame) -> pd.DataFrame:
    """
    Compute degree, strength, eigenvector, PageRank, and clustering on M.
    Returns a tidy DataFrame sorted by strength (descending).
    """
    t = _t0("Computing centralities ...")
    categories = list(M.index.astype(str))
    W = M.to_numpy(dtype=float)
    W = np.maximum(W, W.T); np.fill_diagonal(W, 0.0)
    A = (W > 0).astype(int)

    degree = A.sum(axis=1).astype(int)
    strength = W.sum(axis=1)
    eig = eigenvector_centrality(W)
    pr = pagerank(W, d=0.85)
    cc = clustering_coefficients(A)

    out = pd.DataFrame({
        "category": categories,
        "degree": degree,
        "strength": np.round(strength, 3),
        "eigencentrality": np.round(eig, 4),
        "pagerank": np.round(pr, 4),
        "clustering": np.round(cc, 4),
    }).sort_values("strength", ascending=False)
    _tend("step20.compute_centralities", t)
    return out

def merge_bw_log2rr(df_c: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Merge centrality table with Black-women log2 RR per category (if available).
    Returns (merged_df, pearson_r or None).
    """
    t = _t0("Merging BW log2 RR and computing correlation ...")
    corr = None
    if BW_PATH.exists():
        bw = pd.read_csv(BW_PATH)
        if {"category", "log2_rr_bw"}.issubset(set(bw.columns)):
            merged = df_c.merge(bw[["category", "log2_rr_bw"]], on="category", how="left")
            x = merged["strength"].to_numpy(float)
            y = merged["log2_rr_bw"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 3:
                corr = float(np.corrcoef(x[m], y[m])[0, 1])
            _tend("step20.merge_bw", t)
            return merged, corr
    df_c["log2_rr_bw"] = np.nan
    _tend("step20.merge_bw", t)
    return df_c, corr

def plot_top_strength(df: pd.DataFrame, *, suffix: str = "") -> None:
    """
    Plot top-20 categories by strength (dual theme).
    """
    t = _t0("Plotting top-20 by strength ...")
    top = df.nlargest(20, "strength").copy()
    names = [c[:28] for c in top["category"]]
    vals = top["strength"].astype(float).to_numpy()
    for dark in (True, False):
        set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top)), vals[::-1])
        ax.set_yticks(range(len(top))); ax.set_yticklabels(names[::-1])
        ax.set_xlabel("Strength (sum of co-occurrence weights)")
        ax.set_title("Top-20 categories by network strength")
        ax.grid(True, axis="x", alpha=0.3)
        fname = FIG_DIR / f"20_top_strength_{'dark' if dark else 'light'}{suffix}.png"
        fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
        print(f"✓ Artefact saved: {fname}")
    _tend("step20.plot_top_strength", t)

def plot_strength_vs_bw(df: pd.DataFrame, *, suffix: str = "") -> None:
    """
    Scatter: strength vs BW log2 RR, when RR is available.
    """
    if "log2_rr_bw" not in df.columns or not df["log2_rr_bw"].notna().any():
        return
    t = _t0("Plotting strength vs BW log2 RR ...")
    x = df["strength"].astype(float).to_numpy()
    y = df["log2_rr_bw"].astype(float).to_numpy()
    for dark in (True, False):
        set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        ax.scatter(x, y, s=20)
        ax.axhline(0.0, ls="--", lw=1)
        ax.set_xlabel("Strength"); ax.set_ylabel("Black-women log2 RR")
        ax.set_title("Centrality vs Black-women over/under-representation")
        ax.grid(True, alpha=0.3)
        fname = FIG_DIR / f"20_strength_vs_bw_log2rr_{'dark' if dark else 'light'}{suffix}.png"
        fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
        print(f"✓ Artefact saved: {fname}")
    _tend("step20.plot_strength_vs_bw", t)

def qualitative_readout(df: pd.DataFrame, corr: Optional[float]) -> None:
    """
    Print a compact qualitative readout, highlighting outliers and interpretation hooks.
    Non-English titles are common; category tags anchor meaning.
    """
    print("\n--- Quick qualitative readout ---")
    if not df.empty:
        top = df.head(5)[["category", "strength", "degree", "clustering"]]
        print("• Top categories by strength (5):")
        for _, r in top.iterrows():
            print(f"   {r['category'][:28]:<28} | strength={float(r['strength']):.1f} | "
                  f"deg={int(r['degree'])} | C={float(r['clustering']):.2f}")
        if "log2_rr_bw" in df.columns and df["log2_rr_bw"].notna().any():
            # Outliers: central but under-represented
            central = df.nlargest(20, "strength")
            under = central.nsmallest(5, "log2_rr_bw")
            print("• Under-represented among central categories (log2 RR, 5):")
            for _, r in under.iterrows():
                print(f"   {r['category'][:28]:<28} | log2RR={float(r['log2_rr_bw']):.3f} | strength={float(r['strength']):.1f}")
    if corr is not None:
        sign = "negative" if corr < 0 else ("positive" if corr > 0 else "near-zero")
        print(f"• Corr(strength, BW log2 RR): {corr:.3f} ({sign})")
    print("*Note:* category totals can exceed N because categories are multi-label per item. "
          "Some titles are not in English; tags/categories (MPU) anchor semantics.")

# ----------------------------- Orchestrator ----------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Run Step 20 end-to-end.

    Options
    -------
    --selfcheck          Randomly sample a subgraph of categories (non-destructive).
    --sample-k  INT      Number of categories to keep in self-check (default 60).
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample-k", type=int, default=60)
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    print("--- Starting Step 20: Network Analysis ---")

    # 1) Load co-occurrence
    M = load_cooccurrence()

    # 1a) Self-check: random subgraph selection (suffix outputs; do not overwrite)
    suffix = ""
    if args.selfcheck:
        k = min(int(args.sample_k), M.shape[0])
        rng = np.random.default_rng(SEED)
        keep_idx = sorted(rng.choice(np.arange(M.shape[0]), size=k, replace=False).tolist())
        M = M.iloc[keep_idx, keep_idx]
        suffix = "_selfcheck"
        print(f"[SELF-CHECK] Random subgraph: {k} categories (seed={SEED}).")

    # 2) Metrics
    df_c = compute_centrality_table(M)

    # 3) Merge BW log2 RR + corr
    df_c, corr = merge_bw_log2rr(df_c)

    # 4) Save tables
    t = _t0("Saving tables (CSV/TEX) ...")
    csv = DATA_DIR / f"20_category_centrality{suffix}.csv"
    tex = TABLE_DIR / f"20_category_centrality{suffix}.tex"
    df_c.to_csv(csv, index=False); print(f"✓ Artefact saved: {csv}")
    try:
        with open(tex, "w", encoding="utf-8") as f:
            f.write(df_c.to_latex(index=False))
        print(f"✓ Artefact saved: {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")
    _tend("step20.save_tables", t)

    # 5) Figures
    plot_top_strength(df_c, suffix=suffix)
    plot_strength_vs_bw(df_c, suffix=suffix)

    # 6) Narrative
    t = _t0("Writing narrative ...")
    lines = [
        "# 20 — Network Analysis of Category Co-occurrence",
        "- Degree/strength/eigenvector/PageRank/clustering computed on the top-K category graph.",
        f"- Correlation(strength, BW log2 RR) = {('n/a' if corr is None else f'{corr:.3f}')} "
        "(negative values suggest under-representation in central categories).",
        "- Dual-theme figures saved. Seed from config; Matplotlib only.",
        "- Multi-label categories can make totals exceed N; non-English titles are common, "
        "but tags/categories anchor semantics (MPU).",
    ]
    md = NARR_DIR / f"20_network_analysis_summary{suffix}.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✓ Narrative saved: {md}")
    _tend("step20.write_narrative", t)

    _tend("step20.total_runtime", t_all)
    print("--- Step 20: Network Analysis Completed Successfully ---")


if __name__ == "__main__":
    main()
