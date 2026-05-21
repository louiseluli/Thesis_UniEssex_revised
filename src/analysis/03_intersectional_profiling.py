# -*- coding: utf-8 -*-
"""
03_intersectional_profiling.py
==============================

Role
----
Intersectional text profiling via PMI for a target group (from config),
with publication-ready artefacts and dual-theme plots.

Goal (precise)
--------------
Compute PMI between 1–2 gram terms and a target intersectional group
(e.g., intersectional_black_female) using **document frequencies** (per-video).
Save full and top-N tables, figure, narrative, LaTeX. Provide a tiny reproducible
self-check that samples the REAL parquet (non-destructive, separate *_selfcheck).

What it is / what it does
-------------------------
1) Loads canonical Step-01 parquet, **seed/paths/names** from config (seed=95).
2) Builds binary 1–2-gram features on `combined_text_clean` (fallback: title+tags+categories).
3) Computes PMI = log2( P(term,group) / (P(term)*P(group)) ) using video-level DF.
4) Prints qualitative, interpretable console summary with outlier call-outs.
5) Saves artefacts under `03_*` naming; dual-theme plot, narrative, LaTeX.
6) Self-check uses a **random sample** (not hardcoded) and keeps separate outputs.
7) Lightweight timers for each step and **total file runtime**.

Interpretability notes
----------------------
- Titles may be non-English; **tags/categories help anchor semantics** (MPU tiles in other
  languages can be harder to interpret; lean on tags/categories).
- Counts are per-video; **totals can exceed N** due to multi-labels.

Artefacts (full run)
--------------------
Data:
  - outputs/data/03_pmi_<target_slug>.csv                 (top-N)
  - outputs/data/03_pmi_<target_slug>_full.csv            (full vocabulary)
Figure (dual theme):
  - outputs/figures/pmi/03_pmi_associations_bar_{light,dark}.png
Narrative:
  - outputs/narratives/automated/03_pmi_summary.md
LaTeX:
  - dissertation/auto_tables/03_pmi_<target_slug>.tex

Self-check (separate, non-destructive)
--------------------------------------
  - outputs/data/03_pmi_<target_slug>_selfcheck.csv
  - outputs/data/03_pmi_<target_slug>_full_selfcheck.csv
  - outputs/narratives/automated/03_pmi_summary_selfcheck.md
  - outputs/figures/pmi/03_pmi_associations_bar_selfcheck_{light,dark}.png

CLI
---
# FULL RUN
python3 src/analysis/03_intersectional_profiling.py

# SELF-CHECK (random sample from real parquet; seed from config)
python3 src/analysis/03_intersectional_profiling.py --selfcheck --sample 80000

Deletions called out
--------------------
- Removed hardcoded filenames like 'pmi_intersectional_black_women*.csv';
  now derive `<target_slug>` from config and prefix all artefacts with **03_**.

"""

from __future__ import annotations

# --- Imports (always at the top) ----------------------------------------------
import re
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import seaborn as sns

# Project utils (config + plotting theme + LaTeX helper)
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.utils.academic_tables import dataframe_to_latex_table


# --- 1) Config & reproducibility ---------------------------------------------
CONFIG = load_config()
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
np.random.seed(SEED)

CORPUS_PATH = Path(CONFIG["paths"]["data"]) / "01_ml_corpus.parquet"
FIG_DIR     = Path(CONFIG["paths"]["figures"]) / "pmi"
DATA_DIR    = Path(CONFIG["paths"]["data"])
NARR_DIR    = Path(CONFIG["paths"]["narratives"]) / "automated"
TEX_DIR     = Path(CONFIG["project"]["root"]) / "dissertation" / "auto_tables"

# Target group column name from config (e.g., 'intersectional_black_female')
PMI_TARGET_GROUP = str(CONFIG["project_specifics"]["intersection"]["output_col_name"])
TOP_N_TERMS_DEFAULT = 25


# --- 2) Lightweight timers ----------------------------------------------------
def _tstart(msg: str) -> float:
    """Start a perf counter and print a standardized header line."""
    t0 = time.perf_counter()
    print(msg)
    return t0

def _tend(label: str, t0: float) -> None:
    """Finish a timer with standardized [TIME] output in seconds."""
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")


# --- 3) Helpers ---------------------------------------------------------------
def _slugify_target(col_name: str) -> str:
    """
    Turn a target column name into a stable slug for filenames.

    Rules:
    - Lowercase, keep letters/digits/underscore; collapse runs of non-alnum to '_'.
    - Cosmetic: trailing '_female' -> '_women', '_male' -> '_men'.
    """
    base = re.sub(r"[^a-zA-Z0-9]+", "_", str(col_name).strip().lower()).strip("_")
    base = re.sub(r"_female$", "_women", base)
    base = re.sub(r"_male$", "_men", base)
    return base or "target"


def _ensure_text_column(df: pd.DataFrame, text_col: str = "combined_text_clean") -> str:
    """
    Ensure a text column exists for vectorization; fallback to title+tags+categories.

    Behavior
    --------
    - Uses `combined_text_clean` if present.
    - Fallback concatenates title/tags/categories, lowercases, normalizes separators
      (hyphens/underscores/slashes → space; collapse multiple spaces).

    Returns
    -------
    str
        The name of the column to be used for text.
    """
    if text_col in df.columns:
        return text_col

    # Fallback: build a lightweight, normalized text field
    tmp = (
        df.get("title", "").fillna("").astype(str) + " " +
        df.get("tags", "").fillna("").astype(str) + " " +
        df.get("categories", "").fillna("").astype(str)
    ).str.lower()

    # Normalize common delimiters that often glue words together
    tmp = (
        tmp
        .str.replace(r"[_/\\\-]+", " ", regex=True)   # hyphen/underscore/slash → space
        .str.replace(r"\s+", " ", regex=True)         # collapse whitespace
        .str.strip()
    )

    df[text_col] = tmp
    return text_col



def _load_corpus(path: Path) -> pd.DataFrame:
    """
    Load the canonical parquet (Step 01). Non-destructive; DB-free.
    """
    t0 = _tstart(f"Loading corpus from {path} ...")
    df = pd.read_parquet(path)
    _tend("pmi.load_corpus", t0)
    return df


# --- 4) PMI core --------------------------------------------------------------
def calculate_pmi(
    df: pd.DataFrame,
    target_group_col: str,
    text_col: str,
    *,
    min_df: int = 50,
    max_features: int = 30000
) -> pd.DataFrame:
    """
    Compute Pointwise Mutual Information (PMI) between terms and a target group.

    Parameters
    ----------
    df : pd.DataFrame
        Corpus with text and a binary column for the target group.
    target_group_col : str
        Column name with {0,1} membership for the group of interest.
    text_col : str
        Column containing the text to vectorize.
    min_df : int, default 50
        Minimum number of videos containing a term to keep it.
    max_features : int, default 30000
        Maximum vocabulary size (cap for speed/reproducibility).

    Returns
    -------
    pd.DataFrame
        Columns: ['Term','PMI','DF','DF_in_group','P(term)','P(group)',
                  'P(term,group)','P(term|group)'] sorted by PMI desc.
    """
    t0 = _tstart(
        f"Calculating PMI for target='{target_group_col}' on '{text_col}' "
        f"(min_df={min_df}, max_features={max_features})..."
    )

    # Vectorizer: binary DF per video, 1–2 grams. Accent stripping reduces weird splits.
    vec = CountVectorizer(
        stop_words="english",
        binary=True,
        ngram_range=(1, 2),
        token_pattern=r"\b[a-zA-Z0-9]+\b",
        strip_accents="unicode",
        lowercase=True,
        min_df=min_df,
        max_features=max_features,
    )
    X = vec.fit_transform(df[text_col].fillna(""))
    terms = vec.get_feature_names_out()

    N = X.shape[0]
    g = df[target_group_col].astype(bool).to_numpy()
    G = int(g.sum())
    if G == 0:
        print(f"⚠ WARNING: '{target_group_col}' has zero members. Cannot calculate PMI.")
        _tend("pmi.calculate_pmi", t0)
        return pd.DataFrame()

    # Document frequencies (unique videos with the term)
    df_term = np.asarray(X.sum(axis=0)).ravel().astype(int)
    df_term_group = np.asarray(X[g].sum(axis=0)).ravel().astype(int)

    # Probabilities (per-video)
    p_term  = df_term / N
    p_group = G / N
    p_joint = df_term_group / N

    # Log-smoothing to avoid -inf for zero joints
    eps = 1e-12
    pmi = np.log2((p_joint + eps) / (p_term * p_group + eps))

    # P(term | group) for interpretability
    p_term_given_group = np.divide(
        df_term_group, G, out=np.zeros_like(df_term_group, dtype=float), where=(G > 0)
    )

    out = (
        pd.DataFrame(
            {
                "Term": terms,
                "PMI": pmi,
                "DF": df_term,
                "DF_in_group": df_term_group,
                "P(term)": p_term,
                "P(group)": p_group,
                "P(term,group)": p_joint,
                "P(term|group)": p_term_given_group,
            }
        )
        .sort_values(["PMI", "DF_in_group", "DF"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    kept = int((df_term >= min_df).sum())
    print(f"✓ PMI complete. Vocabulary={len(terms):,} (kept ≥{min_df}: {kept:,}), N={N:,}, |group|={G:,}")
    _tend("pmi.calculate_pmi", t0)
    return out


# --- 5) Plotting --------------------------------------------------------------
@plot_dual_theme(section="fairness")
def plot_pmi_scores(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """
    Plot top-N PMI terms as a horizontal bar chart.

    Notes
    -----
    - Adds right margin so on-bar labels don't get clipped.
    """
    import textwrap


    n = int(data["Term"].nunique())
    active = (palette[:n] if (palette and len(palette) >= n) else sns.color_palette("magma", n_colors=n))

    # Main plot (legend off because hue==y)
    sns.barplot(
        x="PMI", y="Term", hue="Term",
        data=data, ax=ax, palette=active, legend=False, orient="h",
        errorbar=None  # quiets seaborn>=0.13 warnings about CIs we don't need
    )

    ax.set_title(f"Top {len(data)} Terms Associated with Target (by PMI)")
    ax.set_xlabel("Pointwise Mutual Information (PMI)")
    ax.set_ylabel("Term")

    # Wrap long y-tick labels for readability
    ticks = ax.get_yticks()
    labels = [lab.get_text() for lab in ax.get_yticklabels()]
    ax.set_yticks(ticks)
    ax.set_yticklabels([textwrap.fill(lbl, 24) for lbl in labels])

    # Ensure we have some right-side breathing room for annotations
    ax.margins(x=0.08)

    # On-bar counts: DF_in_group / DF
    xmax = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1.0
    for patch, (_, row) in zip(ax.patches, data.iterrows()):
        x = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        # Place label inside the axes bounds
        x_text = min(x + 0.02 * xmax, 0.98 * xmax)
        ax.text(
            x_text, y,
            f"{int(row['DF_in_group'])}/{int(row['DF'])}",
            va="center", ha="left", fontsize=10, clip_on=True
        )

    # Small explainer box (theme-aware stroke; translucent face so it's readable in dark)
    edge = ax.spines.get("left", None).get_edgecolor() if ax.spines.get("left", None) else "0.5"
    ax.text(
        0.98, 0.02,
        "How to read: bar = PMI; label = videos with term in group / all videos with term.\n"
        "Probabilities are per-video; totals can exceed N due to multi-labels.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", fc=(1, 1, 1, 0.08), ec=edge, lw=0.8)
    )



# --- 6) Self-check ------------------------------------------------------------
def self_check_from_sample(
    df: pd.DataFrame,
    *,
    sample: Optional[int],
    seed: int,
    target_group_col: str,
    text_col: str,
    min_df: int,
    max_features: int,
    top_n: int,
    paths: dict,
) -> None:
    """
    Run a small self-check on a random sample from the REAL parquet.

    Behavior
    --------
    - Uses `sample` rows (or min(total, 80_000) if None), random_state=seed.
    - Writes *_selfcheck artefacts only (does NOT overwrite full-run outputs).
    - Prints qualitative results + high-PMI/low-coverage outliers.
    """
    t0 = _tstart(f"[SELF-CHECK] Drawing random sample (seed={seed}) ...")
    n = sample or min(80_000, len(df))
    d = df.sample(n=n, random_state=seed, replace=False).reset_index(drop=True)
    _tend("pmi.selfcheck.sample", t0)

    df_pmi = calculate_pmi(
        d, target_group_col, text_col, min_df=min_df, max_features=max_features
    )
    if df_pmi.empty:
        print("✗ Self-check halted: empty PMI table.")
        return

    df_top = df_pmi.head(top_n)

    # Console snapshot
    print("\n=== SELF-CHECK SNAPSHOT (PMI) ===")
    print(
        df_top[["Term", "PMI", "DF_in_group", "DF"]]
        .assign(PMI=lambda x: x["PMI"].round(2))
        .to_string(index=False)
    )

    # Outliers: high PMI near min_df
    near_min = df_pmi[(df_pmi["DF"] <= (min_df + 10)) & (df_pmi["DF"] >= min_df)]
    if not near_min.empty:
        print("\n[OUTLIERS] High-PMI, low-coverage terms (DF near min_df):")
        # Save outliers snapshot for self-check
        if "outliers_sc" in paths:
            near_min.sort_values("PMI", ascending=False).to_csv(paths["outliers_sc"], index=False)
            print(f"✓ Outliers saved: {paths['outliers_sc']}")
        for _, r in near_min.sort_values("PMI", ascending=False).head(8).iterrows():
            print(f"  - {r['Term']}: PMI {r['PMI']:.2f}, DF {int(r['DF'])}, in-group {int(r['DF_in_group'])}")

    # Save *_selfcheck artefacts
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NARR_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df_pmi.to_csv(paths["data_full_sc"], index=False)
    df_top.to_csv(paths["data_top_sc"], index=False)
    print(f"✓ Artefacts saved: {paths['data_full_sc']} , {paths['data_top_sc']}")

    # Plot figure (dual theme handles light/dark)
    plot_pmi_scores(
        data=df_top,
        save_path=str(paths["fig_base"]) + "_selfcheck",
        figsize=(10, 12),
    )

    # Narrative (self-check)
    lead = df_top.iloc[0]
    summary = f"""
# Automated Summary (Self-check): Intersectional PMI

Target: `{target_group_col}`. Sample size: {n:,} (seed={seed}).
Top term: **{lead['Term']}** — PMI {lead['PMI']:.2f}; in-group/overall DF = {int(lead['DF_in_group'])}/{int(lead['DF'])}.
Probabilities are per-video. Because videos are multi-label, totals across terms can exceed the number of videos.
"""
    with open(paths["narr_sc"], "w") as f:
        f.write(summary)
    print(f"✓ Narrative saved: {paths['narr_sc']}")


# --- 7) Main ------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> None:
    """
    Run Step 03 PMI pipeline (real corpus by default; tiny --selfcheck available).

    CLI
    ---
    python3 src/analysis/03_intersectional_profiling.py
    python3 src/analysis/03_intersectional_profiling.py --selfcheck --sample 80000
    python3 src/analysis/03_intersectional_profiling.py --min-df 100 --max-features 50000 --top-n 25
    """
    t_all = time.perf_counter()
    print("--- Starting Step 03: Intersectional Profiling (PMI) ---")
    print("[NOTE] Titles may be non-English; tags/categories help anchor semantics (multi-label).")

    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true", help="Run self-check on a random sample of the REAL parquet.")
    p.add_argument("--sample", type=int, default=None, help="Rows to sample for self-check (default=min(N, 80k)).")
    p.add_argument("--min-df", type=int, default=50, help="Min videos required for a term.")
    p.add_argument("--max-features", type=int, default=30000, help="Vocab cap for vectorizer.")
    p.add_argument("--top-n", type=int, default=TOP_N_TERMS_DEFAULT, help="How many PMI terms to show/save.")
    args = p.parse_args(argv)

    # Load corpus
    df = _load_corpus(CORPUS_PATH)

    # Ensure text column present
    text_col = _ensure_text_column(df, text_col="combined_text_clean")

    # Ensure target column exists
    if PMI_TARGET_GROUP not in df.columns:
        print(f"✗ ERROR: Target column '{PMI_TARGET_GROUP}' not found in corpus.")
        return

    # Resolve output paths based on target slug
    slug = _slugify_target(PMI_TARGET_GROUP)
    data_top_path       = DATA_DIR / f"03_pmi_{slug}.csv"
    data_full_path      = DATA_DIR / f"03_pmi_{slug}_full.csv"
    data_top_path_sc    = DATA_DIR / f"03_pmi_{slug}_selfcheck.csv"
    data_full_path_sc   = DATA_DIR / f"03_pmi_{slug}_full_selfcheck.csv"
    fig_base            = FIG_DIR / "03_pmi_associations_bar"
    narr_path           = NARR_DIR / "03_pmi_summary.md"
    narr_path_sc        = NARR_DIR / "03_pmi_summary_selfcheck.md"
    tex_path            = TEX_DIR / f"03_pmi_{slug}.tex"

    outliers_path       = DATA_DIR / f"03_pmi_{slug}_outliers.csv"
    outliers_path_sc    = DATA_DIR / f"03_pmi_{slug}_outliers_selfcheck.csv"

    # Self-check (non-destructive)
    if args.selfcheck:
        self_check_from_sample(
            df,
            sample=args.sample,
            seed=SEED,
            target_group_col=PMI_TARGET_GROUP,
            text_col=text_col,
            min_df=args.min_df,
            max_features=args.max_features,
            top_n=args.top_n,
            paths=dict(
                data_full_sc=data_full_path_sc,
                data_top_sc=data_top_path_sc,
                fig_base=str(fig_base),
                narr_sc=str(narr_path_sc),
                outliers_sc=str(outliers_path_sc),
            ),
        )
        _tend("pmi.step03_total", t_all)
        print("\n--- Step 03: Intersectional Profiling (PMI, self-check) Completed ---")
        return

    # ===== FULL RUN =====
    df_pmi = calculate_pmi(
        df, PMI_TARGET_GROUP, text_col, min_df=args.min_df, max_features=args.max_features
    )
    if df_pmi.empty:
        print("✗ Halting: PMI table is empty.")
        return

    # Keep top-N for plots & table
    top_n = min(args.top_n, len(df_pmi))
    df_top = df_pmi.head(top_n)

    # Console normalization notes
    N = len(df)
    G = int(df[PMI_TARGET_GROUP].sum())
    print("\n=== REAL NUMBERS SUMMARY (PMI) ===")
    print(f"Total videos: {N:,} | Group videos: {G:,}")
    print("All probabilities are per-video (document frequency).")
    print("Videos are multi-label, so totals across terms can exceed N.")

    # Save data artefacts
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NARR_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)

    df_pmi.to_csv(data_full_path, index=False)
    df_top.to_csv(data_top_path, index=False)
    print(f"✓ Artefacts saved: {data_full_path} , {data_top_path}")
    # Save high-PMI, low-coverage outliers (near min_df)
    near_min_full = df_pmi[(df_pmi["DF"] <= (args.min_df + 10)) & (df_pmi["DF"] >= args.min_df)]
    if not near_min_full.empty:
        near_min_full.sort_values("PMI", ascending=False).to_csv(outliers_path, index=False)
        print(f"✓ Outliers saved: {outliers_path}")

    # Plot figure (dual theme handles light/dark)
    plot_pmi_scores(
        data=df_top,
        save_path=str(fig_base),
        figsize=(10, 12),
    )

    # Narrative
    lead = df_top.iloc[0]
    summary = f"""
# Automated Summary: Intersectional Profiling via PMI

Target group: `{PMI_TARGET_GROUP}`. Probabilities are per-video (document frequency).

Top term by PMI: **{lead['Term']}**
- PMI = {lead['PMI']:.2f}
- Videos with term in group / overall: {int(lead['DF_in_group'])} / {int(lead['DF'])}

Note: Because videos are multi-label, counts across terms can exceed the number of videos.
"""
    with open(narr_path, "w") as f:
        f.write(summary)
    print(f"✓ Narrative saved: {narr_path}")

    # LaTeX (top-N)
    dataframe_to_latex_table(
        df=df_top[["Term", "PMI", "P(term|group)"]].set_index("Term"),
        save_path=str(tex_path),
        caption=f"Top {top_n} Terms Associated with the '{PMI_TARGET_GROUP}' Group by PMI.",
        label=f"tab:pmi-{slug}",
        note="Probabilities are per-video; sums across labels can exceed 100% because videos are multi-label."
    )

    _tend("pmi.step03_total", t_all)
    print("\n--- Step 03: Intersectional Profiling (PMI) Completed Successfully ---")


if __name__ == "__main__":
    main()
