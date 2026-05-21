# -*- coding: utf-8 -*-
"""
04_multilayer_harm_analysis.py
==============================

Role
----
Quantify category-specific harmful language (HurtLex) across protected groups,
producing interpretable prevalence profiles and publication-ready artefacts.

Goal (precise)
--------------
For each protected group (race_*, gender_*, intersectional_*), compute the share
of the group’s videos that contain at least one term from each HurtLex category
(**document frequency**, per-video). Save CSV + LaTeX + dual-theme heatmap, with
timers, qualitative analysis, and a reproducible random self-check that writes
*_selfcheck artefacts only.

What it is / what it does
-------------------------
1) Loads seed, paths, and column names from config (seed=95).
2) Loads and normalizes HurtLex; compiles robust regex by category (hyphen/space variants).
3) Builds **harm profiles** (% of videos per group per category).
4) Calls out outliers (spikes) to aid interpretability.
5) Saves artefacts under **04_*** naming; dual-theme plot.
6) Self-check uses a **random sample from the real parquet** (not hardcoded) and
   writes *_selfcheck outputs so the full artefacts are not overwritten.

Interpretability notes
----------------------
- Titles may be non-English; **tags/categories help anchor semantics**.
- All percentages use unique videos (**document frequency**).
- This is a multi-label corpus; **totals across labels can exceed N** — expected.

Artefacts (full run)
--------------------
Data:
  - outputs/data/04_harm_category_by_group.csv                 (index=Group, columns=HurtLex category, values=%)
Figure (dual theme):
  - outputs/figures/harm/04_harm_category_heatmap_{light,dark}.png
Narrative:
  - outputs/narratives/automated/04_harm_analysis_summary.md
LaTeX:
  - dissertation/auto_tables/04_harm_category_by_group.tex

Self-check (separate, non-destructive)
--------------------------------------
  - outputs/data/04_harm_category_by_group_selfcheck.csv
  - outputs/figures/harm/04_harm_category_heatmap_selfcheck_{light,dark}.png
  - outputs/narratives/automated/04_harm_analysis_summary_selfcheck.md

"""

from __future__ import annotations

# --- Imports (always at the top) ----------------------------------------------
import re
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Project utils (config + plotting theme + LaTeX helper)
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.utils.academic_tables import dataframe_to_latex_table


# --- 1) Config & reproducibility ---------------------------------------------
CONFIG = load_config()
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
np.random.seed(SEED)

# Canonical Step-01 parquet (consistent naming)
CORPUS_PATH = Path(CONFIG["paths"]["data"]) / "01_ml_corpus.parquet"

# Outputs (04_* naming + section folder)
DATA_DIR = Path(CONFIG["paths"]["data"])
FIG_DIR  = Path(CONFIG["paths"]["figures"]) / "harm"
NARR_DIR = Path(CONFIG["paths"]["narratives"]) / "automated"
TEX_DIR  = Path(CONFIG["project"]["root"]) / "dissertation" / "auto_tables"

OUT_DATA          = DATA_DIR / "04_harm_category_by_group.csv"
OUT_DATA_SC       = DATA_DIR / "04_harm_category_by_group_selfcheck.csv"
OUT_FIG_BASE      = FIG_DIR  / "04_harm_category_heatmap"
OUT_FIG_BASE_SC   = FIG_DIR  / "04_harm_category_heatmap_selfcheck"
OUT_NARR          = NARR_DIR / "04_harm_analysis_summary.md"
OUT_NARR_SC       = NARR_DIR / "04_harm_analysis_summary_selfcheck.md"
OUT_LATEX         = TEX_DIR  / "04_harm_category_by_group.tex"

OUT_DATA_COUNTS    = DATA_DIR / "04_harm_category_by_group_counts.csv"
OUT_DATA_COUNTS_SC = DATA_DIR / "04_harm_category_by_group_counts_selfcheck.csv"
OUT_DATA_LONG      = DATA_DIR / "04_harm_category_by_group_long.csv"
OUT_DATA_LONG_SC   = DATA_DIR / "04_harm_category_by_group_long_selfcheck.csv"
OUT_OUTLIERS       = DATA_DIR / "04_harm_category_outliers.csv"
OUT_OUTLIERS_SC    = DATA_DIR / "04_harm_category_outliers_selfcheck.csv"


# Lexica location (HurtLex tsv)
LEXICA_PATH = Path(CONFIG["project"]["root"]) / "config" / "abusive_lexica"
HURTLEX_FILENAME = "hurtlex_EN.tsv"


# --- 2) Lightweight timers ----------------------------------------------------
def _tstart(msg: str) -> float:
    """Start a perf counter and print a standardized header line."""
    t0 = time.perf_counter()
    print(msg)
    return t0

def _tend(label: str, t0: float) -> None:
    """Finish a timer with standardized [TIME] output in seconds."""
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")


# --- 3) Lexicon loading & patterns -------------------------------------------
def load_hurtlex(path: Path, filename: str = HURTLEX_FILENAME) -> pd.DataFrame:
    """
    Load and normalize the HurtLex lexicon.

    Parameters
    ----------
    path : Path
        Directory containing HurtLex file.
    filename : str
        Expected TSV file name (default: 'hurtlex_EN.tsv').

    Returns
    -------
    pd.DataFrame
        Columns: ['term','category'] — lowercase, stripped, unique terms.

    Notes
    -----
    - Prints a [TIME] line with elapsed seconds.
    - Drops blank terms and duplicates.
    """
    t0 = _tstart("Loading and normalizing HurtLex...")
    fpath = path / filename
    try:
        raw = pd.read_csv(fpath, sep="\t")
    except FileNotFoundError:
        print(f"✗ ERROR: HurtLex file not found at {fpath}.")
        _tend("harm.load_hurtlex", t0)
        return pd.DataFrame()

    df = (raw[["lemma", "category"]]
          .rename(columns={"lemma": "term"})
          .assign(term=lambda d: d["term"].astype(str).str.lower().str.strip()))
    df = df[df["term"] != ""].drop_duplicates("term").reset_index(drop=True)
    print(f"✓ Loaded {len(df):,} unique terms across {df['category'].nunique()} categories.")
    _tend("harm.load_hurtlex", t0)
    return df


def compile_category_patterns(
    df_lexicon: pd.DataFrame,
    *,
    casefold: bool = True,
    word_boundaries: bool = True,
    allow_hyphen_variants: bool = True
) -> dict[str, re.Pattern]:
    """
    Compile one regex per HurtLex category with safe escaping and hyphen/space variants.

    Strategy
    --------
    - For each term, split on separators `[-_\\s]+` into tokens.
    - Escape each token with `re.escape`.
    - Re-join tokens with the class `[-_\\s]+` (treat '-', '_' and whitespace as equivalent).
    - Optionally wrap with \\b...\\b.

    Returns
    -------
    dict[str, re.Pattern]
        {'category': compiled_regex}
    """
    t0 = _tstart("Compiling per-category regex patterns...")
    flags = re.IGNORECASE if casefold else 0
    cat2pat: dict[str, re.Pattern] = {}

    for category, terms_ser in df_lexicon.groupby("category")["term"]:
        parts: list[str] = []
        for raw in terms_ser:
            if not isinstance(raw, str):
                continue
            s = raw.strip()
            if not s:
                continue

            if allow_hyphen_variants and re.search(r"[-_\s]", s):
                tokens = [t for t in re.split(r"[-_\s]+", s) if t]
                core = r"[-_\s]+".join(re.escape(t) for t in tokens) if tokens else re.escape(s)
            else:
                core = re.escape(s)

            if word_boundaries:
                core = rf"\b{core}\b"
            parts.append(f"(?:{core})")

        if parts:
            try:
                cat2pat[category] = re.compile("|".join(parts), flags)
            except re.error as err:
                print(f"[WARN] Failed to compile pattern for category '{category}': {err}")

    _tend("harm.compile_category_patterns", t0)
    return cat2pat


# --- 4) Harm profiles ---------------------------------------------------------
def _ensure_text_column(df: pd.DataFrame, text_col: str = "combined_text_clean") -> str:
    """
    Ensure a text column exists for matching; fallback to title+tags+categories.
    """
    if text_col in df.columns:
        return text_col
    df[text_col] = (
        df.get("title", "").fillna("").astype(str) + " " +
        df.get("tags", "").fillna("").astype(str) + " " +
        df.get("categories", "").fillna("").astype(str)
    ).str.lower()
    return text_col



def create_harm_profiles(df: pd.DataFrame, df_lexicon: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Build harm profiles:
      - prevalence_df: % of group's videos (document frequency), rounded to 1 d.p.
      - counts_df:     raw counts of videos that hit each category
      - long_df:       tidy table with Group, Category, Count, N, Prevalence (%)
      - group_sizes:   Series of N per group
    """
    t0 = _tstart("Creating harm profiles for each protected group...")

    text_col = _ensure_text_column(df)
    group_cols = sorted([c for c in df.columns if c.startswith("race_") or c.startswith("gender_") or c.startswith("intersectional_")])
    cat2pat = compile_category_patterns(df_lexicon)

    rows = []
    for group in group_cols:
        gmask = (df[group] == 1)
        n_docs = int(gmask.sum())
        if n_docs == 0:
            continue
        text = df.loc[gmask, text_col].astype(str)

        for category, pattern in cat2pat.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                hits = text.str.contains(pattern, regex=True, na=False)
            count = int(hits.sum())
            prevalence = 100.0 * count / n_docs if n_docs else 0.0
            rows.append({"Group": group, "HurtLex Category": category, "Count": count, "N": n_docs, "Prevalence (%)": prevalence})

    if not rows:
        print("✗ No groups/categories produced any rows.")
        _tend("harm.create_harm_profiles", t0)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=int)

    long_df = pd.DataFrame(rows)
    prevalence_df = (
        long_df.pivot(index="Group", columns="HurtLex Category", values="Prevalence (%)")
                .fillna(0.0).sort_index().round(1)
    )
    counts_df = (
        long_df.pivot(index="Group", columns="HurtLex Category", values="Count")
               .fillna(0).sort_index().astype(int)
    )
    group_sizes = (long_df[["Group","N"]].drop_duplicates().set_index("Group")["N"]).astype(int)

    print("✓ Harm profile creation complete.")
    _tend("harm.create_harm_profiles", t0)
    return prevalence_df, counts_df, long_df.round({"Prevalence (%)": 1}), group_sizes

# --- 5) Visualization  -------------------------------------------------

@plot_dual_theme(section="fairness")
def plot_harm_heatmap(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """
    Heatmap of harm category prevalence by group (percent of group's videos).
    - Uses 'magma'.
    - Annotates cells with 1 decimal place.
    """
    sns.heatmap(
        data, ax=ax, cmap="magma", linewidths=.5,
        annot=True, fmt=".1f", annot_kws={"size": 7}, cbar_kws={"label": "Prevalence (%)"}
    )
    ax.set_title("Prevalence (%) of HurtLex Categories by Protected Group")
    ax.set_xlabel("HurtLex Category")
    ax.set_ylabel(None)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)


# --- 6) Qualitative summary ---------------------------------------------------
def print_qualitative_summary(df_harm: pd.DataFrame) -> None:
    """
    Print interpretable highlights and outliers (spikes) for quick diagnostics.
    """
    t0 = _tstart("Summarising qualitative highlights & outliers...")
    if df_harm.empty:
        print("No harm data to summarise."); _tend("harm.summary", t0); return

    # Top 5 cells overall
    cells = (
        df_harm.stack()
        .rename("Prevalence")
        .sort_values(ascending=False)
        .head(5)
        .round(1)
    )
    print("\nTop 5 highest prevalence cells (Group × Category):")
    for (grp, cat), val in cells.items():
        print(f"  - {grp} × {cat}: {val:.1f}%")

    # For each category, top group (spike)
    print("\nPer-category spikes (top group per category):")
    for cat in df_harm.columns:
        col = df_harm[cat]
        if col.max() <= 0:
            continue
        g = col.idxmax()
        v = col.max()
        print(f"  - {cat}: {g} @ {v:.1f}%")

    _tend("harm.summary", t0)


# --- 7) Main ------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> None:
    """
    Run Step 04 harm analysis (real corpus by default; random self-check available).

    CLI
    ---
    python3 src/analysis/04_multilayer_harm_analysis.py
    python3 src/analysis/04_multilayer_harm_analysis.py --selfcheck --sample 80000
    """
    t_all = time.perf_counter()
    print("--- Starting Step 04: Multi-Layered Harm Analysis ---")
    print("[NOTE] Titles may be non-English; tags/categories help anchor semantics (multi-label).")

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true", help="Run on a random sample and write *_selfcheck artefacts.")
    p.add_argument("--sample", type=int, default=None, help="Rows for self-check (default=min(N, 80k)).")
    args = p.parse_args(argv)

    # Load lexicon
    df_lex = load_hurtlex(LEXICA_PATH, HURTLEX_FILENAME)
    if df_lex.empty:
        if args.selfcheck:
            # Fallback tiny lexicon to keep the demo runnable
            print("[WARN] HurtLex missing; using tiny fallback lexicon for self-check.")
            df_lex = pd.DataFrame({
                "term": ["stupid", "lazy", "exotic", "thug", "goddess"],
                "category": ["PS", "PS", "AS", "VI", "PA"]
            })
        else:
            return

    # Load corpus
    print(f"[READ] Parquet: {CORPUS_PATH}")
    df = pd.read_parquet(CORPUS_PATH)
    print(f"[STATS] Total videos in corpus: {len(df):,}")

    # Self-check path: sample real data and write *_selfcheck artefacts
    if args.selfcheck:
        t0 = _tstart(f"[SELF-CHECK] Drawing random sample (seed={SEED}) ...")
        n = args.sample or min(80_000, len(df))
        df = df.sample(n=n, random_state=SEED, replace=False).reset_index(drop=True)
        print(f"[SELF-CHECK] Sample size: {len(df):,}")
        _tend("harm.selfcheck.sample", t0)

    # Compute harm profiles
    df_prev, df_cnt, df_long, group_sizes = create_harm_profiles(df, df_lex)
    print_qualitative_summary(df_prev)
    out_counts_path = OUT_DATA_COUNTS_SC if args.selfcheck else OUT_DATA_COUNTS
    out_long_path   = OUT_DATA_LONG_SC   if args.selfcheck else OUT_DATA_LONG
    out_outliers_path = OUT_OUTLIERS_SC if args.selfcheck else OUT_OUTLIERS

    # Save artefacts
    print("\nSaving data artefacts...")
    for d in (DATA_DIR, FIG_DIR, NARR_DIR, TEX_DIR):
        d.mkdir(parents=True, exist_ok=True)

    out_data_path = OUT_DATA_SC if args.selfcheck else OUT_DATA
    df_prev.to_csv(out_data_path)
    print(f"✓ Artefact saved: {out_data_path.resolve()}")
    df_cnt.to_csv(out_counts_path)
    print(f"✓ Artefact saved: {out_counts_path.resolve()}")
    df_long.to_csv(out_long_path, index=False)
    print(f"✓ Artefact saved: {out_long_path.resolve()}")
    # Outliers: top 15 Group×Category spikes by prevalence
    _ol = df_long.sort_values("Prevalence (%)", ascending=False).head(15)
    _ol.to_csv(out_outliers_path, index=False)
    print(f"✓ Artefact saved: {out_outliers_path.resolve()}")

    # Plot
    print("\nGenerating visualizations...")
    fig_base = OUT_FIG_BASE_SC if args.selfcheck else OUT_FIG_BASE
    plot_harm_heatmap(data=df_prev, save_path=str(fig_base), figsize=(12, 10))

    # Narrative
    print("\nGenerating narrative summary...")
    max_val = float(df_prev.max().max()) if not df_prev.empty else 0.0
    max_cat = str(df_prev.max().idxmax()) if not df_prev.empty else "N/A"
    max_group = str(df_prev[max_cat].idxmax()) if (not df_prev.empty and max_cat in df_prev.columns) else "N/A"
    narr_path = OUT_NARR_SC if args.selfcheck else OUT_NARR
    summary = f"""
# Automated Summary: Multi-Layered Harm Analysis

Group sizes (N videos per group):

- $filled at runtime\n
This analysis quantified the prevalence of HurtLex harm categories across protected groups.
Values are percentages of each group's videos that contain at least one term from the category
(document frequency). Because videos are multi-label, totals across labels may exceed N.

**Highest prevalence:** {max_val:.1f}% in **{max_group}** for category **{max_cat}**.
"""
    with open(narr_path, "w") as f:
        f.write(summary)
        # Append group sizes table (sorted by N desc)
        if not group_sizes.empty:
            f.write('\n\n## Group sizes (N videos)\n')
            for g, n in group_sizes.sort_values(ascending=False).items():
                f.write(f"- {g}: {int(n)}\n")
    print(f"✓ Artefact saved: {narr_path.resolve()}")

    # LaTeX (export the prevalence matrix you computed)
    table_df = df_prev.reset_index().rename(columns={"index": "Group"})
    dataframe_to_latex_table(
        df=table_df,
        save_path=str(OUT_LATEX if not args.selfcheck else OUT_LATEX.with_name(OUT_LATEX.stem + "_selfcheck.tex")),
        caption="Prevalence (%) of HurtLex Harm Categories Across Protected Groups.",
        label="tab:harm-profiles",
        note="Each cell is the percent of a group's videos containing at least one term in the category (document frequency)."
    )


    _tend("harm.step04_total", t_all)
    print("\n--- Step 04: Multi-Layered Harm Analysis Completed Successfully ---")


if __name__ == "__main__":
    main()
