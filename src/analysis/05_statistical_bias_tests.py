# -*- coding: utf-8 -*-
"""
05_statistical_bias_tests.py
============================

Role
----
Quantify outcome disparities (ratings, views) between protected groups and
a privileged baseline via robust non-parametric tests and interpretable plots.

Goal (precise)
--------------
For a set of intersectional groups (e.g., Black/White/Asian/Latina Women),
compare distributions to a privileged baseline (default: White Women) using:
- Mann–Whitney U (two-sided) with Benjamini–Hochberg FDR,
- Effect sizes (Cliff's delta via AUC; Cohen's d),
- Bootstrap 95% CI for *median* differences.
Save dual-theme forest plots and publication-ready tables, with timers and a
reproducible random self-check that **never** overwrites full outputs.

What it is / what it does
-------------------------
1) Reads seed, paths, and column names from config (seed=95).
2) Loads canonical parquet (no DB): outputs/data/01_ml_corpus.parquet.
3) Builds standard group masks (race_* × gender_*), plus All Women/Men.
4) Runs stats on ratings and log10(views+1); rounds to consistent units:
   - years (ints if present), ratings (1 decimal), effects & CI (3 decimals),
     p-values (3 sig figs), counts (ints).
5) Prints a qualitative summary (top signals & outliers).
6) Saves artefacts with **05_*** names; self-check writes *_selfcheck files.

Interpretability & caveats
--------------------------
- Titles may be non-English; **tags/categories help anchor semantics**.
- All metrics are computed per video. This corpus is **multi-label**, so sums
  across labels/groups can exceed N — that’s expected.
- We avoid seaborn boxplots here, so no “palette without hue” warnings.

Artefacts (full run)
--------------------
Data
  - outputs/data/05_bias_tests_ratings.csv
  - outputs/data/05_bias_tests_views.csv
Figures (dual theme)
  - outputs/figures/bias/05_bias_forest_ratings_{light,dark}.png
  - outputs/figures/bias/05_bias_forest_views_{light,dark}.png
Narrative
  - outputs/narratives/automated/05_bias_tests_summary.md
LaTeX
  - dissertation/auto_tables/05_bias_tests_ratings.tex
  - dissertation/auto_tables/05_bias_tests_views.tex

Self-check (separate, non-destructive)
--------------------------------------
  - outputs/data/05_bias_tests_ratings_selfcheck.csv
  - outputs/data/05_bias_tests_views_selfcheck.csv
  - outputs/figures/bias/05_bias_forest_ratings_selfcheck_{light,dark}.png
  - outputs/figures/bias/05_bias_forest_views_selfcheck_{light,dark}.png
  - outputs/narratives/automated/05_bias_tests_summary_selfcheck.md
  - dissertation/auto_tables/05_bias_tests_ratings_selfcheck.tex
  - dissertation/auto_tables/05_bias_tests_views_selfcheck.tex
"""

from __future__ import annotations

# --- Imports (keep at top) ---------------------------------------------------
import sys
import time
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# --- 1) Configuration & Theme ------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.utils.academic_tables import dataframe_to_latex_table

CONFIG = load_config()
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))

# Canonical Step-01 parquet (consistent across steps)
CORPUS_PATH   = Path(CONFIG["paths"]["data"]) / "01_ml_corpus.parquet"

# Output dirs / filenames (05_* naming)
DATA_DIR      = Path(CONFIG["paths"]["data"])
FIGURES_DIR   = Path(CONFIG["paths"]["figures"]) / "bias"
TABLES_DIR    = Path(CONFIG["project"]["root"]) / "dissertation" / "auto_tables"
NARR_DIR      = Path(CONFIG["paths"]["narratives"]) / "automated"

OUT_RATINGS   = DATA_DIR / "05_bias_tests_ratings.csv"
OUT_VIEWS     = DATA_DIR / "05_bias_tests_views.csv"
OUT_RATINGS_SC= DATA_DIR / "05_bias_tests_ratings_selfcheck.csv"
OUT_VIEWS_SC  = DATA_DIR / "05_bias_tests_views_selfcheck.csv"

OUT_FIG_R     = FIGURES_DIR / "05_bias_forest_ratings"
OUT_FIG_V     = FIGURES_DIR / "05_bias_forest_views"
OUT_FIG_R_SC  = FIGURES_DIR / "05_bias_forest_ratings_selfcheck"
OUT_FIG_V_SC  = FIGURES_DIR / "05_bias_forest_views_selfcheck"

NARR_PATH     = NARR_DIR / "05_bias_tests_summary.md"
NARR_PATH_SC  = NARR_DIR / "05_bias_tests_summary_selfcheck.md"

OUT_TEX_R     = TABLES_DIR / "05_bias_tests_ratings.tex"
OUT_TEX_V     = TABLES_DIR / "05_bias_tests_views.tex"
OUT_TEX_R_SC  = TABLES_DIR / "05_bias_tests_ratings_selfcheck.tex"
OUT_TEX_V_SC  = TABLES_DIR / "05_bias_tests_views_selfcheck.tex"


# --- 2) Timers ---------------------------------------------------------------
def _t0(msg: str) -> float:
    """Start a monotonic timer and print a standard log header."""
    t = time.perf_counter()
    print(msg)
    return t

def _tend(label: str, t0: float) -> None:
    """Print a standardized [TIME] line for elapsed seconds."""
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")


# --- 3) Utilities ------------------------------------------------------------
def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    """
    Ensure required columns exist; create empties if missing.
    Missing numeric columns: NaN; text columns: ''.
    """
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan if c in {"rating", "views"} else ""


@dataclass(frozen=True)
class GroupDef:
    """Container describing a group mask and metadata."""
    name: str
    mask: pd.Series


def build_groups(df: pd.DataFrame) -> Dict[str, GroupDef]:
    """
    Construct standard intersectional groups.

    Notes
    -----
    If a one-hot column is missing, it's treated as all zeros.
    """
    def col(name: str) -> pd.Series:
        return (df.get(name, 0) == 1)

    groups = {
        "Black Women":  col("race_ethnicity_black")  & col("gender_female"),
        "White Women":  col("race_ethnicity_white")  & col("gender_female"),
        "Asian Women":  col("race_ethnicity_asian")  & col("gender_female"),
        "Latina Women": col("race_ethnicity_latina") & col("gender_female"),
        "All Women":    col("gender_female"),
        "All Men":      (df.get("gender_female", 0) == 0),
    }
    return {k: GroupDef(k, v.fillna(False)) for k, v in groups.items()}


# --- 4) Stats helpers --------------------------------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta (rank-biserial) via AUC: delta = 2*AUC - 1.

    We compute U for the 'x > y' direction (alternative='greater') to avoid
    the small-U ambiguity from two-sided tests, then AUC = U/(n*m).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return np.nan
    U_greater = stats.mannwhitneyu(x, y, alternative="greater").statistic
    auc = U_greater / (n * m)
    return float(2 * auc - 1)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d for two independent samples (pooled SD).
    Positive -> x mean > y mean.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2 = x.var(ddof=1)
    sy2 = y.var(ddof=1)
    sp = math.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    return 0.0 if sp == 0 else (x.mean() - y.mean()) / sp


def bootstrap_ci_diff_median(
    x: np.ndarray, y: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = SEED
) -> Tuple[float, float, float]:
    """
    Bootstrap the median difference (x - y); return (diff, lo, hi).
    """
    rng = np.random.default_rng(seed)
    nx, ny = len(x), len(y)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        xs = x[rng.integers(0, nx, nx)]
        ys = y[rng.integers(0, ny, ny)]
        diffs[i] = np.median(xs) - np.median(ys)
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return float(np.median(x) - np.median(y)), lo, hi


def fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg FDR correction. Returns (pvals_adj, reject_mask).
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    p_adj_sorted = np.minimum.accumulate((p[order] * n / ranks)[::-1])[::-1]
    p_adj = np.empty_like(p)
    p_adj[order] = p_adj_sorted
    return p_adj, (p_adj <= alpha)


# --- 5) Core testing ---------------------------------------------------------
def run_bias_tests(
    df: pd.DataFrame,
    value_col: str,
    groups: Dict[str, GroupDef],
    privileged: str,
    *,
    min_n: int = 50,
    log10_views: bool = False,
    n_boot: int = 2000,
    seed: int = SEED
) -> pd.DataFrame:
    """
    Run group-vs-privileged bias tests for a numeric column.

    Returns
    -------
    pd.DataFrame
        Rows per group (excluding privileged) with counts, medians/means,
        MWU U + p, FDR, Cliff's δ, Cohen's d, bootstrap median diff (+ CI).
    """
    t0 = _t0(f"Running bias tests for '{value_col}' vs privileged='{privileged}' "
             f"(log10_views={log10_views})...")

    v = df[value_col].astype(float).copy()
    if value_col == "views" and log10_views:
        v = np.log10(np.clip(v, a_min=0, a_max=None) + 1.0)

    base_mask = groups[privileged].mask
    base = v[base_mask].dropna().to_numpy()

    results: List[Dict[str, float]] = []
    for name, g in groups.items():
        if name == privileged:
            continue
        x = v[g.mask].dropna().to_numpy()
        if len(x) < min_n or len(base) < min_n:
            continue

        # Two-sided p-value for hypothesis test
        U, p = stats.mannwhitneyu(x, base, alternative="two-sided")

        # Effect sizes
        delta = cliffs_delta(x, base)
        d = cohens_d(x, base)
        med_diff, lo, hi = bootstrap_ci_diff_median(x, base, n_boot=n_boot, seed=seed)

        results.append({
            "Group": name,
            "Privileged": privileged,
            "N_group": int(len(x)),
            "N_priv": int(len(base)),
            "Mean_group": float(np.mean(x)),
            "Mean_priv": float(np.mean(base)),
            "Median_group": float(np.median(x)),
            "Median_priv": float(np.median(base)),
            "MWU_U": float(U),
            "p_value": float(p),
            "Cliffs_delta": float(delta),
            "Cohens_d": float(d),
            "MedianDiff(group-priv)": float(med_diff),
            "CI_low": float(lo),
            "CI_high": float(hi),
        })

    out = pd.DataFrame(results)
    if not out.empty:
        p_adj, reject = fdr_bh(out["p_value"].to_numpy(), alpha=0.05)
        out["p_fdr"] = p_adj
        out["reject@0.05FDR"] = reject
        def stars(pv: float) -> str:
            return "****" if pv < 1e-4 else "***" if pv < 1e-3 else "**" if pv < 1e-2 else "*" if pv < 0.05 else "ns"
        out["sig"] = [stars(p) for p in out["p_fdr"]]
        out = out.sort_values(["p_fdr", "Group"]).reset_index(drop=True)

    _tend(f"bias.run_bias_tests[{value_col}]", t0)
    return out


# --- 6) Rounding & formatting ------------------------------------------------
def _round_bias_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Round columns to consistent units:
    - N_* as ints; MWU_U as int
    - Ratings means/medians: 1 decimal; views (log): 2 decimals
    - Effect sizes & CI: 3 decimals; p_fdr: 3 sig figs
    """
    if df.empty:
        return df

    out = df.copy()
    for c in ["N_group", "N_priv"]:
        if c in out:
            out[c] = out[c].astype(int)
    if "MWU_U" in out:
        out["MWU_U"] = out["MWU_U"].round(0).astype(int)

    if target == "rating":
        mm = 1
    else:
        mm = 2  # log10 views usually benefits from 2 d.p.

    for c in ["Mean_group", "Mean_priv", "Median_group", "Median_priv"]:
        if c in out:
            out[c] = out[c].round(mm)

    for c in ["Cliffs_delta", "Cohens_d", "MedianDiff(group-priv)", "CI_low", "CI_high"]:
        if c in out:
            out[c] = out[c].astype(float).round(3)

    if "p_fdr" in out:
        out["p_fdr"] = out["p_fdr"].apply(lambda x: float(f"{x:.3g}") if pd.notnull(x) else x)

    return out


# --- 7) Visualization: forest plots -----------------------------------------
def _prep_forest(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a tidy DataFrame for forest plotting from bias test results.
    """
    if data.empty:
        return data
    return data.rename(columns={
        "MedianDiff(group-priv)": "effect",
        "CI_low": "lo",
        "CI_high": "hi"
    })[["Group", "effect", "lo", "hi", "sig", "N_group", "N_priv"]]


@plot_dual_theme(section='fairness')
def plot_forest(data: pd.DataFrame, title: str, ax=None, palette=None, **kwargs):
    """
    Forest plot of median difference with 95% bootstrap CI.
    (No seaborn boxplot -> no palette/hue warnings.)
    """
    if data.empty:
        ax.text(0.5, 0.5, "No groups met minimum N", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_axis_off(); return

    y = np.arange(len(data))[::-1]
    ax.hlines(y, data["lo"], data["hi"])
    ax.plot(data["effect"], y, "o")

    # annotate significance + Ns
    for i, row in enumerate(data.itertuples(index=False)):
        ax.text(row.hi, y[i], f"  {row.sig}  (N={row.N_group:,}/{row.N_priv:,})",
                va="center", ha="left", fontsize=9)

    ax.axvline(0.0, color=ax.spines['left'].get_edgecolor(), linestyle='--', alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(list(data["Group"]))
    ax.set_xlabel("Median difference vs. privileged (group − priv)")
    ax.set_title(title)
    ax.set_ylabel(None)
    plt.tight_layout()


# --- 8) Qualitative summary --------------------------------------------------
def print_qualitative_summary(res_ratings: pd.DataFrame, res_views: pd.DataFrame) -> None:
    """
    Print interpretable highlights and outliers for quick diagnostics.

    Implementation notes
    --------------------
    - We iterate via dicts (records) instead of namedtuples so that columns
      with non-identifier names (e.g., 'MedianDiff(group-priv)') are accessed
      safely without relying on attribute names.
    - Keeps consistent rounding via _round_bias_table().
    - Emits a [TIME] line when done.
    """
    t0 = _t0("Summarising qualitative highlights & outliers...")

    def _top(df: pd.DataFrame, *, label: str) -> None:
        if df.empty:
            print(f"\nNo signals for {label}."); return
        # Pick top-5 by FDR then Group for stable ordering; round for display
        show = df.sort_values(["p_fdr", "Group"]).head(5).copy()
        show = _round_bias_table(show, "rating" if "rating" in label.lower() else "views")

        print(f"\nTop 5 signals — {label}:")
        for r in show.to_dict(orient="records"):
            md = r.get("MedianDiff(group-priv)")
            lo = r.get("CI_low")
            hi = r.get("CI_high")
            print(
                f"  - {r['Group']}: Δ̃={md:.3f} "
                f"[{lo:.3f}, {hi:.3f}] "
                f"p_FDR={r['p_fdr']} ({r['sig']}); "
                f"N={r['N_group']:,}/{r['N_priv']:,}"
            )

    _top(res_ratings, label="ratings")
    _top(res_views,   label="views (log10)")

    _tend("bias.summary", t0)


# --- 9) Main -----------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Run Step 05 bias tests on the REAL corpus by default; reproducible self-check available.

    CLI
    ---
    # Full run (privileged=White Women)
    python3 src/analysis/05_statistical_bias_tests.py

    # Self-check on a random sample (uses real parquet; writes *_selfcheck artefacts)
    python3 src/analysis/05_statistical_bias_tests.py --selfcheck --sample 80000
    """
    tall = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true",
                        help="Sample randomly from the real parquet (seeded) and write *_selfcheck artefacts.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Rows for self-check (default=min(N, 80k)).")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for sampling/bootstrap (defaults to config seed).")
    parser.add_argument("--privileged", type=str, default="White Women",
                        help="Privileged baseline group name.")
    parser.add_argument("--min_n", type=int, default=50,
                        help="Minimum N per group to include in tests.")
    parser.add_argument("--n_boot", type=int, default=2000,
                        help="Bootstrap reps for median CI.")
    args = parser.parse_args(argv)

    print("--- Starting Step 05: Statistical Bias Tests ---")
    print("[NOTE] Titles may be non-English; tags/categories help anchor semantics (multi-label).")
    print(f"[READ] Parquet: {CORPUS_PATH}")

    df = pd.read_parquet(CORPUS_PATH)

    # Self-check sampling (non-destructive)
    if args.selfcheck:
        t0 = _t0(f"[SELF-CHECK] Drawing random sample (seed={args.seed}) ...")
        n = args.sample or min(80_000, len(df))
        df = df.sample(n=n, random_state=args.seed, replace=False).reset_index(drop=True)
        print(f"[SELF-CHECK] Sample size: {len(df):,}")
        _tend("bias.selfcheck.sample", t0)

    _ensure_columns(df, ["rating", "views", "categories", "tags"])
    total_videos = len(df)
    print(f"[STATS] Total videos in corpus used: {total_videos:,}")
    print("Note: per-video metrics; totals can exceed N due to multi-label data.")

    # Groups
    groups = build_groups(df)
    if args.privileged not in groups:
        print(f"✗ ERROR: Privileged group '{args.privileged}' not defined. Exiting.")
        return

    # RATINGS
    t0 = _t0("Testing RATINGS across groups...")
    res_ratings = run_bias_tests(
        df=df, value_col="rating", groups=groups, privileged=args.privileged,
        min_n=args.min_n, log10_views=False, n_boot=args.n_boot, seed=args.seed
    )
    res_ratings = _round_bias_table(res_ratings, "rating")
    _tend("bias.section_ratings", t0)

    # VIEWS (log10)
    t0 = _t0("Testing VIEWS (log10) across groups...")
    res_views = run_bias_tests(
        df=df, value_col="views", groups=groups, privileged=args.privileged,
        min_n=args.min_n, log10_views=True, n_boot=args.n_boot, seed=args.seed
    )
    res_views = _round_bias_table(res_views, "views")
    _tend("bias.section_views", t0)

    print_qualitative_summary(res_ratings, res_views)

    # Save artefacts (separate for self-check)
    print("\nSaving data artefacts...")
    for d in (DATA_DIR, FIGURES_DIR, TABLES_DIR, NARR_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if args.selfcheck:
        res_ratings.to_csv(OUT_RATINGS_SC, index=False)
        res_views.to_csv(OUT_VIEWS_SC, index=False)
        print(f"✓ Artefact saved: {OUT_RATINGS_SC}")
        print(f"✓ Artefact saved: {OUT_VIEWS_SC}")
    else:
        res_ratings.to_csv(OUT_RATINGS, index=False)
        res_views.to_csv(OUT_VIEWS, index=False)
        print(f"✓ Artefact saved: {OUT_RATINGS}")
        print(f"✓ Artefact saved: {OUT_VIEWS}")

    # Forest plots
    print("\nGenerating visualizations...")
    df_r_plot = _prep_forest(res_ratings)
    df_v_plot = _prep_forest(res_views)
    base_r = OUT_FIG_R_SC if args.selfcheck else OUT_FIG_R
    base_v = OUT_FIG_V_SC if args.selfcheck else OUT_FIG_V
    plot_forest(data=df_r_plot, title=f"Ratings: group vs. {args.privileged}",
                save_path=str(base_r), figsize=(11, 7))
    plot_forest(data=df_v_plot, title=f"log10(views+1): group vs. {args.privileged}",
                save_path=str(base_v), figsize=(11, 7))

    # Narrative + LaTeX
    print("\nGenerating narrative & LaTeX tables...")

    def _lead(df_: pd.DataFrame) -> str:
        if df_.empty:
            return "No groups met the minimum sample size threshold."
        row = df_.sort_values(["p_fdr", "Group"]).iloc[0]
        return (
            f"Strongest signal: {row['Group']} vs {row['Privileged']}, "
            f"median diff={row['MedianDiff(group-priv)']:.3f} "
            f"[{row['CI_low']:.3f}, {row['CI_high']:.3f}], "
            f"p_FDR={row['p_fdr']} ({row['sig']})."
        )

    def _sig_counts(df_: pd.DataFrame) -> tuple[int, int]:
        if df_.empty or "reject@0.05FDR" not in df_.columns:
            return 0, 0
        return int(df_["reject@0.05FDR"].sum()), int(len(df_))

    def _top_abs_delta(df_: pd.DataFrame) -> Optional[str]:
        if df_.empty or "Cliffs_delta" not in df_.columns or df_["Cliffs_delta"].isna().all():
            return None
        i = df_["Cliffs_delta"].abs().idxmax()
        row = df_.loc[i]
        return f"{row['Group']} (δ={float(row['Cliffs_delta']):.3f})"

    sig_r, tot_r = _sig_counts(res_ratings)
    sig_v, tot_v = _sig_counts(res_views)
    top_delta_r = _top_abs_delta(res_ratings)
    top_delta_v = _top_abs_delta(res_views)

    narrative = f"""
    # Automated Summary: Statistical Bias Tests

    **Corpus size:** {total_videos:,} videos.
    Metrics are per video (document frequency). Because videos are multi-label,
    sums across labels may exceed N — expected.

    **Significance @ 5% FDR:** ratings {sig_r}/{tot_r}; views {sig_v}/{tot_v}.
    {f"Strongest |Cliff’s δ| (ratings): {top_delta_r}" if top_delta_r else ""}
    {f"Strongest |Cliff’s δ| (views): {top_delta_v}" if top_delta_v else ""}

    ## Ratings (two-sided Mann–Whitney U, FDR-corrected)
    {_lead(res_ratings)}

    ## Views (log10(views+1))
    {_lead(res_views)}
    """.strip() + "\n"

    with open(NARR_PATH_SC if args.selfcheck else NARR_PATH, "w") as f:
        f.write(narrative)
    print(f"✓ Narrative saved: {(NARR_PATH_SC if args.selfcheck else NARR_PATH)}")


    if not res_ratings.empty:
        dataframe_to_latex_table(
            df=res_ratings.set_index("Group"),
            save_path=str(OUT_TEX_R_SC if args.selfcheck else OUT_TEX_R),
            caption=f"Ratings: Group vs {args.privileged} (Mann–Whitney U; FDR-corrected).",
            label="tab:05-bias-ratings",
            note="Effect sizes include Cliff's δ and Cohen's d. CI is bootstrap for median difference."
        )
    if not res_views.empty:
        dataframe_to_latex_table(
            df=res_views.set_index("Group"),
            save_path=str(OUT_TEX_V_SC if args.selfcheck else OUT_TEX_V),
            caption=f"log10(views+1): Group vs {args.privileged} (Mann–Whitney U; FDR-corrected).",
            label="tab:05-bias-views",
            note="CI is bootstrap for median difference."
        )

    _tend("bias.step05_total", tall)
    print("\n--- Step 05: Statistical Bias Tests Completed Successfully ---")


if __name__ == "__main__":
    main()
