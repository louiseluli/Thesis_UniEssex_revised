# -*- coding: utf-8 -*-
"""
12_postprocessing_mitigation.py
===============================

Purpose
-------
Post-processing bias mitigation using Fairlearn's ThresholdOptimizer with
Equalized Odds. Starting from a pre-trained baseline pipeline (prefers
'rf_baseline.joblib', falls back to 'rf_reweighed.joblib'), we learn
group-conditional thresholds on the validation split and apply them to Val/Test.

What it does
------------
1) Loads canonical parquet and Step-06 splits (or creates an internal split for
   self-check). Enforces required columns; aligns features to baseline expectations.
2) Loads a fitted baseline model pipeline (joblib).
3) Scores baseline probabilities for Val/Test.
4) Fits ThresholdOptimizer (Equalized Odds) on Val with intersectional sensitive
   features ('Asian/Black/Latina/White Women'; else 'Other').
5) Predicts mitigated labels on Val/Test; evaluates overall & by-group; computes
   disparities vs White Women; exports CSVs, LaTeX, dual-theme margin plots, and a
   narrative. Shows the top-10 most confident mistakes.

Reproducibility & config
------------------------
- Reads seed, paths, and column names from your config (seed=95).


CLI
---
# Full run (writes 12_* artefacts + legacy egpp_*):
python3 src/fairness/12_postprocessing_mitigation.py --use-gold

# Self-check (random sample; non-destructive 12_*_selfcheck artefacts):
python3 src/fairness/12_postprocessing_mitigation.py --selfcheck --sample 80000 --use-gold
"""

from __future__ import annotations

# --- Imports (keep at top) ---------------------------------------------------
import sys
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.postprocessing import ThresholdOptimizer

# Project utils
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.utils.academic_tables import dataframe_to_latex_table
from src.fairness.fairness_evaluation_utils import (
    load_gold_table, maybe_override_targets, maybe_override_groups
)

# --- 1) Config & Paths -------------------------------------------------------
CONFIG = load_config()

DATA_DIR    = Path(CONFIG['paths']['data'])
FIG_DIR     = Path(CONFIG['paths']['figures']) / 'eval'
MODEL_DIR   = Path(CONFIG['paths'].get('models', Path(CONFIG['project']['root']) / 'outputs' / 'models'))
TEX_DIR     = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'
NARR_DIR    = Path(CONFIG['paths']['narratives']) / 'automated'
OUTPUT_DIR  = Path(CONFIG['paths']['outputs'])

# Align to Steps 09–11 conventions
CORPUS_PATH = DATA_DIR / '01_ml_corpus.parquet'
TRAIN_IDS   = DATA_DIR / '06_train_ids.csv'
VAL_IDS     = DATA_DIR / '06_val_ids.csv'
TEST_IDS    = DATA_DIR / '06_test_ids.csv'

BASELINE_CANDIDATES = [
    MODEL_DIR / 'rf_baseline.joblib',   # Step-07 baseline (preferred)
    MODEL_DIR / 'rf_reweighed.joblib',  # Step-10 model (fallback)
]

SEED        = int(CONFIG.get('reproducibility', {}).get('seed', 95))
TEXT_COL    = 'model_input_text'
NUM_FALLBACK= ['duration', 'ratings']  # if baseline expects others, we align
CATS_COL    = 'categories'

# --- 2) Lightweight timers ---------------------------------------------------
def _t0(msg: str) -> float:
    """
    Start a timer and print a standardized header.

    Parameters
    ----------
    msg : str
        Message to print at the start.

    Returns
    -------
    float
        High-resolution start time.
    """
    t = time.perf_counter()
    print(msg)
    return t

def _tend(label: str, t0: float) -> None:
    """
    Stop a timer and print a standardized [TIME] line.

    Parameters
    ----------
    label : str
        Short label for the timed block.
    t0 : float
        Start time returned by _t0.
    """
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")

# --- 3) Data helpers ----------------------------------------------------------
def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required text/numeric columns exist. If TEXT_COL is missing, fallback to title+tags.
    """
    d = df.copy()
    if TEXT_COL not in d.columns:
        d[TEXT_COL] = (d.get('title', '').fillna('').astype(str) + ' ' +
                       d.get('tags',  '').fillna('').astype(str))
    for c in NUM_FALLBACK:
        if c not in d.columns:
            d[c] = 0
    if CATS_COL not in d.columns:
        d[CATS_COL] = ''
    return d

def _load_corpus() -> pd.DataFrame:
    """
    Load the canonical parquet and enforce required columns.

    Returns
    -------
    pd.DataFrame
        Canonical corpus with safe columns present.
    """
    t0 = _t0(f"[READ] Parquet: {CORPUS_PATH}")
    df = pd.read_parquet(CORPUS_PATH)
    _tend("postproc.load_corpus", t0)
    return _ensure_columns(df)

def _read_ids_or_none(path: Path) -> Optional[pd.Index]:
    """
    Read a single-column CSV of video IDs.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Optional[pd.Index]
        Index of IDs or None if not found.
    """
    if path.exists():
        ids = pd.read_csv(path)['video_id']
        return pd.Index(ids)
    return None

def _subset_by_ids(df: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    """
    Subset dataframe by IDs, preserving order of df.

    Parameters
    ----------
    df : pd.DataFrame
    ids : pd.Index

    Returns
    -------
    pd.DataFrame
    """
    return df[df['video_id'].isin(ids)].reset_index(drop=True)

def _make_internal_split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a 60/20/20 internal split for self-check, stratified on female intersections.

    Returns
    -------
    (train, val, test) : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    def stratify_key(d: pd.DataFrame) -> pd.Series:
        key = np.full(len(d), "Other", dtype=object)
        gf = (d.get('gender_female', 0) == 1)
        bw = (d.get('race_ethnicity_black', 0) == 1) & gf
        ww = (d.get('race_ethnicity_white', 0) == 1) & gf & ~bw
        aw = (d.get('race_ethnicity_asian', 0) == 1) & gf & ~bw & ~ww
        lw = (d.get('race_ethnicity_latina', 0) == 1) & gf & ~bw & ~ww & ~aw
        key[bw] = "Black_Women"; key[ww] = "White_Women"; key[aw] = "Asian_Women"; key[lw] = "Latina_Women"
        return pd.Series(key, index=d.index, name="stratify_key")

    d = df.copy()
    d['stratify_key'] = stratify_key(d)
    from sklearn.model_selection import train_test_split
    trv, te = train_test_split(d, test_size=0.20, random_state=seed, stratify=d['stratify_key'])
    tr, va = train_test_split(trv, test_size=0.25, random_state=seed, stratify=trv['stratify_key'])
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

# --- 4) Groups & targets ------------------------------------------------------
def _prepare_binary_target(df: pd.DataFrame, positive_class: str = "Amateur") -> np.ndarray:
    """
    Default binary target: 1 if primary category equals positive_class, else 0.

    Parameters
    ----------
    df : pd.DataFrame
    positive_class : str

    Returns
    -------
    np.ndarray
        0/1 labels.
    """
    primary = df[CATS_COL].fillna("").astype(str).str.split(",").str[0].str.strip()
    return (primary == positive_class).astype(int).to_numpy()

def _group_labels(df: pd.DataFrame) -> pd.Series:
    """
    Readable group labels for fairness diagnostics (not fed to model).

    Returns
    -------
    pd.Series
        {'Black Women','White Women','Asian Women','Latina Women','Other'}.
    """
    labels = np.full(len(df), "Other", dtype=object)
    gf = (df.get('gender_female', 0) == 1)
    bw = (df.get('race_ethnicity_black', 0) == 1) & gf
    ww = (df.get('race_ethnicity_white', 0) == 1) & gf & ~bw
    aw = (df.get('race_ethnicity_asian', 0) == 1) & gf & ~bw & ~ww
    lw = (df.get('race_ethnicity_latina', 0) == 1) & gf & ~bw & ~ww & ~aw
    labels[bw] = "Black Women"; labels[ww] = "White Women"; labels[aw] = "Asian Women"; labels[lw] = "Latina Women"
    return pd.Series(labels, index=df.index, name="Group")

# --- 5) Metrics ---------------------------------------------------------------
@dataclass
class ClassifMetrics:
    """Container for standard binary classification metrics."""
    acc: float; prec: float; rec: float; f1: float

def _overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassifMetrics:
    """
    Compute Accuracy, Precision, Recall, F1 (rounded to 3 decimals).

    Returns
    -------
    ClassifMetrics
    """
    return ClassifMetrics(
        acc = float(np.round(accuracy_score(y_true, y_pred), 3)),
        prec= float(np.round(precision_score(y_true, y_pred, zero_division=0), 3)),
        rec = float(np.round(recall_score(y_true, y_pred, zero_division=0), 3)),
        f1  = float(np.round(f1_score(y_true, y_pred, zero_division=0), 3)),
    )

def _group_metrics(df_meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Compute metrics per intersectional group (N, Accuracy, Precision, Recall, F1).

    Returns
    -------
    pd.DataFrame
        Group-wise metrics table.
    """
    g = df_meta["Group"] if "Group" in df_meta.columns else _group_labels(df_meta)
    out = []
    for grp, idx in df_meta.groupby(g).groups.items():
        yt, yp = y_true[idx], y_pred[idx]
        out.append({
            "Group": grp,
            "N": int(len(idx)),
            "Accuracy": round(accuracy_score(yt, yp), 3),
            "Precision": round(precision_score(yt, yp, zero_division=0), 3),
            "Recall": round(recall_score(yt, yp, zero_division=0), 3),
            "F1": round(f1_score(yt, yp, zero_division=0), 3)
        })
    return pd.DataFrame(out).sort_values("Group").reset_index(drop=True)

def _disparities_vs_priv(df_group: pd.DataFrame, privileged: str = "White Women") -> pd.DataFrame:
    """
    Disparities vs a privileged group (priv - group) for Accuracy, TPR, Precision.

    Returns
    -------
    pd.DataFrame
        Columns: Comparison Group, Accuracy Disparity, Equal Opportunity Difference, Precision Disparity
    """
    if df_group.empty or privileged not in df_group["Group"].values:
        return pd.DataFrame(columns=["Comparison Group","Accuracy Disparity","Equal Opportunity Difference","Precision Disparity"])
    base = df_group.loc[df_group['Group'] == privileged].iloc[0]
    rows = []
    for _, r in df_group.iterrows():
        if r['Group'] == privileged:
            continue
        rows.append({
            "Comparison Group": r['Group'],
            "Accuracy Disparity": round(base["Accuracy"] - r["Accuracy"], 3),
            "Equal Opportunity Difference": round(base["Recall"] - r["Recall"], 3),
            "Precision Disparity": round(base["Precision"] - r["Precision"], 3),
        })
    return pd.DataFrame(rows).sort_values("Comparison Group").reset_index(drop=True)

# --- 6) Baseline alignment & scoring -----------------------------------------
def _align_to_baseline_expected(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Align incoming features to baseline RF pipeline expectations:
    - text column 'combined_text_clean'
    - numeric 'rating', 'views'

    Parameters
    ----------
    df_in : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Feature-aligned frame.
    """
    d = df_in.copy()
    if "combined_text_clean" not in d.columns:
        if TEXT_COL in d.columns:
            d["combined_text_clean"] = d[TEXT_COL].astype(str)
        else:
            d["combined_text_clean"] = (d.get("title", "").astype(str) + " " + d.get("tags", "").astype(str)).fillna("")
    if "rating" not in d.columns:
        d["rating"] = pd.to_numeric(d.get("ratings", 0.0), errors="coerce").fillna(0.0)
    if "views" not in d.columns:
        d["views"] = pd.to_numeric(d.get("views", d.get("view_count", 0.0)), errors="coerce").fillna(0.0)
    return d

def _baseline_probs(model, X: pd.DataFrame) -> np.ndarray:
    """
    Get baseline probabilities of positive class (class 1).

    Parameters
    ----------
    model : Any
        Prefit sklearn pipeline supporting predict_proba.
    X : pd.DataFrame

    Returns
    -------
    np.ndarray
        Probabilities in [0,1].
    """
    return model.predict_proba(X)[:, 1]

# --- 7) Plotting & outliers ---------------------------------------------------
@plot_dual_theme(section='fairness')
def _plot_margins(margins: np.ndarray, title: str, ax=None, palette=None, **kwargs):
    """
    Histogram of decision margins (p - 0.5).

    Notes
    -----
    - Matplotlib histogram, so no palette/hue issues.
    """
    ax.hist(margins, bins=50)
    ax.set_title(title)
    ax.set_xlabel("Decision margin (p - 0.5)")
    ax.set_ylabel("Count")

def _log_basic_outliers(df: pd.DataFrame) -> None:
    """
    Log simple 99th-percentile outliers (non-destructive) for situational awareness.

    Formatting
    ----------
    - ratings rounded to 1 decimal
    - other counts/durations rounded to integers
    """
    for col in ["duration", "views", "ratings"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() == 0:
                continue
            q99 = s.quantile(0.99)
            rounded = round(float(q99), 1 if col == "ratings" else 0)
            n_hi = int((s > q99).sum())
            print(f"[OUTLIERS] {col}: {n_hi:,} above 99th percentile (~{rounded})")

# --- 8) Train + Evaluate (Post-process) --------------------------------------
def fit_postprocess_and_eval(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    model_candidates: List[Path],
    use_gold: bool,
    gold: Optional[pd.DataFrame]
) -> Dict[str, object]:
    """
    Fit ThresholdOptimizer (Equalized Odds) on Val using a prefit baseline model,
    optionally overriding labels/groups with GOLD; evaluate on Val/Test; return artefacts.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        Splits.
    model_candidates : List[Path]
        Ordered list of candidate joblib models to load.
    use_gold : bool
        Whether to attempt GOLD overrides.
    gold : Optional[pd.DataFrame]
        GOLD table if available.

    Returns
    -------
    Dict[str, object]
        Keys: optimizer, metrics_overall, metrics_group, disparities,
              preds_test_full, outliers_topk, margins
    """
    # Load model (prefer baseline; fallback to reweighed)
    chosen = None
    for p in model_candidates:
        if p.exists():
            chosen = p
            break
    if chosen is None:
        raise FileNotFoundError(
            f"No baseline model found. Tried: {', '.join(str(p) for p in model_candidates)}"
        )
    print(f"Loading baseline model from {chosen}...")
    base = joblib.load(chosen)

    # Align features
    t0 = _t0("Aligning features to baseline pipeline expectations (Val/Test) ...")
    X_val  = _align_to_baseline_expected(df_val)
    X_test = _align_to_baseline_expected(df_test)
    _tend("postproc.align_features", t0)

    # GOLD-aware targets & groups
    y_val_def = _prepare_binary_target(df_val)
    y_tst_def = _prepare_binary_target(df_test)
    g_val_def = _group_labels(df_val)
    g_tst_def = _group_labels(df_test)

    y_val, cov_val = maybe_override_targets(df_val,  y_val_def, gold)
    y_tst, cov_tst = maybe_override_targets(df_test, y_tst_def, gold)
    g_val = maybe_override_groups(df_val,  g_val_def, gold)
    g_tst = maybe_override_groups(df_test, g_tst_def, gold)

    # Baseline probabilities
    t0 = _t0("Scoring baseline probabilities (Val/Test) ...")
    p_val_base = _baseline_probs(base, X_val)
    p_tst_base = _baseline_probs(base, X_test)
    _tend("postproc.baseline_score", t0)

    # ThresholdOptimizer fit & predict
    print("Fitting ThresholdOptimizer (constraint=equalized_odds) on Val ...")
    t0 = _t0("postproc.fit")
    opt = ThresholdOptimizer(estimator=base, constraints="equalized_odds", prefit=True)
    opt.fit(X_val, y_val, sensitive_features=g_val)
    _tend("postproc.fit", t0)

    print("Predicting with ThresholdOptimizer on Val/Test ...")
    t0 = _t0("postproc.predict")
    y_val_mit = opt.predict(X_val, sensitive_features=g_val)
    y_tst_mit = opt.predict(X_test, sensitive_features=g_tst)
    _tend("postproc.predict", t0)
    print(f"[INFO] GOLD coverage (Val/Test): {cov_val:.2%} / {cov_tst:.2%}")

    # Overall metrics
    mo_val = _overall_metrics(y_val, y_val_mit)
    mo_tst = _overall_metrics(y_tst, y_tst_mit)
    metrics_overall = pd.DataFrame([
        {"Split": "Val",  "Accuracy": mo_val.acc, "Precision": mo_val.prec, "Recall": mo_val.rec, "F1": mo_val.f1},
        {"Split": "Test", "Accuracy": mo_tst.acc, "Precision": mo_tst.prec, "Recall": mo_tst.rec, "F1": mo_tst.f1},
    ])

    # Group metrics & disparities (Test)
    df_test_groups = df_test.copy()
    df_test_groups["Group"] = g_tst.values
    mg   = _group_metrics(df_test_groups, y_tst, y_tst_mit)
    disp = _disparities_vs_priv(mg, privileged="White Women")

    # Full Test predictions + margins (use baseline probs for interpretability)
    preds_test_full = pd.DataFrame({
        "video_id": df_test["video_id"].to_numpy(),
        "title": df_test.get("title", pd.Series([""]*len(df_test))).to_numpy(),
        "Group": g_tst.to_numpy(),
        "y_true": y_tst,
        "y_pred": y_tst_mit,
        "prob":  np.round(p_tst_base, 3)
    })
    preds_test_full["margin"] = preds_test_full["prob"] - 0.5

    # Top-10 outliers (most confident mistakes under postproc decisions)
    wrong = np.where(y_tst_mit != y_tst)[0]
    margins = np.abs(p_tst_base[wrong] - 0.5)
    top_idx = wrong[np.argsort(-margins)[:10]]
    out_rows = []
    labels = g_tst.to_numpy()
    for i in top_idx:
        out_rows.append({
            "video_id": df_test.iloc[i]["video_id"],
            "title": df_test.iloc[i].get("title", ""),
            "Group": labels[i],
            "y_true": int(y_tst[i]),
            "y_pred": int(y_tst_mit[i]),
            "prob": float(np.round(p_tst_base[i], 3)),
            "margin_abs": float(np.round(abs(p_tst_base[i] - 0.5), 3))
        })
    outliers_topk = pd.DataFrame(out_rows)

    return {
        "optimizer": opt,
        "metrics_overall": metrics_overall,
        "metrics_group": mg,
        "disparities": disp,
        "preds_test_full": preds_test_full,
        "outliers_topk": outliers_topk,
        "margins": preds_test_full["margin"].to_numpy()
    }

# --- 9) Save & Narrative ------------------------------------------------------
@plot_dual_theme(section='fairness')
def _plot_margins_wrapper(margins: np.ndarray, title: str, ax=None, palette=None, **kwargs):
    """
    Wrapper for project theme; draws histogram of margins.

    Parameters
    ----------
    margins : np.ndarray
    title : str
    ax, palette, **kwargs : Any
    """
    ax.hist(margins, bins=50)
    ax.set_title(title)
    ax.set_xlabel("Decision margin (p - 0.5)")
    ax.set_ylabel("Count")

def _save_all(bundle: Dict[str, object], *, is_selfcheck: bool = False) -> None:
    """
    Persist metrics, predictions, plots, and LaTeX tables for post-processing.

    Notes
    -----
    - Writes step-aligned names (12_postproc_*) and legacy egpp_* for compatibility.
    - Self-check writes *_selfcheck and does not overwrite full artefacts.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    NARR_DIR.mkdir(parents=True, exist_ok=True)

    suff = "_selfcheck" if is_selfcheck else ""

    # Step-12 friendly names
    mo_12 = DATA_DIR / f"12_postproc_overall_metrics{suff}.csv"
    mg_12 = DATA_DIR / f"12_postproc_group_metrics{suff}.csv"
    md_12 = DATA_DIR / f"12_postproc_disparities{suff}.csv"
    pt_12 = DATA_DIR / f"12_postproc_predictions_test{suff}.csv"
    ot_12 = DATA_DIR / f"12_postproc_outliers_top10{suff}.csv"

    # Legacy (kept)
    mo_eg = DATA_DIR / f"egpp_overall_metrics{'_postproc_selfcheck' if is_selfcheck else '_postproc'}.csv"
    mg_eg = DATA_DIR / f"egpp_group_metrics{'_postproc_selfcheck' if is_selfcheck else '_postproc'}.csv"
    md_eg = DATA_DIR / f"egpp_disparities{'_postproc_selfcheck' if is_selfcheck else '_postproc'}.csv"
    pt_eg = DATA_DIR / f"egpp_predictions_test{'_postproc_selfcheck' if is_selfcheck else '_postproc'}.csv"
    ot_eg = DATA_DIR / f"egpp_outliers_top10{'_postproc_selfcheck' if is_selfcheck else '_postproc'}.csv"

    # Save CSVs (both new + legacy)
    bundle["metrics_overall"].to_csv(mo_12, index=False); bundle["metrics_overall"].to_csv(mo_eg, index=False)
    bundle["metrics_group"].to_csv(mg_12, index=False);   bundle["metrics_group"].to_csv(mg_eg, index=False)
    bundle["disparities"].to_csv(md_12, index=False);     bundle["disparities"].to_csv(md_eg, index=False)
    bundle["preds_test_full"].to_csv(pt_12, index=False); bundle["preds_test_full"].to_csv(pt_eg, index=False)
    bundle["outliers_topk"].to_csv(ot_12, index=False);   bundle["outliers_topk"].to_csv(ot_eg, index=False)

    print("✓ Artefacts saved:",
          mo_12.name, ",", mg_12.name, ",", md_12.name, ",", pt_12.name, ",", ot_12.name,
          f"(+ legacy: {mo_eg.name}, {mg_eg.name}, {md_eg.name}, {pt_eg.name}, {ot_eg.name})")

    # Plots
    _plot_margins_wrapper(
        margins=bundle["margins"],
        title="Decision Margins (Test) — Post-processing (Equalized Odds)",
        save_path=str(FIG_DIR / f"12_postproc_margins{suff}"),
        figsize=(9, 6)
    )

    # LaTeX (group metrics)
    dataframe_to_latex_table(
        bundle["metrics_group"].set_index("Group"),
        str(TEX_DIR / f"12_postproc_group_metrics{suff}.tex"),
        "Group-wise performance under post-processing (ThresholdOptimizer, Equalized Odds).",
        "tab:12-postproc-group-metrics"
    )

def _write_narrative(bundle: Dict[str,object], *, is_selfcheck: bool) -> None:
    """
    Save a short Markdown narrative summarizing performance & fairness notes.

    Parameters
    ----------
    bundle : Dict[str, object]
    is_selfcheck : bool
    """
    suff = "_selfcheck" if is_selfcheck else ""
    lines = [f"# Automated Summary: Post-Processing (ThresholdOptimizer, EO){' — self-check' if is_selfcheck else ''}\n"]
    m = bundle["metrics_overall"]
    lines.append("## Overall Metrics\n")
    lines.append(m.to_string(index=False))
    lines.append("\n## Group Metrics\n")
    lines.append(bundle["metrics_group"].to_string(index=False))
    lines.append("\n## Disparities vs. White Women\n")
    lines.append(bundle["disparities"].to_string(index=False))
    lines.append("\n## Top 5 Outliers (by confident mistakes)\n")
    lines.append(bundle["outliers_topk"].head(5).to_string(index=False))
    lines.append("\n*Note:* Titles in other languages may be harder to interpret; tags/categories help anchor semantics.")
    path = NARR_DIR / f"12_postprocessing_summary{suff}.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Narrative saved: {path.resolve()}")

# --- 10) Main ----------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Run post-processing mitigation with ThresholdOptimizer (Equalized Odds),
    optionally overriding labels/groups with GOLD. Includes timers, outlier logs,
    qualitative console output, and a total runtime line.
    """
    t_all = time.perf_counter()
    print("--- Starting Step 12: Post-processing Mitigation (ThresholdOptimizer) ---")

    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--use-gold", action="store_true", help="Use GOLD labels/groups for evaluation if available.")
    p.add_argument("--gold-path", type=str, default=str(OUTPUT_DIR / "data" / "gold" / "gold_final.csv"))
    args = p.parse_args(argv)

    from pathlib import Path as _P
    gold = load_gold_table(_P(args.gold_path)) if args.use_gold else None

    # Load corpus
    df = _load_corpus()
    total = len(df)
    print(f"[STATS] Total videos available: {total:,}")
    print("[NOTE] Titles may be non-English; tags/categories help anchor semantics.")
    print("[NOTE] Logging basic outliers for situational awareness (non-destructive).")
    _log_basic_outliers(df)

    # Build splits
    if args.selfcheck:
        n = args.sample or min(80000, total)
        df = df.sample(n=n, random_state=args.seed, replace=False).reset_index(drop=True)
        print(f"[SELF-CHECK] Random sample drawn: {len(df):,} rows (seed={args.seed}).")
        train_df, val_df, test_df = _make_internal_split(df, seed=args.seed)
    else:
        if args.sample:
            df = df.sample(n=min(args.sample, total), random_state=args.seed, replace=False).reset_index(drop=True)
            print(f"[SAMPLE] Using {len(df):,} rows (seed={args.seed}).")
        tr_ids, va_ids, te_ids = _read_ids_or_none(TRAIN_IDS), _read_ids_or_none(VAL_IDS), _read_ids_or_none(TEST_IDS)
        if not (tr_ids is not None and va_ids is not None and te_ids is not None):
            print("✗ Step-06 IDs not found; falling back to internal split on full corpus.")
            train_df, val_df, test_df = _make_internal_split(df, seed=args.seed)
        else:
            train_df, val_df, test_df = _subset_by_ids(df, tr_ids), _subset_by_ids(df, va_ids), _subset_by_ids(df, te_ids)

    print(f"Split sizes: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    # Fit + evaluate post-processing
    bundle = fit_postprocess_and_eval(
        df_train=train_df, df_val=val_df, df_test=test_df,
        model_candidates=BASELINE_CANDIDATES,
        use_gold=args.use_gold,
        gold=gold
    )
    _save_all(bundle, is_selfcheck=args.selfcheck)

    print("\n=== Overall Metrics (Val/Test: Accuracy / Precision / Recall / F1) ===")
    print(bundle["metrics_overall"].to_string(index=False))
    print("\n=== Group Metrics (Test) ===")
    print(bundle["metrics_group"].to_string(index=False))
    print("\n=== Disparities vs. White Women (privileged) ===")
    print(bundle["disparities"].to_string(index=False))
    print("\n=== Top 10 outliers (Test) — most confident mistakes ===")
    print(bundle["outliers_topk"].to_string(index=False))

    _write_narrative(bundle, is_selfcheck=args.selfcheck)
    _tend("postproc.step12_total", t_all)
    print("\n--- Step 12: Post-processing Mitigation Completed Successfully ---")

if __name__ == '__main__':
    main()
