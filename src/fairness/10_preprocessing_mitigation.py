# -*- coding: utf-8 -*-
"""
10_preprocessing_mitigation.py
==============================

Purpose
-------
Pre-processing bias mitigation using the Kamiran–Calders Reweighing scheme.
We compute instance weights per (Group, Label) and train a RandomForest
classifier on text+numeric features. Evaluation is overall & by intersectional
groups, with disparities vs White Women, interpretable outliers, dual-theme plots,
LaTeX, and a short narrative. Optional GOLD can override labels/groups.

What it does
------------
1) Loads canonical parquet and Step-06 splits (or creates an internal split for
   self-check). Enforces required columns and aligns to baseline RF expectations.
2) Computes reweighing weights w(a,y) ∝ P(A=a)P(Y=y) / P(A=a,Y=y).
3) Trains a RF pipeline on Train with sample_weight; scores Val/Test.
4) Evaluates by group, computes disparities vs White Women, exports CSV/LaTeX,
   plots margins, writes a narrative with qualitative notes + top outliers.

Interpretability & language notes
---------------------------------
- Some titles are not in English; tags/categories often anchor semantics.
- We therefore surface **title + Group** for outliers to support qualitative review.

CLI
---
# Full run (writes step-10 artefacts):
python3 src/fairness/10_preprocessing_mitigation.py --use-gold

# Self-check (random sample & internal split; non-destructive *_selfcheck artefacts):
python3 src/fairness/10_preprocessing_mitigation.py --selfcheck --sample 80000 --use-gold
"""

from __future__ import annotations

# --- Imports (keep at top) ---------------------------------------------------
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

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
FIG_DIR     = Path(CONFIG['paths']['figures'])
MODEL_DIR   = Path(CONFIG['paths'].get('models', Path(CONFIG['project']['root']) / 'outputs' / 'models'))
TEX_DIR     = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'
NARR_DIR    = Path(CONFIG['paths']['narratives']) / 'automated'
OUTPUT_DIR  = Path(CONFIG['paths']['outputs'])

# Align to Step-01/06 names (consistent with 08/09)
CORPUS_PATH = DATA_DIR / '01_ml_corpus.parquet'
TRAIN_IDS   = DATA_DIR / '06_train_ids.csv'
VAL_IDS     = DATA_DIR / '06_val_ids.csv'
TEST_IDS    = DATA_DIR / '06_test_ids.csv'

SEED        = int(CONFIG.get('reproducibility', {}).get('seed', 95))

# Baseline-style columns used elsewhere
TEXT_COL        = 'model_input_text'
TEXT_BASELINE   = 'combined_text_clean'
NUM_FALLBACK    = ['duration', 'ratings']
NUM_BASELINE    = ['rating', 'views']
CATS_COL        = 'categories'

# --- 2) Lightweight timers ---------------------------------------------------
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
        Perf counter start time.
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
        Label describing the timed block.
    t0 : float
        Start time returned by _t0.
    """
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")

# --- 3) Data helpers ----------------------------------------------------------
def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist; build TEXT_COL if missing via title+tags; ensure numerics exist.

    Returns
    -------
    pd.DataFrame
        A defensive copy with TEXT_COL, CATS_COL, and basic numeric fallbacks.
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

def _align_to_baseline_expected(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Align to baseline RF expectations:
    - text column 'combined_text_clean' (fallback: TEXT_COL or title+tags)
    - numeric 'rating', 'views' (fallback: NUM_FALLBACK or zeros)

    Returns
    -------
    pd.DataFrame
        DataFrame guaranteed to have combined_text_clean, rating, views.
    """
    d = df_in.copy()
    if TEXT_BASELINE not in d.columns:
        d[TEXT_BASELINE] = d.get(TEXT_COL, (d.get("title", "").astype(str) + " " + d.get("tags", "").astype(str))).astype(str)
    if "rating" not in d.columns:
        d["rating"] = pd.to_numeric(d.get("rating", 0.0), errors="coerce").fillna(0.0)
    if "views" not in d.columns:
        d["views"] = pd.to_numeric(d.get("views", d.get("view_count", 0.0)), errors="coerce").fillna(0.0)
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
    _tend("reweigh.load_corpus", t0)
    return _ensure_columns(df)

def _read_ids_or_none(path: Path) -> Optional[pd.Index]:
    """Read a single-column CSV of video IDs; return Index or None."""
    if path.exists():
        ids = pd.read_csv(path)['video_id']
        return pd.Index(ids)
    return None

def _subset_by_ids(df: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    """Subset dataframe by IDs, preserving order of df."""
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
    trv, te = train_test_split(d, test_size=0.20, random_state=seed, stratify=d['stratify_key'])
    tr, va = train_test_split(trv, test_size=0.25, random_state=seed, stratify=trv['stratify_key'])
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

# --- 4) Task & groups ---------------------------------------------------------
def _prepare_binary_target(df: pd.DataFrame, positive_class: str = "Amateur") -> np.ndarray:
    """
    Default binary target: primary_category == positive_class.

    Returns
    -------
    np.ndarray of dtype int {0,1}
    """
    primary = df[CATS_COL].fillna("").astype(str).str.split(",").str[0].str.strip()
    return (primary == positive_class).astype(int).to_numpy()

def _group_labels(df: pd.DataFrame) -> pd.Series:
    """
    Readable group labels for fairness diagnostics (not fed to model).

    Priority order matches stratification key: Black > White > Asian > Latina > Other.

    Returns
    -------
    pd.Series named 'Group'
    """
    labels = np.full(len(df), "Other", dtype=object)
    gf = (df.get('gender_female', 0) == 1)
    bw = (df.get('race_ethnicity_black', 0) == 1) & gf
    ww = (df.get('race_ethnicity_white', 0) == 1) & gf & ~bw
    aw = (df.get('race_ethnicity_asian', 0) == 1) & gf & ~bw & ~ww
    lw = (df.get('race_ethnicity_latina', 0) == 1) & gf & ~bw & ~ww & ~aw
    labels[bw] = "Black Women"; labels[ww] = "White Women"; labels[aw] = "Asian Women"; labels[lw] = "Latina Women"
    return pd.Series(labels, index=df.index, name="Group")

# --- 5) Reweighing weights ----------------------------------------------------
def _reweighing_weights(y: np.ndarray, groups: pd.Series) -> np.ndarray:
    """
    Compute Kamiran–Calders reweighing weights:
      w(a,y) ∝ P(A=a) * P(Y=y) / P(A=a, Y=y)

    Returns
    -------
    np.ndarray
        Sample-weight per instance (normalized to mean ~ 1.0).
    """
    a = groups.to_numpy()
    dfw = pd.DataFrame({"a": a, "y": y})
    pa = dfw["a"].value_counts(normalize=True)
    py = dfw["y"].value_counts(normalize=True)
    pay = dfw.value_counts(normalize=True)
    w = np.ones(len(y), dtype=float)
    for (ai, yi), p_ay in pay.items():
        w_ay = (pa[ai] * py[yi]) / max(p_ay, 1e-12)
        w[(a == ai) & (y == yi)] = w_ay
    w = w * (len(y) / w.sum())  # normalize
    return w

# --- 6) Features & model ------------------------------------------------------
def build_preprocessor(seed: int, n_text_components: int = 256, n_hash_features: int = 2**18) -> ColumnTransformer:
    """
    ColumnTransformer:
      - text -> HashingVectorizer -> TruncatedSVD (dense)
      - numeric -> passthrough

    Uses baseline-friendly columns (combined_text_clean, rating, views).
    """
    text_pipe = Pipeline(steps=[
        ("hash", HashingVectorizer(
            n_features=n_hash_features,
            alternate_sign=False,
            ngram_range=(1, 2),
            norm="l2",
            stop_words="english"
        )),
        ("svd", TruncatedSVD(n_components=n_text_components, random_state=seed))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("text", text_pipe, TEXT_BASELINE),
            ("num", "passthrough", NUM_BASELINE)
        ],
        remainder="drop",
        sparse_threshold=0.0
    )
    return pre

def build_model(seed: int, n_estimators: int = 300, max_depth: Optional[int] = None,
                n_svd_components: int = 256) -> Pipeline:
    """
    RandomForest pipeline on top of the preprocessor.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with ('prep', ColumnTransformer) and ('rf', RandomForestClassifier).
    """
    pre = build_preprocessor(seed=seed, n_text_components=n_svd_components)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
        class_weight=None
    )
    return Pipeline(steps=[("prep", pre), ("rf", rf)])

# --- 7) Metrics ---------------------------------------------------------------
@dataclass
class ClassifMetrics:
    """Container for standard binary classification metrics."""
    acc: float
    prec: float
    rec: float
    f1: float

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
        Sorted by Group name for readability.
    """
    g = df_meta["Group"] if "Group" in df_meta.columns else _group_labels(df_meta)
    out = []
    # ensure stable iteration order
    for grp in sorted(g.unique().tolist()):
        idx = np.where(g.values == grp)[0]
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

# --- 8) Plotting --------------------------------------------------------------
@plot_dual_theme(section='fairness')
def _plot_margins(margins: np.ndarray, title: str, ax=None, palette=None, **kwargs):
    """
    Histogram of decision margins (p - 0.5).

    Notes
    -----
    - Uses matplotlib's default style.
    - Avoids 'palette without hue' warnings completely.
    """
    ax.hist(margins, bins=50)
    ax.set_title(title)
    ax.set_xlabel("Decision margin (p - 0.5)")
    ax.set_ylabel("Count")

# --- 9) Train + Evaluate ------------------------------------------------------
def _log_basic_outliers(df: pd.DataFrame) -> None:
    """
    Log simple 99th-percentile outliers for situational awareness (non-destructive).

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

def train_eval_reweigh(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    seed: int,
    gold: Optional[pd.DataFrame] = None,
    n_svd_components: int = 256,
    n_estimators: int = 300,
) -> Dict[str, object]:
    """
    Train a RF with reweighing weights on Train; evaluate on Val/Test; return artefacts.
    GOLD (if provided) overrides labels/groups where available, with coverage printed.

    Returns
    -------
    Dict[str, object]
        Keys: model, metrics_overall, metrics_group, disparities, preds_val,
              preds_test_full, outliers_topk, margins
    """
    # Targets and groups (default)
    y_tr_def = _prepare_binary_target(df_train)
    y_va_def = _prepare_binary_target(df_val)
    y_te_def = _prepare_binary_target(df_test)

    g_tr_def = _group_labels(df_train)
    g_va_def = _group_labels(df_val)
    g_te_def = _group_labels(df_test)

    # GOLD overrides (if provided)
    y_tr, cov_tr = maybe_override_targets(df_train, y_tr_def, gold)
    y_va, cov_va = maybe_override_targets(df_val,   y_va_def, gold)
    y_te, cov_te = maybe_override_targets(df_test,  y_te_def, gold)

    g_tr = maybe_override_groups(df_train, g_tr_def, gold)
    g_va = maybe_override_groups(df_val,   g_va_def, gold)
    g_te = maybe_override_groups(df_test,  g_te_def, gold)

    # Reweighing: compute sample weights on TRAIN
    w_tr = _reweighing_weights(y_tr, g_tr)

    # Build & fit model (align features first)
    model = build_model(seed=seed, n_estimators=n_estimators, n_svd_components=n_svd_components)
    X_train = _align_to_baseline_expected(df_train)
    X_val   = _align_to_baseline_expected(df_val)
    X_test  = _align_to_baseline_expected(df_test)

    t0 = _t0("Fitting RF with reweighing on Train ...")
    model.fit(X_train, y_tr, rf__sample_weight=w_tr)
    _tend("reweigh.fit", t0)
    print(f"[INFO] GOLD coverage (Train/Val/Test): {cov_tr:.2%} / {cov_va:.2%} / {cov_te:.2%}")

    # Predict probabilities for margins/outliers; predict labels
    t0 = _t0("Scoring probabilities on Val/Test ...")
    p_val = model.predict_proba(X_val)[:, 1]
    p_tst = model.predict_proba(X_test)[:, 1]
    _tend("reweigh.predict_proba", t0)

    ypv = (p_val >= 0.5).astype(int)
    ypt = (p_tst >= 0.5).astype(int)

    # Overall metrics
    mo_val = _overall_metrics(y_va, ypv)
    mo_tst = _overall_metrics(y_te, ypt)
    metrics_overall = pd.DataFrame([
        {"Split": "Val",  "Accuracy": mo_val.acc, "Precision": mo_val.prec, "Recall": mo_val.rec, "F1": mo_val.f1},
        {"Split": "Test", "Accuracy": mo_tst.acc, "Precision": mo_tst.prec, "Recall": mo_tst.rec, "F1": mo_tst.f1},
    ])

    # Group metrics & disparities (Test)
    df_test_groups = df_test.copy()
    df_test_groups["Group"] = g_te.values
    mg   = _group_metrics(df_test_groups, y_te, ypt)
    disp = _disparities_vs_priv(mg, privileged="White Women")

    # Full Test predictions + margins (for interpretability)
    preds_test_full = pd.DataFrame({
        "video_id": df_test["video_id"].to_numpy(),
        "title": df_test.get("title", pd.Series([""]*len(df_test))).to_numpy(),
        "Group": g_te.to_numpy(),
        "y_true": y_te,
        "y_pred": ypt,
        "prob":  np.round(p_tst, 3)
    })
    preds_test_full["margin"] = preds_test_full["prob"] - 0.5

    # Val predictions (for compatibility with downstream steps)
    preds_val = pd.DataFrame({
        "video_id": df_val["video_id"].to_numpy(),
        "y_true": y_va,
        "y_pred": ypv,
        "prob":  np.round(p_val, 3)
    })

    # Top-K outliers on Test (most confident mistakes)
    wrong = np.where(ypt != y_te)[0]
    margins_abs = np.abs(p_tst[wrong] - 0.5)
    top_idx = wrong[np.argsort(-margins_abs)[:10]]
    out_rows = []
    labels = g_te.to_numpy()
    for i in top_idx:
        out_rows.append({
            "video_id": df_test.iloc[i]["video_id"],
            "title": df_test.iloc[i].get("title", ""),
            "Group": labels[i],
            "y_true": int(y_te[i]),
            "y_pred": int(ypt[i]),
            "prob": float(np.round(p_tst[i], 3)),
            "margin_abs": float(np.round(abs(p_tst[i] - 0.5), 3))
        })
    outliers_topk = pd.DataFrame(out_rows)

    return {
        "model": model,
        "metrics_overall": metrics_overall,
        "metrics_group": mg,
        "disparities": disp,
        "preds_val": preds_val,
        "preds_test_full": preds_test_full,
        "outliers_topk": outliers_topk,
        "margins": preds_test_full["margin"].to_numpy()
    }

# --- 10) Save & Narrative -----------------------------------------------------
@plot_dual_theme(section='fairness')
def _plot_margins_wrapper(margins: np.ndarray, title: str, ax=None, palette=None, **kwargs):
    """Wrapper for theme decorator; draws histogram of margins (no seaborn)."""
    ax.hist(margins, bins=50)
    ax.set_title(title)
    ax.set_xlabel("Decision margin (p - 0.5)")
    ax.set_ylabel("Count")

def _save_all(bundle: Dict[str, object], *, is_selfcheck: bool = False) -> None:
    """
    Persist metrics, predictions, plots, and LaTeX tables for reweighing.

    Notes
    -----
    - Self-check writes *_selfcheck and does not overwrite full artefacts.
    - Writes both Step-10-prefixed files and legacy RF filename for downstream.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    NARR_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "_selfcheck" if is_selfcheck else ""

    # Step-10 friendly names
    mo = DATA_DIR / f"10_reweigh_overall_metrics{suffix}.csv"
    mg = DATA_DIR / f"10_reweigh_group_metrics{suffix}.csv"
    md = DATA_DIR / f"10_reweigh_disparities{suffix}.csv"
    pt = DATA_DIR / f"10_reweigh_predictions_test{suffix}.csv"
    pv_10 = DATA_DIR / f"10_reweigh_val_predictions{suffix}.csv"
    ot = DATA_DIR / f"10_reweigh_outliers_top10{suffix}.csv"

    # Legacy name for compatibility with later steps (kept)
    pv_legacy = DATA_DIR / "rf_reweighed_val_predictions.csv"

    bundle["metrics_overall"].to_csv(mo, index=False)
    bundle["metrics_group"].to_csv(mg, index=False)
    bundle["disparities"].to_csv(md, index=False)
    bundle["preds_test_full"].to_csv(pt, index=False)
    bundle["preds_val"].to_csv(pv_10, index=False)
    bundle["preds_val"].to_csv(pv_legacy, index=False)
    bundle["outliers_topk"].to_csv(ot, index=False)
    print(f"✓ Artefacts saved: {mo.name}, {mg.name}, {md.name}, {pt.name}, {pv_10.name} (+ {pv_legacy.name}), {ot.name}")

    # Persist model
    import joblib
    joblib.dump(bundle["model"], MODEL_DIR / "rf_reweighed.joblib")

    # Plot margins (dual theme)
    _plot_margins_wrapper(
        margins=bundle["margins"],
        title=f"Decision Margins (Test) — Pre-processing (Reweighing){' (self-check)' if is_selfcheck else ''}",
        save_path=str(FIG_DIR / f"10_reweigh_margins{suffix}"),
        figsize=(9, 6)
    )

    # LaTeX (group metrics)
    dataframe_to_latex_table(
        df=bundle["metrics_group"].set_index("Group"),
        save_path=str(TEX_DIR / f"10_reweigh_group_metrics{suffix}.tex"),
        caption="Group-wise performance under pre-processing (Kamiran–Calders Reweighing).",
        label="tab:10-reweigh-group-metrics"
    )

def _write_narrative(bundle: Dict[str,object], *, is_selfcheck: bool) -> None:
    """
    Save a short Markdown narrative summarizing performance & fairness notes.
    """
    suff = "_selfcheck" if is_selfcheck else ""
    lines = [f"# Automated Summary: Pre-Processing (Reweighing){' — self-check' if is_selfcheck else ''}\n"]
    m = bundle["metrics_overall"]
    lines.append("## Overall Metrics\n")
    lines.append(m.to_string(index=False))
    lines.append("\n## Group Metrics\n")
    lines.append(bundle["metrics_group"].to_string(index=False))
    lines.append("\n## Disparities vs. White Women\n")
    lines.append(bundle["disparities"].to_string(index=False))
    lines.append("\n## Top 5 Outliers (by confident mistakes)\n")
    lines.append(bundle["outliers_topk"].head(5).to_string(index=False))
    lines.append("\n*Note:* Some titles are non-English; tags/categories help anchor semantics.")
    path = NARR_DIR / f"10_reweigh_summary{suff}.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Narrative saved: {path.resolve()}")

# --- 11) Main ----------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Run pre-processing mitigation with reweighing, optionally using GOLD.

    Options
    -------
    --selfcheck           Use a random sample and internal split (safe).
    --sample INT          Random sample size for self-check (default: min(80k, N)).
    --seed INT            Seed (from config by default; do not use 42).
    --use-gold            Use GOLD labels/groups where available.
    --gold-path STR       Path to GOLD file (default: outputs/data/gold/gold_final.csv).

    Notes
    -----
    - Upstream tasks are multi-label; totals there can exceed N. Here we evaluate
      a *single* binary target per ID, so counts sum to N.
    """
    t_all = time.perf_counter()
    print("--- Starting Step 10: Pre-processing Mitigation (Reweighing) ---")

    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=None,
                   help="Sample N rows (works with or without --selfcheck; writes to canonical paths)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--use-gold", action="store_true", help="Use GOLD labels/groups where available.")
    p.add_argument("--gold-path", type=str, default=str(OUTPUT_DIR / "data" / "gold" / "gold_final.csv"))
    p.add_argument("--n-svd-components", type=int, default=256,
                   help="TruncatedSVD components (reduce to 64 for fast runs)")
    p.add_argument("--n-estimators", type=int, default=500)
    args = p.parse_args(argv)

    gold = load_gold_table(Path(args.gold_path)) if args.use_gold else None

    # Load corpus
    df = _load_corpus()
    total = len(df)
    print(f"[STATS] Total videos available: {total:,}")

    # Self-check note for interpretability
    print("[NOTE] Titles may be non-English; tags/categories help anchor semantics.")
    print("[NOTE] Logging basic outliers for situational awareness (non-destructive).")
    _log_basic_outliers(df)

    # Build splits
    if args.selfcheck:
        n = args.sample or min(80_000, total)
        df = df.sample(n=n, random_state=args.seed, replace=False).reset_index(drop=True)
        print(f"[SELF-CHECK] Random sample drawn: {len(df):,} rows (seed={args.seed}).")
        train_df, val_df, test_df = _make_internal_split(df, seed=args.seed)
    else:
        if args.sample:
            df = df.sample(n=min(args.sample, total), random_state=args.seed, replace=False).reset_index(drop=True)
            print(f"[SAMPLE] Using {len(df):,} rows (seed={args.seed}). Canonical artefacts will reflect this sample.")
        tr_ids, va_ids, te_ids = _read_ids_or_none(TRAIN_IDS), _read_ids_or_none(VAL_IDS), _read_ids_or_none(TEST_IDS)
        if not (tr_ids is not None and va_ids is not None and te_ids is not None):
            print("✗ Step-06 IDs not found; falling back to internal split on full corpus.")
            train_df, val_df, test_df = _make_internal_split(df, seed=args.seed)
        else:
            train_df, val_df, test_df = _subset_by_ids(df, tr_ids), _subset_by_ids(df, va_ids), _subset_by_ids(df, te_ids)

    print(f"Split sizes: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    # Train & evaluate
    bundle = train_eval_reweigh(
        df_train=train_df, df_val=val_df, df_test=test_df,
        seed=args.seed, gold=gold,
        n_svd_components=getattr(args, 'n_svd_components', 256),
        n_estimators=getattr(args, 'n_estimators', 500),
    )
    _save_all(bundle, is_selfcheck=args.selfcheck)

    # Console highlights (qualitative)
    print("\n=== Overall Metrics (Val/Test: Accuracy / Precision / Recall / F1) ===")
    print(bundle["metrics_overall"].to_string(index=False))
    print("\n=== Group Metrics (Test) ===")
    print(bundle["metrics_group"].to_string(index=False))
    print("\n=== Disparities vs. White Women (privileged) ===")
    print(bundle["disparities"].to_string(index=False))
    print("\n=== Top 10 outliers (Test) — most confident mistakes ===")
    print(bundle["outliers_topk"].to_string(index=False))
    print("\nNote: Outliers intentionally include title + Group for interpretability; "
          "some titles may be non-English; use tags/categories for context.")

    _write_narrative(bundle, is_selfcheck=args.selfcheck)
    _tend("reweigh.step10_total", t_all)
    print("\n--- Step 10: Pre-processing Mitigation Completed Successfully ---")

if __name__ == '__main__':
    main()
