# -*- coding: utf-8 -*-
"""
07_rf_baseline.py
=================

Purpose
-------
Train and evaluate a Random Forest baseline for the project's canonical binary
task (default: 'Amateur' vs not) using text features (HashingVectorizer → SVD)
and light numeric features. Outputs remain fully compatible with downstream
evaluation and mitigation steps. Adds optional GOLD evaluation.

What it does
------------
1) Loads canonical parquet + Step-06 splits (or performs an internal split in
   self-check mode), enforcing required columns and deriving the binary target.
2) Builds a scikit-learn Pipeline:
      text(model_input_text) → HashingVectorizer → TruncatedSVD (dense)
      + numeric passthrough  → RandomForestClassifier
3) Evaluates overall and intersectional group metrics; prints top-10 confident
   mistakes; saves metrics, predictions, plots (margins), model, and a narrative.
4) If GOLD annotations exist, evaluates the fitted model on GOLD (non-destructive).
5) Self-check uses a random sample and writes *_selfcheck artefacts only.

Interpretability note
---------------------
Some titles are non-English (MPU). Tags/categories often anchor semantics. We
highlight confident mistakes for qualitative inspection.

CLI
---
# Full run (canonical artefacts):
python -m src.models.07_rf_baseline

# Self-check (safe; never overwrites full artefacts):
python -m src.models.07_rf_baseline --selfcheck --sample 120000
"""

from __future__ import annotations

# --- Imports (keep at top) ----------------------------------------------------
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Project utils
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.fairness.fairness_evaluation_utils import (
    overall_metrics,
    calculate_group_metrics,
    calculate_fairness_disparities,
    top_confident_outliers,
    group_labels_intersectional,
    load_gold_table,
    align_gold_to_frame,
    evaluate_against_gold,
)

# --- 1) Config & paths --------------------------------------------------------
CONFIG = load_config()
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))

DATA_DIR    = Path(CONFIG["paths"]["data"])
OUT_DIR     = Path(CONFIG["paths"]["outputs"])
FIG_DIR     = Path(CONFIG["paths"]["figures"])
FIG_MODELS_DIR = FIG_DIR / "models"
MODEL_DIR   = OUT_DIR / "models"
NARR_DIR    = Path(CONFIG["paths"]["narratives"]) / "automated"

# Inputs (standardised)
CORPUS_PATH = DATA_DIR / "01_ml_corpus.parquet"
TRAIN_IDS   = DATA_DIR / "06_train_ids.csv"
VAL_IDS     = DATA_DIR / "06_val_ids.csv"
TEST_IDS    = DATA_DIR / "06_test_ids.csv"

# Columns (config-driven with fallbacks)
COLCFG     = CONFIG.get("columns", {})
TEXT_COL   = COLCFG.get("text", "model_input_text")
CATS_COL   = COLCFG.get("categories", "categories")
NUM_COLS   = COLCFG.get("numeric_rf", ["duration", "ratings"])  # keep minimal, fast
POSITIVE_CLASS = CONFIG.get("task", {}).get("positive_class", "Amateur")

# Outputs (standardised 07_*)
VAL_METRICS_CSV       = DATA_DIR / "07_rf_val_metrics.csv"
VAL_PREDICTIONS_CSV   = DATA_DIR / "07_rf_val_predictions.csv"
TEST_PREDICTIONS_CSV  = DATA_DIR / "07_rf_test_predictions.csv"
OVERALL_METRICS_CSV   = DATA_DIR / "07_overall_metrics.csv"
GROUP_METRICS_CSV     = DATA_DIR / "07_fairness_group_metrics.csv"
DISPARITIES_CSV       = DATA_DIR / "07_fairness_disparities.csv"
FEATURE_IMP_CSV       = DATA_DIR / "07_rf_feature_importance.csv"
MODEL_PATH            = MODEL_DIR / "07_rf.joblib"

# Optional GOLD (keeps numbering)
GOLD_METRICS_CSV     = DATA_DIR / "07_rf_gold_eval_metrics.csv"
GOLD_PREDICTIONS_CSV = DATA_DIR / "07_rf_gold_eval_predictions.csv"
GOLD_PATH            = DATA_DIR / "gold/gold_final.csv"  # if missing, GOLD eval is skipped gracefully


# --- 2) Lightweight timers ----------------------------------------------------
def _t0(msg: str) -> float:
    """
    Start a timer and print a standardized header.

    Parameters
    ----------
    msg : str
        Human-readable message for the step start.

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
        Short label, e.g. 'rf.train'.
    t0 : float
        Start time from _t0().
    """
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")


# --- 3) Data helpers ----------------------------------------------------------
def _coerce_types_and_round(d: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce numeric conventions and rounding, aligned to source API semantics.

    - 'ratings' : integer (#votes)
    - 'rating'  : float grade (rounded to 1 decimal; not a model feature here)
    - 'views'   : integer (if present)
    - 'duration': seconds as numeric; string 'MM:SS'/'HH:MM:SS' coerced to seconds
    - 'year'    : integer (if present)

    Notes
    -----
    The Redtube-like `searchVideos` API uses:
      * ratings = integer (vote count)
      * rating  = float (grade)
    We respect that here, ensuring clean dtypes for interpretability.
    """
    out = d.copy()

    if "ratings" in out.columns:
        out["ratings"] = pd.to_numeric(out["ratings"], errors="coerce").round(0).astype("Int64")
    if "rating" in out.columns:
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce").round(1)
    if "views" in out.columns:
        out["views"] = pd.to_numeric(out["views"], errors="coerce").round(0).astype("Int64")

    if "duration" in out.columns and out["duration"].dtype == "object":
        def _to_sec(x):
            s = str(x)
            if s.isdigit():
                return float(s)
            parts = s.split(":")
            try:
                if len(parts) == 2:
                    m, sec = int(parts[0]), float(parts[1])
                    return m * 60 + sec
                if len(parts) == 3:
                    h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
                    return h * 3600 + m * 60 + sec
            except Exception:
                return np.nan
            return np.nan
        out["duration"] = out["duration"].map(_to_sec)
    if "duration" in out.columns:
        out["duration"] = pd.to_numeric(out["duration"], errors="coerce")

    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").round(0).astype("Int64")

    return out


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist and are correctly typed.

    - If TEXT_COL missing, fallback to: title + tags (handles multilingual titles).
    - Ensure NUM_COLS exist; missing ones are filled with 0 (numeric).
    - Categories column exists for target derivation.
    """
    d = df.copy()
    if TEXT_COL not in d.columns:
        d[TEXT_COL] = (d.get("title", "").fillna("").astype(str) + " " +
                       d.get("tags", "").fillna("").astype(str))

    for c in NUM_COLS:
        if c not in d.columns:
            d[c] = 0

    if CATS_COL not in d.columns:
        d[CATS_COL] = ""

    # Coerce types and rounding per conventions
    d = _coerce_types_and_round(d)
    return d


def _load_corpus() -> pd.DataFrame:
    """
    Load canonical parquet and enforce required columns + types.

    Returns
    -------
    pd.DataFrame
        Cleaned corpus ready for splitting/modeling.
    """
    t0 = _t0(f"[READ] Parquet: {CORPUS_PATH}")
    df = pd.read_parquet(CORPUS_PATH)
    _tend("rf.load_corpus", t0)
    return _ensure_columns(df)


def _read_ids_or_none(path: Path) -> Optional[pd.Index]:
    """
    Read a single-column CSV of video IDs; return Index or None.

    Parameters
    ----------
    path : Path
        Path to CSV with a 'video_id' column.

    Returns
    -------
    Optional[pd.Index]
        Index of IDs or None if path does not exist.
    """
    if path.exists():
        ids = pd.read_csv(path)["video_id"]
        return pd.Index(ids)
    return None


def _subset_by_ids(df: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    """
    Subset dataframe by IDs, preserving dataframe order.

    Parameters
    ----------
    df : pd.DataFrame
        Full corpus.
    ids : pd.Index
        IDs to keep.

    Returns
    -------
    pd.DataFrame
        Subset.
    """
    return df[df["video_id"].isin(ids)].reset_index(drop=True)


def _make_internal_split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a 60/20/20 internal split for self-check, stratified on intersectional key.

    Parameters
    ----------
    df : pd.DataFrame
        Clean corpus.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of pd.DataFrame
    """
    def stratify_key(d: pd.DataFrame) -> pd.Series:
        key = np.full(len(d), "Other", dtype=object)
        gf = (d.get("gender_female", 0) == 1)
        bw = (d.get("race_ethnicity_black", 0) == 1) & gf
        ww = (d.get("race_ethnicity_white", 0) == 1) & gf & ~bw
        aw = (d.get("race_ethnicity_asian", 0) == 1) & gf & ~bw & ~ww
        lw = (d.get("race_ethnicity_latina", 0) == 1) & gf & ~bw & ~ww & ~aw
        key[bw] = "Black_Women"; key[ww] = "White_Women"; key[aw] = "Asian_Women"; key[lw] = "Latina_Women"
        return pd.Series(key, index=d.index, name="stratify_key")

    d = df.copy()
    d["stratify_key"] = stratify_key(d)
    trv, te = train_test_split(d, test_size=0.20, random_state=seed, stratify=d["stratify_key"])
    tr, va = train_test_split(trv, test_size=0.25, random_state=seed, stratify=trv["stratify_key"])
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def _prepare_binary_target(df: pd.DataFrame, positive_class: str = POSITIVE_CLASS) -> pd.Series:
    """
    Binary target from first listed category; 1 if equals positive_class else 0.

    Parameters
    ----------
    df : pd.DataFrame
        Frame with categories column.
    positive_class : str
        Positive class label (default from config: 'Amateur').

    Returns
    -------
    pd.Series of dtype int
        Binary target.
    """
    primary = df[CATS_COL].fillna("").astype(str).str.split(",").str[0].str.strip()
    return (primary == positive_class).astype(int)


# --- 4) Features & model ------------------------------------------------------
def build_preprocessor(seed: int,
                       n_text_components: int = 256,
                       n_hash_features: int = 2**18) -> ColumnTransformer:
    """
    Build a ColumnTransformer:
      - text -> HashingVectorizer (CSR) -> TruncatedSVD (dense n_components)
      - numeric -> passthrough

    Parameters
    ----------
    seed : int
        Random seed for SVD reproducibility.
    n_text_components : int
        Number of SVD components (dense output).
    n_hash_features : int
        HashingVectorizer dimensionality.

    Returns
    -------
    ColumnTransformer
        Preprocessing transformer producing a dense feature matrix.
    """
    text_pipe = Pipeline(steps=[
        ("hash", HashingVectorizer(
            n_features=n_hash_features,
            alternate_sign=False,
            ngram_range=(1, 2),
            norm="l2",
            stop_words="english",
        )),
        ("svd", TruncatedSVD(n_components=n_text_components, random_state=seed)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("text", text_pipe, TEXT_COL),
            ("num", "passthrough", NUM_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre


def make_rf_baseline(seed: int,
                     *,
                     n_estimators: int = 500,
                     max_depth: Optional[int] = None,
                     n_svd_components: int = 256) -> Pipeline:
    """
    Create the baseline RF pipeline: preprocessor + RandomForest.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_estimators : int, default 500
        Number of trees.
    max_depth : Optional[int]
        Limit tree depth for speed/regularization.
    n_svd_components : int, default 256
        Number of TruncatedSVD components (reduce for faster runs).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline(preprocessor → RandomForestClassifier)
    """
    pre = build_preprocessor(seed=seed, n_text_components=n_svd_components)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline(steps=[("prep", pre), ("rf", rf)])


def _extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    """
    Build a feature-importance table combining SVD components and numeric columns.

    Parameters
    ----------
    model : Pipeline
        Fitted Pipeline(prep→rf).

    Returns
    -------
    pd.DataFrame
        Columns: ['feature','importance'] sorted desc by importance.
    """
    rf = model.named_steps["rf"]
    importances = rf.feature_importances_
    svd = model.named_steps["prep"].named_transformers_["text"].named_steps["svd"]
    n_svd = int(getattr(svd, "n_components", len(importances) - len(NUM_COLS)))
    names = [f"svd_comp_{i+1}" for i in range(n_svd)] + NUM_COLS
    return pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)


# --- 5) Plotting (dual theme) -------------------------------------------------
@plot_dual_theme(section="fairness")
def _plot_margins(margins: np.ndarray, title: str, ax=None, palette=None, **kwargs):
    """
    Histogram of decision margins (p - 0.5). Uses project-wide dual theme.

    Parameters
    ----------
    margins : np.ndarray
        p - 0.5 for the positive class probabilities.
    title : str
        Plot title.
    ax, palette, **kwargs : injected by decorator
    """
    ax.hist(margins, bins=50)
    ax.set_title(title)
    ax.set_xlabel("Decision margin (p - 0.5)")
    ax.set_ylabel("Count")


# --- 6) Train + evaluate ------------------------------------------------------
def _prob(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Get positive-class probabilities from a scikit-learn pipeline.

    Parameters
    ----------
    estimator : Pipeline
        Fitted estimator.
    X : pd.DataFrame
        Feature frame with TEXT_COL and NUM_COLS.

    Returns
    -------
    np.ndarray
        Probabilities for positive class.
    """
    return estimator.predict_proba(X)[:, 1]


def train_eval_rf(df_train: pd.DataFrame,
                  df_val: pd.DataFrame,
                  df_test: pd.DataFrame,
                  *,
                  n_estimators: int = 500,
                  max_depth: Optional[int] = None,
                  n_svd_components: int = 256) -> Dict[str, object]:
    """
    Train the RF baseline and evaluate on val/test; return artefacts for saving.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        Splits with columns TEXT_COL, NUM_COLS, CATS_COL, and one-hots for groups.
    n_estimators : int
        Number of trees for RF.
    max_depth : Optional[int]
        Maximum tree depth.

    Returns
    -------
    dict
        {
          'model': Pipeline,
          'val_metrics': pd.DataFrame,
          'test_metrics': pd.DataFrame,
          'val_preds': pd.DataFrame,
          'test_preds': pd.DataFrame,
          'group_metrics_test': pd.DataFrame,
          'disparities_test': pd.DataFrame,
          'outliers_test': pd.DataFrame,
        }
    """
    # Targets
    y_tr = _prepare_binary_target(df_train).to_numpy()
    y_va = _prepare_binary_target(df_val).to_numpy()
    y_te = _prepare_binary_target(df_test).to_numpy()

    # Estimator
    model = make_rf_baseline(SEED, n_estimators=n_estimators, max_depth=max_depth,
                             n_svd_components=n_svd_components)

    # Fit
    t0 = _t0("Training RF baseline ...")
    model.fit(df_train[[TEXT_COL] + NUM_COLS], y_tr)
    _tend("rf.train", t0)

    # Predict proba
    t0 = _t0("Scoring probabilities on Val/Test ...")
    p_val = _prob(model, df_val[[TEXT_COL] + NUM_COLS])
    p_tst = _prob(model, df_test[[TEXT_COL] + NUM_COLS])
    _tend("rf.predict", t0)

    ypv = (p_val >= 0.5).astype(int)
    ypt = (p_tst >= 0.5).astype(int)

    # Overall metrics
    mo_val = overall_metrics(y_va, ypv)
    mo_tst = overall_metrics(y_te, ypt)
    val_metrics  = pd.DataFrame([{"Split": "Val",  "Accuracy": round(mo_val.acc, 3), "Precision": round(mo_val.prec, 3), "Recall": round(mo_val.rec, 3), "F1": round(mo_val.f1, 3)}])
    test_metrics = pd.DataFrame([{"Split": "Test", "Accuracy": round(mo_tst.acc, 3), "Precision": round(mo_tst.prec, 3), "Recall": round(mo_tst.rec, 3), "F1": round(mo_tst.f1, 3)}])

    # Predictions frames (rounded probs to 3 decimals; margins derived)
    val_preds = pd.DataFrame({
        "video_id": df_val["video_id"].to_numpy(),
        "title": df_val.get("title", pd.Series([""]*len(df_val))).to_numpy(),
        "Group": group_labels_intersectional(df_val).to_numpy(),
        "y_true": y_va,
        "y_pred": ypv,
        "prob": np.round(p_val, 3),
    })
    val_preds["margin"] = val_preds["prob"] - 0.5

    test_preds = pd.DataFrame({
        "video_id": df_test["video_id"].to_numpy(),
        "title": df_test.get("title", pd.Series([""]*len(df_test))).to_numpy(),
        "Group": group_labels_intersectional(df_test).to_numpy(),
        "y_true": y_te,
        "y_pred": ypt,
        "prob": np.round(p_tst, 3),
    })
    test_preds["margin"] = test_preds["prob"] - 0.5

    # Group metrics & disparities (Test). Multi-label groups can overlap → sums > N.
    gm_test = calculate_group_metrics(test_preds)
    disp = calculate_fairness_disparities(gm_test, privileged="White Women")

    # Outliers
    outliers = top_confident_outliers(test_preds, probs=p_tst, k=10)

    return {
        "model": model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "group_metrics_test": gm_test,
        "disparities_test": disp,
        "outliers_test": outliers,
    }


# --- 7) Save + narrative ------------------------------------------------------
@plot_dual_theme(section="fairness")
def _plot_dummy_for_decorator(ax=None, **kwargs):
    """No-op plot to satisfy decorator when needed (not used directly)."""
    pass


def _save_all(bundle: Dict[str, object], *, selfcheck: bool = False) -> None:
    """
    Persist metrics, predictions, model, plot, and narrative.

    - Always writes standard 07_* artefacts (val/test metrics & predictions,
      group metrics, disparities, feature importance, margins plot, model, narrative).
    - Also writes **compatibility Step-08-style** artefacts needed by downstream
      steps: 08_rf_overall_metrics*.csv, 08_rf_group_metrics*.csv,
      08_rf_disparities*.csv, 08_rf_predictions_test*.csv, and outliers CSV.

    Self-check writes *_selfcheck files ONLY (non-destructive).
    """
    t0 = _t0("Saving RF artefacts ...")
    for p in [DATA_DIR, FIG_DIR, FIG_MODELS_DIR, MODEL_DIR, NARR_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    suffix = "_selfcheck" if selfcheck else ""

    # --- Overall metrics (Val + Test) → 07_overall_metrics*.csv
    overall_df = pd.concat([bundle["val_metrics"], bundle["test_metrics"]], ignore_index=True)
    overall_path = OVERALL_METRICS_CSV if not selfcheck else DATA_DIR / f"07_overall_metrics{suffix}.csv"
    overall_df.to_csv(overall_path, index=False)

    # --- Val/Test artefacts (07_*)
    val_metrics_csv = VAL_METRICS_CSV if not selfcheck else DATA_DIR / f"07_rf_val_metrics{suffix}.csv"
    val_preds_csv   = VAL_PREDICTIONS_CSV if not selfcheck else DATA_DIR / f"07_rf_val_predictions{suffix}.csv"
    test_preds_csv  = TEST_PREDICTIONS_CSV if not selfcheck else DATA_DIR / f"07_rf_test_predictions{suffix}.csv"
    bundle["val_metrics"].to_csv(val_metrics_csv, index=False)
    bundle["val_preds"].to_csv(val_preds_csv, index=False)
    bundle["test_preds"].to_csv(test_preds_csv, index=False)

    # --- Group metrics & disparities (07_*)
    gm_csv   = GROUP_METRICS_CSV if not selfcheck else DATA_DIR / f"07_fairness_group_metrics{suffix}.csv"
    disp_csv = DISPARITIES_CSV   if not selfcheck else DATA_DIR / f"07_fairness_disparities{suffix}.csv"
    bundle["group_metrics_test"].to_csv(gm_csv, index=False)
    bundle["disparities_test"].to_csv(disp_csv, index=False)

    # --- Plot margins (dual theme)
    _plot_margins(
        margins=bundle["test_preds"]["margin"].to_numpy(),
        title="RF Baseline — Decision Margins (Test)",
        save_path=str(FIG_MODELS_DIR / f"07_margins_baseline{suffix}"),
        figsize=(9, 6),
    )

    # --- Feature importance (component-level)
    fi = _extract_feature_importance(bundle["model"])
    fi_csv = FEATURE_IMP_CSV if not selfcheck else DATA_DIR / f"07_rf_feature_importance{suffix}.csv"
    fi.to_csv(fi_csv, index=False)

    # --- Save model (both numbered and alias)
    model_path = MODEL_PATH if not selfcheck else MODEL_DIR / f"07_rf{suffix}.joblib"
    baseline_model_path = MODEL_DIR / ("rf_baseline.joblib" if not selfcheck else f"rf_baseline{suffix}.joblib")
    joblib.dump(bundle["model"], model_path)
    joblib.dump(bundle["model"], baseline_model_path)
    print(f"✓ Model saved: {model_path.resolve()} and {baseline_model_path.resolve()}")

    # --- Narrative (rounded tables for legibility)
    lines = []
    lines.append(f"# Automated Summary: RF Baseline{' — self-check' if selfcheck else ''}\n")
    lines.append("Some titles are non-English; tags/categories help anchor semantics (MPU).")
    lines.append("\n## Overall Metrics\n")
    lines.append(overall_df.round(3).to_string(index=False))
    lines.append("\n## Group Metrics (Test)\n")
    lines.append(bundle["group_metrics_test"].round(3).to_string(index=False))
    lines.append("\n## Disparities vs. White Women\n")
    lines.append(bundle["disparities_test"].round(3).to_string(index=False))
    lines.append("\n## Top 10 Outliers (Test) — most confident mistakes\n")
    lines.append(bundle["outliers_test"].to_string(index=False))
    narr_path = NARR_DIR / f"07_rf_baseline_summary{suffix}.md"
    with open(narr_path, "w") as f:
        f.write("\n".join(lines))

    # --- NEW: Compatibility artefacts for later steps (Step-08-style names)
    comp_overall = DATA_DIR / f"08_rf_overall_metrics{suffix}.csv"
    comp_group   = DATA_DIR / f"08_rf_group_metrics{suffix}.csv"
    comp_disp    = DATA_DIR / f"08_rf_disparities{suffix}.csv"
    comp_preds   = DATA_DIR / f"08_rf_predictions_test{suffix}.csv"

    overall_df.to_csv(comp_overall, index=False)
    bundle["group_metrics_test"].to_csv(comp_group, index=False)
    bundle["disparities_test"].to_csv(comp_disp, index=False)
    bundle["test_preds"].to_csv(comp_preds, index=False)

    # Outliers (save both 07_* and 08_* names for discovery)
    out_csv_07 = DATA_DIR / f"07_rf_outliers_top10{suffix}.csv"
    out_csv_08 = DATA_DIR / f"08_rf_outliers_top10{suffix}.csv"
    bundle["outliers_test"].to_csv(out_csv_07, index=False)
    bundle["outliers_test"].to_csv(out_csv_08, index=False)

    print("✓ Artefacts saved:",
          overall_path.name, ",",
          val_metrics_csv.name, ",", val_preds_csv.name, ",", test_preds_csv.name, ",",
          gm_csv.name, ",", disp_csv.name, ",",
          fi_csv.name, ",",
          narr_path.name, ",",
          comp_overall.name, ",", comp_group.name, ",", comp_disp.name, ",", comp_preds.name, ",",
          out_csv_07.name, ",", out_csv_08.name)
    _tend("rf.save_all", t0)


def _maybe_eval_gold(model: Pipeline, df_all: pd.DataFrame) -> None:
    """
    If a GOLD table exists, evaluate the fitted model on that subset and save
    07_rf_gold_eval_metrics.csv and 07_rf_gold_eval_predictions.csv.

    Parameters
    ----------
    model : Pipeline
        Fitted baseline.
    df_all : pd.DataFrame
        Corpus (concatenated train/val/test for feature access).
    """
    gold = load_gold_table(GOLD_PATH)
    if gold is None:
        return
    joined = align_gold_to_frame(df_all, gold)
    if joined.empty:
        print("✗ GOLD join produced 0 rows; skipping GOLD evaluation.")
        return

    t0 = _t0("Scoring probabilities on GOLD subset (RF) ...")
    probs = _prob(model, joined[[TEXT_COL] + NUM_COLS])
    _tend("rf.gold_predict", t0)

    metrics_df, preds_df = evaluate_against_gold(probs, joined)
    metrics_df.to_csv(GOLD_METRICS_CSV, index=False)
    preds_df.to_csv(GOLD_PREDICTIONS_CSV, index=False)
    print(f"✓ GOLD artefacts saved: {GOLD_METRICS_CSV.name}, {GOLD_PREDICTIONS_CSV.name}")


# --- 8) Main ------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Run the RF baseline full pipeline with optional self-check.

    Options
    -------
    --selfcheck             Use a random sample and internal split (safe).
    --sample INT            Random sample size for self-check (default: min(150k, N)).
    --n-estimators INT      Number of trees (default: 500).
    --max-depth INT         Optional limit on tree depth.

    Notes
    -----
    - Totals in multi-label analyses elsewhere can exceed N; here we evaluate a
      *single* binary target per ID, so counts sum to N. Group aggregates can
      nonetheless overlap across protected axes.
    - Non-English titles may appear among outliers; tags/categories provide anchors.
    """
    # argparse imported at top

    t_all = time.perf_counter()
    print("--- Starting Step 07: RF Baseline ---")
    print("[NOTE] Titles may be non-English; tags/categories help anchor semantics (multi-label upstream; single-label target here).")

    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=None,
                   help="Sample N rows from corpus (works with or without --selfcheck; writes to canonical paths)")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--n-svd-components", type=int, default=256,
                   help="Number of TruncatedSVD components (reduce to 64 for fast dev runs)")
    args = p.parse_args(argv)

    # Load corpus
    df = _load_corpus()
    total = len(df)
    print(f"[STATS] Total videos available: {total:,}")

    # Build splits
    if args.selfcheck:
        n = args.sample or min(150_000, total)
        df = df.sample(n=n, random_state=SEED, replace=False).reset_index(drop=True)
        print(f"[SELF-CHECK] Random sample drawn: {len(df):,} rows (seed={SEED}).")
        train_df, val_df, test_df = _make_internal_split(df, seed=SEED)
    else:
        if args.sample:
            df = df.sample(n=min(args.sample, total), random_state=SEED, replace=False).reset_index(drop=True)
            print(f"[SAMPLE] Using {len(df):,} rows (seed={SEED}). Canonical artefacts will reflect this sample.")

        tr_ids, va_ids, te_ids = _read_ids_or_none(TRAIN_IDS), _read_ids_or_none(VAL_IDS), _read_ids_or_none(TEST_IDS)
        if not (tr_ids is not None and va_ids is not None and te_ids is not None):
            print("✗ Step-06 IDs not found; falling back to internal split on full corpus.")
            train_df, val_df, test_df = _make_internal_split(df, seed=SEED)
        else:
            train_df = _subset_by_ids(df, tr_ids)
            val_df   = _subset_by_ids(df, va_ids)
            test_df  = _subset_by_ids(df, te_ids)

    print(f"Split sizes: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    # Train & evaluate
    bundle = train_eval_rf(
        df_train=train_df,
        df_val=val_df,
        df_test=test_df,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_svd_components=args.n_svd_components,
    )

    # Save artefacts
    _save_all(bundle, selfcheck=args.selfcheck)

    # Print interpretable highlights
    print("\n=== Overall Metrics (Accuracy / Precision / Recall / F1) ===")
    print(pd.concat([bundle['val_metrics'], bundle['test_metrics']], ignore_index=True).to_string(index=False))
    print("\n=== Group Metrics (Test) ===")
    print(bundle["group_metrics_test"].round(3).to_string(index=False))
    print("\n=== Disparities vs. White Women (privileged) ===")
    print(bundle["disparities_test"].round(3).to_string(index=False))
    print("\n=== Top 10 Outliers (Test) — most confident mistakes ===")
    print(bundle["outliers_test"].to_string(index=False))

    # Optional GOLD evaluation (only in full run)
    if not args.selfcheck:
        _maybe_eval_gold(bundle["model"], pd.concat([train_df, val_df, test_df], ignore_index=True))

    _tend("rf.step07_total", t_all)
    print("\n--- Step 07: RF Baseline Completed Successfully ---")


if __name__ == "__main__":
    main()
