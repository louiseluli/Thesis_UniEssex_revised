# -*- coding: utf-8 -*-
"""
11_inprocessing_mitigation.py
=============================

Purpose
-------
In-processing bias mitigation with Fairlearn's Exponentiated Gradient (EG) under
Demographic Parity (DP). We train a binary classifier ("Amateur" vs. not) on
hashing+SVD text features plus light numeric columns. Sensitive groups are
intersectional (Black/White/Asian/Latina Women; else Other) and are not given
to the model—only to the fairness constraint.

What it does
------------
1) Loads canonical parquet + Step-06 splits (or internal split for self-check).
2) Builds a scikit-learn Pipeline: text → HashingVectorizer → TruncatedSVD (dense),
   numeric passthrough; classifier = LogisticRegression (lbfgs, L2).
3) Wraps the pipeline with Fairlearn ExponentiatedGradient under Demographic Parity.
4) Evaluates overall and by group; computes disparities vs White Women (privileged).
5) Prints top-10 outliers (most confident mistakes), saves dual-theme plots, LaTeX,
   CSVs, and a narrative. Includes lightweight timers and total runtime.

Reproducibility & config
------------------------
- Reads seed, paths, and column names from your config (seed=95).
- Keeps imports at the top. Years remain ints; ratings show with 1 decimal when logged.
- Notes that non-English titles occur; tags/categories help anchor semantics.

CLI
---
# Full run (writes 11_* artefacts + legacy egdp_* for compatibility):
python3 src/fairness/11_inprocessing_mitigation.py --use-gold

# Self-check (random sample; non-destructive *_selfcheck artefacts):
python3 src/fairness/11_inprocessing_mitigation.py --selfcheck --sample 80000 --use-gold

# Options:
python3 src/fairness/11_inprocessing_mitigation.py \
  --eps 0.02 --svd-components 256 --hash-features 262144 --max-iter 1500 --skip-save-model
"""

from __future__ import annotations

# --- Imports (keep at top) ---------------------------------------------------
import sys
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Project utils
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.utils.academic_tables import dataframe_to_latex_table
from src.fairness.fairness_evaluation_utils import (
    load_gold_table,
    maybe_override_groups,
    maybe_override_targets,
)

# --- 1) Config & Paths -------------------------------------------------------
CONFIG = load_config()

DATA_DIR   = Path(CONFIG["paths"]["data"])
FIG_DIR    = Path(CONFIG["paths"]["figures"]) / "eval"
MODEL_DIR  = Path(CONFIG["paths"].get("models", Path(CONFIG["project"]["root"]) / "outputs" / "models"))
TEX_DIR    = Path(CONFIG["project"]["root"]) / "dissertation" / "auto_tables"
NARR_DIR   = Path(CONFIG["paths"]["narratives"]) / "automated"
OUTPUT_DIR = Path(CONFIG["paths"]["outputs"])

# Align to Steps 09–10 names
CORPUS_PATH = DATA_DIR / "01_ml_corpus.parquet"
TRAIN_IDS   = DATA_DIR / "06_train_ids.csv"
VAL_IDS     = DATA_DIR / "06_val_ids.csv"
TEST_IDS    = DATA_DIR / "06_test_ids.csv"

SEED     = int(CONFIG.get("reproducibility", {}).get("seed", 95))
TEXT_COL = "model_input_text"       # from earlier steps
NUM_COLS = ["duration", "ratings"]  # light numeric features (small & safe)
CATS_COL = "categories"             # derive binary target

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
    Ensure required columns exist. If TEXT_COL is missing, build via title+tags;
    ensure NUM_COLS exist; ensure CATS_COL exists.

    Returns
    -------
    pd.DataFrame
        Safe copy with minimum required columns present.
    """
    d = df.copy()
    if TEXT_COL not in d.columns:
        d[TEXT_COL] = (d.get("title", "").fillna("").astype(str) + " " +
                       d.get("tags",  "").fillna("").astype(str))
    for c in NUM_COLS:
        if c not in d.columns:
            d[c] = 0
    if CATS_COL not in d.columns:
        d[CATS_COL] = ""
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
    _tend("inproc.load_corpus", t0)
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
        ids = pd.read_csv(path)["video_id"]
        return pd.Index(ids)
    return None

def _subset_by_ids(df: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    """
    Subset dataframe by IDs, preserving original row order.

    Parameters
    ----------
    df : pd.DataFrame
    ids : pd.Index

    Returns
    -------
    pd.DataFrame
    """
    return df[df["video_id"].isin(ids)].reset_index(drop=True)

def _make_internal_split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a 60/20/20 internal split for self-check, stratified on female intersections.

    Returns
    -------
    (train, val, test) : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
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

# --- 4) Task & groups ---------------------------------------------------------
def _prepare_binary_target(df: pd.DataFrame, positive_class: str = "Amateur") -> pd.Series:
    """
    Build the binary target: 1 if primary category == positive_class, else 0.

    Parameters
    ----------
    df : pd.DataFrame
    positive_class : str

    Returns
    -------
    pd.Series
        0/1 labels.
    """
    primary = df[CATS_COL].fillna("").astype(str).str.split(",").str[0].str.strip()
    return (primary == positive_class).astype(int)

def _group_labels(df: pd.DataFrame) -> pd.Series:
    """
    Intersectional group labels for fairness diagnostics (not fed to the model).

    Returns
    -------
    pd.Series
        Values in {'Black Women', 'White Women', 'Asian Women', 'Latina Women', 'Other'}.
    """
    labels = np.full(len(df), "Other", dtype=object)
    gf = (df.get("gender_female", 0) == 1)
    bw = (df.get("race_ethnicity_black", 0) == 1) & gf
    ww = (df.get("race_ethnicity_white", 0) == 1) & gf & ~bw
    aw = (df.get("race_ethnicity_asian", 0) == 1) & gf & ~bw & ~ww
    lw = (df.get("race_ethnicity_latina", 0) == 1) & gf & ~bw & ~ww & ~aw
    labels[bw] = "Black Women"; labels[ww] = "White Women"; labels[aw] = "Asian Women"; labels[lw] = "Latina Women"
    return pd.Series(labels, index=df.index, name="Group")

# --- 5) Features & model ------------------------------------------------------
def build_preprocessor(seed: int, n_text_components: int = 256, n_hash_features: int = 2**18) -> ColumnTransformer:
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

    # scale numeric features for faster/cleaner LogReg convergence
    num_pipe = Pipeline(steps=[
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("text", text_pipe, TEXT_COL),
            ("num",  num_pipe,  NUM_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.0
    )
    return pre


def make_base_estimator(
    seed: int,
    *,
    n_components: int = 256,
    n_hash_features: int = 2**18,
    solver: str = "lbfgs",
    C: float = 0.5,
    max_iter: int = 1500,   # ↑ give LBFGS more room to converge
    tol: float = 1e-4       # ↓ slightly tighter tolerance
) -> Pipeline:
    pre = build_preprocessor(seed=seed, n_text_components=n_components, n_hash_features=n_hash_features)
    clf = LogisticRegression(
        solver=solver,
        penalty="l2",
        C=C,
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
        class_weight="balanced",  # more robust when positives are rarer
    )
    return Pipeline(steps=[("prep", pre), ("clf", clf)])


# --- 6) Metrics & plotting ----------------------------------------------------
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
        acc=float(np.round(accuracy_score(y_true, y_pred), 3)),
        prec=float(np.round(precision_score(y_true, y_pred, zero_division=0), 3)),
        rec=float(np.round(recall_score(y_true, y_pred, zero_division=0), 3)),
        f1=float(np.round(f1_score(y_true, y_pred, zero_division=0), 3)),
    )

def _group_metrics(df_meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Metrics per intersectional group.

    Returns
    -------
    pd.DataFrame
        Columns: Group, N, Accuracy, Precision, Recall, F1
    """
    g = df_meta["Group"] if "Group" in df_meta.columns else _group_labels(df_meta)
    out = []
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
    base = df_group.loc[df_group["Group"] == privileged].iloc[0]
    rows = []
    for _, r in df_group.iterrows():
        if r["Group"] == privileged:
            continue
        rows.append({
            "Comparison Group": r["Group"],
            "Accuracy Disparity": round(base["Accuracy"] - r["Accuracy"], 3),
            "Equal Opportunity Difference": round(base["Recall"] - r["Recall"], 3),
            "Precision Disparity": round(base["Precision"] - r["Precision"], 3),
        })
    return pd.DataFrame(rows).sort_values("Comparison Group").reset_index(drop=True)

@plot_dual_theme(section="fairness")
def _plot_importances(imp_df: pd.DataFrame, title: str, ax=None, palette=None, **kwargs):
    """
    Top feature importances with type coloring and % contribution labels.
    Drop-in replacement (no external changes required).
    """
    # Keep top 30 for readability
    d = imp_df.head(min(30, len(imp_df))).copy()

    # Two-type palette (no 'viridis')
    color_map = {
        "Text (SVD)": sns.color_palette("magma", 1)[0],
        "Numeric":    sns.color_palette("Set2", 1)[0],
    }
    d["color"] = d["type"].map(color_map)

    # Core plot
    sns.barplot(
        data=d, x="importance", y="feature",
        ax=ax, hue="type", dodge=False, palette=color_map, legend=True
    )
    ax.set_title(title)
    ax.set_xlabel("|Weighted coefficient|")
    ax.set_ylabel(None)
    ax.legend(title="Feature type", loc="lower right")

    # On-bar annotations: share %
    xmax = float(d["importance"].max()) if d["importance"].max() > 0 else 1.0
    for patch, (_, row) in zip(ax.patches, d.iterrows()):
        x = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(
            min(x + 0.02 * xmax, 0.98 * xmax), y,
            f"{row['share']*100:,.1f}%  |  {row['importance']:.2f}",
            va="center", ha="left", fontsize=9, clip_on=True
        )

    # Subtle grid to help reading values
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.margins(x=0.15)

    # Small explainer (works in dark/light)
    edge = ax.spines.get("left").get_edgecolor()
    face = ax.get_facecolor()
    ax.text(
        0.98, 0.02,
        "SVD components are combinations of many n-grams; bars show how much\n"
        "each component (and numeric feature) contributes to the EG decision.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor=face, edgecolor=edge, lw=0.8, alpha=0.95)
    )

    try:
        ax.figure.tight_layout(rect=[0.0, 0.0, 0.98, 1.0])
    except Exception:
        pass


@plot_dual_theme(section="fairness")
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

# --- 7) Outlier logging -------------------------------------------------------
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

# --- 8) Train + Evaluate ------------------------------------------------------
def _eg_predict_proba(mitigator: ExponentiatedGradient, X: pd.DataFrame) -> np.ndarray:
    """
    Predict p(y=1) by weighted averaging predictor predict_proba outputs.

    Parameters
    ----------
    mitigator : ExponentiatedGradient
    X : pd.DataFrame

    Returns
    -------
    np.ndarray
        Probabilities in [0,1].
    """
    probs = None
    w = np.array(mitigator.weights_)
    for i, pred in enumerate(mitigator.predictors_):
        p = pred.predict_proba(X)[:, 1]
        probs = p * w[i] if probs is None else probs + p * w[i]
    return probs / max(w.sum(), 1e-12)

def _top_outliers(df_meta: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray, k: int = 10) -> pd.DataFrame:
    """
    Top-k wrong predictions with largest |p - 0.5| (most confident mistakes).

    Returns
    -------
    pd.DataFrame
        Columns: video_id, title, Group, y_true, y_pred, prob, margin_abs
    """
    y_pred = (y_prob >= 0.5).astype(int)
    wrong = np.where(y_pred != y_true)[0]
    margin = np.abs(y_prob[wrong] - 0.5)
    labels = _group_labels(df_meta).to_numpy()
    rows = []
    for idx in np.argsort(-margin)[:k]:
        i = wrong[idx]
        rows.append({
            "video_id": df_meta.iloc[i]["video_id"],
            "title": df_meta.iloc[i].get("title", ""),
            "Group": labels[i],
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
            "prob": float(np.round(y_prob[i], 3)),
            "margin_abs": float(np.round(margin[idx], 3))
        })
    return pd.DataFrame(rows)

def _eg_weighted_importances(mitigator: ExponentiatedGradient, n_components: int) -> pd.DataFrame:
    """
    Weight-averaged absolute coefficient importances across EG predictors.

    Returns
    -------
    pd.DataFrame
        Columns: feature, type, importance, share, cum_share (sorted by importance desc).
    """
    w = np.array(mitigator.weights_)
    coefs = []
    for pred in mitigator.predictors_:
        # Handle both Pipeline({"clf": LR}) and plain LR
        clf = pred.named_steps["clf"] if hasattr(pred, "named_steps") and "clf" in pred.named_steps else pred
        coefs.append(np.abs(clf.coef_.ravel()))
    W = np.average(np.stack(coefs, axis=0), axis=0, weights=w)

    # Human-friendlier names + types
    text_feats = [f"SVD #{i+1:03d}" for i in range(n_components)]
    feat_names = text_feats + NUM_COLS
    feat_types = (["Text (SVD)"] * len(text_feats)) + (["Numeric"] * len(NUM_COLS))

    k = min(len(W), len(feat_names))
    df = pd.DataFrame({"feature": feat_names[:k], "type": feat_types[:k], "importance": W[:k]})

    # Percent contribution & cumulative percent
    total = float(df["importance"].sum()) or 1.0
    df["share"] = df["importance"] / total
    df["cum_share"] = df["share"].cumsum()

    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def train_eval_inproc(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    seed: int,
    eps: float = 0.02,
    n_components: int = 256,
    n_hash_features: int = 2**18,
    max_iter: int = 1000,
    gold: Optional[pd.DataFrame] = None
) -> Dict[str, object]:
    """
    Train Fairlearn EG (Demographic Parity) on the binary task, evaluate overall &
    group metrics, compute disparities, identify top outliers, and export artefacts.
    GOLD can override targets/groups per split where available.

    Returns
    -------
    Dict[str, object]
        Keys: mitigator, metrics_overall, metrics_group, disparities,
              preds_test_full, outliers_topk, importances
    """
    # GOLD-aware labels/groups
    y_tr_def = _prepare_binary_target(df_train).to_numpy()
    y_va_def = _prepare_binary_target(df_val).to_numpy()
    y_te_def = _prepare_binary_target(df_test).to_numpy()

    g_train_def = _group_labels(df_train)
    g_val_def   = _group_labels(df_val)
    g_test_def  = _group_labels(df_test)

    y_tr, cov_tr = maybe_override_targets(df_train, y_tr_def, gold)
    y_va, cov_va = maybe_override_targets(df_val,   y_va_def, gold)
    y_te, cov_te = maybe_override_targets(df_test,  y_te_def, gold)

    g_train = maybe_override_groups(df_train, g_train_def, gold)
    g_val   = maybe_override_groups(df_val,   g_val_def,   gold)
    g_test  = maybe_override_groups(df_test,  g_test_def,  gold)

    # --- Pre-transform with SVD once (so EG only refits LR, not SVD, each iteration) ---
    t0 = _t0("Pre-transforming features with SVD (fit once on Train) ...")
    preprocessor = build_preprocessor(seed=seed, n_text_components=n_components,
                                      n_hash_features=n_hash_features)
    X_tr_dense = preprocessor.fit_transform(df_train[[TEXT_COL] + NUM_COLS])
    X_va_dense = preprocessor.transform(df_val[[TEXT_COL] + NUM_COLS])
    X_te_dense = preprocessor.transform(df_test[[TEXT_COL] + NUM_COLS])
    _tend("inproc.pretransform", t0)

    # Store full pipeline for saving (wrap preprocessor + dummy estimator for joblib compat)
    _preprocessor_ref = preprocessor  # kept for _eg_weighted_importances

    # Base estimator: LR only (no re-SVD per EG iteration)
    from sklearn.linear_model import LogisticRegression
    lr_base = LogisticRegression(
        solver="lbfgs", penalty="l2", C=0.5,
        max_iter=max_iter, tol=1e-4,
        random_state=seed, class_weight="balanced"
    )

    t0 = _t0(f"Fitting in-processing EG (Demographic Parity, eps={eps:.2f}) on pre-transformed features ...")
    mitigator = ExponentiatedGradient(
        estimator=lr_base,
        constraints=DemographicParity(),
        eps=eps,
        sample_weight_name="sample_weight"
    )
    mitigator.fit(X_tr_dense, y_tr, sensitive_features=g_train)
    _tend("inproc.fit", t0)
    print(f"[INFO] GOLD coverage (Train/Val/Test): {cov_tr:.2%} / {cov_va:.2%} / {cov_te:.2%}")

    # Predict (on pre-transformed features)
    t0 = _t0("Predicting probabilities on Val/Test ...")
    p_val = _eg_predict_proba(mitigator, X_va_dense)
    p_tst = _eg_predict_proba(mitigator, X_te_dense)
    _tend("inproc.predict", t0)

    ypv = (p_val >= 0.5).astype(int)
    ypt = (p_tst >= 0.5).astype(int)

    # Metrics
    mo_val = _overall_metrics(y_va, ypv)
    mo_tst = _overall_metrics(y_te, ypt)
    metrics_overall = pd.DataFrame([
        {"Split": "Val",  "Accuracy": mo_val.acc, "Precision": mo_val.prec, "Recall": mo_val.rec, "F1": mo_val.f1},
        {"Split": "Test", "Accuracy": mo_tst.acc, "Precision": mo_tst.prec, "Recall": mo_tst.rec, "F1": mo_tst.f1},
    ])

    df_test_groups = df_test.copy()
    df_test_groups["Group"] = g_test.values
    mg   = _group_metrics(df_test_groups, y_te, ypt)
    disp = _disparities_vs_priv(mg, privileged="White Women")

    # Test predictions + margins
    preds_test_full = pd.DataFrame({
        "video_id": df_test["video_id"].to_numpy(),
        "title": df_test.get("title", pd.Series([""]*len(df_test))).to_numpy(),
        "Group": g_test.to_numpy(),
        "y_true": y_te,
        "y_pred": ypt,
        "prob":  np.round(p_tst, 3)
    })
    preds_test_full["margin"] = preds_test_full["prob"] - 0.5

    # Outliers + importances
    outliers_topk = _top_outliers(df_test, y_te, p_tst, k=10)
    imp_df = _eg_weighted_importances(mitigator, n_components=n_components)

    return {
        "mitigator": mitigator,
        "metrics_overall": metrics_overall,
        "metrics_group": mg,
        "disparities": disp,
        "preds_test_full": preds_test_full,
        "outliers_topk": outliers_topk,
        "importances": imp_df
    }

# --- 9) IO helpers ------------------------------------------------------------
def _save_all(bundle: Dict[str, object], *, is_selfcheck: bool = False, save_model: bool = True) -> None:
    """
    Persist metrics, predictions, plots, LaTeX tables, narrative, and (optionally) model.

    Notes
    -----
    - Writes step-aligned names (11_inproc_*) and legacy egdp_* for compatibility.
    - Self-check writes *_selfcheck and does not overwrite full artefacts.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    NARR_DIR.mkdir(parents=True, exist_ok=True)

    suff = "_selfcheck" if is_selfcheck else ""

    # Step-11 friendly names
    mo_11 = DATA_DIR / f"11_inproc_overall_metrics{suff}.csv"
    mg_11 = DATA_DIR / f"11_inproc_group_metrics{suff}.csv"
    md_11 = DATA_DIR / f"11_inproc_disparities{suff}.csv"
    pt_11 = DATA_DIR / f"11_inproc_predictions_test{suff}.csv"
    ot_11 = DATA_DIR / f"11_inproc_outliers_top10{suff}.csv"
    imp_11 = DATA_DIR / f"11_inproc_importances{suff}.csv"

    # Legacy (kept)
    mo_eg = DATA_DIR / f"egdp_overall_metrics{'_inproc_selfcheck' if is_selfcheck else '_inproc'}.csv"
    mg_eg = DATA_DIR / f"egdp_group_metrics{'_inproc_selfcheck' if is_selfcheck else '_inproc'}.csv"
    md_eg = DATA_DIR / f"egdp_disparities{'_inproc_selfcheck' if is_selfcheck else '_inproc'}.csv"
    pt_eg = DATA_DIR / f"egdp_predictions_test{'_inproc_selfcheck' if is_selfcheck else '_inproc'}.csv"
    ot_eg = DATA_DIR / f"egdp_outliers_topk{'_inproc_selfcheck' if is_selfcheck else '_inproc'}.csv"

    # Save CSVs (both new + legacy)
    bundle["metrics_overall"].to_csv(mo_11, index=False); bundle["metrics_overall"].to_csv(mo_eg, index=False)
    bundle["metrics_group"].to_csv(mg_11, index=False);   bundle["metrics_group"].to_csv(mg_eg, index=False)
    bundle["disparities"].to_csv(md_11, index=False);     bundle["disparities"].to_csv(md_eg, index=False)
    bundle["preds_test_full"].to_csv(pt_11, index=False); bundle["preds_test_full"].to_csv(pt_eg, index=False)
    bundle["outliers_topk"].to_csv(ot_11, index=False);   bundle["outliers_topk"].to_csv(ot_eg, index=False)
    bundle["importances"].to_csv(imp_11, index=False)

    print("✓ Artefacts saved:",
          mo_11.name, ",", mg_11.name, ",", md_11.name, ",", pt_11.name, ",", ot_11.name, ",", imp_11.name,
          f"(+ legacy: {mo_eg.name}, {mg_eg.name}, {md_eg.name}, {pt_eg.name}, {ot_eg.name})")

    # Plots
    _plot_importances(
        imp_df=bundle["importances"],
        title="EG (Demographic Parity) — Weighted Importances",
        save_path=str(FIG_DIR / f"11_inproc_importances{suff}"),
        figsize=(9, 8)
    )
    _plot_margins(
        margins=bundle["preds_test_full"]["margin"].to_numpy(),
        title="Decision Margins (Test) — EG (DP)",
        save_path=str(FIG_DIR / f"11_inproc_margins{suff}"),
        figsize=(9, 6)
    )

    # LaTeX (group metrics)
    dataframe_to_latex_table(
        bundle["metrics_group"].set_index("Group"),
        str(TEX_DIR / f"11_inproc_group_metrics{suff}.tex"),
        "Group-wise performance under in-processing (Exponentiated Gradient, Demographic Parity).",
        "tab:11-inproc-group-metrics"
    )

    # Narrative (qualitative analysis)
    lines = [f"# Automated Summary: In-Processing (EG, DP){' — self-check' if is_selfcheck else ''}\n"]
    lines.append("## Overall Metrics\n")
    lines.append(bundle["metrics_overall"].to_string(index=False))
    lines.append("\n## Group Metrics\n")
    lines.append(bundle["metrics_group"].to_string(index=False))
    lines.append("\n## Disparities vs. White Women\n")
    lines.append(bundle["disparities"].to_string(index=False))
    lines.append("\n## Top 5 Outliers (by confident mistakes)\n")
    lines.append(bundle["outliers_topk"].head(5).to_string(index=False))
    lines.append("\n*Note:* Some titles are non-English; tags/categories help anchor semantics.")
    narr_path = NARR_DIR / f"11_inprocessing_summary{suff}.md"
    with open(narr_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Narrative saved: {narr_path.resolve()}")

    # Model artefact (optional; compressed; guarded)
    if save_model:
        try:
            joblib.dump(bundle["mitigator"], MODEL_DIR / "egdp_mitigator.joblib", compress=3, protocol=4)
            print(f"✓ Model saved: {(MODEL_DIR / 'egdp_mitigator.joblib').resolve()}")
        except OSError as e:
            print(f"✗ Skipping model save ({e}). Tip: re-run with --skip-save-model or free disk space.")

# --- 10) Main -----------------------------------------------------------------
def main(argv: Optional[list] = None) -> None:
    """
    Run in-processing mitigation with EG (DP) on the binary task.

    Options
    -------
    --selfcheck --sample INT  (random sample; non-destructive)
    --eps FLOAT, --svd-components INT, --hash-features INT, --max-iter INT
    --use-gold --gold-path PATH
    --skip-save-model
    """
    t_all = time.perf_counter()
    print("--- Starting Step 11: In-processing Mitigation (EG, Demographic Parity) ---")

    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--svd-components", type=int, default=256)
    p.add_argument("--hash-features", type=int, default=2**18)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--use-gold", action="store_true", help="Use GOLD labels/groups if available.")
    p.add_argument("--gold-path", type=str, default=str(OUTPUT_DIR / "data" / "gold" / "gold_final.csv"))
    p.add_argument("--skip-save-model", action="store_true", help="Skip saving the (large) EG model.")
    args = p.parse_args(argv)

    # Load GOLD (optional)
    gold = load_gold_table(Path(args.gold_path)) if args.use_gold else None

    # Load corpus
    df = _load_corpus()
    total = len(df)
    print(f"[STATS] Total videos available: {total:,}")
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
    bundle = train_eval_inproc(
        df_train=train_df,
        df_val=val_df,
        df_test=test_df,
        seed=args.seed,
        eps=args.eps,
        n_components=args.svd_components,
        n_hash_features=args.hash_features,
        max_iter=args.max_iter,
        gold=gold
    )
    _save_all(bundle, is_selfcheck=args.selfcheck, save_model=not args.skip_save_model)

    # Console highlights (qualitative)
    print("\n=== Overall Metrics (Accuracy/Precision/Recall/F1) ===")
    print(bundle["metrics_overall"].to_string(index=False))
    print("\n=== Group Metrics (Test) ===")
    print(bundle["metrics_group"].to_string(index=False))
    print("\n=== Disparities vs. White Women (privileged) ===")
    print(bundle["disparities"].to_string(index=False))
    print("\n=== Top outliers (Test) — most confident mistakes ===")
    print(bundle["outliers_topk"].to_string(index=False))
    print("\nNote: Outliers intentionally include title + Group for interpretability; "
          "some titles may be non-English; use tags/categories for context.")

    _tend("inproc.step11_total", t_all)
    print("\n--- Step 11: In-processing Mitigation Completed Successfully ---")

if __name__ == "__main__":
    main()
