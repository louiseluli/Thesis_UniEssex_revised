# -*- coding: utf-8 -*-
"""
fairness_evaluation_utils.py
============================

Purpose
-------
Reusable utilities for fairness evaluation across modelling steps (RF, BERT, and
pre-/in-/post-processing mitigations). This module centralizes:
- Overall and group-wise metrics (Accuracy, Precision, Recall/TPR, F1).
- Disparity tables vs. a privileged group (White Women, by default), with a
  robust fallback if the privileged group is absent in the current slice.
- Outlier selection for qualitative analysis (most confident mistakes).
- Optional GOLD standard helpers (load, align by video_id, evaluate).

What it does
------------
1) Provides a consistent intersectional grouping function used across the repo:
   Black Women, White Women, Asian Women, Latina Women, and a catch-all "Other".
2) Implements standard rounded metrics and disparity tables with stable column
   names ("Accuracy Disparity", "Equal Opportunity Difference", "Precision Disparity")
   to remain compatible with Step-13 consolidation.
3) Adds lightweight timers for quick diagnostics.
4) Exposes GOLD helpers that do NOT overwrite canonical artefacts. The GOLD file
   is optional and kept separate from default metrics.

Interpretability notes
----------------------
- Some titles are not English; tags/categories often anchor the semantics.
  Outlier lists explicitly include titles so you can inspect them qualitatively.

CLI (tiny self-check; safe & non-destructive)
---------------------------------------------
python -m src.fairness.fairness_evaluation_utils --selfcheck --k 5 --sample 8000

- Reads a small sample from the canonical parquet.
- Builds a trivial "coin-flip" predictor to exercise metrics, group tables,
  disparities, and outliers.
- Writes self-check artefacts with 'selfcheck_' prefix only.
"""

# --- Imports (keep at top) ----------------------------------------------------
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Project config (for consistent seed/paths)
from src.utils.theme_manager import load_config

CONFIG = load_config()
if CONFIG is None:
    raise RuntimeError("Failed to load config/settings.yaml — check the file exists and is valid YAML.")
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))

# Common paths (used only in self-check; imported, not required by helpers)
DATA_DIR = Path(CONFIG["paths"]["data"])
# Step 01 writes outputs/data/01_ml_corpus.parquet
CORPUS_PATH = DATA_DIR / "01_ml_corpus.parquet"

__all__ = [
    "group_labels_intersectional",
    "ClassifMetrics",
    "overall_metrics",
    "calculate_group_metrics",
    "calculate_fairness_disparities",
    "top_confident_outliers",
    "load_gold_table",
    "align_gold_to_frame",
    "evaluate_against_gold",
    "maybe_override_targets",
    "maybe_override_groups",
    "json_safe",
    "df_json_records",
    "package_fairness_summary",
    # Standalone fairness-gap helpers (used by tests)
    "demographic_parity_difference",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "expected_calibration_error",
    "compute_all_metrics",
]


# --- Lightweight timers -------------------------------------------------------
def _t0(msg: str) -> float:
    """
    Start a timer and print a standardized header.

    Parameters
    ----------
    msg : str
        The message printed before timing starts.

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
        A short label that describes the timed block.
    t0 : float
        Start time returned by _t0.
    """
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")

# --- JSON-safe helpers --------------------------------------------------------
def json_safe(x: any) -> any:
    """
    Convert common pandas/numpy objects to plain Python types for JSON encoding.

    Rules
    -----
    - numpy scalar -> Python scalar
    - pandas NA/NaT -> None
    - pandas Timestamp -> ISO string
    - numpy arrays / pandas Series -> list
    - lists/tuples/dicts -> recurse
    """
    if x is None:
        return None

    # pandas NA/NaT and Timestamp
    if x is pd.NA:
        return None
    if isinstance(x, pd.Timestamp):
        # always tz-naive ISO
        return x.tz_localize(None).isoformat() if x.tzinfo else x.isoformat()

    # numpy scalar -> Python scalar
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(x, (np.bool_,)):
        return bool(x)

    # numpy / pandas containers
    if isinstance(x, (np.ndarray,)):
        return [json_safe(v) for v in x.tolist()]
    if isinstance(x, (pd.Series,)):
        return [json_safe(v) for v in x.to_list()]
    if isinstance(x, (pd.DataFrame,)):
        return df_json_records(x)

    # builtins
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}

    # numeric weirdness
    if isinstance(x, float):
        if not np.isfinite(x):
            return None
        return x

    return x


def df_json_records(df: pd.DataFrame) -> List[Dict[str, any]]:
    """
    Convert a DataFrame to a list of JSON-safe dicts (records orientation).
    """
    recs = []
    for _, row in df.iterrows():
        rec = {}
        for c, v in row.items():
            rec[str(c)] = json_safe(v)
        recs.append(rec)
    return recs


def package_fairness_summary(
    group_metrics: pd.DataFrame,
    disparities: pd.DataFrame,
    *,
    notes: Optional[str] = None
) -> Dict[str, any]:
    """
    Package group metrics & disparities into a JSON-safe structure for dashboards.

    Parameters
    ----------
    group_metrics : pd.DataFrame
        Output of calculate_group_metrics.
    disparities : pd.DataFrame
        Output of calculate_fairness_disparities.
    notes : Optional[str]
        Additional context for the consumer (e.g., "privileged fallback used").

    Returns
    -------
    Dict[str, Any]
        { 'group_metrics': [...], 'disparities': [...], 'notes': str|None }
    """
    return {
        "group_metrics": df_json_records(group_metrics) if group_metrics is not None else [],
        "disparities": df_json_records(disparities) if disparities is not None else [],
        "notes": notes,
    }

# --- Grouping and metrics -----------------------------------------------------
def _as_bool(df: pd.DataFrame, name: str) -> pd.Series:
    """Interpret 1/0, True/False, '1'/'0' consistently as booleans."""
    if name not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[name]
    # numeric/boolean → numeric; strings '1'/'true' handled by to_numeric
    if s.dtype == bool:
        return s.fillna(False)
    s_num = pd.to_numeric(s, errors="coerce")  # True→1, '1'→1, False/None→NaN
    return s_num.fillna(0).astype(float) > 0.5

def group_labels_intersectional(df: pd.DataFrame) -> pd.Series:
    """Create intersectional group labels for fairness diagnostics (robust).

    Priority order matches stratification key in step 06:
    Black Women > White Women > Asian Women > Latina Women > Other.
    """
    labels = np.full(len(df), "Other", dtype=object)
    gf = _as_bool(df, "gender_female")
    bw = _as_bool(df, "race_ethnicity_black")  & gf
    ww = _as_bool(df, "race_ethnicity_white")  & gf & ~bw
    aw = _as_bool(df, "race_ethnicity_asian")  & gf & ~bw & ~ww
    lw = _as_bool(df, "race_ethnicity_latina") & gf & ~bw & ~ww & ~aw
    labels[bw] = "Black Women"
    labels[ww] = "White Women"
    labels[aw] = "Asian Women"
    labels[lw] = "Latina Women"
    return pd.Series(labels, index=df.index, name="Group")


@dataclass
class ClassifMetrics:
    """
    Container for rounded binary classification metrics.

    Attributes
    ----------
    acc : float
        Accuracy rounded to 3 decimals.
    prec : float
        Precision rounded to 3 decimals (safe with zero_division=0).
    rec : float
        Recall/TPR rounded to 3 decimals (safe with zero_division=0).
    f1 : float
        F1-score rounded to 3 decimals (safe with zero_division=0).
    """
    acc: float
    prec: float
    rec: float
    f1: float


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassifMetrics:
    """
    Compute Accuracy, Precision, Recall (TPR), and F1 with 3-decimal rounding.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels {0,1}.
    y_pred : np.ndarray
        Predicted binary labels {0,1}.

    Returns
    -------
    ClassifMetrics
        A simple dataclass bundle for logging and table generation.
    """
    return ClassifMetrics(
        acc=float(np.round(accuracy_score(y_true, y_pred), 3)),
        prec=float(np.round(precision_score(y_true, y_pred, zero_division=0), 3)),
        rec=float(np.round(recall_score(y_true, y_pred, zero_division=0), 3)),
        f1=float(np.round(f1_score(y_true, y_pred, zero_division=0), 3)),
    )


def calculate_group_metrics(
    df_meta: pd.DataFrame,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    group_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute per-group metrics for fairness reporting.

    Preference order:
      1) explicit 'group_col' if provided and present
      2) existing 'Group' column if present
      3) fallback: derive via group_labels_intersectional(df_meta)
    """
    d = df_meta.copy()
    if y_true is None:
        if "y_true" not in d.columns:
            raise ValueError("calculate_group_metrics: y_true not provided and 'y_true' column not found.")
        y_true = d["y_true"].to_numpy()
    if y_pred is None:
        if "y_pred" not in d.columns:
            raise ValueError("calculate_group_metrics: y_pred not provided and 'y_pred' column not found.")
        y_pred = d["y_pred"].to_numpy()

    if group_col and group_col in d.columns:
        g = d[group_col]
    elif "Group" in d.columns:
        g = d["Group"]
    else:
        g = group_labels_intersectional(d)

    out = []
    for grp, idx in d.groupby(g).groups.items():
        yt, yp = y_true[idx], y_pred[idx]
        out.append({
            "Group": grp,
            "N": int(len(idx)),
            "Accuracy": round(accuracy_score(yt, yp), 3),
            "Precision": round(precision_score(yt, yp, zero_division=0), 3),
            "Recall": round(recall_score(yt, yp, zero_division=0), 3),
            "F1": round(f1_score(yt, yp, zero_division=0), 3),
        })
    return pd.DataFrame(out).sort_values("Group").reset_index(drop=True)
# ============================================================================ #

# ===================== DROP-IN: robust disparities function ====================
def calculate_fairness_disparities(df_group: pd.DataFrame,
                                   privileged: str = "White Women") -> pd.DataFrame:
    """
    Compute disparity table vs. a privileged group.
    If fewer than 2 groups or the privileged group is absent, return empty.
    """
    if "Group" not in df_group.columns or len(df_group["Group"].unique()) < 2:
        return pd.DataFrame(columns=[
            "Comparison Group", "Accuracy Disparity", "Equal Opportunity Difference", "Precision Disparity"
        ])

    base = df_group.loc[df_group["Group"] == privileged]
    if base.empty:
        return pd.DataFrame(columns=[
            "Comparison Group", "Accuracy Disparity", "Equal Opportunity Difference", "Precision Disparity"
        ])
    base = base.iloc[0]

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
# ==============================================================================


def top_confident_outliers(
    df_meta: pd.DataFrame,
    probs: np.ndarray,
    k: int = 10
) -> pd.DataFrame:
    """
    Select top-k confident mistakes: those with largest |p - 0.5| among errors.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Frame aligned to probs and including 'y_true' (binary).
    probs : np.ndarray
        Predicted probability for the positive class (float in [0, 1]).
    k : int
        Number of outliers to return.

    Returns
    -------
    pd.DataFrame
        Columns: video_id, title, Group, y_true, y_pred, prob, margin_abs
    """
    if "y_true" not in df_meta.columns:
        raise ValueError("top_confident_outliers: required column 'y_true' missing.")
    y_true = df_meta["y_true"].to_numpy()
    y_pred = (probs >= 0.5).astype(int)
    wrong = np.where(y_pred != y_true)[0]
    if wrong.size == 0:
        return pd.DataFrame(columns=["video_id", "title", "Group", "y_true", "y_pred", "prob", "margin_abs"])

    margin = np.abs(probs[wrong] - 0.5)
    g = group_labels_intersectional(df_meta).to_numpy()
    rows: List[Dict[str, object]] = []
    for idx in np.argsort(-margin)[:k]:
        i = wrong[idx]
        rows.append({
            "video_id": df_meta.iloc[i].get("video_id", None),
            "title": df_meta.iloc[i].get("title", ""),
            "Group": g[i],
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
            "prob": float(np.round(probs[i], 3)),
            "margin_abs": float(np.round(margin[idx], 3)),
        })
    return pd.DataFrame(rows)


# --- GOLD helpers --------------------------------------------------------------
def load_gold_table(path: Union[Path, str]) -> Optional[pd.DataFrame]:
    """
    Load a GOLD annotation table if present; else return None.

    Expected schema
    ---------------
    - video_id : join key present in the corpus
    - gold_label or is_amateur_gold : int in {0,1} (1 = positive class)
    - optional: `group_gold`, `notes`, `annotator_id` are carried along for reporting

    Parameters
    ----------
    path : Union[pathlib.Path, str]
        Path to CSV.

    Returns
    -------
    Optional[pd.DataFrame]
        The loaded DataFrame (unchanged) or None if not found / malformed.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"✗ GOLD read error: {e}")
        return None

    if "video_id" not in df.columns:
        print("✗ GOLD file missing 'video_id' column. Ignoring.")
        return None

    # Normalize/ensure label column
    if "gold_label" not in df.columns and "is_amateur_gold" in df.columns:
        df["gold_label"] = df["is_amateur_gold"]
    if "gold_label" not in df.columns:
        print("✗ GOLD file missing 'gold_label' (or 'is_amateur_gold'). Ignoring.")
        return None

    df["gold_label"] = pd.to_numeric(df["gold_label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return df


def align_gold_to_frame(df: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join a dataframe to the GOLD table by 'video_id'.

    Parameters
    ----------
    df : pd.DataFrame
        The base frame (predictions/meta) containing 'video_id'.
    gold : pd.DataFrame
        Output of `load_gold_table`.

    Returns
    -------
    pd.DataFrame
        Joined frame with a 'gold_label' column. May be empty if no overlap.
    """
    if "video_id" not in df.columns:
        raise ValueError("align_gold_to_frame: base frame missing 'video_id'.")

    keep_cols = ["video_id", "gold_label"]
    keep_cols += [c for c in ("group_gold", "notes", "annotator_id") if c in gold.columns]
    res = df.merge(gold[keep_cols], on="video_id", how="inner")
    print(f"[INFO] GOLD alignment: {len(res)}/{len(df)} videos have gold labels")
    return res


def evaluate_against_gold(
    pred_probs: np.ndarray,
    gold_frame: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate predictions against GOLD labels (non-destructive helper).

    Parameters
    ----------
    pred_probs : np.ndarray
        Positive-class probabilities aligned to `gold_frame`.
    gold_frame : pd.DataFrame
        Frame with at least 'video_id' and 'gold_label'. Optional title is used
        for interpretability if present.

    Returns
    -------
    (metrics_df, preds_df)
        metrics_df: DataFrame with one row ('Gold') and Accuracy/Precision/Recall/F1.
        preds_df: per-ID GOLD predictions with prob and margin.
    """
    if "gold_label" not in gold_frame.columns:
        raise ValueError("evaluate_against_gold: 'gold_label' column not found.")

    y_true = gold_frame["gold_label"].astype(int).to_numpy()
    y_pred = (pred_probs >= 0.5).astype(int)

    met = ClassifMetrics(
        acc=float(np.round(accuracy_score(y_true, y_pred), 3)),
        prec=float(np.round(precision_score(y_true, y_pred, zero_division=0), 3)),
        rec=float(np.round(recall_score(y_true, y_pred, zero_division=0), 3)),
        f1=float(np.round(f1_score(y_true, y_pred, zero_division=0), 3)),
    )
    metrics_df = pd.DataFrame([{
        "Split": "Gold",
        "Accuracy": met.acc, "Precision": met.prec, "Recall": met.rec, "F1": met.f1
    }])

    preds_df = pd.DataFrame({
        "video_id": gold_frame["video_id"].to_numpy(),
        "title": gold_frame.get("title", pd.Series([""]*len(gold_frame))).to_numpy(),
        "Group": group_labels_intersectional(gold_frame).to_numpy()
                 if "group_gold" not in gold_frame.columns
                 else gold_frame["group_gold"].fillna("Other").to_numpy(),
        "y_true": y_true,
        "y_pred": y_pred,
        "prob": np.round(pred_probs, 3),
    })
    preds_df["margin"] = preds_df["prob"] - 0.5
    return metrics_df, preds_df


def maybe_override_targets(
    df: pd.DataFrame,
    default_targets: np.ndarray,
    gold: Optional[pd.DataFrame]
) -> Tuple[np.ndarray, float]:
    """
    Override targets with GOLD labels where available.

    Parameters
    ----------
    df : pd.DataFrame
        Base dataframe with 'video_id'.
    default_targets : np.ndarray
        Original targets (binary vector).
    gold : Optional[pd.DataFrame]
        GOLD table with 'video_id' and 'gold_label'/'is_amateur_gold'.

    Returns
    -------
    (targets, coverage_fraction)
    """
    if gold is None:
        return default_targets, 0.0

    g = gold.copy()
    if "gold_label" not in g.columns and "is_amateur_gold" in g.columns:
        g["gold_label"] = g["is_amateur_gold"]

    merged = df[["video_id"]].merge(
        g[["video_id", "gold_label"]],
        on="video_id", how="left"
    )
    has_gold = merged["gold_label"].notna()
    coverage = float(has_gold.mean())
    targets = default_targets.copy()
    targets[has_gold.to_numpy()] = merged.loc[has_gold, "gold_label"].astype(int).to_numpy()
    return targets, coverage


def maybe_override_groups(
    df: pd.DataFrame,
    default_groups: pd.Series,
    gold: Optional[pd.DataFrame]
) -> pd.Series:
    """
    Override groups with GOLD group labels where available.

    Parameters
    ----------
    df : pd.DataFrame
        Base dataframe with 'video_id'.
    default_groups : pd.Series
        Default group labels (derived intersectionally).
    gold : Optional[pd.DataFrame]
        GOLD table optionally containing 'group_gold'.

    Returns
    -------
    pd.Series
        Group labels with GOLD overrides where available.
    """
    if gold is None or "group_gold" not in gold.columns:
        return default_groups

    merged = df[["video_id"]].merge(
        gold[["video_id", "group_gold"]],
        on="video_id", how="left"
    )
    groups = default_groups.copy()
    has_gold = merged["group_gold"].notna()
    groups.loc[has_gold] = merged.loc[has_gold, "group_gold"].astype(str).values
    return groups


# --- Standalone fairness-gap helpers (test-friendly, no corpus dependency) ----

def demographic_parity_difference(
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    Maximum positive-prediction rate gap across groups (demographic parity).

    Parameters
    ----------
    y_pred : np.ndarray
        Binary predictions ``{0, 1}``.
    groups : np.ndarray
        Group membership labels aligned to ``y_pred``.

    Returns
    -------
    float
        ``max(rate) - min(rate)`` across groups; 0 if fewer than 2 groups.
    """
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        return 0.0
    rates = [float(y_pred[groups == g].mean()) for g in unique]
    return float(max(rates) - min(rates))


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    Maximum true-positive-rate (recall) gap across groups (equal opportunity).

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels ``{0, 1}``.
    y_pred : np.ndarray
        Binary predictions ``{0, 1}``.
    groups : np.ndarray
        Group membership labels aligned to ``y_true``/``y_pred``.

    Returns
    -------
    float
        ``max(TPR) - min(TPR)`` across groups; 0 if fewer than 2 groups.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        return 0.0
    tprs = []
    for g in unique:
        mask = (groups == g) & (y_true == 1)
        tprs.append(float(y_pred[mask].mean()) if mask.sum() > 0 else 0.0)
    return float(max(tprs) - min(tprs))


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    Maximum of the TPR gap and FPR gap across groups (equalized odds).

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels.
    y_pred : np.ndarray
        Binary predictions.
    groups : np.ndarray
        Group membership labels.

    Returns
    -------
    float
        ``max(TPR_gap, FPR_gap)``; 0 if fewer than 2 groups.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        return 0.0
    tprs, fprs = [], []
    for g in unique:
        pos_mask = (groups == g) & (y_true == 1)
        neg_mask = (groups == g) & (y_true == 0)
        tprs.append(float(y_pred[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0)
        fprs.append(float(y_pred[neg_mask].mean()) if neg_mask.sum() > 0 else 0.0)
    tpr_gap = max(tprs) - min(tprs)
    fpr_gap = max(fprs) - min(fprs)
    return float(max(tpr_gap, fpr_gap))


def expected_calibration_error(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) — weighted mean absolute calibration gap.

    Bins are equal-width over ``[0, 1]``. Empty bins are skipped. Works with
    both continuous probability scores and binary 0/1 predictions (the latter
    produces a two-bin ECE).

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels.
    y_scores : np.ndarray
        Probability scores or binary predictions in ``[0, 1]``.
    n_bins : int
        Number of equal-width bins. Default 10.

    Returns
    -------
    float
        ECE in ``[0, 1]``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    n = len(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_scores >= lo) & (y_scores < hi)
        if not mask.any():
            continue
        frac = mask.sum() / n
        mean_conf = float(y_scores[mask].mean())
        mean_acc = float(y_true[mask].mean())
        ece += frac * abs(mean_conf - mean_acc)
    return float(ece)


def compute_all_metrics(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """
    Compute accuracy, per-group fairness gaps, and max gap for a fitted model.

    Parameters
    ----------
    model : sklearn estimator
        A fitted model that implements ``predict``.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Ground-truth binary labels.
    groups : np.ndarray
        Group membership labels.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``demographic_parity``, ``equal_opportunity``,
        ``equalized_odds``, ``max_gap``.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))
    dpd = demographic_parity_difference(y_pred, groups)
    eod = equal_opportunity_difference(y, y_pred, groups)
    eo = equalized_odds_difference(y, y_pred, groups)
    return {
        "accuracy": acc,
        "demographic_parity": dpd,
        "equal_opportunity": eod,
        "equalized_odds": eo,
        "max_gap": max(dpd, eod, eo),
    }


# --- Tiny self-check -----------------------------------------------------------
def _selfcheck(sample: int = 8000, k_outliers: int = 5) -> None:
    """
    Run a tiny self-check on a random sample to exercise all utilities.
    Writes only 'selfcheck_' artefacts in outputs/data (non-destructive).

    Parameters
    ----------
    sample : int
        Number of rows to sample from the corpus.
    k_outliers : int
        How many confident mistakes to list.
    """
    if not CORPUS_PATH.exists():
        print(f"✗ Corpus not found at {CORPUS_PATH}; self-check skipped.")
        return

    t0 = _t0(f"[READ] Parquet (self-check): {CORPUS_PATH}")
    df = pd.read_parquet(CORPUS_PATH)
    _tend("fairutils.load_corpus_selfcheck", t0)

    df = df.sample(n=min(sample, len(df)), random_state=SEED).reset_index(drop=True)

    # Build a toy "coin flip" predictor around the empirical positive rate
    primary = df.get("categories", "").fillna("").astype(str).str.split(",").str[0].str.strip()
    y_true = (primary == "Amateur").astype(int).to_numpy()
    pos_rate = float(np.mean(y_true))
    rng = np.random.default_rng(SEED)
    probs = rng.uniform(0, 1, size=len(df))
    y_pred = (probs < pos_rate).astype(int)

    # Group metrics and disparities
    dfx = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    dfx["video_id"] = df.get("video_id", pd.RangeIndex(len(df)))
    dfx["title"] = df.get("title", "")
    dfx = pd.concat([dfx, df], axis=1)

    gm = calculate_group_metrics(dfx)
    disp = calculate_fairness_disparities(gm, privileged="White Women")
    outs = top_confident_outliers(dfx, probs=probs, k=k_outliers)

    # Save self-check artefacts
    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    gm.to_csv(out_dir / "selfcheck_fairness_metrics.csv", index=False)
    disp.to_csv(out_dir / "selfcheck_fairness_disparities.csv", index=False)
    outs.to_csv(out_dir / "selfcheck_outliers.csv", index=False)
    print("✓ Self-check artefacts saved: selfcheck_fairness_metrics.csv, selfcheck_fairness_disparities.csv, selfcheck_outliers.csv")


# --- Module entrypoint (self-check only) --------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Small CLI to run a tiny self-check. Never overwrites canonical artefacts.

    Options
    -------
    --selfcheck           Run the self-check.
    --sample INT          Random sample size (default: 8000).
    --k INT               Number of outliers to list (default: 5).
    """
    import argparse

    t_all = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=8000)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args(argv)

    if args.selfcheck:
        _selfcheck(sample=args.sample, k_outliers=args.k)
    else:
        print("Nothing to do. Use --selfcheck to run a tiny validation.")

    _tend("fairutils.total", t_all)


if __name__ == "__main__":
    main()