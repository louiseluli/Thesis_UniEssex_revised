# -*- coding: utf-8 -*-
"""
08_comprehensive_evaluation.py
==============================

Purpose
-------
Comprehensively evaluate model predictions: global curves (ROC/PR), threshold
sweep, per-group metrics, disparities vs White Women, fairness-vs-threshold,
confusion matrix, and a narrative. Self-check is random and non-destructive.

What it does
------------
1) Loads Val/Test predictions (Step 07 defaults). Prefers existing 'Group'
   labels; if there are <2 unique groups, **repairs** groups by joining the
   corpus and re-deriving via robust one-hot parsing.
2) Computes AUROC/AP, selects an operating threshold (F1-optimal on Val, else
   0.50), and reports per-group metrics + disparities at that threshold.
3) Computes a fairness curve (max |EOD| vs threshold). If <2 groups overall,
   returns NaNs without spamming warnings.
4) Saves CSVs, dual-theme figures, LaTeX, and a narrative with
   qualitative notes and confident mistakes.

Counts note
-----------
Elsewhere (PMI/harm/EDA) totals can exceed N due to multi-label; here the target
is a single binary per video, so counts sum to N (groups can be imbalanced).

Interpretability (MPU)
----------------------
Some titles are non-English; tags/categories help anchor semantics. We call out
confident mistakes (largest |p-0.5| among errors).
"""

from __future__ import annotations

# --- Imports ----------------------------------------------------
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix
)

# Project utils
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.fairness.fairness_evaluation_utils import (
    calculate_group_metrics,
    calculate_fairness_disparities,
    top_confident_outliers,
    group_labels_intersectional
)
from src.utils.academic_tables import dataframe_to_latex_table
try:
    from metrics.cadp import cadp_score  # preferred name
except Exception:
    try:
        from metrics.cadp import cadp as cadp_score  # alt name used in some repos
    except Exception:
        cadp_score = None

# --- 1) Config & paths --------------------------------------------------------
CONFIG = load_config()
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))

DATA_DIR    = Path(CONFIG["paths"]["data"])
FIG_DIR     = Path(CONFIG["paths"]["figures"]) / "eval"
NARR_DIR    = Path(CONFIG["paths"]["narratives"]) / "automated"
TABLES_DIR  = Path(CONFIG["project"]["root"]) / "dissertation" / "auto_tables"

# Inputs default to Step-07 output names
VAL_PREDS_CSV_DEFAULT  = DATA_DIR / "07_rf_val_predictions.csv"
TEST_PREDS_CSV_DEFAULT = DATA_DIR / "07_rf_test_predictions.csv"
# Corpus for group repair (do not change)
CORPUS_PATH            = DATA_DIR / "01_ml_corpus.parquet"

# Outputs (08_*)
TH_SWEEP_CSV           = DATA_DIR / "08_threshold_sweep.csv"
FAIRNESS_CURVE_CSV     = DATA_DIR / "08_fairness_curve.csv"
GROUP_METRICS_AT_OPT   = DATA_DIR / "08_group_metrics_at_opt.csv"
DISPARITIES_AT_OPT     = DATA_DIR / "08_disparities_at_opt.csv"
CONFUSION_AT_OPT_CSV   = DATA_DIR / "08_confusion_matrix_at_opt.csv"
NARRATIVE_PATH         = NARR_DIR / "08_comprehensive_evaluation.md"
CADP_AT_OPT_CSV       = DATA_DIR / "08_cadp_at_opt.csv"
CADP_SWEEP_CSV        = DATA_DIR / "08_cadp_curve.csv"


def _with_suffix(path: Path, suffix: str) -> Path:
    """Append a suffix before file extension."""
    return path.with_name(path.stem + suffix + path.suffix)

# --- 2) Lightweight timers ----------------------------------------------------
def _t0(msg: str) -> float:
    t = time.perf_counter()
    print(msg)
    return t

def _tend(label: str, t0: float) -> None:
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")


# --- 3) I/O helpers -----------------------------------------------------------
def _read_preds_or_die(path: Path) -> pd.DataFrame:
    """
    Read a predictions CSV produced by Step 07.

    Expected columns:
      - video_id, title (optional)
      - y_true, y_pred, prob
      - Group  (preferred; else we will derive)
    """
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    d = pd.read_csv(path)
    if "prob" in d.columns:
        d["prob"] = pd.to_numeric(d["prob"], errors="coerce").round(3)
    for c in ("y_true", "y_pred"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int).clip(0, 1)
    # If missing, derive; if present but degenerate, we'll repair later.
    if "Group" not in d.columns:
        d["Group"] = group_labels_intersectional(d)
    return d


def _repair_groups_if_needed(preds: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    """
    If there are <2 unique groups (or force=True), join to the corpus to re-derive
    'Group' via robust one-hot parsing. Leaves original columns intact.
    """
    need = force or ("Group" not in preds.columns) or (preds["Group"].nunique(dropna=True) < 2)
    if not need:
        return preds

    if not CORPUS_PATH.exists():
        print("⚠ Group repair requested but corpus parquet not found; continuing without repair.")
        return preds

    t0 = _t0("[REPAIR] Re-deriving 'Group' by joining to corpus ...")
    corpus = pd.read_parquet(CORPUS_PATH)[[
        "video_id", "race_ethnicity_black", "race_ethnicity_white",
        "race_ethnicity_asian", "race_ethnicity_latina", "gender_female"
    ]]
    merged = preds.merge(corpus, on="video_id", how="left", suffixes=("", "_c"))
    repaired = preds.copy()
    repaired["Group"] = group_labels_intersectional(merged)
    n_unique = repaired["Group"].nunique(dropna=True)
    print(f"[REPAIR] Unique groups after repair: {n_unique}")
    _tend("eval.group_repair", t0)
    return repaired


# --- 4) Metrics & selection ---------------------------------------------------
def _global_curves(y_true: np.ndarray, p: np.ndarray):
    """Return AUROC/AP and ROC & PR points."""
    auroc = float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan")
    ap    = float(average_precision_score(y_true, p))
    fpr, tpr, _ = roc_curve(y_true, p)
    prec, rec, _ = precision_recall_curve(y_true, p)
    return auroc, ap, fpr, tpr, prec, rec


def _threshold_sweep(y_true: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Sweep thresholds and compute Accuracy/Precision/Recall/F1."""
    rows = []
    for th in thresholds:
        yp = (p >= th).astype(int)
        tp = int((yp & y_true).sum())
        fp = int((yp & (1 - y_true)).sum())
        fn = int(((1 - yp) & y_true).sum())
        tn = int(((1 - yp) & (1 - y_true)).sum())
        acc = (tp + tn) / max(len(y_true), 1)
        prec = tp / max(tp + fp, 1) if (tp + fp) else 0.0
        rec  = tp / max(tp + fn, 1) if (tp + fn) else 0.0
        f1   = (2 * prec * rec / max(prec + rec, 1e-12)) if (prec + rec) else 0.0
        rows.append({"threshold": float(np.round(th, 3)),
                     "accuracy": float(np.round(acc, 3)),
                     "precision": float(np.round(prec, 3)),
                     "recall": float(np.round(rec, 3)),
                     "f1": float(np.round(f1, 3))})
    return pd.DataFrame(rows)


def _pick_threshold(val_preds: Optional[pd.DataFrame]) -> float:
    """Operating threshold: F1-optimal on Val if available, else 0.50."""
    if val_preds is None or val_preds.empty:
        return 0.50
    yv = val_preds["y_true"].to_numpy()
    pv = val_preds["prob"].to_numpy()
    sweep = _threshold_sweep(yv, pv, np.linspace(0.0, 1.0, 201))
    best = sweep[sweep["f1"] == sweep["f1"].max()].sort_values("threshold").iloc[-1]
    return float(best["threshold"])


# --- 5) Plotting (dual theme; no boxplots) ---------------------
@plot_dual_theme(section="fairness")
def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auroc: float, ax=None, **kwargs):
    ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.6)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC (AUROC={auroc:.3f})"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

@plot_dual_theme(section="fairness")
def plot_pr(rec: np.ndarray, prec: np.ndarray, ap: float, ax=None, **kwargs):
    ax.plot(rec, prec)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall (AP={ap:.3f})"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

@plot_dual_theme(section="fairness")
def plot_threshold_metrics(df_sweep: pd.DataFrame, ax=None, **kwargs):
    for col in ["accuracy", "precision", "recall", "f1"]:
        ax.plot(df_sweep["threshold"], df_sweep[col], label=col)
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Metrics vs Threshold"); ax.legend(loc="lower left")

@plot_dual_theme(section="fairness")
def plot_fairness_curve(thresholds: np.ndarray, eod: np.ndarray, ax=None, **kwargs):
    ax.plot(thresholds, eod)
    ax.set_xlabel("Threshold"); ax.set_ylabel("Max |EOD|")
    ax.set_title("Fairness vs Threshold (Equal Opportunity Difference)")

@plot_dual_theme(section="fairness")
def plot_confusion(cm: np.ndarray, ax=None, **kwargs):
    """
    Readable confusion matrix with auto-contrast annotations.
    Drop-in replacement; palette uses Matplotlib 'cividis'.
    """
    # Heatmap
    im = ax.imshow(cm, cmap="cividis")

    # Auto-contrast text color by cell intensity
    vmax = float(cm.max()) if np.isfinite(cm.max()) else 0.0
    cutoff = vmax * 0.5
    for (i, j), v in np.ndenumerate(cm):
        color = "white" if v >= cutoff else "black"
        ax.text(j, i, f"{int(v):,}", ha="center", va="center", color=color, fontsize=12)

    # Ticks, labels
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title("Confusion Matrix (operating)")

    # Thin separators (theme-aware)
    spine_color = ax.spines['left'].get_edgecolor()
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color=spine_color, linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)


# --- 6) Fairness helpers ------------------------------------------------------
def _fairness_curve(df: pd.DataFrame, thresholds: np.ndarray, privileged: str = "White Women") -> pd.DataFrame:
    """
    Compute max absolute EOD across groups for each threshold.
    Returns NaNs (single line) if fewer than 2 groups overall to avoid spam.
    """
    groups_unique = df["Group"].dropna().astype(str).nunique()
    if groups_unique < 2:
        return pd.DataFrame({
            "threshold": thresholds,
            "max_abs_eod": np.full_like(thresholds, np.nan, dtype=float)
        })

    y = df["y_true"].to_numpy()
    p = df["prob"].to_numpy()

    rows = []
    for th in thresholds:
        yp = (p >= th).astype(int)
        tmp = df[["Group"]].copy()
        tmp["y_true"] = y
        tmp["y_pred"] = yp
        gm = calculate_group_metrics(tmp)  # uses 'Group' if present
        disp = calculate_fairness_disparities(gm, privileged=privileged)
        max_abs = float(np.abs(disp["Equal Opportunity Difference"]).max()) if not disp.empty else np.nan
        rows.append({"threshold": float(np.round(th,3)), "max_abs_eod": np.round(max_abs, 3)})
    return pd.DataFrame(rows)

def _load_label_corr_matrix(path: Path) -> np.ndarray:
    """
    Expect a symmetric correlation/co-occurrence matrix CSV with columns/rows in the same order.
    Returns a normalized correlation matrix suitable for cadp_score.
    """
    M = pd.read_csv(path, index_col=0)
    # Normalize to correlation-like range if needed
    M = M / (np.maximum(1.0, np.abs(M.values).max()))
    return M.to_numpy()


def _cadp_at_threshold(df: pd.DataFrame, threshold: float, corr: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Correlation-Adjusted Demographic Parity (CADP) at a fixed threshold.
    Falls back to Demographic Parity if cadp_score is unavailable.
    """
    # Operate at threshold
    tmp = df[["Group"]].copy()
    tmp["y_pred"] = (df["prob"].to_numpy() >= threshold).astype(int)

    # Use provided correlation (multi-label); else identity (single-label Step-08)
    C = corr if corr is not None else np.eye(1)

    # --- clean fallback if CADP is missing ---
    if cadp_score is None:
        # No warnings spam; just compute DP once.
        rates = (tmp.groupby("Group")["y_pred"].mean().rename("AdjustedRate")).reset_index()
        priv = "White Women"
        priv_rate = float(rates.loc[rates["Group"] == priv, "AdjustedRate"].iloc[0]) if (rates["Group"] == priv).any() else np.nan
        rates["Gap_vs_Priv"] = rates["AdjustedRate"] - priv_rate
        rates["threshold"] = float(np.round(threshold, 3))
        return rates

    # --- CADP path ---
    try:
        res = cadp_score(
            y_or_pred=tmp["y_pred"].to_numpy(),
            groups=tmp["Group"].astype(str).to_numpy(),
            corr=C,
            eps=1e-9,
            delta=0.0,
            return_by_group=True
        )
        cadp_df = pd.DataFrame(res)
        if "Gap_vs_Priv" not in cadp_df.columns:
            priv = "White Women"
            rr = cadp_df.set_index("Group")["AdjustedRate"]
            cadp_df["Gap_vs_Priv"] = cadp_df["Group"].map(lambda g: float(rr.get(g, np.nan) - rr.get(priv, np.nan)))
        cadp_df["threshold"] = float(np.round(threshold, 3))
        return cadp_df[["Group", "AdjustedRate", "Gap_vs_Priv", "threshold"]]
    except Exception as e:
        print(f"⚠ CADP call failed; falling back to DP: {e}")
        rates = (tmp.groupby("Group")["y_pred"].mean().rename("AdjustedRate")).reset_index()
        priv = "White Women"
        priv_rate = float(rates.loc[rates["Group"] == priv, "AdjustedRate"].iloc[0]) if (rates["Group"] == priv).any() else np.nan
        rates["Gap_vs_Priv"] = rates["AdjustedRate"] - priv_rate
        rates["threshold"] = float(np.round(threshold, 3))
        return rates



# --- 7) Main ------------------------------------------------------------------
def main(argv: Optional[list] = None) -> None:
    """
    Comprehensive evaluation from prediction files.

    CLI
    ---
    # Full run (uses Step-07 CSVs by default)
    python -m src.fairness.08_comprehensive_evaluation

    # Self-check (random sample; non-destructive)
    python -m src.fairness.08_comprehensive_evaluation --selfcheck --sample 80000
    """
    import argparse

    t_all = time.perf_counter()
    print("--- Starting Step 08: Comprehensive Evaluation ---")
    print("[NOTE] Titles may be non-English; tags/categories help anchor semantics (multi-label upstream; single-label target here).")

    p = argparse.ArgumentParser()
    p.add_argument("--val-csv", type=str, default=str(VAL_PREDS_CSV_DEFAULT))
    p.add_argument("--test-csv", type=str, default=str(TEST_PREDS_CSV_DEFAULT))
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--use-label-corr", action="store_true",
                   help="Use label correlation from outputs/data/18_category_cooccurrence.csv (multi-label runs).")
    args = p.parse_args(argv)
    corr_mat = None
    if args.use_label_corr:
        corr_mat = _load_label_corr_matrix(DATA_DIR / "18_category_cooccurrence.csv")
        print("[CADP] Using label correlation from 18_category_cooccurrence.csv")


    # Read predictions
    t0 = _t0("Loading predictions ...")
    val_df = None
    try:
        if Path(args.val_csv).exists():
            val_df = _read_preds_or_die(Path(args.val_csv))
    except FileNotFoundError:
        val_df = None
    test_df = _read_preds_or_die(Path(args.test_csv))
    _tend("eval.load_preds", t0)

    # Optional self-check sampling (non-destructive)
    suffix = ""
    if args.selfcheck:
        n = args.sample or min(80_000, len(test_df))
        test_df = test_df.sample(n=n, random_state=SEED, replace=False).reset_index(drop=True)
        suffix = "_selfcheck"
        print(f"[SELF-CHECK] Random sample drawn from Test: {len(test_df):,} rows (seed={SEED}).")

    # If groups are degenerate (or missing), repair by joining corpus
    test_df = _repair_groups_if_needed(test_df)

    # Curves on Test
    t0 = _t0("Computing global curves on Test ...")
    y_te = test_df["y_true"].to_numpy()
    p_te = test_df["prob"].to_numpy()
    auroc, ap, fpr, tpr, prec, rec = _global_curves(y_te, p_te)
    _tend("eval.global_curves", t0)

    # Threshold selection (Val if available)
    t0 = _t0("Selecting operating threshold ...")
    th = _pick_threshold(val_df)
    print(f"[SELECTION] Operating threshold = {th:.2f} (F1-optimal on Val if available; else 0.50)")
    _tend("eval.select_threshold", t0)

    # Threshold sweep (Test)
    t0 = _t0("Sweeping thresholds on Test ...")
    thresholds = np.linspace(0.0, 1.0, 201)
    sweep = _threshold_sweep(y_te, p_te, thresholds)
    _tend("eval.threshold_sweep", t0)

    # Operating point metrics (Test)
    ypt = (p_te >= th).astype(int)

    # Group metrics + disparities at operating point
    t0 = _t0("Computing per-group metrics & disparities at operating point ...")
    op_frame = test_df[["video_id","title","Group"]].copy()
    op_frame["y_true"] = y_te
    op_frame["y_pred"] = ypt
    gm = calculate_group_metrics(op_frame)  # uses 'Group' by default
    disp = calculate_fairness_disparities(gm, privileged="White Women")
    _tend("eval.group_metrics", t0)
    # --- CADP at operating threshold -----------------------------------
    t0 = _t0("Computing CADP at operating threshold ...")
    cadp_at_opt = _cadp_at_threshold(test_df[["Group","y_true","prob"]].copy(), threshold=th, corr=corr_mat)
    _tend("eval.cadp_at_opt", t0)


    # Fairness vs threshold
    t0 = _t0("Computing fairness curve (EOD vs threshold) ...")
    fair_curve = _fairness_curve(test_df, thresholds, privileged="White Women")
    _tend("eval.fairness_curve", t0)
    
    # CADP sweep vs threshold (DP when single label)
    t0 = _t0("Sweeping CADP vs threshold ...")
    cadp_rows = []
    for th_s in thresholds:
        cadp_rows.append(_cadp_at_threshold(test_df[["Group","y_true","prob"]], threshold=float(th_s), corr=corr_mat))
    cadp_curve = pd.concat(cadp_rows, ignore_index=True)
    _tend("eval.cadp_sweep", t0)


    # Confusion matrix at operating point
    t0 = _t0("Building confusion matrix ...")
    cm = confusion_matrix(y_te, ypt, labels=[0,1])
    cm_df = pd.DataFrame(cm, index=["True_0","True_1"], columns=["Pred_0","Pred_1"])
    _tend("eval.confusion", t0)

    # Save artefacts (with suffix if selfcheck)
    t0 = _t0("Saving artefacts ...")
    for pth in [DATA_DIR, FIG_DIR, NARR_DIR, TABLES_DIR]:
        pth.mkdir(parents=True, exist_ok=True)

    sweep.to_csv(_with_suffix(TH_SWEEP_CSV, suffix), index=False)
    fair_curve.to_csv(_with_suffix(FAIRNESS_CURVE_CSV, suffix), index=False)
    gm.to_csv(_with_suffix(GROUP_METRICS_AT_OPT, suffix), index=False)
    disp.to_csv(_with_suffix(DISPARITIES_AT_OPT, suffix), index=False)
    cm_df.to_csv(_with_suffix(CONFUSION_AT_OPT_CSV, suffix))
    cadp_curve.to_csv(_with_suffix(CADP_SWEEP_CSV, suffix), index=False)
    cadp_at_opt.to_csv(_with_suffix(CADP_AT_OPT_CSV, suffix), index=False)



    # Plots (dual-theme)
    plot_roc(fpr=fpr, tpr=tpr, auroc=auroc,
             save_path=str(FIG_DIR / f"08_roc{suffix}"), figsize=(7,5))
    plot_pr(rec=rec, prec=prec, ap=ap,
            save_path=str(FIG_DIR / f"08_pr{suffix}"), figsize=(7,5))
    plot_threshold_metrics(df_sweep=sweep,
            save_path=str(FIG_DIR / f"08_threshold_metrics{suffix}"), figsize=(8,5))
    plot_fairness_curve(thresholds=sweep["threshold"].to_numpy(),
            eod=fair_curve["max_abs_eod"].to_numpy(),
            save_path=str(FIG_DIR / f"08_fairness_curve{suffix}"), figsize=(8,5))
    plot_confusion(cm=cm,
            save_path=str(FIG_DIR / f"08_confusion{suffix}"), figsize=(5,4))
    _tend("eval.save_artefacts", t0)

    # LaTeX tables
    t0 = _t0("Generating LaTeX tables ...")
    dataframe_to_latex_table(
        df=sweep.set_index("threshold"),
        save_path=str(_with_suffix(TABLES_DIR / "08_threshold_sweep.tex", suffix)),
        caption="Threshold sweep on Test (accuracy, precision, recall, F1).",
        label=f"tab:08-threshold-sweep{'-selfcheck' if suffix else ''}",
        note="Scores rounded to 3 decimals."
    )
    dataframe_to_latex_table(
        df=gm.set_index("Group"),
        save_path=str(_with_suffix(TABLES_DIR / "08_group_metrics_at_opt.tex", suffix)),
        caption="Per-group metrics at operating threshold on Test.",
        label=f"tab:08-group-metrics{'-selfcheck' if suffix else ''}",
        note="Group sizes can be imbalanced."
    )
    if not disp.empty:
        dataframe_to_latex_table(
            df=disp.set_index("Comparison Group"),
            save_path=str(_with_suffix(TABLES_DIR / "08_disparities_at_opt.tex", suffix)),
            caption="Disparities vs. White Women at operating threshold.",
            label=f"tab:08-disparities{'-selfcheck' if suffix else ''}",
            note="Equal Opportunity Difference is Recall(priv) − Recall(group)."
        )
    _tend("academic_tables.dataframe_to_latex_table runtime", t0)

    # Narrative (qualitative + outliers)
    t0 = _t0("Writing narrative ...")
    outs = top_confident_outliers(
        df_meta=test_df.assign(y_pred=ypt),
        probs=p_te,
        k=10
    )
    lines = []
    lines.append("# Automated Summary: Comprehensive Evaluation\n")
    lines.append("Some titles are non-English; tags/categories help anchor semantics (MPU). Totals in other steps can exceed N due to multi-label; here counts are per-ID.\n")
    lines.append(f"**Operating threshold**: {th:.2f} (Val-optimal F1 when available; else 0.50).\n")
    lines.append(f"**Test AUROC**: {auroc:.3f} | **AP**: {ap:.3f}\n")
    if not disp.empty:
        worst = disp.iloc[np.argmax(np.abs(disp["Equal Opportunity Difference"].to_numpy()))]
        lines.append(f"**Largest |EOD| at operating point**: {abs(worst['Equal Opportunity Difference']):.3f} ({worst['Comparison Group']})\n")
    lines.append("\n## Fairness curve (EOD vs threshold)\nLower is better; seek a flat region with acceptable utility.\n")
    lines.append("\n## Top 10 confident errors (Test)\n")
    lines.append(outs.to_string(index=False))
    with open(_with_suffix(NARRATIVE_PATH, suffix), "w") as f:
        f.write("\n".join(lines))
    _tend("eval.narrative", t0)

    _tend("eval.step08_total", t_all)
    print("\n--- Step 08: Comprehensive Evaluation Completed Successfully ---")


if __name__ == "__main__":
    main()
