# -*- coding: utf-8 -*-
"""
07a_category_sweep.py
=====================

Purpose
-------
Compare candidate positive classes for the baseline task by training a small RF
model per category (same features, separate heads) and reporting metrics. Uses
a shared text+numeric preprocessor for speed; each category trains its own RF.

Enhancements in this version
----------------------------
1) Decision threshold is **tuned on the validation set** (grid 0.1–0.9) to
   maximize F1 for each category, then **applied to the test set**.
2) Plotting is robust: assigns `hue`, ensures the palette has ≥ the number of
   bars → no seaborn FutureWarnings/UserWarnings.

Outputs (non-destructive)
-------------------------
- outputs/data/07a_category_sweep_results.csv
- outputs/narratives/automated/07a_category_sweep_summary.md
- outputs/figures/models/07a_category_sweep_{light,dark}.png

CLI
---
python -m src.models.07a_category_sweep
python -m src.models.07a_category_sweep --n-estimators 300 --max-depth 28
python -m src.models.07a_category_sweep --categories "Amateur,Big Tits,Anal,Asian,Blonde"
"""

from __future__ import annotations

# --- Imports (keep at top) ----------------------------------------------------
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Project utils
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme

# --- 1) Config & paths --------------------------------------------------------
CONFIG = load_config()
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))

DATA_DIR = Path(CONFIG["paths"]["data"])
OUT_DIR  = Path(CONFIG["paths"]["outputs"])
FIG_DIR  = Path(CONFIG["paths"]["figures"])
FIG_MODELS_DIR = FIG_DIR / "models"
NARR_DIR = Path(CONFIG["paths"]["narratives"]) / "automated"

CORPUS_PATH = DATA_DIR / "01_ml_corpus.parquet"
TRAIN_IDS   = DATA_DIR / "06_train_ids.csv"
VAL_IDS     = DATA_DIR / "06_val_ids.csv"
TEST_IDS    = DATA_DIR / "06_test_ids.csv"

COLCFG   = CONFIG.get("columns", {})
TEXT_COL = COLCFG.get("text", "model_input_text")
CATS_COL = COLCFG.get("categories", "categories")
NUM_COLS = COLCFG.get("numeric_rf", ["duration", "ratings"])

DEFAULT_CATEGORIES: List[str] = [
    "Amateur", "Big Tits", "Anal", "Asian",
    "Blonde", "Blowjob", "Masturbation", "Creampie",
    "Fetish", "Cumshot", "Feet", "Brunette",
]

# --- 2) Timers ----------------------------------------------------------------
def _t0(msg: str) -> float:
    t = time.perf_counter()
    print(msg)
    return t

def _tend(label: str, t0: float) -> None:
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")


# --- 3) Data helpers ----------------------------------------------------------
def _coerce_types(d: pd.DataFrame) -> pd.DataFrame:
    """
    Light coercions aligned to Step-07.
    """
    out = d.copy()
    if TEXT_COL not in out.columns:
        out[TEXT_COL] = (out.get("title", "").fillna("").astype(str) + " " +
                         out.get("tags", "").fillna("").astype(str))
    for c in NUM_COLS:
        if c not in out.columns:
            out[c] = 0
    if "ratings" in out.columns:
        out["ratings"] = pd.to_numeric(out["ratings"], errors="coerce").round(0).astype("Int64")
    if "duration" in out.columns and out["duration"].dtype == "object":
        def _to_sec(x):
            s = str(x)
            if s.isdigit():
                return float(s)
            parts = s.split(":")
            try:
                if len(parts) == 2:
                    m, sec = int(parts[0]), float(parts[1]); return m * 60 + sec
                if len(parts) == 3:
                    h, m, sec = int(parts[0]), int(parts[1]), float(parts[2]); return h * 3600 + m * 60 + sec
            except Exception:
                return np.nan
            return np.nan
        out["duration"] = out["duration"].map(_to_sec)
    if "duration" in out.columns:
        out["duration"] = pd.to_numeric(out["duration"], errors="coerce")
    if CATS_COL not in out.columns:
        out[CATS_COL] = ""
    return out


def _load_corpus() -> pd.DataFrame:
    t0 = _t0(f"[READ] Parquet: {CORPUS_PATH}")
    df = pd.read_parquet(CORPUS_PATH)
    _tend("cat.load_corpus", t0)
    return _coerce_types(df)


def _read_ids_or_none(path: Path) -> Optional[pd.Index]:
    if path.exists():
        ids = pd.read_csv(path)["video_id"]
        return pd.Index(ids)
    return None


def _subset_by_ids(df: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    return df[df["video_id"].isin(ids)].reset_index(drop=True)


def _primary_category(df: pd.DataFrame) -> pd.Series:
    return df[CATS_COL].fillna("").astype(str).str.split(",").str[0].str.strip()


# --- 4) Features --------------------------------------------------------------
def build_preprocessor(seed: int,
                       n_text_components: int = 256,
                       n_hash_features: int = 2**18) -> ColumnTransformer:
    """
    Shared text+numeric preprocessor:
      TEXT_COL -> HashingVectorizer (CSR) -> TruncatedSVD (dense)
      NUM_COLS -> passthrough
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


def _fit_transform_shared(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Fit the shared preprocessor on Train; transform Val/Test. Returns dense arrays.
    """
    pre = build_preprocessor(seed=seed)
    t0 = _t0("Fitting shared text+numeric preprocessor on Train ...")
    Xtr = pre.fit_transform(df_train[[TEXT_COL] + NUM_COLS])
    _tend("cat.prep_fit", t0)

    t0 = _t0("Transforming Val/Test features ...")
    Xva = pre.transform(df_val[[TEXT_COL] + NUM_COLS])
    Xte = pre.transform(df_test[[TEXT_COL] + NUM_COLS])
    _tend("cat.prep_transform", t0)
    return Xtr, Xva, Xte, pre


# --- 5) Threshold & metrics ---------------------------------------------------
def _best_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    """
    Threshold in [0.1, 0.9] that maximizes F1 on y_true. Falls back to 0.5 when degenerate.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    grid = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y_true, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def _compute_metrics(y_true: np.ndarray, p: np.ndarray, *, threshold: float) -> Dict[str, float]:
    """
    Compute Acc/Prec/Rec/F1 (at threshold) and AUROC (threshold-free).
    """
    yhat = (p >= threshold).astype(int)
    out = {
        "Accuracy":  accuracy_score(y_true, yhat),
        "Precision": precision_score(y_true, yhat, zero_division=0),
        "Recall":    recall_score(y_true, yhat, zero_division=0),
        "F1":        f1_score(y_true, yhat, zero_division=0),
        "AUROC":     roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else np.nan,
    }
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


# --- 6) Plotting --------------------------------------------------------------
@plot_dual_theme(section="fairness")
def _plot_results_bar(df: pd.DataFrame, metric: str, title: str, ax=None, palette=None, **kwargs):
    """
    Horizontal bar plot of a metric across categories.

    Fixes:
    - Assigns hue='Category' when a palette is passed (FutureWarning fix).
    - Ensures palette length >= #bars to avoid the cycling warning.
    """
    import seaborn as sns

    d = df.sort_values(metric, ascending=True).copy()
    n = len(d)

    try:
        pal = (palette if (palette and len(palette) >= n) else sns.color_palette("magma", n_colors=n))
        pal = pal[:n]
    except Exception:
        pal = sns.color_palette("magma", n_colors=n)

    sns.barplot(
        y="Category", x=metric, data=d, ax=ax,
        hue="Category", palette=pal, orient="h",
        dodge=False, legend=False
    )
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel("Category")


# --- 7) Core sweep ------------------------------------------------------------
def _labels_for_category(primary: pd.Series, positive_class: str) -> np.ndarray:
    return (primary == positive_class).astype(int).to_numpy()


def run_sweep(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
    categories: Iterable[str], *, n_estimators: int, max_depth: Optional[int]
) -> pd.DataFrame:
    """
    Train + evaluate an RF per category using a shared preprocessor. The decision
    threshold is tuned on Val to maximize F1 and then used on Test.
    """
    # Prepare features once
    Xtr, Xva, Xte, _pre = _fit_transform_shared(df_train, df_val, df_test, seed=SEED)

    # Primary categories (target source)
    cat_train = _primary_category(df_train)
    cat_val   = _primary_category(df_val)
    cat_test  = _primary_category(df_test)

    rows = []

    for cat in categories:
        print(f"\n[CAT] Evaluating positive class: '{cat}'")
        y_tr = _labels_for_category(cat_train, cat)
        y_va = _labels_for_category(cat_val, cat)
        y_te = _labels_for_category(cat_test, cat)

        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=SEED, n_jobs=-1
        )

        t0 = _t0("  Training RF ...")
        clf.fit(Xtr, y_tr)
        _tend("cat.train_rf", t0)

        pv = clf.predict_proba(Xva)[:, 1]
        pt = clf.predict_proba(Xte)[:, 1]

        thr = _best_threshold(y_va, pv)
        mv  = _compute_metrics(y_va, pv, threshold=thr)
        mt  = _compute_metrics(y_te, pt, threshold=thr)

        rows.append({
            "Category": cat,
            "TrainPos": int(y_tr.sum()), "TrainN": len(y_tr),
            "ValPos":   int(y_va.sum()), "ValN": len(y_va),
            "TestPos":  int(y_te.sum()), "TestN": len(y_te),
            "ValAcc": mv["Accuracy"], "ValPrec": mv["Precision"], "ValRec": mv["Recall"],
            "ValF1": mv["F1"], "ValAUROC": mv["AUROC"],
            "TestAcc": mt["Accuracy"], "TestPrec": mt["Precision"], "TestRec": mt["Recall"],
            "TestF1": mt["F1"], "TestAUROC": mt["AUROC"],
            "PrevalenceTrain": round(y_tr.mean(), 6),
            "DecisionThreshold": thr,
        })

    return pd.DataFrame(rows)


# --- 8) Save artefacts --------------------------------------------------------
def _save_results_table(df: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "07a_category_sweep_results.csv"
    df.to_csv(out, index=False)
    return out


def _save_narrative(df: pd.DataFrame) -> Path:
    NARR_DIR.mkdir(parents=True, exist_ok=True)
    d = df.sort_values("TestF1", ascending=False)
    top = d.head(5)

    lines = []
    lines.append("# Automated Summary: Category Sweep (RF with tuned thresholds)\n")
    lines.append("Validation thresholds were tuned per category to maximize F1 and then applied to Test.\n")
    lines.append("## Top 5 by Test F1\n")
    lines.append(top[["Category", "TestF1", "TestAUROC", "DecisionThreshold"]].to_string(index=False))
    lines.append("\n\nFull table saved as `07a_category_sweep_results.csv`.\n")
    out = NARR_DIR / "07a_category_sweep_summary.md"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    return out


# --- 9) Main ------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    t_all = time.perf_counter()
    print("--- Starting 07a: Category Sweep (RF) ---")

    p = argparse.ArgumentParser()
    p.add_argument("--categories", type=str, default=",".join(DEFAULT_CATEGORIES),
                   help="Comma-separated list of categories to try (primary category match).")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=None)
    args = p.parse_args(argv)

    # Load corpus and Step-06 splits (fallback to internal stratified split)
    df = _load_corpus()
    print(f"[STATS] Total videos available: {len(df):,}")

    tr_ids = _read_ids_or_none(TRAIN_IDS)
    va_ids = _read_ids_or_none(VAL_IDS)
    te_ids = _read_ids_or_none(TEST_IDS)

    if tr_ids is None or va_ids is None or te_ids is None:
        print("✗ Step-06 IDs not found; using internal 60/20/20 split stratified by primary category.")
        prim = _primary_category(df)
        trv, te = train_test_split(df, test_size=0.20, random_state=SEED, stratify=prim)
        tr, va  = train_test_split(trv, test_size=0.25, random_state=SEED, stratify=_primary_category(trv))
        train_df, val_df, test_df = tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)
    else:
        train_df = _subset_by_ids(df, tr_ids)
        val_df   = _subset_by_ids(df, va_ids)
        test_df  = _subset_by_ids(df, te_ids)

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    results = run_sweep(
        train_df, val_df, test_df, cats,
        n_estimators=args.n_estimators, max_depth=args.max_depth
    )

    # Save artefacts
    table_path = _save_results_table(results)
    narr_path  = _save_narrative(results)

    # Plot (Test F1)
    _plot_results_bar(
        df=results, metric="TestF1",
        title="Category Sweep — Test F1",
        save_path=str(FIG_MODELS_DIR / "07a_category_sweep"),
        figsize=(10, 8),
    )

    print("\n✓ Artefacts saved:",
          table_path.name, ",", narr_path.name, ",",
          "and figures/models/07a_category_sweep_*.png")
    _tend("cat.sweep_total", t_all)
    print("\n--- 07a: Category Sweep Completed Successfully ---")


if __name__ == "__main__":
    main()
