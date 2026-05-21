# -*- coding: utf-8 -*-
"""
09_bert_baseline.py
===================

Purpose
-------
Train and evaluate a DistilBERT text baseline for the canonical binary task
(default: 'Amateur' vs not). Outputs and file names align with the RF baseline
for downstream comparability. Adds optional GOLD evaluation.

What it does
------------
1) Loads canonical parquet + Step-06 splits (or makes an internal split in
   self-check mode), enforces required columns, derives the binary target.
2) Builds a Hugging Face Transformers pipeline:
      tokenizer ⟶ (Distil)BERT ⟶ classification head
3) Trains briefly (configurable), evaluates overall + intersectional metrics,
   computes disparities with robust fallback, and surfaces qualitative outliers.
4) Saves metrics, predictions, model, a dual-theme margin plot, and a narrative.
5) If GOLD annotations exist, evaluates the fitted model on GOLD (non-destructive).
6) Self-check uses a random sample and writes *_selfcheck artefacts only.

Interpretability note (MPU)
---------------------------
Some titles are not English; tags/categories often anchor semantics. We include
titles and groups in the outliers for qualitative inspection.

CLI
---
# Full run (canonical artefacts):
python -m src.models.09_bert_baseline

# Self-check (small sample; non-destructive):
python -m src.models.09_bert_baseline --selfcheck --sample 16000 --epochs 1 --model prajjwal1/bert-tiny

# With GOLD (auto if file exists at default path; override with --gold-path):
python -m src.models.09_bert_baseline --gold-path outputs/data/gold/gold_final.csv
"""

from __future__ import annotations

# --- Imports (keep at top) ----------------------------------------------------
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

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

DATA_DIR       = Path(CONFIG["paths"]["data"])
OUT_DIR        = Path(CONFIG["paths"]["outputs"])
FIG_DIR        = Path(CONFIG["paths"]["figures"])
FIG_MODELS_DIR = FIG_DIR / "models"
MODEL_DIR      = OUT_DIR / "models" / "09_bert"     # numbered model dir
NARR_DIR       = Path(CONFIG["paths"]["narratives"]) / "automated"
GOLD_DIR       = DATA_DIR / "gold"

# Inputs (from Step 01 & 06)
CORPUS_PATH = DATA_DIR / "01_ml_corpus.parquet"
TRAIN_IDS   = DATA_DIR / "06_train_ids.csv"
VAL_IDS     = DATA_DIR / "06_val_ids.csv"
TEST_IDS    = DATA_DIR / "06_test_ids.csv"

# Outputs (all start with 09_*)
OVERALL_METRICS_CSV   = DATA_DIR / "09_overall_metrics.csv"
VAL_METRICS_CSV       = DATA_DIR / "09_bert_val_metrics.csv"
VAL_PREDICTIONS_CSV   = DATA_DIR / "09_bert_val_predictions.csv"
TEST_METRICS_CSV      = DATA_DIR / "09_bert_test_metrics.csv"
TEST_PREDICTIONS_CSV  = DATA_DIR / "09_bert_test_predictions.csv"
GROUP_METRICS_CSV     = DATA_DIR / "09_fairness_group_metrics.csv"
DISPARITIES_CSV       = DATA_DIR / "09_fairness_disparities.csv"
NARRATIVE_PATH        = NARR_DIR / "09_bert_baseline_summary.md"

# GOLD (numbered outputs, shared input path)
DEFAULT_GOLD_PATH     = GOLD_DIR / "gold_final.csv"
GOLD_METRICS_CSV      = DATA_DIR / "09_bert_gold_eval_metrics.csv"
GOLD_PREDICTIONS_CSV  = DATA_DIR / "09_bert_gold_eval_predictions.csv"

# Task columns
TEXT_COL = "model_input_text"
CATS_COL = "categories"
POSITIVE_CLASS = "Amateur"  # positive class is Amateur (consistent with RF)

# Model default (overridable via CLI or settings)
DEFAULT_MODEL_NAME = (
    CONFIG.get("models", {})
          .get("bert", {})
          .get("model_name", "distilbert-base-uncased")
)

# --- 2) Lightweight timers ----------------------------------------------------
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
    Ensure required columns exist. If TEXT_COL missing, fallback to title + tags.

    This accounts for multilingual titles; tags/categories frequently ground the semantics.
    """
    d = df.copy()
    if TEXT_COL not in d.columns:
        d[TEXT_COL] = (d.get("title", "").fillna("").astype(str) + " " +
                       d.get("tags", "").fillna("").astype(str))
    if CATS_COL not in d.columns:
        d[CATS_COL] = ""
    return d

def _load_corpus() -> pd.DataFrame:
    """
    Load canonical parquet and enforce required columns.

    Returns
    -------
    pd.DataFrame
        Canonical corpus with TEXT_COL present.
    """
    t0 = _t0(f"[READ] Parquet: {CORPUS_PATH}")
    df = pd.read_parquet(CORPUS_PATH)
    _tend("bert.load_corpus", t0)
    return _ensure_columns(df)

def _read_ids_or_none(path: Path) -> Optional[pd.Index]:
    """
    Read a single-column CSV of video IDs; return Index or None.
    """
    if path.exists():
        ids = pd.read_csv(path)["video_id"]
        return pd.Index(ids)
    return None

def _subset_by_ids(df: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    """
    Subset dataframe by IDs, preserving the original row order.
    """
    return df[df["video_id"].isin(ids)].reset_index(drop=True)

def _make_internal_split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a 60/20/20 internal split for self-check, stratified on intersectional key.

    Returns
    -------
    (train_df, val_df, test_df)
        Three dataframes, stratified by a simple priority intersectional scheme.
    """
    from sklearn.model_selection import train_test_split

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
    """
    primary = df[CATS_COL].fillna("").astype(str).str.split(",").str[0].str.strip()
    return (primary == positive_class).astype(int)

# --- 4) HF dataset ------------------------------------------------------------
class TextBinDataset(Dataset):
    """
    Lightweight Dataset for text binary classification with Hugging Face.

    Notes
    -----
    - Expects columns: TEXT_COL and 'label' (int).
    - Metadata (video_id/title) stays in ds.df; do NOT return strings here.
    """
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int = 128):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return ONLY model inputs and labels to keep the collator happy.
        """
        row = self.df.iloc[idx]
        enc = self.tok(
            str(row[TEXT_COL]),
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        enc["labels"] = int(row["label"])
        # strictly supported types for collator
        return {k: torch.tensor(v) if isinstance(v, list) else v for k, v in enc.items()}

# --- 5) Plotting --------------------------------------------------------------
@plot_dual_theme(section="fairness")
def _plot_margins(margins: np.ndarray, title: str, ax=None, palette=None, **kwargs):
    """Histogram of decision margins (p - 0.5). Uses project-wide dual theme."""
    ax.hist(margins, bins=50)
    ax.set_title(title)
    ax.set_xlabel("Decision margin (p - 0.5)")
    ax.set_ylabel("Count")

# --- 6) Train + evaluate ------------------------------------------------------
@dataclass
class RunBundle:
    """Container for all artefacts produced by a training run."""
    model_dir: Path
    val_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    val_preds: pd.DataFrame
    test_preds: pd.DataFrame
    group_metrics_test: pd.DataFrame
    disparities_test: pd.DataFrame
    outliers_test: pd.DataFrame

def _build_model_and_tokenizer(model_name: str, num_labels: int = 2):
    """
    Initialize tokenizer and model (num_labels=2). Returns (tokenizer, model).
    """
    t0 = _t0(f"[HF] Load tokenizer & model: {model_name}")
    cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
    _tend("bert.load_hf_components", t0)
    return tok, mdl

def _build_frame_and_datasets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    tokenizer,
    max_len: int = 128
) -> Tuple[TextBinDataset, TextBinDataset, TextBinDataset]:
    """
    Prepare HF datasets from train/val/test frames. Returns (ds_train, ds_val, ds_test).
    """
    for d in (df_train, df_val, df_test):
        d["label"] = _prepare_binary_target(d).astype(int).values
    return (
        TextBinDataset(df_train, tokenizer, max_len=max_len),
        TextBinDataset(df_val, tokenizer, max_len=max_len),
        TextBinDataset(df_test, tokenizer, max_len=max_len),
    )

def _make_training_args(out_dir: Path, *, epochs: int, batch_size: int, lr: float, seed: int) -> TrainingArguments:
    """
    Construct TrainingArguments in a version-robust way with graceful fallbacks.
    """
    import torch as _torch
    _use_mps = _torch.backends.mps.is_available() and not _torch.cuda.is_available()
    base = dict(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        seed=seed,
        logging_steps=200,
    )
    # Build kwargs defensively — drop any param the installed version doesn't accept
    optional = {
        "eval_strategy": "epoch",
        "save_strategy": "no",
        "report_to": [],
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "use_mps_device": _use_mps,
    }
    import inspect
    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    # eval_strategy renamed from evaluation_strategy in newer versions
    if "eval_strategy" not in valid and "evaluation_strategy" in valid:
        optional["evaluation_strategy"] = optional.pop("eval_strategy")
    elif "eval_strategy" not in valid:
        optional.pop("eval_strategy", None)
    optional = {k: v for k, v in optional.items() if k in valid}
    return TrainingArguments(**base, **optional)

def _trainer(
    model,
    tokenizer,
    ds_train: TextBinDataset,
    ds_val: TextBinDataset,
    *,
    out_dir: Path,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
    seed: int = SEED
) -> Trainer:
    """
    Build a Hugging Face Trainer (version-robust) with eval hooks when supported.
    """
    args = _make_training_args(out_dir=out_dir, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    try:
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
    except Exception:
        try:
            from transformers import default_data_collator as _default_collator  # type: ignore
            collator = _default_collator
        except Exception:
            def collator(features):
                keep = []
                for f in features:
                    keep.append({k: v for k, v in f.items() if k in ("input_ids","attention_mask","token_type_ids","labels")})
                return tokenizer.pad(keep, return_tensors="pt")
    # 'tokenizer' was renamed to 'processing_class' in transformers ≥ 4.46
    import inspect as _ins
    t_params = set(_ins.signature(Trainer.__init__).parameters)
    tok_kwarg = "processing_class" if "processing_class" in t_params else "tokenizer"
    return Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        **{tok_kwarg: tokenizer},
    )

def _predict_proba(trainer: Trainer, ds: TextBinDataset) -> np.ndarray:
    """
    Predict positive-class probabilities for a dataset. Returns np.ndarray in [0,1] for class 1.
    """
    t0 = _t0("[HF] Predict logits ...")
    preds = trainer.predict(ds)
    _tend("bert.predict_logits", t0)
    logits = preds.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    return probs

def train_eval_bert(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 2,
    batch_size: int = 16,
    max_len: int = 128,
    lr: float = 2e-5
) -> RunBundle:
    """
    Train the BERT baseline and evaluate on val/test; return artefacts for saving.
    """
    # HF components
    tok, mdl = _build_model_and_tokenizer(model_name)

    # Datasets
    ds_tr, ds_va, ds_te = _build_frame_and_datasets(df_train.copy(), df_val.copy(), df_test.copy(), tok, max_len=max_len)

    # Trainer
    trainer = _trainer(mdl, tok, ds_tr, ds_va, out_dir=MODEL_DIR, epochs=epochs, batch_size=batch_size, lr=lr)

    # Train
    t0 = _t0("Training BERT baseline ...")
    trainer.train()
    trainer.save_model(str(MODEL_DIR))
    tok.save_pretrained(str(MODEL_DIR))
    _tend("bert.train", t0)

    # Predict (Val/Test)
    p_val = _predict_proba(trainer, ds_va)
    p_tst = _predict_proba(trainer, ds_te)

    y_va = ds_va.df["label"].to_numpy()
    y_te = ds_te.df["label"].to_numpy()
    ypv = (p_val >= 0.5).astype(int)
    ypt = (p_tst >= 0.5).astype(int)

    # Overall metrics
    mo_val = overall_metrics(y_va, ypv)
    mo_tst = overall_metrics(y_te, ypt)
    val_metrics  = pd.DataFrame([{"Split": "Val",  "Accuracy": mo_val.acc, "Precision": mo_val.prec, "Recall": mo_val.rec, "F1": mo_val.f1}])
    test_metrics = pd.DataFrame([{"Split": "Test", "Accuracy": mo_tst.acc, "Precision": mo_tst.prec, "Recall": mo_tst.rec, "F1": mo_tst.f1}])

    # Predictions frames (carry IDs/titles for interpretability via backing df)
    def _mk_preds_frame(ds, probs, ypred) -> pd.DataFrame:
        dfp = pd.DataFrame({
            "video_id": ds.df.get("video_id", pd.RangeIndex(len(ds.df))),
            "title": ds.df.get("title", pd.Series([""]*len(ds.df))),
            "Group": group_labels_intersectional(ds.df).to_numpy(),
            "y_true": ds.df["label"].to_numpy(),
            "y_pred": ypred,
            "prob": np.round(probs, 3),
        })
        dfp["margin"] = dfp["prob"] - 0.5
        return dfp

    val_preds  = _mk_preds_frame(ds_va, p_val, ypv)
    test_preds = _mk_preds_frame(ds_te, p_tst, ypt)

    # Group metrics & disparities (Test)
    gm_test = calculate_group_metrics(test_preds)
    disp = calculate_fairness_disparities(gm_test, privileged="White Women")

    # Outliers (qualitative)
    outliers = top_confident_outliers(test_preds, probs=p_tst, k=10)

    return RunBundle(
        model_dir=MODEL_DIR,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_preds=val_preds,
        test_preds=test_preds,
        group_metrics_test=gm_test,
        disparities_test=disp,
        outliers_test=outliers,
    )

# --- 7) Save + narrative ------------------------------------------------------
def _save_all(bundle: RunBundle, *, selfcheck: bool = False) -> None:
    """
    Persist metrics, predictions, model path (directory), plot, and narrative.
    Self-check writes *_selfcheck files ONLY (non-destructive).
    """
    t0 = _t0("Saving BERT artefacts ...")
    for p in [DATA_DIR, FIG_DIR, FIG_MODELS_DIR, MODEL_DIR, NARR_DIR, GOLD_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    suffix = "_selfcheck" if selfcheck else ""

    # 09_* artefacts
    overall_df = pd.concat([bundle.val_metrics, bundle.test_metrics], ignore_index=True)
    overall_df.to_csv(DATA_DIR / f"09_overall_metrics{suffix}.csv", index=False)

    val_metrics_csv  = DATA_DIR / f"09_bert_val_metrics{suffix}.csv"
    val_preds_csv    = DATA_DIR / f"09_bert_val_predictions{suffix}.csv"
    test_metrics_csv = DATA_DIR / f"09_bert_test_metrics{suffix}.csv"
    test_preds_csv   = DATA_DIR / f"09_bert_test_predictions{suffix}.csv"
    group_csv        = DATA_DIR / f"09_fairness_group_metrics{suffix}.csv"
    disp_csv         = DATA_DIR / f"09_fairness_disparities{suffix}.csv"

    bundle.val_metrics.to_csv(val_metrics_csv, index=False)
    bundle.val_preds.to_csv(val_preds_csv, index=False)
    bundle.test_metrics.to_csv(test_metrics_csv, index=False)
    bundle.test_preds.to_csv(test_preds_csv, index=False)
    bundle.group_metrics_test.to_csv(group_csv, index=False)
    bundle.disparities_test.to_csv(disp_csv, index=False)

    # Save outliers CSV for interpretability (top-10 confident mistakes)
    out_csv = DATA_DIR / f"09_bert_outliers_top10{suffix}.csv"
    bundle.outliers_test.to_csv(out_csv, index=False)


    # Standardised figure path/name
    _plot_margins(
        margins=bundle.test_preds["margin"].to_numpy(),
        title=f"BERT Baseline — Decision Margins (Test){' (self-check)' if selfcheck else ''}",
        save_path=str(FIG_MODELS_DIR / f"09_margins_baseline{suffix}"),
        figsize=(9, 6),
    )

    # Narrative (09_*)
    narr_path = NARRATIVE_PATH if not selfcheck else NARR_DIR / f"09_bert_baseline_summary{suffix}.md"
    lines = []
    lines.append(f"# Automated Summary: BERT Baseline{' — self-check' if selfcheck else ''}\n")
    lines.append("## Overall Metrics\n")
    lines.append(overall_df.round(3).to_string(index=False))
    lines.append("\n## Group Metrics (Test)\n")
    lines.append(bundle.group_metrics_test.round(3).to_string(index=False))
    lines.append("\n## Disparities vs. White Women\n")
    lines.append((bundle.disparities_test.round(3) if not bundle.disparities_test.empty else
                  pd.DataFrame(columns=["Comparison Group","Accuracy Disparity","Equal Opportunity Difference","Precision Disparity"])
                 ).to_string(index=False))
    lines.append("\n## Top 10 Outliers (Test) — most confident mistakes\n")
    lines.append(bundle.outliers_test.to_string(index=False))
    lines.append("\n*Note*: Some titles are non-English; tags/categories often anchor semantics; we inspected outliers accordingly.")
    with open(narr_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✓ Narrative saved: {narr_path.resolve()}")
    print(f"✓ Model directory contains weights/tokenizer: {MODEL_DIR.resolve()}")
    print("✓ Artefacts saved:",
          (DATA_DIR / f"09_overall_metrics{suffix}.csv").name, ",",
          val_metrics_csv.name, ",", val_preds_csv.name, ",",
          test_metrics_csv.name, ",", test_preds_csv.name, ",",
          group_csv.name, ",", disp_csv.name, ",", out_csv.name)
    _tend("bert.save_all", t0)

def _maybe_eval_gold(
    trainer: Trainer,
    df_all: pd.DataFrame,
    tokenizer,
    gold_path: Path
) -> None:
    """
    Evaluate the trained model on a GOLD subset if a GOLD CSV exists at gold_path.

    Saves two artefacts under outputs/data/:
      - 09_bert_gold_eval_metrics.csv
      - 09_bert_gold_eval_predictions.csv

    Notes
    -----
    - Non-destructive: separate files; does NOT alter train/val/test artefacts.
    - Aligns GOLD rows to DF by 'video_id'; uses GOLD-provided labels ('gold_label').
    """
    if not gold_path.exists():
        print(f"⚠ GOLD evaluation skipped: file not found at {gold_path}")
        return

    print(f"[GOLD] Using file: {gold_path}")
    gold = load_gold_table(gold_path)
    if gold is None:
        print("⚠ GOLD evaluation skipped: loader returned None (empty/invalid file).")
        return

    joined = align_gold_to_frame(df_all, gold)
    if joined.empty:
        print("⚠ GOLD join produced 0 rows; skipping GOLD evaluation.")
        return

    tmp = joined.copy()
    if TEXT_COL not in tmp.columns:
        tmp[TEXT_COL] = (tmp.get("title", "").fillna("").astype(str) + " " +
                         tmp.get("tags", "").fillna("").astype(str))
    tmp["label"] = tmp["gold_label"].astype(int)

    ds_gold = TextBinDataset(tmp, tokenizer, max_len=128)

    t0 = _t0("Scoring probabilities on GOLD subset (BERT) ...")
    probs = _predict_proba(trainer, ds_gold)
    _tend("bert.gold_predict", t0)

    metrics_df, preds_df = evaluate_against_gold(probs, tmp)
    metrics_df.to_csv(GOLD_METRICS_CSV, index=False)
    preds_df.to_csv(GOLD_PREDICTIONS_CSV, index=False)
    print(f"✓ GOLD artefacts saved: {GOLD_METRICS_CSV.name}, {GOLD_PREDICTIONS_CSV.name}")

def _log_basic_outliers(df: pd.DataFrame) -> None:
    """
    Log simple 99th-percentile outliers for interpretability. Non-destructive.

    - ratings are rounded to 1 decimal; other counts/durations to integers.
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

# --- 8) Main ------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Run the BERT baseline full pipeline with optional self-check and GOLD eval.

    Options
    -------
    --selfcheck             Use a random sample and internal split (safe).
    --sample INT            Random sample size for self-check (default: min(16k, N)).
    --epochs INT            Training epochs (default: 2).
    --batch INT             Per-device batch size (default: 16).
    --max-len INT           Max sequence length (default: 128).
    --lr FLOAT              Learning rate (default: 2e-5).
    --model STR             HF model name/id (default from settings.yaml or distilbert-base-uncased).
    --gold-path STR         Path to GOLD CSV (default: outputs/data/gold/gold_final.csv).
    --no-gold               Force skip GOLD evaluation even if file exists.

    Notes
    -----
    - Totals elsewhere can exceed N due to multi-label tasks; here we evaluate a
      *single* binary target per ID, so counts sum to N.
    - Non-English titles may appear among outliers; tags/categories provide anchors.
    """
    # argparse imported at top

    t_all = time.perf_counter()
    print("--- Starting Step 09: BERT Baseline ---")
    np.random.seed(SEED)
    try:
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--gold-path", type=str, default=str(DEFAULT_GOLD_PATH))
    p.add_argument("--no-gold", action="store_true")
    args = p.parse_args(argv)

    # Load corpus
    df = _load_corpus()
    total = len(df)
    print(f"[STATS] Total videos available: {total:,}")

    # Build splits
    if args.selfcheck:
        n = args.sample or min(16_000, total)
        df = df.sample(n=n, random_state=SEED, replace=False).reset_index(drop=True)
        print(f"[SELF-CHECK] Random sample drawn: {len(df):,} rows (seed={SEED}).")
        train_df, val_df, test_df = _make_internal_split(df, seed=SEED)
    else:
        tr_ids, va_ids, te_ids = _read_ids_or_none(TRAIN_IDS), _read_ids_or_none(VAL_IDS), _read_ids_or_none(TEST_IDS)
        if not (tr_ids is not None and va_ids is not None and te_ids is not None):
            print("✗ Step-06 IDs not found; falling back to internal split on full corpus.")
            train_df, val_df, test_df = _make_internal_split(df, seed=SEED)
        else:
            train_df = _subset_by_ids(df, tr_ids)
            val_df   = _subset_by_ids(df, va_ids)
            test_df  = _subset_by_ids(df, te_ids)

    print(f"Split sizes: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    # Brief qualitative outlier note + log
    print("[NOTE] Titles may be non-English; tags/categories help anchor semantics.")
    print("[NOTE] Logging basic outliers for situational awareness (non-destructive).")
    _log_basic_outliers(train_df)

    # Train & evaluate
    bundle = train_eval_bert(
        df_train=train_df,
        df_val=val_df,
        df_test=test_df,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        max_len=args.max_len,
        lr=args.lr,
    )

    # Save artefacts
    selfcheck = bool(args.selfcheck)
    _save_all(bundle, selfcheck=selfcheck)

    # Print interpretable highlights
    print("\n=== Overall Metrics (Accuracy/Precision/Recall/F1) ===")
    print(pd.concat([bundle.val_metrics, bundle.test_metrics], ignore_index=True).to_string(index=False))
    print("\n=== Group Metrics (Test) ===")
    print(bundle.group_metrics_test.to_string(index=False))
    print("\n=== Disparities vs. White Women (privileged) ===")
    print(bundle.disparities_test.to_string(index=False))
    print("\n=== Top 10 Outliers (Test) — most confident mistakes ===")
    print(bundle.outliers_test.to_string(index=False))

    # Optional GOLD evaluation (full run only and if available)
    if not selfcheck and not args.no_gold:
        # Reuse the trained model for prediction on GOLD
        tok, _ = _build_model_and_tokenizer(args.model)  # tokenizer only
        mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        import inspect as _ins2
        _t2_params = set(_ins2.signature(Trainer.__init__).parameters)
        _tok_kw2 = "processing_class" if "processing_class" in _t2_params else "tokenizer"
        trainer_gold = Trainer(
            model=mdl,
            args=TrainingArguments(
                output_dir=str(MODEL_DIR),
                per_device_eval_batch_size=args.batch,
                report_to=[],
                seed=SEED,
                save_strategy="no",
                dataloader_pin_memory=False,
            ),
            **{_tok_kw2: tok},
            data_collator=DataCollatorWithPadding(tokenizer=tok),
        )
        print("Running optional GOLD evaluation ...")
        _maybe_eval_gold(
            trainer_gold,
            pd.concat([train_df, val_df, test_df], ignore_index=True),
            tok,
            Path(args.gold_path),
        )

    _tend("bert.step09_total", t_all)
    print("\n--- Step 09: BERT Baseline Completed Successfully ---")


if __name__ == "__main__":
    main()
