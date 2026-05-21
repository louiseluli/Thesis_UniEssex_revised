# -*- coding: utf-8 -*-
"""
15_qualitative_deep_dive.py
===========================

Purpose
-------
Collect the highest-confidence model mistakes (top-K by |p - 0.5|) for a chosen
intersectional focus group (default: "Black Women") to enable qualitative review.
Works with baseline or mitigation predictions artefacts (auto-discovery).

What it does
------------
1) Loads config (seed=95 by default) and canonical corpus.
2) Auto-discovers the first available predictions CSV among baseline/mitigation artefacts,
   or use --pred-path to specify the file explicitly.
3) Merges predictions with corpus metadata (id/title/tags/categories read from config).
4) Identifies errors (y_pred != y_true), computes |margin| = |prob - 0.5|,
   filters to the focus group, selects top-K (or a random sample in --selfcheck).
5) Saves CSV + a short narrative; prints a small preview and lightweight timers.
6) Prints a total runtime line.

Self-check
----------
--selfcheck samples randomly from the focus-group errors (non-destructive);
artefacts are suffixed with *_selfcheck and never overwrite full-run files.

Interpretability & language note
--------------------------------
Titles in other languages can be harder to interpret; **tags** and **categories**
often anchor semantics—use your MPU when reviewing.

Totals vs N
-----------
Previous steps can be multi-label (totals may exceed N there). Here we work over
per-ID predictions on the Test (or Val) split, so counts sum to N at this step.

CLI
---
# Full run (most important for your testing)
python -m src.analysis.15_qualitative_deep_dive --k 50 --focus "Black Women"

# Self-check (safe random sample; writes *_selfcheck outputs only)
python -m src.analysis.15_qualitative_deep_dive --selfcheck --k 50 --sample 200 --focus "Black Women"

# Explicit predictions file (skips auto-discovery)
python -m src.analysis.15_qualitative_deep_dive --pred-path outputs/data/09_bert_test_predictions.csv
"""
from __future__ import annotations

# --- Imports (keep at top) ---------------------------------------------------
import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# Project utils (theme_manager.load_config prints its own [TIME] line)
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config  # noqa: E402


# --- 1) Config & Paths -------------------------------------------------------
_t0_cfg = time.perf_counter()
CONFIG = load_config()
print(f"[TIME] theme_manager.load_config: {time.perf_counter() - _t0_cfg:.2f}s")

SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))

ROOT = Path(CONFIG["project"]["root"])
DATA_DIR = Path(CONFIG["paths"]["data"])              # typically outputs/data
NARR_DIR = Path(CONFIG["paths"]["narratives"]) / "automated"
DATA_DIR.mkdir(parents=True, exist_ok=True)
NARR_DIR.mkdir(parents=True, exist_ok=True)

# Column names (config-driven with robust defaults)
COLS = CONFIG.get("columns", {})
ID_COL = COLS.get("id", "video_id")
TITLE_COL = COLS.get("title", "title")
TAGS_COL = COLS.get("tags", "tags")
CATS_COL = COLS.get("categories", "categories")
GROUP_COL = COLS.get("group", "Group")

# Canonical corpus paths (prefer step-01 parquet; tolerate fallbacks)
CORPUS_CANDIDATES = [
    DATA_DIR / "01_ml_corpus.parquet",
    DATA_DIR / "ml_corpus.parquet",
    DATA_DIR / "01_ml_corpus.snappy.parquet",
]

# Predictions candidates (priority order; baseline & mitigators + legacy + BERT + val)
PRED_CANDIDATES = [
    DATA_DIR / "12_postproc_predictions_test.csv",
    DATA_DIR / "12_postproc_predictions_val.csv",
    DATA_DIR / "11_inproc_predictions_test.csv",
    DATA_DIR / "11_inproc_predictions_val.csv",
    DATA_DIR / "10_reweigh_predictions_test.csv",
    DATA_DIR / "10_reweigh_predictions_val.csv",
    DATA_DIR / "09_bert_test_predictions.csv",
    DATA_DIR / "09_bert_val_predictions.csv",
    DATA_DIR / "07_rf_test_predictions.csv",
    DATA_DIR / "07_rf_val_predictions.csv",
    # legacy naming
    DATA_DIR / "egpp_predictions_test_postproc.csv",
    DATA_DIR / "egdp_predictions_test_inproc.csv",
    DATA_DIR / "rf_reweighed_val_predictions.csv",
]

# --- 2) Timers ---------------------------------------------------------------
def _t0(msg: str) -> float:
    print(msg)
    return time.perf_counter()

def _tend(label: str, t0: float) -> None:
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")


# --- 3) Helpers --------------------------------------------------------------
def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _read_corpus() -> pd.DataFrame:
    path = _first_existing(CORPUS_CANDIDATES)
    if path is None:
        raise FileNotFoundError(
            "Corpus not found under outputs/data. Expected one of: "
            + ", ".join(str(p.name) for p in CORPUS_CANDIDATES)
        )
    t0 = _t0(f"[READ] Parquet: {path}")
    df = pd.read_parquet(path)
    _tend("step15.load_corpus", t0)

    # Ensure minimal text fields for qualitative review
    for col, default in [(TITLE_COL, ""), (TAGS_COL, ""), (CATS_COL, "")]:
        if col not in df.columns:
            df[col] = default

    # Provide a generic text field if needed
    if "model_input_text" not in df.columns:
        df["model_input_text"] = (
            df.get(TITLE_COL, "").fillna("").astype(str) + " " +
            df.get(TAGS_COL, "").fillna("").astype(str)
        )
    return df


def _coerce_numeric_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(-1).astype(int)


def _resolve_prob_and_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'prob' and 'margin' exist; derive margin as (prob - 0.5).
    Accepts tolerant names: prob/proba/prob_pos/p1/score.
    """
    if "prob" not in df.columns:
        cand = [c for c in df.columns if c.lower() in {"prob", "proba", "prob_pos", "p1", "score"}]
        if cand:
            df = df.rename(columns={cand[0]: "prob"})
    if "prob" not in df.columns:
        df["prob"] = 0.5  # neutral fallback if no probability available
    if "margin" not in df.columns:
        df["margin"] = df["prob"] - 0.5
    return df


def _load_predictions(pred_path_cli: Optional[str]) -> Tuple[pd.DataFrame, Path]:
    """
    Load predictions either from --pred-path or the first available candidate.
    Returns df minimally with: ID_COL, y_true, y_pred, prob, margin, (optional) GROUP_COL.
    """
    path: Optional[Path] = Path(pred_path_cli) if pred_path_cli else _first_existing(PRED_CANDIDATES)
    if path is None:
        raise FileNotFoundError(
            "No predictions file found in outputs/data. "
            "Run Step-07/09/10/11/12 first, or pass --pred-path."
        )
    t0 = _t0(f"[READ] Predictions: {path}")
    df = pd.read_csv(path)
    _tend("step15.load_predictions", t0)

    # Normalize columns
    if ID_COL not in df.columns:
        for alt in ("video_id", "id"):
            if alt in df.columns:
                df = df.rename(columns={alt: ID_COL})
                break
    if ID_COL not in df.columns:
        raise KeyError(f"ID column '{ID_COL}' not found in predictions CSV.")

    for c in ("y_true", "y_pred"):
        if c in df.columns:
            df[c] = _coerce_numeric_int(df[c])
        else:
            raise KeyError(f"Required column '{c}' missing in predictions CSV.")

    df = _resolve_prob_and_margin(df)
    return df, path


_FOCUS_ALIASES: Dict[str, Tuple[str, str]] = {
    "Black Women": ("race_ethnicity_black", "gender_female"),
    "White Women": ("race_ethnicity_white", "gender_female"),
    "Asian Women": ("race_ethnicity_asian", "gender_female"),
    "Latina Women": ("race_ethnicity_latina", "gender_female"),
}

def _mask_focus_group(df_meta: pd.DataFrame, focus: str) -> np.ndarray:
    if GROUP_COL in df_meta.columns:
        return (df_meta[GROUP_COL].astype(str).str.strip() == focus).to_numpy()
    if "intersection_group" in df_meta.columns:
        return (df_meta["intersection_group"].astype(str).str.strip() == focus).to_numpy()
    if focus in _FOCUS_ALIASES:
        race_col, gender_col = _FOCUS_ALIASES[focus]
        return ((df_meta.get(race_col, 0) == 1) & (df_meta.get(gender_col, 0) == 1)).to_numpy()
    return np.ones(len(df_meta), dtype=bool)


def _collect_topk_errors(
    df_corpus: pd.DataFrame,
    df_pred: pd.DataFrame,
    focus: str,
    k: int,
    *,
    random_sample: bool,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge metadata, identify errors, filter to focus group, then select top-K mistakes.
    Returns (selected_subset, full_focus_error_pool).
    """
    t0 = _t0("Merging predictions with corpus ...")
    keep_cols = [c for c in {ID_COL, TITLE_COL, TAGS_COL, CATS_COL,
                             "intersection_group",
                             "race_ethnicity_black", "race_ethnicity_white",
                             "race_ethnicity_asian", "race_ethnicity_latina",
                             "gender_female",
                             GROUP_COL} if c in df_corpus.columns]
    meta = df_corpus[keep_cols].copy()
    df = df_pred.merge(meta, on=ID_COL, how="left")
    _tend("step15.merge", t0)

    t0 = _t0("Filtering errors and focus-group subset ...")
    errors = df[(df["y_true"] != df["y_pred"]) & (df["y_true"] >= 0) & (df["y_pred"] >= 0)].copy()
    errors["margin_abs"] = errors["margin"].abs()
    mask_focus = _mask_focus_group(errors, focus)
    errors_focus = errors[mask_focus].copy()
    _tend("step15.filter", t0)

    if errors_focus.empty:
        return errors_focus, errors_focus

    if random_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(errors_focus), size=min(k, len(errors_focus)), replace=False)
        sel = errors_focus.iloc[np.sort(idx)]
    else:
        sel = errors_focus.sort_values("margin_abs", ascending=False).head(k)

    # Readability rounding
    if "prob" in sel.columns:
        sel["prob"] = sel["prob"].map(lambda x: float(np.round(x, 3)))
    for col in ("margin", "margin_abs"):
        if col in sel.columns:
            sel[col] = sel[col].map(lambda x: float(np.round(x, 3)))

    return sel.reset_index(drop=True), errors_focus.reset_index(drop=True)


def _qual_summary(df_sel: pd.DataFrame) -> Dict[str, object]:
    """
    Build small, interpretable summaries (counts + tag frequency + non-ASCII heuristic).
    """
    def _non_ascii_ratio(s: str) -> float:
        if not isinstance(s, str) or not s:
            return 0.0
        return sum(ord(ch) > 127 for ch in s) / len(s)

    non_ascii_share = float(np.round(
        df_sel.get(TITLE_COL, pd.Series([""] * len(df_sel))).fillna("").map(_non_ascii_ratio).mean(), 3
    ))

    tag_freq: Dict[str, int] = {}
    if TAGS_COL in df_sel.columns:
        for raw in df_sel[TAGS_COL].fillna("").astype(str):
            for t in [x.strip().lower() for x in raw.split(",") if x.strip()]:
                tag_freq[t] = tag_freq.get(t, 0) + 1
    top_tags = sorted(tag_freq.items(), key=lambda kv: kv[1], reverse=True)[:10]

    return {"n_selected": int(len(df_sel)),
            "non_ascii_title_share": non_ascii_share,
            "top_tags": top_tags}


# --- 4) Main -----------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Run Step-15 qualitative deep-dive.

    Options
    -------
    --k INT           Number of examples to select (default 50).
    --focus STR       Focus group label (default: 'Black Women').
    --selfcheck       Sample randomly instead of top-K (non-destructive).
    --sample INT      Max random samples (default: k if not provided).
    --pred-path STR   Explicit predictions CSV path (skips auto-discovery).

    Notes
    -----
    - Uses seed from config (not 42).
    - Imports remain at the top.
    - Self-check does not overwrite full-run artefacts.
    - Titles may be non-English; tags/categories help anchor semantics.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--focus", type=str, default="Black Women")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--pred-path", type=str, default=None)
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    print("--- Starting Step 15: Qualitative Deep Dive ---")

    # Load assets
    df_corpus = _read_corpus()
    df_pred, pred_path = _load_predictions(args.pred_path)

    # Select pool and the final subset
    random_sample = bool(args.selfcheck)
    k_eff = (args.sample or args.k) if random_sample else args.k
    df_top, df_pool = _collect_topk_errors(
        df_corpus=df_corpus,
        df_pred=df_pred,
        focus=args.focus,
        k=k_eff,
        random_sample=random_sample,
        seed=SEED,
    )

    if df_pool.empty:
        print(f"✗ No errors found for focus group '{args.focus}'. Nothing to save.")
        _tend("step15.total_runtime", t_all)
        print("\n--- Step 15: Qualitative Deep Dive Completed (No Results) ---")
        return

    suffix = "_selfcheck" if args.selfcheck else ""
    out_csv = DATA_DIR / f"15_qualitative_error_samples_topk{suffix}.csv"
    df_top.to_csv(out_csv, index=False)

    # Narrative
    summ = _qual_summary(df_top)
    # quick confusion breakdown inside focus errors
    try:
        confu = (df_pool.groupby(["y_true", "y_pred"])
                 .size().rename("count").reset_index()
                 .sort_values("count", ascending=False))
        confu_lines = ["- Confusions (y_true → y_pred → count):"] + [
            f"  - {int(r.y_true)} → {int(r.y_pred)} → {int(r['count'])}" for r in confu.itertuples(index=False)
        ]
    except Exception:
        confu_lines = []

    md = [
        "# 15 — Qualitative Deep Dive",
        f"- Source predictions: `{pred_path.name}`.",
        f"- Focus group: **{args.focus}**.",
        f"- Pool size (focus-group errors): **{len(df_pool):,}**.",
        f"- Selected: **{summ['n_selected']}** "
        f"{'random samples' if args.selfcheck else 'top-K by |p-0.5|'}.\n",
        "## Quick readout",
        f"- Share of titles likely non-English (non-ASCII heuristic): **{summ['non_ascii_title_share']:.3f}**.",
        "- Top tags in selected errors (tag → count):",
    ] + [f"  - {tag}: {cnt}" for tag, cnt in summ["top_tags"]] + ([""] + confu_lines if confu_lines else []) + [
        "\n*Note:* Titles in other languages can be harder to interpret; "
        "tags and categories help anchor meaning (lean on your MPU during review)."
    ]
    md_path = NARR_DIR / f"15_qualitative_summary{suffix}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # Console preview (sanity check)
    preview_cols = [c for c in [ID_COL, TITLE_COL, GROUP_COL, "y_true", "y_pred", "prob", "margin_abs"] if c in df_top.columns]
    print("\n=== Preview (top 10) ===")
    print(df_top[preview_cols].head(10).to_string(index=False))

    print(f"\n✓ Artefacts saved: {out_csv.name}, {md_path.name}")
    _tend("step15.total_runtime", t_all)
    print("\n--- Step 15: Qualitative Deep Dive Completed Successfully ---")


if __name__ == "__main__":
    main()
