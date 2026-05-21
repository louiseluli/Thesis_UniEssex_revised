# -*- coding: utf-8 -*-
"""
13_mitigation_effectiveness.py
==============================

Purpose
-------
Synthesize mitigation effectiveness (RQ4) by comparing overall Test performance
and fairness disparities across four systems:

  1) Baseline RF (Step 08)
  2) Pre-processing (Reweighing; Step 10)
  3) In-processing (Exponentiated Gradient, Demographic Parity; Step 11)
  4) Post-processing (ThresholdOptimizer; Step 12)

What it does
------------
- Discovers previously saved artefacts under outputs/data/ using robust filename
  patterns (supports both full-run and self-check variants).
- Extracts Test Accuracy and disparities vs the privileged group ("White Women"),
  focused on a configurable comparison group (default: "Black Women").
- Builds a compact comparison table (Accuracy, Accuracy Δ vs White, TPR Δ vs White)
  to illustrate the fairness–accuracy trade-off.
- Saves CSV & LaTeX, prints a concise qualitative analysis, can optionally peek
  at top outliers saved by prior steps, and times each step.

Interpretability & language notes
---------------------------------
- Positive Δ means the privileged group (White Women) outperforms the comparison
  group on that metric; negative Δ means the reverse.
- Titles in other languages can be harder to interpret; tags/categories often
  anchor semantics—expect some non-English items among outliers in earlier steps.
- Totals can exceed N in multi-label tagging upstream; here we summarize per-ID
  Test metrics, so splits sum to N.

CLI
---
# Full run (writes *_effectiveness.* artefacts)
python3 src/fairness/13_mitigation_effectiveness.py

# Self-check (safe): writes *_effectiveness_selfcheck.* only
python3 src/fairness/13_mitigation_effectiveness.py --selfcheck --sample-models 2

# Options
python3 src/fairness/13_mitigation_effectiveness.py --focus-group "Black Women" --peek-outliers
"""

from __future__ import annotations

# --- Imports (keep at top) ---------------------------------------------------
import sys
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project utils
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config
from src.utils.academic_tables import dataframe_to_latex_table


# --- 1) Config & Paths -------------------------------------------------------
_t0_cfg = time.perf_counter()
CONFIG = load_config()
print(f"[TIME] theme_manager.load_config: {time.perf_counter() - _t0_cfg:.2f}s")

DATA_DIR = Path(CONFIG['paths']['data'])               # typically outputs/data
OUT_DIR  = Path(CONFIG['paths']['outputs']) / 'data'   # explicitly outputs/data
TEX_DIR  = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'
SEED     = int(CONFIG.get('reproducibility', {}).get('seed', 95))


# --- 2) Timers ---------------------------------------------------------------
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
        perf_counter start time.
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
        Short label describing the timed block.
    t0 : float
        Start time from _t0.
    """
    print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")


# --- 3) Robust artefact discovery -------------------------------------------
@dataclass
class ModelFiles:
    """
    Container for a model's artefact paths.

    Attributes
    ----------
    name : str
        Human-readable model name.
    overall : Optional[Path]
        Path to overall metrics CSV (must include a Test row with 'Accuracy').
    disparities : Optional[Path]
        Path to disparities CSV with 'Comparison Group' and Δ columns.
    group : Optional[Path]
        Path to group metrics CSV (optional here).
    outliers : Optional[Path]
        Path to outliers CSV (optional; for interpretability peek).
    """
    name: str
    overall: Optional[Path]
    disparities: Optional[Path]
    group: Optional[Path]
    outliers: Optional[Path]


def _first_existing_glob(patterns: List[str]) -> Optional[Path]:
    """
    Return the first existing path matching any of the glob patterns under OUT_DIR,
    then under DATA_DIR as fallback.

    Parameters
    ----------
    patterns : List[str]
        Glob patterns to evaluate in order.

    Returns
    -------
    Optional[Path]
        First matching path or None.
    """
    for base in (OUT_DIR, DATA_DIR):
        for pat in patterns:
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0]
    return None


def _model_candidates() -> Dict[str, ModelFiles]:
    """
    Define filename patterns per model and resolve to actual files.

    Returns
    -------
    Dict[str, ModelFiles]
        Mapping from model name to resolved file handles (or None when not found).
    """
    # ---- Baseline RF (Step 07/08): cover common variants ----
    base_overall = _first_existing_glob([
        "07_overall_metrics*.csv",
        "08_rf_overall_metrics*.csv",
        "rf_overall_metrics*.csv",
        "baseline_overall_metrics*.csv",
    ])
    base_disp = _first_existing_glob([
        "07_fairness_disparities*.csv",
        "08_rf_disparities*.csv",
        "rf_disparities*.csv",
        "fairness_disparities_rf*.csv",
    ])
    base_group = _first_existing_glob([
        "07_fairness_group_metrics*.csv",
        "08_rf_group_metrics*.csv",
        "rf_group_metrics*.csv",
        "fairness_group_metrics_rf*.csv",
    ])
    base_outliers = _first_existing_glob([
        "07_rf_outliers_top*.csv",
        "08_rf_outliers_top*.csv",
        "rf_outliers_top*.csv",
    ])


    # ---- Pre-processing (Step 10): your actual names + alternates ----
    pre_overall = _first_existing_glob([
        "10_reweigh_overall_metrics*.csv", "egpr_overall_metrics*_reweigh*.csv"
    ])
    pre_disp = _first_existing_glob([
        "10_reweigh_disparities*.csv", "egpr_disparities*_reweigh*.csv"
    ])
    pre_group = _first_existing_glob([
        "10_reweigh_group_metrics*.csv", "egpr_group_metrics*_reweigh*.csv"
    ])
    pre_outliers = _first_existing_glob([
        "10_reweigh_outliers_top*.csv"
    ])

    # ---- In-processing (Step 11): both new + legacy ----
    in_overall = _first_existing_glob(["11_inproc_overall_metrics*.csv", "egdp_overall_metrics*inproc*.csv"])
    in_disp    = _first_existing_glob(["11_inproc_disparities*.csv",    "egdp_disparities*inproc*.csv"])
    in_group   = _first_existing_glob(["11_inproc_group_metrics*.csv",  "egdp_group_metrics*inproc*.csv"])
    in_outliers= _first_existing_glob(["11_inproc_outliers_top*.csv",   "egdp_outliers_topk*inproc*.csv"])

    # ---- Post-processing (Step 12): both new + legacy ----
    post_overall = _first_existing_glob(["12_postproc_overall_metrics*.csv", "egpp_overall_metrics*postproc*.csv"])
    post_disp    = _first_existing_glob(["12_postproc_disparities*.csv",     "egpp_disparities*postproc*.csv"])
    post_group   = _first_existing_glob(["12_postproc_group_metrics*.csv",   "egpp_group_metrics*postproc*.csv"])
    post_outliers= _first_existing_glob(["12_postproc_outliers_top*.csv",    "egpp_outliers_top10*postproc*.csv"])

    registry: Dict[str, ModelFiles] = {
        "Baseline RF": ModelFiles("Baseline RF", base_overall, base_disp, base_group, base_outliers),
        "Reweighed RF": ModelFiles("Reweighed RF", pre_overall, pre_disp, pre_group, pre_outliers),
        "In-Processing (EG, DP)": ModelFiles("In-Processing (EG, DP)", in_overall, in_disp, in_group, in_outliers),
        "Post-Processing (Threshold)": ModelFiles("Post-Processing (Threshold)", post_overall, post_disp, post_group, post_outliers),
    }

    print("\n[DISCOVERY] Resolved artefacts:")
    for k, v in registry.items():
        print(f"  • {k}: overall={v.overall}, disparities={v.disparities}, group={v.group}, outliers={v.outliers}")
    return registry


# --- 4) Loading & extraction helpers -----------------------------------------
def _load_overall_accuracy(path: Optional[Path]) -> Optional[float]:
    """
    Load Test Accuracy from an 'overall metrics' CSV.

    Parameters
    ----------
    path : Optional[Path]
        Candidate file path.

    Returns
    -------
    Optional[float]
        Test Accuracy rounded to 3 decimals, or None if not found.
    """
    if path is not None and path.exists():
        try:
            df = pd.read_csv(path)
            if "Split" in df.columns and "Accuracy" in df.columns:
                row = df.loc[df["Split"].astype(str).str.lower() == "test"]
                if len(row) == 1:
                    return float(np.round(row["Accuracy"].iloc[0], 3))
            if "Accuracy" in df.columns and len(df) == 1:
                return float(np.round(df["Accuracy"].iloc[0], 3))
        except Exception as e:
            print(f"✗ Failed to load overall accuracy from {path}: {e}")
    return None


def _extract_deltas(df: pd.DataFrame, focus_group: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract Accuracy Δ and TPR Δ for the focus group from a disparities DataFrame.

    Handles both 'Accuracy Disparity'/'Equal Opportunity Difference' and
    'Accuracy Δ'/'TPR Δ' naming variants.

    Parameters
    ----------
    df : pd.DataFrame
        Disparities table.
    focus_group : str
        Human-readable group label (e.g., 'Black Women').

    Returns
    -------
    (acc_delta, tpr_delta) : Tuple[Optional[float], Optional[float]]
    """
    # Identify group column
    cg = None
    for cand in ["Comparison Group", "Group", "group", "comparison_group"]:
        if cand in df.columns:
            cg = cand
            break
    if cg is None:
        return (None, None)

    row = df.loc[df[cg].astype(str).str.strip().str.lower() == focus_group.strip().lower()]
    if row.empty:
        return (None, None)

    # Identify delta columns
    acc_cols = ["Accuracy Δ", "Accuracy Delta", "Accuracy Disparity", "Accuracy Difference"]
    tpr_cols = ["TPR Δ", "TPR Delta", "Equal Opportunity Difference", "Recall Δ", "Recall Difference"]

    accd = next((c for c in acc_cols if c in df.columns), None)
    tprd = next((c for c in tpr_cols if c in df.columns), None)

    acc_delta = float(row[accd].iloc[0]) if accd else None
    tpr_delta = float(row[tprd].iloc[0]) if tprd else None
    return (acc_delta, tpr_delta)


def _load_black_disparities(path: Path, focus_group: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Load disparities CSV and extract Accuracy Δ and TPR Δ for the focus group.

    Parameters
    ----------
    path : Path
        Disparities CSV produced by Steps 10/11/12.
    focus_group : str
        Group label (e.g., 'Black Women').

    Returns
    -------
    (acc_delta, tpr_delta) : Tuple[Optional[float], Optional[float]]
    """
    try:
        df = pd.read_csv(path)
        return _extract_deltas(df, focus_group)
    except Exception as e:
        print(f"✗ Failed to load disparities from {path}: {e}")
        return (None, None)


# --- 5) Core synthesis --------------------------------------------------------
def build_comparison_table(
    focus_group: str,
    *,
    available: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build the mitigation effectiveness comparison table.

    Parameters
    ----------
    focus_group : str
        Group to compare against the privileged group (White Women).
    available : Optional[List[str]]
        Optional subset of model names to include; if None, include all
        discovered models that have both 'overall' and 'disparities'.

    Returns
    -------
    pd.DataFrame
        Comparison table indexed by model (rounded to 3 decimals).
    """
    t0 = _t0("Compiling mitigation effectiveness table ...")
    registry = _model_candidates()
    rows = []

    names_to_use = available if available is not None else list(registry.keys())

    for model_name in names_to_use:
        mf = registry.get(model_name)
        if mf is None:
            continue

        if mf.disparities is None or mf.overall is None:
            print(f"  - Skipping {model_name}: missing files (overall={mf.overall}, disparities={mf.disparities}).")
            continue

        acc = _load_overall_accuracy(mf.overall)
        acc_delta, tpr_delta = _load_black_disparities(mf.disparities, focus_group)

        if acc is None or (acc_delta is None and tpr_delta is None):
            print(f"  - Skipping {model_name}: could not extract metrics (acc={acc}, Δ={acc_delta}, TPRΔ={tpr_delta}).")
            continue

        rows.append({
            "Model": model_name,
            "Test Accuracy": acc,
            f"Accuracy Δ (vs White) [{focus_group}]": acc_delta,
            f"TPR Δ (vs White) [{focus_group}]": tpr_delta
        })

    if not rows:
        print("✗ No models could be summarized. Check previous steps' artefacts in outputs/data.")
        _tend("effectiveness.compile", t0)
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Model").sort_index()
    df = df.round(3)
    _tend("effectiveness.compile", t0)
    return df


# --- 6) IO helpers ------------------------------------------------------------
def _save_all(df_comp: pd.DataFrame, *, is_selfcheck: bool) -> Tuple[Path, Path]:
    """
    Save CSV and LaTeX artefacts for the comparison table.

    Parameters
    ----------
    df_comp : pd.DataFrame
        Comparison table from build_comparison_table.
    is_selfcheck : bool
        If True, suffix filenames with '_effectiveness_selfcheck' to avoid overwriting.

    Returns
    -------
    (csv_path, tex_path) : Tuple[Path, Path]
        Paths to saved artefacts.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "_effectiveness_selfcheck" if is_selfcheck else "_effectiveness"

    csv_path = OUT_DIR / f"mitigation{suffix}.csv"
    tex_path = TEX_DIR / f"mitigation{suffix}.tex"

    df_comp.to_csv(csv_path)
    dataframe_to_latex_table(
        df_comp,
        str(tex_path),
        "Fairness–Accuracy trade-off across baseline and mitigation strategies (Test split).",
        "tab:mitigation-effectiveness"
    )
    return csv_path, tex_path


# --- 7) Qualitative analysis --------------------------------------------------
def _qualitative_readout(df_comp: pd.DataFrame, focus_group: str) -> None:
    """
    Print a short, interpretable summary of the table.

    Parameters
    ----------
    df_comp : pd.DataFrame
        Comparison table with accuracy and deltas.
    focus_group : str
        Group compared against White Women.
    """
    if df_comp.empty:
        print("No results to analyze.")
        return

    print("\n--- Mitigation Effectiveness (Qualitative) ---")
    print("• Positive Δ means White Women outperform the comparison group; negative Δ means the reverse.")
    best_acc = df_comp["Test Accuracy"].idxmax()
    print(f"• Highest Test Accuracy: {best_acc} ({df_comp.loc[best_acc, 'Test Accuracy']:.3f}).")

    col_accd = f"Accuracy Δ (vs White) [{focus_group}]"
    col_tprd = f"TPR Δ (vs White) [{focus_group}]"

    if col_accd in df_comp.columns and not df_comp[col_accd].isna().all():
        accd_mag = df_comp[col_accd].abs()
        accd_best = accd_mag.idxmin()
        print(f"• Smallest |Accuracy Δ| (closer parity): {accd_best} ({df_comp.loc[accd_best, col_accd]:.3f}).")

    if col_tprd in df_comp.columns and not df_comp[col_tprd].isna().all():
        tprd_mag = df_comp[col_tprd].abs()
        tprd_best = tprd_mag.idxmin()
        print(f"• Smallest |TPR Δ| (closer equal opportunity): {tprd_best} ({df_comp.loc[tprd_best, col_tprd]:.3f}).")

    print("\nTable:")
    print(df_comp.to_string())


# --- 8) Optional: peek at outliers from previous steps -----------------------
def _peek_outliers(registry: Dict[str, ModelFiles], k: int = 3) -> None:
    """
    Print the first k rows from each available outliers CSV (non-destructive).

    Parameters
    ----------
    registry : Dict[str, ModelFiles]
        Model registry with resolved paths.
    k : int
        Number of outliers to display per model.
    """
    print("\n[PEEK] Top outliers from previous steps (titles may be non-English; use tags/categories for context):")
    any_found = False
    for name, mf in registry.items():
        if mf.outliers and mf.outliers.exists():
            try:
                df = pd.read_csv(mf.outliers)
                head = df.head(k)
                print(f"\n— {name}: {mf.outliers.name}")
                print(head.to_string(index=False))
                any_found = True
            except Exception as e:
                print(f"  (Could not read {mf.outliers}: {e})")
    if not any_found:
        print("  No outliers files found to peek.")


# --- 9) Main -----------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Run the mitigation effectiveness synthesis.

    Options
    -------
    --selfcheck               Use a subset of discovered models; artifacts saved as *_effectiveness_selfcheck.* only.
    --sample-models INT       How many models to randomly sample in self-check (default: 2).
    --seed INT                Random seed for self-check sampling (default: from CONFIG).
    --focus-group STR         Comparison group label (default: 'Black Women').
    --peek-outliers           If set, show the first few outliers from each step (non-destructive).

    Notes
    -----
    Prior steps may produce totals that exceed N because of multi-label tasks.
    Here, we summarize per-ID Test metrics so splits sum to N.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--sample-models", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--focus-group", type=str, default="Black Women")
    p.add_argument("--peek-outliers", action="store_true")
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    print("--- Starting Step 13: Mitigation Effectiveness Analysis ---")

    # Decide which models to include
    registry = _model_candidates()
    discovered = [k for k, mf in registry.items() if (mf.overall and mf.disparities)]
    if not discovered:
        print("✗ No candidate models discovered in outputs/data or data/. Exiting.")
        _tend("step13.total_runtime", t_all)
        return

    if args.selfcheck:
        rng = np.random.default_rng(args.seed)
        k = min(max(1, args.sample_models), len(discovered))
        chosen = sorted(rng.choice(discovered, size=k, replace=False).tolist())
        print(f"[SELF-CHECK] Sampling {k} model(s): {', '.join(chosen)}")
        df_comp = build_comparison_table(args.focus_group, available=chosen)
    else:
        df_comp = build_comparison_table(args.focus_group, available=discovered)

    if df_comp.empty:
        print("✗ No comparison table produced (missing or incompatible artefacts).")
        _tend("step13.total_runtime", t_all)
        return

    # Save artefacts
    csv_path, tex_path = _save_all(df_comp, is_selfcheck=args.selfcheck)
    print(f"\n✓ Artefacts saved: {csv_path.name}, {tex_path.name}")

    # Qualitative summary
    _qualitative_readout(df_comp, args.focus_group)

    # Optional interpretability
    if args.peek_outliers:
        _peek_outliers(_model_candidates(), k=3)

    _tend("step13.total_runtime", t_all)
    print("\n--- Step 13: Mitigation Effectiveness Analysis Completed Successfully ---")


if __name__ == "__main__":
    main()
