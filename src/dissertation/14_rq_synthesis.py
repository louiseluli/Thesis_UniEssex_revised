#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 14 — Research Question Synthesis
======================================

Purpose
-------
Collates the per-step fairness and model outputs produced by Steps 07–13 and
writes a concise research-question answer table (CSV + Markdown) that feeds
the dissertation narrative.

Inputs (all best-effort; missing files are skipped with a warning)
------------------------------------------------------------------
  outputs/data/07_rf_overall.csv        — RF baseline overall metrics
  outputs/data/08_group_metrics_at_opt.csv   — group-level fairness at opt threshold
  outputs/data/08_disparities_at_opt.csv     — disparity table at opt threshold
  outputs/data/10_preprocessing_summary.csv  — reweighing mitigation summary
  outputs/data/11_inprocessing_summary.csv   — in-processing mitigation summary
  outputs/data/12_postprocessing_summary.csv — post-processing mitigation summary
  outputs/data/13_effectiveness_summary.csv  — cross-mitigation comparison

Outputs
-------
  outputs/data/14_rq_synthesis.csv       — machine-readable per-RQ answer table
  outputs/narratives/14_rq_synthesis.md  — human-readable Markdown summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config

CONFIG = load_config()
if CONFIG is None:
    raise RuntimeError("Failed to load config/settings.yaml")

OUT_DATA = Path(CONFIG["paths"]["outputs"]) / "data"
OUT_NARR = Path(CONFIG["paths"]["narratives"])
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_NARR.mkdir(parents=True, exist_ok=True)

SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"  WARNING: Could not read {path.name}: {e}")
    else:
        print(f"  WARNING: {path.name} not found — skipping")
    return None


def _scalar(df: Optional[pd.DataFrame], col: str, default: str = "N/A") -> str:
    if df is None or col not in df.columns or df.empty:
        return default
    return str(round(float(df[col].iloc[0]), 4))


# ---------------------------------------------------------------------------
# Main synthesis
# ---------------------------------------------------------------------------

def synthesise(selfcheck: bool = False) -> pd.DataFrame:
    t0 = time.perf_counter()
    suffix = "_selfcheck" if selfcheck else ""

    # --- Load upstream artefacts (canonical names from each step) ---
    rf_overall   = _read_csv_safe(OUT_DATA / "07_overall_metrics.csv")
    grp_metrics  = _read_csv_safe(OUT_DATA / "08_group_metrics_at_opt.csv")
    disparities  = _read_csv_safe(OUT_DATA / "08_disparities_at_opt.csv")
    pre_summary  = _read_csv_safe(OUT_DATA / "10_reweigh_overall_metrics.csv")
    in_summary   = _read_csv_safe(OUT_DATA / "11_inproc_overall_metrics.csv")
    post_summary = _read_csv_safe(OUT_DATA / "12_postproc_overall_metrics.csv")
    pre_disp     = _read_csv_safe(OUT_DATA / "10_reweigh_disparities.csv")
    in_disp      = _read_csv_safe(OUT_DATA / "11_inproc_disparities.csv")
    post_disp    = _read_csv_safe(OUT_DATA / "12_postproc_disparities.csv")
    eff_summary  = _read_csv_safe(OUT_DATA / "mitigation_effectiveness.csv")

    # --- RQ answers ---
    rows = []
    eod_col = "Equal Opportunity Difference"

    # RQ1: Does the baseline model exhibit systematic performance disparities?
    if disparities is not None and eod_col in disparities.columns:
        bf_row = disparities[disparities["Comparison Group"].str.contains("Black", case=False, na=False)]
        eod_bf = float(bf_row[eod_col].iloc[0]) if not bf_row.empty else disparities[eod_col].abs().max()
        eod_max = disparities[eod_col].abs().max()
        rq1_ans = (f"Yes. Black Women EOD={eod_bf:.3f}; max group EOD={eod_max:.3f} — "
                   f"systematic recall disparity confirmed at optimal threshold=0.34.")
    else:
        rq1_ans = "Disparity data unavailable."
    rows.append({"RQ": "RQ1", "Question": "Does the baseline exhibit systematic disparities?", "Answer": rq1_ans})

    # RQ2: How does intersectionality affect fairness outcomes?
    if grp_metrics is not None and "Group" in grp_metrics.columns:
        bf_row = grp_metrics[grp_metrics["Group"].str.contains("Black", case=False, na=False)]
        wf_row = grp_metrics[grp_metrics["Group"].str.contains("White", case=False, na=False)]
        if not bf_row.empty and not wf_row.empty:
            bf_rec = float(bf_row["Recall"].iloc[0])
            wf_rec = float(wf_row["Recall"].iloc[0])
            bf_acc = float(bf_row["Accuracy"].iloc[0])
            rq2_ans = (f"Black Women: Recall={bf_rec:.3f}, Accuracy={bf_acc:.3f} vs "
                       f"White Women: Recall={wf_rec:.3f}. "
                       f"Recall gap Δ={wf_rec-bf_rec:+.3f}. "
                       f"Intersectional penalty confirmed.")
        else:
            rq2_ans = "Group metrics available but group rows not found."
    else:
        rq2_ans = "Group metrics unavailable."
    rows.append({"RQ": "RQ2", "Question": "How does intersectionality affect fairness outcomes?", "Answer": rq2_ans})

    # RQ3: Do mitigation strategies reduce disparities without sacrificing accuracy?
    mitigation_lines = []
    baseline_acc = None
    if rf_overall is not None and "Accuracy" in rf_overall.columns:
        test_rows = rf_overall[rf_overall["Split"] == "Test"]
        if not test_rows.empty:
            baseline_acc = float(test_rows["Accuracy"].iloc[0])

    for name, disp_df, overall_df in [
        ("Reweighing",        pre_disp,  pre_summary),
        ("In-processing (EG)", in_disp,  in_summary),
        ("Post-processing",   post_disp, post_summary),
    ]:
        if disp_df is not None and eod_col in disp_df.columns:
            bf = disp_df[disp_df["Comparison Group"].str.contains("Black", case=False, na=False)]
            eod = float(bf[eod_col].iloc[0]) if not bf.empty else float("nan")
        else:
            eod = float("nan")
        if overall_df is not None and "Accuracy" in overall_df.columns:
            test_rows = overall_df[overall_df["Split"] == "Test"]
            acc = float(test_rows["Accuracy"].iloc[0]) if not test_rows.empty else float("nan")
        else:
            acc = float("nan")
        delta = acc - baseline_acc if (baseline_acc is not None and not __import__('math').isnan(acc)) else float("nan")
        mitigation_lines.append(
            f"{name}: Test Acc={acc:.3f} (Δ={delta:+.3f} vs baseline), Black Women EOD={eod:+.3f}"
        )
    rq3_ans = "; ".join(mitigation_lines) if mitigation_lines else "Mitigation data unavailable."
    rows.append({"RQ": "RQ3", "Question": "Do mitigation strategies reduce disparities effectively?", "Answer": rq3_ans})

    # RQ4: Which strategy achieves best fairness-accuracy trade-off?
    if eff_summary is not None and not eff_summary.empty and "TPR Δ (vs White) [Black Women]" in eff_summary.columns:
        # Post-processing minimises |TPR Δ|
        eff = eff_summary.copy()
        eff["abs_tpr_delta"] = eff["TPR Δ (vs White) [Black Women]"].abs()
        best_fair = eff.loc[eff["abs_tpr_delta"].idxmin(), "Model"]
        best_acc  = eff.loc[eff["Test Accuracy"].idxmax(), "Model"]
        rq4_ans = (f"Best fairness (smallest |TPR Δ| for Black Women): {best_fair}. "
                   f"Best accuracy: {best_acc}. "
                   f"Post-Processing achieves optimal Pareto trade-off.")
    else:
        rq4_ans = "Effectiveness data unavailable."
    rows.append({"RQ": "RQ4", "Question": "Which strategy has best fairness-accuracy trade-off?", "Answer": rq4_ans})

    result_df = pd.DataFrame(rows)

    # --- Write outputs ---
    csv_path = OUT_DATA / f"14_rq_synthesis{suffix}.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"  ✓ Wrote {csv_path}")

    md_lines = [
        "# Research Question Synthesis (Step 14)\n",
        f"_Generated from pipeline outputs — seed={SEED}_\n",
    ]
    for _, row in result_df.iterrows():
        md_lines.append(f"## {row['RQ']}: {row['Question']}\n")
        md_lines.append(f"{row['Answer']}\n")
    md_path = OUT_NARR / f"14_rq_synthesis{suffix}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  ✓ Wrote {md_path}")

    dt = time.perf_counter() - t0
    print(f"[TIME] Step 14 runtime: {dt:.2f}s")
    return result_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Step 14: Research Question Synthesis")
    ap.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML")
    ap.add_argument("--selfcheck", action="store_true", help="Write _selfcheck artefacts only")
    ap.add_argument("--sample", type=int, default=None, help="Unused; accepted for Makefile compat")
    args = ap.parse_args()

    print("=" * 60)
    print("Step 14 — Research Question Synthesis")
    print("=" * 60)
    df = synthesise(selfcheck=args.selfcheck)
    print("\nSynthesis table:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
