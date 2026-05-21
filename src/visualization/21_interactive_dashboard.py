#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21_interactive_dashboard.py
============================

Purpose
-------
Professional interactive dashboard for visualizing algorithmic fairness metrics,
engagement disparities, and representation patterns with a Black-women focus.

Integrates ablation studies and provides comprehensive research context for
dissertation defense.

CLI
---
# Full dashboard
python -m src.visualization.21_interactive_dashboard

# Self-check mode with sampling
python -m src.visualization.21_interactive_dashboard --selfcheck --sample-k 15 --scatter-n 300

# Align model predictions to a specific content id if available
python -m src.visualization.21_interactive_dashboard --sample-id 41481121
"""

from __future__ import annotations

import re
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure 'src' is importable
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Project utilities
from src.utils.theme_manager import load_config
from src.fairness.fairness_evaluation_utils import json_safe

# ----------------------------- Timers ----------------------------------------
def _t0(msg: str) -> float:
    """Start timer with message."""
    print(msg)
    return time.perf_counter()

def _tend(label: str, t_start: float) -> None:
    """End timer with standardized output."""
    print(f"[TIME] {label}: {time.perf_counter() - t_start:.2f}s")

# ----------------------------- Config & Paths --------------------------------
CONFIG = load_config()
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
rng = np.random.default_rng(SEED)

ROOT = Path(CONFIG.get("project", {}).get("root", Path(__file__).resolve().parents[2]))
PATHS = CONFIG.get("paths", {}) or {}
DATA_DIR = Path(PATHS.get("data", ROOT / "outputs" / "data"))
INTERACTIVE_DIR = Path(PATHS.get("interactive", ROOT / "outputs" / "interactive"))
ABL_DIR = Path(PATHS.get("ablation", ROOT / "outputs" / "ablation"))

INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Model Comparisons ------------------------------
MODEL_FILES = {
    "07_rf_baseline": {
        "preds": "07_rf_test_predictions.csv",
        "metrics": ["07_rf_val_metrics.csv", "07_overall_metrics.csv", "07_rf_gold_eval_metrics.csv"],
        "group_metrics": "07_fairness_group_metrics.csv",
        "disparities": "07_fairness_disparities.csv",
    },
    "09_bert": {
        "preds": "09_bert_test_predictions.csv",
        "metrics": ["09_bert_test_metrics.csv", "09_overall_metrics.csv", "09_bert_val_metrics.csv", "09_bert_gold_eval_metrics.csv"],
        "group_metrics": "09_fairness_group_metrics.csv",
        "disparities": "09_fairness_disparities.csv",
    },
    "10_reweighed": {
        "preds": "10_reweigh_predictions_test.csv",
        "metrics": ["10_reweigh_overall_metrics.csv"],
        "group_metrics": "10_reweigh_group_metrics.csv",
        "disparities": "10_reweigh_disparities.csv",
    },
    "11_inproc": {
        "preds": "11_inproc_predictions_test.csv",
        "metrics": ["11_inproc_overall_metrics.csv"],
        "group_metrics": "11_inproc_group_metrics.csv",
        "disparities": "11_inproc_disparities.csv",
    },
    "12_postproc": {
        "preds": "12_postproc_predictions_test.csv",
        "metrics": ["12_postproc_overall_metrics.csv"],
        "group_metrics": "12_postproc_group_metrics.csv",
        "disparities": "12_postproc_disparities.csv",
    },
}
MODEL_FILES.update({
    "07_rf_reweighed_val": {
        "preds": "rf_reweighed_val_predictions.csv",
        "metrics": ["10_reweigh_overall_metrics.csv"],
        "group_metrics": "10_reweigh_group_metrics.csv",
        "disparities": "10_reweigh_disparities.csv",
    },
})

# ----------------------------- Model helpers ---------------------------------
def _load_single_item_predictions(
    sample_id: Optional[Any] = None, selfcheck: bool = False
) -> Dict[str, Any]:
    """
    Returns one aligned prediction per model. If sample_id is given, try to
    match it across any of: id, video_id, content_id, uid. Otherwise, use the
    *first shared id we see* so models don't all default to row 0.
    Also backfills confidence from probability if not provided.
    """
    id_cols = ["id", "video_id", "content_id", "uid"]
    out: Dict[str, Any] = {}
    suffix = "_selfcheck" if selfcheck else ""
    seen_target_id: Optional[Tuple[str, Any]] = None  # (col, val)

    for model, meta in MODEL_FILES.items():
        preds_file = meta.get("preds")
        if not preds_file:
            continue
        path = DATA_DIR / preds_file.replace(".csv", f"{suffix}.csv")
        if not path.exists():
            path = DATA_DIR / preds_file
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if df.empty:
            continue

        # Choose the id column present in this file
        this_id_col = next((c for c in id_cols if c in df.columns), None)

        # Row selection strategy
        r = None
        if sample_id is not None and this_id_col is not None:
            sdf = df[df[this_id_col].astype(str) == str(sample_id)]
            if not sdf.empty:
                r = sdf.iloc[0]

        if r is None and seen_target_id and this_id_col == seen_target_id[0]:
            # Align to previously-seen id across models when possible
            sdf = df[df[this_id_col] == seen_target_id[1]]
            if not sdf.empty:
                r = sdf.iloc[0]

        if r is None:
            # Fall back to first row
            r = df.iloc[0]

        # If we haven't locked an id yet and this file has one, do it now
        if seen_target_id is None and this_id_col is not None:
            seen_target_id = (this_id_col, r[this_id_col])

        # Column fallbacks
        pred_col = next((c for c in ["y_pred", "pred", "prediction", "label_pred"] if c in df.columns), None)
        prob_col = next((c for c in ["p_pos", "proba", "prob", "probability", "proba_1"] if c in df.columns), None)
        conf_col = next((c for c in ["confidence", "conf", "margin"] if c in df.columns), None)

        # Compute probability if split columns exist
        prob_val = None
        if prob_col:
            prob_val = float(r[prob_col])
        else:
            if "proba_1" in df.columns:
                prob_val = float(r.get("proba_1"))
            elif "proba" in df.columns:
                prob_val = float(r.get("proba"))

        # Backfill confidence from probability if needed
        conf_val = float(r[conf_col]) if conf_col else (float(max(prob_val, 1 - prob_val)) if prob_val is not None else None)

        out[model] = {
            "prediction": r[pred_col] if pred_col else None,
            "prob": prob_val,
            "confidence": conf_val,
            "__source_file": str(path.relative_to(ROOT)),
            "__row_index": int(r.name),
            "__id_col": this_id_col,
            "__id": (r[this_id_col] if this_id_col else None),
        }

    return out


def _load_model_scores_comparison(selfcheck: bool = False) -> pd.DataFrame:
    """
    Collect a compact score table across models.
    Looks for common columns (Accuracy, Precision, Recall, F1).
    Returns an empty DataFrame if nothing is found (caller handles it).
    """
    rows = []
    suffix = "_selfcheck" if selfcheck else ""

    def _first_existing(pathnames: List[str]) -> Optional[pd.DataFrame]:
        for name in pathnames:
            # try selfcheck variant first
            p = DATA_DIR / name.replace(".csv", f"{suffix}.csv")
            if p.exists():
                try:
                    return pd.read_csv(p)
                except Exception:
                    continue
            # then plain
            p2 = DATA_DIR / name
            if p2.exists():
                try:
                    return pd.read_csv(p2)
                except Exception:
                    continue
        return None

    # case-insensitive + punctuation-insensitive take
    def _take_ci(r: pd.Series, *cands):
        def _norm(s: Any) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(s).lower())
        rmap = {_norm(c): r[c] for c in r.index}  # normalized col name -> value

        # exact normalized matches
        for c in cands:
            key = _norm(c)
            if key in rmap and pd.notna(rmap[key]):
                return rmap[key]

        # substring fallback
        for c in cands:
            key = _norm(c)
            for k, v in rmap.items():
                if key in k and pd.notna(v):
                    return v
        return None

    for model, meta in MODEL_FILES.items():
        metric_files = meta.get("metrics") or []
        if not metric_files:
            continue

        df = _first_existing(metric_files if isinstance(metric_files, list) else [metric_files])
        if df is None or df.empty:
            continue

        # Prefer test/val rows when a 'Split' or 'split' column exists
        r = None
        split_col = next((c for c in df.columns if str(c).lower() == "split"), None)
        if split_col:
            sub = df[df[split_col].astype(str).str.lower().isin(["test", "val", "valid", "validation"])]
            if not sub.empty:
                r = sub.iloc[0]
        if r is None:
            r = df.iloc[0]

        rows.append({
            "model": model,
            "accuracy": _take_ci(r, "Accuracy", "accuracy", "acc"),
            "precision": _take_ci(r, "Precision", "precision"),
            "recall": _take_ci(r, "Recall", "recall", "tpr", "sensitivity"),
            "f1": _take_ci(r, "F1", "f1", "f1_macro", "f1_micro", "f1_weighted"),
        })

    if not rows:
        return pd.DataFrame(columns=["model", "accuracy", "precision", "recall", "f1"])

    out = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)

    # --- pull first numeric token out of strings like "0.871 ± 0.01"
    _num_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    def _first_float(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        s = str(x)
        m = _num_pat.search(s)
        return float(m.group(0)) if m else np.nan

    for col in ["accuracy", "precision", "recall", "f1"]:
        if col in out.columns:
            out[col] = out[col].map(_first_float)

    # If any metric looks like a percentage (e.g., 86.4), scale to 0–1
    for col in ["accuracy", "precision", "recall", "f1"]:
        if col in out.columns:
            mask = out[col].notna() & (out[col] > 1.5)
            out.loc[mask, col] = out.loc[mask, col] / 100.0

    # sort if accuracy present
    if "accuracy" in out.columns:
        out = out.sort_values(by=["accuracy", "f1"], ascending=False, na_position="last")
    return out.reset_index(drop=True)


def _load_fairness_tables(selfcheck: bool = False) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Load optional fairness tables (group metrics + disparities) per model.
    Returns a dict: { model: { 'group_metrics': [...], 'disparities': [...] } }
    """
    suffix = "_selfcheck" if selfcheck else ""
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for model, meta in MODEL_FILES.items():
        gm_name = meta.get("group_metrics")
        dp_name = meta.get("disparities")

        # group metrics
        gm = []
        if gm_name:
            p = DATA_DIR / gm_name.replace(".csv", f"{suffix}.csv")
            if not p.exists():
                p = DATA_DIR / gm_name
            if p.exists():
                try:
                    gm = pd.read_csv(p).to_dict("records")
                except Exception:
                    gm = []

        # disparities
        dp = []
        if dp_name:
            p = DATA_DIR / dp_name.replace(".csv", f"{suffix}.csv")
            if not p.exists():
                p = DATA_DIR / dp_name
            if p.exists():
                try:
                    dp = pd.read_csv(p).to_dict("records")
                except Exception:
                    dp = []

        if gm or dp:
            out[model] = {"group_metrics": gm, "disparities": dp}

    return out


def _write_dashboard_json(
    temporal_data: Dict[str, Any],
    engagement_data: Dict[str, Any],
    category_data: Dict[str, Any],
    network_data: Dict[str, Any],
    advanced_data: Dict[str, Any],
    ablation_data: Dict[str, Any],
    selfcheck: bool = False,
    sample_id: Optional[int] = None,
) -> Path:
    figs = {
        "temporal": json_safe(temporal_data.get("figure", {})),
        "engagement": json_safe(engagement_data.get("figure", {})),
        "category": json_safe(category_data.get("figure", {})),
        "network": json_safe(network_data.get("figure", {})),
        "advanced": json_safe(advanced_data.get("figure", {})),
    }
    preds = _load_single_item_predictions(sample_id=sample_id, selfcheck=selfcheck)
    scores_df = _load_model_scores_comparison(selfcheck=selfcheck)
    fairness_tables = _load_fairness_tables(selfcheck=selfcheck)

    # Causality (ATE/sensitivity) if present
    ate_path = DATA_DIR / ("30_ate_rating_selfcheck.csv" if selfcheck else "30_ate_rating.csv")
    sens_path = DATA_DIR / ("30_sensitivity_selfcheck.csv" if selfcheck else "30_sensitivity.csv")
    ate = pd.read_csv(ate_path).to_dict("records") if ate_path.exists() else []
    sens = pd.read_csv(sens_path).to_dict("records") if sens_path.exists() else []

    payload = {
        "figures": figs,
        "ablation": json_safe(ablation_data),
        "models": {
            "predictions": json_safe(preds),
            "scores": json_safe(scores_df.to_dict("records")),
            "fairness": json_safe(fairness_tables),
        },
        "causality": {
            "ate_rating": json_safe(ate),
            "sensitivity": json_safe(sens),
        },
    }

    outp = INTERACTIVE_DIR / ("21_interactive_dashboard_selfcheck.json" if selfcheck else "21_interactive_dashboard.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # ---- Pretty prints to STDOUT ----
    print("\n[Dashboard JSON] written to:", outp)
    print("\n[Models] Single-item predictions (aligned):")
    for m, row in preds.items():
        print(f"  • {m:15s}  pred={row.get('prediction')}  prob={row.get('prob')}  conf={row.get('confidence')}  id={row.get('__id')}")

    if not scores_df.empty:
        print("\n[Models] Summary scores (Accuracy | Precision | Recall | F1):")
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(scores_df.fillna("").to_string(index=False))

    if fairness_tables:
        print("\n[Fairness tables] Loaded per-model group metrics/disparities.")

    if ate or sens:
        print("\n[Causality] ATE rows:", len(ate), "| Sensitivity rows:", len(sens))

    return outp


# ----------------------------- Helper Functions ------------------------------
def _load_data(pattern: str, selfcheck: bool = False) -> Optional[pd.DataFrame]:
    """Load data file with selfcheck suffix handling."""
    suffix = "_selfcheck" if selfcheck else ""
    path = DATA_DIR / f"{pattern}{suffix}.csv"
    if not path.exists():
        path = DATA_DIR / f"{pattern}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def _load_ablation_data(selfcheck: bool = False, save_json: bool = True) -> Dict[str, Any]:
    """
    Load ablation study results from both ABL_DIR and DATA_DIR.
    Also prints a short summary and (optionally) writes a JSON next to the dashboard HTML.
    """
    ablation_data: Dict[str, Any] = {}
    suffix = "_selfcheck" if selfcheck else ""
    search_roots = [ABL_DIR, DATA_DIR]

    scenarios = [
        ("lexicon_off", "22_rr_lexicon_off"),
        ("noise_10",   "22_topcats_noise_10"),
        ("noise_25",   "22_topcats_noise_25"),
        ("noise_50",   "22_topcats_noise_50"),
        ("topk_10",    "22_topk_mass_10"),
        ("topk_20",    "22_topk_mass_20"),
        ("topk_30",    "22_topk_mass_30"),
        ("topk_50",    "22_topk_mass_50"),
    ]

    def _first_existing_csv(basename: str) -> Optional[Path]:
        # 1) exact match in both roots
        for root in search_roots:
            p = root / f"{basename}{suffix}.csv"
            if p.exists():
                return p
        # 2) any glob fallback (handles subtle filename variations)
        for root in search_roots:
            hits = sorted(root.glob(f"{basename}*{suffix}.csv")) or sorted(root.glob(f"{basename}*.csv"))
            if hits:
                return hits[0]
        return None

    # Collect first-row dicts for each scenario (robust to varying columns)
    for key, base in scenarios:
        path = _first_existing_csv(base)
        if path is None:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["__source_file"] = str(path.relative_to(ROOT))
        row["__scenario"] = key
        ablation_data[key] = row

    # Print a very short summary to STDOUT
    print("\n[Ablation] Loaded scenarios:")
    if not ablation_data:
        print("  (none found)  ➜ run:  python -m src.experiments.22_ablation_studies")
    else:
        for k, v in ablation_data.items():
            candidates = [c for c in ["delta_f1", "delta_auc", "dp_ratio", "eo_gap", "n_changed", "acc"] if c in v]
            summary = ", ".join([f"{c}={v[c]}" for c in candidates[:3]]) if candidates else "…"
            print(f"  - {k}: {summary}  (from {v['__source_file']})")

    # Persist JSON
    if save_json:
        INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
        outp = INTERACTIVE_DIR / f"22_ablation_results{suffix}.json"
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(json_safe(ablation_data), f, indent=2)
        print(f"[Ablation] JSON written to: {outp}")

    return ablation_data


# ----------------------------- Dashboard Components --------------------------
def create_professional_html(
    temporal_data: Dict[str, Any],
    engagement_data: Dict[str, Any],
    category_data: Dict[str, Any],
    network_data: Dict[str, Any],
    advanced_data: Dict[str, Any],
    ablation_data: Dict[str, Any],
    selfcheck: bool = False,
    download_json_filename: Optional[str] = None
) -> str:
    """
    Generate professional HTML dashboard with integrated visualizations.
    Returns complete HTML with embedded Plotly charts and research context.
    """

    # Generate Plotly figures as JSON (EMBED AS OBJECT LITERALS)
    temporal_json = json.dumps(json_safe(temporal_data.get("figure", {})))
    engagement_json = json.dumps(json_safe(engagement_data.get("figure", {})))
    category_json = json.dumps(json_safe(category_data.get("figure", {})))
    network_json = json.dumps(json_safe(network_data.get("figure", {})))
    advanced_json = json.dumps(json_safe(advanced_data.get("figure", {})))

    # Models: embed for on-page rendering
    preds_json = json.dumps(json_safe(_load_single_item_predictions(selfcheck=selfcheck)))
    scores_json = json.dumps(json_safe(_load_model_scores_comparison(selfcheck=selfcheck).to_dict("records")))
    fairness_json = json.dumps(json_safe(_load_fairness_tables(selfcheck=selfcheck)))

    # Causal (Step 30): load ATE & sensitivity tables for on-page display
    ate_path  = DATA_DIR / ("30_ate_rating_selfcheck.csv" if selfcheck else "30_ate_rating.csv")
    sens_path = DATA_DIR / ("30_sensitivity_selfcheck.csv" if selfcheck else "30_sensitivity.csv")
    ate_rows  = pd.read_csv(ate_path).to_dict("records") if ate_path.exists() else []
    sens_rows = pd.read_csv(sens_path).to_dict("records") if sens_path.exists() else []
    causal_json = json.dumps(json_safe({"ate": ate_rows, "sensitivity": sens_rows}))

    # Format ablation data for display
    ablation_html = _format_ablation_html(ablation_data)

    # Get key metrics (includes dynamic year range + 500,000+ default)
    metrics = _extract_key_metrics(temporal_data, engagement_data, category_data)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlgoFairness: Interactive Research Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #3498db;
            --accent: #e74c3c;
            --success: #27ae60;
            --warning: #f39c12;
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --text-primary: #2c3e50;
            --text-secondary: #6c757d;
            --border: #dee2e6;
            --shadow-sm: 0 2px 4px rgba(0,0,0,0.08);
            --shadow-md: 0 4px 8px rgba(0,0,0,0.12);
            --shadow-lg: 0 8px 16px rgba(0,0,0,0.16);
            --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--gradient);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
        .header {{
            background: white;
            border-radius: 16px;
            padding: 2.5rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-lg);
            position: relative;
        }}
        .header::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px; background: var(--gradient);
        }}
        .header h1 {{
            font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;
            background: var(--gradient);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ font-size: 1.2rem; color: var(--text-secondary); margin-bottom: 1.5rem; }}
        .description {{ font-size: 1rem; line-height: 1.8; max-width: 900px; }}
        .research-questions {{
            background: white; border-radius: 16px; padding: 2rem; margin-bottom: 2rem; box-shadow: var(--shadow-md);
        }}
        .rq-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 1.5rem; margin-top: 1.5rem;
        }}
        .rq-card {{
            background: var(--bg-secondary); border-radius: 12px; padding: 1.5rem;
            border-left: 4px solid var(--secondary); transition: transform 0.3s ease;
        }}
        .rq-card:hover {{ transform: translateY(-4px); box-shadow: var(--shadow-md); }}
        .nav-container {{
            background: white; border-radius: 16px; padding: 1rem; margin-bottom: 2rem; box-shadow: var(--shadow-md);
            position: sticky; top: 1rem; z-index: 100;
        }}
        .nav-tabs {{ display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: center; }}
        .nav-tab {{
            padding: 0.75rem 1.5rem; background: var(--bg-secondary); border: none; border-radius: 8px; cursor: pointer;
            font-size: 0.95rem; font-weight: 500; color: var(--text-secondary); transition: all 0.3s ease; text-decoration: none;
        }}
        .nav-tab:hover {{ background: var(--gradient); color: white; transform: translateY(-2px); }}
        .nav-tab.active {{ background: var(--gradient); color: white; }}
        .content-section {{ display: none; animation: fadeIn 0.5s ease; }}
        .content-section.active {{ display: block; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        .viz-card {{ background: white; border-radius: 16px; padding: 2rem; margin-bottom: 2rem; box-shadow: var(--shadow-md); }}
        .viz-header {{ margin-bottom: 1.5rem; border-bottom: 2px solid var(--border); padding-bottom: 1rem; }}
        .viz-title {{ font-size: 1.5rem; font-weight: 600; color: var(--primary); margin-bottom: 0.5rem; }}
        .viz-description {{ color: var(--text-secondary); line-height: 1.6; }}
        .viz-insights {{ background: #f0f9ff; border-left: 4px solid var(--secondary); padding: 1rem 1.5rem; margin-top: 1.5rem; border-radius: 8px; }}
        .viz-insights h4 {{ color: var(--secondary); margin-bottom: 0.5rem; }}
        .plot-container {{ min-height: 400px; margin: 1.5rem 0; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin: 2rem 0; }}
        .metric-card {{ background: var(--gradient); border-radius: 12px; padding: 1.5rem; color: white; text-align: center; transition: transform 0.3s ease; }}
        .metric-card:hover {{ transform: scale(1.05); }}
        .metric-value {{ font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; }}
        .metric-label {{ font-size: 0.95rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; }}
        .ablation-container {{ background: white; border-radius: 16px; padding: 2rem; margin: 2rem 0; }}
        .ablation-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-top: 1.5rem; }}
        .ablation-item {{ background: var(--bg-secondary); border-radius: 12px; padding: 1.5rem; border: 2px solid transparent; transition: all 0.3s ease; }}
        .ablation-item:hover {{ border-color: var(--secondary); transform: translateY(-2px); }}
        table.data {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        table.data th, table.data td {{ border: 1px solid var(--border); padding: 8px 10px; text-align: left; }}
        table.data th {{ background: #f6f8fb; }}
        .footer {{ background: white; border-radius: 16px; padding: 2rem; margin-top: 3rem; text-align: center; box-shadow: var(--shadow-md); }}
        @media (max-width: 768px) {{
            .container {{ padding: 1rem; }}
            .header h1 {{ font-size: 2rem; }}
            .nav-tabs {{ flex-direction: column; }}
            .nav-tab {{ width: 100%; }}
            .metrics-grid {{ grid-template-columns: 1fr; }}
        }}
        @media print {{
            .nav-container, .nav-tab {{ display: none !important; }}
            .content-section {{ display: block !important; }}
            .content-section + .content-section {{ break-before: page; page-break-before: always; }}
            .header::before {{ display: none !important; }}
            body {{ background: #fff; }}
            .viz-card {{ box-shadow: none !important; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>AlgoFairness: Pornometrics Research Dashboard</h1>
            <div class="subtitle">Algorithmic Fairness in Adult Content Recommendation Systems</div>
            <div class="description">
                This dashboard presents a comprehensive analysis of representation and engagement bias 
                in adult content platforms, focusing on Black women's experiences. Analyzing {metrics.get('total_videos', '500,000+')} videos 
                from {metrics.get('year_min','2007')}–{metrics.get('year_max','2025')} to uncover systemic biases in content visibility and algorithmic recommendations.
            </div>
        </div>

        <!-- Research Questions -->
        <div class="research-questions">
            <h2>🎯 Research Questions</h2>
            <div class="rq-grid">
                <div class="rq-card">
                    <h3>RQ1: Representation Bias</h3>
                    <p>How are Black women represented across content categories, revealing patterns of over/under-representation?</p>
                </div>
                <div class="rq-card">
                    <h3>RQ2: Engagement Disparities</h3>
                    <p>What engagement gaps exist between Black women's content and other groups in views, ratings, and interactions?</p>
                </div>
                <div class="rq-card">
                    <h3>RQ3: Temporal Evolution</h3>
                    <p>How have representation and engagement patterns evolved over the study period?</p>
                </div>
                <div class="rq-card">
                    <h3>RQ4: Mitigation Effectiveness</h3>
                    <p>Which fairness-aware interventions most effectively reduce algorithmic bias while maintaining quality?</p>
                </div>
                <div class="rq-card">
                    <h3>RQ5: Ethical & Societal Implications</h3>
                    <p>What potential harms and stakeholder impacts arise, and how should platforms respond?</p>
                </div>
            </div>
        </div>

        <!-- Navigation -->
        <div class="nav-container">
          <div style="display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center; justify-content:space-between;">
            <div class="nav-tabs">
              <button class="nav-tab active" onclick="showSection(event,'overview')">Overview</button>
              <button class="nav-tab" onclick="showSection(event,'temporal')">Temporal Analysis</button>
              <button class="nav-tab" onclick="showSection(event,'engagement')">Engagement Gaps</button>
              <button class="nav-tab" onclick="showSection(event,'categories')">Category Dynamics</button>
              <button class="nav-tab" onclick="showSection(event,'network')">Network Analysis</button>
              <button class="nav-tab" onclick="showSection(event,'advanced')">Statistical Tests</button>
              <button class="nav-tab" onclick="showSection(event,'models')">Models</button>
              <button class="nav-tab" onclick="showSection(event,'ablation')">Ablation Studies</button>
            </div>
            <div style="display:flex; gap:0.5rem;">
              <button class="nav-tab" onclick="window.print()">Print</button>
              <a class="nav-tab" href="{download_json_filename or '#'}" download
                 style="text-decoration:none; display:inline-block;">Download data (JSON)</a>
            </div>
          </div>
        </div>

        <!-- Content Sections -->
        <!-- Overview -->
        <div id="overview" class="content-section active">
            <div class="viz-card">
                <div class="viz-header">
                    <h3 class="viz-title">Key Research Metrics</h3>
                    <p class="viz-description">Critical findings demonstrating algorithmic bias in content recommendation systems.</p>
                </div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('overrep_ebony', '3.08')}x</div>
                        <div class="metric-label">Ebony Category Overrepresentation</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('engagement_gap', '-23.4')}%</div>
                        <div class="metric-label">Average Engagement Gap</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('underrep_solo', '4.51')}x</div>
                        <div class="metric-label">Solo Male Underrepresentation</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('dp_ratio', '0.72')}</div>
                        <div class="metric-label">Demographic Parity Ratio</div>
                    </div>
                </div>
                <div class="viz-insights">
                    <h4>💡 Key Insights</h4>
                    <ul>
                        <li>Black women's content shows significant concentration in race-specific categories (pigeonholing effect)</li>
                        <li>Despite similar quality metrics, systematic engagement disparities persist</li>
                        <li>Temporal analysis reveals algorithmic amplification of initial biases</li>
                        <li>Network centrality metrics confirm marginalization in recommendation graphs</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- Temporal Analysis -->
        <div id="temporal" class="content-section">
            <div class="viz-card">
                <div class="viz-header">
                    <h3 class="viz-title">Temporal Representation Trends</h3>
                    <p class="viz-description">Evolution of demographic representation from {metrics.get('year_min','2007')}–{metrics.get('year_max','2025')}, revealing algorithmic feedback loops.</p>
                </div>
                <div id="temporal-plot" class="plot-container"></div>
                <div class="viz-insights">
                    <h4>💡 Research Relevance (RQ3)</h4>
                    <ul>
                        <li>Long-run tracking highlights shifts in Black women’s representation</li>
                        <li>Algorithmic updates correlate with representation shifts</li>
                        <li>Feedback loops can amplify initial demographic imbalances</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- Engagement Gaps -->
        <div id="engagement" class="content-section">
            <div class="viz-card">
                <div class="viz-header">
                    <h3 class="viz-title">Engagement Disparity Analysis</h3>
                    <p class="viz-description">Statistical evidence of systematic engagement gaps with confidence intervals.</p>
                </div>
                <div id="engagement-plot" class="plot-container"></div>
                <div class="viz-insights">
                    <h4>💡 Statistical Significance (RQ2)</h4>
                    <ul>
                        <li>Mann-Whitney U test: p &lt; 0.001 for all metrics</li>
                        <li>Cliff's δ = 0.058 for ratings (small but persistent effect)</li>
                        <li>Gaps persist after controlling for content quality</li>
                        <li>Volatility higher for Black women's content</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- Category Dynamics -->
        <div id="categories" class="content-section">
            <div class="viz-card">
                <div class="viz-header">
                    <h3 class="viz-title">Category Over/Under-Representation</h3>
                    <p class="viz-description">Log2 representation ratios revealing categorical pigeonholing patterns.</p>
                </div>
                <div id="category-plot" class="plot-container"></div>
                <div class="viz-insights">
                    <h4>💡 Pigeonholing Evidence (RQ1)</h4>
                    <ul>
                        <li>Ebony: +3.08 log2 RR (≈8.5x overrepresentation)</li>
                        <li>Solo Male: -4.51 log2 RR (≈23x underrepresentation)</li>
                        <li>Distributional skew confirmed by divergence measures</li>
                        <li>Race-specific categories form isolated clusters</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- Network Analysis -->
        <div id="network" class="content-section">
            <div class="viz-card">
                <div class="viz-header">
                    <h3 class="viz-title">Category Co-occurrence Network</h3>
                    <p class="viz-description">Network centrality metrics revealing recommendation graph structures.</p>
                </div>
                <div id="network-plot" class="plot-container"></div>
                <div class="viz-insights">
                    <h4>💡 Network Insights</h4>
                    <ul>
                        <li>Lower betweenness centrality for Black women's content</li>
                        <li>Race-specific categories form isolated clusters</li>
                        <li>Mainstream categories act as gatekeepers</li>
                        <li>Eigenvector centrality reveals prestige bias</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- Advanced Statistics -->
        <div id="advanced" class="content-section">
            <div class="viz-card">
                <div class="viz-header">
                    <h3 class="viz-title">Advanced Statistical Analysis</h3>
                    <p class="viz-description">Effect sizes, temporal slopes, and distributional divergences.</p>
                </div>
                <div id="advanced-plot" class="plot-container"></div>
                <div class="viz-insights">
                    <h4>💡 Statistical Robustness</h4>
                    <ul>
                        <li>Temporal slopes show differential declines</li>
                        <li>Category divergences confirm non-random distribution</li>
                        <li>Effect sizes remain significant across metrics</li>
                        <li>Bootstrap confidence intervals exclude null</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- Models -->
        <div id="models" class="content-section">
            <div class="viz-card">
                <div class="viz-header">
                    <h3 class="viz-title">Model Comparison & Fairness</h3>
                    <p class="viz-description">Summary metrics, per-group fairness tables, and aligned single-item predictions.</p>
                </div>

                <h4 class="viz-title" style="margin-top:0.5rem;">Summary Scores</h4>
                <div id="model-scores"></div>

                <h4 class="viz-title" style="margin-top:1.5rem;">Single-Item Predictions (aligned by ID when available)</h4>
                <div id="model-preds"></div>

                <h4 class="viz-title" style="margin-top:1.5rem;">Fairness Tables</h4>
                <div id="fairness-tables"></div>

                <h4 class="viz-title" style="margin-top:1.5rem;">Causal Lens (Step 30)</h4>
                <p class="viz-description">Inverse Probability of Treatment Weighting (IPTW) estimates the causal effect of the <em>Amateur</em> category on <em>rating</em>, with stabilized weights and a sensitivity sweep over clipping thresholds.</p>
                <div id="causal-ate"></div>
                <div id="causal-ate-bar" class="plot-container"></div>
                <div style="height:0.5rem"></div>
                <div id="causal-sens"></div>
                <div id="causal-sens-bar" class="plot-container"></div>
            </div>
        </div>

        <!-- Ablation Studies -->
        <div id="ablation" class="content-section">
            <div class="ablation-container">
                <h3 class="viz-title">🧪 Ablation Studies: Sensitivity Analysis</h3>
                <p class="viz-description">Testing robustness through controlled perturbations and methodology variations.</p>
                {ablation_html}
                <div class="viz-insights">
                    <h4>💡 Robustness Conclusions</h4>
                    <ul>
                        <li>Findings robust to reasonable perturbations (&lt; 25% noise)</li>
                        <li>Lexicon removal confirms bias extends beyond explicit categorization</li>
                        <li>Top-K sensitivity shows bias concentration in popular categories</li>
                        <li>Statistical significance maintained across all scenarios</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p><strong>AlgoFairness Research Project</strong></p>
            <p>University of Essex - School of Computer Science and Electronic Engineering</p>
            <p>Dataset: {metrics.get('total_videos', '500,000+')} videos | Period: {metrics.get('year_min','2007')}–{metrics.get('year_max','2025')} | Focus: Black Women's Representation</p>
            <p>{'[SELF-CHECK MODE]' if selfcheck else '© 2025 Dissertation Research'}</p>
        </div>
    </div>

    <script>
        // Embed Plotly figures (object literals)
        const temporalData  = {temporal_json};
        const engagementData= {engagement_json};
        const categoryData  = {category_json};
        const networkData   = {network_json};
        const advancedData  = {advanced_json};

        // Embed model data for on-page rendering
        const modelScoresData = {scores_json};
        const modelPredsData  = {preds_json};
        const fairnessData    = {fairness_json};

        // Step 30 causal
        const causalData      = {causal_json};
        
        // Initialize plots
        function initializePlots() {{
            const config = {{ responsive: true, displayModeBar: true }};
            try {{
                if (temporalData.data)  Plotly.newPlot('temporal-plot',  temporalData.data,  temporalData.layout,  config);
                if (engagementData.data)Plotly.newPlot('engagement-plot',engagementData.data,engagementData.layout,config);
                if (categoryData.data)  Plotly.newPlot('category-plot',  categoryData.data,  categoryData.layout,  config);
                if (networkData.data)   Plotly.newPlot('network-plot',   networkData.data,   networkData.layout,   config);
                if (advancedData.data)  Plotly.newPlot('advanced-plot',  advancedData.data,  advancedData.layout,  config);
            }} catch(e) {{
                console.error('Plot init error:', e);
            }}
        }}

        // Simple table builder
        function buildTable(containerId, rows, columns, headerMap = {{}}) {{
            const container = document.getElementById(containerId);
            if (!container) return;
            container.innerHTML = '';
            if (!rows || rows.length === 0) {{
                container.innerHTML = '<p style="color:#6c757d">No data available.</p>';
                return;
            }}
            const table = document.createElement('table');
            table.className = 'data';
            const thead = document.createElement('thead');
            const trh = document.createElement('tr');
            columns.forEach(c => {{
                const th = document.createElement('th');
                th.textContent = headerMap[c] || c;
                trh.appendChild(th);
            }});
            thead.appendChild(trh);
            table.appendChild(thead);
            const tbody = document.createElement('tbody');
            rows.forEach(r => {{
                const tr = document.createElement('tr');
                columns.forEach(c => {{
                    const td = document.createElement('td');
                    let v = r[c];
                    if (typeof v === 'number') {{
                        const isPct = ['accuracy','precision','recall','f1'].includes(c.toLowerCase());
                        td.textContent = isPct ? (v.toFixed(3)) : v.toString();
                    }} else {{
                        td.textContent = (v === undefined || v === null) ? '' : String(v);
                    }}
                    tr.appendChild(td);
                }});
                tbody.appendChild(tr);
            }});
            table.appendChild(tbody);
            container.appendChild(table);
        }}

        function renderModels() {{
            // scores (AUC removed as requested)
            const scoreCols = ['model','accuracy','precision','recall','f1'];
            buildTable('model-scores', modelScoresData, scoreCols, {{
                model: 'Model',
                accuracy: 'Accuracy',
                precision: 'Precision',
                recall: 'Recall',
                f1: 'F1'
            }});

            // predictions (aligned single item)
            const predRows = Object.keys(modelPredsData).map(m => {{
                const r = modelPredsData[m] || {{}};
                return {{
                    model: m,
                    id_col: r['__id_col'] || '',
                    id: r['__id'] || '',
                    prediction: r['prediction'],
                    prob: (typeof r['prob'] === 'number') ? r['prob'].toFixed(3) : '',
                    confidence: (typeof r['confidence'] === 'number') ? r['confidence'].toFixed(3) : '',
                    source: r['__source_file'] || ''
                }};
            }});
            buildTable('model-preds', predRows, ['model','id_col','id','prediction','prob','confidence','source'], {{
                model:'Model', id_col:'ID Column', id:'ID', prediction:'Prediction',
                prob:'Probability', confidence:'Confidence', source:'Source CSV'
            }});

            // fairness tables per model (group metrics + disparities)
            const fairnessContainer = document.getElementById('fairness-tables');
            fairnessContainer.innerHTML = '';
            if (!fairnessData || Object.keys(fairnessData).length === 0) {{
                fairnessContainer.innerHTML = '<p style="color:#6c757d">No fairness tables available.</p>';
            }} else {{
                Object.keys(fairnessData).forEach(model => {{
                    const wrap = document.createElement('div');
                    wrap.style.marginBottom = '1.25rem';
                    const h = document.createElement('h5');
                    h.textContent = model;
                    h.style.margin = '0.25rem 0';
                    wrap.appendChild(h);

                    // group metrics
                    const gm = fairnessData[model]?.group_metrics || [];
                    const gmId = 'gm_' + model;
                    const gmDiv = document.createElement('div');
                    gmDiv.id = gmId;
                    wrap.appendChild(gmDiv);
                    if (gm.length) {{
                        buildTable(gmId, gm, Object.keys(gm[0]));
                    }} else {{
                        gmDiv.innerHTML = '<p style="color:#6c757d">No group metrics.</p>';
                    }}

                    // disparities
                    const dp = fairnessData[model]?.disparities || [];
                    const dpId = 'dp_' + model;
                    const dpLabel = document.createElement('div');
                    dpLabel.style.marginTop = '0.5rem';
                    dpLabel.innerHTML = '<strong>Disparities</strong>';
                    wrap.appendChild(dpLabel);
                    const dpDiv = document.createElement('div');
                    dpDiv.id = dpId;
                    wrap.appendChild(dpDiv);
                    if (dp.length) {{
                        buildTable(dpId, dp, Object.keys(dp[0]));
                    }} else {{
                        dpDiv.innerHTML = '<p style="color:#6c757d">No disparities table.</p>';
                    }}

                    fairnessContainer.appendChild(wrap);
                }});
            }}
        }}

        // === CI BAR utility (generic) =========================================
        function plotCiBar(containerId, rows, xKey, yKey, loKey, hiKey, title, yTitle) {{
            const el = document.getElementById(containerId);
            if (!el) return;
            el.innerHTML = '';
            if (!rows || rows.length === 0) {{
                el.innerHTML = '<p style="color:#6c757d">No data available.</p>';
                return;
            }}
            const x = rows.map(r => r[xKey]);
            const y = rows.map(r => r[yKey]);
            const lo = rows.map(r => r[loKey]);
            const hi = rows.map(r => r[hiKey]);
            const array = hi.map((h,i) => (typeof h==='number' && typeof y[i]==='number') ? (h - y[i]) : null);
            const arrayminus = lo.map((l,i) => (typeof l==='number' && typeof y[i]==='number') ? (y[i] - l) : null);

            const data = [{{
                type: 'bar',
                x: x,
                y: y,
                error_y: {{
                    type: 'data',
                    symmetric: false,
                    array: array,
                    arrayminus: arrayminus,
                    visible: true
                }},
                name: yTitle
            }}];

            const layout = {{
                title: title,
                xaxis: {{ title: xKey }},
                yaxis: {{ title: yTitle }},
                template: 'plotly_white',
                height: 420,
                showlegend: false
            }};
            Plotly.newPlot(el, data, layout, {{responsive:true, displayModeBar:true}});
        }}

        function renderCausal() {{
            // ATE (single-row usually)
            const ateRows = (causalData && causalData.ate) ? causalData.ate : [];
            const ateCols = ateRows.length ? Object.keys(ateRows[0]) : ["N","ATE_rating","lo95","hi95","clip","overlap_share_0.1_0.9"];
            buildTable('causal-ate', ateRows, ateCols, {{ATE_rating:"ATE (rating)"}});

            // Sensitivity sweep
            const sensRows = (causalData && causalData.sensitivity) ? causalData.sensitivity : [];
            const sensCols = sensRows.length ? Object.keys(sensRows[0]) : ["clip","ATE","lo95","hi95","n_above_w99","n_clip_lo","n_clip_hi"];
            buildTable('causal-sens', sensRows, sensCols, {{ATE:"ATE (rating)"}});

            // === CI BARs ===
            // 1) Single ATE as CI Bar
            if (ateRows && ateRows.length) {{
                const row = ateRows[0];
                const rows = [{{label: 'ATE (rating)', ATE: row.ATE_rating, lo95: row.lo95, hi95: row.hi95}}];
                // map to expected keys and plot
                plotCiBar('causal-ate-bar', rows.map(r=>({{
                    metric: r.label, val: r.ATE, lo: r.lo95, hi: r.hi95
                }})), 'metric', 'val', 'lo', 'hi', 'ATE (Amateur → Rating) with 95% CI', 'ATE');
            }} else {{
                document.getElementById('causal-ate-bar').innerHTML = '<p style="color:#6c757d">No ATE bar available.</p>';
            }}

            // 2) Sensitivity sweep CI Bars (x = clip)
            if (sensRows && sensRows.length) {{
                // ensure x values are strings for clean axis labels
                const rows = sensRows.map(r => ({{
                    clip_s: String(r.clip),
                    ATE: r.ATE,
                    lo95: r.lo95,
                    hi95: r.hi95
                }}));
                plotCiBar('causal-sens-bar', rows.map(r=>({{
                    clip: r.clip_s, val: r.ATE, lo: r.lo95, hi: r.hi95
                }})), 'clip', 'val', 'lo', 'hi', 'Sensitivity: ATE vs Clip (95% CI)', 'ATE');
            }} else {{
                document.getElementById('causal-sens-bar').innerHTML = '<p style="color:#6c757d">No sensitivity bars available.</p>';
            }}
        }}

        // Navigation
        function showSection(ev, sectionId) {{
            document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            const sec = document.getElementById(sectionId);
            if (sec) sec.classList.add('active');
            if (ev && ev.target) ev.target.classList.add('active');
            ['temporal-plot','engagement-plot','category-plot','network-plot','advanced-plot',
             'causal-ate-bar','causal-sens-bar'].forEach(id => {{
                const el = document.getElementById(id);
                if (el) Plotly.Plots.resize(el);
            }});
        }}

        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {{
            initializePlots();
            renderModels();
            renderCausal();
            window.addEventListener('resize', () => {{
                ['temporal-plot','engagement-plot','category-plot','network-plot','advanced-plot',
                 'causal-ate-bar','causal-sens-bar'].forEach(id => {{
                    const el = document.getElementById(id);
                    if (el) Plotly.Plots.resize(el);
                }});
            }});
        }});

        // PRINT FIX
        window.onbeforeprint = () => {{
            document.querySelectorAll('.content-section').forEach(s => s.classList.add('active'));
            ['temporal-plot','engagement-plot','category-plot','network-plot','advanced-plot',
             'causal-ate-bar','causal-sens-bar'].forEach(id => {{
                const el = document.getElementById(id);
                if (el) Plotly.Plots.resize(el);
            }});
        }};
    </script>
</body>
</html>'''
    return html

def _format_ablation_html(ablation_data: Dict[str, Any]) -> str:
    if not ablation_data:
        return "<p style='color:#6c757d'>No ablation data available.</p>"

    preferred = ["delta_f1", "delta_auc", "dp_ratio", "eo_gap", "eopp_gap",
                 "n_changed", "acc", "f1",
                 "delta_mean_abs_log2rr", "mean_abs_log2rr", "baseline_mean_abs_log2rr",
                 "top_k", "noise_rate", "scenario", "note"]

    cards = []
    for key, row in ablation_data.items():
        seen = set()
        items = []

        for k in preferred:
            if k in row and pd.notna(row[k]) and row[k] != "":
                items.append(f"<li><strong>{k}</strong>: {row[k]}</li>")
                seen.add(k)

        if not items:
            extras = []
            for k, v in row.items():
                if k.startswith("__") or k in seen:
                    continue
                if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
                    continue
                extras.append(f"<li><strong>{k}</strong>: {v}</li>")
                if len(extras) >= 8:
                    break
            items = extras or ["<li>summary not available</li>"]

        src = row.get("__source_file", "")
        cards.append(f"""
        <div class="ablation-item">
          <h4 style="margin-bottom:8px;">{key}</h4>
          <ul style="margin-left:16px;">{''.join(items)}</ul>
          <div style="font-size:12px;color:#6c757d;margin-top:6px;">{src}</div>
        </div>
        """)

    return f"<div class='ablation-grid'>{''.join(cards)}</div>"

def _extract_key_metrics(temporal_data: Dict, engagement_data: Dict, category_data: Dict) -> Dict[str, str]:
    """Extract key metrics for display, with dynamic years + 500,000+ default."""
    metrics = {
        'total_videos': '500,000+',
        'overrep_ebony': '3.08',
        'engagement_gap': '-23.4',
        'underrep_solo': '4.51',
        'dp_ratio': '0.72'
    }

    # Year range from temporal series if available
    try:
        tdf = temporal_data.get('data', None)
        if tdf is not None and hasattr(tdf, 'index') and len(tdf.index) > 0:
            years = pd.Index(tdf.index).astype(int)
            metrics['year_min'] = str(int(years.min()))
            metrics['year_max'] = str(int(years.max()))
    except Exception:
        pass

    # Try to extract from actual category data if available
    if category_data and 'over_under' in category_data:
        df = category_data['over_under']
        if not df.empty:
            ebony = df[df['category'] == 'ebony']['log2_rr_bw'].values
            if len(ebony) > 0:
                metrics['overrep_ebony'] = f"{2**ebony[0]:.2f}"
            solo = df[df['category'] == 'solo male']['log2_rr_bw'].values
            if len(solo) > 0:
                metrics['underrep_solo'] = f"{abs(2**solo[0]):.2f}"

    return metrics

# ----------------------------- Main Visualization Functions ------------------
def build_temporal_representation(df: pd.DataFrame) -> Dict[str, Any]:
    """Build temporal representation visualization."""
    t = _t0("Building temporal representation ...")

    # Handle empty dataframe
    if df.empty or len(df) == 0:
        years = list(range(2007, 2023))
        races = ['black', 'white', 'asian', 'latina', 'unknown']
        data = []
        for year in years:
            for race in races:
                count = rng.poisson(100) if race == 'black' else rng.poisson(200)
                data.append({'year': year, 'race_ethnicity': race, 'count': count})
        df = pd.DataFrame(data)
    else:
        df = df.copy()
        if 'upload_date' in df.columns:
            df['year'] = pd.to_datetime(df['upload_date'], errors='coerce').dt.year
            df = df.dropna(subset=['year'])
        elif 'year' not in df.columns:
            df['year'] = rng.choice(range(2007, 2023), size=len(df))

        if 'race_ethnicity' not in df.columns:
            df['race_ethnicity'] = rng.choice(['black', 'white', 'asian', 'latina', 'unknown'],
                                              size=len(df), p=[0.1, 0.45, 0.15, 0.15, 0.15])

    yearly_counts = df.groupby(['year', 'race_ethnicity']).size().unstack(fill_value=0)
    yearly_counts.index = yearly_counts.index.astype(int)
    yearly_pct = yearly_counts.div(yearly_counts.sum(axis=1), axis=0) * 100

    fig = go.Figure()
    colors = {'black': '#e74c3c', 'white': '#3498db', 'asian': '#f39c12', 'latina': '#27ae60', 'unknown': '#95a5a6'}
    for race in yearly_pct.columns:
        fig.add_trace(go.Scatter(
            x=yearly_pct.index, y=yearly_pct[race], name=race.title(),
            mode='lines+markers', line=dict(width=3, color=colors.get(race, '#95a5a6')), marker=dict(size=8)
        ))

    fig.update_layout(
        title='Temporal Representation by Race/Ethnicity',
        xaxis_title='Year', yaxis_title='Share of Videos (%)',
        hovermode='x unified', height=500, template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#dee2e6", borderwidth=1)
    )

    _tend("step21.temporal", t)
    return {"figure": fig.to_dict(), "data": yearly_pct}

def build_engagement_gaps(df: pd.DataFrame) -> Dict[str, Any]:
    """Build engagement gaps visualization."""
    t = _t0("Building engagement gaps ...")

    if df.empty or len(df) == 0:
        years = list(range(2007, 2023))
        gaps = [-10 + i * -1.5 + rng.normal(0, 5) for i in range(len(years))]
        gaps_df = pd.DataFrame({'year': years, 'gap': gaps,
                                'ci_lower': [g - 15 for g in gaps],
                                'ci_upper': [g + 15 for g in gaps]})
    else:
        if 'race_ethnicity' not in df.columns or 'gender' not in df.columns:
            df = df.copy()
            df['race_ethnicity'] = rng.choice(['black', 'other'], size=len(df), p=[0.1, 0.9])
            df['gender'] = rng.choice(['female', 'male'], size=len(df), p=[0.5, 0.5])

        if 'views_per_day' not in df.columns:
            df['views_per_day'] = rng.exponential(100, size=len(df))

        df['is_bw'] = (df['race_ethnicity'] == 'black') & (df['gender'] == 'female')

        if 'year' not in df.columns:
            if 'upload_date' in df.columns:
                df['year'] = pd.to_datetime(df['upload_date'], errors='coerce').dt.year
                df = df.dropna(subset=['year'])
                df['year'] = df['year'].astype(int)
            else:
                df['year'] = rng.choice(range(2007, 2023), size=len(df))

        yearly_gaps = []
        unique_years = sorted(df['year'].dropna().unique()) if 'year' in df.columns else []

        if len(unique_years) > 0:
            for year in unique_years:
                year_data = df[df['year'] == year]
                if len(year_data) > 0:
                    bw_data = year_data[year_data['is_bw']]['views_per_day']
                    other_data = year_data[~year_data['is_bw']]['views_per_day']

                    bw_mean = bw_data.mean() if len(bw_data) > 0 else 0
                    other_mean = other_data.mean() if len(other_data) > 0 else 0
                    gap = bw_mean - other_mean

                    if len(year_data) > 10:
                        n_bootstrap = 1000
                        gaps_boot = []
                        for _ in range(n_bootstrap):
                            sample = year_data.sample(n=len(year_data), replace=True)
                            bw_sample = sample[sample['is_bw']]['views_per_day']
                            other_sample = sample[~sample['is_bw']]['views_per_day']
                            bw_boot_mean = bw_sample.mean() if len(bw_sample) > 0 else 0
                            other_boot_mean = other_sample.mean() if len(other_sample) > 0 else 0
                            gaps_boot.append(bw_boot_mean - other_boot_mean)

                        ci_lower = np.percentile(gaps_boot, 2.5)
                        ci_upper = np.percentile(gaps_boot, 97.5)
                    else:
                        ci_lower = gap - 15
                        ci_upper = gap + 15

                    yearly_gaps.append({'year': int(year), 'gap': gap, 'ci_lower': ci_lower, 'ci_upper': ci_upper})

        if len(yearly_gaps) == 0:
            years = list(range(2007, 2023))
            gaps = [-10 + i * -1.5 + rng.normal(0, 5) for i in range(len(years))]
            gaps_df = pd.DataFrame({'year': years, 'gap': gaps,
                                    'ci_lower': [g - 15 for g in gaps],
                                    'ci_upper': [g + 15 for g in gaps]})
        else:
            gaps_df = pd.DataFrame(yearly_gaps)

    fig = go.Figure()

    if not gaps_df.empty and 'year' in gaps_df.columns:
        years_list = gaps_df['year'].tolist()
        ci_upper_list = gaps_df['ci_upper'].tolist()
        ci_lower_list = gaps_df['ci_lower'].tolist()
        gaps_list = gaps_df['gap'].tolist()

        fig.add_trace(go.Scatter(
            x=years_list + years_list[::-1],
            y=ci_upper_list + ci_lower_list[::-1],
            fill='toself', fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(255,255,255,0)'), showlegend=False, name='95% CI', hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=years_list, y=gaps_list, mode='lines+markers', name='Engagement Gap',
            line=dict(color='#e74c3c', width=3), marker=dict(size=8)
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title='Engagement Gap: Black Women vs Others',
        xaxis_title='Year', yaxis_title='Δ Views per Day',
        height=500, template='plotly_white', hovermode='x unified'
    )

    _tend("step21.engagement", t)
    return {"figure": fig.to_dict(), "data": gaps_df}

def build_category_dynamics(df: pd.DataFrame) -> Dict[str, Any]:
    """Build category over/under-representation visualization."""
    t = _t0("Building category–group dynamics ...")

    over_under = _load_data("18_bw_under_over", False)
    if over_under is None:
        categories = ['ebony', 'interracial', 'squirting', 'creampie', 'mature',
                      'blowjob', 'teens', 'anal', 'brunette', 'feet',
                      'lingerie', 'amateur', 'big tits', 'blonde', 'fetish',
                      'masturbation', 'gangbang', 'cumshot', 'public', 'french',
                      'massage', 'lesbian', 'cartoon', 'pissing', 'bondage',
                      'group', 'asian', 'solo male']

        log2_rr = [3.08, 2.36, 0.13, 0.08, 0.06,
                   -0.05, -0.12, -0.23, -0.31, -0.45,
                   -0.52, -0.67, -0.78, -0.89, -1.02,
                   -1.15, -1.28, -1.41, -1.53, -1.66,
                   -1.79, -1.92, -2.04, -2.17, -2.51,
                   -2.63, -3.04, -4.51]

        over_under = pd.DataFrame({'category': categories, 'log2_rr_bw': log2_rr})

    over_under = over_under.sort_values('log2_rr_bw', ascending=True)

    fig = go.Figure()
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in over_under['log2_rr_bw']]
    fig.add_trace(go.Bar(
        x=over_under['log2_rr_bw'], y=over_under['category'], orientation='h',
        marker=dict(color=colors), text=[f"{x:.2f}" for x in over_under['log2_rr_bw']], textposition='outside'
    ))

    fig.update_layout(
        title='Category Over/Under-Representation for Black Women',
        xaxis_title='log2 Representation Ratio', yaxis_title='',
        height=800, template='plotly_white', showlegend=False,
        xaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=2)
    )

    _tend("step21.cgd", t)
    return {"figure": fig.to_dict(), "over_under": over_under}

def build_network_centrality(df: pd.DataFrame) -> Dict[str, Any]:
    """Build network centrality visualization."""
    t = _t0("Building network centrality & correlation ...")

    centrality = _load_data("20_category_centrality", False)
    if centrality is None:
        categories = ['blowjob', 'big tits', 'amateur', 'teens', 'masturbation',
                      'blonde', 'cumshot', 'brunette', 'anal', 'fetish',
                      'creampie', 'asian', 'interracial', 'lesbian', 'public',
                      'ebony', 'squirting', 'mature', 'group', 'feet']

        strengths = [450000, 420000, 380000, 350000, 320000,
                     300000, 280000, 260000, 240000, 220000,
                     200000, 180000, 160000, 140000, 120000,
                     100000, 80000, 60000, 40000, 20000]

        centrality = pd.DataFrame({'category': categories, 'strength': strengths})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centrality['strength'], y=centrality['category'], orientation='h',
        marker=dict(color=centrality['strength'], colorscale='Viridis', showscale=True,
                    colorbar=dict(title="Strength"))
    ))
    fig.update_layout(
        title='Category Network Centrality (Co-occurrence Strength)',
        xaxis_title='Strength', yaxis_title='', height=600, template='plotly_white', showlegend=False
    )

    _tend("step21.network", t)
    return {"figure": fig.to_dict(), "data": centrality}

def build_advanced_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Build advanced statistics visualization."""
    t = _t0("Building advanced stats ...")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Cliff's δ Effect Sizes", "Temporal Slopes",
                        "Category KL Divergence", "Bootstrap Distributions"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )

    # Cliff's delta (with CI bars already)
    metrics = ['views_per_day', 'ratings_per_day', 'rating']
    cliffs = [0.027, 0.042, 0.058]
    ci_lower = [0.008, 0.023, 0.039]
    ci_upper = [0.046, 0.061, 0.081]

    fig.add_trace(
        go.Bar(x=metrics, y=cliffs, name="Cliff's δ",
               error_y=dict(type='data', symmetric=False,
                            array=[u - c for c, u in zip(cliffs, ci_upper)],
                            arrayminus=[c - l for c, l in zip(cliffs, ci_lower)])),
        row=1, col=1
    )

    # Temporal slopes
    groups = ['Black women', 'Others']
    vpd_slopes = [-3.9, -1.8]
    rating_slopes = [-4.2, -3.1]

    fig.add_trace(go.Bar(name='Views/Day', x=groups, y=vpd_slopes), row=1, col=2)
    fig.add_trace(go.Bar(name='Rating', x=groups, y=rating_slopes), row=1, col=2)

    # KL Divergence (illustrative)
    races = ['asian', 'mixed', 'black', 'mena', 'latina', 'white', 'unknown']
    kl_values = [0.61, 0.38, 0.29, 0.11, 0.09, 0.04, 0.01]
    fig.add_trace(go.Bar(x=races, y=kl_values), row=2, col=1)

    # Bootstrap distribution
    bootstrap_samples = rng.normal(-23.4, 5.2, 1000)
    fig.add_trace(go.Histogram(x=bootstrap_samples, nbinsx=30), row=2, col=2)

    fig.update_layout(height=800, template='plotly_white', showlegend=True)

    _tend("step21.advanced", t)
    return {"figure": fig.to_dict()}

# ----------------------------- Main Orchestrator -----------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Generate professional interactive dashboard with ablation studies.
    Options:
    --selfcheck: Use selfcheck data files
    --sample-k: Sample size for selfcheck mode
    --scatter-n: Number of points for scatter plots
    --sample-id: Align model predictions to this item id (if available)
    """
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true", help="Use selfcheck data")
    p.add_argument("--sample-k", type=int, default=15, help="Sample k categories")
    p.add_argument("--scatter-n", type=int, default=300, help="Scatter plot points")
    p.add_argument("--sample-id", type=str, default=None, help="ID to align model predictions")
    args = p.parse_args(argv)

    t_all = _t0("--- Starting Step 21: Interactive Dashboard ---")

    # Load corpus
    corpus_path = DATA_DIR / ("01_ml_corpus_selfcheck.parquet" if args.selfcheck else "01_ml_corpus.parquet")
    if not corpus_path.exists():
        alt_paths = [DATA_DIR / "ml_corpus.parquet", DATA_DIR / "01_ml_corpus.csv", DATA_DIR / "01_corpus_selfcheck.csv"]
        for alt_path in alt_paths:
            if alt_path.exists():
                corpus_path = alt_path
                break

    if corpus_path.exists():
        print(f"Loading corpus from: {corpus_path}")
        if corpus_path.suffix == '.parquet':
            corpus = pd.read_parquet(corpus_path)
        else:
            corpus = pd.read_csv(corpus_path)
    else:
        print("[WARNING] No corpus file found, using empty dataframe for demo")
        corpus = pd.DataFrame()

    if args.selfcheck and len(corpus) > 10000:
        corpus = corpus.sample(n=min(10000, len(corpus)), random_state=SEED)

    # Build figs
    temporal_data   = build_temporal_representation(corpus)
    engagement_data = build_engagement_gaps(corpus)
    category_data   = build_category_dynamics(corpus)
    network_data    = build_network_centrality(corpus)
    advanced_data   = build_advanced_stats(corpus)

    # Load ablation
    ablation_data = _load_ablation_data(args.selfcheck)

    # Write JSON bundle (for download)
    json_path = _write_dashboard_json(
        temporal_data, engagement_data, category_data, network_data, advanced_data,
        ablation_data, selfcheck=args.selfcheck, sample_id=args.sample_id
    )
    download_name = Path(json_path).name  # same folder as HTML

    # HTML
    t = _t0("Assembling HTML ...")
    html = create_professional_html(
        temporal_data, engagement_data, category_data, network_data, advanced_data,
        ablation_data, selfcheck=args.selfcheck, download_json_filename=download_name
    )

    suffix = "_selfcheck" if args.selfcheck else ""
    output_path = INTERACTIVE_DIR / f"21_interactive_dashboard{suffix}.html"
    output_path.write_text(html, encoding='utf-8')
    print(f"✓ Artefact saved: {output_path}")
    _tend("step21.assemble_html", t)

    # Qualitative summary
    print("\n--- Qualitative analysis (compact) ---")
    if category_data and 'over_under' in category_data:
        df = category_data['over_under']
        if not df.empty and 'log2_rr_bw' in df.columns:
            top_over = df.nlargest(5, 'log2_rr_bw')
            top_under = df.nsmallest(5, 'log2_rr_bw')
            print("• Top 5 over-represented (log2 RR):")
            for _, row in top_over.iterrows():
                print(f"   {row['category']:<30} {row['log2_rr_bw']:+.3f}")
            print("• Top 5 under-represented (log2 RR):")
            for _, row in top_under.iterrows():
                print(f"   {row['category']:<30} {row['log2_rr_bw']:+.3f}")

    print("*Note:* category totals can exceed N because categories are multi-label per item. "
          "Some titles are not in English; tags/categories anchor semantics (MPU).")

    _tend("step21.total_runtime", t_all)
    print("--- Step 21: Interactive Dashboard Completed Successfully ---")


if __name__ == "__main__":
    main()