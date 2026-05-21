#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 25 — Pareto Frontier (Accuracy vs Fairness)
================================================

What it is / what it does
-------------------------
Builds the Accuracy–Fairness Pareto frontier from prior evaluation artefacts:
- X-axis: Overall accuracy
- Y-axis: Fairness = 1 − max_abs_disparity  (clamped to [0, 1])
- Points: Each model (e.g., RF baseline, BERT baseline, reweighing/inproc/postproc variants)
- Highlight: Pareto-optimal solutions (upper-right frontier)

Inputs (robust to partial availability)
--------------------------------------
Scans `outputs/data/**/*.csv` for likely accuracy/fairness sources:
  • Accuracy columns (any): accuracy, acc, overall_accuracy, val_accuracy,
    macro_accuracy, micro_accuracy, balanced_accuracy
  • Fairness sources:
      - direct fairness columns: fairness  (already in [0,1] or 0–100)
      - or disparity-like columns: max_disparity, disparity, gap, delta, diff
        (we take max absolute value across them; fairness = 1 − max(abs(.)))

Also supports: mitigation_effectiveness_comparison.csv, fairness_disparities_*.csv,
*_val_metrics.csv — but does not require those exact names.

Outputs (canonical; self-check writes *_selfcheck.*) — number-first
-------------------------------------------------------------------
  CSVs:
    outputs/data/25_pareto_points[ _selfcheck].csv
    outputs/data/25_pareto_frontier[ _selfcheck].csv
  LaTeX:
    dissertation/auto_tables/25_pareto_points[ _selfcheck].tex
    dissertation/auto_tables/25_pareto_frontier[ _selfcheck].tex
  Figures:
    outputs/figures/dark/25_pareto_frontier_dark[ _selfcheck].png
    outputs/figures/light/25_pareto_frontier_light[ _selfcheck].png
  Narrative:
    outputs/narratives/automated/25_pareto_frontier[ _selfcheck].md

Notes & conventions
-------------------
• Uses reproducibility seed from the project config (NOT 42).
• All imports at the top of the file (never mid-file).
• When printing, round numbers sensibly (e.g., accuracies/fairness to 3 decimals).
• If no overlapping real entries exist:
    - FULL run: clean stale artefacts and exit (with [DELETE] notices).
    - SELF-CHECK: synthesize a tiny demo so the plot/narrative render for testing.
• Titles may be in other languages; tags/categories (MPU) keep semantics interpretable.
"""

from __future__ import annotations

# ----------------------------- Imports (top only) -----------------------------
import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------- Config & theming ------------------------------
try:
    from src.utils.theme_manager import ThemeManager, load_config
    THEME = ThemeManager()
    CONFIG = load_config() or {}
except Exception:
    THEME = None
    CONFIG = {}

ROOT = Path(CONFIG.get("project", {}).get("root", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(CONFIG.get("paths", {}).get("data", ROOT / "outputs" / "data"))
FIG_DARK = ROOT / "outputs" / "figures" / "dark"
FIG_LIGHT = ROOT / "outputs" / "figures" / "light"
NARR_DIR = ROOT / "outputs" / "narratives" / "automated"
AUTO_TEX = ROOT / "dissertation" / "auto_tables"

for d in (FIG_DARK, FIG_LIGHT, DATA_DIR, NARR_DIR, AUTO_TEX):
    d.mkdir(parents=True, exist_ok=True)

# reproducibility seed from config (NOT 42)
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
np.random.seed(SEED)

EPS = 1e-12

# ----------------------------- Lightweight timers ----------------------------
def _t0(msg: str) -> float:
    """Start a high-resolution timer and print a heading."""
    print(msg)
    return time.perf_counter()

def _tend(label: str, t_start: float) -> None:
    """Stop timer and print standardized [TIME] message."""
    print(f"[TIME] {label}: {time.perf_counter() - t_start:.2f}s")

# ----------------------------- IO helpers ------------------------------------
def _suffix(name: str, selfcheck: bool) -> str:
    """Append _selfcheck to a base filename (before extension) when requested."""
    if not selfcheck:
        return name
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        return f"{stem}_selfcheck.{ext}"
    return f"{name}_selfcheck"

def _out_csv(name: str, selfcheck: bool = False) -> Path:
    """Build a CSV output path in DATA_DIR with optional self-check suffix."""
    return DATA_DIR / _suffix(name, selfcheck)

def _out_tex(name: str, selfcheck: bool = False) -> Path:
    """Build a LaTeX output path in AUTO_TEX with optional self-check suffix."""
    return AUTO_TEX / _suffix(name, selfcheck)

def _out_md(name: str, selfcheck: bool = False) -> Path:
    """Build a Markdown output path in NARR_DIR with optional self-check suffix."""
    return NARR_DIR / _suffix(name, selfcheck)

def _plot_path(base: str, dark: bool, selfcheck: bool = False) -> Path:
    """Build a plot path (dark/light) with optional self-check suffix."""
    side = "dark" if dark else "light"
    name = _suffix(f"{base}_{side}.png", selfcheck)
    return (FIG_DARK if dark else FIG_LIGHT) / name

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV with a friendly confirmation line."""
    df.to_csv(path, index=False)
    print(f"[WRITE] {path}")

def _write_tex(df: pd.DataFrame, path: Path) -> None:
    """Write a minimal LaTeX table; avoids heavy dependencies."""
    try:
        tex = df.to_latex(index=False)
    except Exception:
        tex = df.to_string(index=False)
    path.write_text(tex, encoding="utf-8")
    print(f"[TEX]   {path}")

def _delete_stale(paths: List[Path]) -> None:
    """Delete stale artefacts; report deletions."""
    for p in paths:
        try:
            if p.exists():
                p.unlink()
                print(f"[DELETE] {p}")
        except Exception:
            pass

# ----------------------------- Legacy cleanup --------------------------------
def _cleanup_legacy_pareto25_outputs() -> None:
    """
    Remove legacy 'pareto25_*' artefacts now that we write '25_pareto_*'.
    Safe to run even if nothing exists.
    """
    to_del: List[Path] = []
    for base in ["pareto25_points", "pareto25_frontier"]:
        to_del += [
            _out_csv(f"{base}.csv", False),
            _out_csv(f"{base}.csv", True),
            _out_tex(f"{base}.tex", False),
            _out_tex(f"{base}.tex", True),
        ]
    for dark in (True, False):
        for sc in ("", "_selfcheck"):
            side = "dark" if dark else "light"
            to_del.append((FIG_DARK if dark else FIG_LIGHT) / f"pareto25_frontier_{side}{sc}.png")
    to_del += [_out_md("25_pareto_frontier.md", True)]  # selfcheck name unchanged; included for completeness
    if to_del:
        print("[INFO] Cleaning legacy pareto25_* artefacts (one-time).")
        _delete_stale(to_del)

# ----------------------------- Plot theming ----------------------------------
def _set_theme(dark: bool) -> None:
    """Apply dark/light plotting theme. Uses ThemeManager if available; else rcParams."""
    if THEME is not None:
        THEME.apply(dark=dark)
        return
    plt.rcParams.update({
        "figure.facecolor": "black" if dark else "white",
        "axes.facecolor": "black" if dark else "white",
        "axes.labelcolor": "white" if dark else "black",
        "xtick.color": "white" if dark else "black",
        "ytick.color": "white" if dark else "black",
        "text.color": "white" if dark else "black",
        "savefig.facecolor": "black" if dark else "white",
        "savefig.edgecolor": "black" if dark else "white",
        "grid.color": "gray",
        "grid.alpha": 0.25,
    })

# ----------------------------- Readers & utilities ---------------------------
def _iter_csvs() -> Iterable[Path]:
    """Iterate over CSV files under outputs/data recursively.

    Excludes:
    - Our own 25_pareto_* artefacts (avoid circular reads)
    - *_selfcheck.* files (small samples give wrong per-group accuracies)
    - *_group_metrics* files (per-group rows, not overall model accuracy)
    - *_predictions* files (per-video rows, not metrics)
    - *_ids.csv files (split ID lists, no metrics)
    """
    for p in sorted(DATA_DIR.rglob("*.csv")):  # sorted for deterministic order
        nm = p.name.lower()
        if nm.startswith("25_pareto_"):
            continue
        if "_selfcheck" in nm:
            continue
        if "_group_metrics" in nm:
            continue
        if "_predictions" in nm:
            continue
        if nm.endswith("_ids.csv") or nm in ("06_train_ids.csv", "06_val_ids.csv", "06_test_ids.csv"):
            continue
        yield p

def _read_csv(p: Path) -> Optional[pd.DataFrame]:
    """Safe CSV reader: returns None if file missing or broken."""
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None

def _norm_model_key(s: str) -> str:
    """Normalize a model string to a matching key (lower, alnum-only)."""
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def _humanize_label(stem: str) -> str:
    """Map filename hints to friendly labels (best-effort).

    Explicit step-number prefixes (09_ → BERT, 07_/08_rf_ → RF baseline) are
    needed because those files don't carry the model name in the filename itself.
    More-specific patterns must come before the generic step-prefix fallbacks.
    """
    m = stem.lower()
    # Named model patterns (most specific first)
    if "bert" in m:
        return "BERT baseline"
    if "reweigh" in m or "reweighed" in m:
        return "RF reweighed"
    if "postprocessing" in m or "postproc" in m:
        return "RF postproc"
    if "inprocessing" in m or "inproc" in m:
        return "RF inproc"
    if "rf_baseline" in m or m.endswith("rf"):
        return "RF baseline"
    # Step-number prefix fallbacks for files whose names carry no model hint
    if "adversarial" in m or "adv_fairness" in m or m.startswith("11b_"):
        return "RF adversarial"
    if m.startswith("09_") or m.startswith("09overall") or m.startswith("09fairness"):
        return "BERT baseline"
    if (m.startswith("07_") or m.startswith("07overall") or m.startswith("07fairness")
            or m.startswith("08_rf") or m.startswith("08rf")):
        return "RF baseline"
    return stem

def _fix_percent(x: float) -> Optional[float]:
    """Normalize metric to [0,1]: if 1<x<=100 treat as percent; else clamp to [0,1]."""
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    if 1.0 < v <= 100.0:
        v = v / 100.0
    return float(np.clip(v, 0.0, 1.0))

def _find_numeric(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """Return the first numeric column (Series) found among candidates; else None."""
    cols = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in cols:
            s = pd.to_numeric(df[cols[key]], errors="coerce")
            if s.notna().any():
                return s
    return None

def _extract_model_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Try to get a 'model' column (case-insensitive)."""
    for name in ("model", "models", "classifier", "estimator", "name"):
        if name in {c.lower() for c in df.columns}:
            real = [c for c in df.columns if c.lower() == name][0]
            return df[real].astype(str)
    return None

# ----------------------------- Accuracy & Fairness loaders -------------------
def _load_accuracies() -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Collect overall accuracy per model by scanning all CSVs for common accuracy
    fields. Returns both normalized key->accuracy and key->display label.
    """
    acc: Dict[str, float] = {}
    label: Dict[str, str] = {}
    ACC_CANDS = [
        "accuracy", "acc", "overall_accuracy", "val_accuracy",
        "macro_accuracy", "micro_accuracy", "balanced_accuracy"
    ]

    for p in _iter_csvs():
        df = _read_csv(p)
        if df is None or df.empty:
            continue
        s = _find_numeric(df, ACC_CANDS)
        if s is None:
            continue

        # per-row model column first
        mcol = _extract_model_series(df)
        if mcol is not None:
            for i, v in s.dropna().items():
                m = str(mcol.iloc[i]) if i in mcol.index else None
                if not m:
                    continue
                k = _norm_model_key(m)
                val = _fix_percent(v)
                if val is None:
                    continue
                acc.setdefault(k, val)   # keep first seen
                label.setdefault(k, str(m))
            continue

        # fallback: single-row file; derive label from filename
        try:
            val = _fix_percent(s.dropna().iloc[0])
        except Exception:
            val = None
        if val is None:
            continue
        human = _humanize_label(p.stem)
        k = _norm_model_key(human)
        acc.setdefault(k, val)
        label.setdefault(k, human)

    return acc, label

def _max_abs_disparity(df: pd.DataFrame) -> Optional[float]:
    """
    Robustly extract the max absolute disparity from a disparities table.
    Accepts columns containing: disparity/gap/delta/diff/max_disparity (any).
    If a direct 'fairness' column is present, convert to disparity as (1 - fairness).
    """
    if df is None or df.empty:
        return None

    # direct fairness column → convert to disparity
    for c in df.columns:
        if c.lower().strip() == "fairness":
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if s.empty:
                return None
            f = _fix_percent(s.iloc[0])
            return None if f is None else float(np.clip(1.0 - f, 0.0, 1.0))

    cands = [c for c in df.columns
             if any(tok in c.lower() for tok in ("dispar", "gap", "delta", "diff"))]
    if not cands:
        return None

    best = None
    for c in cands:
        vals = pd.to_numeric(df[c], errors="coerce").dropna().abs()
        if vals.empty:
            continue
        m = float(vals.max())
        best = m if best is None else max(best, m)
    return best

def _load_fairness() -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Collect fairness per model as 1 − max_abs_disparity (clamped to [0,1]) by
    scanning all CSVs. Returns normalized key->fairness and key->display label.
    """
    out: Dict[str, float] = {}
    label: Dict[str, str] = {}

    for p in _iter_csvs():
        df = _read_csv(p)
        if df is None or df.empty:
            continue

        # per-row model column
        mcol = _extract_model_series(df)
        if mcol is not None:
            # direct fairness column?
            fser = _find_numeric(df, ["fairness"])
            if fser is not None:
                for i, v in fser.dropna().items():
                    m = str(mcol.iloc[i]) if i in mcol.index else None
                    if not m:
                        continue
                    k = _norm_model_key(m)
                    f = _fix_percent(v)
                    if f is None:
                        continue
                    out.setdefault(k, f)
                    label.setdefault(k, str(m))
                continue

            # derive from disparities
            cand_dispar = [c for c in df.columns if any(tok in c.lower() for tok in ("dispar", "gap", "delta", "diff"))]
            if cand_dispar:
                arr = pd.concat([pd.to_numeric(df[c], errors="coerce").abs() for c in cand_dispar], axis=1)
                row_max = arr.max(axis=1)
                for i, d in row_max.dropna().items():
                    m = str(mcol.iloc[i]) if i in mcol.index else None
                    if not m:
                        continue
                    k = _norm_model_key(m)
                    d_unit = _fix_percent(d)
                    if d_unit is None:
                        continue
                    fairness = float(np.clip(1.0 - d_unit, 0.0, 1.0))
                    out.setdefault(k, fairness)
                    label.setdefault(k, str(m))
                continue

        # file-per-model fallback
        mx = _max_abs_disparity(df)
        if mx is None:
            continue
        d_unit = _fix_percent(mx)
        if d_unit is None:
            continue
        fairness = float(np.clip(1.0 - d_unit, 0.0, 1.0))
        human = _humanize_label(p.stem.replace("fairness_disparities_", ""))
        k = _norm_model_key(human)
        out.setdefault(k, fairness)
        label.setdefault(k, human)

    return out, label

# ----------------------------- Pareto logic ----------------------------------
def _pareto_frontier(points: pd.DataFrame) -> pd.DataFrame:
    """
    Compute non-dominated points when maximizing (accuracy, fairness).
    """
    if points.empty:
        return points
    pts = points.sort_values(["accuracy", "fairness"], ascending=[False, False]).reset_index(drop=True)
    frontier_idx = []
    best_y = -np.inf
    for i, row in pts.iterrows():
        if row["fairness"] >= best_y - 1e-12:
            frontier_idx.append(i)
            best_y = row["fairness"]
    return pts.loc[frontier_idx].copy()

# ----------------------------- Orchestration steps ---------------------------
def build_points(selfcheck: bool, sample_models: Optional[int]) -> Optional[pd.DataFrame]:
    """
    Merge accuracy and fairness into a single dataframe. Optionally sub-sample
    models in self-check mode for faster rendering.

    FULL run (no points): delete stale artefacts and return None.
    SELF-CHECK (no points): synthesize a tiny demo so the plot renders.
    """
    t = _t0("A. Collecting accuracy & fairness ...")
    acc, acc_label = _load_accuracies()
    fai, fai_label = _load_fairness()

    keys = sorted(set(acc.keys()) & set(fai.keys()))
    rows = [{"model": acc_label.get(k, fai_label.get(k, k)),
             "accuracy": float(acc[k]),
             "fairness": float(fai[k])} for k in keys]
    df = pd.DataFrame(rows)

    if df.empty:
        if not selfcheck:
            print("[INFO] No overlapping accuracy & fairness entries found; cleaning stale Pareto artefacts and skipping.")
            _delete_stale([
                _out_csv("25_pareto_points.csv", False), _out_csv("25_pareto_points.csv", True),
                _out_csv("25_pareto_frontier.csv", False), _out_csv("25_pareto_frontier.csv", True),
                _out_tex("25_pareto_points.tex", False), _out_tex("25_pareto_points.tex", True),
                _out_tex("25_pareto_frontier.tex", False), _out_tex("25_pareto_frontier.tex", True),
                _plot_path("25_pareto_frontier", True, False), _plot_path("25_pareto_frontier", False, False),
                _plot_path("25_pareto_frontier", True, True),  _plot_path("25_pareto_frontier", False, True),
                _out_md("25_pareto_frontier.md", False), _out_md("25_pareto_frontier.md", True)
            ])
            _tend("step25.collect", t)
            return None
        # SELF-CHECK fallback: synthesize a small demo
        rng = np.random.default_rng(SEED)
        models = ["RF baseline", "BERT baseline", "RF reweighed"]
        accs = np.clip(rng.normal(0.78, 0.04, size=3), 0.6, 0.92)
        fairs = np.clip(rng.normal(0.70, 0.07, size=3), 0.45, 0.95)
        df = pd.DataFrame({"model": models, "accuracy": accs, "fairness": fairs})
        print("[INFO] SELF-CHECK: synthesized demo points (no real overlap found).")

    if selfcheck and sample_models is not None and sample_models > 0 and len(df) > sample_models:
        df = df.sample(n=sample_models, random_state=SEED).sort_values(["accuracy", "fairness"], ascending=[False, False])

    _write_csv(df, _out_csv("25_pareto_points.csv", selfcheck))
    _write_tex(df, _out_tex("25_pareto_points.tex", selfcheck))
    _tend("step25.collect", t)
    return df

def plot_frontier(df: pd.DataFrame, selfcheck: bool) -> pd.DataFrame:
    """
    Compute Pareto frontier and generate dark/light plots.
    """
    t = _t0("B. Computing & plotting frontier ...")
    fr = _pareto_frontier(df)
    _write_csv(fr, _out_csv("25_pareto_frontier.csv", selfcheck))
    _write_tex(fr, _out_tex("25_pareto_frontier.tex", selfcheck))

    for dark in (True, False):
        _set_theme(dark)
        fig, ax = plt.subplots(figsize=(7.5, 5.2))

        # All points
        pts = ax.scatter(
            df["accuracy"], df["fairness"],
            s=70, alpha=0.95, zorder=2, label="_nolegend_"
        )

        # Annotate points
        for _, r in df.iterrows():
            ax.annotate(
                str(r["model"]),
                (r["accuracy"], r["fairness"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9
            )

        # Frontier line + star overlays for frontier points
        f = fr.sort_values("accuracy")
        if not f.empty:
            line, = ax.plot(
                f["accuracy"], f["fairness"],
                "-o", lw=2.4, alpha=1.0, zorder=3, label="_nolegend_"
            )
            stars = ax.scatter(
                f["accuracy"], f["fairness"],
                marker="*", s=220, linewidths=1.2, edgecolors="white",
                zorder=4, label="_nolegend_"
            )

        ax.set_xlabel("Overall accuracy")
        ax.set_ylabel("Fairness = 1 − max disparity")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.set_title("Accuracy–Fairness Pareto Frontier")

        # Legend (proxy handles; avoids plotting stray markers at (0,0))
        legend_handles = [
            Line2D([], [], marker="o", linestyle="None", markersize=7, label="Models"),
            Line2D([], [], marker="*", linestyle="-", markersize=10, label="Pareto frontier"),
        ]
        ax.legend(handles=legend_handles, frameon=False, loc="lower right")

        fname = _plot_path("25_pareto_frontier", dark, selfcheck)
        fig.tight_layout()
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        print(f"[PLOT] {fname}")


    _tend("step25.frontier", t)
    return fr

def write_narrative(points: pd.DataFrame, frontier: pd.DataFrame, selfcheck: bool) -> None:
    """
    Write a short narrative (markdown) with key counts and frontier model names.
    """
    t = _t0("C. Narrative write-up ...")
    n_pts = int(len(points))
    n_front = int(len(frontier))
    front_names = ", ".join(frontier["model"].tolist()) if n_front else "—"
    md = [
        "# Step 25 — Pareto Frontier",
        "",
        f"- Total models considered: **{n_pts}**",
        f"- Pareto-optimal models: **{n_front}**",
        f"- Frontier set: {front_names}",
        "",
        "Interpretation:",
        "- Points further to the upper-right jointly improve accuracy and fairness.",
        "- Frontier models are **non-dominated** (no other model strictly better on both axes).",
        "- Large gaps between neighbors on the frontier suggest headroom (algorithm/mitigation).",
        "",
        "Notes:",
        "- Totals in other steps can exceed *N* due to multi-label assignment; here we work per model.",
        "- Titles can be non-English; tags/categories (MPU) keep semantics interpretable.",
        "",
        f"*Seed={SEED}. Figures in `outputs/figures/(dark|light)`; tables in `dissertation/auto_tables`. "
        f"Self-check mode writes `_selfcheck` artefacts only.*"
    ]
    outp = _out_md("25_pareto_frontier.md", selfcheck)
    outp.write_text("\n".join(md), encoding="utf-8")
    print(f"[WRITE] {outp}")
    _tend("step25.narrative", t)

def qualitative_readout(points: Optional[pd.DataFrame], frontier: Optional[pd.DataFrame]) -> None:
    """
    Print a compact qualitative analysis, highlighting outliers for quick inspection.
    """
    print("\n--- Quick qualitative readout ---")
    if points is None or points.empty:
        print("• No points available.")
        return

    acc_m = float(points["accuracy"].mean())
    fai_m = float(points["fairness"].mean())
    print(f"• Means: accuracy={acc_m:.3f}, fairness={fai_m:.3f}")

    worst_f = points.sort_values("fairness").head(3)
    worst_a = points.sort_values("accuracy").head(3)

    def _fmt_row(r):
        return f"{r['model']} (acc={float(r['accuracy']):.3f}, fair={float(r['fairness']):.3f})"

    if not worst_f.empty:
        print("• Lowest fairness:", "; ".join(_fmt_row(r) for _, r in worst_f.iterrows()))
    if not worst_a.empty:
        print("• Lowest accuracy:", "; ".join(_fmt_row(r) for _, r in worst_a.iterrows()))
    if frontier is not None and not frontier.empty:
        print(f"• Frontier size: {len(frontier)} — " + ", ".join(frontier['model'].tolist()))
    print("*Note:* Values are rounded for readability. Titles may be non-English; tags/categories (MPU) keep semantics interpretable.")

# ----------------------------- CLI ------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Orchestrate Step 25 with timers, self-check, robust artefacts, and narrative.

    CLI
    ---
    Full run (canonical artefacts):
        python -m src.dissertation.25_pareto_frontier

    Self-check (no overwrites; random subset of models if desired):
        python -m src.dissertation.25_pareto_frontier --selfcheck --sample-models 12
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true",
                    help="Write *_selfcheck artefacts; do not overwrite canonical files.")
    ap.add_argument("--sample-models", type=int, default=None,
                    help="Optional model cap for self-check (random sample, uses config seed).")
    args = ap.parse_args(argv)

    t_all = _t0("--- Starting Step 25: Pareto Frontier ---")

    # One-time cleanup of legacy pareto25_* files (safe if not present)
    _cleanup_legacy_pareto25_outputs()

    points = build_points(selfcheck=args.selfcheck, sample_models=args.sample_models)
    if points is None:
        _tend("step25.total_runtime", t_all)
        print("--- Step 25: Pareto Frontier Completed Successfully ---")
        return

    frontier = plot_frontier(points, selfcheck=args.selfcheck)
    write_narrative(points, frontier, selfcheck=args.selfcheck)
    qualitative_readout(points, frontier)

    _tend("step25.total_runtime", t_all)
    print("--- Step 25: Pareto Frontier Completed Successfully ---")

if __name__ == "__main__":
    main()
