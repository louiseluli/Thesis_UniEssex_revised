#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 24 — Results Synthesis
===========================

What it is / what it does
-------------------------
Collates and summarizes core outputs from Step 22 (Ablations) and Step 23 (Limitations)
into a compact set of CSVs, LaTeX tables, figures, and a short narrative for the
dissertation chapter.

Inputs (best-effort; all optional but used when present)
-------------------------------------------------------
Ablations (Step 22):
  outputs/ablation/22_summary.csv
  outputs/ablation/22_rr_lexicon_off.csv
  outputs/ablation/22_topk_mass_*.csv
  outputs/ablation/22_topcats_noise_*.csv
  outputs/ablation/22_bootstrap_*.csv

Limitations (Step 23) — number-first naming:
  outputs/data/23_missingness.csv
  outputs/data/23_representation_skew.csv
  outputs/data/23_temporal_drift.csv
  outputs/data/23_category_longtail.csv
  outputs/data/23_hurtlex_dependency.csv   (optional)

Outputs (canonical; self-check writes *_selfcheck.*)
----------------------------------------------------
  CSVs:
    outputs/data/24_ablation_kpis[ _selfcheck].csv
    outputs/data/24_limitations_kpis[ _selfcheck].csv
    outputs/data/24_checklist[ _selfcheck].csv
  LaTeX:
    dissertation/auto_tables/24_ablation_kpis[ _selfcheck].tex
    dissertation/auto_tables/24_limitations_kpis[ _selfcheck].tex
    dissertation/auto_tables/24_checklist[ _selfcheck].tex
  Figures (dark|light):
    outputs/figures/dark|light/24_ablation_dashboard_[dark|light][ _selfcheck].png
    outputs/figures/dark|light/24_limitations_dashboard_[dark|light][ _selfcheck].png
  Narrative:
    outputs/narratives/automated/24_results_synthesis[ _selfcheck].md

Conventions & notes
-------------------
• Uses reproducibility seed from the project config (NOT 42).
• Years are kept as integers; ratings are reported with one decimal place; other numbers
  are rounded sensibly when printed.
• Category totals can exceed N due to multi-label assignment.
• Titles may be in other languages; tags/categories preserve interpretable semantics (MPU).
"""

from __future__ import annotations

# ----------------------------- Imports (top only) -----------------------------
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- Config & theming ------------------------------
try:
    from src.utils.theme_manager import ThemeManager, load_config
    THEME = ThemeManager()
    CONFIG = load_config() or {}
except Exception:
    THEME = None
    CONFIG = {}

# project roots & paths (from config when present)
ROOT = Path(CONFIG.get("project", {}).get("root", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(CONFIG.get("paths", {}).get("data", ROOT / "outputs" / "data"))
ABL_DIR  = ROOT / "outputs" / "ablation"
FIG_DARK = ROOT / "outputs" / "figures" / "dark"
FIG_LIGHT= ROOT / "outputs" / "figures" / "light"
NARR_DIR = ROOT / "outputs" / "narratives" / "automated"
AUTO_TEX = ROOT / "dissertation" / "auto_tables"

for d in (DATA_DIR, ABL_DIR, FIG_DARK, FIG_LIGHT, NARR_DIR, AUTO_TEX):
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

def _set_theme(dark: bool) -> None:
    """Apply dark/light plotting theme. Uses ThemeManager if available; else rcParams."""
    if THEME is not None:
        THEME.apply(dark=dark)
        return
    plt.rcParams.update({
        "figure.facecolor": "black" if dark else "white",
        "axes.facecolor": "black" if dark else "white",
        "axes.edgecolor": "white" if dark else "black",
        "axes.labelcolor": "white" if dark else "black",
        "xtick.color": "white" if dark else "black",
        "ytick.color": "white" if dark else "black",
        "text.color": "white" if dark else "black",
        "savefig.facecolor": "black" if dark else "white",
        "savefig.edgecolor": "black" if dark else "white",
        "grid.color": "gray",
        "grid.alpha": 0.25,
    })

def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read CSV if present; else return None."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None

# ----------------------------- Legacy cleanup --------------------------------
def _cleanup_legacy_rs24_outputs() -> None:
    """
    Remove legacy 'rs24_*' artefacts so we don't keep duplicates after renaming to '24_*'.
    Runs quickly; safe if files don't exist.
    """
    to_del: List[Path] = []
    # CSV/TEX
    for base in ["ablation_kpis", "limitations_kpis", "checklist"]:
        to_del += [
            _out_csv(f"rs24_{base}.csv", False),
            _out_csv(f"rs24_{base}.csv", True),
            _out_tex(f"rs24_{base}.tex", False),
            _out_tex(f"rs24_{base}.tex", True),
        ]
    # FIGS
    for dash in ["rs24_ablation_dashboard", "rs24_limitations_dashboard"]:
        for dark in (True, False):
            for sc in ("", "_selfcheck"):
                side = "dark" if dark else "light"
                to_del.append((FIG_DARK if dark else FIG_LIGHT) / f"{dash}_{side}{sc}.png")
    # Delete
    if to_del:
        print("[INFO] Cleaning legacy rs24_* artefacts (one-time).")
        _delete_stale(to_del)

# ----------------------------- Step 22 KPIs ----------------------------------
def gather_ablation_kpis(selfcheck: bool) -> Optional[pd.DataFrame]:
    """
    Load and summarize Step-22 ablation outputs (best-effort).
    Produces a compact KPI table + a simple dashboard figure.

    Logic:
      • Prefer a 'delta' column (scenario − baseline). If absent, fall back to first row mean of numeric cols.
      • Read canonical first; if missing, try *_selfcheck variants.
      • Optionally ingest 22_bootstrap_*.csv and record median CI width as a stability proxy.
    """
    t = _t0("A. Ablations (Step 22) — gathering KPIs ...")

    def _read_either(base: str) -> Optional[pd.DataFrame]:
        """Read canonical file first; if not present, try *_selfcheck variant."""
        df = _safe_read_csv(ABL_DIR / base)
        if df is None:
            df = _safe_read_csv(ABL_DIR / _suffix(base, True))
        return df

    files: Dict[str, str] = {
        "Lexicon OFF": "22_rr_lexicon_off.csv",
        "Noise 10%":   "22_topcats_noise_10.csv",
        "Noise 25%":   "22_topcats_noise_25.csv",
        "Noise 50%":   "22_topcats_noise_50.csv",
        "TopK 10":     "22_topk_mass_10.csv",
        "TopK 20":     "22_topk_mass_20.csv",
        "TopK 30":     "22_topk_mass_30.csv",
        "TopK 50":     "22_topk_mass_50.csv",
    }

    rows = []
    for label, fname in files.items():
        df = _read_either(fname)
        if df is None or df.empty:
            continue
        val: Optional[float] = None
        if "delta" in df.columns:
            try:
                val = float(pd.to_numeric(df["delta"], errors="coerce").iloc[0])
            except Exception:
                val = None
        if val is None:
            nums = df.select_dtypes(include=[np.number])
            if not nums.empty:
                try:
                    val = float(nums.iloc[0].mean())
                except Exception:
                    val = None
        if val is not None and np.isfinite(val):
            rows.append({"scenario": label, "delta": float(val)})

    # Optional bootstrap ingestion (if present)
    for n in (200, 500, 1000):
        dfb = _read_either(f"22_bootstrap_{n}.csv")
        if dfb is None or dfb.empty:
            continue
        width = None
        if all(c in dfb.columns for c in ["metric", "ci_low", "ci_high"]):
            w = (pd.to_numeric(dfb["ci_high"], errors="coerce") -
                 pd.to_numeric(dfb["ci_low"], errors="coerce")).abs()
            width = float(w.median()) if w.notna().any() else None
        else:
            num = dfb.select_dtypes(include="number")
            if not num.empty:
                width = float((num.max(numeric_only=True) - num.min(numeric_only=True)).mean())
        if width is not None and np.isfinite(width):
            rows.append({"scenario": f"bootstrap_median_ci_width_N={n}", "delta": width})

    if not rows:
        print("[INFO] No usable Ablation artefacts found; deleting stale 24 Ablation outputs and skipping.")
        for p in [
            _out_csv("24_ablation_kpis.csv", False), _out_csv("24_ablation_kpis.csv", True),
            _out_tex("24_ablation_kpis.tex", False), _out_tex("24_ablation_kpis.tex", True),
            _plot_path("24_ablation_dashboard", True, False), _plot_path("24_ablation_dashboard", False, False),
            _plot_path("24_ablation_dashboard", True, True),  _plot_path("24_ablation_dashboard", False, True),
        ]:
            try:
                if p.exists():
                    p.unlink()
                    print(f"[DELETE] {p}")
            except Exception:
                pass
        _tend("step24.ablation_kpis", t)
        return None

    out = pd.DataFrame(rows).sort_values("scenario")
    _write_csv(out, _out_csv("24_ablation_kpis.csv", selfcheck))
    _write_tex(out, _out_tex("24_ablation_kpis.tex", selfcheck))

    # Dashboard: bars of Δ by scenario + a hint line at 0
    for dark in (True, False):
        _set_theme(dark)
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(out))
        y = out["delta"].astype(float).to_numpy()
        ax.bar(x, y)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_ylabel("Δ vs baseline (scenario)")
        ax.set_title("Ablations — Scenario Sensitivity (Δ)")
        ax.set_xticks(x)
        ax.set_xticklabels(out["scenario"], rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fn = _plot_path("24_ablation_dashboard", dark, selfcheck)
        fig.savefig(fn, dpi=200); plt.close(fig)
        print(f"[PLOT] {fn}")

    _tend("step24.ablation_kpis", t)
    return out

# ----------------------------- Step 23 KPIs ----------------------------------
def gather_limitations_kpis(selfcheck: bool) -> Optional[pd.DataFrame]:
    """
    Load Step-23 limitation outputs and condense into a single KPI table.
    Tries canonical files first, then *_selfcheck when running in self-check mode.
    Uses the number-first Step-23 filenames.
    """
    t = _t0("B. Limitations (Step 23) — gathering KPIs ...")

    def _read(name: str) -> Optional[pd.DataFrame]:
        # canonical first
        df = _safe_read_csv(DATA_DIR / name)
        if df is None and selfcheck:
            df = _safe_read_csv(DATA_DIR / _suffix(name, True))
        return df

    miss = _read("23_missingness.csv")
    rep  = _read("23_representation_skew.csv")
    drift= _read("23_temporal_drift.csv")
    longt= _read("23_category_longtail.csv")
    lex  = _read("23_hurtlex_dependency.csv")  # optional

    rows = []
    if miss is not None and not miss.empty:
        worst = miss.sort_values("missing_fraction", ascending=False).iloc[0]
        rows.append({"kpi": "missingness_worst_field", "value": worst["field"], "detail": round(float(worst["missing_fraction"])*100, 1)})

    if rep is not None and not rep.empty:
        for dim in ["race_ethnicity", "gender"]:
            row = rep[rep["dimension"] == dim]
            if not row.empty:
                r = row.iloc[0]
                rows.extend([
                    {"kpi": f"{dim}_entropy_norm", "value": float(r["entropy_norm"])},
                    {"kpi": f"{dim}_gini", "value": float(r["gini"])},
                    {"kpi": f"{dim}_min_share_pct", "value": round(float(r["min_share"])*100, 1)},
                    {"kpi": f"{dim}_max_share_pct", "value": round(float(r["max_share"])*100, 1)},
                ])

    if drift is not None and not drift.empty:
        rows.extend([
            {"kpi": "bw_share_slope_per_year", "value": float(drift["bw_share_slope_per_year"].iloc[0])},
            {"kpi": "simpson_flip_detected", "value": bool(drift["simpson_flip_detected"].iloc[0])},
        ])

    if longt is not None and not longt.empty:
        def _get_k(k):
            if (longt["K"]==k).any():
                return float(longt.loc[longt["K"]==k, "mass_captured"].iloc[0])
            return np.nan
        rows.extend([
            {"kpi": "longtail_top10_mass", "value": _get_k(10)},
            {"kpi": "longtail_top50_mass", "value": _get_k(50)},
        ])

    if lex is not None and not lex.empty:
        rows.extend([
            {"kpi": "lexicon_median_mean_prevalence_pct", "value": float(lex["mean_prevalence_pct_median"].iloc[0])},
            {"kpi": "lexicon_median_max_prevalence_pct", "value": float(lex["max_prevalence_pct_median"].iloc[0])},
        ])

    if not rows:
        print("[INFO] No usable Limitations artefacts found; deleting stale 24 Limitations outputs and skipping.")
        for p in [
            _out_csv("24_limitations_kpis.csv", False), _out_csv("24_limitations_kpis.csv", True),
            _out_tex("24_limitations_kpis.tex", False), _out_tex("24_limitations_kpis.tex", True),
            _plot_path("24_limitations_dashboard", True, False), _plot_path("24_limitations_dashboard", False, False),
            _plot_path("24_limitations_dashboard", True, True),  _plot_path("24_limitations_dashboard", False, True),
        ]:
            try:
                if p.exists():
                    p.unlink()
                    print(f"[DELETE] {p}")
            except Exception:
                pass
        _tend("step24.limitations_kpis", t)
        return None

    out = pd.DataFrame(rows)
    _write_csv(out, _out_csv("24_limitations_kpis.csv", selfcheck))
    _write_tex(out, _out_tex("24_limitations_kpis.tex", selfcheck))

    # Simple dashboard: bars for entropy/gini/min/max share and long-tail mass (if present)
    for dark in (True, False):
        _set_theme(dark)
        fig, ax = plt.subplots(figsize=(10, 4))
        sel = out[out["kpi"].isin([
            "race_ethnicity_entropy_norm","race_ethnicity_gini","gender_entropy_norm","gender_gini",
            "longtail_top10_mass","longtail_top50_mass"
        ])].copy()
        if sel.empty:
            plt.close(fig)
            continue
        x = np.arange(len(sel))
        v = sel["value"].astype(float).to_numpy()
        ax.bar(x, v)
        ax.set_title("Limitations Dashboard (selected KPIs)")
        ax.set_ylabel("Value")
        ax.set_xticks(x)
        ax.set_xticklabels(sel["kpi"], rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fn = _plot_path("24_limitations_dashboard", dark, selfcheck)
        fig.savefig(fn, dpi=200); plt.close(fig)
        print(f"[PLOT] {fn}")

    _tend("step24.limitations_kpis", t)
    return out

# ----------------------------- Checklist synthesis ---------------------------
def build_rs24_checklist(abl: Optional[pd.DataFrame],
                         lim: Optional[pd.DataFrame],
                         selfcheck: bool) -> pd.DataFrame:
    """
    Combine a few signals into a high-level checklist for the Results chapter.

    Flags:
      • Representation entropy (race/gender) < 0.80
      • Long-tail severity: Top-10 mass < 0.50
      • Lexicon dependency: median max-prevalence ≥ 10%
      • Bootstrap CI width improvement from N=200 to N=1000
    """
    t = _t0("C. RS24 Checklist — synthesizing ...")
    rows = []

    def _val(df: Optional[pd.DataFrame], key: str) -> Optional[float]:
        """
        Convenience getter for a numeric by key.
        Supports either schema:
          • kpi/value  (limitations table)
          • scenario/delta (ablations table)
        """
        if df is None or df.empty:
            return None
        if "kpi" in df.columns and "value" in df.columns:
            s = df.loc[df["kpi"] == key, "value"]
        elif "scenario" in df.columns and "delta" in df.columns:
            s = df.loc[df["scenario"] == key, "delta"]
        else:
            return None
        if s.empty:
            return None
        try:
            return float(s.iloc[0])
        except Exception:
            return None

    # Representation balance flags
    ent_r = _val(lim, "race_ethnicity_entropy_norm")
    ent_g = _val(lim, "gender_entropy_norm")
    rows.append({"issue": "Representation entropy low (race)", "flag": bool(ent_r is not None and ent_r < 0.80), "metric": ent_r})
    rows.append({"issue": "Representation entropy low (gender)", "flag": bool(ent_g is not None and ent_g < 0.80), "metric": ent_g})

    # Long-tail coverage
    m10 = _val(lim, "longtail_top10_mass")
    rows.append({"issue": "Long-tail severe (Top-10 < 50%)", "flag": bool(m10 is not None and m10 < 0.50), "metric": m10})

    # Lexicon dependency
    lexmax = _val(lim, "lexicon_median_max_prevalence_pct")
    rows.append({"issue": "High lexicon dependency (median max ≥ 10%)", "flag": bool(lexmax is not None and lexmax >= 10.0), "metric": lexmax})

    # Ablation stability — bootstrap width (if present in ablation KPIs)
    boot200  = _val(abl, "bootstrap_median_ci_width_N=200")
    boot1000 = _val(abl, "bootstrap_median_ci_width_N=1000")
    if boot200 is not None and boot1000 is not None:
        rows.append({"issue": "Bootstrap CI width improved (200→1000)", "flag": bool(boot1000 < boot200), "metric": round(boot200 - boot1000, 3)})
    else:
        rows.append({"issue": "Bootstrap CI width improved (200→1000)", "flag": False, "metric": np.nan})

    out = pd.DataFrame(rows)
    _write_csv(out, _out_csv("24_checklist.csv", selfcheck))
    _write_tex(out, _out_tex("24_checklist.tex", selfcheck))
    _tend("step24.checklist", t)
    return out

# ----------------------------- Narrative -------------------------------------
def write_narrative(abl: Optional[pd.DataFrame],
                    lim: Optional[pd.DataFrame],
                    chk: pd.DataFrame,
                    selfcheck: bool) -> None:
    """
    Write a short synthesis narrative with checklist and interpretation notes.
    """
    t = _t0("D. Narrative write-up ...")
    md = [
        "# Step 24 — Results Synthesis",
        "",
        "This synthesis collates ablation sensitivity (Step 22) and limitations (Step 23).",
        "",
        "## Checklist",
        chk.to_markdown(index=False),
        "",
        "## Highlights & Interpretation",
        "- **Representation**: entropy/gini summarize balance; outlier max shares suggest dominant groups.",
        "- **Long-tail**: Top-K mass shows category head coverage; diminishing returns beyond ~30 are typical.",
        "- **Lexicon dependency**: high median max-prevalence suggests results sensitive to HurtLex; corroborate with context models.",
        "- **Uncertainty**: bootstrap CI widths shrink with larger N_BOOT; stability improves accordingly.",
        "",
        "_Totals can exceed N due to multi-label assignment. Titles may be in other languages, "
        "but tags/categories (MPU) preserve interpretable semantics. Years printed as integers; "
        "ratings shown with one decimal._",
        f"\n*Seed={SEED}. Figures saved under `outputs/figures/(dark|light)`; tables in `dissertation/auto_tables`. "
        f"Self-check mode writes `_selfcheck` artefacts only.*",
    ]
    outp = _out_md("24_results_synthesis.md", selfcheck)
    outp.write_text("\n".join(md), encoding="utf-8")
    print(f"[WRITE] {outp}")
    _tend("step24.narrative", t)

# ----------------------------- Readout ---------------------------------------
def qualitative_readout(abl: Optional[pd.DataFrame], lim: Optional[pd.DataFrame]) -> None:
    """
    Print compact qualitative analysis highlighting outliers, with rounded values.
    """
    print("\n--- Quick qualitative readout ---")
    if lim is not None and not lim.empty:
        # worst missingness
        s = lim.loc[lim["kpi"] == "missingness_worst_field", :]
        if not s.empty:
            f = str(s["value"].iloc[0])
            v = float(s["detail"].iloc[0]) if "detail" in s.columns else np.nan
            print(f"• Most missing: {f:<16}  missing={v:.1f}%")

        # representation
        for dim in ["race_ethnicity", "gender"]:
            ent = lim.loc[lim["kpi"] == f"{dim}_entropy_norm", "value"]
            gni = lim.loc[lim["kpi"] == f"{dim}_gini", "value"]
            mn  = lim.loc[lim["kpi"] == f"{dim}_min_share_pct", "value"]
            mx  = lim.loc[lim["kpi"] == f"{dim}_max_share_pct", "value"]
            if not ent.empty and not gni.empty and not mn.empty and not mx.empty:
                print(f"• {dim}: entropy_norm={float(ent.iloc[0]):.2f}, gini={float(gni.iloc[0]):.2f}, "
                      f"min={float(mn.iloc[0]):.0f}%, max={float(mx.iloc[0]):.0f}%")

        # long-tail
        t10 = lim.loc[lim["kpi"] == "longtail_top10_mass", "value"]
        t50 = lim.loc[lim["kpi"] == "longtail_top50_mass", "value"]
        if not t10.empty and not t50.empty:
            print(f"• Top-K coverage rises from {float(t10.iloc[0])*100:.0f}% (K=10) to {float(t50.iloc[0])*100:.0f}% (K=50).")

        # lexicon dependency
        lxm = lim.loc[lim["kpi"] == "lexicon_median_max_prevalence_pct", "value"]
        if not lxm.empty:
            print(f"• HurtLex dependency (median max-prevalence): {float(lxm.iloc[0]):.1f}%.")

    if abl is not None and not abl.empty:
        b200 = abl.loc[abl["scenario"] == "bootstrap_median_ci_width_N=200", "delta"] if "scenario" in abl.columns else pd.Series(dtype=float)
        b1000 = abl.loc[abl["scenario"] == "bootstrap_median_ci_width_N=1000", "delta"] if "scenario" in abl.columns else pd.Series(dtype=float)
        if not b200.empty and not b1000.empty:
            print(f"• Bootstrap width shrinks: {float(b200.iloc[0]):.2f} → {float(b1000.iloc[0]):.2f} (median).")

    print("*Note:* category totals can exceed N due to multi-label assignment. "
          "Titles may be non-English; categories/tags keep semantics interpretable (MPU).")

# ----------------------------- Orchestrator ----------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Orchestrate Step 24 with timers, self-check mode, robust optional artefacts, and narrative.

    CLI:
      python -m src.dissertation.24_results_synthesis
      python -m src.dissertation.24_results_synthesis --selfcheck
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true",
                    help="Write *_selfcheck artefacts; do not overwrite canonical files.")
    args = ap.parse_args(argv)

    t_all = _t0("--- Starting Step 24: Results Synthesis ---")

    # One-time cleanup of old rs24_* files (safe if not present)
    _cleanup_legacy_rs24_outputs()

    # Gather KPIs
    abl = gather_ablation_kpis(selfcheck=args.selfcheck)
    lim = gather_limitations_kpis(selfcheck=args.selfcheck)

    # Checklist + narrative
    chk = build_rs24_checklist(abl, lim, selfcheck=args.selfcheck)
    write_narrative(abl, lim, chk, selfcheck=args.selfcheck)

    # Console readout
    qualitative_readout(abl, lim)

    _tend("step24.total_runtime", t_all)
    print("--- Step 24: Results Synthesis Completed ---")

if __name__ == "__main__":
    main()
