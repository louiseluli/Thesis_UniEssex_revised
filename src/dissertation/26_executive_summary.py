#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 26 — Executive Summary (≤ 500 words)
=========================================

What it is / what it does
-------------------------
Builds a concise, quantified executive summary using existing artefacts:
  • outputs/data/ml_corpus.parquet (size; also supports 01_ml_corpus.parquet)
  • outputs/data/eng17_yearly_bw_gaps.csv (latest engagement gaps)
  • outputs/data/adv19_bw_effect_sizes.csv (Cliff’s δ)
  • outputs/data/cgd_black_women_under_over.csv (category extremes)
  • outputs/data/adv19_harm_relative_risks.csv (if present)

Output (canonical; self-check writes a suffixed file and never overwrites):
  • dissertation/executive_summary[ _selfcheck].md

Role & Goal (precise)
---------------------
Summarize the full pipeline’s most decision-relevant signals for non-technical
readers, with tight wording (≤ 500 words), rounded numbers, and explicit caveats
(multi-label category totals can exceed N; non-English titles ⇒ tags/cats/MPU).

CLI
---
Full run:
    python -m src.dissertation.26_executive_summary

Self-check (safe suffix, no overwrites):
    python -m src.dissertation.26_executive_summary --selfcheck --max-words 450

Conventions
-----------
• Imports at the top only. Lightweight timers printed as: [TIME] …: X.XXs
• Seed pulled from config (NOT 42).
• Years as integers; ratings to 1 decimal; others sensibly rounded.
• Deletions called out when removing stale *_selfcheck outputs in full mode.
• MPU note preserved: titles can be in other languages; tags/categories help.
"""

from __future__ import annotations

# ----------------------------- Imports (top only) -----------------------------
import argparse
import math
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# Optional fast rowcount via pyarrow (top-only import to respect your rule)
try:
    import pyarrow.parquet as PQ  # type: ignore
except Exception:  # pragma: no cover
    PQ = None  # graceful fallback

# ----------------------------- Config / Paths --------------------------------
THEME = None  # not plotting here; still attempt to load config like other steps
CONFIG = {}
try:
    from src.utils.theme_manager import ThemeManager, load_config
    THEME = ThemeManager()
    CONFIG = load_config() or {}
except Exception:
    pass  # graceful fallback

ROOT = Path(CONFIG.get("project", {}).get("root", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(CONFIG.get("paths", {}).get("data", ROOT / "outputs" / "data"))
DISC_DIR = ROOT / "dissertation"
for d in (DATA_DIR, DISC_DIR):
    d.mkdir(parents=True, exist_ok=True)

# reproducibility seed from config (NOT 42)
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
np.random.seed(SEED)

# ----------------------------- Lightweight timers ----------------------------
def _t0(msg: str) -> float:
    """
    Start a high-resolution timer and print a heading.

    Parameters
    ----------
    msg : str
        Message printed before timing starts.

    Returns
    -------
    float
        perf_counter start time.
    """
    print(msg)
    return time.perf_counter()


def _tend(label: str, t_start: float) -> None:
    """
    Stop timer and print standardized [TIME] message.

    Parameters
    ----------
    label : str
        Short label describing the timed block.
    t_start : float
        Start time from _t0.
    """
    print(f"[TIME] {label}: {time.perf_counter() - t_start:.2f}s")

# ----------------------------- IO helpers & utils ----------------------------
def _suffix(name: str, selfcheck: bool) -> str:
    """
    Append _selfcheck to a base filename (before extension) when requested.

    Parameters
    ----------
    name : str
        Base filename.
    selfcheck : bool
        Whether to append the suffix.

    Returns
    -------
    str
        Filename with optional suffix.
    """
    if not selfcheck:
        return name
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        return f"{stem}_selfcheck.{ext}"
    return f"{name}_selfcheck"


def _delete_if_exists(p: Path) -> None:
    """
    Delete a path if it exists, with a loud console log.

    Parameters
    ----------
    p : Path
        Target path to delete.
    """
    try:
        if p.exists():
            p.unlink()
            print(f"[DELETE] {p}")
    except Exception:
        pass


def _read_csv(p: Path) -> Optional[pd.DataFrame]:
    """
    Safe CSV reader returning None if missing or unreadable.

    Parameters
    ----------
    p : Path
        CSV path.

    Returns
    -------
    Optional[pd.DataFrame]
        Dataframe or None on failure.
    """
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None


def _round_or_none(x: Optional[float], digits: int) -> Optional[float]:
    """
    Round finite floats; return None for non-finite/None.

    Parameters
    ----------
    x : Optional[float]
        Value to round.
    digits : int
        Number of decimal places.

    Returns
    -------
    Optional[float]
        Rounded value or None.
    """
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return round(v, digits)


def _fmt_sig(x: Optional[float], sig: int = 3) -> str:
    """
    Format with significant figures, or '—' if missing.

    Parameters
    ----------
    x : Optional[float]
        Value to format.
    sig : int
        Significant figures.

    Returns
    -------
    str
        Formatted string or '—'.
    """
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    try:
        return f"{float(x):.{sig}g}"
    except Exception:
        return "—"


def _fmt_1dp(x: Optional[float]) -> str:
    """
    Format as one decimal place (or '—' if missing).

    Parameters
    ----------
    x : Optional[float]
        Value to format.

    Returns
    -------
    str
        Formatted string or '—'.
    """
    try:
        v = float(x)
        if not math.isfinite(v):
            return "—"
        return f"{v:.1f}"
    except Exception:
        return "—"


def _enforce_word_cap(text: str, max_words: int) -> str:
    """
    Hard cap the number of words to max_words.

    Parameters
    ----------
    text : str
        Input text.
    max_words : int
        Maximum word count.

    Returns
    -------
    str
        Possibly truncated text, with an ellipsis if truncated.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"

# ----------------------------- Data extractors --------------------------------
def _corpus_size() -> Optional[int]:
    """
    Return number of rows in the main parquet corpus.

    Robust path search (in priority order):
      • outputs/data/ml_corpus.parquet
      • outputs/data/01_ml_corpus.parquet
      • any outputs/data/ml_corpus*.parquet (first match)

    Uses pyarrow metadata when available (faster).

    Returns
    -------
    Optional[int]
        Row count or None if unavailable.
    """
    candidates = [
        DATA_DIR / "ml_corpus.parquet",
        DATA_DIR / "01_ml_corpus.parquet",
    ]
    # add any ml_corpus*.parquet
    candidates.extend(sorted(DATA_DIR.glob("ml_corpus*.parquet")))
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        return None

    if PQ is not None:
        try:
            meta = PQ.ParquetFile(target)
            return int(meta.metadata.num_rows)
        except Exception:
            pass

    try:
        return int(len(pd.read_parquet(target)))
    except Exception:
        return None


def _latest_bw_gaps() -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float],
                                Optional[float], Optional[float], Optional[float]]:
    """
    Extract the latest year's BW engagement gaps (views/day & rating) with 95% CIs.

    Returns
    -------
    (year, vpd_gap, vpd_lo, vpd_hi, rat_gap, rat_lo, rat_hi)
    """
    g = _read_csv(DATA_DIR / "eng17_yearly_bw_gaps.csv")
    if g is None or g.empty:
        return (None,) * 7

    cols = {c.lower(): c for c in g.columns}
    ycol = cols.get("year")
    v_gap = cols.get("gap_views_per_day") or cols.get("views_per_day_gap") or cols.get("gap_vpd")
    v_lo  = cols.get("gap_views_ci_lo") or cols.get("views_ci_lo") or cols.get("vpd_ci_lo")
    v_hi  = cols.get("gap_views_ci_hi") or cols.get("views_ci_hi") or cols.get("vpd_ci_hi")
    r_gap = cols.get("gap_rating") or cols.get("rating_gap") or cols.get("gap_score")
    r_lo  = cols.get("gap_rating_ci_lo") or cols.get("rating_ci_lo")
    r_hi  = cols.get("gap_rating_ci_hi") or cols.get("rating_ci_hi")

    try:
        gg = g.sort_values(ycol).tail(1) if ycol else g.tail(1)
        year = int(gg[ycol].iloc[0]) if ycol else None
    except Exception:
        year = None

    def _get(frame, c):
        try:
            return float(pd.to_numeric(frame[c], errors="coerce").iloc[0]) if c and c in frame.columns else None
        except Exception:
            return None

    vpd_gap = _get(gg, v_gap)
    vpd_lo  = _get(gg, v_lo)
    vpd_hi  = _get(gg, v_hi)
    rat_gap = _get(gg, r_gap)
    rat_lo  = _get(gg, r_lo)
    rat_hi  = _get(gg, r_hi)
    return year, vpd_gap, vpd_lo, vpd_hi, rat_gap, rat_lo, rat_hi


def _effect_sizes() -> Tuple[Optional[float], Optional[float]]:
    """
    Extract Cliff's δ for views/day and rating (Black women − others).

    Returns
    -------
    (delta_vpd, delta_rating)
    """
    eff = _read_csv(DATA_DIR / "adv19_bw_effect_sizes.csv")
    if eff is None or eff.empty:
        return (None, None)

    l = {c.lower(): c for c in eff.columns}
    mcol = l.get("metric")
    dcol = l.get("cliffs_delta") or l.get("cliffsdelta") or l.get("delta")

    if mcol is None or dcol is None:
        return (None, None)

    def _pick(metric_name: str) -> Optional[float]:
        try:
            val = eff.loc[eff[mcol].astype(str).str.lower() == metric_name, dcol]
            v = float(pd.to_numeric(val, errors="coerce").iloc[0])
            return v
        except Exception:
            return None

    delta_vpd = _pick("views_per_day")
    delta_rat = _pick("rating")
    return delta_vpd, delta_rat


def _category_extremes() -> Tuple[Optional[Tuple[str, float]], Optional[Tuple[str, float]]]:
    """
    Return most under- and over-represented categories for BW by log2 RR.

    Returns
    -------
    (under_tuple, over_tuple)
        Each tuple is (category, log2_rr). Either may be None.
    """
    cgd = _read_csv(DATA_DIR / "cgd_black_women_under_over.csv")
    if cgd is None or cgd.empty:
        return (None, None)

    cols = {c.lower(): c for c in cgd.columns}
    cat = cols.get("category") or cols.get("tag") or cols.get("label")
    rr  = cols.get("log2_rr_bw") or cols.get("log2rrbw") or cols.get("log2_rr")

    if cat is None or rr is None:
        return (None, None)

    try:
        cgd[rr] = pd.to_numeric(cgd[rr], errors="coerce")
        under = cgd.nsmallest(1, rr).dropna(subset=[rr])
        over  = cgd.nlargest(1, rr).dropna(subset=[rr])
        under_cat = (str(under[cat].iloc[0]), float(under[rr].iloc[0])) if not under.empty else None
        over_cat  = (str(over[cat].iloc[0]),  float(over[rr].iloc[0]))  if not over.empty else None
        return under_cat, over_cat
    except Exception:
        return (None, None)


def _harm_rr_count_gt1() -> Optional[int]:
    """
    Count how many HurtLex categories have RR_bw_vs_others > 1 (if available).

    Returns
    -------
    Optional[int]
        Count or None.
    """
    rr = _read_csv(DATA_DIR / "adv19_harm_relative_risks.csv")
    if rr is None or rr.empty:
        return None
    cols = {c.lower(): c for c in rr.columns}
    rcol = cols.get("rr_bw_vs_others") or cols.get("rr") or cols.get("relative_risk")
    if rcol is None:
        return None
    try:
        vals = pd.to_numeric(rr[rcol], errors="coerce")
        return int((vals > 1.0).sum())
    except Exception:
        return None

# ----------------------------- Narrative builder -----------------------------
def _build_summary(
    n_rows: Optional[int],
    latest_year: Optional[int],
    vpd_gap: Optional[float], vpd_lo: Optional[float], vpd_hi: Optional[float],
    rat_gap: Optional[float], rat_lo: Optional[float], rat_hi: Optional[float],
    delta_vpd: Optional[float], delta_rat: Optional[float],
    under_cat: Optional[Tuple[str, float]], over_cat: Optional[Tuple[str, float]],
    rr_count_gt1: Optional[int],
    max_words: int
) -> str:
    """
    Create a ≤ max_words Markdown executive summary with rounded metrics.

    Parameters
    ----------
    n_rows : Optional[int]
        Corpus size.
    latest_year : Optional[int]
        Most recent year in the gaps table.
    vpd_gap, vpd_lo, vpd_hi : Optional[float]
        Views-per-day gap and CI.
    rat_gap, rat_lo, rat_hi : Optional[float]
        Rating gap and CI.
    delta_vpd, delta_rat : Optional[float]
        Cliff's δ effect sizes.
    under_cat, over_cat : Optional[Tuple[str, float]]
        Category extremes as (name, log2_rr).
    rr_count_gt1 : Optional[int]
        Count of HurtLex categories with RR>1.
    max_words : int
        Hard cap on words in the final summary.

    Returns
    -------
    str
        Markdown executive summary (≤ max_words).
    """
    lines: list[str] = []
    lines.append("# Executive Summary\n")

    if n_rows is not None:
        lines.append(
            f"This dissertation presents a large-scale, metadata-driven fairness audit in the adult-content domain, "
            f"analysing **{n_rows:,}** videos from a major platform. We assess representational and allocative harms "
            f"with a focus on **Black women**, combining temporal, engagement and linguistic harm analyses with "
            f"model-based fairness evaluation and mitigation."
        )
    else:
        lines.append(
            "This dissertation presents a large-scale, metadata-driven fairness audit in the adult-content domain. "
            "We assess representational and allocative harms with a focus on **Black women**, combining temporal, "
            "engagement and linguistic harm analyses with model-based fairness evaluation and mitigation."
        )

    lines.append("\n\n## Key findings")

    # 1) Engagement gaps (VPD ~ 3 sig figs; Rating = 1dp)
    if latest_year is not None and vpd_gap is not None:
        vpd_gap_s = _fmt_sig(vpd_gap, 3); vpd_lo_s = _fmt_sig(vpd_lo, 3); vpd_hi_s = _fmt_sig(vpd_hi, 3)
        rat_gap_s = _fmt_1dp(rat_gap);    rat_lo_s = _fmt_1dp(rat_lo);    rat_hi_s = _fmt_1dp(rat_hi)
        lines.append(
            f"- **Engagement gap persists:** In **{int(latest_year)}**, Black-women videos received "
            f"**Δ views/day = {vpd_gap_s}** (95% CI [{vpd_lo_s}, {vpd_hi_s}]) relative to others; "
            f"rating gap **Δ rating = {rat_gap_s}** (95% CI [{rat_lo_s}, {rat_hi_s}])."
        )

    # 2) Effect sizes
    if delta_vpd is not None:
        lines.append(f"- **Effect size (views/day):** Cliff’s δ (BW − Others) = **{_fmt_sig(delta_vpd, 3)}**, indicating a distribution shift.")
    if delta_rat is not None:
        lines.append(f"- **Effect size (rating):** Cliff’s δ = **{_fmt_sig(delta_rat, 3)}**, evidencing systematic rating differences.")

    # 3) Category extremes
    if under_cat and over_cat:
        lines.append(
            f"- **Category skew:** Most under-represented for BW: **{under_cat[0]}** *(log2 RR={under_cat[1]:.2f})*; "
            f"most over-represented: **{over_cat[0]}** *(log2 RR={over_cat[1]:.2f})*."
        )

    # 4) Linguistic harms
    if rr_count_gt1 is not None:
        lines.append(
            f"- **Linguistic harms:** In **{rr_count_gt1}** HurtLex categories, the relative risk is **RR>1** for BW videos, "
            f"suggesting elevated exposure to denigrating terms."
        )

    lines.append("\n## Implications for industry")
    lines.append("- **Fairness-aware ranking:** Construct a Pareto frontier of accuracy vs fairness to constrain disparities while preserving engagement.")
    lines.append("- **Taxonomy-guided moderation:** Use category-specific over/under-representation and harm RRs for targeted, auditable interventions.")
    lines.append("- **Monitoring at scale:** Use uncertainty-aware metrics (bootstrap CIs, effect sizes) as a continuous fairness monitor.")

    lines.append("\n## Future research")
    lines.append("- **Causal identification:** Move beyond correlational gaps via causal graphs and interventional tests.")
    lines.append("- **Intersectional constraints:** Evaluate differential and counterfactual-individual fairness under domain-specific utility constraints.")

    # Footnote/caveats
    lines.append(
        "\n*Notes:* Category totals can exceed N due to **multi-label** assignment. Some titles are not in English; "
        "tags/categories (MPU) preserve interpretable semantics. Years are integers; ratings shown with one decimal place; "
        "other metrics rounded sensibly."
    )

    text = "\n".join(lines).strip() + "\n"
    return _enforce_word_cap(text, max_words)

# ----------------------------- Orchestrator ----------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Orchestrate Step 26 with timers, safe self-check, robust optional inputs,
    deletions called out, and deterministic rounding.

    Examples
    --------
    python -m src.dissertation.26_executive_summary
    python -m src.dissertation.26_executive_summary --selfcheck --max-words 450
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true", help="Write *_selfcheck artefact; never overwrite canonical.")
    ap.add_argument("--max-words", type=int, default=500, help="Hard cap on summary length (default: 500).")
    args = ap.parse_args(argv)

    t_all = _t0("--- Starting Step 26: Executive Summary ---")

    # In full mode, delete stale selfcheck file to avoid confusion
    if not args.selfcheck:
        _delete_if_exists(DISC_DIR / "executive_summary_selfcheck.md")

    # 1) Corpus size
    t = _t0("A) Corpus size ...")
    n_rows = _corpus_size()
    _tend("step26.corpus_size", t)

    # 2) Latest yearly BW gaps
    t = _t0("B) Latest BW engagement gaps ...")
    latest_year, vpd_gap, vpd_lo, vpd_hi, rat_gap, rat_lo, rat_hi = _latest_bw_gaps()
    _tend("step26.gaps", t)

    # 3) BW effect sizes
    t = _t0("C) Effect sizes (Cliff’s δ) ...")
    delta_vpd, delta_rat = _effect_sizes()
    _tend("step26.effect_sizes", t)

    # 4) Category extremes
    t = _t0("D) Category extremes (log2 RR) ...")
    under_cat, over_cat = _category_extremes()
    _tend("step26.category_extremes", t)

    # 5) HurtLex RR>1 count
    t = _t0("E) HurtLex RR>1 count ...")
    rr_count_gt1 = _harm_rr_count_gt1()
    _tend("step26.hurtlex_rr", t)

    # 6) Build narrative
    t = _t0("F) Narrative build & write ...")
    md = _build_summary(
        n_rows,
        latest_year, vpd_gap, vpd_lo, vpd_hi, rat_gap, rat_lo, rat_hi,
        delta_vpd, delta_rat,
        under_cat, over_cat,
        rr_count_gt1,
        max_words=int(args.max_words),
    )
    out = DISC_DIR / _suffix("executive_summary.md", args.selfcheck)
    out.write_text(md, encoding="utf-8")
    print(f"[WRITE] {out}")
    _tend("step26.narrative", t)

    # 7) Quick qualitative readout
    print("\n--- Quick qualitative readout ---")
    if n_rows is not None:
        print(f"• Corpus size: {n_rows:,} rows")
    if latest_year is not None and vpd_gap is not None:
        print(f"• {int(latest_year)}: Δ views/day = {_fmt_sig(vpd_gap)} "
              f"(95% CI [{_fmt_sig(vpd_lo)}, {_fmt_sig(vpd_hi)}]); "
              f"Δ rating = {_fmt_1dp(rat_gap)} (95% CI [{_fmt_1dp(rat_lo)}, {_fmt_1dp(rat_hi)}])")
    if (delta_vpd is not None) or (delta_rat is not None):
        dv = _fmt_sig(delta_vpd); dr = _fmt_sig(delta_rat)
        print(f"• Cliff’s δ (VPD, Rating): {dv}, {dr}")
    if under_cat and over_cat:
        print(f"• Category extremes (log2 RR): under={under_cat[0]} ({under_cat[1]:.2f}); "
              f"over={over_cat[0]} ({over_cat[1]:.2f})")
    if rr_count_gt1 is not None:
        print(f"• HurtLex categories with RR>1 (BW): {rr_count_gt1}")
    print("*Note:* category totals can exceed N (multi-label). Some titles are not in English; "
          "tags/categories (MPU) preserve semantics.")

    _tend("step26.total_runtime", t_all)
    print("--- Step 26: Executive Summary Completed Successfully ---")

# ----------------------------- Entry point -----------------------------------
if __name__ == "__main__":
    main()
