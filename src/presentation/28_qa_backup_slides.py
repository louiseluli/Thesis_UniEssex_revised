#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 28 — Q&A Backup Slides (HTML, self-contained)
==================================================

What it is / what it does
-------------------------
Generates a lightweight, single-file HTML deck (keyboard-navigable) that covers
ten high-importance Q&A topics an examiner is likely to press on. Content is
anchored in the literature (intersectionality, fairness metrics, impossibility,
model cards/datasheets, causal claims) and tailored to the project context.

Role & Goal (precise)
---------------------
Provide concise, defensible answers with citations you can reference in viva,
while keeping the slides implementation-simple and dependency-free.

Outputs (canonical; self-check writes a suffixed variant)
---------------------------------------------------------
  Canonical:
    outputs/interactive/qa_backup_slides.html

  Self-check (safe; never overwrites canonical):
    outputs/interactive/qa_backup_slides_selfcheck.html

CLI
---
# Full run (canonical output)
python -m src.presentation.28_qa_backup_slides

# Self-check (random K slides sampled with project seed; safe suffix)
python -m src.presentation.28_qa_backup_slides --selfcheck --sample-k 5

Conventions
-----------
• Imports at the top only, no mid-file imports.
• Timers printed as: [TIME] step28.*: X.XXs and total runtime.
• Seed comes from config (NOT 42); used for self-check sampling.
• Numbers rounded sensibly if/when used in slides.
• MPU note preserved: titles may be non-English; tags/categories preserve semantics.
"""

from __future__ import annotations

# ----------------------------- Imports (top only) -----------------------------
import argparse
import html
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

# ----------------------------- Config & Paths --------------------------------
THEME = None
CONFIG = {}
try:
    from src.utils.theme_manager import ThemeManager, load_config
    THEME = ThemeManager()  # prints its own tiny timer
    CONFIG = load_config() or {}
except Exception:
    pass  # graceful fallback if theming/config not present

ROOT = Path(CONFIG.get("project", {}).get("root", Path(__file__).resolve().parents[2]))
INTER_DIR = ROOT / "outputs" / "interactive"
INTER_DIR.mkdir(parents=True, exist_ok=True)

HTML_CANON = INTER_DIR / "qa_backup_slides.html"
HTML_SCHECK = INTER_DIR / "qa_backup_slides_selfcheck.html"

# reproducibility seed from config (NOT 42)
SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
RNG = np.random.default_rng(SEED)

# ----------------------------- Lightweight timers ----------------------------
def _t0(msg: str) -> float:
    """
    Start a high-resolution timer and print a short heading.

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

# ----------------------------- Slide content ---------------------------------
def _base_slides() -> List[Dict[str, str]]:
    """
    Build the canonical list of 10 slides.
    Content is concise, academically anchored, and tailored for rigorous Q&A.

    Returns
    -------
    List[Dict[str, str]]
        Each slide dict has keys: 'title', 'body' (HTML), and optional 'refs'.
    """
    return [
        {
            "title": "Why focus on Black women?",
            "body": (
                "<ul>"
                "<li><b>Intersectionality</b> (Crenshaw, 1989): harms can <i>compound</i> at race×gender intersections.</li>"
                "<li><b>Empirics</b>: our audit shows persistent engagement gaps and category skew concentrated on Black women.</li>"
                "<li><b>Equity reporting</b>: focusing on the worst-affected group clarifies mitigation priorities.</li>"
                "</ul>"
            ),
            "refs": "Crenshaw (1989); Barocas & Selbst (2016)."
        },
        {
            "title": "Ethical considerations",
            "body": (
                "<ul>"
                "<li><b>Scope</b>: metadata-only analyses—no raw content—minimises exposure risk in a sensitive domain.</li>"
                "<li><b>Use</b>: results are for <i>harm reduction</i> (fairer ranking/moderation), not profiling creators.</li>"
                "<li><b>Documentation</b>: follow Datasheets/Model Cards to disclose limits and uncertainty.</li>"
                "</ul>"
            ),
            "refs": "Gebru et al. (2018); Mitchell et al. (2019); Nissenbaum (2004)."
        },
        {
            "title": "Industry applicability",
            "body": (
                "<ul>"
                "<li><b>Fairness-aware re-ranking</b>: constrain disparities with minimal accuracy loss (Pareto frontier).</li>"
                "<li><b>Moderation levers</b>: denigration lexicon filters; per-category promotion caps.</li>"
                "<li><b>Monitoring</b>: bootstrap CIs, Cliff’s δ, and drift checks for continuous fairness telemetry.</li>"
                "</ul>"
            ),
            "refs": "Corbett-Davies & Goel (2018); Kleinberg et al. (2017)."
        },
        {
            "title": "Computational complexity",
            "body": (
                "<ul>"
                "<li><b>Descriptives</b>: mostly O(N) vectorised passes.</li>"
                "<li><b>Bootstrap</b>: O(B·N) with small B and batching; parallelisable.</li>"
                "<li><b>Models</b>: classical baselines fast; transformer baselines batched; mitigation adds minor overhead.</li>"
                "</ul>"
            ),
            "refs": "Efron & Tibshirani (1994) on bootstrap; scalable vectorisation best practices."
        },
        {
            "title": "Alternative approaches considered",
            "body": (
                "<ul>"
                "<li><b>Causal identification</b>: DAGs & interventions; deferred for future A/B validation.</li>"
                "<li><b>Counterfactual fairness</b> (Kusner et al., 2017): requires structural causal models & sensitive counterfactuals.</li>"
                "<li><b>Differential fairness</b> (Foulds et al., 2020): guarantees across many intersections; heavier estimation burden.</li>"
                "</ul>"
            ),
            "refs": "Pearl (2009); Kusner et al. (2017); Foulds et al. (2020)."
        },
        {
            "title": "Data & representation caveats",
            "body": (
                "<ul>"
                "<li><b>Multi-label taxonomy</b>: category totals can exceed N; sparsity in the long tail.</li>"
                "<li><b>Language</b>: titles may be non-English; <i>tags/categories</i> preserve semantics (MPU).</li>"
                "<li><b>Measurement</b>: inferred attributes/tags can be noisy; report missingness & sensitivity.</li>"
                "</ul>"
            ),
            "refs": "Suresh & Guttag (2019) on bias sources; Datasheets / Model Cards."
        },
        {
            "title": "Metric selection & uncertainty",
            "body": (
                "<ul>"
                "<li><b>Uncertainty</b>: bootstrap CIs; Cliff’s δ for effect sizes; Simpson’s check across years.</li>"
                "<li><b>Fairness metrics</b>: parity vs equalised odds trade-offs; no single metric fits all.</li>"
                "<li><b>Impossibility</b>: incompatible constraints under differing base rates.</li>"
                "</ul>"
            ),
            "refs": "Chouldechova (2017); Kleinberg et al. (2017); Cliff (1993)."
        },
        {
            "title": "Mitigation trade-offs",
            "body": (
                "<ul>"
                "<li><b>Pre-processing</b>: reweighing changes sample distribution; stable but may shift calibration.</li>"
                "<li><b>In-processing</b>: fairness constraints in training; needs careful tuning.</li>"
                "<li><b>Post-processing</b>: per-segment thresholds/ranking tweaks; auditable, product-friendly.</li>"
                "</ul>"
            ),
            "refs": "Hardt et al. (2016); Zafar et al. (2017)."
        },
        {
            "title": "External validity & generalisability",
            "body": (
                "<ul>"
                "<li><b>Platform-specificity</b>: behaviour & policies differ; replicate on other platforms.</li>"
                "<li><b>Temporal drift</b>: re-run regularly; report year-conditioned metrics.</li>"
                "<li><b>Cross-lingual</b>: taxonomy helps transfer despite title language differences.</li>"
                "</ul>"
            ),
            "refs": "Mitchell et al. (2019); Suresh & Guttag (2019)."
        },
        {
            "title": "Limitations & future work",
            "body": (
                "<ul>"
                "<li><b>Causal tests</b>: move beyond associations; interventional audits; online A/B.</li>"
                "<li><b>Richer attributes</b>: more nuanced protected groups; consent-aware collection.</li>"
                "<li><b>Product integration</b>: deploy Pareto frontier guard-rails; governance & logging.</li>"
                "</ul>"
            ),
            "refs": "Barocas & Selbst (2016); Mitchell et al. (2019); Gebru et al. (2018)."
        },
    ]

# ----------------------------- HTML template ---------------------------------
# IMPORTANT: we use a literal token (%%SLIDES%%) and string .replace(), NOT str.format()
# CSS/JS contains braces, so .format would break with 'unexpected { in field name'.
_HTML_TMPL = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Q&A Backup Slides</title>
<style>
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  .slide { width:100vw; height:100vh; display:flex; flex-direction:column; justify-content:center;
            padding:6vh 10vw; box-sizing:border-box; }
  .slide h1 { margin:0 0 .6em 0; font-size:2.2em; }
  .slide p, .slide ul { font-size:1.2em; line-height:1.55; max-width: 72ch; margin:.2em 0; }
  .refs { margin-top:1em; font-size:.95em; opacity:.8; }
  nav { position:fixed; bottom:12px; right:16px; opacity:.75; font-size:.9em; }
  .pager { position:fixed; bottom:12px; left:16px; opacity:.75; font-size:.9em; }
  .active { outline: 4px solid rgba(0,0,0,.06); outline-offset: -4px; }
</style>
<script>
  (function(){
    function show(idx){
      const slides=[...document.querySelectorAll('.slide')];
      slides.forEach(s=>s.classList.remove('active'));
      slides[Math.max(0,Math.min(idx,slides.length-1))].classList.add('active');
      document.querySelector('.pager').textContent = (idx+1)+'/'+slides.length;
      window.__idx = idx;
    }
    document.addEventListener('keydown', e => {
      const slides=document.querySelectorAll('.slide');
      if (!slides.length) return;
      const n = slides.length, i = (window.__idx||0);
      if (e.key==='ArrowRight' || e.key==='PageDown') show(Math.min(i+1,n-1));
      if (e.key==='ArrowLeft'  || e.key==='PageUp')   show(Math.max(i-1,0));
      if (e.key==='Home') show(0);
      if (e.key==='End')  show(n-1);
    });
    window.addEventListener('load', ()=>show(0));
  })();
</script>
</head>
<body>
%%SLIDES%%
<nav>Use ← / → (Home/End) to navigate</nav>
<div class="pager"></div>
</body>
</html>
"""

# ----------------------------- Rendering helpers -----------------------------
def _render_slides(slides: List[Dict[str, str]]) -> str:
    """
    Render a list of slides (title/body/refs) into HTML <section> blocks.

    Parameters
    ----------
    slides : List[Dict[str, str]]
        Each dict has 'title', 'body' (HTML), and optional 'refs' (string).

    Returns
    -------
    str
        HTML string for all slides joined.
    """
    parts = []
    for s in slides:
        title = html.escape(s["title"])
        body_html = s["body"]  # already HTML
        refs = s.get("refs")
        refs_html = f'<div class="refs">Refs: {html.escape(refs)}</div>' if refs else ""
        parts.append(f'<section class="slide"><h1>{title}</h1>{body_html}{refs_html}</section>')
    return "\n".join(parts)


def _write_html(html_text: str, path: Path) -> None:
    """
    Write HTML text to disk with a confirmation log line.

    Parameters
    ----------
    html_text : str
        Rendered HTML content.
    path : Path
        Output path.
    """
    path.write_text(html_text, encoding="utf-8")
    print(f"[WRITE] {path}")


def _delete_stale(path: Path) -> None:
    """
    Delete a file if it exists, loudly (used to clear stale self-check in full runs).

    Parameters
    ----------
    path : Path
        File to delete.
    """
    try:
        if path.exists():
            path.unlink()
            print(f"[DELETE] {path}")
    except Exception:
        pass

# ----------------------------- Qualitative checks ----------------------------
def _qualitative_readout(slides: List[Dict[str, str]]) -> None:
    """
    Print a compact readout about slide lengths and potential outliers.

    Outlier heuristic
    -----------------
    Flags any slide whose combined title+body+refs length exceeds the 90th
    percentile + 1.5×IQR (simple robust rule), so you can decide to trim.

    Parameters
    ----------
    slides : List[Dict[str, str]]
        Slide dictionaries rendered for the deck.
    """
    print("\n--- Quick qualitative readout ---")
    lens = []
    for s in slides:
        text = s["title"] + " " + s["body"] + " " + s.get("refs", "")
        lens.append(len(html.unescape(text)))
    if not lens:
        print("• No slides rendered.")
        return
    lens = np.asarray(lens, dtype=float)
    p90 = float(np.percentile(lens, 90))
    q1, q3 = np.percentile(lens, [25, 75])
    iqr = float(q3 - q1)
    thresh = p90 + 1.5 * iqr
    max_len = int(lens.max())
    outliers = [i for i, L in enumerate(lens) if L > thresh]

    print(f"• Slides rendered: {len(lens)}; max length: {max_len} chars; outlier threshold: {int(thresh)}.")
    if outliers:
        print("• Length outliers:", ", ".join(str(i+1) for i in outliers))
    else:
        print("• No length outliers detected.")
    print("*Note:* titles may be non-English; tags/categories (MPU) preserve semantics.")

# ----------------------------- Orchestrator ----------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Orchestrate Step 28 with timers, self-check sampling, and robust output.

    CLI
    ---
    Full:
      python -m src.presentation.28_qa_backup_slides
    Self-check:
      python -m src.presentation.28_qa_backup_slides --selfcheck --sample-k 5

    Parameters
    ----------
    argv : Optional[List[str]]
        Optional CLI argument list for programmatic invocation.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true",
                    help="Write *_selfcheck.html and DO NOT overwrite canonical output.")
    ap.add_argument("--sample-k", type=int, default=None,
                    help="In selfcheck mode, sample K slides at random (seeded by config).")
    args = ap.parse_args(argv)

    t_all = _t0("--- Starting Step 28: Q&A Backup Slides ---")

    # Build slide list
    t = _t0("A) Building slide content ...")
    slides = _base_slides()

    # Self-check sampling (safe & deterministic via config seed)
    if args.selfcheck and args.sample_k:
        k = int(max(1, min(args.sample_k, len(slides))))
        idx = RNG.choice(len(slides), size=k, replace=False)
        slides = [slides[i] for i in sorted(idx)]
    _tend("step28.build", t)

    # Delete stale self-check on full run (avoid confusion)
    if not args.selfcheck:
        _delete_stale(HTML_SCHECK)

    # Render & write
    t = _t0("B) Rendering & writing HTML ...")
    html_slides = _render_slides(slides)
    # Use literal token replacement; do NOT use str.format due to braces in CSS/JS
    html_text = _HTML_TMPL.replace("%%SLIDES%%", html_slides)
    out_path = HTML_SCHECK if args.selfcheck else HTML_CANON
    _write_html(html_text, out_path)
    _tend("step28.render", t)

    # Qualitative readout
    _qualitative_readout(slides)

    _tend("step28.total_runtime", t_all)
    print("--- Step 28: Q&A Backup Slides Completed Successfully ---")

# ----------------------------- Entry point -----------------------------------
if __name__ == "__main__":
    main()
