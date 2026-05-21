# -*- coding: utf-8 -*-
"""
theme_manager.py

Purpose
-------
Centralize configuration loading and provide a decorator that renders any
matplotlib plot in both light and dark themes with consistent palettes, DPI,
and file naming.

Creates
-------
- Figures saved as <save_path>_light.png and <save_path>_dark.png (and PDFs
  if enabled in YAML).

Performance
-----------
This module prints elapsed times for config loading and each themed plot
render (per-theme + total), useful for your reproducibility appendix.
"""

import os
import functools
import time
import yaml
import matplotlib.pyplot as plt
from pathlib import Path


# --- 1. Advanced Configuration Loading ---

def _resolve_paths(config, key, value):
    """
    Recursively resolve ${project.root} placeholders inside the config.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration object.
    key : Any
        Unused, retained for compatibility with earlier versions.
    value : Any
        A nested value (str/dict/list/other) from the config.

    Returns
    -------
    Any
        The same structure with ${project.root} expanded to an absolute path.
    """
    if isinstance(value, str) and "${project.root}" in value:
        return value.replace("${project.root}", config['project']['root'])
    if isinstance(value, dict):
        return {k: _resolve_paths(config, k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(config, i, v) for i, v in enumerate(value)]
    return value

def load_config():
    """
    Load the master YAML config and resolve ${project.root} placeholders.

    Returns
    -------
    dict | None
        Resolved configuration dict, or None if load/parse fails.

    Notes
    -----
    Prints elapsed time for reproducibility reporting.
    """
    t0 = time.perf_counter()
    config_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        resolved = _resolve_paths(config, 'root', config)
        dt = time.perf_counter() - t0
        print(f"[TIME] theme_manager.load_config: {dt:.2f}s")
        return resolved
    except Exception as e:
        print(f"ERROR: Failed to load or parse configuration file: {e}")
        return None


CONFIG = load_config()

# --- 2. Core Plotting Wrapper ---

def plot_dual_theme(section: str):
    """
    Decorator factory to render a plotting function in both light and dark themes.

    Parameters
    ----------
    section : str
        Palette section to use from settings.yaml (e.g., 'eda', 'fairness', 'models').

    Returns
    -------
    Callable
        A decorator that wraps a function with signature like:
        `func(*args, ax=None, palette=None, **kwargs)` and expects `save_path` in kwargs.

    Notes
    -----
    - Prints per-theme runtime and total runtime.
    - Requires CONFIG loaded successfully.
    """
    def decorator(plot_func):
        @functools.wraps(plot_func)
        def wrapper(*args, **kwargs):
            if CONFIG is None:
                print("Aborting plot generation due to missing configuration.")
                return

            save_path_base = kwargs.get("save_path")
            if not save_path_base:
                raise ValueError("Plotting function must be called with 'save_path'.")

            figsize = kwargs.get("figsize", (10, 8))
            Path(save_path_base).parent.mkdir(parents=True, exist_ok=True)

            t_all = time.perf_counter()
            for theme in ['light', 'dark']:
                t0 = time.perf_counter()
                print(f"Generating '{theme}' theme plot for section '{section}'...")

                theme_config = CONFIG['viz']['themes'][theme]
                plt.style.use('seaborn-v0_8-whitegrid' if theme == 'light' else 'seaborn-v0_8-darkgrid')
                plt.rcParams.update({
                    'figure.facecolor': theme_config['facecolor'],
                    'axes.facecolor': theme_config['facecolor'],
                    'axes.labelcolor': theme_config['textcolor'],
                    'axes.edgecolor': theme_config['gridcolor'],
                    'xtick.color': theme_config['textcolor'],
                    'ytick.color': theme_config['textcolor'],
                    'text.color': theme_config['textcolor'],
                    'grid.color': theme_config['gridcolor'],
                    'legend.facecolor': theme_config['facecolor'],
                    'legend.edgecolor': theme_config['gridcolor']
                })

                fig, ax = plt.subplots(figsize=figsize)
                palette = CONFIG['viz']['palettes']['sections'][section][theme]

                try:
                    # NOTE: ensure *args precede keyword args in calls
                    plot_func(*args, ax=ax, palette=palette, **kwargs)
                except Exception as e:
                    print(f"ERROR executing plotting function '{plot_func.__name__}': {e}")
                    plt.close(fig)
                    continue

                ax.title.set_color(theme_config['textcolor'])
                plt.tight_layout()

                if CONFIG['viz'].get('save_png', True):
                    final_save_path_png = f"{save_path_base}_{theme}.png"
                    fig.savefig(final_save_path_png, dpi=CONFIG['viz']['dpi'], bbox_inches='tight')
                    print(f"✓ Artefact saved: {Path(final_save_path_png).resolve()}")

                if CONFIG['viz'].get('save_pdf', False):
                    final_save_path_pdf = f"{save_path_base}_{theme}.pdf"
                    fig.savefig(final_save_path_pdf, bbox_inches='tight')
                    print(f"✓ Artefact saved: {Path(final_save_path_pdf).resolve()}")

                plt.close(fig)
                dt = time.perf_counter() - t0
                print(f"[TIME] plot_dual_theme[{theme}] {plot_func.__name__}: {dt:.2f}s")

            dt_all = time.perf_counter() - t_all
            print(f"[TIME] plot_dual_theme[total] {plot_func.__name__}: {dt_all:.2f}s")
        return wrapper
    return decorator

if __name__ == "__main__":
    # Minimal self-check: generate a tiny 2-point line using both themes.
    @plot_dual_theme(section="eda")
    def _demo(ax=None, palette=None, **kwargs):
        ax.plot([0, 1], [0, 1], marker="o", lw=2, color=palette[0])
        ax.set_title("Theme Manager Self-Check")
        ax.set_xlabel("x"); ax.set_ylabel("y")

    outdir = Path(CONFIG["paths"]["figures"])
    outdir.mkdir(parents=True, exist_ok=True)
    _demo(save_path=str(outdir / "test_theme_manager"), figsize=(4, 3))
    print("Self-check complete. See test_theme_manager_{light,dark}.png")