"""
postprocessing_mitigation.py
=============================

Clean, importable interface for post-processing (threshold optimization)
bias mitigation. The numbered pipeline script (12_postprocessing_mitigation.py)
uses Fairlearn's ThresholdOptimizer for the full corpus run; this module
exposes a pure-function API that tests import directly.

Public API
----------
find_optimal_thresholds(y_true, y_scores, groups, metric, target) -> dict
    Find per-group decision thresholds that equalize ``metric`` across groups.

apply_group_thresholds(y_scores, groups, thresholds) -> np.ndarray
    Apply a per-group threshold dictionary to a score array.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tpr_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, t: float) -> float:
    pred = (y_scores >= t).astype(int)
    pos_mask = y_true == 1
    if pos_mask.sum() == 0:
        return 0.0
    return float(pred[pos_mask].mean())


def _fpr_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, t: float) -> float:
    pred = (y_scores >= t).astype(int)
    neg_mask = y_true == 0
    if neg_mask.sum() == 0:
        return 0.0
    return float(pred[neg_mask].mean())


def _precision_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, t: float) -> float:
    pred = (y_scores >= t).astype(int)
    if pred.sum() == 0:
        return 0.0
    return float(y_true[pred == 1].mean())


_METRIC_FNS = {
    "tpr": _tpr_at_threshold,
    "fpr": _fpr_at_threshold,
    "precision": _precision_at_threshold,
}


def _best_threshold_for_group(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str,
    target: Optional[float],
) -> float:
    """
    Find the threshold for one group that gets ``metric`` as close to
    ``target`` as possible. If ``target`` is None, find the threshold that
    maximises the group's own TPR subject to a precision floor.

    We scan a fine grid of thresholds rather than calling a black-box optimiser
    so the function is deterministic, dependency-free, and O(n·grid_size).
    """
    metric_fn = _METRIC_FNS[metric]
    grid = np.linspace(0.01, 0.99, 199)

    if target is not None:
        # Minimise |metric(t) - target|
        diffs = [abs(metric_fn(y_true, y_scores, t) - target) for t in grid]
        return float(grid[int(np.argmin(diffs))])

    # target is None: pick the threshold that maximises TPR, but not trivially 0.
    # Use the elbow of the ROC curve via Youden's J statistic (TPR - FPR).
    tprs = np.array([_tpr_at_threshold(y_true, y_scores, t) for t in grid])
    fprs = np.array([_fpr_at_threshold(y_true, y_scores, t) for t in grid])
    youden = tprs - fprs
    return float(grid[int(np.argmax(youden))])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_optimal_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    groups: np.ndarray,
    metric: str = "tpr",
    target: Optional[float] = None,
) -> Dict[str, float]:
    """
    Find per-group decision thresholds that equalize ``metric`` across groups.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels ``{0, 1}``.
    y_scores : np.ndarray
        Continuous scores / probabilities in ``[0, 1]``.
    groups : np.ndarray
        Group membership labels (any hashable dtype).
    metric : {'tpr', 'fpr', 'precision'}
        The rate metric to equalize. Default ``'tpr'`` (equal opportunity).
    target : float or None
        Target value for the metric. If ``None``, the target is the
        population-wide average of ``metric`` at threshold 0.5, which
        equalizes each group toward the current global average — the natural
        fairness-conscious default (equalized opportunity / demographic parity).

    Returns
    -------
    dict
        Mapping ``{group_label: threshold}`` with one entry per unique group.

    Notes
    -----
    * When ``target`` is provided, each group's threshold is chosen to bring
      its metric as close to ``target`` as possible (grid search on [0.01, 0.99]).
    * The function is fully deterministic — no random state is involved.
    * Different ``metric`` values produce different targets (e.g. average TPR ≠
      average FPR ≠ average precision at threshold 0.5), so per-group thresholds
      will generally differ across metrics.
    """
    if metric not in _METRIC_FNS:
        raise ValueError(f"metric must be one of {list(_METRIC_FNS)}; got '{metric}'")

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    # Resolve target: if None, equalize to the population-wide average of the
    # metric computed at the default threshold of 0.5.
    if target is None:
        metric_fn = _METRIC_FNS[metric]
        group_values = []
        for grp in unique_groups:
            mask = groups == grp
            group_values.append(metric_fn(y_true[mask], y_scores[mask], 0.5))
        target = float(np.mean(group_values))

    thresholds: Dict[str, float] = {}
    for grp in unique_groups:
        mask = groups == grp
        thresholds[grp] = _best_threshold_for_group(
            y_true[mask], y_scores[mask], metric=metric, target=target
        )
    return thresholds


def apply_group_thresholds(
    y_scores: np.ndarray,
    groups: np.ndarray,
    thresholds: Dict[str, float],
) -> np.ndarray:
    """
    Apply per-group decision thresholds to a score array.

    Parameters
    ----------
    y_scores : np.ndarray
        Continuous scores / probabilities in ``[0, 1]``.
    groups : np.ndarray
        Group membership labels aligned to ``y_scores``.
    thresholds : dict
        Mapping ``{group_label: threshold}`` as returned by
        ``find_optimal_thresholds``.

    Returns
    -------
    np.ndarray
        Binary prediction array ``{0, 1}`` of the same length as ``y_scores``.
    """
    y_scores = np.asarray(y_scores, dtype=float)
    groups = np.asarray(groups)
    y_pred = np.zeros(len(y_scores), dtype=int)

    for grp, t in thresholds.items():
        mask = groups == grp
        y_pred[mask] = (y_scores[mask] >= t).astype(int)

    return y_pred
