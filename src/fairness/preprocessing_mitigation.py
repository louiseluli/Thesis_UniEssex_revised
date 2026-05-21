"""
preprocessing_mitigation.py
============================

Clean, importable interface to the Kamiran–Calders Reweighing scheme.
The numbered pipeline script (10_preprocessing_mitigation.py) is the full
corpus-aware runner; this module exposes the pure-function API that tests
and other modules import directly.

Public API
----------
compute_reweighing_weights(df, protected_attr, label) -> np.ndarray
    Compute per-instance Kamiran–Calders weights for a DataFrame.

apply_reweighing(X, y, groups, random_state=42) -> RandomForestClassifier
    Train a RandomForest with reweighing applied; returns fitted model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def compute_reweighing_weights(
    df: pd.DataFrame,
    protected_attr: str,
    label: str,
) -> np.ndarray:
    """
    Compute Kamiran–Calders instance weights for pre-processing bias mitigation.

    Each weight is::

        w(a, y) = P(A=a) * P(Y=y) / P(A=a, Y=y)

    The resulting vector is normalized so that ``weights.sum() == len(df)``,
    which keeps the effective sample size constant and is compatible with
    ``sklearn`` estimators' ``sample_weight`` parameter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains at least ``protected_attr`` and ``label`` columns.
    protected_attr : str
        Column name of the protected / sensitive attribute.
    label : str
        Column name of the binary outcome label (0 / 1).

    Returns
    -------
    np.ndarray
        Float array of shape ``(len(df),)`` with positive weights.

    Notes
    -----
    * Weights are purely determined by marginal statistics; the function is
      therefore deterministic for a given input.
    * Single-group datasets return uniform weights (all 1.0).
    * Empty (A=a, Y=y) cells get weight 1.0 (no adjustment where no signal).
    """
    a = df[protected_attr].to_numpy()
    y = df[label].to_numpy().astype(float)
    n = len(df)

    groups = np.unique(a)
    if len(groups) <= 1:
        return np.ones(n, dtype=float)

    weights = np.ones(n, dtype=float)

    labels = np.unique(y)
    for grp in groups:
        p_a = np.mean(a == grp)
        for lbl in labels:
            p_y = np.mean(y == lbl)
            p_ay = np.mean((a == grp) & (y == lbl))
            if p_ay > 0:
                w = (p_a * p_y) / p_ay
            else:
                w = 1.0
            mask = (a == grp) & (y == lbl)
            weights[mask] = w

    # Normalize so weights sum to n (keeps effective sample size)
    weights = weights * (n / weights.sum())
    return weights


def apply_reweighing(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier with Kamiran–Calders reweighing.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Binary label vector of shape ``(n_samples,)``.
    groups : np.ndarray
        Sensitive-group labels of shape ``(n_samples,)``.
    random_state : int
        Random seed forwarded to ``RandomForestClassifier`` for reproducibility.

    Returns
    -------
    RandomForestClassifier
        Fitted model with sample weights applied during training.
    """
    df = pd.DataFrame({"group": groups, "label": y})
    weights = compute_reweighing_weights(df, "group", "label")
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X, y, sample_weight=weights)
    return model
