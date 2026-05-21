"""
model_training.py
=================

Importable interface to baseline model training used by integration tests
and the pipeline module. The numbered script (07_rf_baseline.py) is the
full corpus-aware runner; this module exposes the clean API surface.

Public API
----------
train_baseline_model(X, y, random_state=42) -> RandomForestClassifier
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_baseline_model(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a baseline RandomForestClassifier.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Binary label vector of shape ``(n_samples,)``.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X, y)
    return model
