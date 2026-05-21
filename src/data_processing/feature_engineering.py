"""
feature_engineering.py
=======================

Importable feature-engineering interface used by integration tests and the
pipeline module. Takes a raw corpus DataFrame (as produced by
01_corpus_builder.py) and returns (X, y, groups) ready for ML.

Public API
----------
create_features(df) -> (np.ndarray, np.ndarray, np.ndarray)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, issparse


_REQUIRED_COLUMNS = {"is_amateur"}

_TEXT_COLS = ["title", "tags"]
_NUM_COLS = ["duration", "rating"]
_GROUP_RACE_COLS = [
    "race_ethnicity_white",
    "race_ethnicity_black",
    "race_ethnicity_asian",
    "race_ethnicity_latina",
]


def _build_group_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Derive intersectional group labels from boolean race/gender columns.

    Mirrors the logic in fairness_evaluation_utils.group_labels_intersectional
    but operates on a raw corpus DataFrame (may use 'gender_woman' rather than
    'gender_female').
    """
    n = len(df)
    labels = np.full(n, "Other", dtype=object)

    # Accept both column name conventions
    female = (
        df.get("gender_female", df.get("gender_woman", pd.Series(0, index=df.index)))
        .fillna(0)
        .astype(float)
        .to_numpy()
        > 0.5
    )

    for col, name in [
        ("race_ethnicity_black",  "black_woman"),
        ("race_ethnicity_white",  "white_woman"),
        ("race_ethnicity_asian",  "asian_woman"),
        ("race_ethnicity_latina", "latina_woman"),
    ]:
        race = df.get(col, pd.Series(0, index=df.index)).fillna(0).astype(float).to_numpy() > 0.5
        labels[race & female] = name

    return labels


def create_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    """
    Convert a raw corpus DataFrame to (X, y, groups, fitted_scaler).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``is_amateur`` (label). Text columns ``title``/``tags``
        and numeric columns ``duration``/``rating`` are used if present.
    scaler : StandardScaler or None
        If None, a new scaler is fitted on this data (use for training split).
        If provided, this pre-fitted scaler is used to transform the data
        without re-fitting (use for validation / test splits to avoid leakage).

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Dense float feature matrix.
    y : np.ndarray of shape (n_samples,)
        Binary labels (0/1).
    groups : np.ndarray of shape (n_samples,)
        Intersectional group strings.
    fitted_scaler : StandardScaler or None
        The scaler fitted on this call's data (same object when ``scaler`` was
        provided; newly fitted instance otherwise).  Pass to subsequent calls
        on val/test splits to ensure identical normalisation.

    Raises
    ------
    KeyError
        If ``is_amateur`` is not in ``df``.

    Notes
    -----
    HashingVectorizer is stateless so it requires no fitting.  StandardScaler
    must be fitted on the training partition only and then reused on val/test.
    Callers that previously used the single-argument form and ignored the
    fourth return value will continue to work without changes.
    """
    if df.empty:
        raise ValueError("create_features received an empty DataFrame.")

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(f"create_features: required columns missing: {missing}")

    df = df.copy().reset_index(drop=True)

    # Labels
    y = df["is_amateur"].fillna(0).astype(int).to_numpy()

    # Groups
    groups = _build_group_labels(df)

    # Text features (stateless — no fit needed)
    text = (
        df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("tags", pd.Series("", index=df.index)).fillna("").astype(str)
    )
    vec = HashingVectorizer(n_features=2**14, alternate_sign=False, norm="l2")
    X_text = vec.transform(text)  # sparse

    # Numeric features — fit scaler on train only; reuse on val/test
    num_parts = []
    for col in _NUM_COLS:
        series = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)
        num_parts.append(series.to_numpy().reshape(-1, 1))

    fitted_scaler: StandardScaler | None = None
    if num_parts:
        X_num = np.hstack(num_parts)
        if scaler is None:
            fitted_scaler = StandardScaler()
            X_num = fitted_scaler.fit_transform(X_num)
        else:
            fitted_scaler = scaler
            X_num = fitted_scaler.transform(X_num)
        from scipy.sparse import csr_matrix
        X = hstack([X_text, csr_matrix(X_num)]).toarray()
    else:
        X = X_text.toarray()

    return X, y, groups, fitted_scaler
