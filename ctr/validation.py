from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex


def expanding_walk_forward(
    index: pd.DatetimeIndex,
    minimum_training_days: int = 730,
    test_months: int = 12,
    purge_days: int = 7,
) -> list[WalkForwardFold]:
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) < 2:
        return []
    first_test = idx[0] + pd.Timedelta(days=minimum_training_days)
    folds: list[WalkForwardFold] = []
    start = first_test
    while start < idx[-1]:
        end = start + pd.DateOffset(months=test_months)
        train_end = start - pd.Timedelta(days=purge_days)
        train = idx[idx < train_end]
        test = idx[(idx >= start) & (idx < end)]
        if len(train) and len(test):
            folds.append(WalkForwardFold(train, test))
        start = end
    return folds


def select_parameter_on_training(
    folds: list[WalkForwardFold],
    candidates: list[float],
    evaluator,
) -> list[tuple[WalkForwardFold, float]]:
    selections = []
    for fold in folds:
        scores = {candidate: evaluator(fold.train, candidate) for candidate in candidates}
        finite = {k: v for k, v in scores.items() if np.isfinite(v)}
        chosen = max(finite, key=finite.get) if finite else candidates[0]
        selections.append((fold, chosen))
    return selections


def assert_feature_is_point_in_time(feature: pd.DataFrame, source: pd.DataFrame, builder) -> None:
    """Fail if truncating future source values changes any earlier feature value."""
    if len(source) < 6:
        return
    cutoff = len(source) // 2
    truncated = source.iloc[:cutoff].copy()
    rebuilt = builder(truncated)
    expected = feature.loc[rebuilt.index, rebuilt.columns]
    pd.testing.assert_frame_equal(expected, rebuilt)


def reject_future_leakage(feature: pd.DataFrame, source: pd.DataFrame) -> None:
    """Conservative guard: a feature may not equal the next observed source value."""
    future = source.shift(-1).reindex_like(feature)
    overlap = feature.notna() & future.notna()
    if overlap.to_numpy().any() and np.allclose(feature.where(overlap).stack(), future.where(overlap).stack()):
        raise ValueError("feature exactly matches future data")

