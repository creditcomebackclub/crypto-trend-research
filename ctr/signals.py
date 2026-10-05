from __future__ import annotations

import numpy as np
import pandas as pd


def trend_votes(close: pd.DataFrame, lookbacks: list[int]) -> pd.DataFrame:
    """Mean sign-of-return vote using trailing data through the current close."""
    votes = [np.sign(close / close.shift(lb) - 1.0) for lb in lookbacks]
    return sum(votes) / len(votes)


def trend_signal(close: pd.DataFrame, lookbacks: list[int]) -> pd.DataFrame:
    return (trend_votes(close, lookbacks) > 0).astype(float)


def trailing_realized_volatility(close: pd.DataFrame, window: int = 30, periods_per_year: int = 365) -> pd.DataFrame:
    return close.pct_change(fill_method=None).rolling(window, min_periods=window).std() * np.sqrt(periods_per_year)


def volatility_targeted_weights(
    signal: pd.DataFrame,
    realized_volatility: pd.DataFrame,
    target: float,
    cap: float = 1.0,
) -> pd.DataFrame:
    scale = (target / realized_volatility.replace(0, np.nan)).clip(upper=cap)
    return (signal * scale).fillna(0.0)


def funding_zscore(funding: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    mean = funding.rolling(window, min_periods=max(20, window // 3)).mean()
    std = funding.rolling(window, min_periods=max(20, window // 3)).std().replace(0, np.nan)
    return (funding - mean) / std


def apply_funding_crowding_filter(weights: pd.DataFrame, funding_z: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    aligned = funding_z.reindex_like(weights).ffill()
    multiplier = pd.DataFrame(np.where(aligned > threshold, 0.5, 1.0), index=weights.index, columns=weights.columns)
    return weights * multiplier


def cross_sectional_momentum_weights(
    close: pd.DataFrame,
    eligible: pd.DataFrame,
    lookbacks: tuple[int, int] = (30, 90),
    skip: int = 7,
    top_fraction: float = 0.2,
) -> pd.DataFrame:
    scores = []
    for lb in lookbacks:
        ret = close.shift(skip) / close.shift(skip + lb) - 1
        scores.append(ret.rank(axis=1, pct=True))
    score = sum(scores) / len(scores)
    monthly_decision = ~pd.Series(close.index.tz_localize(None).to_period("M"), index=close.index).duplicated()
    selected = ((score >= 1 - top_fraction) & eligible).where(monthly_decision, False)
    selected = selected.where(monthly_decision).astype("boolean").ffill().fillna(False).astype(bool)
    weights = selected.astype(float)
    return weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
