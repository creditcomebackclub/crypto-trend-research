from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


def weekly_block_bootstrap_ci(
    returns: pd.Series,
    statistic: Callable[[pd.Series], float],
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20261003,
) -> tuple[float, float] | None:
    r = returns.dropna()
    periods = r.index.tz_localize(None).to_period("W") if r.index.tz is not None else r.index.to_period("W")
    blocks = [group for _, group in r.groupby(periods)]
    if len(blocks) < 3:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        chosen = rng.integers(0, len(blocks), len(blocks))
        sample = pd.concat([blocks[i] for i in chosen], ignore_index=True)
        spacing = r.index.to_series().diff().median()
        sample.index = pd.date_range("2000-01-01", periods=len(sample), freq=spacing)
        value = statistic(sample)
        if np.isfinite(value):
            values.append(value)
    if len(values) < max(30, samples // 10):
        return None
    alpha = (1 - confidence) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def paired_block_bootstrap_ci(
    strategy: pd.Series,
    benchmark: pd.Series,
    statistic: Callable[[pd.Series], float] = np.mean,
    **kwargs,
) -> tuple[float, float] | None:
    a, b = strategy.align(benchmark, join="inner")
    return weekly_block_bootstrap_ci(a - b, statistic, **kwargs)


def deflated_sharpe_ratio(returns: pd.Series, trials: int, periods_per_year: int = 365) -> float:
    """Bailey/López de Prado-style probability that Sharpe exceeds selection bias."""
    r = returns.dropna().to_numpy()
    if len(r) < 3 or np.std(r, ddof=1) == 0:
        return float("nan")
    observed = np.mean(r) / np.std(r, ddof=1) * math.sqrt(periods_per_year)
    expected_max = norm.ppf(1 - 1 / max(trials, 2))
    sr_std = math.sqrt((1 - skew(r) * observed + ((kurtosis(r, fisher=False) - 1) / 4) * observed**2) / max(len(r) - 1, 1))
    return float(norm.cdf((observed - expected_max / math.sqrt(len(r))) / max(sr_std, 1e-12)))


def probability_of_backtest_overfitting(variant_returns: pd.DataFrame, splits: int = 8) -> float:
    """Deterministic CSCV approximation: fraction where IS winner ranks below median OOS."""
    clean = variant_returns.dropna(how="all")
    if clean.shape[1] < 2 or len(clean) < splits:
        return float("nan")
    blocks = np.array_split(np.arange(len(clean)), splits)
    failures = total = 0
    for mask in range(1, 2**splits - 1):
        if mask.bit_count() != splits // 2:
            continue
        ins = np.concatenate([blocks[i] for i in range(splits) if mask & (1 << i)])
        out = np.concatenate([blocks[i] for i in range(splits) if not mask & (1 << i)])
        is_score = clean.iloc[ins].mean() / clean.iloc[ins].std().replace(0, np.nan)
        winner = is_score.idxmax()
        oos_score = clean.iloc[out].mean() / clean.iloc[out].std().replace(0, np.nan)
        rank_pct = oos_score.rank(pct=True).get(winner, np.nan)
        if np.isfinite(rank_pct):
            failures += rank_pct <= 0.5
            total += 1
    return failures / total if total else float("nan")
