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


def weekly_block_bootstrap_metric_cis(
    returns: pd.Series,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20261003,
) -> dict[str, tuple[float, float] | None]:
    """Bootstrap the three headline metrics without recomputing the full metric suite.

    The resampling unit and random seed are identical to ``weekly_block_bootstrap_ci``.
    Keeping the metric calculations in NumPy makes the 1,000-sample production report
    practical while preserving the pre-registered weekly-block design.
    """
    r = returns.dropna().astype(float)
    periods = r.index.tz_localize(None).to_period("W") if r.index.tz is not None else r.index.to_period("W")
    blocks = [np.asarray(group, dtype=float) for _, group in r.groupby(periods)]
    empty = {"cagr": None, "sharpe": None, "max_drawdown": None}
    if len(blocks) < 3:
        return empty
    spacing_days = max(float(r.index.to_series().diff().median().total_seconds() / 86400), 1 / 24)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in empty}
    for _ in range(samples):
        chosen = rng.integers(0, len(blocks), len(blocks))
        sample = np.concatenate([blocks[i] for i in chosen])
        if len(sample) < 3:
            continue
        curve = np.cumprod(1.0 + sample)
        years = max((len(sample) - 1) * spacing_days / 365.25, spacing_days / 365.25)
        values["cagr"].append(float(curve[-1] ** (1 / years) - 1) if curve[-1] > 0 else -1.0)
        std = float(np.std(sample, ddof=1))
        values["sharpe"].append(float(np.mean(sample) / std * np.sqrt(365.25 / spacing_days)) if std > 0 else np.nan)
        values["max_drawdown"].append(float(np.min(curve / np.maximum.accumulate(curve) - 1.0)))
    alpha = (1 - confidence) / 2
    result: dict[str, tuple[float, float] | None] = {}
    for name, observations in values.items():
        clean = np.asarray(observations, dtype=float)
        clean = clean[np.isfinite(clean)]
        result[name] = (
            (float(np.quantile(clean, alpha)), float(np.quantile(clean, 1 - alpha)))
            if len(clean) >= max(30, samples // 10)
            else None
        )
    return result


def paired_weekly_block_bootstrap_metric_cis(
    strategy: pd.Series,
    benchmark: pd.Series,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20261003,
) -> dict[str, tuple[float, float] | None]:
    """Paired CIs for differences in CAGR, Sharpe, and max drawdown."""
    frame = pd.concat({"strategy": strategy, "benchmark": benchmark}, axis=1).dropna()
    periods = frame.index.tz_localize(None).to_period("W") if frame.index.tz is not None else frame.index.to_period("W")
    blocks = [group.to_numpy(dtype=float) for _, group in frame.groupby(periods)]
    empty = {"cagr": None, "sharpe": None, "max_drawdown": None}
    if len(blocks) < 3:
        return empty
    spacing_days = max(float(frame.index.to_series().diff().median().total_seconds() / 86400), 1 / 24)
    annualization = 365.25 / spacing_days
    rng = np.random.default_rng(seed)
    differences = {name: [] for name in empty}

    def metrics(values: np.ndarray) -> tuple[float, float, float]:
        curve = np.cumprod(1.0 + values)
        years = max((len(values) - 1) * spacing_days / 365.25, spacing_days / 365.25)
        cagr = float(curve[-1] ** (1 / years) - 1) if curve[-1] > 0 else -1.0
        std = float(np.std(values, ddof=1))
        sharpe = float(np.mean(values) / std * np.sqrt(annualization)) if std > 0 else np.nan
        max_dd = float(np.min(curve / np.maximum.accumulate(curve) - 1.0))
        return cagr, sharpe, max_dd

    for _ in range(samples):
        chosen = rng.integers(0, len(blocks), len(blocks))
        sample = np.concatenate([blocks[i] for i in chosen])
        if len(sample) < 3:
            continue
        left = metrics(sample[:, 0])
        right = metrics(sample[:, 1])
        for name, value in zip(differences, np.subtract(left, right), strict=True):
            differences[name].append(float(value))
    alpha = (1 - confidence) / 2
    result: dict[str, tuple[float, float] | None] = {}
    for name, observations in differences.items():
        clean = np.asarray(observations, dtype=float)
        clean = clean[np.isfinite(clean)]
        result[name] = (
            (float(np.quantile(clean, alpha)), float(np.quantile(clean, 1 - alpha)))
            if len(clean) >= max(30, samples // 10)
            else None
        )
    return result


def deflated_sharpe_ratio(returns: pd.Series, trials: int, periods_per_year: int = 365) -> float:
    """Bailey/López de Prado-style probability that Sharpe exceeds selection bias."""
    r = returns.dropna().to_numpy()
    if len(r) < 3 or np.std(r, ddof=1) == 0:
        return float("nan")
    # Work in per-period Sharpe units throughout. Mixing an annualized observed
    # Sharpe with a per-period selection threshold makes the probability
    # spuriously converge to 100%.
    observed = float(np.mean(r) / np.std(r, ddof=1))
    count = max(trials, 2)
    euler_gamma = 0.5772156649015329
    expected_max_z = (
        (1 - euler_gamma) * norm.ppf(1 - 1 / count)
        + euler_gamma * norm.ppf(1 - 1 / (count * math.e))
    )
    selection_threshold = expected_max_z / math.sqrt(max(len(r) - 1, 1))
    sr_std = math.sqrt(
        max(1 - skew(r) * observed + ((kurtosis(r, fisher=False) - 1) / 4) * observed**2, 0)
        / max(len(r) - 1, 1)
    )
    return float(norm.cdf((observed - selection_threshold) / max(sr_std, 1e-12)))


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
