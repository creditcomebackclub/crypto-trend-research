from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import PerformanceMetrics, calculate_metrics
from .portfolio import PortfolioResult, simulate_portfolio
from .signals import trend_signal, trailing_realized_volatility, volatility_targeted_weights


@dataclass
class StrategyRun:
    name: str
    portfolio: PortfolioResult
    metrics: PerformanceMetrics


def run_trend(
    name: str,
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    volume: pd.DataFrame,
    lookbacks: list[int],
    volatility_target: float,
    bps_per_side: float,
    eligible: pd.DataFrame | None = None,
) -> StrategyRun:
    signal = trend_signal(close_prices, lookbacks)
    vol = trailing_realized_volatility(close_prices)
    weights = volatility_targeted_weights(signal, vol, volatility_target)
    if eligible is not None:
        weights = weights.where(eligible, 0.0)
        weights = weights.div(weights.abs().sum(axis=1).where(lambda x: x > 1, 1), axis=0)
    dv = close_prices * volume
    portfolio = simulate_portfolio(open_prices, weights, dv, bps_per_side=bps_per_side)
    metrics = calculate_metrics(portfolio.returns, portfolio.turnover, portfolio.exposure)
    return StrategyRun(name, portfolio, metrics)


def run_weights(
    name: str,
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    volume: pd.DataFrame,
    weights: pd.DataFrame,
    bps_per_side: float,
) -> StrategyRun:
    portfolio = simulate_portfolio(open_prices, weights, close_prices * volume, bps_per_side=bps_per_side)
    return StrategyRun(name, portfolio, calculate_metrics(portfolio.returns, portfolio.turnover, portfolio.exposure))

