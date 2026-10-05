from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    exposure: pd.Series


def simulate_portfolio(
    open_prices: pd.DataFrame,
    decision_weights: pd.DataFrame,
    dollar_volume: pd.DataFrame | None = None,
    bps_per_side: float = 25.0,
    notional_usd: float = 10_000.0,
    impact_coefficient: float = 10.0,
) -> PortfolioResult:
    """Execute close-time decisions at the next open and apply turnover costs."""
    prices, decisions = open_prices.align(decision_weights, join="inner", axis=0)
    decisions = decisions.reindex(columns=prices.columns).fillna(0.0)
    decisions = decisions.clip(lower=-1.0, upper=1.0)
    gross_exposure = decisions.abs().sum(axis=1)
    decisions = decisions.div(gross_exposure.where(gross_exposure > 1.0, 1.0), axis=0)
    # A decision formed at close t executes at open t+1. It can therefore earn
    # only the open(t+1)-to-open(t+2) return. Using pct_change at t+1 would
    # incorrectly award the strategy price movement that occurred before entry.
    executed = decisions.shift(1).fillna(0.0)
    asset_returns = (prices.shift(-1) / prices - 1.0).replace([np.inf, -np.inf], np.nan)
    active_missing = (executed.abs() > 1e-12) & asset_returns.isna()
    gross = (executed * asset_returns).fillna(0.0).sum(axis=1)
    gross.loc[active_missing.any(axis=1)] = np.nan
    trades = executed.diff().abs().fillna(executed.abs())
    turnover = trades.sum(axis=1)
    linear_cost = turnover * bps_per_side / 10_000.0
    if dollar_volume is None:
        impact = pd.Series(0.0, index=prices.index)
    else:
        dv = dollar_volume.reindex_like(prices).shift(1).replace(0, np.nan)
        trade_usd = trades * notional_usd
        impact_bps = impact_coefficient * np.sqrt((trade_usd / dv).clip(lower=0)).fillna(0)
        impact = (trades * impact_bps / 10_000.0).sum(axis=1)
    costs = linear_cost + impact
    costs.loc[gross.isna()] = np.nan
    return PortfolioResult(
        returns=(gross - costs).rename("net_return"),
        gross_returns=gross.rename("gross_return"),
        weights=executed,
        turnover=turnover.rename("turnover"),
        costs=costs.rename("cost"),
        exposure=executed.abs().sum(axis=1).rename("exposure"),
    )


def equal_weight(eligible: pd.DataFrame) -> pd.DataFrame:
    weights = eligible.astype(float)
    return weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
