from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    cagr: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    longest_underwater_days: int
    worst_month: float
    worst_week: float
    hit_rate: float
    turnover: float
    average_holding_days: float
    exposure: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    curve = equity_curve(returns)
    return curve / curve.cummax() - 1.0


def _annualization(index: pd.Index) -> float:
    if len(index) < 2:
        return 365.25
    days = max((pd.Timestamp(index[-1]) - pd.Timestamp(index[0])).total_seconds() / 86400, 1)
    return 365.25 * (len(index) - 1) / days


def _longest_underwater(index: pd.Index, dd: pd.Series) -> int:
    longest = current = 0
    previous = None
    for ts, value in dd.items():
        if value < -1e-12:
            if previous is None:
                current = 1
            else:
                current += max((pd.Timestamp(ts) - pd.Timestamp(previous)).days, 1)
            longest = max(longest, current)
        else:
            current = 0
        previous = ts
    return int(longest)


def _average_holding(exposure: pd.Series) -> float:
    active = exposure.fillna(0).abs() > 1e-12
    groups = active.ne(active.shift()).cumsum()
    lengths = active.groupby(groups).sum()
    held = lengths[lengths > 0]
    return float(held.mean()) if len(held) else 0.0


def calculate_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    exposure: pd.Series | None = None,
) -> PerformanceMetrics:
    r = returns.dropna().astype(float)
    if r.empty:
        return PerformanceMetrics(*(float("nan"),) * 5, 0.0, 0, *(float("nan"),) * 3, 0.0, 0.0, 0.0)
    ann = _annualization(r.index)
    curve = equity_curve(r)
    years = max((pd.Timestamp(r.index[-1]) - pd.Timestamp(r.index[0])).total_seconds() / (365.25 * 86400), 1 / ann)
    cagr = float(curve.iloc[-1] ** (1 / years) - 1) if curve.iloc[-1] > 0 else -1.0
    vol = float(r.std(ddof=1) * math.sqrt(ann))
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(ann)) if r.std(ddof=1) > 0 else float("nan")
    downside = r[r < 0].std(ddof=1)
    sortino = float(r.mean() / downside * math.sqrt(ann)) if downside and downside > 0 else float("nan")
    dd = drawdown(r)
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    monthly = (1 + r).resample("ME").prod() - 1
    weekly = (1 + r).resample("W-SUN").prod() - 1
    exp = exposure.reindex(r.index).fillna(0) if exposure is not None else pd.Series(1.0, index=r.index)
    return PerformanceMetrics(
        cagr=cagr,
        annualized_volatility=vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        longest_underwater_days=_longest_underwater(r.index, dd),
        worst_month=float(monthly.min()),
        worst_week=float(weekly.min()),
        hit_rate=float((r > 0).mean()),
        turnover=float(turnover.reindex(r.index).fillna(0).sum()) if turnover is not None else 0.0,
        average_holding_days=_average_holding(exp),
        exposure=float(exp.abs().mean()),
    )

