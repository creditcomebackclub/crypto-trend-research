from __future__ import annotations

import numpy as np
import pandas as pd


def audit_bars(frame: pd.DataFrame, expected_frequency: str) -> dict[str, object]:
    if frame.empty:
        return {"rows": 0, "start": None, "end": None, "missing_intervals": 0, "zero_volume": 0, "return_outliers": 0}
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq=expected_frequency, tz=frame.index.tz)
    returns = frame["close"].pct_change()
    median = returns.median()
    mad = (returns - median).abs().median()
    outliers = ((returns - median).abs() > 20 * mad).sum() if mad > 0 else 0
    return {
        "rows": int(len(frame)),
        "start": frame.index.min().isoformat(),
        "end": frame.index.max().isoformat(),
        "missing_intervals": int(len(expected.difference(frame.index))),
        "zero_volume": int((frame["volume"] <= 0).sum()),
        "return_outliers": int(outliers),
    }


def cross_exchange_divergence(left: pd.Series, right: pd.Series) -> dict[str, float]:
    a, b = left.align(right, join="inner")
    divergence = (a / b - 1).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "observations": int(len(divergence)),
        "median_absolute_pct": float(divergence.abs().median() * 100) if len(divergence) else float("nan"),
        "p99_absolute_pct": float(divergence.abs().quantile(0.99) * 100) if len(divergence) else float("nan"),
    }

