from __future__ import annotations

import pandas as pd


def point_in_time_universe(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    size: int,
    excluded: set[str] | None = None,
    min_history_days: int = 90,
    volume_lookback: int = 30,
) -> pd.DataFrame:
    """Monthly top-N eligibility using only volume and history known then."""
    excluded = excluded or set()
    dollar_volume = (close * volume).rolling(volume_lookback, min_periods=volume_lookback).sum()
    history = close.notna().cumsum() >= min_history_days
    candidates = history.copy()
    for col in candidates.columns:
        if col.upper().split("-")[0] in excluded:
            candidates[col] = False
    month = pd.Series(close.index.tz_localize(None).to_period("M"), index=close.index)
    result = pd.DataFrame(False, index=close.index, columns=close.columns)
    for _, dates in month.groupby(month).groups.items():
        first = dates[0]
        prior_rows = close.index[close.index < first]
        if not len(prior_rows):
            continue
        decision = prior_rows[-1]
        valid = candidates.loc[decision]
        ranked = dollar_volume.loc[decision].where(valid).nlargest(size).index
        result.loc[dates, ranked] = True
    return result


def membership_changes(eligible: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prior: set[str] = set()
    periods = eligible.index.tz_localize(None).to_period("M")
    for period in periods.unique():
        first = eligible.index[periods == period][0]
        current = set(eligible.columns[eligible.loc[first].astype(bool)])
        rows.append({"month": str(period), "entered": sorted(current - prior), "left": sorted(prior - current)})
        prior = current
    return pd.DataFrame(rows)
