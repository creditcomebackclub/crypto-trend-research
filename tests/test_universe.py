import numpy as np
import pandas as pd

from ctr.universe import point_in_time_universe


def test_future_volume_cannot_change_past_membership():
    idx = pd.date_range("2023-01-01", periods=180, freq="D", tz="UTC")
    close = pd.DataFrame({"A-USD": 10.0, "B-USD": 10.0, "C-USD": 10.0}, index=idx)
    volume = pd.DataFrame({"A-USD": 100.0, "B-USD": 90.0, "C-USD": 1.0}, index=idx)
    before = point_in_time_universe(close, volume, 2, min_history_days=30)
    changed = volume.copy()
    changed.loc[idx[-20]:, "C-USD"] = 1_000_000.0
    after = point_in_time_universe(close, changed, 2, min_history_days=30)
    pd.testing.assert_frame_equal(before.loc[: idx[-21]], after.loc[: idx[-21]])


def test_stablecoins_are_excluded():
    idx = pd.date_range("2023-01-01", periods=120, freq="D", tz="UTC")
    close = pd.DataFrame({"BTC-USD": 10.0, "USDC-USD": 1.0}, index=idx)
    volume = pd.DataFrame({"BTC-USD": 1.0, "USDC-USD": 1_000_000.0}, index=idx)
    eligible = point_in_time_universe(close, volume, 1, {"USDC"}, 30)
    assert not eligible["USDC-USD"].any()
    assert eligible["BTC-USD"].any()


def test_volume_lookback_uses_calendar_days_with_missing_bars():
    idx = pd.date_range("2023-01-01", periods=150, freq="D", tz="UTC").delete(slice(100, 105))
    close = pd.DataFrame({"A-USD": 1.0, "B-USD": 1.0}, index=idx)
    volume = pd.DataFrame({"A-USD": 100.0, "B-USD": 90.0}, index=idx)
    eligible = point_in_time_universe(close, volume, 1, min_history_days=30)
    assert eligible["A-USD"].any()
    assert not eligible["B-USD"].any()


def test_undersized_early_universe_does_not_include_future_listings():
    idx = pd.date_range("2020-01-01", periods=180, freq="D", tz="UTC")
    close = pd.DataFrame({"OLD-USD": 10.0, "FUTURE-USD": np.nan}, index=idx)
    volume = pd.DataFrame({"OLD-USD": 100.0, "FUTURE-USD": np.nan}, index=idx)
    close.loc[idx[150]:, "FUTURE-USD"] = 10.0
    volume.loc[idx[150]:, "FUTURE-USD"] = 1_000.0
    eligible = point_in_time_universe(close, volume, 20, min_history_days=30)
    assert eligible["OLD-USD"].any()
    assert not eligible.loc[:idx[149], "FUTURE-USD"].any()
