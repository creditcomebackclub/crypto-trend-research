import numpy as np
import pandas as pd
import pytest

from ctr.signals import trend_signal
from ctr.validation import assert_feature_is_point_in_time, expanding_walk_forward, reject_future_leakage


def test_walk_forward_is_purged_and_ordered():
    index = pd.date_range("2018-01-01", "2024-12-31", freq="D", tz="UTC")
    folds = expanding_walk_forward(index, minimum_training_days=730, test_months=6, purge_days=7)
    assert folds
    for fold in folds:
        assert fold.train.max() < fold.test.min() - pd.Timedelta(days=6)
        assert set(fold.train).isdisjoint(fold.test)


def test_trend_feature_uses_no_future_data():
    index = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC")
    close = pd.DataFrame({"BTC-USD": np.linspace(100, 200, 400)}, index=index)
    feature = trend_signal(close, [20, 60, 120, 250])
    assert_feature_is_point_in_time(feature, close, lambda x: trend_signal(x, [20, 60, 120, 250]))


def test_deliberately_leaked_feature_fails_guard():
    index = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    source = pd.DataFrame({"BTC-USD": np.arange(10.0)}, index=index)
    with pytest.raises(ValueError, match="future"):
        reject_future_leakage(source.shift(-1), source)

