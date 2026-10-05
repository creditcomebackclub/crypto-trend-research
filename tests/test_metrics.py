import numpy as np
import pandas as pd

from ctr.metrics import calculate_metrics, drawdown, equity_curve


def test_equity_and_drawdown_match_hand_calculation():
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    returns = pd.Series([0.10, -0.10, 0.05], index=index)
    expected = pd.Series([1.1, 0.99, 1.0395], index=index)
    pd.testing.assert_series_equal(equity_curve(returns), expected)
    assert np.isclose(drawdown(returns).iloc[1], -0.10)
    metrics = calculate_metrics(returns)
    assert np.isclose(metrics.max_drawdown, -0.10)
    assert np.isclose(metrics.hit_rate, 2 / 3)

