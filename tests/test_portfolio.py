import numpy as np
import pandas as pd

from ctr.portfolio import simulate_portfolio


def test_signal_executes_at_next_open_and_costs_turnover():
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    opens = pd.DataFrame({"BTC-USD": [100, 110, 121, 121]}, index=idx)
    decisions = pd.DataFrame({"BTC-USD": [1, 1, 0, 0]}, index=idx)
    result = simulate_portfolio(opens, decisions, bps_per_side=100)
    assert result.weights.iloc[0, 0] == 0
    assert result.weights.iloc[1, 0] == 1
    assert np.isclose(result.gross_returns.iloc[1], 0.10)
    assert np.isclose(result.costs.iloc[1], 0.01)
    assert np.isclose(result.turnover.sum(), 2.0)


def test_shifting_decision_changes_result():
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    opens = pd.DataFrame({"BTC-USD": [100, 120, 90, 110, 80, 130]}, index=idx)
    decisions = pd.DataFrame({"BTC-USD": [0, 1, 0, 1, 0, 1]}, index=idx)
    normal = simulate_portfolio(opens, decisions, bps_per_side=0).returns
    leaked_timing = simulate_portfolio(opens, decisions.shift(-1).fillna(0), bps_per_side=0).returns
    assert not np.isclose(normal.sum(), leaked_timing.sum())

