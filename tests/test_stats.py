import numpy as np
import pandas as pd

from ctr.stats import deflated_sharpe_ratio, paired_weekly_block_bootstrap_metric_cis, weekly_block_bootstrap_ci, weekly_block_bootstrap_metric_cis


def test_bootstrap_is_deterministic():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.001, 0.02, 200), index=pd.date_range("2024-01-01", periods=200, freq="D"))
    first = weekly_block_bootstrap_ci(returns, np.mean, samples=100, seed=42)
    second = weekly_block_bootstrap_ci(returns, np.mean, samples=100, seed=42)
    assert first == second


def test_bootstrap_requires_three_week_blocks():
    returns = pd.Series([0.01, -0.01], index=pd.date_range("2024-01-01", periods=2, freq="D"))
    assert weekly_block_bootstrap_ci(returns, np.mean, samples=100) is None


def test_fast_metric_bootstrap_is_deterministic_and_non_degenerate():
    rng = np.random.default_rng(11)
    returns = pd.Series(rng.normal(0.0008, 0.015, 365), index=pd.date_range("2024-01-01", periods=365, freq="D"))
    first = weekly_block_bootstrap_metric_cis(returns, samples=100, seed=9)
    second = weekly_block_bootstrap_metric_cis(returns, samples=100, seed=9)
    assert first == second
    assert all(interval is not None and interval[0] < interval[1] for interval in first.values())


def test_fast_metric_bootstrap_requires_three_week_blocks():
    returns = pd.Series([0.01, -0.01], index=pd.date_range("2024-01-01", periods=2, freq="D"))
    assert all(value is None for value in weekly_block_bootstrap_metric_cis(returns, samples=100).values())


def test_paired_metric_bootstrap_preserves_pairing():
    rng = np.random.default_rng(21)
    index = pd.date_range("2023-01-01", periods=400, freq="D")
    benchmark = pd.Series(rng.normal(0.0005, 0.01, len(index)), index=index)
    strategy = benchmark + 0.0002
    cis = paired_weekly_block_bootstrap_metric_cis(strategy, benchmark, samples=100, seed=3)
    assert cis["sharpe"] is not None
    assert cis["cagr"] is not None and cis["cagr"][0] > 0


def test_deflated_sharpe_penalizes_more_trials():
    rng = np.random.default_rng(31)
    returns = pd.Series(rng.normal(0.0007, 0.012, 1200))
    few = deflated_sharpe_ratio(returns, trials=2)
    many = deflated_sharpe_ratio(returns, trials=100)
    assert 0 <= many <= few <= 1
