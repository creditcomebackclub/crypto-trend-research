import numpy as np
import pandas as pd

from ctr.stats import weekly_block_bootstrap_ci


def test_bootstrap_is_deterministic():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.001, 0.02, 200), index=pd.date_range("2024-01-01", periods=200, freq="D"))
    first = weekly_block_bootstrap_ci(returns, np.mean, samples=100, seed=42)
    second = weekly_block_bootstrap_ci(returns, np.mean, samples=100, seed=42)
    assert first == second


def test_bootstrap_requires_three_week_blocks():
    returns = pd.Series([0.01, -0.01], index=pd.date_range("2024-01-01", periods=2, freq="D"))
    assert weekly_block_bootstrap_ci(returns, np.mean, samples=100) is None

