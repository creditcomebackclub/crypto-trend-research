import numpy as np
import pandas as pd

from ctr.signals import apply_funding_crowding_filter, funding_zscore


def test_funding_filter_uses_trailing_z_score_and_reduces_extreme_exposure():
    index = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    funding = pd.DataFrame({"BTC-USD": np.r_[np.zeros(99), 0.05]}, index=index)
    weights = pd.DataFrame({"BTC-USD": 1.0}, index=index)
    z = funding_zscore(funding, window=90)
    filtered = apply_funding_crowding_filter(weights, z, threshold=2.0)
    assert filtered.iloc[-2, 0] == 1.0
    assert filtered.iloc[-1, 0] == 0.5
