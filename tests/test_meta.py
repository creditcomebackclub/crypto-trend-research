import numpy as np
import pandas as pd
import pytest

from ctr.meta import (
    Preprocessor,
    _event_multiplier,
    build_meta_features,
    build_triple_barrier_events,
    predict_logistic_artifact,
    oof_meta_predictions,
)
from ctr.config import load_config
from ctr.universe import point_in_time_universe
from ctr.validation import reject_future_leakage


def _market(periods: int = 320):
    index = pd.date_range("2022-01-01", periods=periods, freq="D", tz="UTC")
    phase = np.arange(periods)
    close = pd.DataFrame(
        {
            "BTC-USD": 100 * np.exp(phase * 0.001 + np.sin(phase / 13) * 0.02),
            "ETH-USD": 50 * np.exp(phase * 0.0008 + np.cos(phase / 17) * 0.03),
            "SOL-USD": 20 * np.exp(phase * 0.0006 + np.sin(phase / 11) * 0.04),
        },
        index=index,
    )
    volume = pd.DataFrame(1_000_000.0, index=index, columns=close.columns)
    funding = pd.DataFrame({"BTC-USD": np.sin(phase / 9) / 10_000}, index=index)
    return close, volume, funding


def test_triple_barrier_uses_next_open_and_normalizes_uniqueness():
    index = pd.date_range("2024-01-01", periods=70, freq="D", tz="UTC")
    close = pd.Series(100.0, index=index)
    close.iloc[36] = 102.0
    close.iloc[38:] = 100.0
    opens = pd.Series(100.0, index=index)
    volume = pd.Series(1_000_000.0, index=index)
    signal = pd.Series(0.0, index=index)
    signal.iloc[35] = 1.0
    signal.iloc[37] = 1.0
    events = build_triple_barrier_events(opens, close, close * volume, signal, bps_per_side=0)
    assert len(events) == 2
    assert events.iloc[0]["entry_time"] == index[36]
    assert events.iloc[0]["label"] == 1
    assert np.isclose(events["uniqueness"].mean(), 1.0)
    assert (events["uniqueness"] > 0).all()


def test_meta_features_are_unchanged_when_future_rows_are_removed():
    close, volume, funding = _market()
    excluded = set()
    top20 = point_in_time_universe(close, volume, 20, excluded, 90)
    full = build_meta_features(close, volume, top20, funding, [20, 60, 120, 250])
    cutoff = 280
    close_cut = close.iloc[:cutoff]
    volume_cut = volume.iloc[:cutoff]
    funding_cut = funding.iloc[:cutoff]
    top20_cut = point_in_time_universe(close_cut, volume_cut, 20, excluded, 90)
    truncated = build_meta_features(close_cut, volume_cut, top20_cut, funding_cut, [20, 60, 120, 250])
    pd.testing.assert_frame_equal(full.loc[truncated.index], truncated)


def test_deliberately_future_shifted_meta_feature_fails_guard():
    close, _, _ = _market(40)
    future_close = close[["BTC-USD"]].shift(-1)
    with pytest.raises(ValueError, match="future"):
        reject_future_leakage(future_close, close[["BTC-USD"]])


def test_episode_multiplier_is_fixed_until_rule_exit():
    index = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    predictions = pd.DataFrame(
        {
            "probability": [0.75],
            "threshold": [0.60],
            "episode_end_time": [index[5]],
        },
        index=[index[2]],
    )
    sized = _event_multiplier(predictions, index, "sized")
    threshold = _event_multiplier(predictions, index, "threshold")
    assert (sized.loc[index[2] : index[4]] == 0.5).all()
    assert (threshold.loc[index[2] : index[4]] == 1.0).all()
    assert sized.loc[index[5]] == 0.0


def test_serialized_logistic_artifact_reproduces_probability():
    frame = pd.DataFrame({"a": [-1.0, 0.0, 1.0], "b": [1.0, 0.0, -1.0]})
    preprocessor = Preprocessor.fit(frame)
    artifact = {
        "preprocessor": preprocessor.to_dict(),
        "coefficient": [0.4, -0.2],
        "intercept": 0.1,
        "calibration": {"coefficient": 1.1, "intercept": -0.05},
    }
    probability = predict_logistic_artifact(frame, artifact)
    assert probability.between(0, 1).all()
    assert probability.is_monotonic_increasing


def test_future_labels_cannot_change_earlier_outer_fold_predictions():
    index = pd.date_range("2020-01-01", periods=640, freq="D", tz="UTC")
    event_index = index[100:620:10]
    phase = np.arange(len(event_index))
    features = pd.DataFrame(
        {"signal": np.sin(phase / 3), "volatility": 0.02 + (phase % 7) / 1_000},
        index=event_index,
    )
    events = pd.DataFrame(
        {
            "entry_time": event_index + pd.Timedelta(days=1),
            "label_end_time": event_index + pd.Timedelta(days=5),
            "episode_end_time": event_index + pd.Timedelta(days=8),
            "label": (phase % 3 == 0).astype(int),
            "realized_net_return": np.where(phase % 3 == 0, 0.04, -0.02),
            "barrier_width": 0.04,
            "uniqueness": 1.0,
        },
        index=event_index,
    )
    config = load_config()
    config["validation"]["minimum_training_days"] = 180
    config["validation"]["test_block_months"] = 3
    config["meta"]["ci_mode"] = True
    first = oof_meta_predictions("logistic", features, events, index, config)
    changed = events.copy()
    future = changed.index >= index[430]
    changed.loc[future, "label"] = 1 - changed.loc[future, "label"]
    second = oof_meta_predictions("logistic", features, changed, index, config)
    earlier = first.index < index[430]
    pd.testing.assert_series_equal(first.loc[earlier, "probability"], second.loc[earlier, "probability"])
