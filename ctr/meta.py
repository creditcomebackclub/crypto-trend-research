from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .backtest import StrategyRun
from .metrics import calculate_metrics
from .portfolio import simulate_portfolio
from .signals import funding_zscore, trend_signal, trend_votes, trailing_realized_volatility, volatility_targeted_weights
from .stats import deflated_sharpe_ratio, paired_weekly_block_bootstrap_metric_cis, probability_of_backtest_overfitting
from .validation import expanding_walk_forward, reject_future_leakage


FEATURE_GROUPS: dict[str, list[str]] = {
    "trend state": ["vote_strength", "vote_agreement", "distance_ma200", "trend_acceleration"],
    "volatility": ["vol_10", "vol_30", "vol_90", "vol_of_vol", "vol_percentile"],
    "positioning": ["funding_level", "funding_z90", "funding_trend", "funding_missing"],
    "market breadth": ["breadth_above_ma50", "btc_volume_share"],
    "calendar": [f"dow_{day}" for day in range(7)],
}


@dataclass(frozen=True)
class Preprocessor:
    columns: list[str]
    lower: dict[str, float]
    upper: dict[str, float]
    median: dict[str, float]
    mean: dict[str, float]
    scale: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "Preprocessor":
        clean = frame.astype(float).replace([np.inf, -np.inf], np.nan)
        lower = clean.quantile(0.01).fillna(0.0)
        upper = clean.quantile(0.99).fillna(0.0)
        clipped = clean.clip(lower=lower, upper=upper, axis=1)
        median = clipped.median().fillna(0.0)
        filled = clipped.fillna(median)
        mean = filled.mean().fillna(0.0)
        scale = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
        return cls(
            list(clean.columns),
            lower.to_dict(),
            upper.to_dict(),
            median.to_dict(),
            mean.to_dict(),
            scale.to_dict(),
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        clean = frame.reindex(columns=self.columns).astype(float).replace([np.inf, -np.inf], np.nan)
        lower = pd.Series(self.lower)
        upper = pd.Series(self.upper)
        median = pd.Series(self.median)
        mean = pd.Series(self.mean)
        scale = pd.Series(self.scale)
        filled = clean.clip(lower=lower, upper=upper, axis=1).fillna(median)
        return ((filled - mean) / scale).to_numpy(dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Preprocessor":
        return cls(**value)


@dataclass
class MetaStudyResult:
    runs: dict[str, StrategyRun]
    report_section: str
    events: pd.DataFrame
    classification: dict[str, dict[str, Any]]
    paired: dict[str, dict[str, tuple[float, float] | None]]
    ablations: list[dict[str, Any]]
    permutation: dict[str, dict[str, float]]
    supported: bool
    artifact: dict[str, Any] | None


def _btc_column(frame: pd.DataFrame) -> str:
    column = next((col for col in frame if col.upper().split("-")[0] in {"BTC", "XBT"}), None)
    if column is None:
        raise ValueError("BTC column is required")
    return column


def _expanding_percentile(series: pd.Series, minimum: int = 30) -> pd.Series:
    def percentile(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.mean(finite <= finite[-1])) if len(finite) else np.nan

    return series.expanding(min_periods=minimum).apply(percentile, raw=True)


def build_meta_features(
    close_prices: pd.DataFrame,
    volume: pd.DataFrame,
    top20: pd.DataFrame,
    funding: pd.DataFrame | None,
    lookbacks: list[int],
) -> pd.DataFrame:
    """Build features using only values available through each decision close."""
    btc = _btc_column(close_prices)
    close = close_prices[btc].astype(float)
    returns = close.pct_change(fill_method=None)
    votes = trend_votes(close_prices[[btc]], lookbacks)[btc]
    individual = pd.concat(
        [np.sign(close / close.shift(lookback) - 1.0).rename(str(lookback)) for lookback in lookbacks],
        axis=1,
    )
    vol30 = returns.rolling(30, min_periods=30).std()
    features = pd.DataFrame(index=close_prices.index)
    features["vote_strength"] = votes
    features["vote_agreement"] = (individual > 0).mean(axis=1)
    features["distance_ma200"] = close / close.rolling(200, min_periods=200).mean() - 1.0
    features["trend_acceleration"] = close / close.shift(20) - close.shift(20) / close.shift(40)
    features["vol_10"] = returns.rolling(10, min_periods=10).std()
    features["vol_30"] = vol30
    features["vol_90"] = returns.rolling(90, min_periods=90).std()
    features["vol_of_vol"] = vol30.rolling(30, min_periods=30).std()
    features["vol_percentile"] = _expanding_percentile(vol30)

    if funding is not None and btc in funding.columns:
        aligned = funding[btc].reindex(close_prices.index).ffill()
    else:
        aligned = pd.Series(np.nan, index=close_prices.index, dtype=float)
    features["funding_missing"] = aligned.isna().astype(float)
    features["funding_level"] = aligned
    features["funding_z90"] = funding_zscore(aligned.to_frame(btc), 90)[btc]
    features["funding_trend"] = aligned.rolling(7, min_periods=3).mean() - aligned.rolling(30, min_periods=10).mean()

    above = close_prices > close_prices.rolling(50, min_periods=50).mean()
    eligible_count = top20.sum(axis=1).replace(0, np.nan)
    features["breadth_above_ma50"] = (above & top20).sum(axis=1) / eligible_count
    dollar_volume = close_prices * volume
    universe_volume = dollar_volume.where(top20).sum(axis=1, min_count=1)
    features["btc_volume_share"] = dollar_volume[btc] / universe_volume.replace(0, np.nan)
    for day in range(7):
        features[f"dow_{day}"] = (features.index.dayofweek == day).astype(float)
    return features


def build_triple_barrier_events(
    open_price: pd.Series,
    close: pd.Series,
    dollar_volume: pd.Series,
    signal: pd.Series,
    *,
    horizon: int = 20,
    volatility_multiple: float = 2.0,
    barrier_floor: float = 0.01,
    barrier_cap: float = 0.25,
    bps_per_side: float = 25.0,
    impact_coefficient: float = 10.0,
    notional_usd: float = 10_000.0,
) -> pd.DataFrame:
    """Create close-monitored triple-barrier labels for cash-to-long H1 entries."""
    index = close.index.intersection(open_price.index).sort_values()
    open_price = open_price.reindex(index)
    close = close.reindex(index)
    signal = signal.reindex(index).fillna(0.0)
    dollar_volume = dollar_volume.reindex(index)
    daily_vol = close.pct_change(fill_method=None).rolling(30, min_periods=30).std()
    starts = signal.gt(0) & signal.shift(1, fill_value=0).le(0)
    rows: list[dict[str, Any]] = []
    for signal_time in index[starts]:
        signal_position = index.get_loc(signal_time)
        if signal_position + 1 >= len(index):
            continue
        entry_position = signal_position + 1
        entry_time = index[entry_position]
        entry = float(open_price.iloc[entry_position])
        sigma = float(daily_vol.loc[signal_time])
        if not np.isfinite(entry) or not np.isfinite(sigma):
            continue
        width = float(np.clip(volatility_multiple * sigma, barrier_floor, barrier_cap))
        future = close.iloc[entry_position : min(entry_position + horizon, len(index))].dropna()
        if future.empty:
            continue
        upper, lower = entry * (1 + width), entry * (1 - width)
        profit_hits = future.index[future >= upper]
        stop_hits = future.index[future <= lower]
        profit_time = profit_hits[0] if len(profit_hits) else None
        stop_time = stop_hits[0] if len(stop_hits) else None
        if profit_time is not None and (stop_time is None or profit_time < stop_time):
            label, exit_time = 1, profit_time
        elif stop_time is not None:
            label, exit_time = 0, stop_time
        else:
            label, exit_time = 0, future.index[-1]
        exit_close = float(close.loc[exit_time])
        entry_dv = float(dollar_volume.loc[signal_time])
        exit_dv = float(dollar_volume.loc[exit_time])
        entry_impact = impact_coefficient * math.sqrt(notional_usd / entry_dv) if entry_dv > 0 else 0.0
        exit_impact = impact_coefficient * math.sqrt(notional_usd / exit_dv) if exit_dv > 0 else 0.0
        cost = 2 * bps_per_side / 10_000.0 + (entry_impact + exit_impact) / 10_000.0
        zero_after = index[(index > signal_time) & signal.le(0)]
        episode_end = zero_after[0] if len(zero_after) else index[-1] + pd.Timedelta(days=1)
        rows.append(
            {
                "signal_time": signal_time,
                "entry_time": entry_time,
                "label_end_time": exit_time,
                "episode_end_time": episode_end,
                "label": label,
                "realized_net_return": exit_close / entry - 1.0 - cost,
                "barrier_width": width,
            }
        )
    events = pd.DataFrame(rows).set_index("signal_time") if rows else pd.DataFrame()
    if events.empty:
        return events
    concurrency = pd.Series(0.0, index=index)
    for row in events.itertuples():
        concurrency.loc[(concurrency.index >= row.entry_time) & (concurrency.index <= row.label_end_time)] += 1.0
    uniqueness = []
    for row in events.itertuples():
        active = concurrency.loc[(concurrency.index >= row.entry_time) & (concurrency.index <= row.label_end_time)]
        uniqueness.append(float((1.0 / active.replace(0, np.nan)).mean()))
    events["uniqueness"] = uniqueness
    events["uniqueness"] /= events["uniqueness"].mean()
    return events


def assert_meta_features_point_in_time(features: pd.DataFrame, close_prices: pd.DataFrame) -> None:
    reject_future_leakage(features[["distance_ma200"]], close_prices[[_btc_column(close_prices)]])
    future_return = close_prices[[_btc_column(close_prices)]].shift(-1) / close_prices[[_btc_column(close_prices)]] - 1
    reject_future_leakage(features[["trend_acceleration"]].rename(columns={"trend_acceleration": future_return.columns[0]}), future_return)


def _inner_splits(events: pd.DataFrame, embargo_days: int = 20) -> list[tuple[pd.Index, pd.Index]]:
    ordered = events.sort_index()
    if len(ordered) < 16:
        return []
    boundaries = np.linspace(max(8, len(ordered) // 2), len(ordered), 4, dtype=int)
    splits = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        validation = ordered.iloc[left:right]
        if validation.empty:
            continue
        cutoff = validation.index.min() - pd.Timedelta(days=embargo_days)
        training = ordered[(ordered.index < cutoff) & (ordered["label_end_time"] < cutoff)]
        if len(training) >= 8 and training["label"].nunique() == 2 and len(validation) >= 2:
            splits.append((training.index, validation.index))
    return splits


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def _weighted_base_rate(labels: pd.Series, weights: pd.Series) -> float:
    return float(np.average(labels.astype(float), weights=weights.astype(float)))


def _fit_raw_model(kind: str, params: dict[str, Any], x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    if kind == "logistic":
        model = LogisticRegression(C=float(params["C"]), penalty="l2", solver="lbfgs", max_iter=2000, random_state=20261003)
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            max_iter=150,
            early_stopping=False,
            random_state=20261003,
        )
    model.fit(x, y, sample_weight=weights)
    return model


def _candidate_grid(kind: str, cfg: dict) -> list[dict[str, Any]]:
    if kind == "logistic":
        values = cfg["meta"]["logistic_c"]
        if cfg["meta"].get("ci_mode"):
            values = [0.1, 1.0]
        return [{"C": float(value)} for value in values]
    if cfg["meta"].get("ci_mode"):
        return [{"learning_rate": 0.05, "max_leaf_nodes": 3, "max_depth": 2, "min_samples_leaf": 20, "l2_regularization": 10.0}]
    return [
        {
            "learning_rate": learning_rate,
            "max_leaf_nodes": leaves,
            "max_depth": depth,
            "min_samples_leaf": minimum,
            "l2_regularization": regularization,
        }
        for learning_rate in (0.03, 0.05)
        for leaves in (3, 7)
        for depth in (2, 3)
        for minimum in (20, 40)
        for regularization in (1.0, 10.0)
    ]


def _tie_key(kind: str, params: dict[str, Any]) -> tuple:
    if kind == "logistic":
        return (float(params["C"]),)
    return (
        int(params["max_leaf_nodes"]),
        int(params["max_depth"]),
        -int(params["min_samples_leaf"]),
        -float(params["l2_regularization"]),
        float(params["learning_rate"]),
    )


def _inner_oof(
    kind: str,
    params: dict[str, Any],
    x: pd.DataFrame,
    events: pd.DataFrame,
    splits: list[tuple[pd.Index, pd.Index]],
) -> pd.Series:
    predictions = pd.Series(np.nan, index=events.index, dtype=float)
    for training, validation in splits:
        y_train = events.loc[training, "fit_label"].astype(int)
        if y_train.nunique() < 2:
            continue
        preprocessor = Preprocessor.fit(x.loc[training])
        model = _fit_raw_model(
            kind,
            params,
            preprocessor.transform(x.loc[training]),
            y_train.to_numpy(),
            events.loc[training, "uniqueness"].to_numpy(dtype=float),
        )
        predictions.loc[validation] = model.predict_proba(preprocessor.transform(x.loc[validation]))[:, 1]
    return predictions


def _select_model(
    kind: str,
    x: pd.DataFrame,
    events: pd.DataFrame,
    cfg: dict,
) -> tuple[dict[str, Any], pd.Series]:
    splits = _inner_splits(events, cfg["meta"]["embargo_days"])
    candidates = _candidate_grid(kind, cfg)
    if not splits:
        return candidates[0], pd.Series(np.nan, index=events.index, dtype=float)
    scores: list[tuple[float, tuple, dict[str, Any], pd.Series]] = []
    for params in candidates:
        prediction = _inner_oof(kind, params, x, events, splits)
        valid = prediction.notna()
        if not valid.any():
            score = float("inf")
        else:
            score = log_loss(
                events.loc[valid, "fit_label"],
                prediction.loc[valid],
                sample_weight=events.loc[valid, "uniqueness"],
                labels=[0, 1],
            )
        scores.append((float(score), _tie_key(kind, params), params, prediction))
    _, _, selected, prediction = min(scores, key=lambda item: (item[0], item[1]))
    return selected, prediction


def _fit_calibrator(prediction: pd.Series, events: pd.DataFrame):
    valid = prediction.notna()
    labels = events.loc[valid, "fit_label"].astype(int)
    if valid.sum() < 8 or labels.nunique() < 2:
        return None
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=20261003)
    calibrator.fit(
        _logit(prediction.loc[valid].to_numpy()),
        labels.to_numpy(),
        sample_weight=events.loc[valid, "uniqueness"].to_numpy(dtype=float),
    )
    return calibrator


def _calibrate(calibrator, probability: np.ndarray) -> np.ndarray:
    return probability if calibrator is None else calibrator.predict_proba(_logit(probability))[:, 1]


def _choose_threshold(prediction: pd.Series, events: pd.DataFrame, thresholds: list[float]) -> float:
    valid = prediction.notna()
    best: tuple[float, float] | None = None
    for threshold in thresholds:
        accepted = events.loc[valid & prediction.gt(threshold), ["label_end_time", "fit_return"]]
        if len(accepted) < 10:
            continue
        series = accepted.groupby("label_end_time")["fit_return"].sum().sort_index()
        std = float(series.std(ddof=1))
        score = float(series.mean() / std) if std > 0 else float("-inf")
        candidate = (score, float(threshold))
        if best is None or candidate > best:
            best = candidate
    return best[1] if best is not None else 0.50


def _permute_training(events: pd.DataFrame, seed: int) -> pd.DataFrame:
    shuffled = events.copy()
    rng = np.random.default_rng(seed)
    quarter = shuffled.index.tz_localize(None).to_period("Q")
    for _, positions in pd.Series(np.arange(len(shuffled)), index=shuffled.index).groupby(quarter):
        values = positions.to_numpy()
        permutation = rng.permutation(values)
        shuffled.iloc[values, shuffled.columns.get_loc("fit_label")] = events.iloc[permutation]["label"].to_numpy()
        shuffled.iloc[values, shuffled.columns.get_loc("fit_return")] = events.iloc[permutation]["realized_net_return"].to_numpy()
    return shuffled


def oof_meta_predictions(
    kind: str,
    features: pd.DataFrame,
    events: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    cfg: dict,
    *,
    permutation_seed: int | None = None,
) -> pd.DataFrame:
    output = events.copy()
    output["probability"] = np.nan
    output["base_rate"] = np.nan
    output["threshold"] = np.nan
    output["fold"] = np.nan
    folds = expanding_walk_forward(
        daily_index,
        cfg["validation"]["minimum_training_days"],
        cfg["validation"]["test_block_months"],
        cfg["meta"]["purge_days"],
    )
    for fold_number, fold in enumerate(folds):
        test_start, test_end = fold.test.min(), fold.test.max()
        cutoff = test_start - pd.Timedelta(days=cfg["meta"]["embargo_days"])
        training = output[(output.index < cutoff) & (output["label_end_time"] < cutoff)].copy()
        testing = output[(output.index >= test_start) & (output.index <= test_end)].copy()
        if training.empty or testing.empty or training["label"].nunique() < 2:
            continue
        training["fit_label"] = training["label"]
        training["fit_return"] = training["realized_net_return"]
        if permutation_seed is not None:
            training = _permute_training(training, permutation_seed + fold_number * 10_000)
        selected, inner_prediction = _select_model(kind, features.loc[training.index], training, cfg)
        calibrator = _fit_calibrator(inner_prediction, training)
        threshold = _choose_threshold(_calibrated_series(inner_prediction, calibrator), training, cfg["meta"]["thresholds"])
        preprocessor = Preprocessor.fit(features.loc[training.index])
        model = _fit_raw_model(
            kind,
            selected,
            preprocessor.transform(features.loc[training.index]),
            training["fit_label"].astype(int).to_numpy(),
            training["uniqueness"].to_numpy(dtype=float),
        )
        raw = model.predict_proba(preprocessor.transform(features.loc[testing.index]))[:, 1]
        probability = _calibrate(calibrator, raw)
        output.loc[testing.index, "probability"] = probability
        output.loc[testing.index, "base_rate"] = _weighted_base_rate(training["fit_label"], training["uniqueness"])
        output.loc[testing.index, "threshold"] = threshold
        output.loc[testing.index, "fold"] = fold_number
    return output


def _calibrated_series(prediction: pd.Series, calibrator) -> pd.Series:
    result = prediction.copy()
    valid = result.notna()
    if valid.any():
        result.loc[valid] = _calibrate(calibrator, result.loc[valid].to_numpy())
    return result


def _slice_run(run: StrategyRun, dates: pd.DatetimeIndex) -> StrategyRun:
    portfolio = run.portfolio
    selected = portfolio.returns.index.intersection(dates)
    sliced = type(portfolio)(
        portfolio.returns.loc[selected],
        portfolio.gross_returns.loc[selected],
        portfolio.weights.loc[selected],
        portfolio.turnover.loc[selected],
        portfolio.costs.loc[selected],
        portfolio.exposure.loc[selected],
    )
    return StrategyRun(run.name, sliced, calculate_metrics(sliced.returns, sliced.turnover, sliced.exposure))


def _event_multiplier(predictions: pd.DataFrame, daily_index: pd.DatetimeIndex, mode: str) -> pd.Series:
    multiplier = pd.Series(0.0, index=daily_index)
    valid = predictions[predictions["probability"].notna()]
    for signal_time, row in valid.iterrows():
        if mode == "sized":
            value = float(np.clip(2 * row["probability"] - 1, 0, 1))
        else:
            value = float(row["probability"] > row["threshold"])
        active = (multiplier.index >= signal_time) & (multiplier.index < row["episode_end_time"])
        multiplier.loc[active] = value
    return multiplier


def _meta_run(
    name: str,
    predictions: pd.DataFrame,
    mode: str,
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    volume: pd.DataFrame,
    base_decision: pd.Series,
    oos_dates: pd.DatetimeIndex,
    cfg: dict,
) -> StrategyRun:
    btc = _btc_column(close_prices)
    multiplier = _event_multiplier(predictions, close_prices.index, mode)
    decisions = (base_decision * multiplier).to_frame(btc)
    portfolio = simulate_portfolio(
        open_prices[[btc]],
        decisions,
        (close_prices * volume)[[btc]],
        bps_per_side=cfg["costs"]["headline_bps_per_side"],
        notional_usd=cfg["costs"]["paper_trade_notional_usd"],
        impact_coefficient=cfg["costs"]["slippage_impact_coefficient"],
    )
    return _slice_run(StrategyRun(name, portfolio, calculate_metrics(portfolio.returns, portfolio.turnover, portfolio.exposure)), oos_dates)


def _event_block_cis(frame: pd.DataFrame, samples: int, seed: int) -> dict[str, tuple[float, float] | None]:
    clean = frame.dropna(subset=["label", "probability", "base_rate", "uniqueness"]).copy()
    if clean.empty:
        return {"auc": None, "brier": None, "log_loss": None}
    weeks = clean.index.tz_localize(None).to_period("W")
    blocks = [group for _, group in clean.groupby(weeks)]
    if len(blocks) < 3:
        return {"auc": None, "brier": None, "log_loss": None}
    rng = np.random.default_rng(seed)
    values = {"auc": [], "brier": [], "log_loss": []}
    for _ in range(samples):
        sample = pd.concat([blocks[index] for index in rng.integers(0, len(blocks), len(blocks))], ignore_index=True)
        y = sample["label"].astype(int).to_numpy()
        p = np.clip(sample["probability"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        base = np.clip(sample["base_rate"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        weight = sample["uniqueness"].to_numpy(dtype=float)
        if len(np.unique(y)) == 2:
            values["auc"].append(roc_auc_score(y, p, sample_weight=weight) - roc_auc_score(y, base, sample_weight=weight))
        values["brier"].append(brier_score_loss(y, p, sample_weight=weight) - brier_score_loss(y, base, sample_weight=weight))
        values["log_loss"].append(log_loss(y, p, sample_weight=weight, labels=[0, 1]) - log_loss(y, base, sample_weight=weight, labels=[0, 1]))
    result: dict[str, tuple[float, float] | None] = {}
    for name, observations in values.items():
        finite = np.asarray(observations, dtype=float)
        finite = finite[np.isfinite(finite)]
        result[name] = (
            (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)))
            if len(finite) >= max(30, samples // 10)
            else None
        )
    return result


def classification_metrics(predictions: pd.DataFrame, samples: int, seed: int) -> dict[str, Any]:
    clean = predictions.dropna(subset=["label", "probability", "base_rate", "uniqueness"])
    if clean.empty:
        return {
            "n": 0,
            "auc": np.nan,
            "base_auc": np.nan,
            "brier": np.nan,
            "base_brier": np.nan,
            "log_loss": np.nan,
            "base_log_loss": np.nan,
            "cis": {"auc": None, "brier": None, "log_loss": None},
        }
    y = clean["label"].astype(int).to_numpy()
    p = np.clip(clean["probability"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    base = np.clip(clean["base_rate"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    weight = clean["uniqueness"].to_numpy(dtype=float)
    auc = roc_auc_score(y, p, sample_weight=weight) if len(np.unique(y)) == 2 else np.nan
    base_auc = roc_auc_score(y, base, sample_weight=weight) if len(np.unique(y)) == 2 else np.nan
    return {
        "n": len(clean),
        "auc": float(auc),
        "base_auc": float(base_auc),
        "brier": float(brier_score_loss(y, p, sample_weight=weight)),
        "base_brier": float(brier_score_loss(y, base, sample_weight=weight)),
        "log_loss": float(log_loss(y, p, sample_weight=weight, labels=[0, 1])),
        "base_log_loss": float(log_loss(y, base, sample_weight=weight, labels=[0, 1])),
        "cis": _event_block_cis(clean, samples, seed),
    }


def _fit_final_logistic(features: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> dict[str, Any] | None:
    training = events.dropna(subset=["label", "realized_net_return"]).copy()
    if training.empty or training["label"].nunique() < 2:
        return None
    training["fit_label"] = training["label"]
    training["fit_return"] = training["realized_net_return"]
    selected, inner_prediction = _select_model("logistic", features.loc[training.index], training, cfg)
    calibrator = _fit_calibrator(inner_prediction, training)
    calibrated_inner = _calibrated_series(inner_prediction, calibrator)
    threshold = _choose_threshold(calibrated_inner, training, cfg["meta"]["thresholds"])
    preprocessor = Preprocessor.fit(features.loc[training.index])
    model = _fit_raw_model(
        "logistic",
        selected,
        preprocessor.transform(features.loc[training.index]),
        training["label"].astype(int).to_numpy(),
        training["uniqueness"].to_numpy(dtype=float),
    )
    calibration = None
    if calibrator is not None:
        calibration = {
            "coefficient": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
        }
    return {
        "version": "meta-labeling-v1",
        "training_end": str(training["label_end_time"].max()),
        "feature_groups": FEATURE_GROUPS,
        "preprocessor": preprocessor.to_dict(),
        "C": selected["C"],
        "coefficient": model.coef_[0].astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
        "calibration": calibration,
        "threshold": threshold,
        "sizing": "clip(2p-1,0,1)",
    }


def predict_logistic_artifact(features: pd.DataFrame, artifact: dict[str, Any]) -> pd.Series:
    preprocessor = Preprocessor.from_dict(artifact["preprocessor"])
    x = preprocessor.transform(features)
    score = x @ np.asarray(artifact["coefficient"], dtype=float) + float(artifact["intercept"])
    probability = 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
    calibration = artifact.get("calibration")
    if calibration:
        calibrated_score = calibration["coefficient"] * _logit(probability).ravel() + calibration["intercept"]
        probability = 1.0 / (1.0 + np.exp(-np.clip(calibrated_score, -40, 40)))
    return pd.Series(probability, index=features.index, name="probability")


def write_meta_artifact(artifact: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt_pct(value: float) -> str:
    return "n/e" if not np.isfinite(value) else f"{value * 100:.1f}%"


def _fmt_num(value: float) -> str:
    return "n/e" if not np.isfinite(value) else f"{value:.3f}"


def _fmt_ci(value: tuple[float, float] | None, percent: bool = False) -> str:
    if value is None:
        return "not estimable"
    if np.isclose(value[0], value[1], rtol=0, atol=1e-12):
        return "not estimable (zero-width bootstrap)"
    if percent:
        return f"[{value[0] * 100:.1f}%, {value[1] * 100:.1f}%]"
    return f"[{value[0]:.3f}, {value[1]:.3f}]"


def _classification_sentence(name: str, metrics: dict[str, Any]) -> str:
    cis = metrics["cis"]
    ranking = bool(np.isfinite(metrics["auc"]) and metrics["auc"] > metrics["base_auc"] and cis["auc"] is not None and cis["auc"][0] > 0)
    quality = bool(
        metrics["brier"] < metrics["base_brier"]
        and metrics["log_loss"] < metrics["base_log_loss"]
        and (
            (cis["brier"] is not None and cis["brier"][1] < 0)
            or (cis["log_loss"] is not None and cis["log_loss"][1] < 0)
        )
    )
    return (
        f"{name} {'beats' if quality else 'does not beat'} the base-rate baseline on probability quality "
        f"(log loss and Brier) and {'beats' if ranking else 'does not beat'} it on ranking (AUC), under the locked CI rule."
    )


def _prediction_difference_cis(
    full: pd.DataFrame,
    ablated: pd.DataFrame,
    samples: int,
    seed: int,
) -> dict[str, tuple[float, float] | None]:
    joined = pd.DataFrame(
        {
            "label": full["label"],
            "weight": full["uniqueness"],
            "full": full["probability"],
            "ablated": ablated["probability"],
        }
    ).dropna()
    if joined.empty:
        return {"auc": None, "brier": None}
    weeks = joined.index.tz_localize(None).to_period("W")
    blocks = [group for _, group in joined.groupby(weeks)]
    if len(blocks) < 3:
        return {"auc": None, "brier": None}
    rng = np.random.default_rng(seed)
    auc_values, brier_values = [], []
    for _ in range(samples):
        sample = pd.concat([blocks[index] for index in rng.integers(0, len(blocks), len(blocks))], ignore_index=True)
        y = sample["label"].astype(int).to_numpy()
        weight = sample["weight"].to_numpy(dtype=float)
        full_probability = sample["full"].to_numpy(dtype=float)
        ablated_probability = sample["ablated"].to_numpy(dtype=float)
        if len(np.unique(y)) == 2:
            auc_values.append(
                roc_auc_score(y, ablated_probability, sample_weight=weight)
                - roc_auc_score(y, full_probability, sample_weight=weight)
            )
        brier_values.append(
            brier_score_loss(y, ablated_probability, sample_weight=weight)
            - brier_score_loss(y, full_probability, sample_weight=weight)
        )

    def interval(values: list[float]) -> tuple[float, float] | None:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) < max(30, samples // 10):
            return None
        return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))

    return {"auc": interval(auc_values), "brier": interval(brier_values)}


def _empty_meta_result(reason: str) -> MetaStudyResult:
    section = f"""## Meta-labeling

Not estimable: {reason}. No model result or forward artifact was produced.
"""
    return MetaStudyResult({}, section, pd.DataFrame(), {}, {}, [], {}, False, None)


def run_meta_study(
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    volume: pd.DataFrame,
    top20: pd.DataFrame,
    funding: pd.DataFrame | None,
    prior_runs: dict[str, StrategyRun],
    cfg: dict,
    source_label: str,
) -> MetaStudyResult:
    cfg = copy.deepcopy(cfg)
    if "synthetic" in source_label.lower():
        cfg["meta"]["ci_mode"] = True
        cfg["meta"]["permutations"] = min(int(cfg["meta"]["permutations"]), 3)
    btc = _btc_column(close_prices)
    lookbacks = cfg["strategy"]["daily_lookbacks"]
    signal = trend_signal(close_prices[[btc]], lookbacks)[btc]
    realized_volatility = trailing_realized_volatility(close_prices[[btc]])[btc]
    base_decision = volatility_targeted_weights(
        signal.to_frame(btc),
        realized_volatility.to_frame(btc),
        cfg["strategy"]["headline_volatility_target"],
        cfg["strategy"]["max_exposure"],
    )[btc]
    features = build_meta_features(close_prices, volume, top20, funding, lookbacks)
    assert_meta_features_point_in_time(features, close_prices)
    events = build_triple_barrier_events(
        open_prices[btc],
        close_prices[btc],
        (close_prices * volume)[btc],
        signal,
        horizon=cfg["meta"]["label_horizon_days"],
        volatility_multiple=cfg["meta"]["barrier_volatility_multiple"],
        barrier_floor=cfg["meta"]["barrier_floor"],
        barrier_cap=cfg["meta"]["barrier_cap"],
        bps_per_side=cfg["costs"]["headline_bps_per_side"],
        impact_coefficient=cfg["costs"]["slippage_impact_coefficient"],
        notional_usd=cfg["costs"]["paper_trade_notional_usd"],
    )
    if events.empty:
        return _empty_meta_result("the fixed H1 rule generated no eligible labeled entry events")
    eligible = events.index.intersection(features.dropna(how="all").index)
    events = events.loc[eligible].sort_index()
    features = features.loc[events.index]
    if len(events) < 16 or events["label"].nunique() < 2:
        return _empty_meta_result(f"only {len(events)} labeled events or one label class was available")

    folds = expanding_walk_forward(
        close_prices.index,
        cfg["validation"]["minimum_training_days"],
        cfg["validation"]["test_block_months"],
        cfg["meta"]["purge_days"],
    )
    oos_dates = pd.DatetimeIndex(sorted(set().union(*(set(fold.test) for fold in folds)))) if folds else close_prices.index[:0]
    if not len(oos_dates) or "BTC trend" not in prior_runs:
        return _empty_meta_result("no purged out-of-sample dates were available")

    samples = cfg["validation"]["bootstrap_samples"]
    seed = cfg["analysis"]["seed"]
    predictions: dict[str, pd.DataFrame] = {}
    classification: dict[str, dict[str, Any]] = {}
    runs: dict[str, StrategyRun] = {}
    for offset, kind in enumerate(("logistic", "boosting")):
        prediction = oof_meta_predictions(kind, features, events, close_prices.index, cfg)
        predictions[kind] = prediction
        classification[kind] = classification_metrics(prediction, samples, seed + 1_000 + offset * 100)
        for mode in ("sized", "threshold"):
            display = f"BTC trend + {kind} {mode}"
            runs[display] = _meta_run(
                display,
                prediction,
                mode,
                open_prices,
                close_prices,
                volume,
                base_decision,
                oos_dates,
                cfg,
            )

    rule = prior_runs["BTC trend"]
    paired = {
        name: paired_weekly_block_bootstrap_metric_cis(
            run.portfolio.returns,
            rule.portfolio.returns,
            samples=samples,
            seed=seed + 2_000 + number * 10,
        )
        for number, (name, run) in enumerate(runs.items())
    }

    ablations: list[dict[str, Any]] = []
    for model_number, kind in enumerate(("logistic", "boosting")):
        full_name = f"BTC trend + {kind} sized"
        full_run = runs[full_name]
        full_metrics = classification[kind]
        for group_number, (group, columns) in enumerate(FEATURE_GROUPS.items()):
            remaining = features.drop(columns=columns)
            prediction = oof_meta_predictions(kind, remaining, events, close_prices.index, cfg)
            metrics = classification_metrics(prediction, samples, seed + 3_000 + model_number * 500 + group_number * 20)
            run = _meta_run(
                f"{full_name} without {group}",
                prediction,
                "sized",
                open_prices,
                close_prices,
                volume,
                base_decision,
                oos_dates,
                cfg,
            )
            strategy_ci = paired_weekly_block_bootstrap_metric_cis(
                run.portfolio.returns,
                full_run.portfolio.returns,
                samples=samples,
                seed=seed + 4_000 + model_number * 500 + group_number * 20,
            )
            prediction_ci = _prediction_difference_cis(
                predictions[kind],
                prediction,
                samples,
                seed + 5_000 + model_number * 500 + group_number * 20,
            )
            ablations.append(
                {
                    "model": kind,
                    "group": group,
                    "auc_change": metrics["auc"] - full_metrics["auc"],
                    "auc_ci": prediction_ci["auc"],
                    "brier_change": metrics["brier"] - full_metrics["brier"],
                    "brier_ci": prediction_ci["brier"],
                    "cagr_change": run.metrics.cagr - full_run.metrics.cagr,
                    "sharpe_change": run.metrics.sharpe - full_run.metrics.sharpe,
                    "max_drawdown_change": run.metrics.max_drawdown - full_run.metrics.max_drawdown,
                    "strategy_ci": strategy_ci,
                }
            )

    permutation: dict[str, dict[str, float]] = {}
    permutations = int(cfg["meta"]["permutations"])
    if cfg["meta"].get("ci_mode"):
        permutations = min(permutations, 3)
    for model_number, kind in enumerate(("logistic", "boosting")):
        real = runs[f"BTC trend + {kind} sized"].metrics.sharpe - rule.metrics.sharpe
        null = []
        for number in range(permutations):
            prediction = oof_meta_predictions(
                kind,
                features,
                events,
                close_prices.index,
                cfg,
                permutation_seed=seed + 10_000 + model_number * 100_000 + number,
            )
            run = _meta_run(
                f"{kind} permutation {number}",
                prediction,
                "sized",
                open_prices,
                close_prices,
                volume,
                base_decision,
                oos_dates,
                cfg,
            )
            null.append(run.metrics.sharpe - rule.metrics.sharpe)
        finite = np.asarray(null, dtype=float)
        finite = finite[np.isfinite(finite)]
        permutation[kind] = {
            "count": float(len(finite)),
            "real_sharpe_change": float(real),
            "null_95": float(np.quantile(finite, 0.95)) if len(finite) else np.nan,
            "p_value": float((1 + np.sum(finite >= real)) / (1 + len(finite))) if len(finite) else np.nan,
        }

    prior_names = [
        "BTC trend",
        "ETH trend",
        "Top-10 trend",
        "Top-20 momentum",
        "Momentum + BTC regime",
        "BTC trend + funding filter",
        "ETH trend + funding filter",
    ]
    prior_variant_runs = {name: prior_runs[name] for name in prior_names if name in prior_runs}
    combined_runs = {**prior_variant_runs, **runs}
    common_returns = pd.concat({name: run.portfolio.returns for name, run in combined_runs.items()}, axis=1).dropna()
    prior_common = pd.concat({name: run.portfolio.returns for name, run in prior_variant_runs.items()}, axis=1).dropna()
    combined_pbo = probability_of_backtest_overfitting(common_returns)
    prior_pbo = probability_of_backtest_overfitting(prior_common)
    dsr = {name: deflated_sharpe_ratio(run.portfolio.returns, 11) for name, run in {"rule": rule, **runs}.items()}

    model_support: dict[str, bool] = {}
    for kind in ("logistic", "boosting"):
        name = f"BTC trend + {kind} sized"
        run = runs[name]
        interval = paired[name]
        c = classification[kind]
        quality = (
            c["brier"] < c["base_brier"]
            and c["log_loss"] < c["base_log_loss"]
            and (
                (c["cis"]["brier"] is not None and c["cis"]["brier"][1] < 0)
                or (c["cis"]["log_loss"] is not None and c["cis"]["log_loss"][1] < 0)
            )
        )
        risk_return = (
            (run.metrics.cagr >= rule.metrics.cagr - 0.02 and run.metrics.max_drawdown >= rule.metrics.max_drawdown)
            or (
                interval["max_drawdown"] is not None
                and interval["max_drawdown"][0] > 0
                and interval["cagr"] is not None
                and interval["cagr"][0] > -0.02
            )
        )
        model_support[kind] = bool(
            interval["sharpe"] is not None
            and interval["sharpe"][0] > 0
            and risk_return
            and quality
            and permutation[kind]["real_sharpe_change"] > permutation[kind]["null_95"]
            and dsr[name] > dsr["rule"]
            and np.isfinite(combined_pbo)
            and np.isfinite(prior_pbo)
            and combined_pbo <= prior_pbo + 0.05
        )
    supported = any(model_support.values())

    classification_rows = [
        "| Model | n | AUC vs base | Brier vs base | Log loss vs base | Paired difference CIs (model − base) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for kind in ("logistic", "boosting"):
        metrics = classification[kind]
        classification_rows.append(
            f"| {kind} | {metrics['n']} | {_fmt_num(metrics['auc'])} vs {_fmt_num(metrics['base_auc'])} | "
            f"{_fmt_num(metrics['brier'])} vs {_fmt_num(metrics['base_brier'])} | "
            f"{_fmt_num(metrics['log_loss'])} vs {_fmt_num(metrics['base_log_loss'])} | "
            f"AUC {_fmt_ci(metrics['cis']['auc'])}; Brier {_fmt_ci(metrics['cis']['brier'])}; log loss {_fmt_ci(metrics['cis']['log_loss'])} |"
        )

    strategy_rows = [
        "| Strategy | CAGR | Vol | Sharpe | Sortino | Calmar | Max DD | Underwater days | Worst month | Turnover | Paired Sharpe CI vs rule |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["BTC trend", *runs]:
        run = rule if name == "BTC trend" else runs[name]
        m = run.metrics
        sharpe_ci = None if name == "BTC trend" else paired[name]["sharpe"]
        strategy_rows.append(
            f"| {name} | {_fmt_pct(m.cagr)} | {_fmt_pct(m.annualized_volatility)} | {_fmt_num(m.sharpe)} | "
            f"{_fmt_num(m.sortino)} | {_fmt_num(m.calmar)} | {_fmt_pct(m.max_drawdown)} | {m.longest_underwater_days} | "
            f"{_fmt_pct(m.worst_month)} | {m.turnover:.1f}× | {'—' if sharpe_ci is None else _fmt_ci(sharpe_ci)} |"
        )

    ablation_rows = [
        "| Model | Dropped group | ΔAUC (95% CI) | ΔBrier (95% CI) | ΔCAGR | ΔSharpe (95% CI) | ΔMax DD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in ablations:
        ablation_rows.append(
            f"| {item['model']} | {item['group']} | {_fmt_num(item['auc_change'])} {_fmt_ci(item['auc_ci'])} | "
            f"{_fmt_num(item['brier_change'])} {_fmt_ci(item['brier_ci'])} | {_fmt_pct(item['cagr_change'])} | "
            f"{_fmt_num(item['sharpe_change'])} {_fmt_ci(item['strategy_ci']['sharpe'])} | {_fmt_pct(item['max_drawdown_change'])} |"
        )

    permutation_rows = ["| Model | Real ΔSharpe | Null 95th percentile | Empirical p |", "|---|---:|---:|---:|"]
    for kind, item in permutation.items():
        permutation_rows.append(
            f"| {kind} | {_fmt_num(item['real_sharpe_change'])} | {_fmt_num(item['null_95'])} | {_fmt_num(item['p_value'])} |"
        )

    result_text = (
        "At least one sized meta-model passed every pre-registered success gate."
        if supported
        else "No sized meta-model passed every pre-registered success gate."
    )
    report_section = f"""## Meta-labeling

The fixed BTC H1 rule generated {len(events)} labeled entry events; {int(events['label'].sum())} ({events['label'].mean() * 100:.1f}%) hit the +2σ close barrier first. Every probability below is outer-fold out of sample, and every strategy return is next-open and net of the locked costs. **Result: {result_text}**

{chr(10).join(strategy_rows)}

### Probability quality

{chr(10).join(classification_rows)}

{_classification_sentence('Logistic regression', classification['logistic'])}

{_classification_sentence('Gradient boosting', classification['boosting'])}

### Feature-group ablation

Negative ΔAUC and positive ΔBrier mean the dropped group helped the full model. Strategy CIs are paired weekly-block differences for the ablated sized strategy minus its full-model counterpart.

{chr(10).join(ablation_rows)}

### Permutation null

The null refits model selection, preprocessing, calibration, and sizing after quarter-block label/return shuffles. Production uses {permutations} seeded permutations.

{chr(10).join(permutation_rows)}

### Multiple-testing correction

- Deflated Sharpe probability with 11 total strategy trials: rule {_fmt_pct(dsr['rule'])}; logistic sized {_fmt_pct(dsr['BTC trend + logistic sized'])}; boosting sized {_fmt_pct(dsr['BTC trend + boosting sized'])}.
- CSCV PBO: prior seven-variant set {_fmt_pct(prior_pbo)}; combined eleven-variant set {_fmt_pct(combined_pbo)}.
- Optional on-chain features were skipped because no qualifying free, keyless source with a locked historical publication lag was available.
- The logistic sized model remains a paper-only forward candidate regardless of this backtest verdict; nothing trades automatically.
"""

    combined_events = events.copy()
    for kind, prediction in predictions.items():
        combined_events[f"{kind}_probability"] = prediction["probability"]
        combined_events[f"{kind}_base_rate"] = prediction["base_rate"]
    artifact = _fit_final_logistic(features, events, cfg)
    return MetaStudyResult(runs, report_section, combined_events, classification, paired, ablations, permutation, supported, artifact)
