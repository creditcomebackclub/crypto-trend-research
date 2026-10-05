#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctr.config import load_config
from ctr.data.public import coinbase_candles, coinbase_products, deribit_funding_history
from ctr.signals import cross_sectional_momentum_weights, funding_zscore, trend_signal, trailing_realized_volatility, volatility_targeted_weights
from ctr.universe import point_in_time_universe


FIELDS = ["date_utc", "strategy", "asset", "target_weight", "reference_close", "prior_day_pnl", "source"]


def fetch_current_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    end = pd.Timestamp(datetime.now(timezone.utc)).floor("D")
    start = end - pd.Timedelta(days=290)
    excluded = set(cfg["universe"]["excluded_symbols"])
    quote_rank = {"USD": 0, "USDC": 1, "USDT": 2}
    preferred: dict[str, tuple[int, str]] = {}
    for product in coinbase_products():
        quote = str(product.get("quote_currency", ""))
        base = str(product.get("base_currency", "")).upper()
        if (
            quote not in quote_rank
            or product.get("status") != "online"
            or product.get("fx_stablecoin", False)
            or base in excluded
        ):
            continue
        candidate = (quote_rank[quote], str(product["id"]))
        if base not in preferred or candidate[0] < preferred[base][0]:
            preferred[base] = candidate
    candidates = [item[1] for item in preferred.values()]
    opens: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    for product in sorted(candidates):
        try:
            bars = coinbase_candles(product, start, end, 86400)
            bars = bars.loc[bars.index < end]  # exclude the still-forming UTC day
            if len(bars) >= 90:
                opens[product] = bars["open"]
                closes[product] = bars["close"]
                volumes[product] = bars["volume"]
        except Exception as error:
            print(f"skip {product}: {type(error).__name__}")
    if "BTC-USD" not in closes:
        raise RuntimeError("BTC-USD unavailable; refusing to write partial paper targets")
    return pd.DataFrame(opens).sort_index(), pd.DataFrame(closes).sort_index(), pd.DataFrame(volumes).sort_index()


def targets(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.Series]:
    cfg = load_config()
    excluded = set(cfg["universe"]["excluded_symbols"])
    top10 = point_in_time_universe(close, volume, 10, excluded, 90)
    top20 = point_in_time_universe(close, volume, 20, excluded, 90)
    signal = trend_signal(close, cfg["strategy"]["daily_lookbacks"])
    vol = trailing_realized_volatility(close)
    trend = volatility_targeted_weights(signal, vol, cfg["strategy"]["headline_volatility_target"])
    top10_trend = trend.where(top10, 0.0)
    top10_trend = top10_trend.div(top10_trend.abs().sum(axis=1).where(lambda x: x > 1, 1), axis=0)
    momentum = cross_sectional_momentum_weights(close, top20)
    btc_regime = signal["BTC-USD"]
    result = {
        "H1 BTC trend": trend[["BTC-USD"]].iloc[-1],
        "H1 top-10 trend": top10_trend.iloc[-1],
        "H3 top-20 momentum": momentum.iloc[-1],
        "H3 momentum + BTC regime": (momentum.mul(btc_regime, axis=0)).iloc[-1],
    }
    try:
        end = close.index[-1] + pd.Timedelta(days=1)
        funding = deribit_funding_history("BTC", end - pd.Timedelta(days=120), end)
        column = "interest_1h" if "interest_1h" in funding else funding.columns[0]
        daily_funding = funding[column].resample("1D").sum().to_frame(column)
        z = funding_zscore(daily_funding, cfg["strategy"]["funding_z_lookback"])[column].iloc[-1]
        h2 = result["H1 BTC trend"].copy()
        if z > cfg["strategy"]["funding_extreme_z"]:
            h2 *= 0.5
        result["H2 BTC funding filter"] = h2
    except Exception:
        result["H2 BTC funding filter"] = pd.Series({"BTC-USD": np.nan})
    return result


def main() -> None:
    open_prices, close, volume = fetch_current_panel()
    current = targets(close, volume)
    log = ROOT / "forward" / "paper_log.csv"
    existing = pd.read_csv(log) if log.exists() else pd.DataFrame(columns=FIELDS)
    # A bar labelled D closes at the next UTC midnight; targets execute there.
    today = (close.index[-1] + pd.Timedelta(days=1)).date().isoformat()
    prior = existing[existing["date_utc"] < today] if len(existing) else existing
    prior_date = prior["date_utc"].max() if len(prior) else None
    prior_rows = prior[prior["date_utc"] == prior_date] if prior_date else prior
    one_day = open_prices.pct_change(fill_method=None).iloc[-1]
    rows = []
    for strategy, weights in current.items():
        old = prior_rows[prior_rows["strategy"] == strategy].set_index("asset") if len(prior_rows) else pd.DataFrame()
        for asset, weight in weights.dropna().items():
            if abs(weight) < 1e-12:
                continue
            old_weight = float(old.loc[asset, "target_weight"]) if len(old) and asset in old.index else 0.0
            pnl = old_weight * float(one_day.get(asset, 0.0))
            rows.append({
                "date_utc": today, "strategy": strategy, "asset": asset,
                "target_weight": float(weight), "reference_close": float(close[asset].iloc[-1]),
                "prior_day_pnl": pnl, "source": "Coinbase Exchange public candles",
            })
    fresh = pd.DataFrame(rows)
    updated = pd.concat([existing, fresh], ignore_index=True) if len(existing) else fresh
    updated = updated.drop_duplicates(["date_utc", "strategy", "asset"], keep="last").sort_values(["date_utc", "strategy", "asset"])
    log.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(log, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Recorded {len(rows)} targets for {today}")


if __name__ == "__main__":
    main()
