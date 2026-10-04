from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


USER_AGENT = "crypto-trend-research/0.1 (public research client)"


def get_json(url: str, params: dict[str, object] | None = None, attempts: int = 5) -> object:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(2**attempt, 15))
    raise RuntimeError("unreachable")


def coinbase_products() -> list[dict[str, object]]:
    data = get_json("https://api.exchange.coinbase.com/products")
    if not isinstance(data, list):
        raise ValueError("unexpected Coinbase products response")
    return data


def coinbase_candles(product: str, start: pd.Timestamp, end: pd.Timestamp, granularity: int) -> pd.DataFrame:
    """Fetch Coinbase candles in <=300-candle chunks."""
    frames = []
    cursor = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    stop = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    span = pd.Timedelta(seconds=granularity * 299)
    while cursor < stop:
        chunk_end = min(cursor + span, stop)
        rows = get_json(
            f"https://api.exchange.coinbase.com/products/{urllib.parse.quote(product)}/candles",
            {"start": cursor.isoformat(), "end": chunk_end.isoformat(), "granularity": granularity},
        )
        if isinstance(rows, list) and rows:
            frame = pd.DataFrame(rows, columns=["timestamp", "low", "high", "open", "close", "volume"])
            frames.append(frame)
        cursor = chunk_end + pd.Timedelta(seconds=granularity)
        time.sleep(0.12)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    result = result.set_index("timestamp").sort_index()
    return result[["open", "high", "low", "close", "volume"]].astype(float)[~result.index.duplicated(keep="last")]


def kraken_asset_pairs() -> dict[str, object]:
    data = get_json("https://api.kraken.com/0/public/AssetPairs")
    if not isinstance(data, dict) or data.get("error"):
        raise ValueError(f"Kraken error: {data}")
    return data["result"]


def kraken_ohlc(pair: str, interval_minutes: int = 1440, since: int | None = None) -> pd.DataFrame:
    params: dict[str, object] = {"pair": pair, "interval": interval_minutes}
    if since is not None:
        params["since"] = since
    data = get_json("https://api.kraken.com/0/public/OHLC", params)
    if not isinstance(data, dict) or data.get("error"):
        raise ValueError(f"Kraken error: {data}")
    key = next(key for key in data["result"] if key != "last")
    rows = data["result"][key]
    columns = ["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"]
    frame = pd.DataFrame(rows, columns=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    return frame[["open", "high", "low", "close", "volume"]].astype(float)


def deribit_funding_history(currency: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    instrument = f"{currency.upper()}-PERPETUAL"
    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000) if pd.Timestamp(start).tzinfo is None else int(pd.Timestamp(start).timestamp() * 1000)
    stop = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if pd.Timestamp(end).tzinfo is None else int(pd.Timestamp(end).timestamp() * 1000)
    rows: list[dict[str, object]] = []
    while cursor < stop:
        payload = get_json(
            "https://www.deribit.com/api/v2/public/get_funding_rate_history",
            {"instrument_name": instrument, "start_timestamp": cursor, "end_timestamp": stop, "count": 1000},
        )
        batch = payload.get("result", []) if isinstance(payload, dict) else []
        if not batch:
            break
        rows.extend(batch)
        latest = max(int(row["timestamp"]) for row in batch)
        if latest <= cursor:
            break
        cursor = latest + 1
        time.sleep(0.12)
    if not rows:
        return pd.DataFrame(columns=["interest_8h", "interest_1h"])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame.set_index("timestamp").sort_index()[[c for c in ["interest_8h", "interest_1h"] if c in frame]]


def deribit_perpetual_candles(currency: str, start: pd.Timestamp, end: pd.Timestamp, resolution: str = "1D") -> pd.DataFrame:
    instrument = f"{currency.upper()}-PERPETUAL"
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    payload = get_json(
        "https://www.deribit.com/api/v2/public/get_tradingview_chart_data",
        {"instrument_name": instrument, "start_timestamp": start_ms, "end_timestamp": end_ms, "resolution": resolution},
    )
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    if result.get("status") != "ok" or not result.get("ticks"):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(result["ticks"], unit="ms", utc=True),
        "open": result["open"], "high": result["high"], "low": result["low"],
        "close": result["close"], "volume": result["volume"],
    })
    return frame.set_index("timestamp").sort_index().astype(float)


def merge_cache(path: Path, fresh: pd.DataFrame) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    merged = pd.concat([old, fresh]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.to_parquet(path)
    return merged
