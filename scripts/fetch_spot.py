#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctr.config import load_config
from ctr.data.public import coinbase_candles, coinbase_products, kraken_asset_pairs, kraken_ohlc, merge_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch resumable public spot OHLCV data")
    parser.add_argument("--source", choices=["coinbase", "kraken"], default="coinbase")
    parser.add_argument("--frequency", choices=["1d", "4h"], default="1d")
    parser.add_argument("--products", nargs="*", help="Explicit product IDs; defaults to all eligible USD/stable-USD pairs")
    parser.add_argument("--max-products", type=int, default=None, help="Audit/debug limit; omit for the research universe")
    return parser.parse_args()


def coinbase_ids(explicit: list[str] | None, quotes: set[str]) -> list[str]:
    if explicit:
        return explicit
    products = coinbase_products()
    metadata = ROOT / "data" / "raw" / "coinbase" / "products.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(products, indent=2) + "\n", encoding="utf-8")
    return sorted(
        str(p["id"])
        for p in products
        if p.get("quote_currency") in quotes and p.get("status") == "online" and not p.get("trading_disabled", False)
    )


def kraken_ids(explicit: list[str] | None, quotes: set[str]) -> list[str]:
    if explicit:
        return explicit
    pairs = kraken_asset_pairs()
    return sorted(name for name, meta in pairs.items() if str(meta.get("quote", "")).replace("Z", "") in quotes)


def main() -> None:
    args = parse_args()
    config = load_config()
    start = pd.Timestamp(config["analysis"]["start"], tz="UTC")
    end = pd.Timestamp(config["analysis"]["end"], tz="UTC") + pd.Timedelta(days=1)
    quotes = set(config["data"]["quote_currencies"])
    ids = coinbase_ids(args.products, quotes) if args.source == "coinbase" else kraken_ids(args.products, quotes)
    if args.max_products:
        ids = ids[: args.max_products]
    for number, product in enumerate(ids, 1):
        safe = product.replace("/", "-")
        path = ROOT / "data" / "raw" / args.source / args.frequency / f"{safe}.parquet"
        resume = start
        if path.exists():
            old = pd.read_parquet(path)
            if len(old):
                resume = max(resume, old.index.max() + pd.Timedelta(args.frequency))
        if resume >= end:
            print(f"[{number}/{len(ids)}] {product}: current")
            continue
        try:
            if args.source == "coinbase":
                granularity = 86400 if args.frequency == "1d" else 3600
                fresh = coinbase_candles(product, resume, end, granularity)
                if args.frequency == "4h" and len(fresh):
                    fresh = fresh.resample("4h", label="left", closed="left").agg(
                        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                    ).dropna(subset=["open", "close"])
            else:
                interval = 1440 if args.frequency == "1d" else 240
                fresh = kraken_ohlc(product, interval, int(resume.timestamp()))
            merged = merge_cache(path, fresh.loc[:end])
            print(f"[{number}/{len(ids)}] {product}: {len(merged)} rows")
        except Exception as error:
            print(f"[{number}/{len(ids)}] {product}: ERROR {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
