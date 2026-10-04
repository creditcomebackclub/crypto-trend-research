#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctr.config import load_config
from ctr.data.public import deribit_funding_history, deribit_perpetual_candles, merge_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public Deribit perpetual funding history")
    parser.add_argument("--currencies", nargs="*", default=["BTC", "ETH", "SOL"])
    args = parser.parse_args()
    config = load_config()
    start = pd.Timestamp(config["analysis"]["start"], tz="UTC")
    end = pd.Timestamp(config["analysis"]["end"], tz="UTC") + pd.Timedelta(days=1)
    for currency in args.currencies:
        path = ROOT / "data" / "raw" / "deribit" / "funding" / f"{currency.upper()}.parquet"
        resume = start
        if path.exists():
            old = pd.read_parquet(path)
            if len(old):
                resume = max(resume, old.index.max() + pd.Timedelta(milliseconds=1))
        try:
            fresh = deribit_funding_history(currency, resume, end)
            merged = merge_cache(path, fresh.loc[:end])
            price_path = ROOT / "data" / "raw" / "deribit" / "perpetual" / f"{currency.upper()}.parquet"
            prices = deribit_perpetual_candles(currency, start, end)
            price_merged = merge_cache(price_path, prices.loc[:end])
            print(f"{currency}: {len(merged)} funding rows, {len(price_merged)} perpetual bars")
        except Exception as error:
            print(f"{currency}: unavailable ({type(error).__name__}: {error})")


if __name__ == "__main__":
    main()
