#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctr.report import run_research, write_results
from ctr.config import load_config


def load_long(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(path)
    required = {"timestamp", "asset", "open", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain {sorted(required)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return tuple(frame.pivot(index="timestamp", columns="asset", values=value).sort_index() for value in ["open", "close", "volume"])


def load_fixture(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return tuple(frame.pivot(index="timestamp", columns="asset", values=value).sort_index() for value in ["open", "close", "volume"])


def load_deribit(root: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    funding_columns, close_columns, high_columns = {}, {}, {}
    for path in sorted((root / "raw" / "deribit" / "funding").glob("*.parquet")):
        frame = pd.read_parquet(path)
        if len(frame):
            column = "interest_8h" if "interest_8h" in frame else frame.columns[0]
            funding_columns[f"{path.stem}-USD"] = frame[column].resample("1D").sum()
    for path in sorted((root / "raw" / "deribit" / "perpetual").glob("*.parquet")):
        frame = pd.read_parquet(path)
        if len(frame):
            close_columns[f"{path.stem}-USD"] = frame["close"]
            high_columns[f"{path.stem}-USD"] = frame["high"]
    make = lambda values: pd.DataFrame(values).sort_index() if values else None
    return make(funding_columns), make(close_columns), make(high_columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the locked research report")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data" / "processed" / "daily.parquet")
    parser.add_argument("--synthetic", action="store_true", help="Run the committed CI fixture; results are visibly labeled synthetic")
    args = parser.parse_args()
    if args.synthetic:
        inputs = load_fixture(ROOT / "tests" / "fixtures" / "synthetic_daily.csv")
        label = "committed synthetic CI fixture (not empirical evidence)"
        config = load_config()
        config["validation"]["bootstrap_samples"] = 100
        extras = {}
    else:
        if not args.snapshot.exists():
            raise SystemExit("Frozen processed snapshot missing. Run scripts/fetch_snapshot.py or use --synthetic for CI.")
        inputs = load_long(args.snapshot)
        label = "frozen public-data snapshot"
        config = load_config()
        funding, perp_close, perp_high = load_deribit(ROOT / "data")
        extras = {"funding": funding, "perp_close": perp_close, "perp_high": perp_high}
    report, runs = run_research(*inputs, source_label=label, config=config, **extras)
    write_results(report, runs)
    print(f"Wrote reports/REPORT.md with {len(runs)} out-of-sample strategy rows")


if __name__ == "__main__":
    main()
