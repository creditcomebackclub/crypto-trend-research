#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctr.config import load_config
from ctr.data.audit import audit_bars, cross_exchange_divergence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = load_config()
    raw = ROOT / "data" / "raw"
    all_coinbase_daily = sorted((raw / "coinbase" / "1d").glob("*.parquet"))
    if not all_coinbase_daily:
        raise SystemExit("No Coinbase daily cache. Run scripts/fetch_spot.py first.")
    products_path = raw / "coinbase" / "products.json"
    products = json.loads(products_path.read_text()) if products_path.exists() else []
    catalog = {str(item["id"]): item for item in products}
    quote_rank = {"USD": 0, "USDC": 1, "USDT": 2}
    excluded = set(config["universe"]["excluded_symbols"])
    preferred: dict[str, tuple[int, Path]] = {}
    for path in all_coinbase_daily:
        meta = catalog.get(path.stem, {})
        quote = str(meta.get("quote_currency", path.stem.rsplit("-", 1)[-1]))
        base = str(meta.get("base_currency", path.stem[: -(len(quote) + 1)]))
        if meta.get("fx_stablecoin") or base.upper() in excluded:
            continue
        candidate = (quote_rank.get(quote, 99), path)
        if base not in preferred or candidate[0] < preferred[base][0]:
            preferred[base] = candidate
    coinbase_daily = sorted(item[1] for item in preferred.values())
    analysis_end = pd.Timestamp(config["analysis"]["end"], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    long_frames = []
    audit: dict[str, object] = {"assets": {}, "cross_exchange": {}, "notes": []}
    for path in coinbase_daily:
        frame = pd.read_parquet(path).sort_index().loc[:analysis_end]
        asset = path.stem
        audit["assets"][asset] = audit_bars(frame, "1D")
        if len(frame):
            item = frame.reset_index().rename(columns={frame.index.name or "index": "timestamp"})
            item["asset"] = asset
            long_frames.append(item[["timestamp", "asset", "open", "high", "low", "close", "volume"]])
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    daily = pd.concat(long_frames, ignore_index=True).sort_values(["timestamp", "asset"])
    daily.to_parquet(processed / "daily.parquet", index=False)
    for asset, kraken_name in {"BTC-USD": "XXBTZUSD", "ETH-USD": "XETHZUSD"}.items():
        left_path = raw / "coinbase" / "1d" / f"{asset}.parquet"
        right_path = raw / "kraken" / "1d" / f"{kraken_name}.parquet"
        if left_path.exists() and right_path.exists():
            audit["cross_exchange"][asset] = cross_exchange_divergence(
                pd.read_parquet(left_path)["close"], pd.read_parquet(right_path)["close"]
            )
    if products_path.exists():
        audit["catalog"] = {
            "downloaded_products": len(products),
            "offline_or_delisted_visible": sum(p.get("status") != "online" for p in products),
            "cached_offline_or_delisted": sum(
                catalog.get(path.stem, {}).get("status") != "online" for path in all_coinbase_daily
            ),
            "selected_unique_base_assets": len(coinbase_daily),
            "warning": "The current catalog cannot prove completeness for products removed from the API.",
        }
    else:
        audit["notes"].append("Coinbase product catalog was not cached; delisting visibility is unknown.")
    (processed / "data_quality.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in raw.rglob("*") if path.is_file()) + sorted(path for path in processed.rglob("*") if path.is_file())
    release = config["data"]["snapshot_release"]
    out_dir = ROOT / "data" / "releases"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{release}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in files:
            bundle.add(path, arcname=path.relative_to(ROOT / "data"))
    manifest = {
        "release": release,
        "analysis_end": config["analysis"]["end"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset": archive.name,
        "sha256": sha256(archive),
        "files": {str(path.relative_to(ROOT / "data")): sha256(path) for path in files},
    }
    (ROOT / "data" / "SNAPSHOT.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
