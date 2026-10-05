#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the frozen research snapshot")
    parser.add_argument("--url", help="Release asset URL; defaults to release URL recorded in SNAPSHOT.json")
    args = parser.parse_args()
    manifest_path = ROOT / "data" / "SNAPSHOT.json"
    manifest = json.loads(manifest_path.read_text())
    url = args.url or manifest.get("url")
    if not url:
        raise SystemExit("Snapshot URL is not published yet. Pass --url after the GitHub Release is created.")
    archive = ROOT / "data" / "releases" / manifest["asset"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, archive)
    actual = sha256(archive)
    if actual != manifest["sha256"]:
        archive.unlink(missing_ok=True)
        raise SystemExit(f"SHA-256 mismatch: expected {manifest['sha256']}, got {actual}")
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(ROOT / "data", filter="data")
    print(f"Verified and extracted {archive.name}")


if __name__ == "__main__":
    main()

