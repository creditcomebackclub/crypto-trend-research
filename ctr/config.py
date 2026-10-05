from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the JSON-compatible YAML config without an extra YAML dependency."""
    target = Path(path) if path else ROOT / "config.yaml"
    return json.loads(target.read_text(encoding="utf-8"))

