from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def stable_hash(payload: Any) -> str:
    return sha256_bytes(stable_json_dumps(payload).encode("utf-8"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
