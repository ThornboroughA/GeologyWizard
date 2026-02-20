from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_root: Path
    keyframe_interval_myr: int = 5
    default_preview_width: int = 512
    default_preview_height: int = 256


def load_settings() -> Settings:
    env_root = os.environ.get("GW_DATA_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
    else:
        root = Path.home() / ".geologic_wizard"
    return Settings(data_root=root)
