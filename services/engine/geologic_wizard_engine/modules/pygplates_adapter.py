from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PygplatesModelCache:
    available: bool
    status: str
    rotation_model: Any | None = None
    reconstruct_model: Any | None = None
    topological_model: Any | None = None

    @property
    def coverage_hint(self) -> float:
        if self.available and self.topological_model is not None:
            return 0.98
        if self.available:
            return 0.85
        return 0.72


class PygplatesAdapter:
    def __init__(self) -> None:
        try:
            import pygplates  # type: ignore

            self._pygplates = pygplates
            self._available = True
        except Exception:
            self._pygplates = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def build_cache(self, tectonics_dir: Path) -> PygplatesModelCache:
        if not self._available or self._pygplates is None:
            return PygplatesModelCache(available=False, status="pygplates_not_installed")

        rot_path = tectonics_dir / "model.rot"
        gpml_path = tectonics_dir / "model.gpml"

        if not rot_path.exists() or not gpml_path.exists():
            return PygplatesModelCache(available=True, status="tectonic_files_missing")

        try:
            rotation_model = self._pygplates.RotationModel(str(rot_path))
        except Exception:
            return PygplatesModelCache(available=True, status="rotation_model_unavailable")

        reconstruct_model = None
        topological_model = None
        reconstruct_status = ""
        topo_status = ""

        try:
            reconstruct_model = self._pygplates.ReconstructModel(str(gpml_path), rotation_model)
            reconstruct_status = "reconstruct_ready"
        except Exception:
            reconstruct_status = "reconstruct_unavailable"

        try:
            topological_model = self._pygplates.TopologicalModel(str(gpml_path), str(rot_path))
            topo_status = "topology_ready"
        except Exception:
            topo_status = "topology_unavailable"

        status = ",".join(part for part in ["pygplates_loaded", reconstruct_status, topo_status] if part)
        return PygplatesModelCache(
            available=True,
            status=status,
            rotation_model=rotation_model,
            reconstruct_model=reconstruct_model,
            topological_model=topological_model,
        )
