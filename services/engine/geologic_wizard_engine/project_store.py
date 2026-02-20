from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .utils import ensure_dir


class ProjectStore:
    def __init__(self, data_root: Path):
        self.data_root = ensure_dir(data_root)
        self.projects_root = ensure_dir(self.data_root / "projects")

    def project_dir(self, project_id: str) -> Path:
        return self.projects_root / project_id

    def initialize_project(self, project_id: str, config_json: dict[str, Any]) -> Path:
        root = ensure_dir(self.project_dir(project_id))
        ensure_dir(root / "runs")
        ensure_dir(root / "working")
        ensure_dir(root / "exports")
        ensure_dir(root / "cache" / "preview")
        ensure_dir(root / "cache" / "strain")
        ensure_dir(root / "cache" / "refined")
        ensure_dir(root / "tectonics")
        self.write_json(root / "working" / "config.json", config_json)
        # Placeholders for GPML/ROT interoperability files.
        (root / "tectonics" / "model.gpml").write_text("<!-- GPML placeholder -->\n", encoding="utf-8")
        (root / "tectonics" / "model.rot").write_text("# ROT placeholder\n", encoding="utf-8")
        return root

    def run_dir(self, project_id: str, run_id: str) -> Path:
        run_root = ensure_dir(self.project_dir(project_id) / "runs" / run_id)
        ensure_dir(run_root / "frames")
        ensure_dir(run_root / "render_frames")
        ensure_dir(run_root / "diagnostics")
        return run_root

    def frame_path(self, project_id: str, run_id: str, time_ma: int) -> Path:
        return self.run_dir(project_id, run_id) / "frames" / f"{time_ma}.json"

    def render_frame_path(self, project_id: str, run_id: str, time_ma: int) -> Path:
        return self.run_dir(project_id, run_id) / "render_frames" / f"{time_ma}.json"

    def timeline_index_path(self, project_id: str, run_id: str) -> Path:
        return self.run_dir(project_id, run_id) / "timeline_index.json"

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        ensure_dir(path.parent)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def write_frame(self, project_id: str, run_id: str, time_ma: int, frame_payload: dict[str, Any]) -> Path:
        path = self.frame_path(project_id, run_id, time_ma)
        self.write_json(path, frame_payload)
        return path

    def read_frame(self, project_id: str, run_id: str, time_ma: int) -> dict[str, Any] | None:
        path = self.frame_path(project_id, run_id, time_ma)
        if not path.exists():
            return None
        return self.read_json(path)

    def write_render_frame(self, project_id: str, run_id: str, time_ma: int, render_payload: dict[str, Any]) -> Path:
        path = self.render_frame_path(project_id, run_id, time_ma)
        self.write_json(path, render_payload)
        return path

    def read_render_frame(self, project_id: str, run_id: str, time_ma: int) -> dict[str, Any] | None:
        path = self.render_frame_path(project_id, run_id, time_ma)
        if not path.exists():
            return None
        return self.read_json(path)

    def list_frame_times(self, project_id: str, run_id: str) -> list[int]:
        frames_root = self.run_dir(project_id, run_id) / "frames"
        times: list[int] = []
        for path in frames_root.glob("*.json"):
            try:
                times.append(int(path.stem))
            except ValueError:
                continue
        return sorted(times, reverse=True)

    def preview_array_path(self, project_id: str, time_ma: int) -> Path:
        return self.project_dir(project_id) / "cache" / "preview" / f"{time_ma}.npy"

    def strain_array_path(self, project_id: str, time_ma: int) -> Path:
        return self.project_dir(project_id) / "cache" / "strain" / f"{time_ma}.npy"

    def refined_array_path(self, project_id: str, bookmark_id: str, refinement_level: int) -> Path:
        return self.project_dir(project_id) / "cache" / "refined" / f"{bookmark_id}_L{refinement_level}.npy"

    def write_array(self, path: Path, arr: np.ndarray) -> Path:
        ensure_dir(path.parent)
        np.save(path, arr)
        return path

    def read_array(self, path: Path) -> np.ndarray:
        return np.load(path)

    def export_path(self, project_id: str, filename: str) -> Path:
        return self.project_dir(project_id) / "exports" / filename

    def metadata_path(self, project_id: str, artifact_id: str) -> Path:
        return self.project_dir(project_id) / "exports" / f"{artifact_id}.metadata.json"

    def run_manifest_path(self, project_id: str, run_id: str) -> Path:
        return self.run_dir(project_id, run_id) / "manifest.json"

    def run_diagnostics_path(self, project_id: str, run_id: str, time_ma: int) -> Path:
        return self.run_dir(project_id, run_id) / "diagnostics" / f"{time_ma}.json"

    def write_run_diagnostics(self, project_id: str, run_id: str, time_ma: int, payload: dict[str, Any]) -> Path:
        path = self.run_diagnostics_path(project_id, run_id, time_ma)
        self.write_json(path, payload)
        return path

    def read_run_diagnostics(self, project_id: str, run_id: str, time_ma: int) -> dict[str, Any] | None:
        path = self.run_diagnostics_path(project_id, run_id, time_ma)
        if not path.exists():
            return None
        return self.read_json(path)
