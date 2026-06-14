from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from p64.engine.files import iter_metadata_files


@dataclass
class AssetMetadata:
    id: str
    kind: str
    source: str
    groups: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "groups": self.groups,
            "materials": self.materials,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssetMetadata":
        return cls(
            id=str(data.get("id") or uuid4().hex),
            kind=str(data.get("kind", "unknown")),
            source=str(data.get("source", "")),
            groups=list(data.get("groups", [])),
            materials=list(data.get("materials", [])),
            settings=dict(data.get("settings", {})),
        )

    @classmethod
    def load(cls, path: Path) -> "AssetMetadata":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")


def discover_assets(assets_dir: Path) -> list[Path]:
    if not assets_dir.exists():
        return []
    return sorted(path for path in assets_dir.rglob("*") if path.is_file())


def discover_metadata(assets_dir: Path) -> list[Path]:
    return iter_metadata_files(assets_dir)


def relative_asset_path(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def model_info(metadata: AssetMetadata) -> dict[str, Any] | None:
    value = metadata.settings.get("model")
    return value if isinstance(value, dict) else None


def model_meshes(metadata: AssetMetadata) -> list[dict[str, Any]]:
    model = model_info(metadata)
    meshes = model.get("meshes", []) if model else []
    return [item for item in meshes if isinstance(item, dict)]


def model_mesh_ids(metadata: AssetMetadata) -> list[str]:
    return [str(item.get("id")) for item in model_meshes(metadata) if item.get("id")]


def model_mesh_by_id(metadata: AssetMetadata, mesh_id: str) -> dict[str, Any] | None:
    for item in model_meshes(metadata):
        if item.get("id") == mesh_id:
            return item
    return None


def model_mesh_by_name(metadata: AssetMetadata, name: str | None) -> dict[str, Any] | None:
    if not name:
        return model_meshes(metadata)[0] if model_meshes(metadata) else None
    for item in model_meshes(metadata):
        if item.get("name") == name or item.get("source_group") == name or item.get("node_path") == name:
            return item
    return None


def resolve_model_mesh(
    metadata_by_id: dict[str, AssetMetadata],
    mesh_id: str,
    legacy_submesh: str | None = None,
) -> tuple[AssetMetadata | None, dict[str, Any] | None]:
    metadata = metadata_by_id.get(mesh_id)
    if metadata is not None:
        return metadata, model_mesh_by_name(metadata, legacy_submesh)
    for candidate in metadata_by_id.values():
        mesh = model_mesh_by_id(candidate, mesh_id)
        if mesh is not None:
            return candidate, mesh
    return None, None


def safe_model_mesh_id(model_id: str, name: str, index: int = 0) -> str:
    safe = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower() or "mesh"
    suffix = f"_{index + 1}" if index else ""
    return f"mesh_{model_id}_{safe}{suffix}"
