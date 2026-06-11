from __future__ import annotations

import json
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
