from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from p64.engine.assets import AssetMetadata
from p64.engine.builtin import STANDARD_SHADER_RELATIVE, normalize_shader_reference
from p64.engine.files import find_metadata_for_source, metadata_path_for_source
from p64.engine.shader import parse_shader


MATERIAL_SUFFIX = ".material"


@dataclass(slots=True)
class MaterialAsset:
    shader: str = STANDARD_SHADER_RELATIVE
    properties: dict[str, Any] = field(default_factory=dict)
    textures: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shader": self.shader,
            "properties": self.properties,
            "textures": self.textures,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaterialAsset":
        return cls(
            shader=normalize_shader_reference(str(data.get("shader") or STANDARD_SHADER_RELATIVE)) or STANDARD_SHADER_RELATIVE,
            properties=dict(data.get("properties", {})),
            textures=dict(data.get("textures", {})),
        )

    @classmethod
    def load(cls, path: Path) -> "MaterialAsset":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")


def is_material_file(path: Path) -> bool:
    return path.name.lower().endswith(MATERIAL_SUFFIX)


def material_defaults_from_shader(project_root: Path, shader: str | None) -> tuple[dict[str, Any], dict[str, str]]:
    shader_path = project_root / (normalize_shader_reference(shader) or STANDARD_SHADER_RELATIVE)
    properties: dict[str, Any] = {}
    textures: dict[str, str] = {}
    if not shader_path.exists():
        return properties, textures
    try:
        source = parse_shader(shader_path)
    except Exception:
        return properties, textures
    for prop in source.properties:
        if prop.kind == "texture":
            textures[prop.name] = str(prop.default or "")
        else:
            properties[prop.name] = prop.default
    return properties, textures


def create_material_from_defaults(
    project_root: Path,
    path: Path,
    defaults: dict[str, Any] | None = None,
    shader: str | None = None,
) -> MaterialAsset:
    defaults = defaults or {}
    material = MaterialAsset(shader=normalize_shader_reference(shader) or STANDARD_SHADER_RELATIVE)
    shader_properties, shader_textures = material_defaults_from_shader(project_root, material.shader)
    material.properties.update(shader_properties)
    material.textures.update(shader_textures)
    diffuse = defaults.get("diffuse_color")
    if isinstance(diffuse, list):
        material.properties["u_base_color"] = diffuse[:3]
    elif isinstance(diffuse, tuple):
        material.properties["u_base_color"] = list(diffuse[:3])
    diffuse_texture = defaults.get("diffuse_texture")
    if diffuse_texture:
        material.textures["u_texture"] = str(diffuse_texture)
    material.save(path)
    return material


def material_sidecar(path: Path) -> Path:
    return metadata_path_for_source(path)


def load_material_metadata(path: Path) -> AssetMetadata | None:
    metadata_path = find_metadata_for_source(path)
    if not metadata_path:
        return None
    try:
        return AssetMetadata.load(metadata_path)
    except Exception:
        return None


def save_material_metadata(
    project_root: Path,
    material_path: Path,
    defaults: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    usage_cache: list[dict[str, Any]] | None = None,
) -> AssetMetadata:
    relative = material_reference(project_root, material_path)
    existing = load_material_metadata(material_path)
    metadata = existing or AssetMetadata(id=f"material_{_slug(relative[: -len(MATERIAL_SUFFIX)])}", kind="material", source=relative)
    metadata.kind = "material"
    metadata.source = relative
    settings = dict(metadata.settings)
    if defaults is not None:
        settings["defaults"] = defaults
    if source is not None:
        settings["source"] = source
    if usage_cache is not None:
        settings["usage_cache"] = usage_cache
    metadata.settings = settings
    metadata.save(material_sidecar(material_path))
    return metadata


def reset_material_from_metadata(project_root: Path, material_path: Path) -> MaterialAsset:
    metadata = load_material_metadata(material_path)
    defaults = metadata.settings.get("defaults", {}) if metadata else {}
    shader = MaterialAsset.load(material_path).shader if material_path.exists() else STANDARD_SHADER_RELATIVE
    return create_material_from_defaults(project_root, material_path, defaults, shader)


def material_asset_id(project_root: Path, material_path: Path) -> str:
    return material_reference(project_root, material_path)


def material_reference(project_root: Path, material_path: Path) -> str:
    resolved = material_path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_material_reference(project_root: Path, material: str | None) -> Path | None:
    if not material:
        return None
    path = Path(str(material))
    return path if path.is_absolute() else project_root / path


def sanitize_material_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_. -]+", "_", name.strip())
    cleaned = cleaned.strip(" .")
    return cleaned or "Material"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "material"
