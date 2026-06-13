from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from p64.engine.builtin import STANDARD_SHADER_RELATIVE, normalize_shader_reference
from p64.renderer.shaders import STANDARD_VERTEX_LIT_FRAGMENT_SHADER, STANDARD_VERTEX_LIT_VERTEX_SHADER


@dataclass(slots=True)
class ShaderSource:
    name: str
    vertex: str
    fragment: str
    properties: list["ShaderProperty"] = field(default_factory=list)


@dataclass(slots=True)
class ShaderProperty:
    kind: str
    name: str
    default: object = None
    minimum: float | None = None
    maximum: float | None = None


def parse_shader(path: Path) -> ShaderSource:
    text = path.read_text(encoding="utf-8")
    name = _parse_name(text) or path.stem
    properties = _parse_properties(text)
    vertex = _parse_block(text, "Vertex") or _parse_colon_section(text, "vertex") or STANDARD_VERTEX_LIT_VERTEX_SHADER
    fragment = _parse_block(text, "Fragment") or _parse_colon_section(text, "fragment") or STANDARD_VERTEX_LIT_FRAGMENT_SHADER
    return ShaderSource(name=name, vertex=vertex.strip() + "\n", fragment=fragment.strip() + "\n", properties=properties)


def discover_shaders(assets_dir: Path, packages_dir: Path | None = None) -> list[Path]:
    roots = [assets_dir]
    if packages_dir is None:
        inferred = assets_dir.parent / "packages"
        if inferred.exists():
            roots.append(inferred)
    else:
        roots.append(packages_dir)
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(root.rglob("*.shader"))
    return sorted(set(paths))


def shader_asset_id(project_root: Path, shader_path: Path) -> str:
    return shader_path.resolve().relative_to(project_root.resolve()).as_posix()


def default_shader_id() -> str:
    return STANDARD_SHADER_RELATIVE


def normalize_shader_id(shader: str | None) -> str | None:
    return normalize_shader_reference(shader)


def _parse_name(text: str) -> str | None:
    match = re.search(r'\bShader\s+"([^"]+)"', text)
    return match.group(1) if match else None


def _parse_block(text: str, block_name: str) -> str | None:
    match = re.search(rf"\b{block_name}\s*\{{", text, flags=re.IGNORECASE)
    if not match:
        return None
    start = match.end()
    depth = 1
    index = start
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
        index += 1
    return None


def _parse_colon_section(text: str, section: str) -> str | None:
    pattern = re.compile(rf"^\s*{section}\s*:\s*$", flags=re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    start = match.end()
    next_match = re.search(r"^\s*(vertex|fragment)\s*:\s*$", text[start:], flags=re.IGNORECASE | re.MULTILINE)
    end = start + next_match.start() if next_match else len(text)
    return text[start:end]


def _parse_properties(text: str) -> list[ShaderProperty]:
    block = _parse_block(text, "Properties")
    if not block:
        return []
    properties: list[ShaderProperty] = []
    for raw_line in block.splitlines():
        line = raw_line.strip().rstrip(";")
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        match = re.match(r"^(Color|Float|Int|Bool|Texture)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*(.+?))?\s*$", line, flags=re.IGNORECASE)
        if not match:
            continue
        kind = match.group(1).lower()
        name = match.group(2)
        default_text = match.group(3)
        minimum = maximum = None
        if default_text:
            range_match = re.search(r"\bRange\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)\s*$", default_text, flags=re.IGNORECASE)
            if range_match:
                minimum = float(range_match.group(1))
                maximum = float(range_match.group(2))
                default_text = default_text[: range_match.start()].strip()
        properties.append(ShaderProperty(kind=kind, name=name, default=_property_default(kind, default_text), minimum=minimum, maximum=maximum))
    return properties


def _property_default(kind: str, text: str | None) -> object:
    if text is None or text == "":
        if kind == "color":
            return [1.0, 1.0, 1.0]
        if kind == "texture":
            return ""
        if kind == "bool":
            return False
        if kind == "int":
            return 0
        return 0.0
    text = text.strip()
    if kind == "color":
        values = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        return [float(value) for value in values[:3]] if len(values) >= 3 else [1.0, 1.0, 1.0]
    if kind == "bool":
        return text.lower() in {"true", "1", "yes", "on"}
    if kind == "int":
        try:
            return int(float(text))
        except ValueError:
            return 0
    if kind == "texture":
        return text.strip('"')
    try:
        return float(text)
    except ValueError:
        return 0.0
