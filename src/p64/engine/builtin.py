from __future__ import annotations

from pathlib import Path

from p64.renderer.shaders import (
    STANDARD_UNLIT_FRAGMENT_SHADER,
    STANDARD_UNLIT_VERTEX_SHADER,
    STANDARD_VERTEX_LIT_FRAGMENT_SHADER,
    STANDARD_VERTEX_LIT_VERTEX_SHADER,
)


BUILTIN_PACKAGE_NAME = "P64Builtin"
STANDARD_VERTEX_LIT_SHADER_NAME = "P64Builtin/Standard VertexLit"
STANDARD_UNLIT_SHADER_NAME = "P64Builtin/Standard Unlit"
STANDARD_SHADER_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/shaders/standard_vertex_lit.shader"
STANDARD_UNLIT_SHADER_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/shaders/standard_unlit.shader"
LEGACY_STANDARD_SHADER_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/shaders/standard_" + "n" + "64.shader"


def ensure_builtin_package(project_root: Path) -> None:
    package = project_root / "packages" / BUILTIN_PACKAGE_NAME
    for folder in ["shaders", "materials", "scripts", "editor"]:
        (package / folder).mkdir(parents=True, exist_ok=True)
    _write_if_missing(project_root / STANDARD_SHADER_RELATIVE, standard_vertex_lit_shader_source())
    _write_if_missing(project_root / STANDARD_UNLIT_SHADER_RELATIVE, standard_unlit_shader_source())
    _remove_generated_legacy_shader(project_root / LEGACY_STANDARD_SHADER_RELATIVE)


def normalize_shader_reference(shader: str | None) -> str | None:
    if shader in {LEGACY_STANDARD_SHADER_RELATIVE, "packages/P64Builtin/shaders/standard_" + "n" + "64.shader"}:
        return STANDARD_SHADER_RELATIVE
    return shader


def standard_vertex_lit_shader_source() -> str:
    return _shader_source(
        STANDARD_VERTEX_LIT_SHADER_NAME,
        STANDARD_VERTEX_LIT_VERTEX_SHADER,
        STANDARD_VERTEX_LIT_FRAGMENT_SHADER,
    )


def standard_unlit_shader_source() -> str:
    return _shader_source(
        STANDARD_UNLIT_SHADER_NAME,
        STANDARD_UNLIT_VERTEX_SHADER,
        STANDARD_UNLIT_FRAGMENT_SHADER,
    )


def _shader_source(name: str, vertex: str, fragment: str) -> str:
    return (
        f'Shader "{name}"\n'
        "{\n"
        "    Vertex\n"
        "    {\n"
        f"{_indent(vertex.strip())}\n"
        "    }\n\n"
        "    Fragment\n"
        "    {\n"
        f"{_indent(fragment.strip())}\n"
        "    }\n"
        "}\n"
    )


def _write_if_missing(path: Path, source: str) -> None:
    if not path.exists():
        path.write_text(source, encoding="utf-8")


def _remove_generated_legacy_shader(path: Path) -> None:
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    if "P64Builtin/Standard" + "N" + "64" in text:
        path.unlink()


def _indent(text: str) -> str:
    return "\n".join(f"        {line}" for line in text.splitlines())
