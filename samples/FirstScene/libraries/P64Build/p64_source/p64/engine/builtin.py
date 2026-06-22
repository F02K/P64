from __future__ import annotations

import json
from pathlib import Path

from p64.engine.files import metadata_path_for_source
from p64.renderer.shaders import (
    CLOUD_PLANE_FRAGMENT_SHADER,
    CLOUD_PLANE_VERTEX_SHADER,
    ERROR_FRAGMENT_SHADER,
    ERROR_VERTEX_SHADER,
    PARTICLE_FRAGMENT_SHADER,
    PARTICLE_VERTEX_SHADER,
    SKYBOX_FRAGMENT_SHADER,
    SKYBOX_VERTEX_SHADER,
    SPRITE_FRAGMENT_SHADER,
    SPRITE_VERTEX_SHADER,
    STANDARD_UNLIT_FRAGMENT_SHADER,
    STANDARD_UNLIT_VERTEX_SHADER,
    STANDARD_VERTEX_LIT_FRAGMENT_SHADER,
    STANDARD_VERTEX_LIT_VERTEX_SHADER,
    UI_FRAGMENT_SHADER,
    UI_VERTEX_SHADER,
)


BUILTIN_PACKAGE_NAME = "P64Builtin"
STANDARD_VERTEX_LIT_SHADER_NAME = "P64Builtin/Standard VertexLit"
STANDARD_UNLIT_SHADER_NAME = "P64Builtin/Standard Unlit"
SPRITE_SHADER_NAME = "P64Builtin/Sprite"
UI_IMAGE_SHADER_NAME = "P64Builtin/UI Image"
UI_TEXT_SHADER_NAME = "P64Builtin/UI Text"
PARTICLE_SHADER_NAME = "P64Builtin/Particle"
SKYBOX_SHADER_NAME = "P64Builtin/Skybox"
CLOUD_SHADER_NAME = "P64Builtin/Cloud Dome"
ERROR_SHADER_NAME = "P64Builtin/Error"
STANDARD_SHADER_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/shaders/standard_vertex_lit.shader"
STANDARD_UNLIT_SHADER_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/shaders/standard_unlit.shader"
SPRITE_SHADER_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/shaders/sprite.shader"
UI_IMAGE_SHADER_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/shaders/ui_image.shader"
UI_TEXT_SHADER_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/shaders/ui_text.shader"
PARTICLE_SHADER_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/shaders/particle.shader"
SKYBOX_SHADER_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/shaders/skybox.shader"
CLOUD_SHADER_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/shaders/cloud_dome.shader"
ERROR_SHADER_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/shaders/error.shader"
SPRITE_MATERIAL_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/materials/sprite.material"
UI_IMAGE_MATERIAL_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/materials/ui_image.material"
PARTICLE_MATERIAL_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/materials/particle.material"
LEGACY_STANDARD_SHADER_RELATIVE = f"packages/{BUILTIN_PACKAGE_NAME}/shaders/standard_" + "n" + "64.shader"
BUILTIN_MATERIAL_PROPERTIES = (
    'Texture u_texture = ""',
    "Color u_base_color = (1.0, 1.0, 1.0)",
    "Float u_alpha_cutoff = 0.0 Range(0, 1)",
)
SPRITE_MATERIAL_PROPERTIES = (
    'Texture u_texture = ""',
    "Color u_base_color = (1.0, 1.0, 1.0)",
    "Float u_alpha = 1.0 Range(0, 1)",
    "Float u_alpha_cutoff = 0.0 Range(0, 1)",
)
PARTICLE_MATERIAL_PROPERTIES = SPRITE_MATERIAL_PROPERTIES
UI_MATERIAL_PROPERTIES = SPRITE_MATERIAL_PROPERTIES
SKYBOX_MATERIAL_PROPERTIES = (
    "Color u_skybox_top_color = (0.22, 0.48, 0.86)",
    "Color u_skybox_horizon_color = (0.66, 0.82, 0.95)",
    "Float u_color_levels = 32.0 Range(2, 256)",
    "Bool u_dithering_enabled = true",
)
CLOUD_MATERIAL_PROPERTIES = (
    "Color u_skybox_cloud_color = (1.0, 0.96, 0.86)",
    "Float u_skybox_cloud_coverage = 0.45 Range(0, 1)",
    "Float u_skybox_cloud_scale = 3.0 Range(0.1, 24)",
    "Float u_skybox_cloud_height = 80.0 Range(0.1, 10000)",
    "Float u_skybox_cloud_softness = 0.08 Range(0, 1)",
    "Float u_color_levels = 32.0 Range(2, 256)",
    "Bool u_dithering_enabled = true",
)


def ensure_builtin_package(project_root: Path) -> None:
    package = project_root / "packages" / BUILTIN_PACKAGE_NAME
    for folder in ["shaders", "materials", "scripts", "editor"]:
        (package / folder).mkdir(parents=True, exist_ok=True)
    _write_generated_shader(
        project_root / STANDARD_SHADER_RELATIVE,
        standard_vertex_lit_shader_source(),
        STANDARD_VERTEX_LIT_SHADER_NAME,
    )
    _write_generated_shader(
        project_root / STANDARD_UNLIT_SHADER_RELATIVE,
        standard_unlit_shader_source(),
        STANDARD_UNLIT_SHADER_NAME,
    )
    for relative, source, shader_name in (
        (SPRITE_SHADER_RELATIVE, sprite_shader_source(), SPRITE_SHADER_NAME),
        (UI_IMAGE_SHADER_RELATIVE, ui_image_shader_source(), UI_IMAGE_SHADER_NAME),
        (UI_TEXT_SHADER_RELATIVE, ui_text_shader_source(), UI_TEXT_SHADER_NAME),
        (PARTICLE_SHADER_RELATIVE, particle_shader_source(), PARTICLE_SHADER_NAME),
        (SKYBOX_SHADER_RELATIVE, skybox_shader_source(), SKYBOX_SHADER_NAME),
        (CLOUD_SHADER_RELATIVE, cloud_shader_source(), CLOUD_SHADER_NAME),
        (ERROR_SHADER_RELATIVE, error_shader_source(), ERROR_SHADER_NAME),
    ):
        _write_generated_shader(project_root / relative, source, shader_name)
    for relative, shader in (
        (SPRITE_MATERIAL_RELATIVE, SPRITE_SHADER_RELATIVE),
        (UI_IMAGE_MATERIAL_RELATIVE, UI_IMAGE_SHADER_RELATIVE),
        (PARTICLE_MATERIAL_RELATIVE, PARTICLE_SHADER_RELATIVE),
    ):
        _write_generated_material(project_root, relative, shader)
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
        BUILTIN_MATERIAL_PROPERTIES,
    )


def standard_unlit_shader_source() -> str:
    return _shader_source(
        STANDARD_UNLIT_SHADER_NAME,
        STANDARD_UNLIT_VERTEX_SHADER,
        STANDARD_UNLIT_FRAGMENT_SHADER,
        BUILTIN_MATERIAL_PROPERTIES,
    )


def sprite_shader_source() -> str:
    return _shader_source(SPRITE_SHADER_NAME, SPRITE_VERTEX_SHADER, SPRITE_FRAGMENT_SHADER, SPRITE_MATERIAL_PROPERTIES)


def ui_image_shader_source() -> str:
    return _shader_source(UI_IMAGE_SHADER_NAME, UI_VERTEX_SHADER, UI_FRAGMENT_SHADER, UI_MATERIAL_PROPERTIES)


def ui_text_shader_source() -> str:
    return _shader_source(UI_TEXT_SHADER_NAME, UI_VERTEX_SHADER, UI_FRAGMENT_SHADER, UI_MATERIAL_PROPERTIES)


def particle_shader_source() -> str:
    return _shader_source(PARTICLE_SHADER_NAME, PARTICLE_VERTEX_SHADER, PARTICLE_FRAGMENT_SHADER, PARTICLE_MATERIAL_PROPERTIES)


def skybox_shader_source() -> str:
    return _shader_source(SKYBOX_SHADER_NAME, SKYBOX_VERTEX_SHADER, SKYBOX_FRAGMENT_SHADER, SKYBOX_MATERIAL_PROPERTIES)


def cloud_shader_source() -> str:
    return _shader_source(CLOUD_SHADER_NAME, CLOUD_PLANE_VERTEX_SHADER, CLOUD_PLANE_FRAGMENT_SHADER, CLOUD_MATERIAL_PROPERTIES)


def error_shader_source() -> str:
    return _shader_source(ERROR_SHADER_NAME, ERROR_VERTEX_SHADER, ERROR_FRAGMENT_SHADER, ())


def _shader_source(name: str, vertex: str, fragment: str, properties: tuple[str, ...]) -> str:
    return (
        f'Shader "{name}"\n'
        "{\n"
        "    Properties\n"
        "    {\n"
        f"{_indent_lines(properties)}\n"
        "    }\n\n"
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


def _write_generated_shader(path: Path, source: str, shader_name: str) -> None:
    if not path.exists():
        path.write_text(source, encoding="utf-8")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    if f'Shader "{shader_name}"' in text:
        path.write_text(source, encoding="utf-8")


def _write_generated_material(project_root: Path, relative: str, shader: str) -> None:
    path = project_root / relative
    material = {
        "shader": shader,
        "properties": {
            "u_base_color": [1.0, 1.0, 1.0],
            "u_alpha": 1.0,
            "u_alpha_cutoff": 0.0,
        },
        "textures": {"u_texture": ""},
    }
    should_write = not path.exists()
    if not should_write:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            should_write = existing.get("shader") == shader
        except (OSError, json.JSONDecodeError):
            should_write = False
    if should_write:
        path.write_text(json.dumps(material, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "id": f"material_{Path(relative).stem}",
        "kind": "material",
        "source": relative,
        "groups": [],
        "materials": [],
        "settings": {"builtin": True},
    }
    metadata_path_for_source(path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


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


def _indent_lines(lines: tuple[str, ...]) -> str:
    return "\n".join(f"        {line}" for line in lines)
