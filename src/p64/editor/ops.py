from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

from p64.engine.audio import import_audio_clip
from p64.engine.assets import AssetMetadata, model_meshes
from p64.engine.builtin import PARTICLE_MATERIAL_RELATIVE, SPRITE_MATERIAL_RELATIVE, UI_IMAGE_MATERIAL_RELATIVE
from p64.engine.collision import apply_mesh_primitive_defaults
from p64.engine.components import AudioListener, AudioSource, Camera, Canvas, CharacterController, Collider, EntityPhysics, Light, MeshRenderer, ModelRenderer, ParticleEmitter, RectTransform, ScriptComponent, SpawnPoint, SpriteRenderer, UIButton, UIControl, UIImage, UIScrollView, UISlider, UIText, UIToggle
from p64.engine.entity import ENTITY, GAME_OBJECT, Entity, entity_under_canvas
from p64.engine.files import find_metadata_for_source, is_lighting_file, is_metadata_file, is_scene_file, iter_metadata_files, metadata_path_for_source
from p64.engine.lighting import lighting_path_for_scene
from p64.engine.material import (
    create_material_from_defaults,
    material_asset_id,
    material_reference,
    reset_material_from_metadata,
    sanitize_material_name,
    save_material_metadata,
    resolve_material_reference,
)
from p64.engine.math import Vec3
from p64.engine.obj import import_obj_to_project
from p64.engine.project import Project
from p64.engine.scene import Scene


@dataclass
class DirtyTracker:
    dirty: bool = False

    def mark_dirty(self) -> None:
        self.dirty = True

    def mark_saved(self) -> None:
        self.dirty = False


class AssetOperationError(ValueError):
    pass


def duplicate_entity(entity: Entity) -> Entity:
    data = entity.to_dict()
    duplicate = Entity.from_dict(data)
    _regenerate_ids(duplicate)
    duplicate.name = f"{entity.name} Copy"
    return duplicate


def delete_entity(scene: Scene, entity_id: str) -> bool:
    for entity in scene.entities:
        if entity.id == entity_id:
            scene.entities.remove(entity)
            return True
        if _delete_from_children(entity, entity_id):
            return True
    return False


def find_parent(scene: Scene, entity_id: str) -> Entity | None:
    for entity in scene.entities:
        for child in entity.children:
            if child.id == entity_id:
                return entity
        found = _find_parent_in_children(entity, entity_id)
        if found:
            return found
    return None


def insert_obj_scene_entity(project: Project, scene: Scene, obj_or_metadata: Path) -> Entity:
    if is_metadata_file(obj_or_metadata):
        metadata = AssetMetadata.load(obj_or_metadata)
        obj_path = project.root / metadata.source
    else:
        metadata_path = find_metadata_for_source(obj_or_metadata)
        if metadata_path and metadata_path.exists():
            metadata = AssetMetadata.load(metadata_path)
        else:
            metadata = import_obj_to_project(project, obj_or_metadata, add_to_startup_scene=False)
        obj_path = project.root / metadata.source

    root = Entity(obj_path.stem, object_type=GAME_OBJECT)
    root.add_component(
        ModelRenderer(
            model=metadata.id,
            source_materials=_source_materials_for(metadata),
            material_slots=_material_slots_for(metadata),
        )
    )
    scene.add_entity(root)
    return root


def split_mesh_renderer_into_children(entity: Entity, metadata: AssetMetadata) -> list[Entity]:
    existing = {child.name for child in entity.children}
    created: list[Entity] = []
    for mesh_entry in _mesh_entries_for_metadata(metadata):
        name = str(mesh_entry.get("name") or "Mesh")
        if name in existing:
            continue
        source_materials = [str(item) for item in mesh_entry.get("material_slots", [])]
        material = source_materials[0] if source_materials else None
        child = Entity(name, object_type=entity.object_type)
        child.add_component(
            MeshRenderer(
                mesh=str(mesh_entry.get("id") or ""),
                submesh=mesh_entry.get("legacy_submesh"),
                material=material,
                source_materials=source_materials,
                material_slots=_material_slots_for(metadata, source_materials),
            )
        )
        entity.add_child(child)
        created.append(child)
    return created


def extract_materials_for_obj(project: Project, obj_or_metadata: Path, output_dir: Path | None = None) -> list[Path]:
    metadata_path = obj_or_metadata if is_metadata_file(obj_or_metadata) else find_metadata_for_source(obj_or_metadata)
    if metadata_path is None or not metadata_path.exists():
        metadata = import_obj_to_project(project, obj_or_metadata, add_to_startup_scene=False)
        metadata_path = metadata_path_for_source(project.root / metadata.source)
    else:
        metadata = AssetMetadata.load(metadata_path)
    obj_path = project.root / metadata.source
    material_defs = metadata.settings.get("material_defs", {})
    material_assets = dict(metadata.settings.get("material_assets", {}))
    output_dir = output_dir or project.assets_dir / "materials" / obj_path.stem
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    created_or_existing: list[Path] = []
    for material_name in metadata.materials:
        filename = sanitize_material_name(material_name) + ".material"
        material_path = output_dir / filename
        if not material_path.exists():
            create_material_from_defaults(project.root, material_path, material_defs.get(material_name, {}))
        save_material_metadata(
            project.root,
            material_path,
            defaults=material_defs.get(material_name, {}),
            source={"mesh": metadata.id, "obj": metadata.source, "mtl": material_name},
        )
        material_assets[material_name] = material_asset_id(project.root, material_path)
        created_or_existing.append(material_path)
    metadata.settings["material_assets"] = material_assets
    metadata.settings["material_extract_folder"] = material_reference(project.root, output_dir)
    metadata.save(metadata_path)
    return created_or_existing


def reset_material_asset(project: Project, material_path: Path) -> None:
    reset_material_from_metadata(project.root, material_path)


def import_audio_asset(project: Project, path: Path) -> AssetMetadata:
    return import_audio_clip(project, path)


def update_material_usage_cache(project: Project, scene: Scene, scene_path: Path | None = None) -> None:
    usage: dict[str, list[dict[str, str]]] = {}
    scene_id = scene_path.resolve().relative_to(project.root.resolve()).as_posix() if scene_path else scene.name
    for entity in scene.walk():
        for component in entity.components:
            if not isinstance(component, MeshRenderer):
                continue
            for slot in component.material_slots:
                if not slot:
                    continue
                usage.setdefault(str(slot), []).append({
                    "scene": scene_id,
                    "entity_id": entity.id,
                    "entity": entity.name,
                    "mesh": component.mesh,
                    "submesh": component.submesh or "",
                })
    for material_id, items in usage.items():
        material_path = resolve_material_reference(project.root, material_id)
        if material_path and material_path.exists():
            save_material_metadata(project.root, material_path, usage_cache=items)


def move_component(entity: Entity, component: object, direction: int) -> bool:
    if component not in entity.components:
        return False
    index = entity.components.index(component)
    target = index + direction
    if target < 0 or target >= len(entity.components):
        return False
    entity.components[index], entity.components[target] = entity.components[target], entity.components[index]
    return True


def move_script_entry(component: ScriptComponent, index: int, direction: int) -> bool:
    target = index + direction
    if index < 0 or index >= len(component.scripts) or target < 0 or target >= len(component.scripts):
        return False
    component.scripts[index], component.scripts[target] = component.scripts[target], component.scripts[index]
    return True


def asset_path_is_editable(project: Project, path: Path) -> bool:
    return _is_relative_to(path.resolve(), project.assets_dir.resolve())


def create_asset_folder(project: Project, parent: Path, name: str = "New Folder") -> Path:
    folder = _editable_asset_folder(project, parent)
    path = _unique_path(folder / _validate_asset_name(name))
    path.mkdir()
    return path


def create_blank_asset_file(project: Project, parent: Path, name: str = "new_file.txt") -> Path:
    folder = _editable_asset_folder(project, parent)
    path = _unique_path(folder / _validate_asset_name(name))
    path.write_text("", encoding="utf-8")
    return path


def rename_asset_path(project: Project, path: Path, new_name: str) -> Path:
    source = _editable_asset_path(project, path)
    if is_lighting_file(source):
        raise AssetOperationError("Lighting assets are renamed together with their Scene")
    name = _validate_asset_name(new_name)
    destination = source.with_name(name)
    if destination == source:
        return source
    _ensure_under_assets(project, destination)
    if destination.exists():
        raise AssetOperationError(f"Asset already exists: {destination.name}")
    if is_scene_file(source) and lighting_path_for_scene(destination).exists():
        raise AssetOperationError(f"Lighting asset already exists: {lighting_path_for_scene(destination).name}")
    old_relative = _relative_to_project(project, source)
    new_relative = _relative_to_project(project, destination)
    was_dir = source.is_dir()
    source.rename(destination)
    if is_scene_file(destination):
        old_lighting = lighting_path_for_scene(source)
        new_lighting = lighting_path_for_scene(destination)
        if old_lighting.exists():
            old_lighting.rename(new_lighting)
    if was_dir:
        _update_metadata_sources_after_folder_rename(project, destination, old_relative, new_relative)
    elif not is_metadata_file(destination):
        _move_source_metadata(project, source, destination)
    return destination


def delete_asset_path(project: Project, path: Path) -> None:
    source = _editable_asset_path(project, path)
    if is_lighting_file(source):
        raise AssetOperationError("Lighting assets are deleted together with their Scene")
    if source.is_dir():
        shutil.rmtree(source)
        return
    source.unlink()
    if is_scene_file(source):
        lighting = lighting_path_for_scene(source)
        if lighting.exists():
            lighting.unlink()
    if not is_metadata_file(source):
        metadata = find_metadata_for_source(source)
        if metadata and asset_path_is_editable(project, metadata):
            metadata.unlink()


def project_relative_asset_path(project: Project, path: Path) -> str:
    return _relative_to_project(project, path)


def is_project_startup_scene(project: Project, path: Path) -> bool:
    return project_relative_asset_path(project, path) == project.startup_scene


def update_startup_scene_after_asset_rename(project: Project, old_path: Path, new_path: Path) -> bool:
    if not is_project_startup_scene(project, old_path):
        return False
    project.startup_scene = project_relative_asset_path(project, new_path)
    project.save()
    return True


def duplicate_scene_asset(project: Project, source: Path) -> tuple[Path, Path]:
    source = _editable_asset_path(project, source)
    if not is_scene_file(source):
        raise AssetOperationError("Only Scene assets can be duplicated")
    destination = _unique_path(source.with_name(f"{source.stem}_copy{source.suffix}"))
    shutil.copy2(source, destination)
    source_lighting = lighting_path_for_scene(source)
    destination_lighting = lighting_path_for_scene(destination)
    if source_lighting.exists():
        shutil.copy2(source_lighting, destination_lighting)
    else:
        Scene.load(source).save(destination)
    return destination, destination_lighting


def create_shader_template(assets_dir: Path, name: str = "new_shader") -> Path:
    shader_dir = assets_dir / "shaders"
    shader_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(shader_dir / f"{name}.shader")
    path.write_text(
        'Shader "P64/NewShader"\n'
        "{\n"
        "    Properties\n"
        "    {\n"
        "        Texture u_texture = \"\"\n"
        "        Color u_base_color = (1.0, 1.0, 1.0)\n"
        "    }\n\n"
        "    Vertex\n"
        "    {\n"
        "        #version 330\n"
        "        in vec3 in_position;\n"
        "        in vec2 in_uv;\n"
        "        in vec3 in_normal;\n"
        "        uniform mat4 u_model;\n"
        "        uniform mat4 u_view;\n"
        "        uniform mat4 u_projection;\n"
        "        out vec2 v_uv;\n"
        "        void main() {\n"
        "            v_uv = in_uv;\n"
        "            gl_Position = u_projection * u_view * u_model * vec4(in_position, 1.0);\n"
        "        }\n"
        "    }\n\n"
        "    Fragment\n"
        "    {\n"
        "        #version 330\n"
        "        uniform sampler2D u_texture;\n"
        "        in vec2 v_uv;\n"
        "        out vec4 fragColor;\n"
        "        void main() {\n"
        "            fragColor = texture(u_texture, v_uv);\n"
        "        }\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    return path


def create_script_template(scripts_dir: Path, class_name: str = "NewScript") -> Path:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(scripts_dir / f"{class_name.lower()}.py")
    path.write_text(
        "from p64.engine.scripting import GameScript\n\n\n"
        f"class {class_name}(GameScript):\n"
        "    def on_start(self) -> None:\n"
        "        pass\n\n"
        "    def on_update(self, dt: float) -> None:\n"
        "        pass\n",
        encoding="utf-8",
    )
    return path


def open_script_in_vscode_project(project: Project, script_path: Path, fallback_open: Callable[[Path], None] | None = None) -> str | None:
    code_command = shutil.which("code") or shutil.which("code.cmd")
    if code_command:
        subprocess.Popen([
            code_command,
            "-r",
            str(project.root.resolve()),
            "--goto",
            f"{script_path.resolve()}:1:1",
        ])
        return None
    if fallback_open is not None:
        fallback_open(project.root)
    return "VSCode command 'code' was not found. Opened the project folder instead."


def add_component(entity: Entity, component_name: str, project: Project | None = None) -> object:
    if component_name in {"UIButton", "UIToggle", "UISlider", "UIScrollView"} and any(isinstance(component, UIControl) for component in entity.components):
        raise ValueError("An entity can only have one interactive UI control")
    if component_name == "MeshRenderer":
        return entity.add_component(MeshRenderer())
    if component_name == "ModelRenderer":
        return entity.add_component(ModelRenderer())
    if component_name == "SpriteRenderer":
        return entity.add_component(SpriteRenderer(material=SPRITE_MATERIAL_RELATIVE))
    if component_name == "Canvas":
        return entity.add_component(Canvas())
    if component_name == "UIImage":
        _ensure_ui_rect_transform(entity, Vec3(128.0, 128.0, 0.0))
        return entity.add_component(UIImage(material=UI_IMAGE_MATERIAL_RELATIVE))
    if component_name == "UIText":
        _ensure_ui_rect_transform(entity, Vec3(240.0, 40.0, 0.0))
        return entity.add_component(UIText())
    if component_name == "UIButton":
        _ensure_ui_rect_transform(entity)
        return entity.add_component(UIButton())
    if component_name == "UIToggle":
        _ensure_ui_rect_transform(entity)
        return entity.add_component(UIToggle())
    if component_name == "UISlider":
        _ensure_ui_rect_transform(entity, Vec3(240.0, 32.0, 0.0))
        return entity.add_component(UISlider())
    if component_name == "UIScrollView":
        _ensure_ui_rect_transform(entity, Vec3(320.0, 240.0, 0.0))
        return entity.add_component(UIScrollView())
    if component_name == "ParticleEmitter":
        entity.object_type = ENTITY
        return entity.add_component(ParticleEmitter(material=PARTICLE_MATERIAL_RELATIVE))
    if component_name == "Camera":
        return entity.add_component(Camera())
    if component_name == "Light":
        return entity.add_component(Light())
    if component_name == "AudioSource":
        return entity.add_component(AudioSource())
    if component_name == "AudioListener":
        return entity.add_component(AudioListener())
    if component_name == "ScriptComponent":
        entity.object_type = ENTITY
        return entity.add_component(ScriptComponent())
    if component_name == "SpawnPoint":
        return entity.add_component(SpawnPoint())
    if component_name == "Collider":
        collider = Collider()
        apply_mesh_primitive_defaults(project, entity, collider)
        return entity.add_component(collider)
    if component_name == "CharacterController":
        entity.object_type = ENTITY
        return entity.add_component(CharacterController())
    if component_name == "EntityPhysics":
        entity.object_type = ENTITY
        return entity.add_component(EntityPhysics())
    raise ValueError(f"Unknown component: {component_name}")


def _ensure_ui_rect_transform(entity: Entity, size: Vec3 | None = None) -> None:
    if entity.rect_transform is None and entity_under_canvas(entity):
        entity.rect_transform = RectTransform(size=size or Vec3(160.0, 48.0, 0.0))


def _regenerate_ids(entity: Entity) -> None:
    entity.id = uuid4().hex
    for child in entity.children:
        _regenerate_ids(child)


def _delete_from_children(parent: Entity, entity_id: str) -> bool:
    for child in parent.children:
        if child.id == entity_id:
            parent.children.remove(child)
            return True
        if _delete_from_children(child, entity_id):
            return True
    return False


def _find_parent_in_children(parent: Entity, entity_id: str) -> Entity | None:
    for child in parent.children:
        if child.id == entity_id:
            return parent
        found = _find_parent_in_children(child, entity_id)
        if found:
            return found
    return None


def _editable_asset_folder(project: Project, path: Path) -> Path:
    folder = path.resolve()
    if not folder.is_dir():
        raise AssetOperationError(f"Not a folder: {path}")
    _ensure_under_assets(project, folder)
    return folder


def _editable_asset_path(project: Project, path: Path) -> Path:
    source = path.resolve()
    if not source.exists():
        raise AssetOperationError(f"Asset does not exist: {path}")
    _ensure_under_assets(project, source)
    return source


def _ensure_under_assets(project: Project, path: Path) -> None:
    if not _is_relative_to(path.resolve(), project.assets_dir.resolve()):
        raise AssetOperationError("Asset edits are only allowed under the project's assets folder.")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_asset_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise AssetOperationError("Asset name cannot be empty.")
    if cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned or Path(cleaned).name != cleaned:
        raise AssetOperationError("Asset name cannot contain path separators.")
    return cleaned


def _relative_to_project(project: Project, path: Path) -> str:
    return path.resolve().relative_to(project.root.resolve()).as_posix()


def _move_source_metadata(project: Project, old_source: Path, new_source: Path) -> None:
    old_metadata = find_metadata_for_source(old_source)
    if not old_metadata or not old_metadata.exists():
        return
    new_metadata = metadata_path_for_source(new_source)
    if new_metadata.exists():
        raise AssetOperationError(f"Asset metadata already exists: {new_metadata.name}")
    metadata = AssetMetadata.load(old_metadata)
    metadata.source = _relative_to_project(project, new_source)
    old_metadata.rename(new_metadata)
    metadata.save(new_metadata)


def _update_metadata_sources_after_folder_rename(project: Project, renamed_folder: Path, old_relative: str, new_relative: str) -> None:
    old_prefix = old_relative.rstrip("/") + "/"
    new_prefix = new_relative.rstrip("/") + "/"
    for metadata_path in iter_metadata_files(renamed_folder):
        metadata = AssetMetadata.load(metadata_path)
        if metadata.source == old_relative:
            metadata.source = new_relative
        elif metadata.source.startswith(old_prefix):
            metadata.source = new_prefix + metadata.source[len(old_prefix):]
        else:
            source_path = project.root / metadata.source
            if _is_relative_to(source_path.resolve(), renamed_folder.resolve()):
                metadata.source = _relative_to_project(project, source_path)
        metadata.save(metadata_path)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique path for {path}")


def _source_materials_for(metadata: AssetMetadata) -> list[str]:
    return list(metadata.materials)


def _mesh_entries_for_metadata(metadata: AssetMetadata) -> list[dict[str, object]]:
    meshes = model_meshes(metadata)
    if meshes:
        return meshes
    return [
        {
            "id": metadata.id,
            "name": group,
            "source_group": group,
            "legacy_submesh": group,
            "material_slots": list(metadata.materials),
        }
        for group in metadata.groups
    ]


def _material_slots_for(metadata: AssetMetadata, source_materials: list[str] | None = None) -> list[str | None]:
    source_materials = source_materials or _source_materials_for(metadata)
    material_assets = metadata.settings.get("material_assets", {})
    slots: list[str | None] = []
    for material in source_materials:
        if isinstance(material_assets, dict) and material_assets.get(material):
            slots.append(str(material_assets[material]))
        else:
            slots.append(None)
    return slots
