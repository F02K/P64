from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from p64.engine.assets import AssetMetadata
from p64.engine.components import Camera, Fog, Light, MeshRenderer, ScriptComponent
from p64.engine.entity import Entity
from p64.engine.files import find_metadata_for_source, is_metadata_file
from p64.engine.obj import import_obj_to_project, parse_obj
from p64.engine.project import Project
from p64.engine.scene import Scene


@dataclass
class DirtyTracker:
    dirty: bool = False

    def mark_dirty(self) -> None:
        self.dirty = True

    def mark_saved(self) -> None:
        self.dirty = False


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

    mesh = parse_obj(obj_path)
    root = Entity(obj_path.stem)
    for group in mesh.groups:
        child = Entity(group.name)
        material = group.faces[0].material if group.faces else None
        child.add_component(MeshRenderer(mesh=metadata.id, submesh=group.name, material=material))
        root.add_child(child)
    scene.add_entity(root)
    return root


def split_mesh_renderer_into_children(entity: Entity, metadata: AssetMetadata) -> list[Entity]:
    existing = {child.name for child in entity.children}
    created: list[Entity] = []
    for group in metadata.groups:
        if group in existing:
            continue
        child = Entity(group)
        child.add_component(MeshRenderer(mesh=metadata.id, submesh=group, material=metadata.materials[0] if metadata.materials else None))
        entity.add_child(child)
        created.append(child)
    return created


def create_shader_template(assets_dir: Path, name: str = "new_shader") -> Path:
    shader_dir = assets_dir / "shaders"
    shader_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(shader_dir / f"{name}.shader")
    path.write_text(
        'Shader "P64/NewShader"\n'
        "{\n"
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
        "from p64.engine.scripting import UserScript\n\n\n"
        f"class {class_name}(UserScript):\n"
        "    def on_start(self):\n"
        "        pass\n\n"
        "    def on_update(self, dt):\n"
        "        pass\n",
        encoding="utf-8",
    )
    return path


def add_component(entity: Entity, component_name: str) -> object:
    if component_name == "MeshRenderer":
        return entity.add_component(MeshRenderer())
    if component_name == "Camera":
        return entity.add_component(Camera())
    if component_name == "Light":
        return entity.add_component(Light())
    if component_name == "Fog":
        return entity.add_component(Fog())
    if component_name == "ScriptComponent":
        return entity.add_component(ScriptComponent())
    raise ValueError(f"Unknown component: {component_name}")


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
