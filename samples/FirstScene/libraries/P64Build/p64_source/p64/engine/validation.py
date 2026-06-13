from __future__ import annotations

import ast
from pathlib import Path

from p64.engine.assets import AssetMetadata, discover_metadata
from p64.engine.components import CharacterController, Collider, EntityPhysics, MeshRenderer, ScriptComponent
from p64.engine.entity import Entity
from p64.engine.material import resolve_material_reference
from p64.engine.project import Project
from p64.engine.scene import Scene


def asset_metadata_by_id(project: Project) -> dict[str, AssetMetadata]:
    assets: dict[str, AssetMetadata] = {}
    for metadata_path in discover_metadata(project.assets_dir):
        try:
            metadata = AssetMetadata.load(metadata_path)
        except Exception:
            continue
        assets[metadata.id] = metadata
    return assets


def scene_reference_errors(project: Project, scene: Scene) -> dict[str, list[str]]:
    metadata = asset_metadata_by_id(project)
    errors: dict[str, list[str]] = {}
    for entity in scene.walk():
        own = entity_reference_errors(project, entity, metadata)
        if own:
            errors[entity.id] = own
    return errors


def entity_reference_errors(project: Project, entity: Entity, metadata: dict[str, AssetMetadata] | None = None) -> list[str]:
    metadata = metadata or asset_metadata_by_id(project)
    errors: list[str] = []
    for child in entity.children:
        if child.object_type != entity.object_type:
            errors.append("Child SceneObject type must match parent")
    for component in entity.components:
        if isinstance(component, MeshRenderer):
            if component.mesh:
                mesh = metadata.get(component.mesh)
                if mesh is None:
                    errors.append(f"Missing mesh asset: {component.mesh}")
                else:
                    if component.submesh and component.submesh not in mesh.groups:
                        errors.append(f"Missing submesh: {component.submesh}")
                    if component.material and component.material not in mesh.materials:
                        errors.append(f"Missing material: {component.material}")
            if component.shader and not (project.root / component.shader).exists():
                errors.append(f"Missing shader: {component.shader}")
            for material in component.material_slots:
                material_path = resolve_material_reference(project.root, material)
                if material_path and not material_path.exists():
                    errors.append(f"Missing material asset: {material}")
        if isinstance(component, ScriptComponent):
            for entry in component.scripts:
                if not entry.enabled:
                    continue
                if not entry.script.strip():
                    errors.append("Script entry has no script file")
                    continue
                script_path = _find_script(project, entry.script)
                if script_path is None:
                    errors.append(f"Missing script: {entry.script}")
                elif entry.class_name and entry.class_name not in _script_classes(script_path):
                    errors.append(f"Missing script class: {entry.class_name}")
                elif not entry.class_name:
                    errors.append("Script entry has no class name")
        if isinstance(component, Collider):
            if component.shape not in {"box", "sphere", "mesh"}:
                errors.append(f"Invalid collider shape: {component.shape}")
            if component.shape == "box" and (component.size.x <= 0 or component.size.y <= 0 or component.size.z <= 0):
                errors.append("Collider size must be positive")
            if component.shape == "sphere" and component.radius <= 0:
                errors.append("Collider radius must be positive")
            if component.shape == "mesh" and not component.convex and not entity.is_game_object:
                errors.append("Non-convex MeshCollider requires a GameObject")
            if component.shape == "mesh" or component.fit_to_mesh:
                renderer = _mesh_renderer(entity)
                if renderer is None or not renderer.mesh:
                    errors.append("Mesh collider needs a MeshRenderer")
                elif renderer.mesh not in metadata:
                    errors.append(f"Missing mesh asset: {renderer.mesh}")
        if isinstance(component, CharacterController):
            if component.height <= 0 or component.radius <= 0:
                errors.append("CharacterController height and radius must be positive")
        if isinstance(component, EntityPhysics):
            if component.mass <= 0:
                errors.append("EntityPhysics mass must be positive")
            if not entity.is_entity:
                errors.append("EntityPhysics requires an Entity")
    return errors


def has_reference_errors(project: Project, entity: Entity, metadata: dict[str, AssetMetadata] | None = None) -> bool:
    if entity_reference_errors(project, entity, metadata):
        return True
    return any(has_reference_errors(project, child, metadata) for child in entity.children)


def _find_script(project: Project, script: str) -> Path | None:
    for path in project.script_path_candidates(script):
        if path.exists():
            return path
    return None


def _mesh_renderer(entity: Entity) -> MeshRenderer | None:
    for component in entity.components:
        if isinstance(component, MeshRenderer):
            return component
    return None


def _script_classes(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    classes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "UserScript":
                    classes.add(node.name)
                elif isinstance(base, ast.Attribute) and base.attr == "UserScript":
                    classes.add(node.name)
    return classes
