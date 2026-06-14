from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from p64.engine.assets import AssetMetadata, model_meshes, safe_model_mesh_id
from p64.engine.components import MeshRenderer
from p64.engine.entity import GAME_OBJECT, Entity
from p64.engine.files import metadata_path_for_source
from p64.engine.project import Project


@dataclass
class ObjVertex:
    position: tuple[float, float, float]
    texcoord: tuple[float, float] | None = None
    normal: tuple[float, float, float] | None = None
    color: tuple[float, float, float] | None = None


@dataclass
class ObjFace:
    vertices: list[ObjVertex]
    material: str | None = None


@dataclass
class ObjGroup:
    name: str
    faces: list[ObjFace] = field(default_factory=list)


@dataclass
class ObjMesh:
    source: Path
    groups: list[ObjGroup]
    materials: list[str]
    mtllibs: list[str] = field(default_factory=list)
    material_defs: dict[str, "MtlMaterial"] = field(default_factory=dict)

    @property
    def group_names(self) -> list[str]:
        return [group.name for group in self.groups]


@dataclass
class MtlMaterial:
    name: str
    diffuse_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    diffuse_texture: str | None = None
    specular_texture: str | None = None
    bump_texture: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "diffuse_color": list(self.diffuse_color),
            "diffuse_texture": self.diffuse_texture,
            "specular_texture": self.specular_texture,
            "bump_texture": self.bump_texture,
        }


def parse_mtl(path: Path) -> dict[str, MtlMaterial]:
    materials: dict[str, MtlMaterial] = {}
    current: MtlMaterial | None = None
    if not path.exists():
        return materials
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag, values = parts[0], parts[1:]
        if tag == "newmtl" and values:
            current = MtlMaterial(" ".join(values))
            materials[current.name] = current
        elif current and tag == "Kd" and len(values) >= 3:
            current.diffuse_color = (float(values[0]), float(values[1]), float(values[2]))
        elif current and tag == "map_Kd" and values:
            current.diffuse_texture = _texture_name(values)
        elif current and tag == "map_Ks" and values:
            current.specular_texture = _texture_name(values)
        elif current and tag in {"map_bump", "bump"} and values:
            current.bump_texture = _texture_name(values)
    return materials


def parse_obj(path: Path) -> ObjMesh:
    positions: list[tuple[float, float, float]] = []
    vertex_colors: list[tuple[float, float, float] | None] = []
    texcoords: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []
    groups: list[ObjGroup] = []
    material_names: list[str] = []
    mtllibs: list[str] = []
    current = ObjGroup(path.stem)
    current_material: str | None = None

    def ensure_current() -> ObjGroup:
        nonlocal current
        if current not in groups:
            groups.append(current)
        return current

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag, values = parts[0], parts[1:]
        if tag == "mtllib" and values:
            mtllibs.append(" ".join(values))
        elif tag == "v" and len(values) >= 3:
            positions.append((float(values[0]), float(values[1]), float(values[2])))
            if len(values) >= 6:
                vertex_colors.append(_parse_vertex_color(values[3:6]))
            else:
                vertex_colors.append(None)
        elif tag == "vt" and len(values) >= 2:
            texcoords.append((float(values[0]), float(values[1])))
        elif tag == "vn" and len(values) >= 3:
            normals.append((float(values[0]), float(values[1]), float(values[2])))
        elif tag in {"o", "g"} and values:
            if current.faces:
                ensure_current()
            current = ObjGroup(" ".join(values))
        elif tag == "usemtl" and values:
            current_material = " ".join(values)
            if current_material not in material_names:
                material_names.append(current_material)
        elif tag == "f" and len(values) >= 3:
            face_vertices = [_resolve_vertex(token, positions, vertex_colors, texcoords, normals) for token in values]
            for idx in range(1, len(face_vertices) - 1):
                ensure_current().faces.append(
                    ObjFace(
                        vertices=[face_vertices[0], face_vertices[idx], face_vertices[idx + 1]],
                        material=current_material,
                    )
                )

    if current.faces and current not in groups:
        groups.append(current)
    material_defs: dict[str, MtlMaterial] = {}
    for mtllib in mtllibs:
        material_defs.update(parse_mtl(path.parent / mtllib))
    return ObjMesh(
        source=path,
        groups=groups or [current],
        materials=material_names,
        mtllibs=mtllibs,
        material_defs=material_defs,
    )


def import_obj_to_project(project: Project, obj_path: Path, add_to_startup_scene: bool = False) -> AssetMetadata:
    project.ensure_layout()
    obj_path = obj_path.resolve()
    destination = _destination_for_obj(project, obj_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if obj_path != destination.resolve():
        shutil.copy2(obj_path, destination)

    _copy_material_dependencies(obj_path, destination)
    mesh = parse_obj(destination)
    source = destination.resolve().relative_to(project.root.resolve()).as_posix()
    metadata_path = metadata_path_for_source(destination)
    existing = AssetMetadata.load(metadata_path) if metadata_path.exists() else None
    metadata_id = existing.id if existing else f"mesh_{destination.stem}_{uuid4().hex[:8]}"
    settings = dict(existing.settings) if existing else {}
    settings.update({
        "import_mode": "model",
        "model": model_settings_for_obj(metadata_id, mesh),
        "mtllibs": mesh.mtllibs,
        "material_defs": {name: material.to_dict() for name, material in mesh.material_defs.items()},
    })
    metadata = AssetMetadata(
        id=metadata_id,
        kind="obj_mesh",
        source=source,
        groups=mesh.group_names,
        materials=mesh.materials,
        settings=settings,
    )
    metadata.save(metadata_path)

    if add_to_startup_scene:
        scene = project.load_startup_scene()
        root = Entity(destination.stem, object_type=GAME_OBJECT)
        for mesh_entry in model_meshes(metadata):
            child = Entity(str(mesh_entry.get("name") or "Mesh"), object_type=GAME_OBJECT)
            source_materials = [str(item) for item in mesh_entry.get("material_slots", [])]
            material = source_materials[0] if source_materials else None
            child.add_component(
                MeshRenderer(
                    mesh=str(mesh_entry.get("id") or ""),
                    material=material,
                    source_materials=source_materials,
                    material_slots=[None for _material in source_materials],
                )
            )
            root.add_child(child)
        scene.add_entity(root)
        project.save_startup_scene(scene)

    return metadata


def mesh_vertices_for_group(group: ObjGroup, material: str | None = None) -> list[float]:
    vertices: list[float] = []
    for face in group.faces:
        if material is not None and face.material != material:
            continue
        for vertex in face.vertices:
            vertices.extend(vertex.position)
            vertices.extend(vertex.texcoord or (0.0, 0.0))
            vertices.extend(vertex.normal or (0.0, 1.0, 0.0))
            vertices.extend(vertex.color or (1.0, 1.0, 1.0))
    return vertices


def model_settings_for_obj(metadata_id: str, mesh: ObjMesh) -> dict[str, object]:
    used_ids: dict[str, int] = {}
    mesh_entries: list[dict[str, object]] = []
    nodes: list[dict[str, object]] = []
    for group in mesh.groups:
        base_id = safe_model_mesh_id(metadata_id, group.name)
        index = used_ids.get(base_id, 0)
        used_ids[base_id] = index + 1
        mesh_id = safe_model_mesh_id(metadata_id, group.name, index)
        material_slots = _group_materials(group)
        mesh_entries.append({
            "id": mesh_id,
            "name": group.name.split("/")[-1] or group.name,
            "source_group": group.name,
            "node_path": group.name,
            "material_slots": material_slots,
            "bounds": _group_bounds(group),
            "triangle_count": len(group.faces),
            "vertex_count": sum(len(face.vertices) for face in group.faces),
            "wireframe": {"vertices": _wireframe_vertices(group)},
        })
        nodes.append({
            "name": group.name.split("/")[-1] or group.name,
            "path": group.name,
            "mesh": mesh_id,
            "children": [],
        })
    return {
        "import_version": 1,
        "nodes": nodes,
        "meshes": mesh_entries,
        "materials": mesh.materials,
    }


def _group_materials(group: ObjGroup) -> list[str]:
    materials: list[str] = []
    for face in group.faces:
        if face.material and face.material not in materials:
            materials.append(face.material)
    return materials


def _group_bounds(group: ObjGroup) -> dict[str, list[float]]:
    positions = [vertex.position for face in group.faces for vertex in face.vertices]
    if not positions:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    return {
        "min": [min(position[index] for position in positions) for index in range(3)],
        "max": [max(position[index] for position in positions) for index in range(3)],
    }


def _wireframe_vertices(group: ObjGroup) -> list[float]:
    vertices: list[float] = []
    seen: set[tuple[tuple[float, float, float], tuple[float, float, float]]] = set()
    for face in group.faces:
        points = [vertex.position for vertex in face.vertices]
        for start, end in [(points[0], points[1]), (points[1], points[2]), (points[2], points[0])]:
            a = tuple(round(value, 5) for value in start)
            b = tuple(round(value, 5) for value in end)
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            vertices.extend(start)
            vertices.extend(end)
    return vertices


def _resolve_vertex(
    token: str,
    positions: list[tuple[float, float, float]],
    vertex_colors: list[tuple[float, float, float] | None],
    texcoords: list[tuple[float, float]],
    normals: list[tuple[float, float, float]],
) -> ObjVertex:
    refs = token.split("/")
    position_index = _obj_index(refs[0], len(positions))
    position = positions[position_index]
    color = vertex_colors[position_index] if position_index < len(vertex_colors) else None
    texcoord = None
    normal = None
    if len(refs) >= 2 and refs[1]:
        texcoord = texcoords[_obj_index(refs[1], len(texcoords))]
    if len(refs) >= 3 and refs[2]:
        normal = normals[_obj_index(refs[2], len(normals))]
    return ObjVertex(position=position, texcoord=texcoord, normal=normal, color=color)


def _parse_vertex_color(values: list[str]) -> tuple[float, float, float]:
    channels = [float(value) for value in values[:3]]
    if any(channel > 1.0 for channel in channels):
        channels = [channel / 255.0 for channel in channels]
    return (
        max(0.0, min(1.0, channels[0])),
        max(0.0, min(1.0, channels[1])),
        max(0.0, min(1.0, channels[2])),
    )


def _obj_index(value: str, length: int) -> int:
    index = int(value)
    if index < 0:
        return length + index
    return index - 1


def _destination_for_obj(project: Project, obj_path: Path) -> Path:
    try:
        relative = obj_path.relative_to(project.assets_dir.resolve())
        return project.assets_dir / relative
    except ValueError:
        if obj_path.parent.name.lower() != "assets" and parse_obj(obj_path).mtllibs:
            return project.assets_dir / obj_path.parent.name / obj_path.name
        return project.assets_dir / obj_path.name


def _copy_material_dependencies(source_obj: Path, destination_obj: Path) -> None:
    source_mesh = parse_obj(source_obj)
    for mtllib in source_mesh.mtllibs:
        source_mtl = source_obj.parent / mtllib
        destination_mtl = destination_obj.parent / mtllib
        if source_mtl.exists() and source_mtl.resolve() != destination_mtl.resolve():
            shutil.copy2(source_mtl, destination_mtl)
        for material in parse_mtl(source_mtl).values():
            for texture in [material.diffuse_texture, material.specular_texture, material.bump_texture]:
                if not texture:
                    continue
                source_texture = source_mtl.parent / texture
                destination_texture = destination_mtl.parent / texture
                destination_texture.parent.mkdir(parents=True, exist_ok=True)
                if source_texture.exists() and source_texture.resolve() != destination_texture.resolve():
                    shutil.copy2(source_texture, destination_texture)


def _texture_name(values: list[str]) -> str:
    for value in reversed(values):
        if not value.startswith("-"):
            return value.replace("\\", "/")
    return values[-1].replace("\\", "/")
