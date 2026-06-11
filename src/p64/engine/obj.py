from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from p64.engine.assets import AssetMetadata
from p64.engine.components import MeshRenderer
from p64.engine.entity import Entity
from p64.engine.files import metadata_path_for_source
from p64.engine.project import Project


@dataclass
class ObjVertex:
    position: tuple[float, float, float]
    texcoord: tuple[float, float] | None = None
    normal: tuple[float, float, float] | None = None


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
            face_vertices = [_resolve_vertex(token, positions, texcoords, normals) for token in values]
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
    metadata = AssetMetadata(
        id=f"mesh_{destination.stem}_{uuid4().hex[:8]}",
        kind="obj_mesh",
        source=source,
        groups=mesh.group_names,
        materials=mesh.materials,
        settings={
            "import_mode": "groups_as_nodes",
            "mtllibs": mesh.mtllibs,
            "material_defs": {name: material.to_dict() for name, material in mesh.material_defs.items()},
        },
    )
    metadata.save(metadata_path_for_source(destination))

    if add_to_startup_scene:
        scene = project.load_startup_scene()
        root = Entity(destination.stem)
        for group in mesh.groups:
            child = Entity(group.name)
            material = group.faces[0].material if group.faces else None
            child.add_component(MeshRenderer(mesh=metadata.id, submesh=group.name, material=material))
            root.add_child(child)
        scene.add_entity(root)
        project.save_startup_scene(scene)

    return metadata


def mesh_vertices_for_group(group: ObjGroup) -> list[float]:
    vertices: list[float] = []
    for face in group.faces:
        for vertex in face.vertices:
            vertices.extend(vertex.position)
            vertices.extend(vertex.texcoord or (0.0, 0.0))
            vertices.extend(vertex.normal or (0.0, 1.0, 0.0))
    return vertices


def _resolve_vertex(
    token: str,
    positions: list[tuple[float, float, float]],
    texcoords: list[tuple[float, float]],
    normals: list[tuple[float, float, float]],
) -> ObjVertex:
    refs = token.split("/")
    position = positions[_obj_index(refs[0], len(positions))]
    texcoord = None
    normal = None
    if len(refs) >= 2 and refs[1]:
        texcoord = texcoords[_obj_index(refs[1], len(texcoords))]
    if len(refs) >= 3 and refs[2]:
        normal = normals[_obj_index(refs[2], len(normals))]
    return ObjVertex(position=position, texcoord=texcoord, normal=normal)


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
