from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

from p64.engine.assets import AssetMetadata, discover_metadata, resolve_model_mesh
from p64.engine.components import MeshRenderer
from p64.engine.entity import Entity
from p64.engine.files import metadata_path_for_source
from p64.engine.math import Vec3
from p64.engine.obj import ObjGroup, parse_obj
from p64.engine.project import Project

Triangle = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class ConvexHull:
    vertices: tuple[Vec3, ...]
    triangles: tuple[tuple[Vec3, Vec3, Vec3], ...]
    normals: tuple[Vec3, ...]
    bounds: tuple[Vec3, Vec3]


_CONVEX_HULL_CACHE: dict[tuple[str, str, str | None], ConvexHull | None] = {}
_MESH_TRIANGLE_CACHE: dict[tuple[str, str, str | None], list[Triangle]] = {}
_METADATA_BY_ID_CACHE: dict[str, dict[str, AssetMetadata]] = {}
_MAX_EXACT_HULL_POINTS = 64


def mesh_renderer_for(entity: Entity) -> MeshRenderer | None:
    for component in entity.components:
        if isinstance(component, MeshRenderer) and component.enabled:
            return component
    return None


def mesh_bounds(project: Project, component: MeshRenderer) -> tuple[Vec3, Vec3] | None:
    metadata, mesh = resolve_model_mesh(_metadata_by_id(project), component.mesh, component.submesh)
    bounds = mesh.get("bounds") if mesh else None
    if isinstance(bounds, dict) and isinstance(bounds.get("min"), list) and isinstance(bounds.get("max"), list):
        return (Vec3.from_value(bounds["min"]), Vec3.from_value(bounds["max"]))
    positions = [position for triangle in mesh_triangles(project, component) for position in triangle]
    if not positions:
        return None
    return (
        Vec3(*(min(position[index] for position in positions) for index in range(3))),
        Vec3(*(max(position[index] for position in positions) for index in range(3))),
    )


def mesh_triangles(project: Project, component: MeshRenderer) -> list[Triangle]:
    key = (str(project.root.resolve()), component.mesh, component.submesh)
    if key in _MESH_TRIANGLE_CACHE:
        return _MESH_TRIANGLE_CACHE[key]
    metadata, mesh = resolve_model_mesh(_metadata_by_id(project), component.mesh, component.submesh)
    if metadata is None:
        return []
    baked = _baked_mesh_triangles(metadata, mesh, component)
    if baked is not None:
        _MESH_TRIANGLE_CACHE[key] = baked
        return baked
    obj_mesh = parse_obj(project.root / metadata.source)
    group_name = str(mesh.get("source_group")) if mesh and mesh.get("source_group") else component.submesh
    group = next((item for item in obj_mesh.groups if item.name == group_name), obj_mesh.groups[0] if obj_mesh.groups else None)
    if group is None:
        return []
    triangles = [tuple(vertex.position for vertex in face.vertices) for face in group.faces if len(face.vertices) == 3]
    _MESH_TRIANGLE_CACHE[key] = triangles
    return triangles


def ensure_mesh_collision_metadata(project: Project, component: MeshRenderer) -> None:
    metadata_by_id = _metadata_by_id(project)
    metadata, mesh = resolve_model_mesh(metadata_by_id, component.mesh, component.submesh)
    if metadata is None:
        return
    mesh_id = str(mesh.get("id")) if mesh and mesh.get("id") else component.mesh
    collision = metadata.settings.get("collision")
    meshes = collision.get("meshes") if isinstance(collision, dict) else None
    if isinstance(meshes, dict) and mesh_id in meshes:
        return
    source = project.root / metadata.source
    obj_mesh = parse_obj(source)
    collision_meshes: dict[str, object] = dict(meshes) if isinstance(meshes, dict) else {}
    for mesh_entry in _model_mesh_entries(metadata):
        entry_id = str(mesh_entry.get("id") or "")
        if not entry_id or entry_id in collision_meshes:
            continue
        group_name = str(mesh_entry.get("source_group") or mesh_entry.get("name") or "")
        group = next((item for item in obj_mesh.groups if item.name == group_name), None)
        if group is None:
            continue
        collision_meshes[entry_id] = _collision_entry_for_group(group)
    metadata.settings["collision"] = {
        "version": 1,
        "source_mtime": source.stat().st_mtime if source.exists() else 0.0,
        "meshes": collision_meshes,
    }
    metadata.save(metadata_path_for_source(project.root / metadata.source))
    clear_mesh_geometry_cache(project)


def convex_hull(project: Project, component: MeshRenderer) -> ConvexHull | None:
    key = (str(project.root.resolve()), component.mesh, component.submesh)
    if key not in _CONVEX_HULL_CACHE:
        _CONVEX_HULL_CACHE[key] = build_convex_hull(_mesh_points(project, component))
    return _CONVEX_HULL_CACHE[key]


def clear_convex_hull_cache(project: Project | None = None) -> None:
    if project is None:
        _CONVEX_HULL_CACHE.clear()
        _MESH_TRIANGLE_CACHE.clear()
        _METADATA_BY_ID_CACHE.clear()
        return
    prefix = str(project.root.resolve())
    for key in list(_CONVEX_HULL_CACHE):
        if key[0] == prefix:
            _CONVEX_HULL_CACHE.pop(key, None)
    for key in list(_MESH_TRIANGLE_CACHE):
        if key[0] == prefix:
            _MESH_TRIANGLE_CACHE.pop(key, None)
    _METADATA_BY_ID_CACHE.pop(prefix, None)


def clear_mesh_geometry_cache(project: Project | None = None) -> None:
    clear_convex_hull_cache(project)


def build_convex_hull(points: Iterable[tuple[float, float, float] | Vec3]) -> ConvexHull | None:
    unique = _unique_points(points)
    if len(unique) < 4:
        return _bounds_hull(unique)
    source_bounds = bounds_from_points(unique)
    unique = _reduced_support_points(unique) if len(unique) > _MAX_EXACT_HULL_POINTS else unique
    centroid = Vec3(
        sum(point.x for point in unique) / len(unique),
        sum(point.y for point in unique) / len(unique),
        sum(point.z for point in unique) / len(unique),
    )
    planes: dict[tuple[int, int, int, int], tuple[Vec3, float, set[int]]] = {}
    epsilon = 0.00001
    for i in range(len(unique) - 2):
        for j in range(i + 1, len(unique) - 1):
            for k in range(j + 1, len(unique)):
                normal = _normalize_vec(_cross_vec(_sub_vec(unique[j], unique[i]), _sub_vec(unique[k], unique[i])))
                if _length_vec(normal) < epsilon:
                    continue
                distances = [_dot_vec(normal, point) - _dot_vec(normal, unique[i]) for point in unique]
                has_positive = any(distance > epsilon for distance in distances)
                has_negative = any(distance < -epsilon for distance in distances)
                if has_positive and has_negative:
                    continue
                if _dot_vec(normal, _sub_vec(centroid, unique[i])) > 0.0:
                    normal = Vec3(-normal.x, -normal.y, -normal.z)
                offset = _dot_vec(normal, unique[i])
                coplanar = {index for index, distance in enumerate(distances) if abs(distance) <= epsilon}
                plane_key = (
                    round(normal.x * 100000),
                    round(normal.y * 100000),
                    round(normal.z * 100000),
                    round(offset * 100000),
                )
                if plane_key in planes:
                    planes[plane_key][2].update(coplanar)
                else:
                    planes[plane_key] = (normal, offset, set(coplanar))
    triangles: list[tuple[Vec3, Vec3, Vec3]] = []
    normals: list[Vec3] = []
    for normal, _offset, indices in planes.values():
        if len(indices) < 3:
            continue
        polygon = _sort_coplanar_points([unique[index] for index in indices], normal)
        if len(polygon) < 3:
            continue
        anchor = polygon[0]
        for index in range(1, len(polygon) - 1):
            triangle = (anchor, polygon[index], polygon[index + 1])
            triangles.append(triangle)
            normals.append(normal)
    if not triangles:
        return _bounds_hull(unique)
    return ConvexHull(tuple(unique), tuple(triangles), tuple(normals), source_bounds)


def transform_point(matrix: list[float], point: tuple[float, float, float] | Vec3) -> Vec3:
    x, y, z = (point.x, point.y, point.z) if isinstance(point, Vec3) else point
    return Vec3(
        matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3],
        matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7],
        matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11],
    )


def transform_triangle(matrix: list[float], triangle: Triangle) -> tuple[Vec3, Vec3, Vec3]:
    return tuple(transform_point(matrix, point) for point in triangle)  # type: ignore[return-value]


def bounds_from_points(points: Iterable[Vec3]) -> tuple[Vec3, Vec3]:
    values = list(points)
    return (
        Vec3(*(min(getattr(point, axis) for point in values) for axis in ("x", "y", "z"))),
        Vec3(*(max(getattr(point, axis) for point in values) for axis in ("x", "y", "z"))),
    )


def transformed_bounds(entity: Entity, local_bounds: tuple[Vec3, Vec3]) -> tuple[Vec3, Vec3]:
    mins, maxs = local_bounds
    corners = [
        Vec3(mins.x, mins.y, mins.z),
        Vec3(maxs.x, mins.y, mins.z),
        Vec3(maxs.x, maxs.y, mins.z),
        Vec3(mins.x, maxs.y, mins.z),
        Vec3(mins.x, mins.y, maxs.z),
        Vec3(maxs.x, mins.y, maxs.z),
        Vec3(maxs.x, maxs.y, maxs.z),
        Vec3(mins.x, maxs.y, maxs.z),
    ]
    matrix = entity.transform.world_matrix(entity)
    return bounds_from_points(transform_point(matrix, point) for point in corners)


def _mesh_points(project: Project, component: MeshRenderer) -> list[tuple[float, float, float]]:
    return [point for triangle in mesh_triangles(project, component) for point in triangle]


def _model_mesh_entries(metadata: AssetMetadata) -> list[dict[str, object]]:
    model = metadata.settings.get("model")
    meshes = model.get("meshes", []) if isinstance(model, dict) else []
    return [item for item in meshes if isinstance(item, dict)]


def _collision_entry_for_group(group: ObjGroup) -> dict[str, object]:
    positions = [vertex.position for face in group.faces for vertex in face.vertices]
    bounds = {
        "min": [min(position[index] for position in positions) for index in range(3)] if positions else [0.0, 0.0, 0.0],
        "max": [max(position[index] for position in positions) for index in range(3)] if positions else [0.0, 0.0, 0.0],
    }
    return {
        "bounds": bounds,
        "source_group": group.name,
        "triangles": [
            [list(vertex.position) for vertex in face.vertices]
            for face in group.faces
            if len(face.vertices) == 3
        ],
    }


def _baked_mesh_triangles(metadata: AssetMetadata, mesh: dict[str, object] | None, component: MeshRenderer) -> list[Triangle] | None:
    collision = metadata.settings.get("collision")
    if not isinstance(collision, dict) or int(collision.get("version", 0) or 0) < 1:
        return None
    meshes = collision.get("meshes")
    if not isinstance(meshes, dict):
        return None
    mesh_id = str(mesh.get("id")) if mesh and mesh.get("id") else component.mesh
    entry = meshes.get(mesh_id)
    if not isinstance(entry, dict) and mesh is not None:
        source_group = mesh.get("source_group")
        entry = next(
            (
                value for value in meshes.values()
                if isinstance(value, dict) and value.get("source_group") == source_group
            ),
            None,
        )
    if not isinstance(entry, dict):
        return None
    raw_triangles = entry.get("triangles")
    if not isinstance(raw_triangles, list):
        return None
    triangles: list[Triangle] = []
    for raw_triangle in raw_triangles:
        if not isinstance(raw_triangle, list) or len(raw_triangle) != 3:
            continue
        points: list[tuple[float, float, float]] = []
        for raw_point in raw_triangle:
            if not isinstance(raw_point, list) or len(raw_point) < 3:
                break
            try:
                points.append((float(raw_point[0]), float(raw_point[1]), float(raw_point[2])))
            except (TypeError, ValueError):
                break
        if len(points) == 3:
            triangles.append((points[0], points[1], points[2]))
    return triangles


def _unique_points(points: Iterable[tuple[float, float, float] | Vec3]) -> list[Vec3]:
    seen: set[tuple[int, int, int]] = set()
    unique: list[Vec3] = []
    for point in points:
        vec = point if isinstance(point, Vec3) else Vec3(*point)
        key = (round(vec.x * 100000), round(vec.y * 100000), round(vec.z * 100000))
        if key in seen:
            continue
        seen.add(key)
        unique.append(vec)
    return unique


def _reduced_support_points(points: list[Vec3]) -> list[Vec3]:
    directions = _support_directions()
    selected: dict[tuple[int, int, int], Vec3] = {}
    for direction in directions:
        maximum = max(points, key=lambda point, direction=direction: _dot_vec(point, direction))
        minimum = min(points, key=lambda point, direction=direction: _dot_vec(point, direction))
        for point in (maximum, minimum):
            key = (round(point.x * 100000), round(point.y * 100000), round(point.z * 100000))
            selected[key] = point
    return list(selected.values())


def _support_directions() -> list[Vec3]:
    directions: list[Vec3] = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                if x == y == z == 0.0:
                    continue
                directions.append(_normalize_vec(Vec3(x, y, z)))
    for x in (-2.0, -1.0, 1.0, 2.0):
        for y in (-2.0, -1.0, 1.0, 2.0):
            directions.append(_normalize_vec(Vec3(x, y, 1.0)))
            directions.append(_normalize_vec(Vec3(x, 1.0, y)))
            directions.append(_normalize_vec(Vec3(1.0, x, y)))
    return directions


def _bounds_hull(points: list[Vec3]) -> ConvexHull | None:
    if not points:
        return None
    mins, maxs = bounds_from_points(points)
    corners = [
        Vec3(mins.x, mins.y, mins.z),
        Vec3(maxs.x, mins.y, mins.z),
        Vec3(maxs.x, maxs.y, mins.z),
        Vec3(mins.x, maxs.y, mins.z),
        Vec3(mins.x, mins.y, maxs.z),
        Vec3(maxs.x, mins.y, maxs.z),
        Vec3(maxs.x, maxs.y, maxs.z),
        Vec3(mins.x, maxs.y, maxs.z),
    ]
    faces = [
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]
    triangles = tuple((corners[a], corners[b], corners[c]) for a, b, c in faces)
    normals = tuple(_triangle_normal_vec(triangle) for triangle in triangles)
    return ConvexHull(tuple(corners), triangles, normals, (mins, maxs))


def _sort_coplanar_points(points: list[Vec3], normal: Vec3) -> list[Vec3]:
    center = Vec3(
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
        sum(point.z for point in points) / len(points),
    )
    axis = Vec3(1.0, 0.0, 0.0) if abs(normal.x) < 0.9 else Vec3(0.0, 1.0, 0.0)
    tangent = _normalize_vec(_cross_vec(normal, axis))
    bitangent = _cross_vec(normal, tangent)
    from math import atan2

    return sorted(points, key=lambda point: atan2(_dot_vec(_sub_vec(point, center), bitangent), _dot_vec(_sub_vec(point, center), tangent)))


def _triangle_normal_vec(triangle: tuple[Vec3, Vec3, Vec3]) -> Vec3:
    a, b, c = triangle
    return _normalize_vec(_cross_vec(_sub_vec(b, a), _sub_vec(c, a)))


def _sub_vec(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z)


def _dot_vec(a: Vec3, b: Vec3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def _cross_vec(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def _length_vec(vector: Vec3) -> float:
    return sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)


def _normalize_vec(vector: Vec3) -> Vec3:
    length = _length_vec(vector)
    if length < 0.000001:
        return Vec3()
    return Vec3(vector.x / length, vector.y / length, vector.z / length)


def _metadata_by_id(project: Project) -> dict[str, AssetMetadata]:
    key = str(project.root.resolve())
    cached = _METADATA_BY_ID_CACHE.get(key)
    if cached is not None:
        return cached
    metadata_by_id: dict[str, AssetMetadata] = {}
    for metadata_path in discover_metadata(project.assets_dir):
        try:
            metadata = AssetMetadata.load(metadata_path)
        except Exception:
            continue
        metadata_by_id[metadata.id] = metadata
    _METADATA_BY_ID_CACHE[key] = metadata_by_id
    return metadata_by_id
