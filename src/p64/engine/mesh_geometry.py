from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

from p64.engine.assets import AssetMetadata, discover_metadata
from p64.engine.components import MeshRenderer
from p64.engine.entity import Entity
from p64.engine.math import Vec3
from p64.engine.obj import parse_obj
from p64.engine.project import Project

Triangle = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class ConvexHull:
    vertices: tuple[Vec3, ...]
    triangles: tuple[tuple[Vec3, Vec3, Vec3], ...]
    normals: tuple[Vec3, ...]
    bounds: tuple[Vec3, Vec3]


_CONVEX_HULL_CACHE: dict[tuple[str, str, str | None], ConvexHull | None] = {}
_MAX_EXACT_HULL_POINTS = 64


def mesh_renderer_for(entity: Entity) -> MeshRenderer | None:
    for component in entity.components:
        if isinstance(component, MeshRenderer) and component.enabled:
            return component
    return None


def mesh_bounds(project: Project, component: MeshRenderer) -> tuple[Vec3, Vec3] | None:
    positions = [position for triangle in mesh_triangles(project, component) for position in triangle]
    if not positions:
        return None
    return (
        Vec3(*(min(position[index] for position in positions) for index in range(3))),
        Vec3(*(max(position[index] for position in positions) for index in range(3))),
    )


def mesh_triangles(project: Project, component: MeshRenderer) -> list[Triangle]:
    metadata = _metadata_by_id(project).get(component.mesh)
    if metadata is None:
        return []
    obj_mesh = parse_obj(project.root / metadata.source)
    group = next((item for item in obj_mesh.groups if item.name == component.submesh), obj_mesh.groups[0] if obj_mesh.groups else None)
    if group is None:
        return []
    return [tuple(vertex.position for vertex in face.vertices) for face in group.faces if len(face.vertices) == 3]


def convex_hull(project: Project, component: MeshRenderer) -> ConvexHull | None:
    key = (str(project.root.resolve()), component.mesh, component.submesh)
    if key not in _CONVEX_HULL_CACHE:
        _CONVEX_HULL_CACHE[key] = build_convex_hull(_mesh_points(project, component))
    return _CONVEX_HULL_CACHE[key]


def clear_convex_hull_cache(project: Project | None = None) -> None:
    if project is None:
        _CONVEX_HULL_CACHE.clear()
        return
    prefix = str(project.root.resolve())
    for key in list(_CONVEX_HULL_CACHE):
        if key[0] == prefix:
            _CONVEX_HULL_CACHE.pop(key, None)


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
    metadata_by_id: dict[str, AssetMetadata] = {}
    for metadata_path in discover_metadata(project.assets_dir):
        try:
            metadata = AssetMetadata.load(metadata_path)
        except Exception:
            continue
        metadata_by_id[metadata.id] = metadata
    return metadata_by_id
