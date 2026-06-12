from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from p64.engine.components import CharacterController, Collider, EntityPhysics
from p64.engine.entity import Entity
from p64.engine.math import Vec3
from p64.engine.mesh_geometry import bounds_from_points, convex_hull, mesh_bounds, mesh_renderer_for, mesh_triangles, transform_point, transform_triangle, transformed_bounds


@dataclass(slots=True)
class CollisionHit:
    entity: Entity
    collider: Collider
    normal: Vec3
    depth: float
    is_trigger: bool = False


@dataclass(slots=True)
class Bounds:
    min: Vec3
    max: Vec3

    def overlaps(self, other: "Bounds") -> bool:
        return (
            self.min.x <= other.max.x and self.max.x >= other.min.x
            and self.min.y <= other.max.y and self.max.y >= other.min.y
            and self.min.z <= other.max.z and self.max.z >= other.min.z
        )


class CollisionWorld:
    def __init__(self, scene: Any, project: Any | None = None) -> None:
        self.scene = scene
        self.project = project
        self.colliders: list[tuple[Entity, Collider]] = []
        for entity in scene.walk():
            if not entity.active:
                continue
            for component in entity.components:
                if isinstance(component, Collider) and component.enabled:
                    self.colliders.append((entity, component))

    def overlaps(self, entity: Entity, collider: Collider, include_triggers: bool = True) -> list[CollisionHit]:
        bounds = collider_bounds(entity, collider, self.project)
        hits: list[CollisionHit] = []
        for other_entity, other in self.colliders:
            if other_entity is entity or not _layers_can_collide(collider, other):
                continue
            if not include_triggers and other.is_trigger:
                continue
            if other.shape == "mesh" and self.project is not None:
                hits.extend(
                    CollisionHit(other_entity, other, contact.normal, contact.depth, other.is_trigger)
                    for contact in _mesh_contacts(entity, collider, other_entity, other, self.project)
                )
                continue
            other_bounds = collider_bounds(other_entity, other, self.project)
            if not bounds.overlaps(other_bounds):
                continue
            normal, depth = _separation(bounds, other_bounds)
            hits.append(CollisionHit(other_entity, other, normal, depth, other.is_trigger))
        return hits

    def ground_check(self, entity: Entity, controller: CharacterController, distance: float | None = None) -> CollisionHit | None:
        distance = controller.skin_width + 0.08 if distance is None else distance
        bounds = controller_bounds(entity, controller)
        probe = Bounds(
            Vec3(bounds.min.x, bounds.min.y - distance, bounds.min.z),
            Vec3(bounds.max.x, bounds.min.y + controller.skin_width, bounds.max.z),
        )
        best: CollisionHit | None = None
        for other_entity, other in self.colliders:
            if other_entity is entity or other.is_trigger:
                continue
            if other.shape == "mesh" and self.project is not None:
                contacts = (
                    _aabb_convex_contacts(probe, other_entity, other, self.project)
                    if other.convex else _aabb_mesh_contacts(probe, other_entity, self.project)
                )
                if not contacts:
                    continue
                hit = min(contacts, key=lambda item: item.depth)
                if best is None or hit.depth < best.depth:
                    best = CollisionHit(other_entity, other, hit.normal, hit.depth, False)
                continue
            else:
                other_bounds = collider_bounds(other_entity, other, self.project)
            if not probe.overlaps(other_bounds):
                continue
            depth = probe.max.y - other_bounds.min.y
            if best is None or depth < best.depth:
                best = CollisionHit(other_entity, other, Vec3(0.0, 1.0, 0.0), depth, False)
        return best

    def move_character(self, entity: Entity, controller: CharacterController, motion: Vec3, dt: float) -> Vec3:
        dt = max(float(dt), 0.0)
        controller.grounded = False
        controller.velocity.y -= controller.gravity * dt
        total = Vec3(motion.x, motion.y + controller.velocity.y * dt, motion.z)
        actual = Vec3()
        for axis in ("x", "y", "z"):
            delta = getattr(total, axis)
            if abs(delta) < 0.000001:
                continue
            before = getattr(entity.transform.position, axis)
            setattr(entity.transform.position, axis, before + delta)
            hits = self._controller_blocking_hits(entity, controller)
            if hits:
                setattr(entity.transform.position, axis, before)
                if axis == "y" and delta < 0.0:
                    controller.grounded = True
                    controller.velocity.y = 0.0
                else:
                    setattr(controller.velocity, axis, 0.0)
            else:
                setattr(actual, axis, delta)
        grounded = self.ground_check(entity, controller) is not None
        controller.grounded = controller.grounded or grounded
        if controller.grounded and controller.velocity.y < 0.0:
            controller.velocity.y = 0.0
        return actual

    def step_physics(self, dt: float) -> None:
        dt = max(float(dt), 0.0)
        for entity in self.scene.walk():
            if not entity.active:
                continue
            physics = _entity_physics(entity)
            if physics is None or not physics.enabled:
                continue
            self._step_entity_physics(entity, physics, _entity_collider(entity), dt)

    def _step_entity_physics(self, entity: Entity, physics: EntityPhysics, collider: Collider | None, dt: float) -> None:
        if not entity.is_entity or physics.is_kinematic or dt <= 0.0:
            physics.clear_accumulators()
            return
        inverse_mass = physics.inverse_mass
        if physics.use_gravity:
            physics.velocity.y -= 18.0 * dt
        physics.velocity.x += physics._force.x * inverse_mass * dt
        physics.velocity.y += physics._force.y * inverse_mass * dt
        physics.velocity.z += physics._force.z * inverse_mass * dt
        _apply_drag(physics.velocity, physics.drag, dt)

        motion = Vec3(physics.velocity.x * dt, physics.velocity.y * dt, physics.velocity.z * dt)
        self._apply_freeze(physics.velocity, motion, physics.freeze_position)
        if collider is None:
            entity.transform.position.x += motion.x
            entity.transform.position.y += motion.y
            entity.transform.position.z += motion.z
        else:
            self._move_physics_body(entity, physics, collider, motion)

        physics.angular_velocity.x += physics._torque.x * inverse_mass * dt
        physics.angular_velocity.y += physics._torque.y * inverse_mass * dt
        physics.angular_velocity.z += physics._torque.z * inverse_mass * dt
        _apply_drag(physics.angular_velocity, physics.angular_drag, dt)
        rotation_delta = Vec3(
            physics.angular_velocity.x * dt,
            physics.angular_velocity.y * dt,
            physics.angular_velocity.z * dt,
        )
        self._apply_freeze(physics.angular_velocity, rotation_delta, physics.freeze_rotation)
        entity.transform.rotation.x += rotation_delta.x
        entity.transform.rotation.y += rotation_delta.y
        entity.transform.rotation.z += rotation_delta.z
        physics.clear_accumulators()

    def _apply_freeze(self, velocity: Vec3, delta: Vec3, freeze: Vec3) -> None:
        for axis in ("x", "y", "z"):
            if abs(getattr(freeze, axis)) > 0.000001:
                setattr(delta, axis, 0.0)
                setattr(velocity, axis, 0.0)

    def _move_physics_body(self, entity: Entity, physics: EntityPhysics, collider: Collider, motion: Vec3) -> None:
        for axis in ("x", "y", "z"):
            delta = getattr(motion, axis)
            if abs(delta) < 0.000001:
                continue
            before = getattr(entity.transform.position, axis)
            setattr(entity.transform.position, axis, before + delta)
            if self._physics_blocking_hits(entity, collider):
                setattr(entity.transform.position, axis, before)
                setattr(physics.velocity, axis, 0.0)

    def _physics_blocking_hits(self, entity: Entity, collider: Collider) -> list[CollisionHit]:
        bounds = collider_bounds(entity, collider, self.project)
        hits: list[CollisionHit] = []
        for other_entity, other in self.colliders:
            if other_entity is entity or other.is_trigger or not _layers_can_collide(collider, other):
                continue
            if collider.shape == "mesh" and collider.convex and self.project is not None:
                for contact in _convex_collider_contacts(entity, collider, other_entity, other, self.project):
                    hits.append(CollisionHit(other_entity, other, contact.normal, contact.depth, False))
                continue
            if other.shape == "mesh" and self.project is not None:
                for contact in _mesh_contacts(entity, collider, other_entity, other, self.project):
                    hits.append(CollisionHit(other_entity, other, contact.normal, contact.depth, False))
                continue
            other_bounds = collider_bounds(other_entity, other, self.project)
            if bounds.overlaps(other_bounds):
                normal, depth = _separation(bounds, other_bounds)
                hits.append(CollisionHit(other_entity, other, normal, depth, False))
        return hits

    def _controller_blocking_hits(self, entity: Entity, controller: CharacterController) -> list[CollisionHit]:
        bounds = controller_bounds(entity, controller)
        hits: list[CollisionHit] = []
        for other_entity, other in self.colliders:
            if other_entity is entity or other.is_trigger:
                continue
            if other.shape == "mesh" and self.project is not None:
                contacts = (
                    _aabb_convex_contacts(bounds, other_entity, other, self.project)
                    if other.convex else _aabb_mesh_contacts(bounds, other_entity, self.project)
                )
                for contact in contacts:
                    hits.append(CollisionHit(other_entity, other, contact.normal, contact.depth, False))
                continue
            other_bounds = collider_bounds(other_entity, other, self.project)
            if bounds.overlaps(other_bounds):
                normal, depth = _separation(bounds, other_bounds)
                hits.append(CollisionHit(other_entity, other, normal, depth, False))
        return hits


def collider_bounds(entity: Entity, collider: Collider, project: Any | None = None) -> Bounds:
    if project is not None and (collider.fit_to_mesh or collider.shape == "mesh"):
        renderer = mesh_renderer_for(entity)
        local_bounds = mesh_bounds(project, renderer) if renderer else None
        if local_bounds is not None:
            world_min, world_max = transformed_bounds(entity, local_bounds)
            if collider.shape == "sphere":
                center, radius = collider_sphere(entity, collider, project)
                return Bounds(
                    Vec3(center.x - radius, center.y - radius, center.z - radius),
                    Vec3(center.x + radius, center.y + radius, center.z + radius),
                )
            return Bounds(world_min, world_max)
    position = entity.transform.position
    scale = entity.transform.scale
    center = Vec3(
        position.x + collider.center.x * scale.x,
        position.y + collider.center.y * scale.y,
        position.z + collider.center.z * scale.z,
    )
    if collider.shape == "sphere":
        radius = max(collider.radius * max(abs(scale.x), abs(scale.y), abs(scale.z)), 0.001)
        return Bounds(
            Vec3(center.x - radius, center.y - radius, center.z - radius),
            Vec3(center.x + radius, center.y + radius, center.z + radius),
        )
    half = Vec3(
        max(abs(collider.size.x * scale.x) * 0.5, 0.001),
        max(abs(collider.size.y * scale.y) * 0.5, 0.001),
        max(abs(collider.size.z * scale.z) * 0.5, 0.001),
    )
    return Bounds(
        Vec3(center.x - half.x, center.y - half.y, center.z - half.z),
        Vec3(center.x + half.x, center.y + half.y, center.z + half.z),
    )


def collider_sphere(entity: Entity, collider: Collider, project: Any | None = None) -> tuple[Vec3, float]:
    if project is not None and collider.fit_to_mesh:
        renderer = mesh_renderer_for(entity)
        local_bounds = mesh_bounds(project, renderer) if renderer else None
        if local_bounds is not None:
            world_min, world_max = transformed_bounds(entity, local_bounds)
            center = Vec3(
                (world_min.x + world_max.x) * 0.5,
                (world_min.y + world_max.y) * 0.5,
                (world_min.z + world_max.z) * 0.5,
            )
            radius = max(world_max.x - world_min.x, world_max.y - world_min.y, world_max.z - world_min.z) * 0.5
            return center, max(radius, 0.001)
    position = entity.transform.position
    scale = entity.transform.scale
    center = Vec3(
        position.x + collider.center.x * scale.x,
        position.y + collider.center.y * scale.y,
        position.z + collider.center.z * scale.z,
    )
    return center, max(collider.radius * max(abs(scale.x), abs(scale.y), abs(scale.z)), 0.001)


def apply_mesh_primitive_defaults(project: Any | None, entity: Entity, collider: Collider, shape: str | None = None) -> bool:
    if project is None:
        return False
    renderer = mesh_renderer_for(entity)
    local_bounds = mesh_bounds(project, renderer) if renderer else None
    if local_bounds is None:
        return False
    mins, maxs = local_bounds
    center = Vec3(
        (mins.x + maxs.x) * 0.5,
        (mins.y + maxs.y) * 0.5,
        (mins.z + maxs.z) * 0.5,
    )
    size = Vec3(
        max(maxs.x - mins.x, 0.001),
        max(maxs.y - mins.y, 0.001),
        max(maxs.z - mins.z, 0.001),
    )
    radius = max(size.x, size.y, size.z) * 0.5
    if shape is not None:
        collider.shape = shape
    if collider.shape == "box":
        collider.center = center
        collider.size = size
        collider.radius = radius
    elif collider.shape == "sphere":
        collider.center = center
        collider.radius = radius
    return True


def controller_bounds(entity: Entity, controller: CharacterController) -> Bounds:
    position = entity.transform.position
    radius = max(controller.radius - controller.skin_width, 0.001)
    height = max(controller.height, radius * 2.0)
    return Bounds(
        Vec3(position.x - radius, position.y + controller.skin_width, position.z - radius),
        Vec3(position.x + radius, position.y + height, position.z + radius),
    )


def move_character(scene: Any, entity: Entity, controller: CharacterController, motion: Vec3, dt: float) -> Vec3:
    return CollisionWorld(scene).move_character(entity, controller, motion, dt)


def _entity_collider(entity: Entity) -> Collider | None:
    for component in entity.components:
        if isinstance(component, Collider) and component.enabled:
            return component
    return None


def _entity_physics(entity: Entity) -> EntityPhysics | None:
    for component in entity.components:
        if isinstance(component, EntityPhysics):
            return component
    return None


def _apply_drag(vector: Vec3, drag: float, dt: float) -> None:
    if drag <= 0.0:
        return
    factor = max(0.0, 1.0 - drag * dt)
    vector.x *= factor
    vector.y *= factor
    vector.z *= factor


@dataclass(slots=True)
class _TriangleContact:
    normal: Vec3
    depth: float


def _mesh_contacts(entity: Entity, collider: Collider, mesh_entity: Entity, mesh_collider: Collider, project: Any) -> list[_TriangleContact]:
    if mesh_collider.convex:
        if collider.shape == "sphere":
            center, radius = collider_sphere(entity, collider, project)
            return _sphere_convex_contacts(center, radius, mesh_entity, mesh_collider, project)
        return _aabb_convex_contacts(collider_bounds(entity, collider, project), mesh_entity, mesh_collider, project)
    if collider.shape == "sphere":
        center, radius = collider_sphere(entity, collider, project)
        return _sphere_mesh_contacts(center, radius, mesh_entity, project)
    return _aabb_mesh_contacts(collider_bounds(entity, collider, project), mesh_entity, project)


def _convex_collider_contacts(entity: Entity, collider: Collider, other_entity: Entity, other: Collider, project: Any) -> list[_TriangleContact]:
    own_vertices, own_normals = _world_convex_points(entity, collider, project)
    if not own_vertices:
        return []
    if other.shape == "mesh":
        if other.convex:
            other_vertices, other_normals = _world_convex_points(other_entity, other, project)
            return _convex_points_contacts(own_vertices, other_vertices, list(own_normals) + list(other_normals))
        return _aabb_mesh_contacts(collider_bounds(entity, collider, project), other_entity, project)
    if other.shape == "sphere":
        center, radius = collider_sphere(other_entity, other, project)
        return _sphere_against_points_contacts(center, radius, own_vertices, own_normals)
    other_bounds = collider_bounds(other_entity, other, project)
    return _convex_aabb_contacts(own_vertices, own_normals, other_bounds)


def _aabb_mesh_contacts(bounds: Bounds, entity: Entity, project: Any) -> list[_TriangleContact]:
    renderer = mesh_renderer_for(entity)
    if renderer is None:
        return []
    matrix = entity.transform.world_matrix(entity)
    contacts: list[_TriangleContact] = []
    for triangle in mesh_triangles(project, renderer):
        world_triangle = transform_triangle(matrix, triangle)
        tri_min, tri_max = bounds_from_points(world_triangle)
        triangle_bounds = Bounds(tri_min, tri_max)
        if bounds.overlaps(triangle_bounds):
            contact = _aabb_triangle_contact(bounds, world_triangle)
            if contact is not None:
                contacts.append(contact)
    return contacts


def _aabb_convex_contacts(bounds: Bounds, entity: Entity, collider: Collider, project: Any) -> list[_TriangleContact]:
    vertices, normals = _world_convex_points(entity, collider, project)
    if not vertices:
        return []
    return _convex_aabb_contacts(vertices, normals, bounds)


def _sphere_convex_contacts(center: Vec3, radius: float, entity: Entity, collider: Collider, project: Any) -> list[_TriangleContact]:
    vertices, normals = _world_convex_points(entity, collider, project)
    if not vertices:
        return []
    return _sphere_against_points_contacts(center, radius, vertices, normals)


def _convex_aabb_contacts(vertices: list[Vec3], normals: list[Vec3], bounds: Bounds) -> list[_TriangleContact]:
    axes = list(normals) + [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]
    return _convex_points_contacts(vertices, _bounds_corners(bounds), axes)


def _sphere_against_points_contacts(center: Vec3, radius: float, vertices: list[Vec3], normals: list[Vec3]) -> list[_TriangleContact]:
    axes = list(normals)
    closest = min(vertices, key=lambda point: _length(_sub(point, center)), default=None)
    if closest is not None:
        axes.append(_sub(center, closest))
    return _sat_projected_sphere_contacts(vertices, center, radius, axes)


def _convex_points_contacts(a: list[Vec3], b: list[Vec3], axes: list[Vec3]) -> list[_TriangleContact]:
    best_axis = Vec3(0.0, 1.0, 0.0)
    best_overlap: float | None = None
    center_a = _points_center(a)
    center_b = _points_center(b)
    for raw_axis in axes:
        axis = _normalize(raw_axis)
        if _length(axis) < 0.000001:
            continue
        a_min, a_max = _project_points(a, axis)
        b_min, b_max = _project_points(b, axis)
        overlap = min(a_max, b_max) - max(a_min, b_min)
        if overlap < 0:
            return []
        if best_overlap is None or overlap < best_overlap:
            best_overlap = overlap
            best_axis = axis
    if best_overlap is None:
        return []
    if _dot(best_axis, _sub(center_b, center_a)) < 0:
        best_axis = Vec3(-best_axis.x, -best_axis.y, -best_axis.z)
    return [_TriangleContact(best_axis, max(best_overlap, 0.0))]


def _sat_projected_sphere_contacts(vertices: list[Vec3], center: Vec3, radius: float, axes: list[Vec3]) -> list[_TriangleContact]:
    best_axis = Vec3(0.0, 1.0, 0.0)
    best_overlap: float | None = None
    hull_center = _points_center(vertices)
    for raw_axis in axes:
        axis = _normalize(raw_axis)
        if _length(axis) < 0.000001:
            continue
        hull_min, hull_max = _project_points(vertices, axis)
        sphere_center = _dot(center, axis)
        sphere_min, sphere_max = sphere_center - radius, sphere_center + radius
        overlap = min(hull_max, sphere_max) - max(hull_min, sphere_min)
        if overlap < 0:
            return []
        if best_overlap is None or overlap < best_overlap:
            best_overlap = overlap
            best_axis = axis
    if best_overlap is None:
        return []
    if _dot(best_axis, _sub(center, hull_center)) < 0:
        best_axis = Vec3(-best_axis.x, -best_axis.y, -best_axis.z)
    return [_TriangleContact(best_axis, max(best_overlap, 0.0))]


def _world_convex_points(entity: Entity, collider: Collider, project: Any) -> tuple[list[Vec3], list[Vec3]]:
    renderer = mesh_renderer_for(entity)
    hull = convex_hull(project, renderer) if renderer else None
    if hull is None:
        return [], []
    matrix = entity.transform.world_matrix(entity)
    vertices = [transform_point(matrix, point) for point in hull.vertices]
    normals = [_transform_direction(matrix, normal) for normal in hull.normals]
    return vertices, normals


def _sphere_mesh_contacts(center: Vec3, radius: float, entity: Entity, project: Any) -> list[_TriangleContact]:
    renderer = mesh_renderer_for(entity)
    if renderer is None:
        return []
    matrix = entity.transform.world_matrix(entity)
    broadphase = Bounds(
        Vec3(center.x - radius, center.y - radius, center.z - radius),
        Vec3(center.x + radius, center.y + radius, center.z + radius),
    )
    contacts: list[_TriangleContact] = []
    for triangle in mesh_triangles(project, renderer):
        world_triangle = transform_triangle(matrix, triangle)
        tri_min, tri_max = bounds_from_points(world_triangle)
        if not broadphase.overlaps(Bounds(tri_min, tri_max)):
            continue
        contact = _sphere_triangle_contact(center, radius, world_triangle)
        if contact is not None:
            contacts.append(contact)
    return contacts


def _aabb_triangle_contact(bounds: Bounds, triangle: tuple[Vec3, Vec3, Vec3]) -> _TriangleContact | None:
    center = Vec3(
        (bounds.min.x + bounds.max.x) * 0.5,
        (bounds.min.y + bounds.max.y) * 0.5,
        (bounds.min.z + bounds.max.z) * 0.5,
    )
    half = Vec3(
        (bounds.max.x - bounds.min.x) * 0.5,
        (bounds.max.y - bounds.min.y) * 0.5,
        (bounds.max.z - bounds.min.z) * 0.5,
    )
    shifted = [Vec3(point.x - center.x, point.y - center.y, point.z - center.z) for point in triangle]
    edges = [
        _sub(shifted[1], shifted[0]),
        _sub(shifted[2], shifted[1]),
        _sub(shifted[0], shifted[2]),
    ]
    axes = [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1), _cross(edges[0], edges[1])]
    for edge in edges:
        for axis in (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)):
            axes.append(_cross(edge, axis))
    best_axis = Vec3(0, 1, 0)
    best_overlap: float | None = None
    for raw_axis in axes:
        axis = _normalize(raw_axis)
        if _length(axis) < 0.000001:
            continue
        triangle_min, triangle_max = _project_triangle(shifted, axis)
        box_radius = half.x * abs(axis.x) + half.y * abs(axis.y) + half.z * abs(axis.z)
        overlap = min(box_radius, triangle_max) - max(-box_radius, triangle_min)
        if overlap < 0:
            return None
        if best_overlap is None or overlap < best_overlap:
            best_overlap = overlap
            best_axis = axis
    normal = best_axis
    triangle_center = Vec3(
        sum(point.x for point in shifted) / 3.0,
        sum(point.y for point in shifted) / 3.0,
        sum(point.z for point in shifted) / 3.0,
    )
    if _dot(normal, triangle_center) < 0:
        normal = Vec3(-normal.x, -normal.y, -normal.z)
    return _TriangleContact(normal, max(best_overlap or 0.0, 0.0))


def _sphere_triangle_contact(center: Vec3, radius: float, triangle: tuple[Vec3, Vec3, Vec3]) -> _TriangleContact | None:
    closest = _closest_point_on_triangle(center, triangle)
    offset = _sub(center, closest)
    distance = _length(offset)
    if distance > radius:
        return None
    normal = _normalize(offset)
    if _length(normal) < 0.000001:
        normal = _triangle_normal(triangle)
    return _TriangleContact(normal, max(radius - distance, 0.0))


def _closest_point_on_triangle(point: Vec3, triangle: tuple[Vec3, Vec3, Vec3]) -> Vec3:
    a, b, c = triangle
    ab = _sub(b, a)
    ac = _sub(c, a)
    ap = _sub(point, a)
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = _sub(point, b)
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return Vec3(a.x + ab.x * v, a.y + ab.y * v, a.z + ab.z * v)
    cp = _sub(point, c)
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return Vec3(a.x + ac.x * w, a.y + ac.y * w, a.z + ac.z * w)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        bc = _sub(c, b)
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return Vec3(b.x + bc.x * w, b.y + bc.y * w, b.z + bc.z * w)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return Vec3(a.x + ab.x * v + ac.x * w, a.y + ab.y * v + ac.y * w, a.z + ab.z * v + ac.z * w)


def _triangle_normal(triangle: tuple[Vec3, Vec3, Vec3]) -> Vec3:
    a, b, c = triangle
    return _normalize(_cross(_sub(b, a), _sub(c, a)))


def _project_triangle(points: list[Vec3], axis: Vec3) -> tuple[float, float]:
    values = [_dot(point, axis) for point in points]
    return min(values), max(values)


def _project_points(points: list[Vec3], axis: Vec3) -> tuple[float, float]:
    values = [_dot(point, axis) for point in points]
    return min(values), max(values)


def _points_center(points: list[Vec3]) -> Vec3:
    return Vec3(
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
        sum(point.z for point in points) / len(points),
    )


def _bounds_corners(bounds: Bounds) -> list[Vec3]:
    x0, y0, z0 = bounds.min.x, bounds.min.y, bounds.min.z
    x1, y1, z1 = bounds.max.x, bounds.max.y, bounds.max.z
    return [
        Vec3(x0, y0, z0), Vec3(x1, y0, z0), Vec3(x1, y1, z0), Vec3(x0, y1, z0),
        Vec3(x0, y0, z1), Vec3(x1, y0, z1), Vec3(x1, y1, z1), Vec3(x0, y1, z1),
    ]


def _transform_direction(matrix: list[float], direction: Vec3) -> Vec3:
    return _normalize(Vec3(
        matrix[0] * direction.x + matrix[1] * direction.y + matrix[2] * direction.z,
        matrix[4] * direction.x + matrix[5] * direction.y + matrix[6] * direction.z,
        matrix[8] * direction.x + matrix[9] * direction.y + matrix[10] * direction.z,
    ))


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z)


def _dot(a: Vec3, b: Vec3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def _length(vector: Vec3) -> float:
    return sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)


def _normalize(vector: Vec3) -> Vec3:
    length = _length(vector)
    if length < 0.000001:
        return Vec3()
    return Vec3(vector.x / length, vector.y / length, vector.z / length)


def _layers_can_collide(a: Collider, b: Collider) -> bool:
    return _mask_allows(a.mask, b.layer) and _mask_allows(b.mask, a.layer)


def _mask_allows(mask: str, layer: str) -> bool:
    if not mask or mask == "*":
        return True
    return layer in {item.strip() for item in mask.split(",") if item.strip()}


def _separation(a: Bounds, b: Bounds) -> tuple[Vec3, float]:
    overlaps = [
        (min(a.max.x - b.min.x, b.max.x - a.min.x), Vec3(1.0 if a.min.x < b.min.x else -1.0, 0.0, 0.0)),
        (min(a.max.y - b.min.y, b.max.y - a.min.y), Vec3(0.0, 1.0 if a.min.y < b.min.y else -1.0, 0.0)),
        (min(a.max.z - b.min.z, b.max.z - a.min.z), Vec3(0.0, 0.0, 1.0 if a.min.z < b.min.z else -1.0)),
    ]
    depth, normal = min(overlaps, key=lambda item: item[0])
    if sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z) < 0.001:
        normal = Vec3(0.0, 1.0, 0.0)
    return normal, max(depth, 0.0)
