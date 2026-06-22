from __future__ import annotations

from typing import Any

from p64.engine.math import Mat4, Vec3, identity


def world_matrix(entity: Any | None) -> Mat4:
    if entity is None:
        return identity()
    return entity.transform.world_matrix(entity)


def parent_world_matrix(entity: Any | None) -> Mat4:
    parent = getattr(entity, "parent", None)
    return world_matrix(parent)


def world_position(entity: Any | None) -> Vec3:
    matrix = world_matrix(entity)
    return Vec3(matrix[3], matrix[7], matrix[11])


def world_rotation(entity: Any | None) -> Vec3:
    rotation = Vec3()
    current = entity
    while current is not None:
        rotation.x += current.transform.rotation.x
        rotation.y += current.transform.rotation.y
        rotation.z += current.transform.rotation.z
        current = current.parent
    return rotation


def world_scale(entity: Any | None) -> Vec3:
    scale = Vec3(1.0, 1.0, 1.0)
    chain: list[Any] = []
    current = entity
    while current is not None:
        chain.append(current)
        current = current.parent
    for item in reversed(chain):
        local = item.transform.scale
        scale.x *= local.x
        scale.y *= local.y
        scale.z *= local.z
    return scale


def local_to_world_direction(entity: Any | None, direction: Vec3) -> Vec3:
    matrix = world_matrix(entity)
    return Vec3(
        matrix[0] * direction.x + matrix[1] * direction.y + matrix[2] * direction.z,
        matrix[4] * direction.x + matrix[5] * direction.y + matrix[6] * direction.z,
        matrix[8] * direction.x + matrix[9] * direction.y + matrix[10] * direction.z,
    ).normalized()


def world_right(entity: Any | None) -> Vec3:
    return local_to_world_direction(entity, Vec3(1.0, 0.0, 0.0))


def world_up(entity: Any | None) -> Vec3:
    return local_to_world_direction(entity, Vec3(0.0, 1.0, 0.0))


def world_forward(entity: Any | None) -> Vec3:
    return local_to_world_direction(entity, Vec3(0.0, 0.0, -1.0))


def world_to_parent_local_point(entity: Any, point: Vec3) -> Vec3:
    inverse_parent = _inverse_affine(parent_world_matrix(entity))
    return transform_point(inverse_parent, point)


def set_world_position(entity: Any, position: Vec3) -> None:
    entity.transform.position = world_to_parent_local_point(entity, position)


def set_world_rotation(entity: Any, rotation: Vec3) -> None:
    parent_rotation = world_rotation(getattr(entity, "parent", None))
    entity.transform.rotation = Vec3(
        rotation.x - parent_rotation.x,
        rotation.y - parent_rotation.y,
        rotation.z - parent_rotation.z,
    )


def transform_point(matrix: Mat4, point: Vec3) -> Vec3:
    return Vec3(
        matrix[0] * point.x + matrix[1] * point.y + matrix[2] * point.z + matrix[3],
        matrix[4] * point.x + matrix[5] * point.y + matrix[6] * point.z + matrix[7],
        matrix[8] * point.x + matrix[9] * point.y + matrix[10] * point.z + matrix[11],
    )


def _inverse_affine(matrix: Mat4) -> Mat4:
    a, b, c = matrix[0], matrix[1], matrix[2]
    d, e, f = matrix[4], matrix[5], matrix[6]
    g, h, i = matrix[8], matrix[9], matrix[10]
    tx, ty, tz = matrix[3], matrix[7], matrix[11]

    det = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(det) <= 0.000001:
        return identity()
    inv_det = 1.0 / det
    inv = [
        (e * i - f * h) * inv_det,
        (c * h - b * i) * inv_det,
        (b * f - c * e) * inv_det,
        0.0,
        (f * g - d * i) * inv_det,
        (a * i - c * g) * inv_det,
        (c * d - a * f) * inv_det,
        0.0,
        (d * h - e * g) * inv_det,
        (b * g - a * h) * inv_det,
        (a * e - b * d) * inv_det,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    inv[3] = -(inv[0] * tx + inv[1] * ty + inv[2] * tz)
    inv[7] = -(inv[4] * tx + inv[5] * ty + inv[6] * tz)
    inv[11] = -(inv[8] * tx + inv[9] * ty + inv[10] * tz)
    return inv
