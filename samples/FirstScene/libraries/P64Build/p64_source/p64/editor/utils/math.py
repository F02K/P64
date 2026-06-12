from __future__ import annotations

from math import sqrt

from p64.engine.math import Vec3


def _add_vec3(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z)


def _sub_vec3(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z)


def _scale_vec3(v: Vec3, scale: float) -> Vec3:
    return Vec3(v.x * scale, v.y * scale, v.z * scale)


def _vec3_length(v: Vec3) -> float:
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _normalize_vec3(v: Vec3) -> Vec3:
    length = _vec3_length(v)
    if length <= 0.0001:
        return Vec3()
    return Vec3(v.x / length, v.y / length, v.z / length)


def _lerp_vec3(a: Vec3, b: Vec3, amount: float) -> Vec3:
    amount = max(0.0, min(1.0, amount))
    return Vec3(
        a.x + (b.x - a.x) * amount,
        a.y + (b.y - a.y) * amount,
        a.z + (b.z - a.z) * amount,
    )
