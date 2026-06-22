from __future__ import annotations

from p64.engine.math import Vec3, lerp_vec3


def _add_vec3(a: Vec3, b: Vec3) -> Vec3:
    return a + b


def _sub_vec3(a: Vec3, b: Vec3) -> Vec3:
    return a - b


def _scale_vec3(v: Vec3, scale: float) -> Vec3:
    return v * scale


def _vec3_length(v: Vec3) -> float:
    return v.length()


def _normalize_vec3(v: Vec3) -> Vec3:
    return v.normalized()


def _lerp_vec3(a: Vec3, b: Vec3, amount: float) -> Vec3:
    return lerp_vec3(a, b, amount)
