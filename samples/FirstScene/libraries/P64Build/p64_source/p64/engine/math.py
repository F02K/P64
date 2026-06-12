from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin


@dataclass(slots=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_value(cls, value: list[float] | tuple[float, float, float] | "Vec3") -> "Vec3":
        if isinstance(value, Vec3):
            return cls(value.x, value.y, value.z)
        return cls(float(value[0]), float(value[1]), float(value[2]))

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]


Mat4 = list[float]


def identity() -> Mat4:
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def multiply(a: Mat4, b: Mat4) -> Mat4:
    out = [0.0] * 16
    for row in range(4):
        for col in range(4):
            out[row * 4 + col] = sum(a[row * 4 + k] * b[k * 4 + col] for k in range(4))
    return out


def translation(v: Vec3) -> Mat4:
    m = identity()
    m[3] = v.x
    m[7] = v.y
    m[11] = v.z
    return m


def scale(v: Vec3) -> Mat4:
    m = identity()
    m[0] = v.x
    m[5] = v.y
    m[10] = v.z
    return m


def rotation_xyz(degrees: Vec3) -> Mat4:
    rx, ry, rz = radians(degrees.x), radians(degrees.y), radians(degrees.z)
    cx, sx = cos(rx), sin(rx)
    cy, sy = cos(ry), sin(ry)
    cz, sz = cos(rz), sin(rz)

    mx = [
        1.0, 0.0, 0.0, 0.0,
        0.0, cx, -sx, 0.0,
        0.0, sx, cx, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    my = [
        cy, 0.0, sy, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -sy, 0.0, cy, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    mz = [
        cz, -sz, 0.0, 0.0,
        sz, cz, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    return multiply(multiply(mz, my), mx)


def compose_transform(position: Vec3, rotation: Vec3, local_scale: Vec3) -> Mat4:
    return multiply(multiply(translation(position), rotation_xyz(rotation)), scale(local_scale))
