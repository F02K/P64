from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, sqrt


Number = int | float


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

    def copy(self) -> "Vec3":
        return Vec3(self.x, self.y, self.z)

    @classmethod
    def zero(cls) -> "Vec3":
        return cls()

    @classmethod
    def one(cls) -> "Vec3":
        return cls(1.0, 1.0, 1.0)

    @classmethod
    def up(cls) -> "Vec3":
        return cls(0.0, 1.0, 0.0)

    @classmethod
    def forward(cls) -> "Vec3":
        return cls(0.0, 0.0, -1.0)

    @classmethod
    def right(cls) -> "Vec3":
        return cls(1.0, 0.0, 0.0)

    def __add__(self, other: object) -> "Vec3":
        if not isinstance(other, Vec3):
            return NotImplemented
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: object) -> "Vec3":
        if not isinstance(other, Vec3):
            return NotImplemented
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, scalar: object) -> "Vec3":
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: object) -> "Vec3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> float:
        return sqrt(self.length_squared())

    def normalized(self) -> "Vec3":
        vector_length = self.length()
        if vector_length <= 0.0001:
            return Vec3()
        return self / vector_length

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def lerp(self, other: "Vec3", amount: float) -> "Vec3":
        return lerp_vec3(self, other, amount)


Mat4 = list[float]


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def lerp(a: float, b: float, amount: float) -> float:
    amount = clamp(amount, 0.0, 1.0)
    return a + (b - a) * amount


def dot(a: Vec3, b: Vec3) -> float:
    return a.dot(b)


def cross(a: Vec3, b: Vec3) -> Vec3:
    return a.cross(b)


def length(v: Vec3) -> float:
    return v.length()


def normalize(v: Vec3) -> Vec3:
    return v.normalized()


def lerp_vec3(a: Vec3, b: Vec3, amount: float) -> Vec3:
    amount = clamp(amount, 0.0, 1.0)
    return Vec3(
        lerp(a.x, b.x, amount),
        lerp(a.y, b.y, amount),
        lerp(a.z, b.z, amount),
    )


def forward_from_yaw(yaw_degrees: float) -> Vec3:
    yaw = radians(yaw_degrees)
    return Vec3(sin(yaw), 0.0, -cos(yaw)).normalized()


def basis_from_rotation(rotation: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    pitch = radians(rotation.x)
    yaw = radians(rotation.y)
    forward = Vec3(sin(yaw) * cos(pitch), sin(pitch), -cos(yaw) * cos(pitch)).normalized()
    right = Vec3(cos(yaw), 0.0, sin(yaw)).normalized()
    up = right.cross(forward).normalized()
    return forward, right, up


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
