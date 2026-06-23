from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, asin, atan2, cos, degrees, radians, sin, sqrt
from typing import Callable


Number = int | float


@dataclass(slots=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    _on_change: Callable[[], None] | None = field(default=None, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)
        if name in {"x", "y", "z"}:
            try:
                callback = object.__getattribute__(self, "_on_change")
            except AttributeError:
                callback = None
            if callback is not None:
                callback()

    def bind_on_change(self, callback: Callable[[], None] | None) -> "Vec3":
        object.__setattr__(self, "_on_change", callback)
        return self

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


@dataclass(slots=True)
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls()

    @classmethod
    def from_value(cls, value: list[float] | tuple[float, float, float, float] | "Quaternion") -> "Quaternion":
        if isinstance(value, Quaternion):
            return cls(value.x, value.y, value.z, value.w)
        return cls(float(value[0]), float(value[1]), float(value[2]), float(value[3])).normalized()

    @classmethod
    def from_euler(cls, euler_degrees: Vec3) -> "Quaternion":
        hx = radians(euler_degrees.x) * 0.5
        hy = radians(euler_degrees.y) * 0.5
        hz = radians(euler_degrees.z) * 0.5
        qx = cls(sin(hx), 0.0, 0.0, cos(hx))
        qy = cls(0.0, sin(hy), 0.0, cos(hy))
        qz = cls(0.0, 0.0, sin(hz), cos(hz))
        return (qz * qy * qx).normalized()

    @classmethod
    def angle_axis(cls, angle_degrees: float, axis: Vec3) -> "Quaternion":
        normalized_axis = axis.normalized()
        if normalized_axis.length_squared() <= 0.000001:
            return cls.identity()
        half = radians(angle_degrees) * 0.5
        scale_value = sin(half)
        return cls(
            normalized_axis.x * scale_value,
            normalized_axis.y * scale_value,
            normalized_axis.z * scale_value,
            cos(half),
        ).normalized()

    @classmethod
    def look_rotation(cls, forward: Vec3, up: Vec3 | None = None) -> "Quaternion":
        forward = forward.normalized()
        if forward.length_squared() <= 0.000001:
            return cls.identity()
        up = (up or Vec3.up()).normalized()
        right = forward.cross(up).normalized()
        if right.length_squared() <= 0.000001:
            fallback = Vec3.right() if abs(forward.y) > 0.99 else Vec3.up()
            right = forward.cross(fallback).normalized()
        corrected_up = right.cross(forward).normalized()
        return cls.from_matrix([
            right.x, corrected_up.x, -forward.x, 0.0,
            right.y, corrected_up.y, -forward.y, 0.0,
            right.z, corrected_up.z, -forward.z, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])

    @classmethod
    def from_matrix(cls, matrix: "Mat4") -> "Quaternion":
        m00, m11, m22 = matrix[0], matrix[5], matrix[10]
        trace = m00 + m11 + m22
        if trace > 0.0:
            s = sqrt(trace + 1.0) * 2.0
            result = cls((matrix[9] - matrix[6]) / s, (matrix[2] - matrix[8]) / s, (matrix[4] - matrix[1]) / s, 0.25 * s)
        elif m00 > m11 and m00 > m22:
            s = sqrt(max(0.0, 1.0 + m00 - m11 - m22)) * 2.0
            result = cls(0.25 * s, (matrix[1] + matrix[4]) / s, (matrix[2] + matrix[8]) / s, (matrix[9] - matrix[6]) / s)
        elif m11 > m22:
            s = sqrt(max(0.0, 1.0 + m11 - m00 - m22)) * 2.0
            result = cls((matrix[1] + matrix[4]) / s, 0.25 * s, (matrix[6] + matrix[9]) / s, (matrix[2] - matrix[8]) / s)
        else:
            s = sqrt(max(0.0, 1.0 + m22 - m00 - m11)) * 2.0
            result = cls((matrix[2] + matrix[8]) / s, (matrix[6] + matrix[9]) / s, 0.25 * s, (matrix[4] - matrix[1]) / s)
        return result.normalized()

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.w]

    def copy(self) -> "Quaternion":
        return Quaternion(self.x, self.y, self.z, self.w)

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

    def normalized(self) -> "Quaternion":
        magnitude = sqrt(self.length_squared())
        if magnitude <= 0.000001:
            return Quaternion.identity()
        return Quaternion(self.x / magnitude, self.y / magnitude, self.z / magnitude, self.w / magnitude)

    def inverse(self) -> "Quaternion":
        magnitude_squared = self.length_squared()
        if magnitude_squared <= 0.000001:
            return Quaternion.identity()
        return Quaternion(-self.x / magnitude_squared, -self.y / magnitude_squared, -self.z / magnitude_squared, self.w / magnitude_squared)

    def __mul__(self, other: object) -> "Quaternion | Vec3":
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            )
        if isinstance(other, Vec3):
            qvec = Vec3(self.x, self.y, self.z)
            uv = qvec.cross(other)
            uuv = qvec.cross(uv)
            return other + uv * (2.0 * self.w) + uuv * 2.0
        return NotImplemented

    def to_matrix(self) -> "Mat4":
        q = self.normalized()
        xx, yy, zz = q.x * q.x, q.y * q.y, q.z * q.z
        xy, xz, yz = q.x * q.y, q.x * q.z, q.y * q.z
        wx, wy, wz = q.w * q.x, q.w * q.y, q.w * q.z
        return [
            1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0,
            2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0,
            2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]

    def to_euler(self) -> Vec3:
        matrix = self.to_matrix()
        if max(abs(matrix[1]), abs(matrix[4]), abs(matrix[6]), abs(matrix[9])) <= 0.000001:
            return Vec3(0.0, _clean_angle(degrees(atan2(matrix[2], matrix[0]))), 0.0)
        sy = max(-1.0, min(1.0, -matrix[8]))
        y = asin(sy)
        if abs(abs(sy) - 1.0) > 0.000001:
            x = atan2(matrix[9], matrix[10])
            z = atan2(matrix[4], matrix[0])
        else:
            x = atan2(-matrix[6], matrix[5])
            z = 0.0
        return Vec3(_clean_angle(degrees(x)), _clean_angle(degrees(y)), _clean_angle(degrees(z)))

    @classmethod
    def lerp(cls, a: "Quaternion", b: "Quaternion", amount: float) -> "Quaternion":
        amount = clamp(amount, 0.0, 1.0)
        if _quaternion_dot(a, b) < 0.0:
            b = Quaternion(-b.x, -b.y, -b.z, -b.w)
        return Quaternion(
            a.x + (b.x - a.x) * amount,
            a.y + (b.y - a.y) * amount,
            a.z + (b.z - a.z) * amount,
            a.w + (b.w - a.w) * amount,
        ).normalized()

    @classmethod
    def slerp(cls, a: "Quaternion", b: "Quaternion", amount: float) -> "Quaternion":
        amount = clamp(amount, 0.0, 1.0)
        dot_value = _quaternion_dot(a, b)
        if dot_value < 0.0:
            b = Quaternion(-b.x, -b.y, -b.z, -b.w)
            dot_value = -dot_value
        if dot_value > 0.9995:
            return cls.lerp(a, b, amount)
        theta = acos(max(-1.0, min(1.0, dot_value)))
        sin_theta = sin(theta)
        left = sin((1.0 - amount) * theta) / sin_theta
        right = sin(amount * theta) / sin_theta
        return Quaternion(
            a.x * left + b.x * right,
            a.y * left + b.y * right,
            a.z * left + b.z * right,
            a.w * left + b.w * right,
        ).normalized()


def _quaternion_dot(a: Quaternion, b: Quaternion) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w


def _clean_angle(value: float) -> float:
    if abs(value) <= 0.0000000001:
        return 0.0
    rounded = round(value)
    return float(rounded) if abs(value - rounded) <= 0.0000000001 else value


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
    quaternion = Quaternion.from_euler(rotation)
    return quaternion * Vec3.forward(), quaternion * Vec3.right(), quaternion * Vec3.up()


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
    return Quaternion.from_euler(degrees).to_matrix()


def compose_transform(position: Vec3, rotation: Vec3 | Quaternion, local_scale: Vec3) -> Mat4:
    rotation_matrix = rotation.to_matrix() if isinstance(rotation, Quaternion) else rotation_xyz(rotation)
    return multiply(multiply(translation(position), rotation_matrix), scale(local_scale))
