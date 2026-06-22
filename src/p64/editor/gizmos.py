from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from p64.engine.math import Vec3
from p64.engine.transforms import set_world_position, set_world_rotation, world_position, world_rotation


AXIS_COLORS = {
    "x": (220, 70, 70),
    "y": (80, 200, 100),
    "z": (80, 130, 235),
    "center": (240, 210, 80),
}

AXIS_VECTORS = {
    "x": Vec3(1.0, 0.0, 0.0),
    "y": Vec3(0.0, 1.0, 0.0),
    "z": Vec3(0.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class ScreenPoint:
    x: float
    y: float


@dataclass(frozen=True)
class GizmoHandle:
    name: str
    start: ScreenPoint
    end: ScreenPoint
    kind: str = "axis"
    points: tuple[ScreenPoint, ...] = ()


@dataclass(frozen=True)
class TransformSnapshot:
    position: Vec3
    rotation: Vec3
    scale: Vec3
    world_position: Vec3
    world_rotation: Vec3


def transform_snapshot(entity: object) -> TransformSnapshot:
    transform = getattr(entity, "transform")
    return TransformSnapshot(
        position=Vec3.from_value(transform.position),
        rotation=Vec3.from_value(transform.rotation),
        scale=Vec3.from_value(transform.scale),
        world_position=world_position(entity),
        world_rotation=world_rotation(entity),
    )


def hit_test_gizmo(handles: list[GizmoHandle], x: float, y: float, axis_radius: float = 8.0, center_radius: float = 12.0) -> str | None:
    center = next((handle for handle in handles if handle.name == "center"), None)
    if center and _distance(ScreenPoint(x, y), center.start) <= center_radius:
        return "center"

    best: tuple[float, str] | None = None
    point = ScreenPoint(x, y)
    for handle in handles:
        if handle.name == "center":
            continue
        if handle.kind == "ring" and handle.points:
            distance = _polyline_distance(point, handle.points, closed=True)
        else:
            distance = _point_segment_distance(point, handle.start, handle.end)
        if distance <= axis_radius and (best is None or distance < best[0]):
            best = (distance, handle.name)
    return best[1] if best else None


def apply_gizmo_drag(
    target: object,
    tool: str,
    handle: str,
    start: TransformSnapshot,
    dx: float,
    dy: float,
    axis_screen_direction: tuple[float, float] = (1.0, 0.0),
    camera_right: Vec3 | None = None,
    camera_up: Vec3 | None = None,
    camera_forward: Vec3 | None = None,
    world_per_pixel: float = 0.02,
) -> None:
    transform = getattr(target, "transform")
    projected = _project_delta(dx, dy, axis_screen_direction)
    if tool == "move":
        if handle == "center":
            right = camera_right or Vec3(1.0, 0.0, 0.0)
            up = camera_up or Vec3(0.0, 1.0, 0.0)
            set_world_position(target, Vec3(
                start.world_position.x + (right.x * dx - up.x * dy) * world_per_pixel,
                start.world_position.y + (right.y * dx - up.y * dy) * world_per_pixel,
                start.world_position.z + (right.z * dx - up.z * dy) * world_per_pixel,
            ))
            return
        world = start.world_position.copy()
        _set_axis_value(world, handle, _axis_value(start.world_position, handle) + projected * world_per_pixel)
        set_world_position(target, world)
    elif tool == "scale":
        amount = (dx - dy) * 0.01 if handle == "center" else projected * 0.01
        if handle == "center":
            value = max(0.001, start.scale.x + amount)
            transform.scale = Vec3(value, value, value)
        else:
            _set_axis_value(transform.scale, handle, max(0.001, _axis_value(start.scale, handle) + amount))
    elif tool == "rotate":
        axis = _dominant_axis(camera_forward or Vec3(0.0, 0.0, -1.0)) if handle == "center" else handle
        rotation = start.world_rotation.copy()
        _set_axis_value(rotation, axis, _axis_value(start.world_rotation, axis) + projected * 0.5)
        set_world_rotation(target, rotation)


def axis_screen_direction(handle: GizmoHandle) -> tuple[float, float]:
    dx = handle.end.x - handle.start.x
    dy = handle.end.y - handle.start.y
    length = max(sqrt(dx * dx + dy * dy), 0.000001)
    return dx / length, dy / length


def scale_handle_radius(start: TransformSnapshot, current_scale: Vec3, handle: str, base_radius: float = 5.0) -> float:
    if handle == "center":
        start_value = max(abs(start.scale.x), 0.001)
        current_value = max(abs(current_scale.x), 0.001)
    else:
        start_value = max(abs(_axis_value(start.scale, handle)), 0.001)
        current_value = max(abs(_axis_value(current_scale, handle)), 0.001)
    ratio = max(0.25, min(3.0, current_value / start_value))
    return max(3.0, min(13.0, base_radius * sqrt(ratio)))


def _project_delta(dx: float, dy: float, axis_screen_direction: tuple[float, float]) -> float:
    return dx * axis_screen_direction[0] + dy * axis_screen_direction[1]


def _axis_value(vec: Vec3, axis: str) -> float:
    return float(getattr(vec, axis))


def _set_axis_value(vec: Vec3, axis: str, value: float) -> None:
    setattr(vec, axis, value)


def _dominant_axis(direction: Vec3) -> str:
    values = {"x": abs(direction.x), "y": abs(direction.y), "z": abs(direction.z)}
    return max(values, key=values.get)


def _distance(a: ScreenPoint, b: ScreenPoint) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return sqrt(dx * dx + dy * dy)


def _point_segment_distance(point: ScreenPoint, start: ScreenPoint, end: ScreenPoint) -> float:
    vx = end.x - start.x
    vy = end.y - start.y
    length_sq = vx * vx + vy * vy
    if length_sq <= 0.000001:
        return _distance(point, start)
    t = max(0.0, min(1.0, ((point.x - start.x) * vx + (point.y - start.y) * vy) / length_sq))
    closest = ScreenPoint(start.x + vx * t, start.y + vy * t)
    return _distance(point, closest)


def _polyline_distance(point: ScreenPoint, points: tuple[ScreenPoint, ...], closed: bool = False) -> float:
    if not points:
        return float("inf")
    if len(points) == 1:
        return _distance(point, points[0])
    best = float("inf")
    pairs = zip(points, points[1:])
    for start, end in pairs:
        best = min(best, _point_segment_distance(point, start, end))
    if closed:
        best = min(best, _point_segment_distance(point, points[-1], points[0]))
    return best
