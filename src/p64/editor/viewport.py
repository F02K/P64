from __future__ import annotations

from contextlib import nullcontext
from math import cos, radians, sin
from typing import Any, Callable

from p64.engine.entity import Entity
from p64.engine.input import InputState, normalize_mouse_button, normalize_qt_key
from p64.engine.math import Vec3
from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.engine.transforms import world_position
from p64.editor.gizmos import AXIS_COLORS, AXIS_VECTORS, GizmoHandle, ScreenPoint, apply_gizmo_drag, axis_screen_direction, hit_test_gizmo, scale_handle_radius, transform_snapshot
from p64.editor.utils.math import _add_vec3, _lerp_vec3, _normalize_vec3, _scale_vec3, _sub_vec3, _vec3_length
from p64.renderer.scene_renderer import RenderCamera, camera_basis, dot


def create_viewport_class(QOpenGLWidget: Any, QWidget: Any, QLabel: Any, QVBoxLayout: Any, Qt: Any) -> type:
    if QOpenGLWidget is not None:
        class Viewport(QOpenGLWidget):  # type: ignore[misc, valid-type]
            def __init__(
                self,
                project_getter: Callable[[], Project | None],
                scene_getter: Callable[[], Scene | None],
                selected_getter: Callable[[], Entity | None],
                selection_setter: Callable[[str | None], None],
                logger: Callable[[str], None],
                input_getter: Callable[[], InputState | None] | None = None,
                scene_change_callback: Callable[[], None] | None = None,
                edit_begin_callback: Callable[[str], None] | None = None,
                edit_commit_callback: Callable[[], None] | None = None,
                profiler_getter: Callable[[], Any | None] | None = None,
            ) -> None:
                super().__init__()
                self.ctx = None
                self.renderer = None
                self.renderer_project: Project | None = None
                self.qt_framebuffer = None
                self.logged_framebuffer = False
                self.view_mode = "Scene"
                self.scene_camera = RenderCamera(position=Vec3(0.0, 2.5, 8.0), rotation=Vec3(-15.0, 0.0, 0.0), fov=60.0)
                self.keys_down: set[int] = set()
                self.mouse_look = False
                self.last_mouse_pos: QPoint | None = None
                self.move_speed = 5.0
                self.scene_camera_velocity = Vec3()
                self.project_getter = project_getter
                self.scene_getter = scene_getter
                self.selected_getter = selected_getter
                self.selection_setter = selection_setter
                self.logger = logger
                self.input_getter = input_getter or (lambda: None)
                self.scene_change_callback = scene_change_callback or (lambda: None)
                self.edit_begin_callback = edit_begin_callback or (lambda _label: None)
                self.edit_commit_callback = edit_commit_callback or (lambda: None)
                self.profiler_getter = profiler_getter or (lambda: None)
                self._recentering_cursor = False
                self._applied_cursor_mode = "normal"
                self.transform_tool = "move"
                self.gizmo_drag: dict[str, Any] | None = None
                self.missing_game_camera = False
                self.logged_missing_game_camera = False
                self.setFocusPolicy(Qt.StrongFocus)
                self.setMouseTracking(True)

            def initializeGL(self) -> None:
                try:
                    import moderngl

                    self.ctx = moderngl.create_context(require=330)
                    self.ctx.enable(moderngl.DEPTH_TEST)
                    self.logger("ModernGL viewport initialized.")
                except Exception as exc:
                    self.ctx = None
                    self.logger(f"P64 viewport could not initialize ModernGL: {exc}")

            def resizeGL(self, width: int, height: int) -> None:
                runtime_input = self._runtime_input()
                if runtime_input is not None:
                    runtime_input.set_viewport_size(width, height)
                if self.ctx:
                    self._bind_qt_framebuffer()
                    self.ctx.viewport = (0, 0, width, height)

            def paintGL(self) -> None:
                if not self.ctx:
                    return
                profiler = self.profiler_getter()
                frame = None
                owns_frame = False
                paint_profiler = None
                if profiler is not None:
                    try:
                        if profiler.current_frame() is None:
                            frame = profiler.begin_frame(self.view_mode)
                            owns_frame = frame is not None
                            paint_profiler = profiler
                    except Exception:
                        frame = None
                        owns_frame = False
                        paint_profiler = None
                self._bind_qt_framebuffer()
                project = self.project_getter()
                scene = self.scene_getter()
                if not project or not scene:
                    self.ctx.clear(0.16, 0.18, 0.21, 1.0)
                    if owns_frame and profiler is not None:
                        try:
                            profiler.end_frame(frame)
                        except Exception:
                            pass
                    return
                try:
                    with _profiler_section(paint_profiler, "viewport paint"):
                        if self.renderer is None or self.renderer_project != project:
                            from p64.renderer.scene_renderer import SceneRenderer

                            self.renderer = SceneRenderer(self.ctx, project, self.logger)
                            self.renderer_project = project
                        self.renderer.profiler_recorder = paint_profiler
                        camera = self.scene_camera if self.view_mode == "Scene" else None
                        selected = self.selected_getter()
                        self.missing_game_camera = not self.renderer.render(
                            scene,
                            self.width(),
                            self.height(),
                            camera=camera,
                            selected_entity_id=selected.id if selected and self.view_mode != "Game" else None,
                            show_grid=self.view_mode == "Scene",
                            game_view=self.view_mode == "Game",
                            output_framebuffer=self.qt_framebuffer,
                        )
                    if self.missing_game_camera and not self.logged_missing_game_camera:
                        self.logger("Game view camera missing: add an active Camera component to an active entity.")
                        self.logged_missing_game_camera = True
                    elif not self.missing_game_camera:
                        self.logged_missing_game_camera = False
                    with _profiler_section(paint_profiler, "viewport overlays"):
                        self._draw_game_camera_overlay()
                        self._draw_ui_bounds_overlay(scene, selected)
                        self._draw_gizmo_overlay(selected)
                except Exception as exc:
                    self.ctx.clear(0.16, 0.18, 0.21, 1.0)
                    self.logger(f"Render failed: {type(exc).__name__}: {exc!r}")
                finally:
                    if owns_frame and profiler is not None:
                        try:
                            profiler.end_frame(frame)
                        except Exception:
                            pass

            def reload_assets(self) -> None:
                if self.renderer:
                    self.renderer.reload_assets()
                self.update()

            def set_view_mode(self, mode: str) -> None:
                self.view_mode = mode
                if mode != "Game":
                    self.reset_runtime_cursor()
                if mode != "Scene":
                    self.gizmo_drag = None
                self.setFocus()
                self.update()

            def set_transform_tool(self, tool: str) -> None:
                self.transform_tool = tool if tool in {"move", "rotate", "scale"} else "move"
                self.gizmo_drag = None
                self.update()

            def tick(self, dt: float) -> None:
                if self.view_mode == "Scene":
                    self._move_scene_camera(dt)
                else:
                    self.apply_runtime_cursor()
                self.update()

            def keyPressEvent(self, event: Any) -> None:
                if not event.isAutoRepeat():
                    runtime_input = self._runtime_input()
                    if runtime_input is not None:
                        runtime_input.press_key(normalize_qt_key(event.key(), _event_text(event)))
                    else:
                        self.keys_down.add(event.key())
                        self._handle_tool_shortcut(event.key())
                super().keyPressEvent(event)

            def keyReleaseEvent(self, event: Any) -> None:
                if not event.isAutoRepeat():
                    runtime_input = self._runtime_input()
                    if runtime_input is not None:
                        runtime_input.release_key(normalize_qt_key(event.key(), _event_text(event)))
                    else:
                        self.keys_down.discard(event.key())
                super().keyReleaseEvent(event)

            def mousePressEvent(self, event: Any) -> None:
                runtime_input = self._runtime_input()
                if runtime_input is not None:
                    runtime_input.press_mouse(normalize_mouse_button(event.button()))
                    x, y = _event_xy(event)
                    runtime_input.move_mouse(x, y)
                    self.setFocus()
                elif event.button() == Qt.RightButton:
                    self.mouse_look = True
                    self.last_mouse_pos = event.position().toPoint()
                    self.setCursor(Qt.BlankCursor)
                    self.setFocus()
                elif event.button() == Qt.LeftButton:
                    if self._start_gizmo_drag(event.position().x(), event.position().y()):
                        event.accept()
                        return
                    self._pick_scene_object(event.position().x(), event.position().y())
                super().mousePressEvent(event)

            def mouseReleaseEvent(self, event: Any) -> None:
                runtime_input = self._runtime_input()
                if runtime_input is not None:
                    runtime_input.release_mouse(normalize_mouse_button(event.button()))
                    x, y = _event_xy(event)
                    runtime_input.move_mouse(x, y)
                elif event.button() == Qt.RightButton:
                    self.mouse_look = False
                    self.last_mouse_pos = None
                    self.unsetCursor()
                elif event.button() == Qt.LeftButton and self.gizmo_drag is not None:
                    self.gizmo_drag = None
                    self.edit_commit_callback()
                    event.accept()
                    return
                super().mouseReleaseEvent(event)

            def mouseMoveEvent(self, event: Any) -> None:
                runtime_input = self._runtime_input()
                if runtime_input is not None:
                    x, y = _event_xy(event)
                    if runtime_input.cursor_mode == "locked":
                        if self._recentering_cursor:
                            self._recentering_cursor = False
                            center = self.rect().center()
                            runtime_input.mouse_position = (float(center.x()), float(center.y()))
                        else:
                            center = self.rect().center()
                            runtime_input.add_mouse_delta(x - center.x(), y - center.y())
                            runtime_input.mouse_position = (float(center.x()), float(center.y()))
                            self._center_runtime_cursor(ignore_next=True)
                    else:
                        runtime_input.move_mouse(x, y)
                elif self.mouse_look and self.last_mouse_pos is not None:
                    current = event.position().toPoint()
                    delta = current - self.last_mouse_pos
                    self.last_mouse_pos = current
                    self.scene_camera.rotation.y += delta.x() * 0.15
                    self.scene_camera.rotation.x = max(-89.0, min(89.0, self.scene_camera.rotation.x - delta.y() * 0.15))
                    self.update()
                elif self.gizmo_drag is not None:
                    self._update_gizmo_drag(event.position().x(), event.position().y())
                    event.accept()
                    return
                super().mouseMoveEvent(event)

            def wheelEvent(self, event: Any) -> None:
                runtime_input = self._runtime_input()
                if runtime_input is not None:
                    delta = event.angleDelta()
                    runtime_input.add_wheel_delta(delta.x(), delta.y())
                elif self.view_mode == "Scene":
                    self.move_speed = max(0.5, min(40.0, self.move_speed + event.angleDelta().y() / 240.0))
                    self.logger(f"Scene camera speed: {self.move_speed:.1f}")
                super().wheelEvent(event)

            def focusOutEvent(self, event: Any) -> None:
                runtime_input = self._runtime_input()
                if runtime_input is not None:
                    runtime_input.clear()
                self.keys_down.clear()
                self.mouse_look = False
                self.last_mouse_pos = None
                self.gizmo_drag = None
                self.reset_runtime_cursor()
                super().focusOutEvent(event)

            def _handle_tool_shortcut(self, key: int) -> None:
                if self.view_mode != "Scene" or self.mouse_look:
                    return
                if key == Qt.Key_W:
                    self.set_transform_tool("move")
                    self.logger("Transform tool: Move")
                elif key == Qt.Key_E:
                    self.set_transform_tool("rotate")
                    self.logger("Transform tool: Rotate")
                elif key == Qt.Key_R:
                    self.set_transform_tool("scale")
                    self.logger("Transform tool: Scale")

            def _move_scene_camera(self, dt: float) -> None:
                dt = max(0.0, min(dt, 0.05))
                forward, right, up = camera_basis(self.scene_camera.rotation)
                speed = self.move_speed * (3.0 if Qt.Key_Shift in self.keys_down else 1.0)
                movement = Vec3()
                if self.mouse_look and Qt.Key_W in self.keys_down:
                    movement = _add_vec3(movement, forward)
                if self.mouse_look and Qt.Key_S in self.keys_down:
                    movement = _sub_vec3(movement, forward)
                if self.mouse_look and Qt.Key_D in self.keys_down:
                    movement = _add_vec3(movement, right)
                if self.mouse_look and Qt.Key_A in self.keys_down:
                    movement = _sub_vec3(movement, right)
                if self.mouse_look and Qt.Key_E in self.keys_down:
                    movement = _add_vec3(movement, up)
                if self.mouse_look and Qt.Key_Q in self.keys_down:
                    movement = _sub_vec3(movement, up)
                movement = _normalize_vec3(movement)
                target_velocity = _scale_vec3(movement, speed)
                smoothing = min(1.0, dt * 14.0)
                self.scene_camera_velocity = _lerp_vec3(self.scene_camera_velocity, target_velocity, smoothing)
                if _vec3_length(self.scene_camera_velocity) < 0.001 and _vec3_length(target_velocity) < 0.001:
                    self.scene_camera_velocity = Vec3()
                    return
                self.scene_camera.position = _add_vec3(self.scene_camera.position, _scale_vec3(self.scene_camera_velocity, dt))

            def _bind_qt_framebuffer(self) -> None:
                if not self.ctx:
                    return
                framebuffer_id = self.defaultFramebufferObject()
                if self.qt_framebuffer is None or getattr(self.qt_framebuffer, "glo", None) != framebuffer_id:
                    self.qt_framebuffer = self.ctx.detect_framebuffer(framebuffer_id)
                    if not self.logged_framebuffer:
                        self.logger(f"Viewport framebuffer bound: {framebuffer_id}")
                        self.logged_framebuffer = True
                self.qt_framebuffer.use()

            def _pick_scene_object(self, x: float, y: float) -> None:
                if self.view_mode != "Scene" or not self.renderer:
                    return
                scene = self.scene_getter()
                if not scene:
                    return
                selected_id = self.renderer.pick_entity(scene, self.width(), self.height(), x, y, camera=self.scene_camera)
                self.selection_setter(selected_id)

            def _start_gizmo_drag(self, x: float, y: float) -> bool:
                if self.view_mode != "Scene" or self.mouse_look:
                    return False
                selected = self.selected_getter()
                if selected is None:
                    return False
                handles = self._gizmo_handles(selected)
                hit = hit_test_gizmo(handles, x, y)
                if not hit:
                    return False
                handle = next((item for item in handles if item.name == hit), None)
                direction = axis_screen_direction(handle) if handle and hit != "center" and handle.kind != "ring" else (1.0, 0.0)
                forward, right, up = camera_basis(self.scene_camera.rotation)
                self.gizmo_drag = {
                    "entity": selected,
                    "tool": self.transform_tool,
                    "handle": hit,
                    "start": transform_snapshot(selected),
                    "mouse": (float(x), float(y)),
                    "axis_direction": direction,
                    "right": right,
                    "up": up,
                    "forward": forward,
                    "world_per_pixel": self._world_per_pixel(selected),
                }
                self.edit_begin_callback(f"{self.transform_tool.title()} {hit.upper() if hit != 'center' else 'Center'}")
                self.setFocus()
                return True

            def _update_gizmo_drag(self, x: float, y: float) -> None:
                if self.gizmo_drag is None:
                    return
                start_x, start_y = self.gizmo_drag["mouse"]
                apply_gizmo_drag(
                    self.gizmo_drag["entity"],
                    self.gizmo_drag["tool"],
                    self.gizmo_drag["handle"],
                    self.gizmo_drag["start"],
                    float(x) - start_x,
                    float(y) - start_y,
                    axis_screen_direction=self.gizmo_drag["axis_direction"],
                    camera_right=self.gizmo_drag["right"],
                    camera_up=self.gizmo_drag["up"],
                    camera_forward=self.gizmo_drag["forward"],
                    world_per_pixel=self.gizmo_drag["world_per_pixel"],
                )
                self.scene_change_callback()
                self.update()

            def _draw_gizmo_overlay(self, selected: Entity | None) -> None:
                if self.view_mode != "Scene" or selected is None:
                    return
                handles = self._gizmo_handles(selected)
                if not handles:
                    return
                try:
                    from PySide6.QtGui import QBrush, QColor, QPainter, QPen

                    painter = QPainter(self)
                    painter.setRenderHint(QPainter.Antialiasing, True)
                    active_scale = self.gizmo_drag if self.gizmo_drag and self.gizmo_drag.get("tool") == "scale" else None
                    for handle in handles:
                        color = QColor(*AXIS_COLORS[handle.name])
                        if handle.name == "center":
                            painter.setPen(QPen(color, 2))
                            painter.setBrush(QBrush(color))
                            radius = 6.0
                            if active_scale and active_scale.get("handle") == "center":
                                radius = scale_handle_radius(active_scale["start"], selected.transform.scale, "center", 6.0)
                            painter.drawEllipse(int(handle.start.x - radius), int(handle.start.y - radius), int(radius * 2), int(radius * 2))
                            continue
                        if handle.kind == "ring" and handle.points:
                            painter.setPen(QPen(color, 3))
                            for start, end in zip(handle.points, handle.points[1:] + handle.points[:1]):
                                painter.drawLine(int(start.x), int(start.y), int(end.x), int(end.y))
                            label = handle.points[0]
                            painter.drawText(int(label.x + 6), int(label.y - 6), handle.name.upper())
                            continue
                        painter.setPen(QPen(color, 3))
                        painter.drawLine(int(handle.start.x), int(handle.start.y), int(handle.end.x), int(handle.end.y))
                        painter.setBrush(QBrush(color))
                        radius = 4.0
                        if active_scale and active_scale.get("handle") == handle.name:
                            radius = scale_handle_radius(active_scale["start"], selected.transform.scale, handle.name, 4.5)
                        painter.drawEllipse(int(handle.end.x - radius), int(handle.end.y - radius), int(radius * 2), int(radius * 2))
                        painter.drawText(int(handle.end.x + 6), int(handle.end.y - 6), handle.name.upper())
                    painter.end()
                except Exception as exc:
                    self.logger(f"Gizmo overlay failed: {exc}")

            def _draw_game_camera_overlay(self) -> None:
                if self.view_mode != "Game" or not self.missing_game_camera:
                    return
                try:
                    from PySide6.QtGui import QColor, QPainter, QPen

                    painter = QPainter(self)
                    painter.setPen(QPen(QColor(220, 224, 230), 1))
                    painter.drawText(self.rect(), Qt.AlignCenter, "No active camera")
                    painter.end()
                except Exception as exc:
                    self.logger(f"Game camera overlay failed: {exc}")

            def _draw_ui_bounds_overlay(self, scene: Scene, selected: Entity | None) -> None:
                if self.view_mode != "Game" or selected is None or selected.rect_transform is None:
                    return
                try:
                    from PySide6.QtGui import QColor, QPainter, QPen
                    from p64.renderer.scene_renderer import ui_layout_debug

                    entry = next((item for item in ui_layout_debug(scene, self.width(), self.height()) if item.entity_id == selected.id), None)
                    if entry is None:
                        return
                    painter = QPainter(self)
                    painter.setRenderHint(QPainter.Antialiasing, False)
                    painter.setPen(QPen(QColor(255, 218, 82), 1))
                    _draw_rect_outline(painter, entry.rect)
                    painter.setPen(QPen(QColor(74, 175, 255), 1))
                    for rect in entry.image_rects:
                        _draw_rect_outline(painter, rect)
                    painter.setPen(QPen(QColor(156, 255, 132), 1))
                    for rect in entry.text_rects:
                        _draw_rect_outline(painter, rect)
                    painter.end()
                except Exception as exc:
                    self.logger(f"UI bounds overlay failed: {exc}")

            def _gizmo_handles(self, selected: Entity) -> list[GizmoHandle]:
                origin = _world_position(selected)
                center = self._project_world(origin)
                if center is None:
                    return []
                length = max(0.6, min(3.5, _vec3_length(_sub_vec3(origin, self.scene_camera.position)) * 0.18))
                handles = [GizmoHandle("center", center, center)]
                if self.transform_tool == "rotate":
                    return handles + self._rotation_gizmo_handles(origin, center, length)
                for name, axis in AXIS_VECTORS.items():
                    end = self._project_world(Vec3(origin.x + axis.x * length, origin.y + axis.y * length, origin.z + axis.z * length))
                    if end is not None:
                        handles.append(GizmoHandle(name, center, end))
                return handles

            def _rotation_gizmo_handles(self, origin: Vec3, center: ScreenPoint, radius: float) -> list[GizmoHandle]:
                handles: list[GizmoHandle] = []
                planes = {
                    "x": (Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0)),
                    "y": (Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0)),
                    "z": (Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)),
                }
                for name, (a, b) in planes.items():
                    points: list[ScreenPoint] = []
                    for index in range(48):
                        angle = radians(index / 48 * 360.0)
                        world = Vec3(
                            origin.x + (a.x * cos(angle) + b.x * sin(angle)) * radius,
                            origin.y + (a.y * cos(angle) + b.y * sin(angle)) * radius,
                            origin.z + (a.z * cos(angle) + b.z * sin(angle)) * radius,
                        )
                        point = self._project_world(world)
                        if point is not None:
                            points.append(point)
                    if len(points) >= 8:
                        handles.append(GizmoHandle(name, center, points[0], kind="ring", points=tuple(points)))
                return handles

            def _world_per_pixel(self, selected: Entity) -> float:
                distance = _vec3_length(_sub_vec3(_world_position(selected), self.scene_camera.position))
                return max(0.002, distance * 0.0025)

            def _project_world(self, point: Vec3) -> ScreenPoint | None:
                view = _view_matrix(self.scene_camera)
                projection = _perspective_matrix(self.scene_camera.fov, max(self.width(), 1) / max(self.height(), 1), self.scene_camera.near, self.scene_camera.far)
                view_point = _mat4_vec4_multiply(view, (point.x, point.y, point.z, 1.0))
                clip = _mat4_vec4_multiply(projection, view_point)
                if abs(clip[3]) < 0.000001 or clip[3] <= 0.0:
                    return None
                ndc_x = clip[0] / clip[3]
                ndc_y = clip[1] / clip[3]
                return ScreenPoint((ndc_x * 0.5 + 0.5) * self.width(), (0.5 - ndc_y * 0.5) * self.height())

            def _runtime_input(self) -> InputState | None:
                if self.view_mode != "Game":
                    return None
                return self.input_getter()

            def apply_runtime_cursor(self) -> None:
                runtime_input = self._runtime_input()
                if runtime_input is None:
                    return
                if runtime_input.cursor_mode == self._applied_cursor_mode:
                    return
                self._applied_cursor_mode = runtime_input.cursor_mode
                if runtime_input.cursor_mode == "normal":
                    self.unsetCursor()
                else:
                    self.setCursor(Qt.BlankCursor)
                    if runtime_input.cursor_mode == "locked":
                        self._center_runtime_cursor(ignore_next=False)

            def reset_runtime_cursor(self) -> None:
                self._recentering_cursor = False
                self._applied_cursor_mode = "normal"
                self.unsetCursor()

            def _center_runtime_cursor(self, ignore_next: bool) -> None:
                try:
                    from PySide6.QtGui import QCursor

                    center = self.rect().center()
                    self._recentering_cursor = ignore_next
                    QCursor.setPos(self.mapToGlobal(center))
                except Exception as exc:
                    self.logger(f"Cursor lock failed: {exc}")
    else:
        class Viewport(QWidget):  # type: ignore[no-redef]
            def __init__(
                self,
                project_getter: Callable[[], Project | None],
                scene_getter: Callable[[], Scene | None],
                selected_getter: Callable[[], Entity | None],
                selection_setter: Callable[[str | None], None],
                logger: Callable[[str], None],
                input_getter: Callable[[], InputState | None] | None = None,
                scene_change_callback: Callable[[], None] | None = None,
                edit_begin_callback: Callable[[str], None] | None = None,
                edit_commit_callback: Callable[[], None] | None = None,
                profiler_getter: Callable[[], Any | None] | None = None,
            ) -> None:
                super().__init__()
                layout = QVBoxLayout(self)
                label = QLabel("P64 Viewport\nInstall PySide6 OpenGL widgets for accelerated rendering.")
                label.setAlignment(Qt.AlignCenter)
                layout.addWidget(label)

            def reload_assets(self) -> None:
                pass

            def set_view_mode(self, mode: str) -> None:
                pass

            def set_transform_tool(self, tool: str) -> None:
                pass

            def tick(self, dt: float) -> None:
                pass

            def apply_runtime_cursor(self) -> None:
                pass

            def reset_runtime_cursor(self) -> None:
                pass

    return Viewport


def _world_position(entity: Entity) -> Vec3:
    return world_position(entity)


def _view_matrix(camera: RenderCamera) -> list[float]:
    forward, right, up = camera_basis(camera.rotation)
    position = camera.position
    return [
        right.x, right.y, right.z, -dot(right, position),
        up.x, up.y, up.z, -dot(up, position),
        -forward.x, -forward.y, -forward.z, dot(forward, position),
        0.0, 0.0, 0.0, 1.0,
    ]


def _perspective_matrix(fov_degrees: float, aspect: float, near: float, far: float) -> list[float]:
    from math import radians, tan

    f = 1.0 / tan(radians(fov_degrees) / 2.0)
    return [
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) / (near - far), (2 * far * near) / (near - far),
        0.0, 0.0, -1.0, 0.0,
    ]


def _mat4_vec4_multiply(matrix: list[float], vector: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2] + matrix[3] * vector[3],
        matrix[4] * vector[0] + matrix[5] * vector[1] + matrix[6] * vector[2] + matrix[7] * vector[3],
        matrix[8] * vector[0] + matrix[9] * vector[1] + matrix[10] * vector[2] + matrix[11] * vector[3],
        matrix[12] * vector[0] + matrix[13] * vector[1] + matrix[14] * vector[2] + matrix[15] * vector[3],
    )


def _event_text(event: Any) -> str:
    try:
        return event.text()
    except Exception:
        return ""


def _event_xy(event: Any) -> tuple[float, float]:
    try:
        position = event.position()
    except Exception:
        position = event.pos()
    return float(position.x()), float(position.y())


def _draw_rect_outline(painter: Any, rect: tuple[float, float, float, float]) -> None:
    x, y, width, height = rect
    painter.drawRect(int(round(x)), int(round(y)), int(round(width)), int(round(height)))


def _profiler_section(profiler: Any | None, name: str) -> Any:
    if profiler is None:
        return nullcontext()
    try:
        return profiler.section(name)
    except Exception:
        return nullcontext()
