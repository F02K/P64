from __future__ import annotations

from typing import Any, Callable

from p64.engine.entity import Entity
from p64.engine.input import InputState, normalize_mouse_button, normalize_qt_key
from p64.engine.math import Vec3
from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.editor.utils.math import _add_vec3, _lerp_vec3, _normalize_vec3, _scale_vec3, _sub_vec3, _vec3_length
from p64.renderer.scene_renderer import RenderCamera, camera_basis


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
                self._recentering_cursor = False
                self._applied_cursor_mode = "normal"
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
                self._bind_qt_framebuffer()
                project = self.project_getter()
                scene = self.scene_getter()
                if not project or not scene:
                    self.ctx.clear(0.16, 0.18, 0.21, 1.0)
                    return
                try:
                    if self.renderer is None or self.renderer_project != project:
                        from p64.renderer.scene_renderer import SceneRenderer

                        self.renderer = SceneRenderer(self.ctx, project, self.logger)
                        self.renderer_project = project
                    camera = self.scene_camera if self.view_mode == "Scene" else None
                    selected = self.selected_getter()
                    self.renderer.render(
                        scene,
                        self.width(),
                        self.height(),
                        camera=camera,
                        selected_entity_id=selected.id if selected else None,
                        show_grid=self.view_mode == "Scene",
                    )
                except Exception as exc:
                    self.ctx.clear(0.16, 0.18, 0.21, 1.0)
                    self.logger(f"Render failed: {exc}")

            def reload_assets(self) -> None:
                if self.renderer:
                    self.renderer.reload_assets()
                self.update()

            def set_view_mode(self, mode: str) -> None:
                self.view_mode = mode
                if mode != "Game":
                    self.reset_runtime_cursor()
                self.setFocus()
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
                self.reset_runtime_cursor()
                super().focusOutEvent(event)

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

            def tick(self, dt: float) -> None:
                pass

            def apply_runtime_cursor(self) -> None:
                pass

            def reset_runtime_cursor(self) -> None:
                pass

    return Viewport


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
