from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Callable

from p64.engine.components import Canvas, RectTransform, UIButton, UIControl, UIScrollView, UISlider, UIToggle
from p64.engine.entity import Entity, entity_effectively_active
from p64.engine.input import InputState
from p64.engine.math import Vec3
from p64.engine.scene import Scene


Rect = tuple[float, float, float, float]
UICallback = Callable[[Entity, str, tuple[Any, ...]], list[str]]


@dataclass(frozen=True)
class UILayoutEntry:
    entity: Entity
    rect: Rect
    clip_rect: Rect | None
    canvas_entity: Entity
    canvas: Canvas
    order: int


def ui_control(entity: Entity) -> UIControl | None:
    return next((component for component in entity.components if isinstance(component, UIControl)), None)


def canvas_layout_size(canvas: Canvas, width: int, height: int) -> tuple[int, int]:
    if canvas.resolution_mode == "fixed":
        return max(1, int(canvas.reference_resolution.x)), max(1, int(canvas.reference_resolution.y))
    return max(1, int(width)), max(1, int(height))


def ui_rect(anchor: str, offset: Vec3, size: Vec3, pivot: Vec3, width: float, height: float) -> Rect:
    anchors = {
        "top-left": (0.0, 0.0), "top": (0.5, 0.0), "top-right": (1.0, 0.0),
        "left": (0.0, 0.5), "center": (0.5, 0.5), "right": (1.0, 0.5),
        "bottom-left": (0.0, 1.0), "bottom": (0.5, 1.0), "bottom-right": (1.0, 1.0),
    }
    ax, ay = anchors.get(anchor, anchors["center"])
    w, h = max(0.001, float(size.x)), max(0.001, float(size.y))
    return width * ax + offset.x - w * pivot.x, height * ay + offset.y - h * pivot.y, w, h


def rect_transform_rect(rect: RectTransform, parent_rect: Rect) -> Rect:
    parent_x, parent_y, parent_w, parent_h = parent_rect
    x, y, w, h = ui_rect(rect.anchor, rect.offset, rect.size, rect.pivot, parent_w, parent_h)
    return parent_x + x, parent_y + y, w, h


def intersect_rect(a: Rect | None, b: Rect | None) -> Rect | None:
    if a is None:
        return b
    if b is None:
        return a
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else (x0, y0, 0.0, 0.0)


def build_ui_layout(scene: Scene, width: int, height: int) -> list[UILayoutEntry]:
    canvases = [
        (entity, component)
        for entity in scene.walk_active()
        for component in entity.components
        if isinstance(component, Canvas) and component.enabled
    ]
    entries: list[UILayoutEntry] = []
    order = 0
    for canvas_entity, canvas in sorted(canvases, key=lambda item: int(item[1].sort_order)):
        layout_w, layout_h = canvas_layout_size(canvas, width, height)
        root_rect: Rect = (0.0, 0.0, float(layout_w), float(layout_h))

        def visit(entity: Entity, parent_rect: Rect, clip: Rect | None, scroll_offset: tuple[float, float] = (0.0, 0.0)) -> None:
            nonlocal order
            if not entity_effectively_active(entity):
                return
            rect = rect_transform_rect(entity.rect_transform, parent_rect) if entity.rect_transform else parent_rect
            if scroll_offset != (0.0, 0.0):
                rect = (rect[0] - scroll_offset[0], rect[1] - scroll_offset[1], rect[2], rect[3])
            current_clip = clip
            entries.append(UILayoutEntry(entity, rect, current_clip, canvas_entity, canvas, order))
            order += 1
            scroll = next((c for c in entity.components if isinstance(c, UIScrollView) and c.enabled), None)
            for child in entity.children:
                child_offset = (0.0, 0.0)
                child_clip = current_clip
                if scroll is not None and (not scroll.content_entity or child.id == scroll.content_entity):
                    child_offset = (scroll.scroll_position.x, scroll.scroll_position.y)
                    child_clip = intersect_rect(current_clip, rect)
                visit(child, rect, child_clip, child_offset)

        visit(canvas_entity, root_rect, None)

    by_id = {entry.entity.id: entry for entry in entries}
    replacements: dict[str, Rect] = {}
    for entry in entries:
        slider = next((c for c in entry.entity.components if isinstance(c, UISlider)), None)
        if slider is None:
            continue
        span = max(0.000001, slider.maximum - slider.minimum)
        value = max(0.0, min(1.0, (slider.value - slider.minimum) / span))
        x, y, w, h = entry.rect
        if slider.fill_entity and slider.fill_entity in by_id:
            replacements[slider.fill_entity] = (x, y, w * value, h) if slider.direction == "horizontal" else (x, y + h * (1.0 - value), w, h * value)
        if slider.handle_entity and slider.handle_entity in by_id:
            handle = by_id[slider.handle_entity].rect
            replacements[slider.handle_entity] = (
                x + w * value - handle[2] * 0.5, handle[1], handle[2], handle[3]
            ) if slider.direction == "horizontal" else (
                handle[0], y + h * (1.0 - value) - handle[3] * 0.5, handle[2], handle[3]
            )
    return [
        UILayoutEntry(entry.entity, replacements.get(entry.entity.id, entry.rect), entry.clip_rect, entry.canvas_entity, entry.canvas, entry.order)
        for entry in entries
    ]


class UIEventSystem:
    def __init__(self, callback: UICallback | None = None) -> None:
        self.callback = callback
        self._scene: Scene | None = None
        self.focused_entity_id: str | None = None
        self.hovered_entity_id: str | None = None
        self.pressed_entity_id: str | None = None
        self._axis_direction: tuple[int, int] = (0, 0)
        self._axis_hold = 0.0
        self._axis_repeat = 0.0
        self._viewport_size = (0, 0)
        self._pointer_active = False
        self._drag_scroll_entity_id: str | None = None
        self._drag_distance = 0.0

    def reset(self, scene: Scene | None = None) -> None:
        if scene:
            for entity in scene.walk():
                control = ui_control(entity)
                if control:
                    control._runtime_hovered = control._runtime_focused = control._runtime_pressed = False
        self.focused_entity_id = self.hovered_entity_id = self.pressed_entity_id = None
        self._axis_direction = (0, 0)
        self._axis_hold = self._axis_repeat = 0.0
        self._pointer_active = False
        self._drag_scroll_entity_id = None
        self._drag_distance = 0.0

    def process(self, scene: Scene, input_state: InputState, dt: float) -> list[str]:
        self._scene = scene
        width, height = input_state.viewport_size
        self._viewport_size = (width, height)
        if width <= 0 or height <= 0:
            return []
        layout = build_ui_layout(scene, width, height)
        controls = [entry for entry in layout if ui_control(entry.entity) is not None]
        errors: list[str] = []
        self._sync_toggle_visuals(scene, controls)
        self._ensure_focus(controls, errors)
        if input_state.mouse_delta != (0.0, 0.0) or input_state.wheel_delta != (0.0, 0.0) or input_state.was_mouse_pressed("left_mouse") or input_state.was_mouse_released("left_mouse"):
            self._pointer_active = True
        hover = self._hit_test(controls, input_state.mouse_position, width, height) if self._pointer_active else None
        hover_id = hover.entity.id if hover else None
        if hover_id != self.hovered_entity_id:
            self._set_hover(scene, self.hovered_entity_id, False, errors)
            self.hovered_entity_id = hover_id
            self._set_hover(scene, hover_id, True, errors)

        started_pointer_press = input_state.was_mouse_pressed("left_mouse") and hover is not None
        if started_pointer_press and hover:
            self.pressed_entity_id = hover.entity.id
            self._set_pressed(scene, self.pressed_entity_id, True)
            self._set_focus(scene, hover.entity.id, errors)
            if not isinstance(ui_control(hover.entity), UISlider):
                drag_owner = self._scroll_owner(hover, controls)
                self._drag_scroll_entity_id = drag_owner.entity.id if drag_owner else None
                self._drag_distance = 0.0
        if input_state.is_mouse_down("left_mouse") and self.pressed_entity_id:
            pressed = next((entry for entry in controls if entry.entity.id == self.pressed_entity_id), None)
            if pressed:
                control = ui_control(pressed.entity)
                if isinstance(control, UISlider):
                    errors.extend(self._set_slider_from_pointer(pressed, control, input_state.mouse_position, width, height))
            if self._drag_scroll_entity_id and not started_pointer_press:
                drag_entity = scene.find(self._drag_scroll_entity_id)
                drag_control = ui_control(drag_entity) if drag_entity else None
                if isinstance(drag_control, UIScrollView):
                    self._drag_distance += abs(input_state.mouse_delta[0]) + abs(input_state.mouse_delta[1])
                    errors.extend(self._scroll(drag_control, -input_state.mouse_delta[0] * drag_control.drag_speed, -input_state.mouse_delta[1] * drag_control.drag_speed, drag_entity))
        if input_state.was_mouse_released("left_mouse"):
            pressed_id = self.pressed_entity_id
            was_drag = self._drag_distance > 4.0
            self._set_pressed(scene, pressed_id, False)
            self.pressed_entity_id = None
            self._drag_scroll_entity_id = None
            self._drag_distance = 0.0
            if pressed_id and hover_id == pressed_id and not was_drag:
                entry = next((item for item in controls if item.entity.id == pressed_id), None)
                if entry:
                    errors.extend(self._activate(entry.entity))

        if hover and input_state.wheel_delta != (0.0, 0.0):
            scroll_entry = self._scroll_owner(hover, controls)
            if scroll_entry:
                scroll = ui_control(scroll_entry.entity)
                if isinstance(scroll, UIScrollView):
                    errors.extend(self._scroll(scroll, -input_state.wheel_delta[0] / 120.0 * scroll.wheel_speed, -input_state.wheel_delta[1] / 120.0 * scroll.wheel_speed, scroll_entry.entity))

        direction = self._navigation_direction(input_state, dt)
        if direction != (0, 0):
            self._pointer_active = False
            self._set_hover(scene, self.hovered_entity_id, False, errors)
            self.hovered_entity_id = None
            errors.extend(self._navigate(controls, direction))
        if self.focused_entity_id and self._submit_pressed(input_state):
            self.pressed_entity_id = self.focused_entity_id
            self._set_pressed(scene, self.pressed_entity_id, True)
        if self.pressed_entity_id == self.focused_entity_id and self._submit_released(input_state):
            target = scene.find(self.pressed_entity_id or "")
            self._set_pressed(scene, self.pressed_entity_id, False)
            self.pressed_entity_id = None
            if target:
                errors.extend(self._activate(target))
        if self.focused_entity_id and (input_state.was_key_pressed("escape") or input_state.was_button_pressed("east")):
            target = scene.find(self.focused_entity_id)
            if target:
                errors.extend(self._emit(target, "on_ui_cancel"))

        focused = scene.find(self.focused_entity_id or "")
        focused_control = ui_control(focused) if focused else None
        if isinstance(focused_control, UIScrollView):
            errors.extend(self._scroll(
                focused_control,
                input_state.get_axis("right_x") * focused_control.stick_speed * dt,
                input_state.get_axis("right_y") * focused_control.stick_speed * dt,
                focused,
            ))
        return errors

    def _ensure_focus(self, controls: list[UILayoutEntry], errors: list[str]) -> None:
        valid = [entry for entry in controls if self._selectable(entry)]
        if self.focused_entity_id and any(entry.entity.id == self.focused_entity_id for entry in valid):
            return
        target = None
        for entry in valid:
            if entry.canvas.initial_focus:
                target = next((candidate for candidate in valid if candidate.entity.id == entry.canvas.initial_focus), None)
                if target:
                    break
        if target is None and valid:
            target = valid[0]
        if target:
            self.focused_entity_id = target.entity.id
            control = ui_control(target.entity)
            if control:
                control._runtime_focused = True
            errors.extend(self._emit(target.entity, "on_ui_focus"))

    def _hit_test(self, controls: list[UILayoutEntry], pointer: tuple[float, float], viewport_w: int, viewport_h: int) -> UILayoutEntry | None:
        for entry in reversed(controls):
            if not self._selectable(entry):
                continue
            layout_w, layout_h = canvas_layout_size(entry.canvas, viewport_w, viewport_h)
            point = (pointer[0] * layout_w / viewport_w, pointer[1] * layout_h / viewport_h)
            if _point_in_rect(point, entry.rect) and (entry.clip_rect is None or _point_in_rect(point, entry.clip_rect)):
                return entry
        return None

    def _selectable(self, entry: UILayoutEntry) -> bool:
        control = ui_control(entry.entity)
        return bool(control and control.enabled and control.interactable and entry.rect[2] > 0 and entry.rect[3] > 0 and (entry.clip_rect is None or intersect_rect(entry.rect, entry.clip_rect)[2] > 0))

    def _set_focus(self, scene: Scene, entity_id: str | None, errors: list[str]) -> None:
        if entity_id == self.focused_entity_id:
            return
        old = scene.find(self.focused_entity_id or "")
        if old:
            control = ui_control(old)
            if control:
                control._runtime_focused = False
            errors.extend(self._emit(old, "on_ui_blur"))
        self.focused_entity_id = entity_id
        new = scene.find(entity_id or "")
        if new:
            control = ui_control(new)
            if control:
                control._runtime_focused = True
            errors.extend(self._emit(new, "on_ui_focus"))
            errors.extend(self._ensure_focus_visible(new))

    def _set_hover(self, scene: Scene, entity_id: str | None, value: bool, errors: list[str]) -> None:
        entity = scene.find(entity_id or "")
        if entity:
            control = ui_control(entity)
            if control:
                control._runtime_hovered = value
            errors.extend(self._emit(entity, "on_ui_pointer_enter" if value else "on_ui_pointer_exit"))

    def _set_pressed(self, scene: Scene, entity_id: str | None, value: bool) -> None:
        entity = scene.find(entity_id or "")
        control = ui_control(entity) if entity else None
        if control:
            control._runtime_pressed = value

    def _activate(self, entity: Entity) -> list[str]:
        control = ui_control(entity)
        if not control or not control.enabled or not control.interactable:
            return []
        if isinstance(control, UIToggle):
            control.is_on = not control.is_on
            if self._scene and control.checkmark_entity:
                mark = self._scene.find(control.checkmark_entity)
                if mark:
                    mark.active = control.is_on
            return self._emit(entity, "on_ui_value_changed", control.is_on)
        if isinstance(control, UIButton):
            return self._emit(entity, "on_ui_click")
        return []

    def _set_slider_from_pointer(self, entry: UILayoutEntry, slider: UISlider, pointer: tuple[float, float], viewport_w: int, viewport_h: int) -> list[str]:
        layout_w, layout_h = canvas_layout_size(entry.canvas, viewport_w, viewport_h)
        px, py = pointer[0] * layout_w / viewport_w, pointer[1] * layout_h / viewport_h
        x, y, w, h = entry.rect
        normalized = (px - x) / max(w, 0.001) if slider.direction == "horizontal" else 1.0 - (py - y) / max(h, 0.001)
        return self._set_slider(entry.entity, slider, slider.minimum + max(0.0, min(1.0, normalized)) * (slider.maximum - slider.minimum))

    def _set_slider(self, entity: Entity, slider: UISlider, value: float) -> list[str]:
        value = max(min(slider.minimum, slider.maximum), min(max(slider.minimum, slider.maximum), value))
        if slider.step > 0:
            value = slider.minimum + round((value - slider.minimum) / slider.step) * slider.step
        if abs(value - slider.value) < 0.000001:
            return []
        slider.value = value
        return self._emit(entity, "on_ui_value_changed", slider.value)

    def _scroll(self, scroll: UIScrollView, dx: float, dy: float, entity: Entity) -> list[str]:
        old = (scroll.scroll_position.x, scroll.scroll_position.y)
        max_x = max_y = inf
        if self._scene and scroll.content_entity and self._viewport_size[0] > 0:
            entries = build_ui_layout(self._scene, *self._viewport_size)
            owner = next((entry for entry in entries if entry.entity.id == entity.id), None)
            content = next((entry for entry in entries if entry.entity.id == scroll.content_entity), None)
            if owner and content:
                max_x = max(0.0, content.rect[2] - owner.rect[2])
                max_y = max(0.0, content.rect[3] - owner.rect[3])
        if scroll.horizontal:
            scroll.scroll_position.x = min(max_x, max(0.0, scroll.scroll_position.x + dx))
        if scroll.vertical:
            scroll.scroll_position.y = min(max_y, max(0.0, scroll.scroll_position.y + dy))
        if old == (scroll.scroll_position.x, scroll.scroll_position.y):
            return []
        return self._emit(entity, "on_ui_scroll_changed", scroll.scroll_position.x, scroll.scroll_position.y)

    def _navigate(self, controls: list[UILayoutEntry], direction: tuple[int, int]) -> list[str]:
        current = next((entry for entry in controls if entry.entity.id == self.focused_entity_id), None)
        if current is None:
            return []
        control = ui_control(current.entity)
        if isinstance(control, UISlider) and ((control.direction == "horizontal" and direction[0]) or (control.direction == "vertical" and direction[1])):
            amount = control.step if control.step > 0 else (control.maximum - control.minimum) / 20.0
            sign = direction[0] if control.direction == "horizontal" else -direction[1]
            return self._set_slider(current.entity, control, control.value + amount * sign)
        attr = {(0, -1): "navigation_up", (0, 1): "navigation_down", (-1, 0): "navigation_left", (1, 0): "navigation_right"}[direction]
        explicit = getattr(control, attr, "") if control else ""
        target = next((entry for entry in controls if entry.entity.id == explicit and self._selectable(entry)), None)
        if target is None:
            cx, cy = _center(current.rect)
            best, best_score = None, inf
            for candidate in controls:
                if candidate.entity.id == current.entity.id or candidate.canvas_entity.id != current.canvas_entity.id or not self._selectable(candidate):
                    continue
                tx, ty = _center(candidate.rect)
                dx, dy = tx - cx, ty - cy
                primary = dx * direction[0] + dy * direction[1]
                if primary <= 0:
                    continue
                perpendicular = abs(dx * direction[1] - dy * direction[0])
                score = primary + perpendicular * 2.0
                if score < best_score:
                    best, best_score = candidate, score
            target = best
        if target is None:
            return []
        errors: list[str] = []
        if self._scene:
            self._set_focus(self._scene, target.entity.id, errors)
        return errors

    def _navigation_direction(self, input_state: InputState, dt: float) -> tuple[int, int]:
        digital = (
            int(input_state.was_key_pressed("right") or input_state.was_key_pressed("d")) - int(input_state.was_key_pressed("left") or input_state.was_key_pressed("a")),
            int(input_state.was_key_pressed("down") or input_state.was_key_pressed("s")) - int(input_state.was_key_pressed("up") or input_state.was_key_pressed("w")),
        )
        if digital != (0, 0):
            return _cardinal(digital)
        axis = (
            input_state.get_axis("dpad_x") or input_state.get_axis("left_x"),
            -(input_state.get_axis("dpad_y") or 0.0) + (input_state.get_axis("left_y") or 0.0),
        )
        direction = _cardinal(axis) if max(abs(axis[0]), abs(axis[1])) >= 0.55 else (0, 0)
        if direction == (0, 0):
            self._axis_direction, self._axis_hold, self._axis_repeat = (0, 0), 0.0, 0.0
            return (0, 0)
        if direction != self._axis_direction:
            self._axis_direction, self._axis_hold, self._axis_repeat = direction, 0.0, 0.0
            return direction
        self._axis_hold += dt
        self._axis_repeat += dt
        if self._axis_hold >= 0.35 and self._axis_repeat >= 0.10:
            self._axis_repeat = 0.0
            return direction
        return (0, 0)

    def _submit_pressed(self, input_state: InputState) -> bool:
        return input_state.was_key_pressed("enter") or input_state.was_key_pressed("space") or input_state.was_button_pressed("south")

    def _submit_released(self, input_state: InputState) -> bool:
        return input_state.was_key_released("enter") or input_state.was_key_released("space") or input_state.was_button_released("south")

    def _scroll_owner(self, hovered: UILayoutEntry, controls: list[UILayoutEntry]) -> UILayoutEntry | None:
        current: Entity | None = hovered.entity
        while current:
            match = next((entry for entry in controls if entry.entity.id == current.id and isinstance(ui_control(current), UIScrollView)), None)
            if match:
                return match
            current = current.parent
        return None

    def _sync_toggle_visuals(self, scene: Scene, controls: list[UILayoutEntry]) -> None:
        for entry in controls:
            toggle = ui_control(entry.entity)
            if isinstance(toggle, UIToggle) and toggle.checkmark_entity:
                mark = scene.find(toggle.checkmark_entity)
                if mark:
                    mark.active = toggle.is_on

    def _ensure_focus_visible(self, entity: Entity) -> list[str]:
        if not self._scene or self._viewport_size[0] <= 0:
            return []
        errors: list[str] = []
        entries = build_ui_layout(self._scene, *self._viewport_size)
        target = next((entry for entry in entries if entry.entity.id == entity.id), None)
        current = entity.parent
        while target and current:
            scroll = ui_control(current)
            owner = next((entry for entry in entries if entry.entity.id == current.id), None)
            if isinstance(scroll, UIScrollView) and owner:
                dx = dy = 0.0
                if target.rect[0] < owner.rect[0]:
                    dx = target.rect[0] - owner.rect[0]
                elif target.rect[0] + target.rect[2] > owner.rect[0] + owner.rect[2]:
                    dx = target.rect[0] + target.rect[2] - owner.rect[0] - owner.rect[2]
                if target.rect[1] < owner.rect[1]:
                    dy = target.rect[1] - owner.rect[1]
                elif target.rect[1] + target.rect[3] > owner.rect[1] + owner.rect[3]:
                    dy = target.rect[1] + target.rect[3] - owner.rect[1] - owner.rect[3]
                errors.extend(self._scroll(scroll, dx, dy, current))
            current = current.parent
        return errors

    def _emit(self, entity: Entity, method: str, *args: Any) -> list[str]:
        return self.callback(entity, method, args) if self.callback else []


def _point_in_rect(point: tuple[float, float], rect: Rect) -> bool:
    return rect[0] <= point[0] <= rect[0] + rect[2] and rect[1] <= point[1] <= rect[1] + rect[3]


def _center(rect: Rect) -> tuple[float, float]:
    return rect[0] + rect[2] * 0.5, rect[1] + rect[3] * 0.5


def _cardinal(value: tuple[float, float] | tuple[int, int]) -> tuple[int, int]:
    if abs(value[0]) >= abs(value[1]):
        return (1 if value[0] > 0 else -1, 0) if value[0] else (0, 0)
    return (0, 1 if value[1] > 0 else -1)
