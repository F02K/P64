from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


VALID_CURSOR_MODES = {"normal", "hidden", "locked"}


QT_KEY_NAMES = {
    16777216: "escape",
    16777217: "tab",
    16777219: "backspace",
    16777220: "enter",
    16777221: "enter",
    16777222: "insert",
    16777223: "delete",
    16777232: "home",
    16777233: "end",
    16777234: "left",
    16777235: "up",
    16777236: "right",
    16777237: "down",
    16777238: "page_up",
    16777239: "page_down",
    16777248: "shift",
    16777249: "ctrl",
    16777250: "alt",
    16777251: "meta",
    32: "space",
}


QT_MOUSE_BUTTONS = {
    1: "left_mouse",
    2: "right_mouse",
    4: "middle_mouse",
}


PYGAME_BUTTON_NAMES = {
    0: "south",
    1: "east",
    2: "west",
    3: "north",
    4: "left_shoulder",
    5: "right_shoulder",
    6: "back",
    7: "start",
    8: "guide",
    9: "left_stick",
    10: "right_stick",
}


PYGAME_AXIS_NAMES = {
    0: "left_x",
    1: "left_y",
    2: "right_x",
    3: "right_y",
    4: "left_trigger",
    5: "right_trigger",
}


def normalize_key(key: str) -> str:
    return str(key).strip().lower().replace(" ", "_")


def normalize_qt_key(key: int, text: str = "") -> str:
    key_value = _enum_int(key)
    if text and len(text) == 1 and text.isprintable() and not text.isspace():
        return normalize_key(text)
    if 16777264 <= key_value <= 16777275:
        return f"f{key_value - 16777263}"
    if 48 <= key_value <= 57 or 65 <= key_value <= 90:
        return chr(key_value).lower()
    return QT_KEY_NAMES.get(key_value, str(key_value))


def normalize_mouse_button(button: int | str) -> str:
    if isinstance(button, str):
        value = normalize_key(button)
        aliases = {
            "left": "left_mouse",
            "right": "right_mouse",
            "middle": "middle_mouse",
        }
        return aliases.get(value, value)
    button_value = _enum_int(button)
    return QT_MOUSE_BUTTONS.get(button_value, f"mouse_{button_value}")


def _enum_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(getattr(value, "value", value))


@dataclass
class ControllerSnapshot:
    axes: dict[str, float] = field(default_factory=dict)
    buttons_down: set[str] = field(default_factory=set)
    connected_controllers: list[str] = field(default_factory=list)


class NullControllerBackend:
    error: str | None = None

    def poll(self) -> ControllerSnapshot:
        return ControllerSnapshot()


class PygameControllerBackend:
    def __init__(self) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()

    def poll(self) -> ControllerSnapshot:
        pygame = self.pygame
        pygame.event.pump()
        axes: dict[str, float] = {}
        buttons: set[str] = set()
        controllers: list[str] = []
        for index in range(pygame.joystick.get_count()):
            joystick = pygame.joystick.Joystick(index)
            if not joystick.get_init():
                joystick.init()
            controllers.append(joystick.get_name())
            prefix = "" if index == 0 else f"controller{index}_"
            for axis in range(joystick.get_numaxes()):
                name = PYGAME_AXIS_NAMES.get(axis, f"axis_{axis}")
                value = float(joystick.get_axis(axis))
                axes[f"{prefix}{name}"] = 0.0 if abs(value) < 0.08 else value
            for button in range(joystick.get_numbuttons()):
                if joystick.get_button(button):
                    buttons.add(f"{prefix}{PYGAME_BUTTON_NAMES.get(button, f'button_{button}')}")
            if joystick.get_numhats() > 0:
                hat_x, hat_y = joystick.get_hat(0)
                axes[f"{prefix}dpad_x"] = float(hat_x)
                axes[f"{prefix}dpad_y"] = float(hat_y)
        return ControllerSnapshot(axes=axes, buttons_down=buttons, connected_controllers=controllers)


@dataclass
class InputState:
    keys_down: set[str] = field(default_factory=set)
    mouse_buttons_down: set[str] = field(default_factory=set)
    mouse_position: tuple[float, float] = (0.0, 0.0)
    mouse_delta: tuple[float, float] = (0.0, 0.0)
    wheel_delta: tuple[float, float] = (0.0, 0.0)
    viewport_size: tuple[int, int] = (0, 0)
    cursor_mode: str = "normal"
    controller_axes: dict[str, float] = field(default_factory=dict)
    controller_buttons_down: set[str] = field(default_factory=set)
    connected_controllers: list[str] = field(default_factory=list)
    controller_backend: Any | None = None
    logger: Callable[[str], None] | None = None
    _keys_pressed: set[str] = field(default_factory=set, init=False)
    _keys_released: set[str] = field(default_factory=set, init=False)
    _mouse_pressed: set[str] = field(default_factory=set, init=False)
    _mouse_released: set[str] = field(default_factory=set, init=False)
    _controller_pressed: set[str] = field(default_factory=set, init=False)
    _controller_released: set[str] = field(default_factory=set, init=False)
    _controller_backend_initialized: bool = field(default=False, init=False)
    _last_controller_error: str | None = field(default=None, init=False)

    def begin_frame(self) -> None:
        self._poll_controllers()

    def end_frame(self) -> None:
        self._keys_pressed.clear()
        self._keys_released.clear()
        self._mouse_pressed.clear()
        self._mouse_released.clear()
        self._controller_pressed.clear()
        self._controller_released.clear()
        self.mouse_delta = (0.0, 0.0)
        self.wheel_delta = (0.0, 0.0)

    def press_key(self, key: str) -> None:
        normalized = normalize_key(key)
        if normalized not in self.keys_down:
            self._keys_pressed.add(normalized)
        self._keys_released.discard(normalized)
        self.keys_down.add(normalized)

    def release_key(self, key: str) -> None:
        normalized = normalize_key(key)
        if normalized in self.keys_down:
            self._keys_released.add(normalized)
        self._keys_pressed.discard(normalized)
        self.keys_down.discard(normalized)

    def is_key_down(self, key: str) -> bool:
        return normalize_key(key) in self.keys_down

    def was_key_pressed(self, key: str) -> bool:
        return normalize_key(key) in self._keys_pressed

    def was_key_released(self, key: str) -> bool:
        return normalize_key(key) in self._keys_released

    def press_mouse(self, button: int | str) -> None:
        normalized = normalize_mouse_button(button)
        if normalized not in self.mouse_buttons_down:
            self._mouse_pressed.add(normalized)
        self._mouse_released.discard(normalized)
        self.mouse_buttons_down.add(normalized)

    def release_mouse(self, button: int | str) -> None:
        normalized = normalize_mouse_button(button)
        if normalized in self.mouse_buttons_down:
            self._mouse_released.add(normalized)
        self._mouse_pressed.discard(normalized)
        self.mouse_buttons_down.discard(normalized)

    def is_mouse_down(self, button: int | str) -> bool:
        return normalize_mouse_button(button) in self.mouse_buttons_down

    def was_mouse_pressed(self, button: int | str) -> bool:
        return normalize_mouse_button(button) in self._mouse_pressed

    def was_mouse_released(self, button: int | str) -> bool:
        return normalize_mouse_button(button) in self._mouse_released

    def move_mouse(self, x: float, y: float) -> None:
        previous_x, previous_y = self.mouse_position
        self.mouse_position = (float(x), float(y))
        self.add_mouse_delta(float(x) - previous_x, float(y) - previous_y)

    def add_mouse_delta(self, x: float, y: float) -> None:
        current_x, current_y = self.mouse_delta
        self.mouse_delta = (current_x + float(x), current_y + float(y))

    def add_wheel_delta(self, x: float, y: float) -> None:
        current_x, current_y = self.wheel_delta
        self.wheel_delta = (current_x + float(x), current_y + float(y))

    def set_viewport_size(self, width: int, height: int) -> None:
        self.viewport_size = (max(0, int(width)), max(0, int(height)))

    def is_button_down(self, button: str) -> bool:
        return normalize_key(button) in self.controller_buttons_down

    def was_button_pressed(self, button: str) -> bool:
        return normalize_key(button) in self._controller_pressed

    def was_button_released(self, button: str) -> bool:
        return normalize_key(button) in self._controller_released

    def get_axis(self, axis: str) -> float:
        return float(self.controller_axes.get(normalize_key(axis), 0.0))

    def set_cursor_mode(self, mode: str) -> None:
        normalized = normalize_key(mode)
        self.cursor_mode = normalized if normalized in VALID_CURSOR_MODES else "normal"

    def clear(self) -> None:
        self.keys_down.clear()
        self.mouse_buttons_down.clear()
        self.controller_buttons_down.clear()
        self._keys_pressed.clear()
        self._keys_released.clear()
        self._mouse_pressed.clear()
        self._mouse_released.clear()
        self._controller_pressed.clear()
        self._controller_released.clear()
        self.mouse_delta = (0.0, 0.0)
        self.wheel_delta = (0.0, 0.0)

    def _poll_controllers(self) -> None:
        backend = self._controller_backend()
        try:
            snapshot = backend.poll()
        except Exception as exc:
            self._log_controller_error(f"Controller input disabled: {exc}")
            snapshot = ControllerSnapshot()

        axes = {normalize_key(name): float(value) for name, value in snapshot.axes.items()}
        buttons = {normalize_key(button) for button in snapshot.buttons_down}
        self._controller_pressed = buttons - self.controller_buttons_down
        self._controller_released = self.controller_buttons_down - buttons
        self.controller_axes = axes
        self.controller_buttons_down = buttons
        self.connected_controllers = list(snapshot.connected_controllers)

    def _controller_backend(self) -> Any:
        if self.controller_backend is not None:
            return self.controller_backend
        if self._controller_backend_initialized:
            return NullControllerBackend()
        self._controller_backend_initialized = True
        try:
            self.controller_backend = PygameControllerBackend()
        except Exception as exc:
            self._log_controller_error(f"Controller input disabled: {exc}")
            self.controller_backend = NullControllerBackend()
        return self.controller_backend

    def _log_controller_error(self, message: str) -> None:
        if message == self._last_controller_error:
            return
        self._last_controller_error = message
        if self.logger:
            self.logger(message)
