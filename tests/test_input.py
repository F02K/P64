from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from p64.engine.components import ScriptComponent, ScriptEntry
from p64.engine.entity import Entity
from p64.engine.input import ControllerSnapshot, InputState, normalize_qt_key
from p64.engine.project import Project
from p64.engine.runtime_session import RuntimeSession


class FakeControllerBackend:
    def __init__(self, snapshots: list[ControllerSnapshot]) -> None:
        self.snapshots = snapshots

    def poll(self) -> ControllerSnapshot:
        if self.snapshots:
            return self.snapshots.pop(0)
        return ControllerSnapshot()


class InputTests(unittest.TestCase):
    def test_keyboard_pressed_down_released_transitions(self):
        input_state = InputState(controller_backend=FakeControllerBackend([]))

        input_state.press_key("W")

        self.assertTrue(input_state.is_key_down("w"))
        self.assertTrue(input_state.was_key_pressed("w"))
        self.assertFalse(input_state.was_key_released("w"))

        input_state.end_frame()

        self.assertTrue(input_state.is_key_down("w"))
        self.assertFalse(input_state.was_key_pressed("w"))

        input_state.release_key("w")

        self.assertFalse(input_state.is_key_down("w"))
        self.assertTrue(input_state.was_key_released("w"))

    def test_mouse_button_position_delta_and_wheel(self):
        input_state = InputState(controller_backend=FakeControllerBackend([]))

        input_state.press_mouse("left")
        input_state.move_mouse(10, 20)
        input_state.move_mouse(13, 18)
        input_state.add_wheel_delta(0, 120)
        input_state.set_viewport_size(640, 480)

        self.assertTrue(input_state.is_mouse_down("left_mouse"))
        self.assertTrue(input_state.was_mouse_pressed("left_mouse"))
        self.assertEqual(input_state.mouse_position, (13.0, 18.0))
        self.assertEqual(input_state.mouse_delta, (13.0, 18.0))
        self.assertEqual(input_state.wheel_delta, (0.0, 120.0))
        self.assertEqual(input_state.viewport_size, (640, 480))

        input_state.end_frame()
        input_state.release_mouse(1)

        self.assertEqual(input_state.mouse_delta, (0.0, 0.0))
        self.assertEqual(input_state.wheel_delta, (0.0, 0.0))
        self.assertTrue(input_state.was_mouse_released("left_mouse"))

    def test_cursor_mode_validation(self):
        input_state = InputState(controller_backend=FakeControllerBackend([]))

        input_state.set_cursor_mode("locked")
        self.assertEqual(input_state.cursor_mode, "locked")

        input_state.set_cursor_mode("not a mode")
        self.assertEqual(input_state.cursor_mode, "normal")

    def test_qt_key_normalization_uses_script_friendly_names(self):
        self.assertEqual(normalize_qt_key(32, ""), "space")
        self.assertEqual(normalize_qt_key(16777216, ""), "escape")
        self.assertEqual(normalize_qt_key(65, ""), "a")
        self.assertEqual(normalize_qt_key(0, "W"), "w")

    def test_controller_backend_updates_axes_buttons_and_connected_list(self):
        input_state = InputState(
            controller_backend=FakeControllerBackend(
                [
                    ControllerSnapshot(
                        axes={"left_x": 0.75},
                        buttons_down={"south"},
                        connected_controllers=["Fake Pad"],
                    ),
                    ControllerSnapshot(axes={"left_x": 0.0}, buttons_down=set(), connected_controllers=["Fake Pad"]),
                ]
            )
        )

        input_state.begin_frame()

        self.assertEqual(input_state.connected_controllers, ["Fake Pad"])
        self.assertAlmostEqual(input_state.get_axis("left_x"), 0.75)
        self.assertTrue(input_state.is_button_down("south"))
        self.assertTrue(input_state.was_button_pressed("south"))

        input_state.end_frame()
        input_state.begin_frame()

        self.assertFalse(input_state.is_button_down("south"))
        self.assertTrue(input_state.was_button_released("south"))

    def test_runtime_session_advances_input_frame_state(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "input_probe.py").write_text(
                "from p64.engine.scripting import UserScript\n"
                "class InputProbe(UserScript):\n"
                "    def on_update(self, dt):\n"
                "        if self.input.was_key_pressed('space'):\n"
                "            self.transform.position.x += 1\n"
                "        if self.input.is_key_down('space'):\n"
                "            self.transform.position.y += 1\n"
                "        if self.input.was_key_released('space'):\n"
                "            self.transform.position.z += 1\n",
                encoding="utf-8",
            )
            scene = project.load_startup_scene()
            actor = Entity("Actor")
            actor.add_component(ScriptComponent(scripts=[ScriptEntry(script="input_probe.py", class_name="InputProbe")]))
            scene.add_entity(actor)
            session = RuntimeSession(project, scene)
            session.input.controller_backend = FakeControllerBackend([])

            session.input.press_key("space")
            self.assertEqual(session.tick(1 / 60), [])
            runtime_actor = session.scene.find(actor.id)
            self.assertEqual(runtime_actor.transform.position.x, 1)
            self.assertEqual(runtime_actor.transform.position.y, 1)
            self.assertFalse(session.input.was_key_pressed("space"))

            self.assertEqual(session.tick(1 / 60), [])
            self.assertEqual(runtime_actor.transform.position.y, 2)

            session.input.release_key("space")
            self.assertEqual(session.tick(1 / 60), [])
            self.assertEqual(runtime_actor.transform.position.z, 1)


if __name__ == "__main__":
    unittest.main()
