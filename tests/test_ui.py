import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from p64.engine.components import Canvas, RectTransform, ScriptComponent, ScriptEntry, UIButton, UIScrollView, UISlider, UIToggle, component_from_dict
from p64.engine.entity import Entity
from p64.engine.input import ControllerSnapshot, InputState
from p64.engine.math import Vec3
from p64.engine.scene import Scene
from p64.engine.ui import UIEventSystem, build_ui_layout
from p64.engine.project import Project
from p64.engine.runtime_session import RuntimeSession
from p64.engine.validation import entity_reference_errors


class EmptyController:
    def poll(self):
        return ControllerSnapshot()


def button(name: str, x: float, y: float) -> Entity:
    return Entity(
        name,
        rect_transform=RectTransform(anchor="top-left", offset=Vec3(x, y, 0), size=Vec3(100, 40, 0), pivot=Vec3()),
        components=[UIButton()],
    )


class UIInteractionTests(unittest.TestCase):
    def setUp(self):
        self.canvas = Entity("Canvas", components=[Canvas(reference_resolution=Vec3(320, 240, 0), resolution_mode="fixed")])
        self.scene = Scene("UI", [self.canvas])
        self.input = InputState(controller_backend=EmptyController())
        self.input.set_viewport_size(640, 480)
        self.events = []
        self.ui = UIEventSystem(lambda entity, method, args: self.events.append((entity.name, method, args)) or [])

    def test_layout_uses_rect_transforms_and_fixed_canvas(self):
        child = self.canvas.add_child(button("Play", 20, 30))
        entry = next(item for item in build_ui_layout(self.scene, 640, 480) if item.entity is child)
        self.assertEqual(entry.rect, (20.0, 30.0, 100.0, 40.0))

    def test_controls_round_trip_through_component_data(self):
        controls = [
            UIButton(navigation_down="next"),
            UIToggle(is_on=True, checkmark_entity="mark"),
            UISlider(minimum=-1, maximum=1, value=0.5, step=0.25, handle_entity="handle"),
            UIScrollView(content_entity="content", horizontal=True, scroll_position=Vec3(4, 8, 0)),
        ]
        loaded = [component_from_dict(control.to_dict()) for control in controls]
        self.assertEqual([type(control) for control in loaded], [type(control) for control in controls])
        self.assertEqual(loaded[0].navigation_down, "next")
        self.assertTrue(loaded[1].is_on)
        self.assertEqual(loaded[2].value, 0.5)
        self.assertEqual(loaded[3].scroll_position.to_list(), [4.0, 8.0, 0.0])

    def test_validation_reports_invalid_control_references_and_multiple_controls(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            entity = self.canvas.add_child(button("Broken", 0, 0))
            entity.components.extend([UIToggle(checkmark_entity="missing"), UISlider(maximum=0)])
            errors = entity_reference_errors(project, entity)
            self.assertIn("An entity can only have one interactive UI control", errors)
            self.assertIn("Missing UI entity reference: missing", errors)
            self.assertIn("UISlider maximum must be greater than minimum", errors)

    def test_mouse_click_emits_pointer_focus_and_click(self):
        self.canvas.add_child(button("Play", 20, 30))
        self.input.move_mouse(60, 80)  # Fixed canvas coordinates: 30, 40.
        self.input.press_mouse("left")
        self.ui.process(self.scene, self.input, 1 / 60)
        self.input.end_frame()
        self.input.release_mouse("left")
        self.ui.process(self.scene, self.input, 1 / 60)
        methods = [event[1] for event in self.events]
        self.assertIn("on_ui_pointer_enter", methods)
        self.assertIn("on_ui_click", methods)

    def test_keyboard_navigation_uses_geometry(self):
        left = self.canvas.add_child(button("Left", 10, 10))
        right = self.canvas.add_child(button("Right", 150, 10))
        self.ui.process(self.scene, self.input, 1 / 60)
        self.assertEqual(self.ui.focused_entity_id, left.id)
        self.input.press_key("right")
        self.ui.process(self.scene, self.input, 1 / 60)
        self.assertEqual(self.ui.focused_entity_id, right.id)

    def test_toggle_and_slider_values_emit_callbacks(self):
        toggle_entity = self.canvas.add_child(button("Toggle", 10, 10))
        toggle_entity.components[-1] = UIToggle()
        slider_entity = self.canvas.add_child(button("Slider", 10, 70))
        slider_entity.components[-1] = UISlider(minimum=0, maximum=10, value=0, step=1)
        self.input.move_mouse(40, 40)
        self.input.press_mouse("left")
        self.ui.process(self.scene, self.input, 1 / 60)
        self.input.end_frame()
        self.input.release_mouse("left")
        self.ui.process(self.scene, self.input, 1 / 60)
        self.assertTrue(toggle_entity.components[-1].is_on)
        self.input.end_frame()
        self.input.move_mouse(200, 160)
        self.input.press_mouse("left")
        self.ui.process(self.scene, self.input, 1 / 60)
        self.assertGreater(slider_entity.components[-1].value, 0)
        self.assertTrue(any(event[1] == "on_ui_value_changed" for event in self.events))

    def test_scroll_view_clips_children_and_clamps_content(self):
        scroll_entity = Entity(
            "Scroll",
            rect_transform=RectTransform(anchor="top-left", offset=Vec3(10, 10, 0), size=Vec3(100, 80, 0), pivot=Vec3()),
            components=[UIScrollView(vertical=True, wheel_speed=40)],
        )
        content = scroll_entity.add_child(Entity(
            "Content",
            rect_transform=RectTransform(anchor="top-left", size=Vec3(100, 200, 0), pivot=Vec3()),
        ))
        scroll_entity.components[0].content_entity = content.id
        child = content.add_child(button("Child", 0, 150))
        self.canvas.add_child(scroll_entity)
        entries = build_ui_layout(self.scene, 640, 480)
        child_entry = next(entry for entry in entries if entry.entity is child)
        self.assertEqual(child_entry.clip_rect, (10.0, 10.0, 100.0, 80.0))
        self.input.move_mouse(40, 40)
        self.input.add_wheel_delta(0, -120)
        self.ui.process(self.scene, self.input, 1 / 60)
        self.assertEqual(scroll_entity.components[0].scroll_position.y, 40.0)

    def test_runtime_dispatches_button_callback_before_update(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "menu.py").write_text(
                "from p64.engine.scripting import GameScript\n"
                "class Menu(GameScript):\n"
                "    def on_ui_click(self): self.transform.position.x += 10\n"
                "    def on_update(self, dt): self.transform.position.y = self.transform.position.x\n",
                encoding="utf-8",
            )
            target = self.canvas.add_child(button("Play", 20, 30))
            target.add_component(ScriptComponent(scripts=[ScriptEntry(script="menu.py", class_name="Menu")]))
            session = RuntimeSession(project, self.scene)
            session.input.controller_backend = EmptyController()
            session.input.set_viewport_size(640, 480)
            session.input.move_mouse(60, 80)
            session.input.press_mouse("left")
            session.tick(1 / 60)
            session.input.release_mouse("left")
            self.assertEqual(session.tick(1 / 60), [])
            self.assertEqual(target.transform.position.x, 10)
            self.assertEqual(target.transform.position.y, 10)


if __name__ == "__main__":
    unittest.main()
