import unittest

from p64.editor.gizmos import GizmoHandle, ScreenPoint, apply_gizmo_drag, hit_test_gizmo, scale_handle_radius, transform_snapshot
from p64.editor.undo import UndoManager
from p64.engine.entity import Entity
from p64.engine.math import Vec3
from p64.engine.scene import Scene
from p64.engine.transforms import world_position, world_rotation


class EditorUndoTests(unittest.TestCase):
    def test_undo_manager_records_undo_redo_and_discards_redo(self):
        scene = Scene("Scene", [Entity("A")])
        manager = UndoManager()
        manager.reset(scene, scene.entities[0].id)

        scene.entities[0].name = "B"
        manager.record("Rename", scene, scene.entities[0].id)

        undone = manager.undo()
        self.assertIsNotNone(undone)
        self.assertEqual(undone.scene_data["entities"][0]["name"], "A")

        redone = manager.redo()
        self.assertIsNotNone(redone)
        self.assertEqual(redone.scene_data["entities"][0]["name"], "B")

        manager.undo()
        scene.entities[0].name = "C"
        manager.record("Rename Again", scene, scene.entities[0].id)

        self.assertFalse(manager.can_redo)

    def test_undo_manager_dirty_tracks_saved_index(self):
        scene = Scene("Scene", [Entity("A")])
        manager = UndoManager()
        manager.reset(scene)

        self.assertFalse(manager.is_dirty)
        scene.entities[0].name = "B"
        manager.record("Rename", scene)
        self.assertTrue(manager.is_dirty)
        manager.mark_saved()
        self.assertFalse(manager.is_dirty)
        manager.undo()
        self.assertTrue(manager.is_dirty)

    def test_undo_state_restores_parent_links_through_scene_from_dict(self):
        parent = Entity("Parent")
        child = parent.add_child(Entity("Child"))
        scene = Scene("Scene", [parent])
        manager = UndoManager()
        manager.reset(scene, child.id)
        child.name = "Renamed"
        manager.record("Rename Child", scene, child.id)

        state = manager.undo()
        self.assertIsNotNone(state)
        restored = Scene.from_dict(state.scene_data)
        restored_child = restored.find(child.id)

        self.assertIsNotNone(restored_child)
        self.assertIs(restored_child.parent, restored.entities[0])

    def test_begin_commit_creates_one_history_entry_for_drag(self):
        scene = Scene("Scene", [Entity("A")])
        manager = UndoManager()
        manager.reset(scene)

        manager.begin("Move X", scene)
        scene.entities[0].transform.position.x = 1.0
        scene.entities[0].transform.position.x = 2.0
        manager.commit(scene)

        self.assertEqual(manager.history_length, 2)
        self.assertTrue(manager.can_undo)


class EditorGizmoTests(unittest.TestCase):
    def test_hit_test_prefers_center_inside_center_radius(self):
        handles = [
            GizmoHandle("center", ScreenPoint(100, 100), ScreenPoint(100, 100)),
            GizmoHandle("x", ScreenPoint(100, 100), ScreenPoint(180, 100)),
        ]

        self.assertEqual(hit_test_gizmo(handles, 104, 103), "center")

    def test_hit_test_selects_nearest_axis(self):
        handles = [
            GizmoHandle("center", ScreenPoint(100, 100), ScreenPoint(100, 100)),
            GizmoHandle("x", ScreenPoint(100, 100), ScreenPoint(180, 100)),
            GizmoHandle("y", ScreenPoint(100, 100), ScreenPoint(100, 180)),
        ]

        self.assertEqual(hit_test_gizmo(handles, 150, 104), "x")
        self.assertEqual(hit_test_gizmo(handles, 104, 150), "y")

    def test_hit_test_selects_rotation_ring(self):
        ring = tuple(
            [
                ScreenPoint(140, 100),
                ScreenPoint(100, 140),
                ScreenPoint(60, 100),
                ScreenPoint(100, 60),
            ]
        )
        handles = [
            GizmoHandle("center", ScreenPoint(100, 100), ScreenPoint(100, 100)),
            GizmoHandle("x", ScreenPoint(100, 100), ring[0], kind="ring", points=ring),
        ]

        self.assertEqual(hit_test_gizmo(handles, 139, 103), "x")

    def test_move_axis_drag_changes_only_one_position_axis(self):
        entity = Entity("Thing")
        start = transform_snapshot(entity)

        apply_gizmo_drag(entity, "move", "x", start, 20, 0, axis_screen_direction=(1, 0), world_per_pixel=0.1)

        self.assertEqual(entity.transform.position.to_list(), [2.0, 0.0, 0.0])

    def test_center_move_uses_camera_right_and_up(self):
        entity = Entity("Thing")
        start = transform_snapshot(entity)

        apply_gizmo_drag(
            entity,
            "move",
            "center",
            start,
            10,
            -5,
            camera_right=Vec3(1, 0, 0),
            camera_up=Vec3(0, 1, 0),
            world_per_pixel=0.1,
        )

        self.assertEqual(entity.transform.position.to_list(), [1.0, 0.5, 0.0])

    def test_child_move_drag_writes_local_position_for_world_motion(self):
        parent = Entity("Parent")
        parent.transform.position = Vec3(5.0, 0.0, 0.0)
        child = parent.add_child(Entity("Child"))
        child.transform.position = Vec3(1.0, 0.0, 0.0)
        start = transform_snapshot(child)

        apply_gizmo_drag(child, "move", "x", start, 20, 0, axis_screen_direction=(1, 0), world_per_pixel=0.1)

        self.assertEqual(world_position(child), Vec3(8.0, 0.0, 0.0))
        self.assertEqual(child.transform.position, Vec3(3.0, 0.0, 0.0))

    def test_scale_axis_clamps_and_center_scales_uniformly(self):
        entity = Entity("Thing")
        start = transform_snapshot(entity)

        apply_gizmo_drag(entity, "scale", "x", start, -500, 0, axis_screen_direction=(1, 0))
        self.assertEqual(entity.transform.scale.x, 0.001)

        apply_gizmo_drag(entity, "scale", "center", start, 20, -20)
        self.assertEqual(entity.transform.scale.to_list(), [1.4, 1.4, 1.4])

    def test_scale_handle_radius_tracks_axis_scale_ratio(self):
        entity = Entity("Thing")
        start = transform_snapshot(entity)
        entity.transform.scale.x = 2.25

        self.assertGreater(scale_handle_radius(start, entity.transform.scale, "x"), scale_handle_radius(start, Vec3(1.0, 1.0, 1.0), "x"))

    def test_rotate_axis_drag_changes_euler_axis(self):
        entity = Entity("Thing")
        start = transform_snapshot(entity)

        apply_gizmo_drag(entity, "rotate", "y", start, 30, 0, axis_screen_direction=(1, 0))

        self.assertEqual(entity.transform.rotation.to_list(), [0.0, 15.0, 0.0])

    def test_center_rotate_uses_dominant_camera_axis(self):
        entity = Entity("Thing")
        start = transform_snapshot(entity)

        apply_gizmo_drag(entity, "rotate", "center", start, 20, 0, camera_forward=Vec3(0.1, 0.8, 0.2))

        self.assertEqual(entity.transform.rotation.to_list(), [0.0, 10.0, 0.0])

    def test_child_rotate_drag_writes_local_rotation_for_world_rotation(self):
        parent = Entity("Parent")
        parent.transform.rotation = Vec3(0.0, 30.0, 0.0)
        child = parent.add_child(Entity("Child"))
        child.transform.rotation = Vec3(0.0, 10.0, 0.0)
        start = transform_snapshot(child)

        apply_gizmo_drag(child, "rotate", "y", start, 20, 0, axis_screen_direction=(1, 0))

        self.assertEqual(world_rotation(child), Vec3(0.0, 50.0, 0.0))
        self.assertEqual(child.transform.rotation, Vec3(0.0, 20.0, 0.0))


if __name__ == "__main__":
    unittest.main()
