from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from p64.build.pipeline import validate_project
from p64.editor.ops import (
    DirtyTracker,
    add_component,
    create_script_template,
    create_shader_template,
    delete_entity,
    duplicate_entity,
    insert_obj_scene_entity,
    move_component,
    move_script_entry,
)
from p64.engine.assets import AssetMetadata
from p64.engine.components import Camera, Collider, EntityPhysics, MeshRenderer, ScriptComponent, ScriptEntry
from p64.engine.entity import GAME_OBJECT, Entity
from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.engine.validation import entity_reference_errors


OBJ_TEXT = """
mtllib model.mtl
o Body
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
usemtl Mat
f 1/1/1 2/2/1 3/3/1
"""

MTL_TEXT = """
newmtl Mat
map_Kd missing.png
"""


class EditorOpsTests(unittest.TestCase):
    def test_duplicate_entity_regenerates_ids(self):
        root = Entity("Root")
        child = Entity("Child")
        root.add_child(child)
        duplicate = duplicate_entity(root)

        self.assertNotEqual(root.id, duplicate.id)
        self.assertNotEqual(root.children[0].id, duplicate.children[0].id)
        self.assertEqual(duplicate.name, "Root Copy")

    def test_delete_entity_removes_nested_child(self):
        parent = Entity("Parent")
        child = Entity("Child")
        parent.add_child(child)
        scene = Scene("Test", [parent])

        self.assertTrue(delete_entity(scene, child.id))
        self.assertEqual(parent.children, [])

    def test_dirty_tracker_marks_and_clears(self):
        tracker = DirtyTracker()
        tracker.mark_dirty()
        self.assertTrue(tracker.dirty)
        tracker.mark_saved()
        self.assertFalse(tracker.dirty)

    def test_create_shader_and_script_templates(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shader = create_shader_template(root / "assets")
            script = create_script_template(root / "scripts", "Mover")
            self.assertIn("Vertex", shader.read_text(encoding="utf-8"))
            self.assertIn("class Mover", script.read_text(encoding="utf-8"))

    def test_component_and_script_entry_ordering(self):
        entity = Entity("Root")
        mesh = entity.add_component(MeshRenderer())
        camera = entity.add_component(Camera())
        self.assertTrue(move_component(entity, camera, -1))
        self.assertIs(entity.components[0], camera)
        self.assertIs(entity.components[1], mesh)

        scripts = ScriptComponent(scripts=[
            ScriptEntry(script="a.py", class_name="A"),
            ScriptEntry(script="b.py", class_name="B"),
        ])
        self.assertTrue(move_script_entry(scripts, 1, -1))
        self.assertEqual([entry.script for entry in scripts.scripts], ["b.py", "a.py"])

    def test_add_collider_prefills_mesh_bounds_when_mesh_renderer_exists(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.obj"
            source.write_text(OBJ_TEXT, encoding="utf-8")
            (root / "model.mtl").write_text(MTL_TEXT, encoding="utf-8")
            project = Project.create(root / "Game")
            metadata = AssetMetadata(
                id="mesh_body",
                kind="obj_mesh",
                source="assets/model.obj",
                groups=["Body"],
                materials=["Mat"],
            )
            project.assets_dir.mkdir(parents=True, exist_ok=True)
            (project.assets_dir / "model.obj").write_text(OBJ_TEXT, encoding="utf-8")
            entity = Entity("Body")
            entity.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            metadata.save(project.assets_dir / "model.obj.mdp64")

            collider = add_component(entity, "Collider", project)

            self.assertIsInstance(collider, Collider)
            self.assertEqual(collider.shape, "box")
            self.assertEqual(collider.center.to_list(), [0.5, 0.5, 0.0])
            self.assertEqual(collider.size.to_list(), [1.0, 1.0, 0.001])
            self.assertEqual(collider.radius, 0.5)
            self.assertFalse(collider.fit_to_mesh)

    def test_add_collider_without_mesh_keeps_fallback_defaults(self):
        entity = Entity("Empty")

        collider = add_component(entity, "Collider")

        self.assertEqual(collider.center.to_list(), [0.0, 0.0, 0.0])
        self.assertEqual(collider.size.to_list(), [1.0, 1.0, 1.0])
        self.assertEqual(collider.radius, 0.5)

    def test_add_entity_physics_switches_game_object_to_entity(self):
        entity = Entity("Crate", object_type=GAME_OBJECT)

        physics = add_component(entity, "EntityPhysics")

        self.assertIsInstance(physics, EntityPhysics)
        self.assertTrue(entity.is_entity)

    def test_validation_reports_entity_physics_on_game_object(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            entity = Entity("Static Crate", object_type=GAME_OBJECT)
            entity.add_component(EntityPhysics())

            errors = entity_reference_errors(project, entity)

            self.assertIn("EntityPhysics requires an Entity", errors)

    def test_insert_obj_scene_entity_creates_submesh_children(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.obj"
            source.write_text(OBJ_TEXT, encoding="utf-8")
            (root / "model.mtl").write_text(MTL_TEXT, encoding="utf-8")
            project = Project.create(root / "Game")
            scene = project.load_startup_scene()

            entity = insert_obj_scene_entity(project, scene, source)
            self.assertEqual(entity.children[0].name, "Body")
            self.assertTrue(entity.is_game_object)
            self.assertTrue(entity.children[0].is_game_object)
            renderer = entity.children[0].components[0]
            self.assertIsInstance(renderer, MeshRenderer)

    def test_validation_reports_missing_texture(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = AssetMetadata(
                id="mesh_missing_texture",
                kind="obj_mesh",
                source="assets/model.obj",
                groups=["Body"],
                materials=["Mat"],
                settings={"material_defs": {"Mat": {"diffuse_texture": "missing.png"}}},
            )
            (project.assets_dir / "model.obj").write_text("o Body\n", encoding="utf-8")
            metadata.save(project.assets_dir / "model.obj.mdp64")
            scene = project.load_startup_scene()
            entity = Entity("Body")
            entity.add_component(MeshRenderer(mesh=metadata.id, submesh="Body", material="Mat"))
            scene.add_entity(entity)
            project.save_startup_scene(scene)

            report = validate_project(project.root)
            self.assertTrue(any("missing texture" in error for error in report.errors))

    def test_reference_validation_reports_missing_mesh(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            entity = Entity("Broken")
            entity.add_component(MeshRenderer(mesh="missing_mesh"))

            errors = entity_reference_errors(project, entity)

            self.assertTrue(any("Missing mesh asset" in error for error in errors))

    def test_empty_script_entry_reports_validation_error_without_crashing(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            entity = Entity("Empty Script")
            entity.add_component(ScriptComponent(scripts=[ScriptEntry(script="", class_name="")]))

            errors = entity_reference_errors(project, entity)

            self.assertIn("Script entry has no script file", errors)


if __name__ == "__main__":
    unittest.main()
