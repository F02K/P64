from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from p64.build.pipeline import validate_project
from p64.editor.ops import (
    DirtyTracker,
    create_script_template,
    create_shader_template,
    delete_entity,
    duplicate_entity,
    insert_obj_scene_entity,
)
from p64.engine.assets import AssetMetadata
from p64.engine.components import MeshRenderer
from p64.engine.entity import Entity
from p64.engine.project import Project
from p64.engine.scene import Scene


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


if __name__ == "__main__":
    unittest.main()
