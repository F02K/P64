from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from p64.engine.components import Fog, MeshRenderer, ScriptComponent, ScriptEntry
from p64.engine.entity import Entity
from p64.engine.migration import migrate_project_files
from p64.engine.math import Vec3
from p64.engine.project import Project
from p64.engine.scene import Scene


class ProjectSceneTests(unittest.TestCase):
    def test_project_create_and_scene_roundtrip(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            loaded = Project.load(project.root)
            scene = loaded.load_startup_scene()
            scene.add_entity(Entity("Thing"))
            loaded.save_startup_scene(scene)

            reloaded = loaded.load_startup_scene()
            self.assertEqual(loaded.name, "Game")
            self.assertTrue((project.root / "project.p64").exists())
            self.assertTrue((project.root / "scenes" / "main.scenep64").exists())
            self.assertTrue((project.root / "assets").exists())
            self.assertIsNotNone(reloaded.active_camera())
            self.assertEqual(reloaded.entities[-1].name, "Thing")

    def test_legacy_project_files_load_and_migrate(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            scene = project.load_startup_scene()
            old_project = project.root / "project.p64.json"
            old_scene = project.root / "scenes" / "main.p64scene.json"
            (project.root / "project.p64").rename(old_project)
            (project.root / "scenes" / "main.scenep64").rename(old_scene)
            old_project.write_text(
                old_project.read_text(encoding="utf-8").replace("scenes/main.scenep64", "scenes/main.p64scene.json"),
                encoding="utf-8",
            )

            loaded = Project.load(project.root)
            self.assertIsNotNone(loaded.load_startup_scene().active_camera())
            changes = migrate_project_files(project.root)

            self.assertTrue((project.root / "project.p64").exists())
            self.assertTrue((project.root / "scenes" / "main.scenep64").exists())
            self.assertFalse(old_project.exists())
            self.assertFalse(old_scene.exists())
            self.assertTrue(any("project.p64.json" in change for change in changes))

    def test_transform_world_matrix_includes_parent_translation(self):
        parent = Entity("Parent")
        child = Entity("Child")
        parent.transform.position = Vec3(2, 0, 0)
        child.transform.position = Vec3(0, 3, 0)
        parent.add_child(child)

        world = child.transform.world_matrix(child)
        self.assertEqual(world[3], 2)
        self.assertEqual(world[7], 3)

    def test_scene_serializes_components(self):
        root = Entity("Root")
        root.add_component(MeshRenderer(mesh="mesh", submesh="Door", shader="assets/shaders/n64.shader"))
        root.add_component(ScriptComponent(scripts=[ScriptEntry(script="spin.py", class_name="Spin")]))
        scene = Scene("Test", [root])

        loaded = Scene.from_dict(scene.to_dict())
        self.assertEqual(loaded.entities[0].components[0].mesh, "mesh")
        self.assertEqual(loaded.entities[0].components[0].shader, "assets/shaders/n64.shader")
        self.assertEqual(loaded.entities[0].components[1].scripts[0].script, "spin.py")

    def test_old_script_component_json_migrates_to_list(self):
        component = {
            "type": "ScriptComponent",
            "enabled": True,
            "script": "spin.py",
            "class_name": "Spin",
            "properties": {"speed": 2},
        }
        scene = Scene.from_dict({"name": "Test", "entities": [{"name": "Door", "components": [component]}]})
        script_component = scene.entities[0].components[0]
        self.assertEqual(script_component.scripts[0].script, "spin.py")
        self.assertEqual(script_component.scripts[0].class_name, "Spin")

    def test_fog_volume_size_roundtrip(self):
        root = Entity("Fog")
        root.add_component(Fog(size=Vec3(10, 20, 30), density=0.25))
        scene = Scene.from_dict(Scene("Test", [root]).to_dict())
        fog = scene.fog()
        self.assertEqual(fog.size.to_list(), [10.0, 20.0, 30.0])
        self.assertEqual(fog.density, 0.25)


if __name__ == "__main__":
    unittest.main()
