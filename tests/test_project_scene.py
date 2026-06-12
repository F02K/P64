from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import unittest

from p64.engine.builtin import LEGACY_STANDARD_SHADER_RELATIVE, STANDARD_SHADER_RELATIVE, STANDARD_UNLIT_SHADER_RELATIVE
from p64.engine.components import EntityPhysics, Fog, Light, MeshRenderer, ScriptComponent, ScriptEntry, SpawnPoint
from p64.engine.entity import GAME_OBJECT, Entity, set_object_type_recursive
from p64.engine.migration import migrate_project_files
from p64.engine.math import Vec3
from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.engine.scene_manager import SceneManager


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
            self.assertTrue((project.root / "assets" / "scenes" / "main.scenep64").exists())
            self.assertTrue((project.root / "assets" / "scripts").exists())
            self.assertTrue((project.root / STANDARD_SHADER_RELATIVE).exists())
            self.assertTrue((project.root / STANDARD_UNLIT_SHADER_RELATIVE).exists())
            self.assertTrue((project.root / "libraries" / "P64Build" / "builder.py").exists())
            self.assertTrue((project.root / "libraries" / "P64Build" / "requirements-build.txt").exists())
            requirements = (project.root / "libraries" / "P64Build" / "requirements-build.txt").read_text(encoding="utf-8")
            self.assertIn("pygame>=2.5", requirements)
            self.assertTrue((project.root / "libraries" / "P64Build" / "p64_source" / "p64" / "engine" / "runtime.py").exists())
            self.assertTrue((project.root / "assets").exists())
            self.assertIsNotNone(reloaded.active_camera())
            self.assertEqual(reloaded.entities[-1].name, "Thing")

    def test_project_settings_roundtrip_and_defaults(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            project.build_settings["executable_name"] = "CustomGame"
            project.build_settings["icon_path"] = "assets/icon.png"
            project.build_settings["python_executable"] = "C:/Python/python.exe"
            project.build_settings["build_pipeline_path"] = "libraries/CustomBuild"
            project.build_settings["auto_install_build_dependencies"] = False
            project.editor_settings["scene_grid"]["spacing"] = 2.5
            project.render_settings["color_levels"] = 1
            project.save()

            loaded = Project.load(project.root)

            self.assertEqual(loaded.build_settings["executable_name"], "CustomGame")
            self.assertEqual(loaded.build_settings["icon_path"], "assets/icon.png")
            self.assertEqual(loaded.build_settings["python_executable"], "C:/Python/python.exe")
            self.assertEqual(loaded.build_settings["build_pipeline_path"], "libraries/CustomBuild")
            self.assertFalse(loaded.build_settings["auto_install_build_dependencies"])
            self.assertEqual(loaded.editor_settings["scene_grid"]["spacing"], 2.5)
            self.assertEqual(loaded.render_settings["color_levels"], 2)

    def test_existing_project_save_adds_build_pipeline(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            shutil.rmtree(project.root / "libraries")

            Project.load(project.root).save()

            self.assertTrue((project.root / "libraries" / "P64Build" / "builder.py").exists())

    def test_legacy_project_files_load_and_migrate(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            scene = project.load_startup_scene()
            old_project = project.root / "project.p64.json"
            old_scene = project.root / "scenes" / "main.p64scene.json"
            old_scene.parent.mkdir(parents=True, exist_ok=True)
            (project.root / "project.p64").rename(old_project)
            (project.root / "assets" / "scenes" / "main.scenep64").rename(old_scene)
            old_project.write_text(
                old_project.read_text(encoding="utf-8").replace("assets/scenes/main.scenep64", "scenes/main.p64scene.json"),
                encoding="utf-8",
            )

            loaded = Project.load(project.root)
            self.assertIsNotNone(loaded.load_startup_scene().active_camera())
            changes = migrate_project_files(project.root)

            self.assertTrue((project.root / "project.p64").exists())
            self.assertTrue((project.root / "assets" / "scenes" / "main.scenep64").exists())
            self.assertFalse(old_project.exists())
            self.assertFalse(old_scene.exists())
            self.assertTrue(any("project.p64.json" in change for change in changes))

    def test_scene_manager_switches_scene_and_preserves_persistent_entities(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            second = Scene("Second", [Entity("SecondCamera")])
            second_path = project.scenes_dir / "second.scenep64"
            second.save(second_path)
            manager = SceneManager(project)
            persistent = Entity("Player", persistent=True)
            persistent.add_child(Entity("Child"))
            manager.current_scene.add_entity(persistent)

            manager.load_scene("assets/scenes/second.scenep64")
            self.assertTrue(manager.apply_queued_scene())

            names = [entity.name for entity in manager.current_scene.entities]
            self.assertIn("Player", names)
            self.assertIn("SecondCamera", names)
            self.assertEqual(manager.current_scene.find(persistent.children[0].id).name, "Child")

    def test_scene_manager_applies_explicit_spawn_point_to_persistent_entity(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            scene = Scene("Second")
            wrong = Entity("WrongSpawn")
            wrong.transform.position = Vec3(1, 0, 0)
            wrong.add_component(SpawnPoint(spawn_id="wrong", is_default=True))
            gate = Entity("GateSpawn")
            gate.transform.position = Vec3(5, 2, 3)
            gate.transform.rotation = Vec3(0, 90, 0)
            gate.add_component(SpawnPoint(spawn_id="gate"))
            scene.add_entity(wrong)
            scene.add_entity(gate)
            scene.save(project.scenes_dir / "second.scenep64")
            manager = SceneManager(project)
            player = Entity("Player", persistent=True)
            manager.current_scene.add_entity(player)

            manager.load_scene_by_name("second", spawn_id="gate")
            self.assertTrue(manager.apply_queued_scene())

            self.assertEqual(player.transform.position.to_list(), [5, 2, 3])
            self.assertEqual(player.transform.rotation.to_list(), [0, 90, 0])

    def test_scene_manager_uses_previous_scene_spawn_when_no_explicit_id(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            scene = Scene("Second")
            default = Entity("DefaultSpawn")
            default.transform.position = Vec3(1, 0, 0)
            default.add_component(SpawnPoint(spawn_id="default", is_default=True))
            previous = Entity("PreviousSpawn")
            previous.transform.position = Vec3(2, 0, 0)
            previous.add_component(SpawnPoint(spawn_id="from_main", from_scene="main"))
            scene.add_entity(default)
            scene.add_entity(previous)
            scene.save(project.scenes_dir / "second.scenep64")
            manager = SceneManager(project)
            player = Entity("Player", persistent=True)
            manager.current_scene.add_entity(player)

            manager.load_scene_by_name("second")
            self.assertTrue(manager.apply_queued_scene())

            self.assertEqual(player.transform.position.to_list(), [2, 0, 0])

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
        root.add_component(MeshRenderer(mesh="mesh", submesh="Door", shader="assets/shaders/standard.shader"))
        root.add_component(ScriptComponent(scripts=[ScriptEntry(script="spin.py", class_name="Spin")]))
        root.add_component(
            EntityPhysics(
                mass=2.0,
                use_gravity=False,
                drag=0.25,
                angular_drag=0.5,
                is_kinematic=True,
                velocity=Vec3(1, 2, 3),
                angular_velocity=Vec3(4, 5, 6),
                freeze_position=Vec3(1, 0, 0),
                freeze_rotation=Vec3(0, 1, 0),
            )
        )
        scene = Scene("Test", [root])

        loaded = Scene.from_dict(scene.to_dict())
        self.assertEqual(loaded.entities[0].components[0].mesh, "mesh")
        self.assertEqual(loaded.entities[0].components[0].shader, "assets/shaders/standard.shader")
        self.assertEqual(loaded.entities[0].components[1].scripts[0].script, "spin.py")
        physics = loaded.entities[0].components[2]
        self.assertIsInstance(physics, EntityPhysics)
        self.assertEqual(physics.mass, 2.0)
        self.assertFalse(physics.use_gravity)
        self.assertEqual(physics.velocity.to_list(), [1.0, 2.0, 3.0])
        self.assertEqual(physics.freeze_rotation.to_list(), [0.0, 1.0, 0.0])

    def test_entity_serializes_persistent_flag(self):
        entity = Entity("Player", persistent=True)

        data = entity.to_dict()
        loaded = Entity.from_dict(data)

        self.assertTrue(data["persistent"])
        self.assertNotIn("dont" + "_destroy_on_load", data)
        self.assertTrue(loaded.persistent)

    def test_scene_object_type_serializes_and_searches_by_kind(self):
        scene = Scene("Test")
        marker = Entity("QuestMarker", object_type=GAME_OBJECT)
        enemy = Entity("Enemy")
        scene.add_entity(marker)
        scene.add_entity(enemy)

        loaded = Scene.from_dict(scene.to_dict())

        self.assertEqual(loaded.find_scene_object("QuestMarker").object_type, GAME_OBJECT)
        self.assertIsNotNone(loaded.find_game_object("QuestMarker"))
        self.assertIsNone(loaded.find_entity("QuestMarker"))
        self.assertIsNotNone(loaded.find_entity("Enemy"))
        self.assertIsNone(loaded.find_game_object("Enemy"))

    def test_scene_object_type_cascades_to_children(self):
        parent = Entity("Parent", object_type=GAME_OBJECT)
        child = Entity("Child")
        grandchild = child.add_child(Entity("Grandchild", object_type=GAME_OBJECT))

        parent.add_child(child)

        self.assertTrue(child.is_game_object)
        self.assertTrue(grandchild.is_game_object)
        set_object_type_recursive(parent, "entity")
        self.assertTrue(parent.is_entity)
        self.assertTrue(child.is_entity)
        self.assertTrue(grandchild.is_entity)

    def test_mixed_scene_object_types_normalize_on_load_and_validate_when_manual(self):
        loaded = Entity.from_dict({
            "name": "Root",
            "object_type": GAME_OBJECT,
            "children": [{"name": "Child", "object_type": "entity"}],
        })
        self.assertTrue(loaded.children[0].is_game_object)

        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            parent = Entity("Parent", object_type=GAME_OBJECT)
            child = parent.add_child(Entity("Child"))
            child.object_type = "entity"

            from p64.engine.validation import entity_reference_errors

            self.assertIn("Child SceneObject type must match parent", entity_reference_errors(project, parent))

    def test_legacy_scene_object_defaults_to_entity(self):
        loaded = Entity.from_dict({"name": "Old Object"})

        self.assertTrue(loaded.is_entity)
        self.assertEqual(loaded.object_type_label, "Entity")

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

    def test_light_serializes_extended_fields(self):
        light = Light(kind="spot", range=24.0, spot_angle=35.0, falloff=3.0)
        scene = Scene.from_dict(Scene("Test", [Entity("Light", components=[light])]).to_dict())
        loaded = scene.entities[0].components[0]

        self.assertIsInstance(loaded, Light)
        self.assertEqual(loaded.kind, "spot")
        self.assertEqual(loaded.range, 24.0)
        self.assertEqual(loaded.spot_angle, 35.0)
        self.assertEqual(loaded.falloff, 3.0)

    def test_legacy_builtin_shader_reference_normalizes_on_load(self):
        scene = Scene.from_dict({
            "name": "Test",
            "entities": [
                {
                    "name": "Mesh",
                    "components": [
                        {
                            "type": "MeshRenderer",
                            "mesh": "mesh",
                            "shader": LEGACY_STANDARD_SHADER_RELATIVE,
                        }
                    ],
                }
            ],
        })

        self.assertEqual(scene.entities[0].components[0].shader, STANDARD_SHADER_RELATIVE)


if __name__ == "__main__":
    unittest.main()
