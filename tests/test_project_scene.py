from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import shutil
import subprocess
import unittest

from p64.__main__ import main as p64_main
import p64.engine.project as project_module
from p64.engine.builtin import LEGACY_STANDARD_SHADER_RELATIVE, PARTICLE_MATERIAL_RELATIVE, SPRITE_MATERIAL_RELATIVE, STANDARD_SHADER_RELATIVE, STANDARD_UNLIT_SHADER_RELATIVE, UI_IMAGE_MATERIAL_RELATIVE
from p64.engine.components import AudioListener, Camera, Canvas, EntityPhysics, Fog, Light, MeshRenderer, ModelRenderer, ParticleEmitter, RectTransform, ScriptComponent, ScriptEntry, SpawnPoint, SpriteRenderer, Transform, UIImage, UIText
from p64.engine.entity import GAME_OBJECT, Entity, entity_effectively_active, set_object_type_recursive
from p64.engine.migration import migrate_project_files
from p64.engine.math import Vec3
from p64.engine.project import Project, _builder_script_source, _source_p64_package_dir, default_render_settings, ensure_project_runtime_env, is_project_runtime_env_ready
from p64.engine.render_settings import clamp_render_settings
from p64.engine.scene import Scene
from p64.engine.scene_manager import SceneManager
from p64.engine.transforms import local_to_world_direction, set_world_position, world_forward, world_position, world_right, world_rotation, world_scale, world_up
from p64.engine.vscode import setup_vscode_project


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
            self.assertTrue((project.root / SPRITE_MATERIAL_RELATIVE).exists())
            self.assertTrue((project.root / UI_IMAGE_MATERIAL_RELATIVE).exists())
            self.assertTrue((project.root / PARTICLE_MATERIAL_RELATIVE).exists())
            self.assertTrue((project.root / "libraries" / "P64Build" / "builder.py").exists())
            self.assertTrue((project.root / "libraries" / "P64Build" / "requirements-build.txt").exists())
            self.assertTrue((project.root / ".vscode" / "settings.json").exists())
            self.assertTrue((project.root / ".vscode" / "tasks.json").exists())
            self.assertTrue((project.root / ".vscode" / "extensions.json").exists())
            self.assertTrue((project.root / ".vscode" / "launch.json").exists())
            self.assertTrue(project.project_api_path.exists())
            self.assertFalse((project.scripts_dir / "p64_project_api.py").exists())
            requirements = (project.root / "libraries" / "P64Build" / "requirements-build.txt").read_text(encoding="utf-8")
            self.assertIn("pygame>=2.5", requirements)
            self.assertTrue((project.root / "libraries" / "P64Build" / "p64_source" / "p64" / "engine" / "runtime.py").exists())
            self.assertTrue((project.root / "assets").exists())
            self.assertIsNotNone(reloaded.active_camera())
            self.assertEqual(reloaded.entities[-1].name, "Thing")

    def test_project_runtime_python_uses_project_venv(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")

            self.assertEqual(project.runtime_env_dir, project.root / ".venv")
            expected = project.runtime_env_dir / ("Scripts" if project_module.os.name == "nt" else "bin")
            self.assertEqual(project.runtime_python.parent, expected)

    def test_project_load_refreshes_missing_builtin_materials(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            material = project.root / SPRITE_MATERIAL_RELATIVE
            metadata = material.with_suffix(material.suffix + ".mdp64")
            material.unlink()
            metadata.unlink()

            Project.load(project.root)

            self.assertTrue(material.exists())
            self.assertTrue(metadata.exists())

    def test_project_runtime_gui_python_prefers_pythonw_on_windows(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            pythonw = project.runtime_env_dir / "Scripts" / "pythonw.exe"
            pythonw.parent.mkdir(parents=True)
            pythonw.write_text("", encoding="utf-8")

            with mock.patch.object(project_module.os, "name", "nt"):
                self.assertEqual(project.runtime_gui_python, pythonw)

            pythonw.unlink()
            with mock.patch.object(project_module.os, "name", "nt"):
                self.assertEqual(project.runtime_gui_python, project.runtime_python)

    def test_project_runtime_env_ready_requires_python_and_imports(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")

            self.assertFalse(is_project_runtime_env_ready(project))

            project.runtime_python.parent.mkdir(parents=True)
            project.runtime_python.write_text("", encoding="utf-8")
            with mock.patch.object(project_module.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)):
                self.assertFalse(is_project_runtime_env_ready(project))
            with mock.patch.object(project_module.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
                self.assertTrue(is_project_runtime_env_ready(project))

    def test_ensure_project_runtime_env_creates_venv_and_installs_editable_project(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")

            with (
                mock.patch.object(project_module, "is_project_runtime_env_ready", side_effect=[False, True]),
                mock.patch.object(project_module.venv, "EnvBuilder") as builder,
                mock.patch.object(project_module.subprocess, "run") as run,
            ):
                python = ensure_project_runtime_env(project)

            self.assertEqual(python, project.runtime_python)
            builder.assert_called_once_with(with_pip=True, clear=False)
            builder.return_value.create.assert_called_once_with(project.runtime_env_dir)
            command = run.call_args.args[0]
            self.assertEqual(command[:5], [str(project.runtime_python), "-m", "pip", "install", "--upgrade"])
            self.assertIn("-e", command)
            self.assertTrue(command[-1].endswith("[dev]"))

    def test_ensure_project_runtime_env_streams_install_output_to_logger(self):
        class FakeProcess:
            stdout = ["Collecting pygame\n", "Successfully installed\n"]

            def wait(self):
                return 0

        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            messages: list[str] = []

            with (
                mock.patch.object(project_module, "is_project_runtime_env_ready", side_effect=[False, True]),
                mock.patch.object(project_module.venv, "EnvBuilder"),
                mock.patch.object(project_module.subprocess, "Popen", return_value=FakeProcess()) as popen,
            ):
                ensure_project_runtime_env(project, messages.append)

            self.assertTrue(any("Preparing project Python environment" in message for message in messages))
            self.assertIn("Collecting pygame", messages)
            self.assertIn("Successfully installed", messages)
            self.assertEqual(popen.call_args.kwargs["stdout"], project_module.subprocess.PIPE)
            self.assertEqual(popen.call_args.kwargs["stderr"], project_module.subprocess.STDOUT)

    def test_ensure_project_runtime_env_clears_incomplete_existing_venv(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            project.runtime_env_dir.mkdir()

            with (
                mock.patch.object(project_module, "is_project_runtime_env_ready", side_effect=[False, True]),
                mock.patch.object(project_module.venv, "EnvBuilder") as builder,
                mock.patch.object(project_module.subprocess, "run"),
            ):
                ensure_project_runtime_env(project)

            builder.assert_called_once_with(with_pip=True, clear=True)
            builder.return_value.create.assert_called_once_with(project.runtime_env_dir)

    def test_ensure_project_runtime_env_uses_external_python_when_current_executable_is_frozen(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            project.runtime_env_dir.mkdir()
            create_command = ["C:\\Python313\\python.exe", "-m", "venv", "--clear", str(project.runtime_env_dir)]

            with (
                mock.patch.object(project_module, "is_project_runtime_env_ready", side_effect=[False, True]),
                mock.patch.object(project_module, "_can_use_current_python_for_venv", return_value=False),
                mock.patch.object(project_module, "_external_venv_command", return_value=create_command) as external_command,
                mock.patch.object(project_module.venv, "EnvBuilder") as builder,
                mock.patch.object(project_module.subprocess, "run") as run,
            ):
                ensure_project_runtime_env(project)

            external_command.assert_called_once_with(project.runtime_env_dir, clear=True)
            builder.assert_not_called()
            self.assertEqual(run.call_args_list[0].args[0], create_command)
            self.assertEqual(run.call_args_list[0].kwargs["check"], True)

    def test_vscode_setup_merges_existing_files_and_generates_project_api(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            settings_path = project.root / ".vscode" / "settings.json"
            tasks_path = project.root / ".vscode" / "tasks.json"
            settings_path.write_text(
                '{"editor.tabSize": 2, "python.analysis.extraPaths": ["custom"]}\n',
                encoding="utf-8",
            )
            tasks_path.write_text(
                '{"version": "2.0.0", "tasks": [{"label": "Custom", "type": "shell", "command": "echo ok"}]}\n',
                encoding="utf-8",
            )
            Scene("Second").save(project.scenes_dir / "second.scenep64")
            (project.assets_dir / "texture.png").write_text("", encoding="utf-8")

            setup_vscode_project(project)

            import json

            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertEqual(settings["editor.tabSize"], 2)
            self.assertIn("custom", settings["python.analysis.extraPaths"])
            self.assertIn("assets/scripts", settings["python.analysis.extraPaths"])
            self.assertIn("packages/P64Generated/python", settings["python.analysis.extraPaths"])
            tasks = json.loads(tasks_path.read_text(encoding="utf-8"))["tasks"]
            self.assertTrue(any(task["label"] == "Custom" for task in tasks))
            self.assertTrue(any(task["label"] == "P64: Validate" for task in tasks))
            api = project.project_api_path.read_text(encoding="utf-8")
            self.assertIn('SCENE_NAME_SECOND = "second"', api)
            self.assertIn('ASSET_PNG_TEXTURE = "assets/texture.png"', api)

    def test_vscode_cli_refreshes_setup(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            (project.root / ".vscode" / "tasks.json").unlink()

            result = p64_main(["vscode", str(project.root)])

            self.assertEqual(result, 0)
            self.assertTrue((project.root / ".vscode" / "tasks.json").exists())

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

    def test_render_settings_include_default_skybox_values(self):
        settings = default_render_settings()

        self.assertTrue(settings["skybox_enabled"])
        self.assertEqual(len(settings["skybox_top_color"]), 3)
        self.assertEqual(len(settings["skybox_horizon_color"]), 3)
        self.assertEqual(len(settings["skybox_cloud_color"]), 3)
        self.assertGreaterEqual(settings["skybox_cloud_coverage"], 0.0)
        self.assertLessEqual(settings["skybox_cloud_coverage"], 1.0)
        self.assertGreater(settings["skybox_cloud_scale"], 0.0)
        self.assertGreater(settings["skybox_cloud_height"], 0.0)
        self.assertGreaterEqual(settings["skybox_cloud_softness"], 0.0)
        self.assertLessEqual(settings["skybox_cloud_softness"], 1.0)

    def test_legacy_scene_load_gets_skybox_defaults(self):
        scene = Scene.from_dict({"name": "Legacy", "entities": []})

        self.assertTrue(scene.render_settings["skybox_enabled"])
        self.assertIn("skybox_cloud_coverage", scene.render_settings)
        self.assertIn("skybox_cloud_height", scene.render_settings)
        self.assertIn("skybox_cloud_softness", scene.render_settings)

    def test_skybox_cloud_values_are_clamped(self):
        settings = clamp_render_settings({
            "skybox_cloud_height": -5.0,
            "skybox_cloud_softness": 2.0,
            "skybox_cloud_coverage": -1.0,
        })

        self.assertEqual(settings["skybox_cloud_height"], 0.1)
        self.assertEqual(settings["skybox_cloud_softness"], 1.0)
        self.assertEqual(settings["skybox_cloud_coverage"], 0.0)

    def test_existing_project_save_adds_build_pipeline(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            shutil.rmtree(project.root / "libraries")

            Project.load(project.root).save()

            self.assertTrue((project.root / "libraries" / "P64Build" / "builder.py").exists())

    def test_project_load_refreshes_generated_build_pipeline_and_runtime_source(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            builder = project.build_pipeline_dir / "builder.py"
            runtime_copy = project.build_pipeline_dir / "p64_source" / "p64" / "engine" / "runtime.py"
            builder.write_text("# stale builder\n", encoding="utf-8")
            runtime_copy.write_text("# stale runtime\n", encoding="utf-8")

            Project.load(project.root)

            self.assertEqual(builder.read_text(encoding="utf-8"), _builder_script_source())
            source_runtime = _source_p64_package_dir() / "engine" / "runtime.py"
            self.assertEqual(runtime_copy.read_text(encoding="utf-8"), source_runtime.read_text(encoding="utf-8"))

    def test_project_load_recreates_missing_build_pipeline(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            shutil.rmtree(project.root / "libraries")

            loaded = Project.load(project.root)

            self.assertTrue((loaded.build_pipeline_dir / "builder.py").exists())
            self.assertTrue((loaded.build_pipeline_dir / "p64_source" / "p64" / "engine" / "runtime.py").exists())

    def test_project_load_refreshes_custom_build_pipeline_path(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            project.build_settings["build_pipeline_path"] = "libraries/CustomBuild"
            project.save()
            custom_builder = project.root / "libraries" / "CustomBuild" / "builder.py"
            custom_builder.write_text("# stale custom builder\n", encoding="utf-8")

            loaded = Project.load(project.root)

            self.assertEqual(loaded.build_settings["build_pipeline_path"], "libraries/CustomBuild")
            self.assertEqual(custom_builder.read_text(encoding="utf-8"), _builder_script_source())

    def test_project_load_refreshes_generated_builtin_shader_without_touching_user_shader(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game", name="Game")
            builtin_shader = project.root / STANDARD_SHADER_RELATIVE
            user_shader = project.assets_dir / "shaders" / "custom.shader"
            user_shader.parent.mkdir(parents=True, exist_ok=True)
            builtin_shader.write_text('Shader "P64Builtin/Standard VertexLit"\n{ stale }\n', encoding="utf-8")
            user_shader.write_text('Shader "Custom"\n{ keep me }\n', encoding="utf-8")

            Project.load(project.root)

            self.assertIn("u_base_color", builtin_shader.read_text(encoding="utf-8"))
            self.assertEqual(user_shader.read_text(encoding="utf-8"), 'Shader "Custom"\n{ keep me }\n')

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

    def test_child_world_transform_follows_parent_translation_rotation_and_scale(self):
        parent = Entity("Parent")
        child = Entity("Child")
        parent.transform.position = Vec3(10.0, 0.0, 0.0)
        parent.transform.rotation = Vec3(0.0, 90.0, 0.0)
        parent.transform.scale = Vec3(2.0, 3.0, 4.0)
        child.transform.position = Vec3(0.0, 0.0, -1.0)
        child.transform.rotation = Vec3(0.0, 15.0, 0.0)
        child.transform.scale = Vec3(0.5, 2.0, 1.0)
        parent.add_child(child)

        position = world_position(child)
        self.assertAlmostEqual(position.x, 6.0)
        self.assertAlmostEqual(position.y, 0.0)
        self.assertAlmostEqual(position.z, 0.0)
        self.assertEqual(world_rotation(child), Vec3(0.0, 105.0, 0.0))
        self.assertEqual(world_scale(child), Vec3(1.0, 6.0, 4.0))

    def test_set_world_position_writes_child_local_position(self):
        parent = Entity("Parent")
        child = parent.add_child(Entity("Child"))
        parent.transform.position = Vec3(10.0, 0.0, 0.0)
        parent.transform.rotation = Vec3(0.0, 90.0, 0.0)

        set_world_position(child, Vec3(12.0, 3.0, 4.0))

        self.assertAlmostEqual(world_position(child).x, 12.0)
        self.assertAlmostEqual(world_position(child).y, 3.0)
        self.assertAlmostEqual(world_position(child).z, 4.0)
        self.assertNotEqual(child.transform.position, Vec3(12.0, 3.0, 4.0))

    def test_world_direction_helpers_use_matrix_axes(self):
        entity = Entity("Car")
        entity.transform.rotation = Vec3(0.0, 90.0, 0.0)

        self.assertAlmostEqual(world_forward(entity).x, -1.0)
        self.assertAlmostEqual(world_forward(entity).z, 0.0, places=6)
        self.assertAlmostEqual(world_right(entity).z, -1.0)
        self.assertAlmostEqual(world_up(entity).y, 1.0)
        self.assertEqual(local_to_world_direction(entity, Vec3(0.0, 0.0, -2.0)), world_forward(entity))
        self.assertNotEqual(entity.transform.forward, Vec3.forward())
        self.assertEqual(entity.transform.forward, world_forward(entity))
        self.assertEqual(entity.transform.right, world_right(entity))
        self.assertEqual(entity.transform.up, world_up(entity))

    def test_child_world_forward_uses_combined_matrix_rotation(self):
        parent = Entity("Parent")
        child = parent.add_child(Entity("Child"))
        parent.transform.rotation = Vec3(30.0, 45.0, 0.0)
        child.transform.rotation = Vec3(0.0, 45.0, 0.0)

        forward = world_forward(child)
        right = world_right(child)
        up = world_up(child)

        self.assertAlmostEqual(forward.length(), 1.0)
        self.assertAlmostEqual(right.length(), 1.0)
        self.assertAlmostEqual(up.length(), 1.0)
        self.assertAlmostEqual(forward.dot(right), 0.0, places=6)
        self.assertNotEqual(forward, world_forward(parent))
        self.assertEqual(child.transform.forward, forward)
        self.assertNotEqual(child.transform.local_forward, forward)

    def test_transform_owner_binding_survives_assignment_and_child_parenting(self):
        parent = Entity("Parent")
        child = parent.add_child(Entity("Child"))
        replacement = Transform()

        child.transform = replacement

        self.assertIs(parent.transform.scene_object, parent)
        self.assertIs(child.transform.scene_object, child)
        self.assertIs(child.transform.sceneObject, child)
        self.assertIs(replacement.scene_object, child)

    def test_transform_point_helpers_roundtrip_through_hierarchy(self):
        parent = Entity("Parent")
        child = parent.add_child(Entity("Child"))
        parent.transform.position = Vec3(10.0, 0.0, 0.0)
        parent.transform.rotation = Vec3(0.0, 90.0, 0.0)
        child.transform.position = Vec3(0.0, 2.0, -1.0)
        local = Vec3(1.0, 2.0, 3.0)

        world = child.transform.transform_point(local)
        roundtripped = child.transform.inverse_transform_point(world)

        self.assertAlmostEqual(roundtripped.x, local.x)
        self.assertAlmostEqual(roundtripped.y, local.y)
        self.assertAlmostEqual(roundtripped.z, local.z)

    def test_scene_serializes_components(self):
        root = Entity("Root")
        root.add_component(
            MeshRenderer(
                mesh="mesh",
                submesh="Door",
                shader="assets/shaders/standard.shader",
                source_materials=["Frame", "Glass"],
                material_slots=["assets/materials/Door.material"],
            )
        )
        root.add_component(
            ModelRenderer(
                model="model",
                shader="assets/shaders/model.shader",
                source_materials=["Frame"],
                material_slots=["assets/materials/Frame.material"],
            )
        )
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
        self.assertEqual(loaded.entities[0].components[0].source_materials, ["Frame", "Glass"])
        self.assertEqual(loaded.entities[0].components[0].material_slots, ["assets/materials/Door.material"])
        model = loaded.entities[0].components[1]
        self.assertIsInstance(model, ModelRenderer)
        self.assertEqual(model.model, "model")
        self.assertEqual(model.shader, "assets/shaders/model.shader")
        self.assertEqual(model.material_slots, ["assets/materials/Frame.material"])
        self.assertEqual(loaded.entities[0].components[2].scripts[0].script, "spin.py")
        physics = loaded.entities[0].components[3]
        self.assertIsInstance(physics, EntityPhysics)
        self.assertEqual(physics.mass, 2.0)
        self.assertFalse(physics.use_gravity)
        self.assertEqual(physics.velocity.to_list(), [1.0, 2.0, 3.0])
        self.assertEqual(physics.freeze_rotation.to_list(), [0.0, 1.0, 0.0])

    def test_scene_serializes_sprite_ui_and_particle_components(self):
        root = Entity("Root")
        root.add_component(SpriteRenderer(texture="assets/textures/hero.png", alpha=0.5, flipbook_columns=4, flipbook_rows=2))
        root.add_component(Canvas(sort_order=3, reference_resolution=Vec3(640, 480, 0), resolution_mode="fixed"))
        root.add_component(UIImage(texture="assets/textures/hud.png", anchor="top-left", size=Vec3(64, 32, 0), fill_mode="fit"))
        root.add_component(UIText(text="Score", font_source="asset", font_family="Arial", bitmap_font="assets/fonts/ui.ttf", font_size=18, anchor="top"))
        root.add_component(ParticleEmitter(texture="assets/textures/spark.png", max_particles=16, rate=5.0, burst=3, blend_mode="additive"))
        scene = Scene("FX", [root])

        loaded = Scene.from_dict(scene.to_dict())

        self.assertIsInstance(loaded.entities[0].components[0], SpriteRenderer)
        self.assertEqual(loaded.entities[0].components[0].flipbook_columns, 4)
        self.assertIsInstance(loaded.entities[0].components[1], Canvas)
        self.assertEqual(loaded.entities[0].components[1].reference_resolution.to_list(), [640.0, 480.0, 0.0])
        self.assertEqual(loaded.entities[0].components[1].resolution_mode, "fixed")
        self.assertIsInstance(loaded.entities[0].components[2], UIImage)
        self.assertEqual(loaded.entities[0].components[2].anchor, "top-left")
        self.assertEqual(loaded.entities[0].components[2].fill_mode, "fit")
        self.assertIsInstance(loaded.entities[0].components[3], UIText)
        self.assertEqual(loaded.entities[0].components[3].text, "Score")
        self.assertEqual(loaded.entities[0].components[3].font_source, "asset")
        self.assertEqual(loaded.entities[0].components[3].bitmap_font, "assets/fonts/ui.ttf")
        self.assertIsInstance(loaded.entities[0].components[4], ParticleEmitter)
        self.assertEqual(loaded.entities[0].components[4].blend_mode, "additive")

    def test_legacy_ui_fields_use_new_defaults(self):
        scene = Scene.from_dict({
            "name": "Legacy UI",
            "entities": [{
                "name": "Canvas",
                "components": [
                    {"type": "Canvas", "reference_resolution": [640, 480, 0]},
                    {"type": "UIText", "text": "Ready", "font_family": "Arial"},
                ],
            }],
        })

        canvas = scene.entities[0].components[0]
        text = scene.entities[0].components[1]

        self.assertIsInstance(canvas, Canvas)
        self.assertEqual(canvas.resolution_mode, "auto")
        self.assertIsInstance(text, UIText)
        self.assertEqual(text.font_source, "system")

    def test_rect_transform_serializes_optionally(self):
        entity = Entity("Button", rect_transform=RectTransform(anchor="top-left", offset=Vec3(10, 20, 0), size=Vec3(200, 48, 0)))
        scene = Scene("UI", [entity])

        data = scene.to_dict()
        loaded = Scene.from_dict(data)
        legacy = Scene.from_dict({"name": "Legacy", "entities": [{"name": "Old"}]})

        self.assertIn("rect_transform", data["entities"][0])
        self.assertIsNotNone(loaded.entities[0].rect_transform)
        self.assertEqual(loaded.entities[0].rect_transform.size.to_list(), [200.0, 48.0, 0.0])
        self.assertIsNone(legacy.entities[0].rect_transform)

    def test_effective_active_respects_inactive_parent_without_overwriting_child(self):
        parent = Entity("Parent", active=False)
        child = parent.add_child(Entity("Child", components=[Camera()]))
        scene = Scene("Active", [parent])

        self.assertTrue(child.active)
        self.assertFalse(entity_effectively_active(child))
        self.assertEqual(list(scene.walk_active()), [])
        self.assertIsNone(scene.active_camera())

    def test_active_audio_listener_uses_first_enabled_active_listener(self):
        inactive_entity = Entity("Inactive", active=False, components=[AudioListener()])
        disabled_listener = Entity("Disabled", components=[AudioListener(enabled=False)])
        inactive_listener = Entity("Inactive Listener", components=[AudioListener(active=False)])
        active_listener = Entity("Active Listener", components=[AudioListener()])
        second_listener = Entity("Second Listener", components=[AudioListener()])
        scene = Scene("Audio", [inactive_entity, disabled_listener, inactive_listener, active_listener, second_listener])

        self.assertIs(scene.active_audio_listener(), active_listener)

    def test_active_audio_listener_ignores_inherited_inactive_listener(self):
        parent = Entity("Parent", active=False)
        child = parent.add_child(Entity("Child", components=[AudioListener()]))
        fallback = Entity("Fallback", components=[AudioListener()])
        scene = Scene("Audio", [parent, fallback])

        self.assertTrue(child.active)
        self.assertIs(scene.active_audio_listener(), fallback)

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
