from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import importlib.util
import unittest
import sys
import wave
import zipfile

import p64.build.pipeline as build_pipeline
from p64.build.pipeline import build_executable, build_hub_app, create_runtime_bundle, create_runtime_package, validate_project
from p64.engine.builtin import STANDARD_SHADER_RELATIVE
from p64.engine.components import Collider, EntityPhysics, ScriptComponent, ScriptEntry
from p64.engine.entity import Entity
from p64.engine.math import Vec3
from p64.engine.project import Project
from p64.engine.runtime_session import RuntimeSession
from p64.engine.scene import Scene
from p64.engine.scene_manager import SceneManager


class ScriptingBuildTests(unittest.TestCase):
    def test_script_lifecycle_runs_without_crashing(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "spin.py").write_text(
                "from p64.engine.scripting import GameScript\n"
                "class Spin(GameScript):\n"
                "    speed = 60\n"
                "    def on_start(self):\n"
                "        self.started = True\n"
                "    def on_update(self, dt):\n"
                "        self.transform.rotation.y += self.speed * dt\n",
                encoding="utf-8",
            )
            scene = project.load_startup_scene()
            door = Entity("Door")
            door.add_component(ScriptComponent(scripts=[ScriptEntry(script="spin.py", class_name="Spin")]))
            scene.add_entity(door)
            project.save_startup_scene(scene)

            errors = project.load_startup_scene().run_scripts_once(project.root, dt=0.5)
            reloaded = project.load_startup_scene().entities[-1]
            self.assertEqual(errors, [])
            self.assertEqual(reloaded.transform.rotation.y, 0)

    def test_runtime_script_can_import_generated_project_api(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "reader.py").write_text(
                "from p64.engine.scripting import GameScript\n"
                "from p64_project_api import SCENE_NAME_MAIN\n"
                "class Reader(GameScript):\n"
                "    def on_update(self, dt):\n"
                "        self.entity.name = SCENE_NAME_MAIN\n",
                encoding="utf-8",
            )
            scene = project.load_startup_scene()
            actor = Entity("Actor")
            actor.add_component(ScriptComponent(scripts=[ScriptEntry(script="reader.py", class_name="Reader")]))
            scene.add_entity(actor)

            session = RuntimeSession(project, scene)
            self.assertEqual(session.tick(1 / 60), [])
            self.assertEqual(session.scene.find(actor.id).name, "main")

    def test_legacy_gamescript_name_is_not_available(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "old.py").write_text(
                "from p64.engine.scripting import UserScript\n"
                "class Old(UserScript):\n"
                "    pass\n",
                encoding="utf-8",
            )
            scene = project.load_startup_scene()
            actor = Entity("Actor")
            actor.add_component(ScriptComponent(scripts=[ScriptEntry(script="old.py", class_name="Old")]))
            scene.add_entity(actor)

            errors = RuntimeSession(project, scene).tick(1 / 60)

            self.assertTrue(any("UserScript" in error for error in errors))

    def test_runtime_session_runs_start_once_and_update_each_tick(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "counter.py").write_text(
                "from p64.engine.scripting import GameScript\n"
                "class Counter(GameScript):\n"
                "    def on_start(self):\n"
                "        self.transform.position.x += 10\n"
                "    def on_update(self, dt):\n"
                "        self.transform.position.x += 1\n",
                encoding="utf-8",
            )
            scene = project.load_startup_scene()
            actor = Entity("Actor")
            actor.add_component(ScriptComponent(scripts=[ScriptEntry(script="counter.py", class_name="Counter")]))
            scene.add_entity(actor)

            session = RuntimeSession(project, scene)
            self.assertEqual(session.tick(1 / 60), [])
            self.assertEqual(session.tick(1 / 60), [])

            runtime_actor = session.scene.find(actor.id)
            self.assertIsNotNone(runtime_actor)
            self.assertEqual(runtime_actor.transform.position.x, 12)

    def test_runtime_session_applies_entity_physics_after_script_update(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "push.py").write_text(
                "from p64.engine.math import Vec3\n"
                "from p64.engine.scripting import GameScript\n"
                "class Push(GameScript):\n"
                "    def on_update(self, dt):\n"
                "        self.entity_physics.add_force(Vec3(1.0, 0.0, 0.0) * 4.0)\n",
                encoding="utf-8",
            )
            scene = project.load_startup_scene()
            actor = Entity("Actor")
            actor.add_component(Collider())
            actor.add_component(EntityPhysics(use_gravity=False))
            actor.add_component(ScriptComponent(scripts=[ScriptEntry(script="push.py", class_name="Push")]))
            scene.add_entity(actor)

            session = RuntimeSession(project, scene)
            errors = session.tick(1.0)

            self.assertEqual(errors, [])
            runtime_actor = session.scene.find(actor.id)
            self.assertIsNotNone(runtime_actor)
            self.assertGreater(runtime_actor.transform.position.x, 0.0)

    def test_runtime_script_can_push_entity_along_world_forward(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "drive.py").write_text(
                "from p64.engine.scripting import GameScript\n"
                "class Drive(GameScript):\n"
                "    def on_update(self, dt):\n"
                "        self.entity_physics.add_force(self.forward * 4.0)\n",
                encoding="utf-8",
            )
            scene = project.load_startup_scene()
            car = Entity("Car")
            car.transform.rotation = Vec3(0.0, 90.0, 0.0)
            car.add_component(EntityPhysics(use_gravity=False))
            car.add_component(ScriptComponent(scripts=[ScriptEntry(script="drive.py", class_name="Drive")]))
            scene.add_entity(car)

            session = RuntimeSession(project, scene)
            errors = session.tick(1.0)

            self.assertEqual(errors, [])
            runtime_car = session.scene.find(car.id)
            self.assertIsNotNone(runtime_car)
            self.assertLess(runtime_car.transform.position.x, 0.0)
            self.assertAlmostEqual(runtime_car.transform.position.z, 0.0, places=6)

    def test_runtime_script_can_use_transform_forward(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "drive.py").write_text(
                "from p64.engine.scripting import GameScript\n"
                "class Drive(GameScript):\n"
                "    def on_update(self, dt):\n"
                "        self.entity_physics.add_force(self.transform.forward * 4.0)\n",
                encoding="utf-8",
            )
            scene = project.load_startup_scene()
            car = Entity("Car")
            car.transform.rotation = Vec3(0.0, 90.0, 0.0)
            car.add_component(EntityPhysics(use_gravity=False))
            car.add_component(ScriptComponent(scripts=[ScriptEntry(script="drive.py", class_name="Drive")]))
            scene.add_entity(car)

            session = RuntimeSession(project, scene)
            errors = session.tick(1.0)

            self.assertEqual(errors, [])
            runtime_car = session.scene.find(car.id)
            self.assertIsNotNone(runtime_car)
            self.assertLess(runtime_car.transform.position.x, 0.0)
            self.assertAlmostEqual(runtime_car.transform.position.z, 0.0, places=6)

    def test_runtime_session_clamps_physics_dt_but_not_script_dt(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "move_by_dt.py").write_text(
                "from p64.engine.scripting import GameScript\n"
                "class MoveByDt(GameScript):\n"
                "    def on_update(self, dt):\n"
                "        self.transform.position.x += dt\n",
                encoding="utf-8",
            )
            scene = project.load_startup_scene()
            actor = Entity("Actor")
            actor.add_component(EntityPhysics(use_gravity=False, velocity=Vec3(10.0, 0.0, 0.0)))
            actor.add_component(ScriptComponent(scripts=[ScriptEntry(script="move_by_dt.py", class_name="MoveByDt")]))
            scene.add_entity(actor)

            session = RuntimeSession(project, scene)
            errors = session.tick(1.0)

            self.assertEqual(errors, [])
            runtime_actor = session.scene.find(actor.id)
            self.assertIsNotNone(runtime_actor)
            self.assertAlmostEqual(runtime_actor.transform.position.x, 1.5)

    def test_runtime_session_mutates_scene_copy_only(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "move.py").write_text(
                "from p64.engine.scripting import GameScript\n"
                "class Move(GameScript):\n"
                "    def on_update(self, dt):\n"
                "        self.transform.position.x += 5\n",
                encoding="utf-8",
            )
            edit_scene = project.load_startup_scene()
            actor = Entity("Actor")
            actor.add_component(ScriptComponent(scripts=[ScriptEntry(script="move.py", class_name="Move")]))
            edit_scene.add_entity(actor)
            runtime_scene = Scene.from_dict(edit_scene.to_dict())

            session = RuntimeSession(project, runtime_scene)
            self.assertEqual(session.tick(1 / 60), [])

            self.assertEqual(edit_scene.find(actor.id).transform.position.x, 0.0)
            self.assertEqual(session.scene.find(actor.id).transform.position.x, 5.0)

    def test_inactive_entities_and_disabled_entries_do_not_run(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "bad.py").write_text(
                "from p64.engine.scripting import GameScript\n"
                "class Bad(GameScript):\n"
                "    def on_start(self):\n"
                "        raise RuntimeError('should not run')\n",
                encoding="utf-8",
            )
            scene = project.load_startup_scene()
            inactive = Entity("Inactive", active=False)
            inactive.add_component(ScriptComponent(scripts=[ScriptEntry(script="bad.py", class_name="Bad")]))
            disabled = Entity("DisabledEntry")
            disabled.add_component(ScriptComponent(scripts=[ScriptEntry(script="bad.py", class_name="Bad", enabled=False)]))
            scene.add_entity(inactive)
            scene.add_entity(disabled)
            project.save_startup_scene(scene)

            errors = project.load_startup_scene().run_scripts_once(project.root)
            self.assertEqual(errors, [])

    def test_validate_and_bundle_project(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            report = validate_project(project.root)
            bundle = create_runtime_bundle(project.root)

            self.assertTrue(report.ok)
            self.assertTrue((bundle / "run_game.py").exists())
            self.assertTrue((bundle / "project.p64").exists())
            self.assertTrue((bundle / "assets" / "scenes" / "main.scenep64").exists())
            self.assertTrue((bundle / STANDARD_SHADER_RELATIVE).exists())
            launcher = (bundle / "run_game.py").read_text(encoding="utf-8")
            self.assertIn("sys, 'frozen'", launcher)
            self.assertIn("sys, '_MEIPASS'", launcher)
            self.assertIn("Path(__file__).resolve().parent", launcher)

    def test_runtime_package_contains_game_data_only(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "spin.py").write_text("class Spin:\n    pass\n", encoding="utf-8")
            bundle = create_runtime_bundle(project.root)
            (bundle / "libraries").mkdir()
            (bundle / "libraries" / "builder.py").write_text("ignored", encoding="utf-8")
            (bundle / "build").mkdir()
            (bundle / "build" / "temp.txt").write_text("ignored", encoding="utf-8")

            package = create_runtime_package(bundle)

            with zipfile.ZipFile(package) as archive:
                names = set(archive.namelist())
            self.assertIn("project.p64", names)
            self.assertIn("assets/scenes/main.scenep64", names)
            self.assertIn("assets/scripts/spin.py", names)
            self.assertIn(STANDARD_SHADER_RELATIVE, names)
            self.assertNotIn("libraries/builder.py", names)
            self.assertNotIn("build/temp.txt", names)

    def test_runtime_bundle_forces_audio_import_before_copy(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")

            with mock.patch.object(build_pipeline, "ensure_audio_clips_for_assets") as ensure:
                create_runtime_bundle(project.root)

            self.assertTrue(any(call.kwargs.get("force") is True for call in ensure.call_args_list))

    def test_runtime_package_contains_generated_audio(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            _write_test_wav(project.assets_dir / "tone.wav")

            bundle = create_runtime_bundle(project.root)
            package = create_runtime_package(bundle)

            with zipfile.ZipFile(package) as archive:
                names = set(archive.namelist())
            generated = [name for name in names if name.startswith("packages/P64Generated/audio/") and name.endswith(".wav")]
            self.assertTrue(generated)

    def test_runtime_package_reports_missing_generated_audio(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            _write_test_wav(project.assets_dir / "tone.wav")
            bundle = create_runtime_bundle(project.root)
            for path in (bundle / "packages" / "P64Generated" / "audio").glob("*.wav"):
                path.unlink()

            with self.assertRaisesRegex(RuntimeError, "generated audio"):
                create_runtime_package(bundle)

    def test_validate_reports_invalid_settings(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            project.startup_scene = "assets/scenes/not_a_scene.txt"
            project.build_settings["output_folder"] = "../outside"
            project.save()

            report = validate_project(project.root)

            self.assertTrue(any("Startup scene must" in error for error in report.errors))
            self.assertTrue(any("Build output folder" in error for error in report.errors))

    def test_script_can_request_scene_switch(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            Scene("Second", [Entity("Marker")]).save(project.scenes_dir / "second.scenep64")
            (project.scripts_dir / "switcher.py").write_text(
                "from p64.engine.scripting import GameScript\n"
                "class Switcher(GameScript):\n"
                "    def on_update(self, dt):\n"
                "        self.scene_manager.load_scene_by_name('second')\n",
                encoding="utf-8",
            )
            scene = project.load_startup_scene()
            switcher = Entity("Switcher")
            switcher.add_component(ScriptComponent(scripts=[ScriptEntry(script="switcher.py", class_name="Switcher")]))
            scene.add_entity(switcher)
            project.save_startup_scene(scene)
            manager = SceneManager(project)

            errors = manager.current_scene.run_scripts_once(project.root, scene_manager=manager)

            self.assertEqual(errors, [])
            self.assertEqual(manager.current_scene.name, "Second")

    def test_runtime_session_applies_scene_switch_on_tick(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            Scene("Second", [Entity("Marker")]).save(project.scenes_dir / "second.scenep64")
            (project.scripts_dir / "switcher.py").write_text(
                "from p64.engine.scripting import GameScript\n"
                "class Switcher(GameScript):\n"
                "    def on_update(self, dt):\n"
                "        self.scene_manager.load_scene_by_name('second')\n",
                encoding="utf-8",
            )
            scene = project.load_startup_scene()
            switcher = Entity("Switcher")
            switcher.add_component(ScriptComponent(scripts=[ScriptEntry(script="switcher.py", class_name="Switcher")]))
            scene.add_entity(switcher)

            session = RuntimeSession(project, scene)
            self.assertEqual(session.tick(1 / 60), [])

            self.assertEqual(session.scene.name, "Second")

    def test_validate_reports_script_syntax_error(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "broken.py").write_text(
                "def broken(:\n"
                "    pass\n",
                encoding="utf-8",
            )

            report = validate_project(project.root)

            self.assertTrue(any("Script syntax error" in error and "broken.py" in error for error in report.errors))

    def test_bundle_fails_on_script_syntax_error(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "broken.py").write_text("def broken(:\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Script syntax error"):
                create_runtime_bundle(project.root)

    def test_build_executable_runs_pyinstaller_with_python_in_source_mode(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            project.build_settings["python_executable"] = sys.executable
            project.save()

            with mock.patch.object(build_pipeline.subprocess, "run") as run:
                output = build_executable(project.root)

            command = run.call_args.args[0]
            self.assertEqual(command[:2], [sys.executable, str(project.build_pipeline_dir / "builder.py")])
            self.assertIn("--p64-source", command)
            self.assertEqual(command[command.index("--p64-source") + 1], str(project.build_pipeline_dir / "p64_source"))
            self.assertIn("--project-package", command)
            project_package = Path(command[command.index("--project-package") + 1])
            self.assertTrue(project_package.exists())
            self.assertEqual(project_package.name, build_pipeline.PROJECT_PACKAGE_FILE)
            self.assertNotIn("-m", command)
            self.assertEqual(run.call_args.kwargs["cwd"], project.root)
            self.assertTrue(str(output).endswith(str(Path("build/game/Game"))))

    def test_build_executable_uses_external_python_for_frozen_hub(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            python = Path(tmp) / "Python" / "python.exe"
            python.parent.mkdir()
            python.write_text("", encoding="utf-8")
            project.build_settings["python_executable"] = str(python)
            project.save()
            executable = Path(tmp) / "P64" / "P64Hub.exe"
            executable.parent.mkdir()

            with (
                mock.patch.object(build_pipeline.sys, "frozen", True, create=True),
                mock.patch.object(build_pipeline.sys, "executable", str(executable)),
                mock.patch.object(build_pipeline.subprocess, "run") as run,
            ):
                build_executable(project.root)

            command = run.call_args.args[0]
            self.assertEqual(command[0], str(python))
            self.assertNotEqual(command[0], str(executable))
            self.assertNotIn("-m", command)

    def test_build_executable_reports_missing_external_python(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            project.build_settings["python_executable"] = str(Path(tmp) / "missing-python.exe")
            project.save()

            with (
                mock.patch.object(build_pipeline.shutil, "which", return_value=None),
                self.assertRaisesRegex(RuntimeError, "No Python executable found"),
            ):
                build_executable(project.root)

    def test_build_executable_passes_icon_to_project_builder(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            project.build_settings["python_executable"] = sys.executable
            icon = project.root / "assets" / "icon.ico"
            icon.write_text("", encoding="utf-8")
            project.build_settings["icon_path"] = "assets/icon.ico"
            project.save()

            with mock.patch.object(build_pipeline.subprocess, "run") as run:
                build_executable(project.root)

            command = run.call_args.args[0]
            self.assertIn("--icon", command)
            self.assertEqual(command[command.index("--icon") + 1], str(icon))

    def test_build_hub_app_collects_pyinstaller_and_embeds_p64_source(self):
        captured: dict[str, list[str]] = {}

        def fake_run(args: list[str], cwd: Path | None = None) -> None:
            captured["args"] = args

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "P64"
            with mock.patch.object(build_pipeline, "_run_pyinstaller", side_effect=fake_run):
                build_hub_app(output)

        args = captured["args"]
        self.assertNotIn("--collect-all", args)
        self.assertNotIn("--copy-metadata", args)
        self.assertIn("--add-data", args)
        self.assertTrue(args[args.index("--add-data") + 1].endswith(f"{build_pipeline.os.pathsep}p64_source/p64"))

    def test_project_builder_adds_runtime_package_to_pyinstaller_command(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            builder = project.build_pipeline_dir / "builder.py"
            spec = importlib.util.spec_from_file_location("p64_test_builder", builder)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            captured: dict[str, list[str]] = {}

            def fake_run(command: list[str], cwd: Path | str | None = None) -> object:
                captured["command"] = command
                return type("Result", (), {"returncode": 0})()

            module._needs_bootstrap = lambda args: False
            module.subprocess.run = fake_run
            module.sys.argv = [
                str(builder),
                "--bundle",
                str(project.build_dir / "bundle"),
                "--distpath",
                str(project.build_dir / "game"),
                "--workpath",
                str(project.build_dir / "pyinstaller-work"),
                "--specpath",
                str(project.build_dir / "pyinstaller-spec"),
                "--name",
                "Game",
                "--p64-source",
                str(project.build_pipeline_dir / "p64_source"),
                "--env-dir",
                str(project.build_dir / "p64-build-env"),
                "--requirements",
                str(project.build_pipeline_dir / "requirements-build.txt"),
                "--project-package",
                str(project.build_dir / "bundle" / build_pipeline.PROJECT_PACKAGE_FILE),
            ]

            self.assertEqual(module.main(), 0)

            command = captured["command"]
            self.assertIn("--add-data", command)
            add_data = command[command.index("--add-data") + 1]
            self.assertTrue(add_data.endswith(f"{build_pipeline.os.pathsep}."))
            self.assertIn(build_pipeline.PROJECT_PACKAGE_FILE, add_data)
            for import_name in ["pygame", "pygame.mixer", "pygame.sndarray", "numpy"]:
                self.assertIn(import_name, command)
            self.assertIn("--collect-submodules", command)
            self.assertIn("--collect-binaries", command)

def _write_test_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 32)


if __name__ == "__main__":
    unittest.main()
