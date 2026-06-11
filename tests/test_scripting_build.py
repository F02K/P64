from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from p64.build.pipeline import create_runtime_bundle, validate_project
from p64.engine.components import ScriptComponent, ScriptEntry
from p64.engine.entity import Entity
from p64.engine.project import Project


class ScriptingBuildTests(unittest.TestCase):
    def test_script_lifecycle_runs_without_crashing(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "spin.py").write_text(
                "from p64.engine.scripting import UserScript\n"
                "class Spin(UserScript):\n"
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

    def test_inactive_entities_and_disabled_entries_do_not_run(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "bad.py").write_text(
                "from p64.engine.scripting import UserScript\n"
                "class Bad(UserScript):\n"
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


if __name__ == "__main__":
    unittest.main()
