from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import unittest

from p64.engine.project import Project
import p64.hub.app as hub_app
from p64.hub.app import _editor_command, _editor_process_kwargs
from p64.hub.registry import ProjectRegistry


class HubRegistryTests(unittest.TestCase):
    def test_registry_adds_project_file_and_deduplicates(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "projects.json"
            project = Project.create(root / "Game", name="Game")
            registry = ProjectRegistry.load(registry_path)

            registry.add(project.root / "project.p64")
            registry.add(project.root)
            reloaded = ProjectRegistry.load(registry_path)

            self.assertEqual(len(reloaded.projects), 1)
            self.assertEqual(reloaded.projects[0].name, "Game")
            self.assertEqual(reloaded.projects[0].path, project.root)

    def test_remove_only_changes_registry(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.create(root / "Game")
            registry = ProjectRegistry.load(root / "projects.json")
            registry.add(project.root)

            self.assertTrue(registry.remove(project.root))
            self.assertTrue(project.project_file.exists())
            self.assertEqual(ProjectRegistry.load(root / "projects.json").projects, [])

    def test_delete_refuses_non_project_folder(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = ProjectRegistry.load(root / "projects.json")
            folder = root / "NotProject"
            folder.mkdir()

            with self.assertRaises(ValueError):
                registry.delete_project(folder)
            self.assertTrue(folder.exists())

    def test_editor_command_uses_project_runtime_gui_python(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            pythonw = project.runtime_env_dir / "Scripts" / "pythonw.exe"
            pythonw.parent.mkdir(parents=True)
            pythonw.write_text("", encoding="utf-8")

            with mock.patch.object(hub_app.os, "name", "nt"):
                command = _editor_command(project)

            self.assertEqual(command, [str(pythonw), "-m", "p64", "editor", str(project.project_file)])

    def test_editor_process_kwargs_hide_windows_console(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")

            with (
                mock.patch.object(hub_app.os, "name", "nt"),
                mock.patch.object(hub_app.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True),
            ):
                kwargs = _editor_process_kwargs(project)

            self.assertEqual(kwargs["cwd"], str(project.root))
            self.assertEqual(kwargs["creationflags"], 0x08000000)


if __name__ == "__main__":
    unittest.main()
