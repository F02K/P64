from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from p64.engine.project import Project
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


if __name__ == "__main__":
    unittest.main()
