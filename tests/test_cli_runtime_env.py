from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import subprocess
import unittest

import p64.__main__ as p64_main_module
from p64.engine.project import Project


class CliRuntimeEnvTests(unittest.TestCase):
    def test_editor_relaunches_in_project_runtime_env(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            result = subprocess.CompletedProcess([], 7)

            with (
                mock.patch.object(p64_main_module, "is_running_in_project_runtime_env", return_value=False),
                mock.patch.object(p64_main_module, "ensure_project_runtime_env", return_value=project.runtime_python) as ensure_env,
                mock.patch.object(p64_main_module.subprocess, "run", return_value=result) as run,
            ):
                exit_code = p64_main_module.main(["editor", str(project.root)])

            self.assertEqual(exit_code, 7)
            ensure_env.assert_called_once()
            self.assertEqual(run.call_args.args[0], [str(project.runtime_python), "-m", "p64", "editor", str(project.project_file)])

    def test_editor_runs_directly_inside_project_runtime_env(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")

            with (
                mock.patch.object(p64_main_module, "is_running_in_project_runtime_env", return_value=True),
                mock.patch.object(p64_main_module, "ensure_project_runtime_env") as ensure_env,
                mock.patch("p64.editor.app.launch_editor") as launch_editor,
            ):
                exit_code = p64_main_module.main(["editor", str(project.project_file)])

            self.assertEqual(exit_code, 0)
            ensure_env.assert_not_called()
            launch_editor.assert_called_once_with(project.project_file)

    def test_run_relaunches_in_project_runtime_env(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            result = subprocess.CompletedProcess([], 3)

            with (
                mock.patch.object(p64_main_module, "is_running_in_project_runtime_env", return_value=False),
                mock.patch.object(p64_main_module, "ensure_project_runtime_env", return_value=project.runtime_python),
                mock.patch.object(p64_main_module.subprocess, "run", return_value=result) as run,
            ):
                exit_code = p64_main_module.main(["run", str(project.root)])

            self.assertEqual(exit_code, 3)
            self.assertEqual(run.call_args.args[0], [str(project.runtime_python), "-m", "p64", "run", str(project.project_file)])

    def test_run_executes_directly_inside_project_runtime_env(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")

            with (
                mock.patch.object(p64_main_module, "is_running_in_project_runtime_env", return_value=True),
                mock.patch.object(p64_main_module, "ensure_project_runtime_env") as ensure_env,
                mock.patch.object(p64_main_module, "run_project") as run_project,
            ):
                exit_code = p64_main_module.main(["run", str(project.project_file)])

            self.assertEqual(exit_code, 0)
            ensure_env.assert_not_called()
            run_project.assert_called_once_with(project.root)


if __name__ == "__main__":
    unittest.main()
