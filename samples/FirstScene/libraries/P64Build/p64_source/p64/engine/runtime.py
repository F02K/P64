from __future__ import annotations

from pathlib import Path

from p64.engine.project import Project
from p64.engine.runtime_session import RuntimeSession


def run_project(project_root: Path) -> None:
    project = Project.load(project_root)
    session = RuntimeSession(project)
    script_errors = session.tick(1 / 60)
    if script_errors:
        for error in script_errors:
            print(f"script error: {error}")

    try:
        from p64.editor.app import launch_runtime_window
    except ImportError as exc:
        raise RuntimeError("PySide6/ModernGL runtime dependencies are required to run a project.") from exc

    launch_runtime_window(project, session.scene)
