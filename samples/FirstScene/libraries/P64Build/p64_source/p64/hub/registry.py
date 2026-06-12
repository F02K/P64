from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from p64.engine.files import PROJECT_FILE, is_project_root, project_root_from_path
from p64.engine.project import Project


@dataclass
class HubProject:
    path: Path
    name: str
    last_opened: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "name": self.name,
            "last_opened": self.last_opened,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HubProject":
        path = project_root_from_path(Path(str(data.get("path", ""))))
        name = str(data.get("name") or path.name)
        last_opened = data.get("last_opened")
        return cls(path=path, name=name, last_opened=str(last_opened) if last_opened else None)

    @property
    def exists(self) -> bool:
        return is_project_root(self.path)


def default_registry_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "P64" / "projects.json"
    return Path.home() / ".p64" / "projects.json"


class ProjectRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_registry_path()
        self.projects: list[HubProject] = []

    @classmethod
    def load(cls, path: Path | None = None) -> "ProjectRegistry":
        registry = cls(path)
        if not registry.path.exists():
            return registry
        data = json.loads(registry.path.read_text(encoding="utf-8"))
        for item in data.get("projects", []):
            try:
                registry.projects.append(HubProject.from_dict(item))
            except Exception:
                continue
        registry._deduplicate()
        return registry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._deduplicate()
        self.path.write_text(
            json.dumps({"projects": [project.to_dict() for project in self.projects]}, indent=2) + "\n",
            encoding="utf-8",
        )

    def add(self, project_path: Path, mark_opened: bool = False) -> HubProject:
        root = project_root_from_path(project_path)
        project = Project.load(root)
        entry = HubProject(path=project.root, name=project.name)
        existing = self._find(entry.path)
        if existing:
            existing.name = entry.name
            if mark_opened:
                existing.last_opened = _timestamp()
            self.save()
            return existing
        if mark_opened:
            entry.last_opened = _timestamp()
        self.projects.append(entry)
        self.save()
        return entry

    def create_project(self, project_path: Path, name: str | None = None) -> HubProject:
        project = Project.create(project_path, name=name)
        return self.add(project.root)

    def remove(self, project_path: Path) -> bool:
        root = project_root_from_path(project_path)
        before = len(self.projects)
        self.projects = [project for project in self.projects if project.path.resolve() != root.resolve()]
        changed = len(self.projects) != before
        if changed:
            self.save()
        return changed

    def mark_opened(self, project_path: Path) -> HubProject:
        entry = self.add(project_path, mark_opened=True)
        entry.last_opened = _timestamp()
        self.save()
        return entry

    def delete_project(self, project_path: Path) -> None:
        root = project_root_from_path(project_path)
        if not is_project_root(root):
            raise ValueError(f"Refusing to delete non-project folder: {root}")
        shutil.rmtree(root)
        self.remove(root)

    def _find(self, root: Path) -> HubProject | None:
        resolved = root.resolve()
        for project in self.projects:
            if project.path.resolve() == resolved:
                return project
        return None

    def _deduplicate(self) -> None:
        seen: set[Path] = set()
        deduped: list[HubProject] = []
        for project in self.projects:
            resolved = project.path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(project)
        self.projects = deduped


def file_association_command(executable: Path) -> str:
    exe = executable.resolve()
    return (
        f'assoc .p64=P64.Project && '
        f'ftype P64.Project="{exe}" "%1"'
    )


def project_file_path(root: Path) -> Path:
    return project_root_from_path(root) / PROJECT_FILE


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
