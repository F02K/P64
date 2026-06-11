from __future__ import annotations

import json
from pathlib import Path

from p64.engine.files import (
    LEGACY_METADATA_SUFFIX,
    LEGACY_PROJECT_FILE,
    LEGACY_SCENE_SUFFIX,
    METADATA_SUFFIX,
    PROJECT_FILE,
    SCENE_SUFFIX,
    normalize_scene_path,
    project_root_from_path,
)
from p64.engine.project import Project


def migrate_project_files(project_path: Path) -> list[str]:
    root = project_root_from_path(project_path)
    changes: list[str] = []

    legacy_project = root / LEGACY_PROJECT_FILE
    native_project = root / PROJECT_FILE
    if legacy_project.exists() and not native_project.exists():
        data = json.loads(legacy_project.read_text(encoding="utf-8"))
        data["startup_scene"] = normalize_scene_path(str(data.get("startup_scene", "")))
        legacy_project.rename(native_project)
        native_project.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        changes.append(f"{LEGACY_PROJECT_FILE} -> {PROJECT_FILE}")

    project = Project.load(root)
    for scene_path in _source_files(root, f"*{LEGACY_SCENE_SUFFIX}"):
        native = Path(scene_path.as_posix()[: -len(LEGACY_SCENE_SUFFIX)] + SCENE_SUFFIX)
        if not native.exists():
            scene_path.rename(native)
            changes.append(f"{scene_path.relative_to(root)} -> {native.relative_to(root)}")

    for metadata_path in _source_files(root, f"*{LEGACY_METADATA_SUFFIX}"):
        native = Path(metadata_path.as_posix()[: -len(LEGACY_METADATA_SUFFIX)] + METADATA_SUFFIX)
        if not native.exists():
            metadata_path.rename(native)
            changes.append(f"{metadata_path.relative_to(root)} -> {native.relative_to(root)}")

    project.startup_scene = normalize_scene_path(project.startup_scene)
    project.save()
    return changes


def _source_files(root: Path, pattern: str) -> list[Path]:
    build_root = (root / "build").resolve()
    paths: list[Path] = []
    for path in root.rglob(pattern):
        try:
            path.resolve().relative_to(build_root)
            continue
        except ValueError:
            paths.append(path)
    return paths
