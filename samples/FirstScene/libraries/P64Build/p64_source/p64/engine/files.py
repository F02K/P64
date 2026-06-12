from __future__ import annotations

from pathlib import Path


PROJECT_FILE = "project.p64"
LEGACY_PROJECT_FILE = "project.p64.json"

SCENE_SUFFIX = ".scenep64"
LEGACY_SCENE_SUFFIX = ".p64scene.json"
DEFAULT_SCENE = f"assets/scenes/main{SCENE_SUFFIX}"
LEGACY_DEFAULT_SCENE = f"scenes/main{LEGACY_SCENE_SUFFIX}"

METADATA_SUFFIX = ".mdp64"
LEGACY_METADATA_SUFFIX = ".p64asset.json"


def project_file_for(root: Path) -> Path:
    root = root.resolve()
    native = root / PROJECT_FILE
    if native.exists():
        return native
    legacy = root / LEGACY_PROJECT_FILE
    if legacy.exists():
        return legacy
    return native


def project_root_from_path(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        if path.name not in {PROJECT_FILE, LEGACY_PROJECT_FILE}:
            raise ValueError(f"Not a P64 project file: {path}")
        return path.parent
    return path


def is_project_root(path: Path) -> bool:
    root = project_root_from_path(path)
    return (root / PROJECT_FILE).exists() or (root / LEGACY_PROJECT_FILE).exists()


def normalize_scene_path(path: str) -> str:
    if path.endswith(LEGACY_SCENE_SUFFIX):
        return path[: -len(LEGACY_SCENE_SUFFIX)] + SCENE_SUFFIX
    return path


def alternate_scene_path(path: Path) -> Path:
    text = path.as_posix()
    if text.endswith(SCENE_SUFFIX):
        return Path(text[: -len(SCENE_SUFFIX)] + LEGACY_SCENE_SUFFIX)
    if text.endswith(LEGACY_SCENE_SUFFIX):
        return Path(text[: -len(LEGACY_SCENE_SUFFIX)] + SCENE_SUFFIX)
    return path


def is_scene_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(SCENE_SUFFIX) or name.endswith(LEGACY_SCENE_SUFFIX)


def metadata_path_for_source(source: Path) -> Path:
    return source.with_suffix(source.suffix + METADATA_SUFFIX)


def legacy_metadata_path_for_source(source: Path) -> Path:
    return source.with_suffix(source.suffix + LEGACY_METADATA_SUFFIX)


def is_metadata_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(METADATA_SUFFIX) or name.endswith(LEGACY_METADATA_SUFFIX)


def metadata_source_from_path(path: Path) -> Path:
    text = path.as_posix()
    if text.endswith(METADATA_SUFFIX):
        return Path(text[: -len(METADATA_SUFFIX)])
    if text.endswith(LEGACY_METADATA_SUFFIX):
        return Path(text[: -len(LEGACY_METADATA_SUFFIX)])
    return path


def find_metadata_for_source(source: Path) -> Path | None:
    native = metadata_path_for_source(source)
    if native.exists():
        return native
    legacy = legacy_metadata_path_for_source(source)
    if legacy.exists():
        return legacy
    return None


def iter_metadata_files(assets_dir: Path) -> list[Path]:
    if not assets_dir.exists():
        return []
    paths = list(assets_dir.rglob(f"*{METADATA_SUFFIX}"))
    paths.extend(path for path in assets_dir.rglob(f"*{LEGACY_METADATA_SUFFIX}") if path not in paths)
    return sorted(paths)
