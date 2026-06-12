from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from p64.engine.builtin import BUILTIN_PACKAGE_NAME, ensure_builtin_package
from p64.engine.components import Camera, Fog, Light
from p64.engine.entity import Entity
from p64.engine.files import DEFAULT_SCENE, LEGACY_DEFAULT_SCENE, PROJECT_FILE, alternate_scene_path, normalize_scene_path, project_file_for, project_root_from_path
from p64.engine.math import Vec3
from p64.engine.scene import Scene


@dataclass
class Project:
    root: Path
    name: str
    startup_scene: str = DEFAULT_SCENE
    render_settings: dict[str, Any] = field(default_factory=dict)
    build_settings: dict[str, Any] = field(default_factory=dict)
    editor_settings: dict[str, Any] = field(default_factory=dict)

    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def scenes_dir(self) -> Path:
        return self.assets_dir / "scenes"

    @property
    def scripts_dir(self) -> Path:
        return self.assets_dir / "scripts"

    @property
    def legacy_scenes_dir(self) -> Path:
        return self.root / "scenes"

    @property
    def legacy_scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def packages_dir(self) -> Path:
        return self.root / "packages"

    @property
    def builtin_package_dir(self) -> Path:
        return self.packages_dir / BUILTIN_PACKAGE_NAME

    @property
    def build_dir(self) -> Path:
        return self.root / "build"

    @property
    def libraries_dir(self) -> Path:
        return self.root / "libraries"

    @property
    def build_pipeline_dir(self) -> Path:
        return self.root / str(self.build_settings.get("build_pipeline_path", "libraries/P64Build"))

    @classmethod
    def create(cls, root: Path, name: str | None = None) -> "Project":
        root = project_root_from_path(root)
        project = cls(
            root=root,
            name=name or root.name,
            render_settings={
                "internal_resolution": [320, 240],
                "texture_filter": "three_point",
                "color_levels": 32,
                "dithering": True,
                "fog": True,
            },
            build_settings=default_build_settings(name or root.name),
            editor_settings=default_editor_settings(),
        )
        project.ensure_layout()
        scene = default_scene("main")
        scene.render_settings = dict(project.render_settings)
        scene.save(root / project.startup_scene)
        project.save()
        return project

    @classmethod
    def load(cls, root: Path) -> "Project":
        root = project_root_from_path(root)
        project_path = project_file_for(root)
        with project_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        project = cls(
            root=root,
            name=str(data.get("name", root.name)),
            startup_scene=str(data.get("startup_scene", DEFAULT_SCENE)),
            render_settings=dict(data.get("render_settings", {})),
            build_settings=dict(data.get("build_settings", {})),
            editor_settings=dict(data.get("editor_settings", {})),
        )
        project.apply_default_settings()
        return project

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(exist_ok=True)
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self.packages_dir.mkdir(exist_ok=True)
        ensure_builtin_package(self.root)
        self.libraries_dir.mkdir(exist_ok=True)
        ensure_project_build_pipeline(self)
        self.build_dir.mkdir(exist_ok=True)

    def save(self) -> None:
        self.ensure_layout()
        self.apply_default_settings()
        self.startup_scene = normalize_scene_path(self.startup_scene)
        with self.project_file.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "name": self.name,
                    "startup_scene": normalize_scene_path(self.startup_scene),
                    "render_settings": self.render_settings,
                    "build_settings": self.build_settings,
                    "editor_settings": self.editor_settings,
                },
                handle,
                indent=2,
            )
            handle.write("\n")

    def load_startup_scene(self) -> Scene:
        return Scene.load(self.resolve_scene_path(self.startup_scene))

    def save_startup_scene(self, scene: Scene) -> None:
        self.startup_scene = normalize_scene_path(self.startup_scene)
        scene.save(self.root / self.startup_scene)

    def resolve_scene_path(self, scene: str | Path) -> Path:
        scene_path = Path(scene)
        if scene_path.is_absolute() and scene_path.exists():
            return scene_path
        candidates = [self.root / scene_path, self.root / alternate_scene_path(scene_path)]
        if str(scene_path).startswith("assets/scenes/"):
            candidates.append(self.root / "scenes" / scene_path.name)
        elif str(scene_path).startswith("scenes/"):
            candidates.append(self.assets_dir / "scenes" / scene_path.name)
        if scene_path.name == "main.scenep64":
            candidates.append(self.root / LEGACY_DEFAULT_SCENE)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def scene_path_by_name(self, name: str) -> Path | None:
        wanted = Path(name).stem
        for folder in [self.scenes_dir, self.legacy_scenes_dir]:
            if not folder.exists():
                continue
            for path in sorted(folder.rglob("*.scenep64")):
                if path.stem == wanted:
                    return path
        return None

    def script_path_candidates(self, script: str | Path) -> list[Path]:
        script_path = Path(script)
        if not str(script).strip() or script_path.name == "":
            return []
        if script_path.suffix != ".py":
            script_path = script_path.with_suffix(".py")
        candidates = [self.scripts_dir / script_path]
        legacy = self.legacy_scripts_dir / script_path
        if legacy not in candidates:
            candidates.append(legacy)
        return candidates

    def apply_default_settings(self) -> None:
        self.render_settings = clamp_render_settings({**default_render_settings(), **self.render_settings})
        self.build_settings = clamp_build_settings({**default_build_settings(self.name), **self.build_settings})
        self.editor_settings = merge_editor_settings(self.editor_settings)


def default_scene(name: str) -> Scene:
    scene = Scene(name=name)

    camera = Entity("Camera")
    camera.transform.position = Vec3(0.0, 3.0, 8.0)
    camera.transform.rotation = Vec3(-18.0, 0.0, 0.0)
    camera.add_component(Camera(active=True))
    scene.add_entity(camera)

    sun = Entity("Sun")
    sun.transform.rotation = Vec3(-45.0, 35.0, 0.0)
    sun.add_component(Light(kind="directional", intensity=1.25))
    scene.add_entity(sun)

    fog = Entity("Fog")
    fog.add_component(Fog(near=18.0, far=85.0))
    scene.add_entity(fog)
    return scene


def default_render_settings() -> dict[str, Any]:
    return {
        "internal_resolution": [320, 240],
        "texture_filter": "three_point",
        "color_levels": 32,
        "dithering": True,
        "fog": True,
    }


def default_build_settings(project_name: str = "Game") -> dict[str, Any]:
    return {
        "executable_name": project_name,
        "output_folder": "build/game",
        "default_build_mode": "executable",
        "windowed": True,
        "icon_path": "",
        "python_executable": "",
        "build_pipeline_path": "libraries/P64Build",
        "auto_install_build_dependencies": True,
    }


def default_editor_settings() -> dict[str, Any]:
    return {
        "scene_grid": {
            "enabled": True,
            "spacing": 1.0,
            "radius": 40.0,
            "fade_start": 18.0,
            "fade_end": 40.0,
        }
    }


def clamp_render_settings(settings: dict[str, Any]) -> dict[str, Any]:
    resolution = list(settings.get("internal_resolution", [320, 240]))
    if len(resolution) < 2:
        resolution = [320, 240]
    settings["internal_resolution"] = [max(1, int(resolution[0])), max(1, int(resolution[1]))]
    settings["color_levels"] = max(2, int(settings.get("color_levels", 32)))
    filter_value = str(settings.get("texture_filter", "three_point"))
    settings["texture_filter"] = filter_value if filter_value in {"nearest", "linear", "three_point"} else "three_point"
    settings["dithering"] = bool(settings.get("dithering", True))
    settings["fog"] = bool(settings.get("fog", True))
    return settings


def clamp_build_settings(settings: dict[str, Any]) -> dict[str, Any]:
    settings["executable_name"] = str(settings.get("executable_name") or "Game")
    output = str(settings.get("output_folder") or "build/game").replace("\\", "/").strip("/")
    settings["output_folder"] = output or "build/game"
    mode = str(settings.get("default_build_mode", "executable"))
    settings["default_build_mode"] = mode if mode in {"bundle", "executable"} else "executable"
    settings["windowed"] = bool(settings.get("windowed", True))
    settings["icon_path"] = str(settings.get("icon_path") or "").replace("\\", "/")
    settings["python_executable"] = str(settings.get("python_executable") or "")
    pipeline = str(settings.get("build_pipeline_path") or "libraries/P64Build").replace("\\", "/").strip("/")
    settings["build_pipeline_path"] = pipeline or "libraries/P64Build"
    settings["auto_install_build_dependencies"] = bool(settings.get("auto_install_build_dependencies", True))
    return settings


def merge_editor_settings(settings: dict[str, Any]) -> dict[str, Any]:
    merged = default_editor_settings()
    scene_grid = {**merged["scene_grid"], **dict(settings.get("scene_grid", {}))}
    spacing = max(0.01, float(scene_grid.get("spacing", 1.0)))
    radius = max(spacing, float(scene_grid.get("radius", 40.0)))
    fade_start = max(0.0, float(scene_grid.get("fade_start", 18.0)))
    fade_end = max(fade_start + spacing, float(scene_grid.get("fade_end", 40.0)))
    merged["scene_grid"] = {
        "enabled": bool(scene_grid.get("enabled", True)),
        "spacing": spacing,
        "radius": radius,
        "fade_start": fade_start,
        "fade_end": fade_end,
    }
    for key, value in settings.items():
        if key != "scene_grid":
            merged[key] = value
    return merged


def ensure_project_build_pipeline(project: Project) -> None:
    pipeline_dir = project.build_pipeline_dir
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "builder.py").write_text(_builder_script_source(), encoding="utf-8")
    (pipeline_dir / "requirements-build.txt").write_text("pyinstaller>=6.0\npillow>=10.0\npygame>=2.5\n", encoding="utf-8")

    source = _source_p64_package_dir()
    destination = pipeline_dir / "p64_source" / "p64"
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=_ignore_copied_source)


def _source_p64_package_dir() -> Path:
    executable_dir = Path(sys.executable).resolve().parent
    bundle_root = Path(getattr(sys, "_MEIPASS", executable_dir))
    candidates = [
        bundle_root / "p64_source" / "p64",
        executable_dir / "_internal" / "p64_source" / "p64",
        executable_dir / "p64_source" / "p64",
        Path(__file__).resolve().parents[1],
    ]
    for candidate in candidates:
        if (candidate / "engine" / "runtime.py").exists():
            return candidate
    return candidates[-1]


def _ignore_copied_source(_directory: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__"}
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo")))
    return ignored


def _builder_script_source() -> str:
    return r'''from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="P64Build")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--distpath", required=True)
    parser.add_argument("--workpath", required=True)
    parser.add_argument("--specpath", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--p64-source", required=True)
    parser.add_argument("--env-dir", required=True)
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--project-package", required=True)
    parser.add_argument("--icon")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--auto-install", action="store_true")
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args()

    if _needs_bootstrap(args):
        return _bootstrap_and_rerun(args)

    icon = _prepare_icon(Path(args.icon), Path(args.workpath)) if args.icon else None
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--name",
        args.name,
        "--paths",
        str(Path(args.p64_source)),
        "--distpath",
        str(Path(args.distpath)),
        "--workpath",
        str(Path(args.workpath)),
        "--specpath",
        str(Path(args.specpath)),
        "--add-data",
        f"{Path(args.project_package)}{os.pathsep}.",
    ]
    if args.windowed:
        command.append("--windowed")
    if icon:
        command.extend(["--icon", str(icon)])
    command.append(str(Path(args.bundle) / "run_game.py"))

    result = subprocess.run(command, cwd=args.bundle)
    return result.returncode


def _needs_bootstrap(args: argparse.Namespace) -> bool:
    if args.no_bootstrap:
        return False
    try:
        import PyInstaller  # noqa: F401
        if args.icon and Path(args.icon).suffix.lower() == ".png":
            import PIL  # noqa: F401
        return False
    except ImportError:
        return True


def _bootstrap_and_rerun(args: argparse.Namespace) -> int:
    if not args.auto_install:
        print("PyInstaller is not available. Enable auto install or install requirements-build.txt with this Python.")
        return 2

    env_dir = Path(args.env_dir)
    python = _venv_python(env_dir)
    if not python.exists():
        print(f"Creating build environment: {env_dir}")
        venv.EnvBuilder(with_pip=True, clear=False).create(env_dir)

    requirements = Path(args.requirements)
    install = [str(python), "-m", "pip", "install", "--upgrade", "-r", str(requirements)]
    print("Installing build dependencies...")
    result = subprocess.run(install)
    if result.returncode != 0:
        print(f"Could not install build dependencies. Run manually: {' '.join(install)}")
        return result.returncode

    rerun = [str(python), __file__, *sys.argv[1:], "--no-bootstrap"]
    return subprocess.run(rerun).returncode


def _venv_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _prepare_icon(icon_path: Path, work_path: Path) -> Path:
    icon_path = icon_path.resolve()
    if icon_path.suffix.lower() == ".ico":
        return icon_path
    if icon_path.suffix.lower() != ".png":
        raise SystemExit(f"Unsupported icon format: {icon_path}")

    from PIL import Image

    work_path.mkdir(parents=True, exist_ok=True)
    output = work_path / "p64_icon.ico"
    image = Image.open(icon_path)
    image.save(output, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])
    return output


if __name__ == "__main__":
    raise SystemExit(main())
'''
