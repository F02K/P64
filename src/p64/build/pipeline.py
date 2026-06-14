from __future__ import annotations

import shutil
import subprocess
import sys
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from p64.engine.components import MeshRenderer, ScriptComponent
from p64.engine.audio import ensure_audio_clips_for_assets
from p64.engine.assets import AssetMetadata, discover_metadata
from p64.engine.files import PROJECT_FILE, alternate_scene_path, is_scene_file
from p64.engine.material import MaterialAsset, load_material_metadata, resolve_material_reference
from p64.engine.project import Project
from p64.engine.validation import asset_metadata_by_id, scene_reference_errors


PROJECT_PACKAGE_FILE = "p64_project.p64pack"


@dataclass
class BuildReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_project(project_root: Path) -> BuildReport:
    report = BuildReport()
    try:
        project = Project.load(project_root)
    except Exception as exc:
        report.errors.append(f"Could not load project: {exc}")
        return report
    try:
        ensure_audio_clips_for_assets(project)
    except Exception as exc:
        report.errors.append(f"Could not import audio clips: {exc}")

    scene_path = project.root / project.startup_scene
    if not is_scene_file(scene_path):
        report.errors.append(f"Startup scene must be a .scenep64 file: {project.startup_scene}")
    if not scene_path.exists():
        alternate = project.root / alternate_scene_path(Path(project.startup_scene))
        if alternate.exists():
            scene_path = alternate
        else:
            report.errors.append(f"Startup scene missing: {project.startup_scene}")
    if scene_path.exists():
        try:
            scene = project.load_startup_scene()
        except Exception as exc:
            report.errors.append(f"Could not load startup scene: {exc}")
        else:
            if scene.active_camera() is None:
                report.warnings.append("Scene has no camera.")
            metadata_by_id = asset_metadata_by_id(project)
            reference_errors = scene_reference_errors(project, scene)
            for entity in scene.walk():
                for error in reference_errors.get(entity.id, []):
                    report.errors.append(f"{entity.name} reference error: {error}")
                for component in entity.components:
                    if isinstance(component, MeshRenderer):
                        metadata = metadata_by_id.get(component.mesh)
                        if metadata:
                            texture = _diffuse_texture_path(project, metadata, component.material)
                            if texture and not texture.exists():
                                report.errors.append(f"{entity.name} references missing texture: {texture}")
                        for material in component.material_slots:
                            material_path = resolve_material_reference(project.root, material)
                            if material_path and material_path.is_absolute():
                                try:
                                    material_path.resolve().relative_to(project.assets_dir.resolve())
                                except ValueError:
                                    report.warnings.append(f"{entity.name} references external material: {material_path}")
                            texture = _material_texture_path(project, material)
                            if texture and not texture.exists():
                                report.errors.append(f"{entity.name} references missing material texture: {texture}")
    output_folder = Path(str(project.build_settings.get("output_folder", "build/game")))
    if output_folder.is_absolute():
        report.errors.append("Build output folder must be relative to the project.")
    else:
        resolved_output = (project.root / output_folder).resolve()
        try:
            resolved_output.relative_to(project.build_dir.resolve())
        except ValueError:
            report.errors.append("Build output folder must stay inside the project build folder.")

    for metadata_path in discover_metadata(project.assets_dir):
        try:
            metadata = AssetMetadata.load(metadata_path)
        except Exception as exc:
            report.errors.append(f"Invalid asset metadata {metadata_path.name}: {exc}")
            continue
        if metadata.source and not (project.root / metadata.source).exists():
            report.errors.append(f"Asset source missing for {metadata.id}: {metadata.source}")

    for script_path in _script_files(project):
        try:
            compile(script_path.read_text(encoding="utf-8"), str(script_path), "exec")
        except SyntaxError as exc:
            line = f":{exc.lineno}" if exc.lineno else ""
            report.errors.append(f"Script syntax error {script_path.relative_to(project.root).as_posix()}{line}: {exc.msg}")
        except UnicodeDecodeError as exc:
            report.errors.append(f"Script encoding error {script_path.relative_to(project.root).as_posix()}: {exc}")

    return report


def _script_files(project: Project) -> list[Path]:
    scripts: list[Path] = []
    for folder in [project.scripts_dir, project.root / "scripts"]:
        if not folder.exists():
            continue
        scripts.extend(path for path in folder.rglob("*.py") if path.is_file())
    return sorted({path.resolve(): path for path in scripts}.values())


def _asset_ids(project: Project) -> set[str]:
    return set(_asset_metadata(project))


def _asset_metadata(project: Project) -> dict[str, AssetMetadata]:
    assets: dict[str, AssetMetadata] = {}
    for metadata_path in discover_metadata(project.assets_dir):
        try:
            metadata = AssetMetadata.load(metadata_path)
            assets[metadata.id] = metadata
        except Exception:
            continue
    return assets


def _diffuse_texture_path(project: Project, metadata: AssetMetadata, material: str | None) -> Path | None:
    if not material:
        return None
    material_defs = metadata.settings.get("material_defs", {})
    texture_name = material_defs.get(material, {}).get("diffuse_texture")
    if not texture_name:
        return None
    return (project.root / metadata.source).parent / str(texture_name)


def _material_texture_path(project: Project, material: str | None) -> Path | None:
    if not material:
        return None
    material_path = resolve_material_reference(project.root, material)
    if material_path is None:
        return None
    if not material_path.exists():
        return None
    try:
        asset = MaterialAsset.load(material_path)
    except Exception:
        return None
    texture_name = asset.textures.get("u_texture")
    if not texture_name:
        return None
    texture_path = Path(str(texture_name))
    if texture_path.is_absolute():
        return texture_path
    candidates: list[Path] = []
    if str(texture_name).startswith(("assets/", "packages/")):
        candidates.append(project.root / texture_name)
    candidates.append(material_path.parent / texture_name)
    metadata = load_material_metadata(material_path)
    source = metadata.settings.get("source", {}) if metadata else {}
    if isinstance(source, dict) and source.get("obj"):
        candidates.append((project.root / str(source["obj"])).parent / texture_name)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0] if candidates else None)


def create_runtime_bundle(project_root: Path, output_dir: Path | None = None) -> Path:
    report = validate_project(project_root)
    if not report.ok:
        raise RuntimeError("Project validation failed:\n" + "\n".join(report.errors))

    project = Project.load(project_root)
    output_dir = output_dir or (project.build_dir / "bundle")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for name in ["assets", "packages"]:
        source = project.root / name
        destination = output_dir / name
        if source.is_dir():
            shutil.copytree(source, destination)
        elif source.exists():
            shutil.copy2(source, destination)
    for name in ["scenes", "scripts"]:
        source = project.root / name
        if source.exists() and any(path.is_file() for path in source.rglob("*")):
            shutil.copytree(source, output_dir / name)

    (output_dir / PROJECT_FILE).write_text(
        json.dumps(
            {
                "name": project.name,
                "startup_scene": project.startup_scene,
                "render_settings": project.render_settings,
                "build_settings": project.build_settings,
                "editor_settings": project.editor_settings,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    launcher = output_dir / "run_game.py"
    launcher.write_text(
        _runtime_launcher_source(),
        encoding="utf-8",
    )
    return output_dir


def create_runtime_package(bundle_dir: Path, output_file: Path | None = None) -> Path:
    bundle_dir = bundle_dir.resolve()
    output_file = (output_file or (bundle_dir / PROJECT_PACKAGE_FILE)).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    allowed_roots = [PROJECT_FILE, "assets", "packages", "scenes", "scripts"]
    with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in allowed_roots:
            source = bundle_dir / name
            if source.is_file():
                archive.write(source, name)
            elif source.is_dir():
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(bundle_dir).as_posix())
    return output_file


def _runtime_launcher_source() -> str:
    return (
        "from pathlib import Path\n"
        "import sys\n"
        "import tempfile\n"
        "import zipfile\n\n"
        "from p64.engine.runtime import run_project\n\n"
        f"PROJECT_PACKAGE = {PROJECT_PACKAGE_FILE!r}\n"
        "_RUNTIME_TEMP = None\n\n"
        "def _project_root() -> Path:\n"
        "    if getattr(sys, 'frozen', False):\n"
        "        base = Path(getattr(sys, '_MEIPASS', Path(sys.executable).resolve().parent))\n"
        "        package = base / PROJECT_PACKAGE\n"
        "        if not package.exists():\n"
        "            raise FileNotFoundError(f'Missing bundled P64 project package: {package}')\n"
        "        temp = tempfile.TemporaryDirectory(prefix='p64_game_')\n"
        "        root = Path(temp.name)\n"
        "        with zipfile.ZipFile(package) as archive:\n"
        "            archive.extractall(root)\n"
        "        global _RUNTIME_TEMP\n"
        "        _RUNTIME_TEMP = temp\n"
        "        return root\n"
        "    return Path(__file__).resolve().parent\n\n"
        "run_project(_project_root())\n"
    )


def _is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def _p64_source_path() -> Path:
    if not _is_frozen_app():
        return Path(__file__).resolve().parents[2]

    executable_dir = Path(sys.executable).resolve().parent
    bundle_root = Path(getattr(sys, "_MEIPASS", executable_dir))
    candidates = [
        bundle_root / "p64_source",
        executable_dir / "_internal" / "p64_source",
        executable_dir / "p64_source",
    ]
    for candidate in candidates:
        if (candidate / "p64").is_dir():
            return candidate
    return candidates[0]


def _pyinstaller_data_arg(source: Path, destination: str) -> str:
    return f"{source}{os.pathsep}{destination}"


def _run_pyinstaller(args: list[str], cwd: Path | None = None) -> None:
    if _is_frozen_app():
        raise RuntimeError("Hub builds require a source Python environment. Game builds use the project BuildPipeline instead.")

    command = [sys.executable, "-m", "PyInstaller", *args]
    subprocess.run(command, cwd=cwd, check=True)


def _python_command_candidates(project: Project) -> list[list[str]]:
    candidates: list[list[str]] = []
    configured = str(project.build_settings.get("python_executable") or "").strip()
    if configured:
        candidates.append([configured])
    env_python = os.environ.get("P64_PYTHON", "").strip()
    if env_python:
        candidates.append([env_python])
    if shutil.which("py"):
        candidates.append(["py", "-3"])
    if shutil.which("python"):
        candidates.append(["python"])
    return candidates


def _resolve_build_python(project: Project) -> list[str]:
    for candidate in _python_command_candidates(project):
        executable = candidate[0]
        if Path(executable).exists() or shutil.which(executable):
            return candidate
    raise RuntimeError(
        "No Python executable found for the build. Set Python Executable in Build Settings, "
        "set P64_PYTHON, or install Python and make py/python available on PATH."
    )


def describe_build_python(project: Project) -> str:
    try:
        return " ".join(_resolve_build_python(project))
    except RuntimeError as exc:
        return str(exc)


def _resolve_project_path(project: Project, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project.root / path


def _run_project_builder(project: Project, bundle: Path, project_package: Path, dist_root: Path, work_path: Path, spec_path: Path) -> None:
    project.ensure_layout()
    pipeline_dir = project.build_pipeline_dir
    builder = pipeline_dir / "builder.py"
    if not builder.exists():
        raise RuntimeError(f"Missing project BuildPipeline: {builder}")

    python = _resolve_build_python(project)
    args = [
        *python,
        str(builder),
        "--bundle",
        str(bundle),
        "--distpath",
        str(dist_root),
        "--workpath",
        str(work_path),
        "--specpath",
        str(spec_path),
        "--name",
        str(project.build_settings.get("executable_name") or project.name),
        "--p64-source",
        str(pipeline_dir / "p64_source"),
        "--env-dir",
        str(project.build_dir / "p64-build-env"),
        "--requirements",
        str(pipeline_dir / "requirements-build.txt"),
        "--project-package",
        str(project_package),
    ]
    if project.build_settings.get("windowed", True):
        args.append("--windowed")
    if project.build_settings.get("auto_install_build_dependencies", True):
        args.append("--auto-install")
    icon = str(project.build_settings.get("icon_path") or "").strip()
    if icon:
        icon_path = _resolve_project_path(project, icon)
        if not icon_path.exists():
            raise RuntimeError(f"Build icon does not exist: {icon_path}")
        args.extend(["--icon", str(icon_path)])

    subprocess.run(args, cwd=project.root, check=True)


def build_executable(project_root: Path, output_dir: Path | None = None, run_pyinstaller: bool = True) -> Path:
    project = Project.load(project_root)
    bundle = create_runtime_bundle(project.root)
    if not run_pyinstaller:
        return bundle

    dist_root = (output_dir or (project.root / str(project.build_settings.get("output_folder", "build/game")))).resolve()
    work_path = (project.build_dir / "pyinstaller-work").resolve()
    spec_path = (project.build_dir / "pyinstaller-spec").resolve()
    dist_root.mkdir(parents=True, exist_ok=True)
    project_package = create_runtime_package(bundle)

    _run_project_builder(project, bundle, project_package, dist_root, work_path, spec_path)
    return dist_root / str(project.build_settings.get("executable_name") or project.name)


def build_hub_app(output_dir: Path | None = None, run_pyinstaller: bool = True) -> Path:
    output_root = (output_dir or Path("build/app/P64")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not run_pyinstaller:
        return output_root

    dist_parent = output_root.parent
    work_path = Path("build/app/pyinstaller-work").resolve()
    spec_path = Path("build/app/pyinstaller-spec").resolve()
    launcher = Path("build/app/hub_launcher.py").resolve()
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(
        "import sys\n"
        "from p64.__main__ import main\n"
        "args = sys.argv[1:]\n"
        "if args[:1] == ['--editor']:\n"
        "    raise SystemExit(main(['editor', *args[1:]]))\n"
        "raise SystemExit(main(['hub', *args]))\n",
        encoding="utf-8",
    )
    args = [
        "--noconfirm",
        "--windowed",
        "--name",
        "P64Hub",
        "--add-data",
        _pyinstaller_data_arg(Path("src/p64").resolve(), "p64_source/p64"),
        "--distpath",
        str(dist_parent),
        "--workpath",
        str(work_path),
        "--specpath",
        str(spec_path),
        str(launcher),
    ]
    _run_pyinstaller(args)
    built = dist_parent / "P64Hub"
    if built.exists() and built.resolve() != output_root:
        if output_root.exists():
            shutil.rmtree(output_root)
        shutil.move(str(built), str(output_root))
    return output_root
