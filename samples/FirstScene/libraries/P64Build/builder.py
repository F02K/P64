from __future__ import annotations

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
