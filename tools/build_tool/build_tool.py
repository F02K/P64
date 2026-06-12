from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parents[2]
    project = Path(args[0]).resolve() if args else root / "samples" / "FirstScene"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    steps = [
        ("Running tests", [sys.executable, "-m", "unittest", "discover", "tests"]),
        ("Validating project", [sys.executable, "-m", "p64", "validate", str(project)]),
        ("Building P64 Hub", [sys.executable, "-m", "p64", "build-hub"]),
        ("Building game executable", [sys.executable, "-m", "p64", "build", str(project)]),
    ]

    exit_code = 0
    print("P64 Build Tool")
    print()
    print("Root:")
    print(f"  {root}")
    print("Project:")
    print(f"  {project}")
    print()

    for index, (title, command) in enumerate(steps, start=1):
        print(f"[{index}/{len(steps)}] {title}...")
        result = subprocess.run(command, cwd=root, env=env)
        if result.returncode != 0:
            exit_code = result.returncode
            print()
            print(f"Build failed with exit code {exit_code}.")
            break
        print()

    if exit_code == 0:
        print("Build complete.")
        print()
        print("Hub:")
        print(f"  {root / 'build' / 'app' / 'P64' / 'P64Hub.exe'}")
        print()
        print("Game:")
        print(f"  {project / 'build' / 'game'}")
        print()
        print("Keep each .exe together with its generated support folders.")

    if sys.stdin.isatty() and os.environ.get("P64_BUILD_TOOL_NO_PAUSE") != "1":
        print()
        input("Press Enter to close this build window...")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
