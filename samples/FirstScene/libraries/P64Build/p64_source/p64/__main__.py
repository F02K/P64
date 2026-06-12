from __future__ import annotations

import argparse
from pathlib import Path

from p64.build.pipeline import build_executable, build_hub_app, create_runtime_bundle, validate_project
from p64.engine.migration import migrate_project_files
from p64.engine.obj import import_obj_to_project
from p64.engine.project import Project
from p64.engine.runtime import run_project


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="p64")
    sub = parser.add_subparsers(dest="command", required=True)

    new_cmd = sub.add_parser("new", help="Create a new P64 project.")
    new_cmd.add_argument("path")
    new_cmd.add_argument("--name")

    editor_cmd = sub.add_parser("editor", help="Open the P64 editor.")
    editor_cmd.add_argument("project", nargs="?")

    hub_cmd = sub.add_parser("hub", help="Open the P64 Hub.")
    hub_cmd.add_argument("project", nargs="?")

    run_cmd = sub.add_parser("run", help="Run a P64 project.")
    run_cmd.add_argument("project")

    import_cmd = sub.add_parser("import-obj", help="Import an OBJ asset.")
    import_cmd.add_argument("project")
    import_cmd.add_argument("obj")
    import_cmd.add_argument("--scene", action="store_true", help="Add OBJ groups to the startup scene.")

    validate_cmd = sub.add_parser("validate", help="Validate a P64 project.")
    validate_cmd.add_argument("project")

    bundle_cmd = sub.add_parser("bundle", help="Create a runnable project bundle.")
    bundle_cmd.add_argument("project")
    bundle_cmd.add_argument("--out")

    build_cmd = sub.add_parser("build", help="Build a desktop executable with PyInstaller.")
    build_cmd.add_argument("project")
    build_cmd.add_argument("--out")
    build_cmd.add_argument("--skip-pyinstaller", action="store_true")

    hub_build_cmd = sub.add_parser("build-hub", help="Build the portable P64 Hub app with PyInstaller.")
    hub_build_cmd.add_argument("--out")
    hub_build_cmd.add_argument("--skip-pyinstaller", action="store_true")

    migrate_cmd = sub.add_parser("migrate", help="Migrate a project to native P64 file extensions.")
    migrate_cmd.add_argument("project")

    args = parser.parse_args(argv)

    if args.command == "new":
        project = Project.create(Path(args.path), name=args.name)
        print(f"Created project: {project.root}")
        return 0

    if args.command == "editor":
        from p64.editor.app import launch_editor

        launch_editor(Path(args.project) if args.project else None)
        return 0

    if args.command == "hub":
        from p64.hub.app import launch_hub

        launch_hub(Path(args.project) if args.project else None)
        return 0

    if args.command == "run":
        run_project(Path(args.project))
        return 0

    if args.command == "import-obj":
        project = Project.load(Path(args.project))
        metadata = import_obj_to_project(project, Path(args.obj), add_to_startup_scene=args.scene)
        print(f"Imported {metadata.source} as {metadata.id}")
        return 0

    if args.command == "validate":
        report = validate_project(Path(args.project))
        for warning in report.warnings:
            print(f"warning: {warning}")
        for error in report.errors:
            print(f"error: {error}")
        return 0 if report.ok else 1

    if args.command == "bundle":
        bundle = create_runtime_bundle(Path(args.project), Path(args.out) if args.out else None)
        print(f"Created bundle: {bundle}")
        return 0

    if args.command == "build":
        output = build_executable(
            Path(args.project),
            Path(args.out) if args.out else None,
            run_pyinstaller=not args.skip_pyinstaller,
        )
        print(f"Build output: {output}")
        return 0

    if args.command == "build-hub":
        output = build_hub_app(
            Path(args.out) if args.out else None,
            run_pyinstaller=not args.skip_pyinstaller,
        )
        print(f"Hub output: {output}")
        return 0

    if args.command == "migrate":
        changes = migrate_project_files(Path(args.project))
        for change in changes:
            print(change)
        print("Migration complete.")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
