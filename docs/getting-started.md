# Getting Started

P64 is a Python-based N64-style game engine prototype with a PySide6 editor,
ModernGL rendering, Python scripts, OBJ-first asset import, and Windows desktop
builds.

## Install

From the repository root:

```powershell
python -m pip install -e .[dev]
```

If the `p64` script is not on your `PATH`, use `python -m p64` for commands.

## Open The Hub

```powershell
python -m p64 hub
```

The Hub can create new projects, add existing projects, open projects, and remove
or delete registered projects.

## Open The Sample Project

```powershell
python -m p64 editor samples\FirstScene
```

You can also open a project file through the Hub:

```powershell
python -m p64 hub samples\FirstScene\project.p64
```

## Run And Validate

Run the sample:

```powershell
python -m p64 run samples\FirstScene
```

Validate project references, scripts, build settings, and assets:

```powershell
python -m p64 validate samples\FirstScene
```

Refresh VSCode/Pylance support for an existing project:

```powershell
python -m p64 vscode samples\FirstScene
```

New projects receive the same setup automatically.

## Next Steps

- Use [editor.md](editor.md) to learn the editor workflow.
- Use [scripting.md](scripting.md) to attach gameplay code.
- Use [building.md](building.md) when you are ready to package a project.
