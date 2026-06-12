# P64

P64 is a Python-based N64-style game engine prototype. It focuses on small OBJ-first
3D games, a native PySide6 editor, a ModernGL renderer, Python component scripts,
custom `.shader` files, and Windows desktop builds through PyInstaller.

## Current Capabilities

- P64 Hub project manager for creating, adding, opening, removing, and deleting projects.
- Native project folders with `assets/`, `packages/`, and `build/`.
- Project manifests use `project.p64`.
- Scene files use `.scenep64`.
- Generated asset metadata files use `.mdp64`.
- Scenes and scripts live under `assets/scenes/` and `assets/scripts/`.
- Builtin engine assets live under `packages/P64Builtin/`.
- Scene graph with entities, parenting, transforms, cameras, lights, fog volumes,
  mesh renderers, and Python script components.
- OBJ/MTL import with groups, materials, UVs, normals, texture metadata, and
  submesh hierarchy support.
- Scene/Game viewport tabs, scene camera navigation, hierarchy selection, shared
  entity/asset inspector, asset tree, and console.
- Project Settings window for startup scene, render settings, build settings, and
  Scene view grid settings.
- Camera-centered Scene grid with distance fade and subtle world axes.
- Multi-scene editing by double-clicking `.scenep64` files in the asset browser.
- N64-style rendering path with low-resolution upscaling, nearest texture sampling,
  fog, simple lighting, and shader selection.
- Runtime bundle creation and optional PyInstaller executable builds.
- Sample project in `samples/FirstScene`.

## Install For Development

```powershell
python -m pip install -e .[dev]
```

If Windows installs scripts outside your `PATH`, use `python -m p64 ...` instead
of `p64 ...`.

## Quick Start

Open the Hub:

```powershell
python -m p64 hub
```

Open a project directly:

```powershell
python -m p64 hub samples\FirstScene\project.p64
python -m p64 editor samples\FirstScene
```

Run or validate the sample project:

```powershell
python -m p64 run samples\FirstScene
python -m p64 validate samples\FirstScene
```

The batch launcher opens the Hub by default:

```powershell
p64.bat
p64.bat editor samples\FirstScene
p64.bat run samples\FirstScene
```

## Project Layout

```text
MyGame/
  project.p64
  assets/
    scenes/
      main.scenep64
    scripts/
      spin.py
    model.obj
    model.obj.mdp64
    shaders/
      n64_textured.shader
  packages/
    P64Builtin/
      shaders/
        standard_n64.shader
  build/
```

Generated P64 files are JSON internally, but use P64-native extensions:

- `project.p64` is the project manifest.
- `.scenep64` files are editable scenes.
- `.mdp64` files are generated asset metadata.

Legacy files still load and can be migrated:

```powershell
python -m p64 migrate samples\FirstScene
```

## Scenes And Scripts

Open a scene in the editor by double-clicking a `.scenep64` file in the asset
browser. If the current scene has unsaved changes, the editor asks whether to
save, discard, or cancel before switching.

Scripts can request scene changes:

```python
from p64.engine.scripting import UserScript


class ChangeScene(UserScript):
    def on_update(self, dt):
        self.scene_manager.load_scene_by_name("main")
```

Objects can persist across scene switches:

```python
class Player(UserScript):
    def on_start(self):
        self.persistent()
```

## Hub Build

Build the portable Hub app:

```powershell
python -m p64 build-hub
```

Output:

```text
build/app/P64/P64Hub.exe
build/app/P64/_internal/
```

Important: `P64Hub.exe` must stay beside the `_internal` folder. If you copy only
the `.exe`, Windows/PyInstaller can fail with `Failed to load Python DLL`.
Move or zip the whole `build/app/P64/` folder.

The Hub can be associated with `.p64` files on Windows. In the Hub, use the
`File Association` button to copy the command for the current executable.

## Game Builds

Create a fast runtime bundle:

```powershell
python -m p64 bundle samples\FirstScene
```

Build a game executable:

```powershell
python -m p64 build samples\FirstScene
```

Outputs:

```text
samples/FirstScene/build/bundle/
samples/FirstScene/build/game/FirstScene/
```

As with the Hub, keep the built game executable together with its generated
support files/folders.

Build behavior can be changed in the editor through `Project Settings`, including
the executable name, relative build output folder, and windowed/console mode.

## Tests

```powershell
$env:PYTHONPATH="src"
python -m unittest discover tests
python -m p64 validate samples\FirstScene
```
