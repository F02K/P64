# P64

P64 is a Python-based N64-style game engine prototype. It focuses on small OBJ-first
3D games, a native PySide6 editor, a ModernGL renderer, Python component scripts,
custom `.shader` files, and Windows desktop builds through PyInstaller.

## Current Capabilities

- P64 Hub project manager for creating, adding, opening, removing, and deleting projects.
- Native project folders with `assets/`, `packages/`, and `build/`.
- Automatic VSCode setup for new projects, plus manual refresh through the editor
  Project menu or `python -m p64 vscode <project>`.
- Project manifests use `project.p64`.
- Scene files use `.scenep64`.
- Generated asset metadata files use `.mdp64`.
- Runtime material files use `.material`; editor-only material defaults and usage
  caches live in hidden `.material.mdp64` sidecars.
- Scenes and scripts live under `assets/scenes/` and `assets/scripts/`.
- New gameplay scripts inherit from `GameScript`; generated project constants for
  VSCode/Pylance live in `packages/P64Generated/python/p64_project_api.py`.
- Builtin engine assets live under `packages/P64Builtin/`.
- Generated builtin shaders and build-runtime support files are refreshed when a
  project is opened, so older projects pick up engine updates automatically.
- Scene graph with entities, parenting, transforms, cameras, lights, scene-global fog,
  mesh renderers, and Python script components.
- OBJ/MTL model import with internal mesh entries, Source Materials, UVs,
  normals, optional vertex colors, texture metadata, diffuse material tinting,
  material extraction, and reusable `.mdp64` bounds/wireframe data for previews
  and gizmos.
- Automatic WAV AudioClip import with generated mono, 16-bit, max-22050 Hz
  runtime audio under `packages/P64Generated/audio/`, plus AudioSource playback
  and simple AudioListener-based spatial panning.
- Scene/Game viewport tabs, scene camera navigation, hierarchy selection, shared
  entity/asset inspector, asset tree, and console.
- Asset browser file operations for user-owned `assets/`: create folders, create
  blank files, rename files/folders inline, and delete assets with confirmation.
  `packages/` stays read-only in the editor.
- Project Settings window for startup scene, render settings, build settings, and
  Scene view grid settings.
- Camera-centered Scene grid with distance fade and subtle world axes.
- Multi-scene editing by double-clicking `.scenep64` files in the asset browser.
- Scene-coupled `.lightingp64` assets for sky, clouds, and global fog.
- Interactive Canvas UI with Button, Toggle, Slider, ScrollView, mouse and controller navigation.
- Quaternion-based transform hierarchies and collider raycasts for gameplay scripts.
- N64-inspired rendering path with low-resolution upscaling, nearest/three-point
  texture sampling, vertex lighting, material and vertex color tinting,
  quantization, optional dithering, fog, material shader selection, shader
  properties, and selection outlines.
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
```

Opening a project through the Hub creates or refreshes that project's `.venv`
and starts the editor with the project Python environment. Direct `editor` and
`run` CLI commands use the same project environment fallback.

Run or validate the sample project:

```powershell
python -m p64 run samples\FirstScene
python -m p64 validate samples\FirstScene
```

Refresh VSCode support files for an existing project:

```powershell
python -m p64 vscode samples\FirstScene
```

## Documentation

Start with the [documentation index](docs/README.md) for setup, editor workflow,
project structure, rendering, and builds. For gameplay code, see the
[scripting guide](docs/scripting.md), which includes copy-pasteable examples for
input, movement, physics, scene switching, and persistent objects.

## Project Layout

```text
MyGame/
  .vscode/
    settings.json
    tasks.json
    extensions.json
    launch.json
  project.p64
  assets/
    scenes/
      main.scenep64
    scripts/
      spin.py
    model.obj
    model.obj.mdp64
    beep.wav
    beep.wav.mdp64
    materials/
      model/
        Mat.material
    shaders/
      custom_textured.shader
  packages/
    P64Generated/
      python/
        p64_project_api.py
      audio/
        audio_beep.wav
    P64Builtin/
      shaders/
        standard_vertex_lit.shader
        standard_unlit.shader
  libraries/
    P64Build/
  build/
```

Generated P64 files are JSON internally, but use P64-native extensions:

- `project.p64` is the project manifest.
- `.scenep64` files are editable scenes.
- `.material` files are editable runtime materials.
- `.mdp64` files are generated metadata and hidden editor/engine sidecars.
- `.vscode/` and `packages/P64Generated/python/p64_project_api.py` are generated
  VSCode support files and can be refreshed by P64.
- `packages/P64Generated/audio/` contains generated mono WAV files for imported
  AudioClips.

The editor treats `assets/` as the editable project content area. Builtin package
files under `packages/P64Builtin/` are generated engine-owned support files and
are refreshed on project load when they are recognizable builtins.

OBJ files are Model assets. Their `.mdp64` sidecars store imported mesh entries,
per-mesh bounds, material slots, stats, and wireframe data used by previews and
gizmos. Source Materials render with MTL defaults and the standard VertexLit
shader until you extract them into `.material` assets. Extracted materials can
change shader, textures, and shader properties in the material asset inspector or
in the selected MeshRenderer's Materials foldout.

WAV files under `assets/` are automatically imported as AudioClip assets when the
editor refreshes assets or when validation/builds run. Their `.mdp64` sidecars
store original/imported sample rates, duration, sample count, and the generated
mono runtime WAV path. AudioSource components reference those clips, play them on
awake if enabled, and use AudioListener-relative distance and panning for 3D
spatial output. Add an active AudioListener component to the camera or another
scene entity to hear runtime audio.

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
from p64.engine.scripting import GameScript


class ChangeScene(GameScript):
    def on_update(self, dt):
        self.scene_manager.load_scene_by_name("main")
```

Objects can persist across scene switches:

```python
class Player(GameScript):
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

## Root App Build

Use the Windows build helper to open the Build Center for P64 tooling:

```powershell
.\build.bat
```

Useful targets:

```powershell
.\build.bat app
.\build.bat all --skip-tests
.\build.bat hub --skip-pyinstaller
```

`build.bat` builds the portable P64 App/Hub. Game builds are handled from the
editor/project build settings. Test targets use verbose output so the Build
Center log shows individual test names, elapsed time, and step progress.

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
