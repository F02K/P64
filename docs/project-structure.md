# Project Structure

A P64 project is a folder with a `project.p64` manifest and generated support
folders.

```text
MyGame/
  .venv/
  .vscode/
    settings.json
    tasks.json
    extensions.json
    launch.json
  project.p64
  assets/
    scenes/
      main.scenep64
      main.lightingp64
    scripts/
      player.py
    shaders/
      custom_textured.shader
    materials/
      model/
        Mat.material
    model.obj
    model.obj.mdp64
    beep.wav
    beep.wav.mdp64
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

## Important Files

- `project.p64` stores project settings, render settings, build settings, and the
  startup scene path.
- `.scenep64` files store editable scenes.
- `.lightingp64` files store the coupled sky, cloud, and global fog settings for
  the Scene with the same filename stem.
- `.material` files store editable runtime material data.
- `.mdp64` files store generated asset metadata and editor-only sidecar data.
- `.shader` files store custom shader source.
- `.venv/` stores the project Python environment used by the Hub, editor, and
  runtime commands. It is generated and should not be committed.
- `.vscode/` stores generated VSCode workspace support files.

## Editable Content

Put user-authored scenes, scripts, shaders, models, textures, and other files
under `assets/`. The editor allows creating, renaming, and deleting files only
inside this folder.

Material assets created inside `assets/` appear in the asset browser and can be
assigned to MeshRenderer material slots. Material files may also live outside
`assets/` when selected during extraction, but those external files are stored as
absolute references and are not managed by the asset browser.

## Generated Content

`packages/P64Builtin/` contains engine-owned builtin assets. `libraries/P64Build/`
contains build pipeline support and a copied runtime/editor source tree used by
desktop builds.

When a project is opened, P64 refreshes generated builtin shaders and build
support files if they are recognizable generated files. New projects also receive
VSCode support files and `packages/P64Generated/python/p64_project_api.py`.
The Hub also creates or refreshes `.venv/` so editor/runtime dependencies such as
PySide6, ModernGL, pygame, NumPy, and Pillow are available for that project.
Imported WAV runtime copies are generated automatically under
`packages/P64Generated/audio/` when the editor refreshes assets or
validation/builds run.
Re-run `Project > Setup VSCode` or `python -m p64 vscode <project>` to refresh
those files for an existing project. User-authored files under `assets/` are not
overwritten by this refresh.

## Metadata Sidecars

`.mdp64` files are hidden editor/engine sidecars. OBJ sidecars describe the OBJ
as a Model asset: imported mesh entries, source material names, per-mesh bounds,
triangle/vertex stats, wireframe edge data for previews and gizmos, MTL defaults,
and material extraction mappings. Material sidecars such as
`Mat.material.mdp64` store reset defaults, source links, and usage cache
information. WAV sidecars describe AudioClip import settings, original and
runtime sample rates, duration, sample count, and the generated mono WAV path.
Scenes, their coupled `.lightingp64` assets, and `.material` files remain the
authoritative runtime data. Scene and Lighting assets are created, renamed,
duplicated, moved, and deleted together by the editor.

## Legacy Files

Legacy project, scene, and metadata files still load. Use migration when you want
to rewrite them to current extensions:

```powershell
python -m p64 migrate samples\FirstScene
```
