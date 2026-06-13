# Project Structure

A P64 project is a folder with a `project.p64` manifest and generated support
folders.

```text
MyGame/
  project.p64
  assets/
    scenes/
      main.scenep64
    scripts/
      player.py
    shaders/
      custom_textured.shader
    materials/
      model/
        Mat.material
    model.obj
    model.obj.mdp64
  packages/
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
- `.material` files store editable runtime material data.
- `.mdp64` files store generated asset metadata and editor-only sidecar data.
- `.shader` files store custom shader source.

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
support files if they are recognizable generated files. User-authored files under
`assets/` are not overwritten by this refresh.

## Metadata Sidecars

`.mdp64` files are hidden editor/engine sidecars. OBJ sidecars store imported
groups, source material names, MTL defaults, and material extraction mappings.
Material sidecars such as `Mat.material.mdp64` store reset defaults, source links,
and usage cache information. Scenes and `.material` files remain the authoritative
runtime data.

## Legacy Files

Legacy project, scene, and metadata files still load. Use migration when you want
to rewrite them to current extensions:

```powershell
python -m p64 migrate samples\FirstScene
```
