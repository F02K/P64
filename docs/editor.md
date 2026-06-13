# Editor

The P64 editor is split into a scene hierarchy, Scene/Game viewport tabs, an
inspector, an asset browser, and a console.

## Scene Hierarchy

The hierarchy shows scene objects and their children. Use it to create entities,
select objects, duplicate objects, rename objects, and delete objects. Selecting
an entity shows its components in the inspector.

## Inspector

The inspector edits the selected entity or asset. Entity inspectors expose
transforms, object type, active/persistent flags, and component-specific fields.
Asset inspectors show file metadata, previews for images, shader summaries, and
context actions such as open, reveal, import, and material extraction.

## Asset Browser

The asset browser shows two project roots:

- `assets/` contains user-owned project content and is editable in the editor.
- `packages/` contains packages such as `P64Builtin` and is read-only in the
  editor.

Inside `assets/`, the asset browser can:

- create folders
- create blank files
- rename files and folders inline
- delete files and folders after confirmation
- create specialized scenes, shaders, and scripts
- import OBJ assets into the current scene
- open scene, shader, and Python files

Metadata sidecars (`.mdp64`) are internal editor files and are hidden from the
asset browser.

When an imported asset source is renamed or deleted, its `.mdp64` metadata sidecar
is moved or deleted with it. Folder renames also update nested metadata `source`
paths.

The editor blocks deletion of the currently open scene and the configured startup
scene. Open or choose another scene first.

## Materials

OBJ assets can expose Source Materials from their referenced `.mtl` files. These
source entries are read-only import information. To edit shader or property
values, select the OBJ asset and use `Extract Materials`.

`Extract Materials` asks for a target folder. Folders inside `assets/` are shown
in the asset browser. Folders outside `assets/` are allowed after a warning, but
those external materials are referenced by absolute path and are not managed by
the asset browser.

MeshRenderer inspectors show Source Materials as information and show editable
P64 material slots in a `Materials` foldout near the bottom of the inspector. If
a slot has no `.material` asset, it renders with MTL defaults and the standard
VertexLit shader. If a slot references a `.material`, its shader and properties
can be edited there.

Selecting a `.material` asset also shows the same shader, texture, and property
fields. Reset restores values from the hidden `.material.mdp64` defaults.

## Scene And Game Tabs

The Scene tab is for editing. The Game tab is for play mode. Play mode runs a
runtime copy of the scene, so script changes to transforms and physics do not
mutate the editor scene until you explicitly save editor-side changes.

## Useful Shortcuts

- `Ctrl+S`: save the current scene
- `Delete`: delete the selected entity
- `Ctrl+D`: duplicate the selected entity
- `F2`: rename the selected entity
- `F`: frame the selected entity in the Scene view
