# Rendering

P64 uses a ModernGL renderer with builtin shaders and optional custom `.shader`
assets. Runtime materials are stored in `.material` files.

## Builtin Shaders

The builtin package provides:

- `packages/P64Builtin/shaders/standard_vertex_lit.shader`
- `packages/P64Builtin/shaders/standard_unlit.shader`

Generated builtin shaders are refreshed on project open when they are
recognizable generated files.

Both builtin shaders declare material-editable properties:

- `u_texture`: main texture slot
- `u_base_color`: material tint, initialized from MTL `Kd` when available
- `u_alpha_cutoff`: alpha threshold for texture cutout; `0.0` keeps existing opaque/alpha output behavior

## Lit Rendering

The standard Lit shader is vertex-lit. It combines:

- texture color
- material tint (`u_base_color`), usually initialized from MTL diffuse color
  (`Kd`)
- optional OBJ vertex colors
- ambient and scene lights
- fog
- color quantization
- optional screen-space dithering
- optional alpha cutout from texture alpha

Directional, point, and spot lights are supported, but the look is intentionally
N64-inspired rather than physically based.

## Unlit Rendering

The standard Unlit shader ignores scene light contribution but still applies
texture color, material tint, optional vertex colors, fog, quantization, and
dithering. It supports the same alpha cutout property as the Lit shader.

## Texture Filtering

Render settings support nearest, linear, and three-point texture sampling. The
default project setting is three-point for a retro 3D look.

Texture filtering, color levels, and dithering are project render settings.
Skybox, clouds, and global fog are stored per Scene in the coupled
`<scene_name>.lightingp64` asset. Open `Window > Lighting Settings` or
double-click that asset to edit the active Scene. Directional, point, and spot
lights remain Scene components.

Global fog uses color, near/far distance, and density values across the entire
Scene. Local Fog volume components are no longer used.

## Materials

OBJ files are imported as Model assets. A Model can contain multiple mesh entries
from Blender object/group exports, and each mesh can use one or more source
materials. P64 stores the imported mesh list, bounds, stats, and wireframe data
in the OBJ `.mdp64` sidecar so previews, picking, collision helpers, and gizmos
can reuse it.

OBJ files may reference `.mtl` files exported from tools such as Blender. P64
keeps those MTL values as source defaults. A `MeshRenderer` references one
concrete mesh entry from a Model and tracks:

- Source Materials: material names found in the OBJ/MTL data
- Materials: optional `.material` slots assigned by P64

If a slot is empty, P64 renders that source material with the standard VertexLit
shader and MTL defaults. To edit shader or property values, extract the source
materials into `.material` assets from the OBJ asset inspector.

`.material` files store runtime material data: shader reference, texture slots,
and shader property values. Editor-only defaults, source links, and usage cache
data live in hidden `.material.mdp64` sidecars.

## Custom Shaders

Custom shaders are stored as `.shader` files. They may declare a `Properties`
block before `Vertex` and `Fragment`:

```text
Properties
{
    Texture u_texture = ""
    Color u_base_color = (1.0, 1.0, 1.0)
}
```

The material inspector uses these declarations to show editable controls. Mesh
attributes are:

- `in_position`: required for mesh rendering
- `in_uv`: optional
- `in_normal`: optional for most shaders, required by silhouette outline passes
- `in_color`: optional vertex color

Shaders may omit unused optional attributes. The renderer tolerates attributes
that the compiler optimizes out.
