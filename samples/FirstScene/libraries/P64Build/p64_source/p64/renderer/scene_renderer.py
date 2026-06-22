from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, radians, sin, tan
from pathlib import Path
from typing import Any, Callable

from p64.engine.assets import AssetMetadata, discover_metadata, resolve_model_mesh
from p64.engine.builtin import CLOUD_SHADER_RELATIVE, ERROR_SHADER_RELATIVE, PARTICLE_SHADER_RELATIVE, SKYBOX_SHADER_RELATIVE, SPRITE_SHADER_RELATIVE, UI_IMAGE_SHADER_RELATIVE, UI_TEXT_SHADER_RELATIVE
from p64.engine.collision import collider_bounds, collider_sphere, controller_bounds
from p64.engine.components import AudioSource, Camera, Canvas, CharacterController, Collider, Light, MeshRenderer, ParticleEmitter, RectTransform, SpawnPoint, SpriteRenderer, UIImage, UIText
from p64.engine.entity import Entity, entity_effectively_active
from p64.engine.material import MaterialAsset, load_material_metadata, resolve_material_reference
from p64.engine.math import Vec3
from p64.engine.mesh_geometry import clear_mesh_geometry_cache, convex_hull, mesh_triangles, transform_triangle
from p64.engine.obj import mesh_vertices_for_group, parse_obj
from p64.engine.project import Project
from p64.engine.render_settings import clamp_render_settings, default_render_settings
from p64.engine.scene import Scene
from p64.engine.shader import default_shader_id, normalize_shader_id, parse_shader
from p64.engine.transforms import world_forward, world_position, world_right, world_rotation, world_scale, world_up
from p64.renderer.shaders import CLOUD_PLANE_FRAGMENT_SHADER, CLOUD_PLANE_VERTEX_SHADER, ERROR_FRAGMENT_SHADER, ERROR_VERTEX_SHADER, PARTICLE_FRAGMENT_SHADER, PARTICLE_VERTEX_SHADER, SKYBOX_FRAGMENT_SHADER, SKYBOX_VERTEX_SHADER, SPRITE_FRAGMENT_SHADER, SPRITE_VERTEX_SHADER, STANDARD_VERTEX_LIT_FRAGMENT_SHADER, STANDARD_VERTEX_LIT_VERTEX_SHADER, UI_FRAGMENT_SHADER, UI_VERTEX_SHADER


MAX_SHADER_LIGHTS = 8
MESH_VERTEX_FLOATS = 11
MESH_VERTEX_LAYOUT = "3f 2f 3f 3f"
MESH_POSITION_ONLY_LAYOUT = "3f 32x"
MESH_OUTLINE_LAYOUT = "3f 8x 3f 12x"
SPRITE_VERTEX_FLOATS = 8
SPRITE_VERTEX_LAYOUT = "3f 2f 3f"


def _render_exception_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc!r}"


@dataclass
class RenderMeshBatch:
    vao: Any
    buffer: Any
    vertex_count: int
    texture: Any
    shader: str | None
    base_color: tuple[float, float, float]
    material_properties: dict[str, Any] | None = None


@dataclass
class RenderMesh:
    vao: Any
    buffer: Any
    vertex_count: int
    texture: Any
    model_matrix: list[float]
    shader: str | None
    base_color: tuple[float, float, float]
    material_properties: dict[str, Any] | None = None
    batches: list[RenderMeshBatch] | None = None
    outline_vao: Any | None = None
    edge_mesh: RenderLineMesh | None = None


@dataclass
class RenderLineMesh:
    vao: Any
    buffer: Any
    vertex_count: int


@dataclass
class TextTexture:
    texture: Any
    size: tuple[int, int]


@dataclass(frozen=True)
class UILayoutDebugEntry:
    entity_id: str
    entity_name: str
    rect: tuple[float, float, float, float]
    image_rects: tuple[tuple[float, float, float, float], ...] = ()
    text_rects: tuple[tuple[float, float, float, float], ...] = ()


@dataclass
class RenderCamera:
    position: Vec3
    rotation: Vec3
    fov: float = 60.0
    near: float = 0.1
    far: float = 500.0
    forward: Vec3 | None = None
    right: Vec3 | None = None
    up: Vec3 | None = None


class SceneRenderer:
    def __init__(self, ctx: Any, project: Project, log: Any | None = None) -> None:
        import moderngl

        self.ctx = ctx
        self.moderngl = moderngl
        self.project = project
        self.log = log or (lambda message: None)
        self.program = self._compile_default_program()
        self.error_program = self._compile_builtin_program(ERROR_SHADER_RELATIVE, ERROR_VERTEX_SHADER, ERROR_FRAGMENT_SHADER, "error")
        self.selection_outline_program = ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                in vec3 in_normal;
                uniform mat4 u_model;
                uniform mat4 u_view;
                uniform mat4 u_projection;
                uniform float u_outline_width;
                void main() {
                    vec4 world_pos = u_model * vec4(in_position, 1.0);
                    vec3 world_normal = normalize(mat3(u_model) * in_normal);
                    gl_Position = u_projection * u_view * vec4(world_pos.xyz + world_normal * u_outline_width, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 u_color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(u_color, 1.0);
                }
            """,
        )
        self.line_program = ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                uniform mat4 u_model;
                uniform mat4 u_view;
                uniform mat4 u_projection;
                void main() {
                    gl_Position = u_projection * u_view * u_model * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 u_color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(u_color, 1.0);
                }
            """,
        )
        self.skybox_program = self._compile_builtin_program(SKYBOX_SHADER_RELATIVE, SKYBOX_VERTEX_SHADER, SKYBOX_FRAGMENT_SHADER, "skybox")
        self.cloud_plane_program = self._compile_builtin_program(CLOUD_SHADER_RELATIVE, CLOUD_PLANE_VERTEX_SHADER, CLOUD_PLANE_FRAGMENT_SHADER, "cloud")
        self.sprite_program = self._compile_builtin_program(SPRITE_SHADER_RELATIVE, SPRITE_VERTEX_SHADER, SPRITE_FRAGMENT_SHADER, "sprite")
        self.ui_image_program = self._compile_builtin_program(UI_IMAGE_SHADER_RELATIVE, UI_VERTEX_SHADER, UI_FRAGMENT_SHADER, "ui image")
        self.ui_text_program = self._compile_builtin_program(UI_TEXT_SHADER_RELATIVE, UI_VERTEX_SHADER, UI_FRAGMENT_SHADER, "ui text")
        self.particle_program = self._compile_builtin_program(PARTICLE_SHADER_RELATIVE, PARTICLE_VERTEX_SHADER, PARTICLE_FRAGMENT_SHADER, "particle")
        self.skybox_buffer = None
        self.skybox_vao = None
        self.cloud_plane_buffer = None
        self.cloud_plane_vao = None
        self.cloud_plane_cache_key: tuple[float, float, int] | None = None
        self.upscale_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_position;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    v_uv = in_uv;
                    gl_Position = vec4(in_position, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_texture;
                in vec2 v_uv;
                out vec4 fragColor;
                void main() {
                    fragColor = texture(u_texture, v_uv);
                }
            """,
        )
        self._render_target_cache: dict[tuple[int, int], tuple[Any, Any, Any]] = {}
        self._text_texture_cache: dict[tuple[str, str, str, str, float, tuple[float, float, float], float], TextTexture] = {}
        self._program_cache: dict[str | None, Any] = {None: self.program, "__p64_error__": self.error_program}
        self._metadata: dict[str, AssetMetadata] = {}
        self._texture_cache: dict[Path, Any] = {}
        self._mesh_cache: dict[tuple[str, str | None, str | None, str | None, tuple[str, ...], tuple[str | None, ...]], RenderMesh] = {}
        self._mesh_collider_line_cache: dict[tuple[str, str | None], RenderLineMesh] = {}
        self._convex_collider_line_cache: dict[tuple[str, str | None], RenderLineMesh] = {}
        self._white_texture = None
        self._transparent_texture = None
        self._logged_scene_stats = False
        self.reload_assets()

    def reload_assets(self) -> None:
        self._metadata.clear()
        for metadata_path in discover_metadata(self.project.assets_dir):
            try:
                metadata = AssetMetadata.load(metadata_path)
            except Exception as exc:
                self.log(f"Could not load asset metadata {metadata_path}: {exc}")
                continue
            self._metadata[metadata.id] = metadata
        self._mesh_cache.clear()
        self._mesh_collider_line_cache.clear()
        self._convex_collider_line_cache.clear()
        self._release_cloud_plane_cache()
        clear_mesh_geometry_cache(self.project)
        self._program_cache = {None: self.program, "__p64_error__": self.error_program}

    def render(
        self,
        scene: Scene,
        width: int,
        height: int,
        camera: RenderCamera | None = None,
        selected_entity_id: str | None = None,
        show_grid: bool = True,
        game_view: bool = False,
        output_framebuffer: Any | None = None,
    ) -> bool:
        render_settings = clamp_render_settings({**default_render_settings(), **scene.render_settings})
        missing_camera = game_view and scene.active_camera() is None
        if missing_camera:
            self.ctx.viewport = (0, 0, max(width, 1), max(height, 1))
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            return False
        if game_view and _scene_resolution_mode(scene) == "fixed" and hasattr(self.ctx, "framebuffer"):
            target_width, target_height = _game_render_size(scene, render_settings)
            color, _depth, framebuffer = self._render_target(target_width, target_height)
            framebuffer.use()
            self._render_scene(scene, target_width, target_height, camera, selected_entity_id, show_grid, render_settings)
            if output_framebuffer is not None:
                output_framebuffer.use()
            self.ctx.viewport = (0, 0, max(width, 1), max(height, 1))
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self._draw_upscaled_texture(color)
            return True
        render_width, render_height = (max(width, 1), max(height, 1))
        if game_view and _scene_resolution_mode(scene) == "fixed":
            render_width, render_height = _game_render_size(scene, render_settings)
        self._render_scene(scene, render_width, render_height, camera, selected_entity_id, show_grid, render_settings)
        return True

    def _render_scene(
        self,
        scene: Scene,
        width: int,
        height: int,
        camera: RenderCamera | None,
        selected_entity_id: str | None,
        show_grid: bool,
        render_settings: dict[str, Any],
    ) -> None:
        self.ctx.viewport = (0, 0, max(width, 1), max(height, 1))
        self.ctx.clear(0.22, 0.27, 0.33, 1.0)
        self.ctx.enable(self.moderngl.DEPTH_TEST)
        render_camera = camera or _camera_from_entity(scene.active_camera())
        view = _view_matrix(render_camera)
        aspect = max(width / max(height, 1), 0.01)
        projection = _perspective_matrix(render_camera.fov, aspect, render_camera.near, render_camera.far)
        render_settings = clamp_render_settings({**default_render_settings(), **scene.render_settings})
        self._current_scene = scene
        self._current_camera = render_camera
        self._current_aspect = aspect
        self._current_view = view
        self._current_projection = projection
        self._current_render_settings = render_settings
        self._render_time = float(getattr(self, "_render_time", 0.0)) + (1.0 / 60.0)
        if render_settings.get("skybox_enabled", True):
            self._draw_skybox(render_camera, render_settings)
            self._draw_cloud_plane(render_camera, view, projection, render_settings)
        submitted = 0
        for entity in scene.walk_active():
            for component in entity.components:
                if isinstance(component, MeshRenderer) and component.enabled and component.visible:
                    if self._draw_mesh(entity, component):
                        submitted += 1
        self._draw_world_sprites(scene, render_camera, view, projection)
        self._draw_particles(scene, render_camera, view, projection)
        self._draw_ui(scene, width, height)
        if not self._logged_scene_stats:
            self.log(f"Scene renderer submitted {submitted} mesh renderer(s); metadata loaded: {len(self._metadata)}")
            self._logged_scene_stats = True
        selected = scene.find(selected_entity_id) if selected_entity_id else None
        if show_grid:
            self._draw_grid(view, projection)
            self._draw_component_gizmos(scene, view, projection, selected)
        if selected:
            self._draw_selection_outline(selected, view, projection)

    def pick_entity(
        self,
        scene: Scene,
        width: int,
        height: int,
        screen_x: float,
        screen_y: float,
        camera: RenderCamera | None = None,
    ) -> str | None:
        render_camera = camera or _camera_from_entity(scene.active_camera())
        origin, direction = _screen_ray(render_camera, width, height, screen_x, screen_y)
        best_id: str | None = None
        best_distance: float | None = None
        for entity in scene.walk_active():
            matrix = entity.transform.world_matrix(entity)
            for component in entity.components:
                if not isinstance(component, MeshRenderer) or not component.enabled or not component.visible:
                    continue
                for triangle in mesh_triangles(self.project, component):
                    hit = _ray_triangle_intersection(origin, direction, transform_triangle(matrix, triangle))
                    if hit is None:
                        continue
                    if best_distance is None or hit < best_distance:
                        best_distance = hit
                        best_id = entity.id
        return best_id

    def _apply_common_uniforms(self, program: Any, scene: Scene, camera: RenderCamera, view: list[float], projection: list[float]) -> None:
        render_settings = getattr(self, "_current_render_settings", clamp_render_settings({**default_render_settings(), **scene.render_settings}))
        self._set_uniform(program, "u_view", _mat4_bytes(view), write=True)
        self._set_uniform(program, "u_projection", _mat4_bytes(projection), write=True)
        self._set_uniform(program, "u_color_levels", float(render_settings.get("color_levels", 32)))
        self._set_uniform(program, "u_texture_filter", _texture_filter_code(str(render_settings.get("texture_filter", "three_point"))))
        self._set_uniform(program, "u_dithering_enabled", bool(render_settings.get("dithering", True)))
        self._apply_light_uniforms(program, scene)
        self._apply_fog_uniforms(program, scene, camera)

    def _draw_skybox(self, camera: RenderCamera, settings: dict[str, Any]) -> None:
        stage = "vao"
        background_started = False
        try:
            vao = self._skybox_vertex_array()
            stage = "uniform"
            self._set_uniform(self.skybox_program, "u_skybox_top_color", _color3(settings.get("skybox_top_color")))
            self._set_uniform(self.skybox_program, "u_skybox_horizon_color", _color3(settings.get("skybox_horizon_color")))
            self._set_uniform(self.skybox_program, "u_color_levels", float(settings.get("color_levels", 32)))
            self._set_uniform(self.skybox_program, "u_dithering_enabled", bool(settings.get("dithering", True)))
            stage = "begin"
            background_started = self._begin_background_pass(blend=False)
            stage = "render"
            vao.render()
        except Exception as exc:
            self.log(f"Skybox render failed during {stage}: {_render_exception_text(exc)}")
        finally:
            if background_started:
                self._end_background_pass()

    def _skybox_vertex_array(self) -> Any:
        if self.skybox_vao is None:
            import struct

            vertices = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]
            self.skybox_buffer = self.ctx.buffer(struct.pack("6f", *vertices))
            self.skybox_vao = self.ctx.vertex_array(self.skybox_program, [(self.skybox_buffer, "2f", "in_position")])
        return self.skybox_vao

    def _draw_cloud_plane(self, camera: RenderCamera, view: list[float], projection: list[float], settings: dict[str, Any]) -> None:
        stage = "vao"
        background_started = False
        try:
            vao = self._cloud_plane_vertex_array(camera, float(settings.get("skybox_cloud_height", 80.0)))
            stage = "uniform"
            self._set_uniform(self.cloud_plane_program, "u_view", _mat4_bytes(view), write=True)
            self._set_uniform(self.cloud_plane_program, "u_projection", _mat4_bytes(projection), write=True)
            self._set_uniform(self.cloud_plane_program, "u_cloud_origin", _vec3_values(camera.position))
            self._set_uniform(self.cloud_plane_program, "u_skybox_cloud_color", _color3(settings.get("skybox_cloud_color")))
            self._set_uniform(self.cloud_plane_program, "u_skybox_cloud_coverage", float(settings.get("skybox_cloud_coverage", 0.45)))
            self._set_uniform(self.cloud_plane_program, "u_skybox_cloud_scale", float(settings.get("skybox_cloud_scale", 3.0)))
            self._set_uniform(self.cloud_plane_program, "u_skybox_cloud_height", float(settings.get("skybox_cloud_height", 80.0)))
            self._set_uniform(self.cloud_plane_program, "u_skybox_cloud_softness", float(settings.get("skybox_cloud_softness", 0.08)))
            self._set_uniform(self.cloud_plane_program, "u_color_levels", float(settings.get("color_levels", 32)))
            self._set_uniform(self.cloud_plane_program, "u_dithering_enabled", bool(settings.get("dithering", True)))
            stage = "begin"
            background_started = self._begin_background_pass(blend=True)
            stage = "render"
            vao.render()
        except Exception as exc:
            self.log(f"Cloud plane render failed during {stage}: {_render_exception_text(exc)}")
        finally:
            if background_started:
                self._end_background_pass()

    def _cloud_plane_vertex_array(self, camera: RenderCamera, height: float) -> Any:
        import struct

        height = max(0.1, float(height))
        radius = max(240.0, height * 8.0, camera.far * 0.75)
        segments = 48
        key = (round(height, 4), round(radius, 4), segments)
        if self.cloud_plane_vao is not None and self.cloud_plane_cache_key == key:
            return self.cloud_plane_vao
        self._release_cloud_plane_cache()
        vertices = cloud_dome_vertices(height, radius, segments)
        self.cloud_plane_buffer = self.ctx.buffer(struct.pack(f"{len(vertices)}f", *vertices))
        self.cloud_plane_vao = self.ctx.vertex_array(self.cloud_plane_program, [(self.cloud_plane_buffer, "3f", "in_position")])
        self.cloud_plane_cache_key = key
        return self.cloud_plane_vao

    def _release_cloud_plane_cache(self) -> None:
        for resource in (self.cloud_plane_vao, self.cloud_plane_buffer):
            if resource is None:
                continue
            try:
                resource.release()
            except Exception:
                pass
        self.cloud_plane_vao = None
        self.cloud_plane_buffer = None
        self.cloud_plane_cache_key = None

    def _render_target(self, width: int, height: int) -> tuple[Any, Any, Any]:
        key = (max(1, int(width)), max(1, int(height)))
        if key not in self._render_target_cache:
            color = self.ctx.texture(key, 4)
            color.filter = (self.moderngl.NEAREST, self.moderngl.NEAREST)
            depth = self.ctx.depth_renderbuffer(key)
            framebuffer = self.ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
            self._render_target_cache[key] = (color, depth, framebuffer)
        return self._render_target_cache[key]

    def _draw_upscaled_texture(self, texture: Any) -> None:
        import struct

        vertices = [
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            -1.0, -1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            -1.0, 1.0, 0.0, 1.0,
        ]
        buffer = self.ctx.buffer(struct.pack(f"{len(vertices)}f", *vertices))
        vao = self.ctx.vertex_array(self.upscale_program, [(buffer, "2f 2f", "in_position", "in_uv")])
        try:
            texture.use(location=0)
            self._set_uniform(self.upscale_program, "u_texture", 0)
            if hasattr(self.moderngl, "DEPTH_TEST"):
                self._safe_context_call("upscale/disable_depth", lambda: self.ctx.disable(self.moderngl.DEPTH_TEST))
            vao.render()
        finally:
            buffer.release()
            vao.release()

    def _begin_background_pass(self, blend: bool) -> bool:
        self._background_has_depth_mask, self._background_previous_depth_mask = self._safe_get_context_attr("depth_mask", "begin/depth_mask")
        self._background_has_blend_func, self._background_previous_blend_func = self._safe_get_context_attr("blend_func", "begin/blend_func")
        if self._background_has_depth_mask:
            self._safe_set_context_attr("depth_mask", False, "begin/depth_mask")
        if hasattr(self.moderngl, "DEPTH_TEST"):
            self._safe_context_call("begin/disable_depth", lambda: self.ctx.disable(self.moderngl.DEPTH_TEST))
        if blend and hasattr(self.moderngl, "BLEND"):
            self._safe_context_call("begin/enable_blend", lambda: self.ctx.enable(self.moderngl.BLEND))
            if self._background_has_blend_func and hasattr(self.moderngl, "SRC_ALPHA") and hasattr(self.moderngl, "ONE_MINUS_SRC_ALPHA"):
                self._safe_set_context_attr("blend_func", (self.moderngl.SRC_ALPHA, self.moderngl.ONE_MINUS_SRC_ALPHA), "begin/blend_func")
        return True

    def _end_background_pass(self) -> None:
        if hasattr(self.moderngl, "BLEND"):
            self._safe_context_call("end/disable_blend", lambda: self.ctx.disable(self.moderngl.BLEND))
        if getattr(self, "_background_has_blend_func", False):
            self._safe_set_context_attr("blend_func", self._background_previous_blend_func, "end/blend_func")
        if getattr(self, "_background_has_depth_mask", False):
            self._safe_set_context_attr("depth_mask", self._background_previous_depth_mask, "end/depth_mask")
        if hasattr(self.moderngl, "DEPTH_TEST"):
            self._safe_context_call("end/enable_depth", lambda: self.ctx.enable(self.moderngl.DEPTH_TEST))

    def _begin_transparent_pass(self, depth_test: bool, additive: bool = False) -> None:
        _has_depth, self._transparent_previous_depth_mask = self._safe_get_context_attr("depth_mask", "transparent/depth_mask")
        self._transparent_has_blend_func, self._transparent_previous_blend_func = self._safe_get_context_attr("blend_func", "transparent/blend_func")
        if _has_depth:
            self._safe_set_context_attr("depth_mask", False, "transparent/depth_mask")
        if hasattr(self.moderngl, "DEPTH_TEST"):
            if depth_test:
                self._safe_context_call("transparent/enable_depth", lambda: self.ctx.enable(self.moderngl.DEPTH_TEST))
            else:
                self._safe_context_call("transparent/disable_depth", lambda: self.ctx.disable(self.moderngl.DEPTH_TEST))
        if hasattr(self.moderngl, "BLEND"):
            self._safe_context_call("transparent/enable_blend", lambda: self.ctx.enable(self.moderngl.BLEND))
            if self._transparent_has_blend_func and hasattr(self.moderngl, "SRC_ALPHA") and hasattr(self.moderngl, "ONE_MINUS_SRC_ALPHA"):
                dst = self.moderngl.ONE if additive and hasattr(self.moderngl, "ONE") else self.moderngl.ONE_MINUS_SRC_ALPHA
                self._safe_set_context_attr("blend_func", (self.moderngl.SRC_ALPHA, dst), "transparent/blend_func")

    def _end_transparent_pass(self, depth_test: bool) -> None:
        if hasattr(self.moderngl, "BLEND"):
            self._safe_context_call("transparent/disable_blend", lambda: self.ctx.disable(self.moderngl.BLEND))
        if getattr(self, "_transparent_has_blend_func", False):
            self._safe_set_context_attr("blend_func", self._transparent_previous_blend_func, "transparent/blend_func_restore")
        if getattr(self, "_transparent_previous_depth_mask", None) is not None:
            self._safe_set_context_attr("depth_mask", self._transparent_previous_depth_mask, "transparent/depth_mask_restore")
        if hasattr(self.moderngl, "DEPTH_TEST"):
            self._safe_context_call("transparent/enable_depth_restore", lambda: self.ctx.enable(self.moderngl.DEPTH_TEST))

    def _safe_get_context_attr(self, name: str, stage: str) -> tuple[bool, Any]:
        try:
            return True, getattr(self.ctx, name)
        except AttributeError:
            return False, None
        except NotImplementedError:
            return False, None
        except Exception as exc:
            self.log(f"Background pass {stage} failed: {_render_exception_text(exc)}")
            return False, None

    def _safe_set_context_attr(self, name: str, value: Any, stage: str) -> None:
        try:
            setattr(self.ctx, name, value)
        except NotImplementedError:
            return
        except Exception as exc:
            self.log(f"Background pass {stage} failed: {_render_exception_text(exc)}")

    def _safe_context_call(self, stage: str, callback: Any) -> None:
        try:
            callback()
        except Exception as exc:
            self.log(f"Background pass {stage} failed: {_render_exception_text(exc)}")

    def _apply_light_uniforms(self, program: Any, scene: Scene) -> None:
        lights = scene.lights()[:MAX_SHADER_LIGHTS]
        self._set_uniform(program, "u_light_count", len(lights))
        self._set_uniform(program, "u_ambient_color", (0.18, 0.18, 0.18))
        kinds: list[int] = []
        positions: list[tuple[float, float, float]] = []
        directions: list[tuple[float, float, float]] = []
        colors: list[tuple[float, float, float]] = []
        intensities: list[float] = []
        ranges: list[float] = []
        spot_angles: list[float] = []
        falloffs: list[float] = []
        for index in range(MAX_SHADER_LIGHTS):
            if index < len(lights):
                entity, light = lights[index]
                position = _world_position(entity)
                direction = world_forward(entity)
                kind_value = _light_kind_code(light)
                position_value = (position.x, position.y, position.z)
                direction_value = (direction.x, direction.y, direction.z)
                color_value = (light.color.x, light.color.y, light.color.z)
                intensity_value = max(0.0, float(light.intensity))
                range_value = max(0.001, float(light.range))
                spot_angle_value = max(1.0, min(179.0, float(light.spot_angle)))
                falloff_value = max(0.001, float(light.falloff))
            else:
                kind_value = 0
                position_value = (0.0, 0.0, 0.0)
                direction_value = (0.0, -1.0, 0.0)
                color_value = (1.0, 1.0, 1.0)
                intensity_value = 0.0
                range_value = 1.0
                spot_angle_value = 45.0
                falloff_value = 2.0
            kinds.append(kind_value)
            positions.append(position_value)
            directions.append(direction_value)
            colors.append(color_value)
            intensities.append(intensity_value)
            ranges.append(range_value)
            spot_angles.append(spot_angle_value)
            falloffs.append(falloff_value)
            self._set_uniform(program, f"u_light_kind[{index}]", kind_value)
            self._set_uniform(program, f"u_light_position[{index}]", position_value)
            self._set_uniform(program, f"u_light_direction[{index}]", direction_value)
            self._set_uniform(program, f"u_light_color[{index}]", color_value)
            self._set_uniform(program, f"u_light_intensity[{index}]", intensity_value)
            self._set_uniform(program, f"u_light_range[{index}]", range_value)
            self._set_uniform(program, f"u_light_spot_angle[{index}]", spot_angle_value)
            self._set_uniform(program, f"u_light_falloff[{index}]", falloff_value)
        self._set_uniform(program, "u_light_kind", tuple(kinds))
        self._set_uniform(program, "u_light_position", tuple(positions))
        self._set_uniform(program, "u_light_direction", tuple(directions))
        self._set_uniform(program, "u_light_color", tuple(colors))
        self._set_uniform(program, "u_light_intensity", tuple(intensities))
        self._set_uniform(program, "u_light_range", tuple(ranges))
        self._set_uniform(program, "u_light_spot_angle", tuple(spot_angles))
        self._set_uniform(program, "u_light_falloff", tuple(falloffs))

    def _apply_fog_uniforms(self, program: Any, scene: Scene, camera: RenderCamera) -> None:
        fog_volume = scene.fog_volume()
        self._set_uniform(program, "u_camera_position", (camera.position.x, camera.position.y, camera.position.z))
        if fog_volume is None:
            self._set_uniform(program, "u_fog_enabled", False)
            self._set_uniform(program, "u_fog_color", (0.46, 0.58, 0.72))
            self._set_uniform(program, "u_fog_center", (0.0, 0.0, 0.0))
            self._set_uniform(program, "u_fog_size", (1.0, 1.0, 1.0))
            self._set_uniform(program, "u_fog_near", 20.0)
            self._set_uniform(program, "u_fog_far", 120.0)
            self._set_uniform(program, "u_fog_density", 0.0)
            return
        entity, fog = fog_volume
        center = _world_position(entity)
        scale = world_scale(entity)
        size = Vec3(fog.size.x * scale.x, fog.size.y * scale.y, fog.size.z * scale.z)
        self._set_uniform(program, "u_fog_enabled", True)
        self._set_uniform(program, "u_fog_color", (fog.color.x, fog.color.y, fog.color.z))
        self._set_uniform(program, "u_fog_center", (center.x, center.y, center.z))
        self._set_uniform(program, "u_fog_size", (max(size.x, 0.001), max(size.y, 0.001), max(size.z, 0.001)))
        self._set_uniform(program, "u_fog_near", fog.near)
        self._set_uniform(program, "u_fog_far", fog.far)
        self._set_uniform(program, "u_fog_density", fog.density)

    def _draw_mesh(self, entity: Entity, component: MeshRenderer) -> bool:
        mesh = self._load_mesh(entity, component)
        if mesh is None:
            return False
        program = self._program_for(mesh.shader)
        render_camera = getattr(self, "_current_camera", _camera_from_entity(None))
        view = getattr(self, "_current_view", _view_matrix(render_camera))
        projection = getattr(self, "_current_projection", _perspective_matrix(render_camera.fov, 1.0, render_camera.near, render_camera.far))
        scene = getattr(self, "_current_scene", None)
        if scene is not None:
            self._apply_common_uniforms(program, scene, render_camera, view, projection)
        self._set_uniform(program, "u_model", _mat4_bytes(entity.transform.world_matrix(entity)), write=True)
        if mesh.batches:
            for batch in mesh.batches:
                batch_program = self._program_for(batch.shader)
                if scene is not None:
                    self._apply_common_uniforms(batch_program, scene, render_camera, view, projection)
                self._set_uniform(batch_program, "u_model", _mat4_bytes(entity.transform.world_matrix(entity)), write=True)
                batch.texture.use(location=0)
                self._set_uniform(batch_program, "u_texture", 0)
                self._set_uniform(batch_program, "u_base_color", batch.base_color)
                self._set_uniform(batch_program, "u_alpha_cutoff", 0.0)
                self._apply_material_properties(batch_program, batch.material_properties)
                batch.vao.render()
            return True
        mesh.texture.use(location=0)
        self._set_uniform(program, "u_texture", 0)
        self._set_uniform(program, "u_base_color", mesh.base_color)
        self._set_uniform(program, "u_alpha_cutoff", 0.0)
        self._apply_material_properties(program, mesh.material_properties)
        mesh.vao.render()
        return True

    def _draw_world_sprites(self, scene: Scene, camera: RenderCamera, view: list[float], projection: list[float]) -> None:
        items: list[tuple[int, float, Entity, SpriteRenderer]] = []
        for entity in scene.walk_active():
            for component in entity.components:
                if isinstance(component, SpriteRenderer) and component.enabled:
                    position = _world_position(entity)
                    distance = _distance_squared(position, camera.position)
                    items.append((int(component.sorting_order), -distance, entity, component))
        items.sort(key=lambda item: (item[0], item[1]))
        for _order, _distance, entity, component in items:
            vertices = sprite_quad_vertices(entity, component, camera, getattr(self, "_render_time", 0.0))
            material_asset = self._load_material_asset(resolve_material_reference(self.project.root, component.material))
            shader = material_asset.shader if material_asset is not None else SPRITE_SHADER_RELATIVE
            program = self._program_for(shader)
            texture = self._component_texture(component.texture, material_asset, component.material)
            self._draw_textured_quad_batch(vertices, program, texture, view, projection, _identity_matrix(), component.color, component.alpha, material_asset)

    def _draw_particles(self, scene: Scene, camera: RenderCamera, view: list[float], projection: list[float]) -> None:
        dt = 1.0 / 60.0
        for entity in scene.walk_active():
            for component in entity.components:
                if not isinstance(component, ParticleEmitter) or not component.enabled:
                    continue
                self._step_particle_emitter(entity, component, dt)
                vertices = particle_quad_vertices(entity, component, camera)
                if not vertices:
                    continue
                material_asset = self._load_material_asset(resolve_material_reference(self.project.root, component.material))
                shader = material_asset.shader if material_asset is not None else PARTICLE_SHADER_RELATIVE
                program = self._program_for(shader)
                texture = self._component_texture(component.texture, material_asset, component.material)
                self._draw_textured_quad_batch(vertices, program, texture, view, projection, _identity_matrix(), component.start_color, component.start_alpha, material_asset, additive=component.blend_mode == "additive")

    def _draw_ui(self, scene: Scene, width: int, height: int) -> None:
        canvases = _canvas_entities(scene)
        ui_roots: list[tuple[list[Entity], Canvas]] = (
            [(list(canvas_entity.walk()), canvas) for canvas_entity, canvas in sorted(canvases, key=lambda item: int(item[1].sort_order))]
            if canvases
            else [(list(scene.walk_active()), Canvas())]
        )
        for entities, canvas in ui_roots:
            if not canvas.enabled:
                continue
            layout_width, layout_height = _canvas_layout_size(canvas, width, height)
            ui_rects = _canvas_entity_rects(entities[0] if canvases and entities else None, layout_width, layout_height)
            for entity in entities:
                if not entity_effectively_active(entity):
                    continue
                entity_rect = ui_rects.get(entity.id)
                for component in entity.components:
                    if isinstance(component, UIImage) and component.enabled:
                        vertices = ui_quad_vertices(component, layout_width, layout_height, getattr(self, "_render_time", 0.0), entity_rect)
                        material_asset = self._load_material_asset(resolve_material_reference(self.project.root, component.material))
                        shader = material_asset.shader if material_asset is not None else UI_IMAGE_SHADER_RELATIVE
                        program = self._program_for(shader)
                        texture = self._component_texture(component.texture, material_asset, component.material)
                        self._draw_ui_quad_batch(vertices, program, texture, layout_width, layout_height, component.color, component.alpha, material_asset)
                    elif isinstance(component, UIText) and component.enabled:
                        text_texture = self._text_texture(component)
                        vertices = ui_text_vertices(component, layout_width, layout_height, entity_rect, text_texture.size)
                        self._draw_ui_quad_batch(vertices, self._program_for(UI_TEXT_SHADER_RELATIVE), text_texture.texture, layout_width, layout_height, Vec3(1.0, 1.0, 1.0), component.alpha, None)

    def _draw_textured_quad_batch(
        self,
        vertices: list[float],
        program: Any,
        texture: Any,
        view: list[float],
        projection: list[float],
        model: list[float],
        color: Vec3,
        alpha: float,
        material_asset: MaterialAsset | None,
        additive: bool = False,
    ) -> None:
        if not vertices:
            return
        try:
            import struct

            buffer = self.ctx.buffer(struct.pack(f"{len(vertices)}f", *vertices))
            vao = self.ctx.vertex_array(program, [(buffer, SPRITE_VERTEX_LAYOUT, "in_position", "in_uv", "in_color")], skip_errors=True)
            texture.use(location=0)
            self._set_uniform(program, "u_texture", 0)
            self._set_uniform(program, "u_model", _mat4_bytes(model), write=True)
            self._set_uniform(program, "u_view", _mat4_bytes(view), write=True)
            self._set_uniform(program, "u_projection", _mat4_bytes(projection), write=True)
            self._set_uniform(program, "u_base_color", _vec3_values(color))
            self._set_uniform(program, "u_alpha", max(0.0, min(1.0, float(alpha))))
            self._set_uniform(program, "u_alpha_cutoff", 0.0)
            if material_asset is not None:
                self._apply_material_properties(program, material_asset.properties)
            self._begin_transparent_pass(depth_test=True, additive=additive)
            vao.render()
            buffer.release()
            vao.release()
        except Exception as exc:
            self.log(f"Sprite batch render failed: {_render_exception_text(exc)}")
        finally:
            self._end_transparent_pass(depth_test=True)

    def _draw_ui_quad_batch(self, vertices: list[float], program: Any, texture: Any, width: int, height: int, color: Vec3, alpha: float, material_asset: MaterialAsset | None) -> None:
        if not vertices:
            return
        try:
            import struct

            clip_vertices = ui_vertices_to_ndc(vertices, width, height)
            buffer = self.ctx.buffer(struct.pack(f"{len(clip_vertices)}f", *clip_vertices))
            vao = self.ctx.vertex_array(program, [(buffer, SPRITE_VERTEX_LAYOUT, "in_position", "in_uv", "in_color")], skip_errors=True)
            texture.use(location=0)
            self._set_uniform(program, "u_texture", 0)
            self._set_uniform(program, "u_base_color", _vec3_values(color))
            self._set_uniform(program, "u_alpha", max(0.0, min(1.0, float(alpha))))
            self._set_uniform(program, "u_alpha_cutoff", 0.0)
            if material_asset is not None:
                self._apply_material_properties(program, material_asset.properties)
            self._begin_transparent_pass(depth_test=False, additive=False)
            vao.render()
            buffer.release()
            vao.release()
        except Exception as exc:
            self.log(f"UI batch render failed: {_render_exception_text(exc)}")
        finally:
            self._end_transparent_pass(depth_test=False)

    def _step_particle_emitter(self, entity: Entity, emitter: ParticleEmitter, dt: float) -> None:
        if emitter._runtime_playing:
            if emitter.burst > 0 and not emitter._runtime_burst_done:
                emitter.emit(emitter.burst)
                emitter._runtime_burst_done = True
            emitter._runtime_accumulator += max(0.0, float(emitter.rate)) * dt
            while emitter._runtime_accumulator >= 1.0:
                emitter.emit(1)
                emitter._runtime_accumulator -= 1.0
                if not emitter.looping and len(emitter._runtime_particles) >= max(0, int(emitter.max_particles)):
                    emitter.stop()
                    break
        origin = _world_position(entity)
        alive: list[dict[str, Any]] = []
        for particle in emitter._runtime_particles:
            age = float(particle.get("age", 0.0)) + dt
            lifetime = max(0.001, float(particle.get("lifetime", emitter.lifetime)))
            if age >= lifetime:
                continue
            velocity = particle.get("velocity")
            position = particle.get("position")
            if not isinstance(velocity, Vec3) or not isinstance(position, Vec3):
                continue
            velocity.x += emitter.gravity.x * dt
            velocity.y += emitter.gravity.y * dt
            velocity.z += emitter.gravity.z * dt
            position.x += velocity.x * dt
            position.y += velocity.y * dt
            position.z += velocity.z * dt
            particle["age"] = age
            if not emitter.local_space and not particle.get("world_initialized"):
                position.x += origin.x
                position.y += origin.y
                position.z += origin.z
                particle["world_initialized"] = True
            alive.append(particle)
        emitter._runtime_particles = alive

    def _component_texture(self, texture_reference: str, material_asset: MaterialAsset | None = None, material_reference_value: str | None = None) -> Any:
        if material_asset is not None:
            material_path = resolve_material_reference(self.project.root, material_reference_value)
            path = self._material_texture_path(material_asset.textures.get("u_texture"), material_path)
            if path and path.exists():
                return self._load_texture(path)
        path = self._texture_reference_path(texture_reference)
        return self._load_texture(path) if path and path.exists() else self._default_texture()

    def _texture_reference_path(self, texture_reference: str | None) -> Path | None:
        if not texture_reference:
            return None
        path = Path(str(texture_reference))
        if path.is_absolute():
            return path
        candidates = []
        if str(texture_reference).startswith(("assets/", "packages/")):
            candidates.append(self.project.root / texture_reference)
        candidates.append(self.project.assets_dir / texture_reference)
        return next((candidate for candidate in candidates if candidate.exists()), candidates[0] if candidates else None)

    def _text_texture(self, component: UIText) -> TextTexture:
        font_source = component.font_source if component.font_source in {"system", "asset"} else "system"
        key = (component.text, font_source, component.font_family, component.bitmap_font, float(component.font_size), _vec3_values(component.color), float(component.alpha))
        if key in self._text_texture_cache:
            return self._text_texture_cache[key]
        pil_error: Exception | None = None
        try:
            from PIL import Image, ImageDraw, ImageFont

            font_size = max(1, int(component.font_size))
            font = self._resolve_text_font(ImageFont, component, font_source, font_size)
            text = component.text or " "
            dummy = Image.new("RGBA", (1, 1))
            draw = ImageDraw.Draw(dummy)
            bounds = draw.textbbox((0, 0), text, font=font)
            width = max(1, bounds[2] - bounds[0] + 4)
            height = max(1, bounds[3] - bounds[1] + 4)
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.text((2 - bounds[0], 2 - bounds[1]), text, font=font, fill=255)
            pixels = bytearray()
            for alpha in mask.tobytes():
                if alpha:
                    pixels.extend((255, 255, 255, alpha))
                else:
                    pixels.extend((0, 0, 0, 0))
            image = Image.frombytes("RGBA", (width, height), bytes(pixels))
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            texture = self.ctx.texture(image.size, 4, image.tobytes())
            texture.filter = self._texture_filter()
            result = TextTexture(texture, (width, height))
            self._text_texture_cache[key] = result
            return result
        except Exception as exc:
            pil_error = exc
        try:
            result = self._text_texture_qt(component, font_source)
            self._text_texture_cache[key] = result
            return result
        except Exception as exc:
            self.log(f"Could not render UI text texture: PIL={_render_exception_text(pil_error) if pil_error is not None else 'not tried'}; Qt={_render_exception_text(exc)}")
            result = TextTexture(self._transparent_default_texture(), _fallback_text_size(component))
            self._text_texture_cache[key] = result
            return result

    def _text_texture_qt(self, component: UIText, font_source: str) -> TextTexture:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QImage, QPainter

        font_size = max(1, int(component.font_size))
        font = QFont()
        if font_source == "asset" and component.bitmap_font:
            font_path = self._texture_reference_path(component.bitmap_font)
            if font_path and font_path.exists():
                font_id = QFontDatabase.addApplicationFont(str(font_path))
                families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
                if families:
                    font = QFont(families[0])
                else:
                    self.log(f"UI text font asset missing or invalid: {component.bitmap_font}")
            else:
                self.log(f"UI text font asset missing: {component.bitmap_font}")
        elif component.font_family and component.font_family != "System":
            font = QFont(component.font_family)
        font.setPixelSize(font_size)

        text = component.text or " "
        metrics = QFontMetrics(font)
        bounds = metrics.boundingRect(text)
        width = max(1, bounds.width() + 4)
        height = max(1, metrics.height() + 4)
        image = QImage(width, height, QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        try:
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255, 255))
            painter.drawText(2 - bounds.x(), 2 + metrics.ascent(), text)
        finally:
            painter.end()
        image = image.mirrored(False, True)
        data = bytes(image.bits())
        expected = width * height * 4
        if len(data) > expected:
            data = data[:expected]
        texture = self.ctx.texture((width, height), 4, data)
        texture.filter = self._texture_filter()
        return TextTexture(texture, (width, height))

    def _resolve_text_font(self, image_font: Any, component: UIText, font_source: str, font_size: int) -> Any:
        try:
            if font_source == "asset":
                font_path = self._texture_reference_path(component.bitmap_font)
                if font_path and font_path.exists():
                    return image_font.truetype(str(font_path), font_size)
                if component.bitmap_font:
                    self.log(f"UI text font asset missing: {component.bitmap_font}")
                return image_font.load_default()
            if component.font_family and component.font_family != "System":
                return image_font.truetype(component.font_family, font_size)
            return image_font.load_default()
        except Exception as exc:
            label = component.bitmap_font if font_source == "asset" else component.font_family
            self.log(f"UI text font fallback for {label or font_source}: {_render_exception_text(exc)}")
            return image_font.load_default()

    def _load_mesh(self, entity: Entity, component: MeshRenderer) -> RenderMesh | None:
        cache_key = (component.mesh, component.submesh, component.material, component.shader, tuple(component.source_materials), tuple(component.material_slots))
        if cache_key in self._mesh_cache:
            return self._mesh_cache[cache_key]
        metadata, mesh_entry = resolve_model_mesh(self._metadata, component.mesh, component.submesh)
        if metadata is None:
            self.log(f"Missing mesh metadata: {component.mesh}")
            return None
        obj_path = self.project.root / metadata.source
        try:
            obj_mesh = parse_obj(obj_path)
        except Exception as exc:
            self.log(f"Could not parse {obj_path}: {exc}")
            return None
        group_name = str(mesh_entry.get("source_group")) if mesh_entry and mesh_entry.get("source_group") else component.submesh
        group = next((item for item in obj_mesh.groups if item.name == group_name), obj_mesh.groups[0] if obj_mesh.groups else None)
        if group is None:
            return None
        vertices = mesh_vertices_for_group(group)
        if not vertices:
            return None
        try:
            import struct

            buffer = self.ctx.buffer(struct.pack(f"{len(vertices)}f", *vertices))
        except Exception as exc:
            self.log(f"Could not build mesh buffer for {entity.name}: {exc}")
            return None
        source_materials = component.source_materials or _mesh_entry_materials(mesh_entry) or list(metadata.materials)
        material = component.material or (source_materials[0] if len(source_materials) == 1 else None) or (group.faces[0].material if group.faces else None)
        material_asset_path = self._material_slot_path(component, metadata, material)
        material_asset = self._load_material_asset(material_asset_path)
        shader = material_asset.shader if material_asset is not None else component.shader
        program = self._program_for(shader)
        try:
            vao = self._mesh_vertex_array(program, buffer)
        except Exception as exc:
            self.log(f"Could not bind mesh attributes for {entity.name}: {exc}")
            return None
        texture = self._texture_for(metadata, material, material_asset, material_asset_path)
        base_color = self._base_color_for(metadata, obj_mesh, material, material_asset)
        mesh = RenderMesh(
            vao=vao,
            buffer=buffer,
            vertex_count=len(vertices) // MESH_VERTEX_FLOATS,
            texture=texture,
            model_matrix=entity.transform.world_matrix(entity),
            shader=shader,
            base_color=base_color,
            material_properties=dict(material_asset.properties) if material_asset is not None else None,
            batches=self._mesh_material_batches(group, component, metadata, obj_mesh),
        )
        self._mesh_cache[cache_key] = mesh
        return mesh

    def _mesh_material_batches(
        self,
        group: Any,
        component: MeshRenderer,
        metadata: AssetMetadata,
        obj_mesh: Any,
    ) -> list[RenderMeshBatch] | None:
        materials = [component.material] if component.material else _group_materials(group)
        materials = [material for material in materials if material]
        if len(materials) <= 1:
            return None
        batches: list[RenderMeshBatch] = []
        for material in materials:
            vertices = mesh_vertices_for_group(group, material)
            if not vertices:
                continue
            try:
                import struct

                buffer = self.ctx.buffer(struct.pack(f"{len(vertices)}f", *vertices))
            except Exception as exc:
                self.log(f"Could not build material batch for {component.mesh}: {exc}")
                continue
            material_asset_path = self._material_slot_path(component, metadata, material)
            material_asset = self._load_material_asset(material_asset_path)
            shader = material_asset.shader if material_asset is not None else component.shader
            program = self._program_for(shader)
            try:
                vao = self._mesh_vertex_array(program, buffer)
            except Exception as exc:
                self.log(f"Could not bind material batch for {component.mesh}: {exc}")
                continue
            batches.append(
                RenderMeshBatch(
                    vao=vao,
                    buffer=buffer,
                    vertex_count=len(vertices) // MESH_VERTEX_FLOATS,
                    texture=self._texture_for(metadata, material, material_asset, material_asset_path),
                    shader=shader,
                    base_color=self._base_color_for(metadata, obj_mesh, material, material_asset),
                    material_properties=dict(material_asset.properties) if material_asset is not None else None,
                )
            )
        return batches or None

    def _draw_grid(self, view: list[float], projection: list[float]) -> None:
        camera = getattr(self, "_current_camera", RenderCamera(position=Vec3(), rotation=Vec3()))
        scene = getattr(self, "_current_scene", None)
        settings = {}
        if scene is not None:
            settings = self.project.editor_settings.get("scene_grid", {})
        if settings.get("enabled", True) is False:
            return
        for vertices, color in grid_line_batches(camera.position, settings):
            self._draw_lines(vertices, view, projection, color, _identity_matrix())

    def _draw_selection_outline(self, entity: Entity, view: list[float], projection: list[float]) -> None:
        previous_cull_face = getattr(self.ctx, "cull_face", "back")
        previous_depth_mask = getattr(self.ctx, "depth_mask", True)
        try:
            if hasattr(self.moderngl, "CULL_FACE"):
                self.ctx.enable(self.moderngl.CULL_FACE)
            if hasattr(self.ctx, "cull_face"):
                self.ctx.cull_face = "front"
            if hasattr(self.ctx, "depth_mask"):
                self.ctx.depth_mask = False
            for mesh_entity, component in self._selection_mesh_renderers(entity):
                mesh = self._load_mesh(mesh_entity, component)
                if mesh is None:
                    continue
                outline_vao = self._selection_outline_vao(mesh)
                if outline_vao is None:
                    continue
                self._set_uniform(self.selection_outline_program, "u_model", _mat4_bytes(mesh_entity.transform.world_matrix(mesh_entity)), write=True)
                self._set_uniform(self.selection_outline_program, "u_view", _mat4_bytes(view), write=True)
                self._set_uniform(self.selection_outline_program, "u_projection", _mat4_bytes(projection), write=True)
                self._set_uniform(self.selection_outline_program, "u_outline_width", 0.035)
                self._set_uniform(self.selection_outline_program, "u_color", (1.0, 0.84, 0.16))
                outline_vao.render()
                edge_mesh = self._selection_edge_mesh(mesh, component)
                if edge_mesh is not None:
                    self._draw_cached_line_mesh(edge_mesh, view, projection, (1.0, 0.84, 0.16), mesh_entity.transform.world_matrix(mesh_entity))
        except Exception as exc:
            self.log(f"Selection outline render failed: {exc}")
        finally:
            if hasattr(self.ctx, "depth_mask"):
                self.ctx.depth_mask = previous_depth_mask
            if hasattr(self.ctx, "cull_face"):
                self.ctx.cull_face = previous_cull_face
            if hasattr(self.moderngl, "CULL_FACE"):
                try:
                    self.ctx.disable(self.moderngl.CULL_FACE)
                except Exception:
                    pass
            self.ctx.enable(self.moderngl.DEPTH_TEST)

    def _selection_mesh_renderers(self, entity: Entity) -> list[tuple[Entity, MeshRenderer]]:
        renderers: list[tuple[Entity, MeshRenderer]] = []
        for candidate in entity.walk():
            if not entity_effectively_active(candidate):
                continue
            for component in candidate.components:
                if isinstance(component, MeshRenderer) and component.enabled and component.visible:
                    renderers.append((candidate, component))
        return renderers

    def _draw_component_gizmos(self, scene: Scene, view: list[float], projection: list[float], selected: Entity | None = None) -> None:
        for entity in scene.walk_active():
            for component in entity.components:
                if isinstance(component, Collider) and component.enabled:
                    color = (0.25, 0.8, 0.35) if not component.is_trigger else (0.25, 0.65, 1.0)
                    if component.shape == "mesh":
                        if _is_selected_or_child_of_selected(entity, selected):
                            self._draw_mesh_collider(entity, component, view, projection, color)
                    elif component.shape == "sphere":
                        center, radius = collider_sphere(entity, component, self.project)
                        self._draw_world_sphere(center, radius, view, projection, color)
                    else:
                        self._draw_world_bounds(collider_bounds(entity, component, self.project), view, projection, color)
                elif isinstance(component, CharacterController) and component.enabled:
                    self._draw_world_bounds(controller_bounds(entity, component), view, projection, (0.95, 0.55, 0.18))
                elif isinstance(component, SpawnPoint) and component.enabled:
                    self._draw_spawn_marker(entity, view, projection)
                elif selected is entity and isinstance(component, AudioSource) and component.enabled:
                    self._draw_audio_source_ranges(entity, component, view, projection)
                elif selected is entity and isinstance(component, Camera) and component.enabled:
                    self._draw_camera_frustum(entity, component, view, projection)
                elif selected is entity and isinstance(component, SpriteRenderer) and component.enabled:
                    self._draw_sprite_gizmo(entity, component, view, projection)
                elif selected is entity and isinstance(component, ParticleEmitter) and component.enabled:
                    self._draw_particle_emitter_gizmo(entity, component, view, projection)

    def _draw_audio_source_ranges(self, entity: Entity, source: AudioSource, view: list[float], projection: list[float]) -> None:
        center = _world_position(entity)
        min_radius, max_radius = audio_source_range_radii(source)
        if min_radius > 0.0:
            self._draw_world_sphere(center, min_radius, view, projection, (0.25, 0.85, 1.0))
        if max_radius > 0.0 and max_radius != min_radius:
            self._draw_world_sphere(center, max_radius, view, projection, (0.1, 0.45, 1.0))

    def _draw_sprite_gizmo(self, entity: Entity, component: SpriteRenderer, view: list[float], projection: list[float]) -> None:
        camera = getattr(self, "_current_camera", RenderCamera(position=Vec3(), rotation=Vec3()))
        quad = sprite_quad_vertices(entity, component, camera)
        if len(quad) < SPRITE_VERTEX_FLOATS * 6:
            return
        points = [quad[index:index + 3] for index in range(0, len(quad), SPRITE_VERTEX_FLOATS)]
        order = [0, 1, 2, 5, 0]
        vertices: list[float] = []
        for start, end in zip(order, order[1:]):
            vertices.extend(points[start])
            vertices.extend(points[end])
        self._draw_lines(vertices, view, projection, (1.0, 0.72, 0.25), _identity_matrix())

    def _draw_particle_emitter_gizmo(self, entity: Entity, component: ParticleEmitter, view: list[float], projection: list[float]) -> None:
        radius = max(0.1, component.start_size * 2.0)
        self._draw_world_sphere(_world_position(entity), radius, view, projection, (1.0, 0.35, 0.15))

    def _draw_camera_frustum(self, entity: Entity, camera: Camera, view: list[float], projection: list[float]) -> None:
        aspect = float(getattr(self, "_current_aspect", 16.0 / 9.0))
        vertices = camera_frustum_vertices(
            _world_position(entity),
            world_rotation(entity),
            camera.fov,
            camera.near,
            camera.far,
            aspect,
            (world_forward(entity), world_right(entity), world_up(entity)),
        )
        self._draw_lines(vertices, view, projection, (1.0, 0.78, 0.2), _identity_matrix())

    def _draw_spawn_marker(self, entity: Entity, view: list[float], projection: list[float]) -> None:
        p = _world_position(entity)
        size = 0.35
        vertices = [
            p.x - size, p.y, p.z, p.x + size, p.y, p.z,
            p.x, p.y, p.z - size, p.x, p.y, p.z + size,
            p.x, p.y, p.z, p.x, p.y + 1.0, p.z,
            p.x - size * 0.5, p.y + 0.75, p.z, p.x, p.y + 1.0, p.z,
            p.x + size * 0.5, p.y + 0.75, p.z, p.x, p.y + 1.0, p.z,
        ]
        self._draw_lines(vertices, view, projection, (0.8, 0.65, 1.0), _identity_matrix())

    def _draw_world_bounds(
        self,
        bounds: Any,
        view: list[float],
        projection: list[float],
        color: tuple[float, float, float],
    ) -> None:
        x0, y0, z0 = bounds.min.x, bounds.min.y, bounds.min.z
        x1, y1, z1 = bounds.max.x, bounds.max.y, bounds.max.z
        corners = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        vertices: list[float] = []
        for a, b in edges:
            vertices.extend(corners[a])
            vertices.extend(corners[b])
        self._draw_lines(vertices, view, projection, color, _identity_matrix())

    def _draw_mesh_collider(
        self,
        entity: Entity,
        collider: Collider,
        view: list[float],
        projection: list[float],
        color: tuple[float, float, float],
    ) -> None:
        renderer = next((component for component in entity.components if isinstance(component, MeshRenderer) and component.enabled), None)
        if renderer is None:
            return
        line_mesh = self._load_convex_collider_lines(renderer) if collider.convex else self._load_mesh_collider_lines(renderer)
        if line_mesh is None:
            return
        self._draw_cached_line_mesh(line_mesh, view, projection, color, entity.transform.world_matrix(entity))

    def _draw_world_sphere(
        self,
        center: Vec3,
        radius: float,
        view: list[float],
        projection: list[float],
        color: tuple[float, float, float],
    ) -> None:
        segments = 32
        vertices: list[float] = []
        for plane in ("xy", "xz", "yz"):
            for index in range(segments):
                a = radians(index / segments * 360.0)
                b = radians((index + 1) / segments * 360.0)
                if plane == "xy":
                    p0 = (center.x + cos(a) * radius, center.y + sin(a) * radius, center.z)
                    p1 = (center.x + cos(b) * radius, center.y + sin(b) * radius, center.z)
                elif plane == "xz":
                    p0 = (center.x + cos(a) * radius, center.y, center.z + sin(a) * radius)
                    p1 = (center.x + cos(b) * radius, center.y, center.z + sin(b) * radius)
                else:
                    p0 = (center.x, center.y + cos(a) * radius, center.z + sin(a) * radius)
                    p1 = (center.x, center.y + cos(b) * radius, center.z + sin(b) * radius)
                vertices.extend(p0)
                vertices.extend(p1)
        self._draw_lines(vertices, view, projection, color, _identity_matrix())

    def _draw_lines(self, vertices: list[float], view: list[float], projection: list[float], color: tuple[float, float, float], model: list[float]) -> None:
        if not vertices:
            return
        try:
            import struct

            buffer = self.ctx.buffer(struct.pack(f"{len(vertices)}f", *vertices))
            vao = self.ctx.vertex_array(self.line_program, [(buffer, "3f", "in_position")])
            self._set_uniform(self.line_program, "u_model", _mat4_bytes(model), write=True)
            self._set_uniform(self.line_program, "u_view", _mat4_bytes(view), write=True)
            self._set_uniform(self.line_program, "u_projection", _mat4_bytes(projection), write=True)
            self._set_uniform(self.line_program, "u_color", color)
            vao.render(mode=self.moderngl.LINES)
            buffer.release()
            vao.release()
        except Exception as exc:
            self.log(f"Line render failed: {exc}")

    def _load_mesh_collider_lines(self, component: MeshRenderer) -> RenderLineMesh | None:
        cache_key = (component.mesh, component.submesh)
        if cache_key in self._mesh_collider_line_cache:
            return self._mesh_collider_line_cache[cache_key]
        vertices = _mesh_collider_wire_vertices(self.project, component)
        if not vertices:
            return None
        try:
            import struct

            buffer = self.ctx.buffer(struct.pack(f"{len(vertices)}f", *vertices))
            vao = self.ctx.vertex_array(self.line_program, [(buffer, "3f", "in_position")])
        except Exception as exc:
            self.log(f"Could not cache mesh collider wireframe for {component.mesh}: {exc}")
            return None
        line_mesh = RenderLineMesh(vao=vao, buffer=buffer, vertex_count=len(vertices) // 3)
        self._mesh_collider_line_cache[cache_key] = line_mesh
        return line_mesh

    def _load_convex_collider_lines(self, component: MeshRenderer) -> RenderLineMesh | None:
        cache_key = (component.mesh, component.submesh)
        if cache_key in self._convex_collider_line_cache:
            return self._convex_collider_line_cache[cache_key]
        vertices = _convex_collider_wire_vertices(self.project, component)
        if not vertices:
            return None
        try:
            import struct

            buffer = self.ctx.buffer(struct.pack(f"{len(vertices)}f", *vertices))
            vao = self.ctx.vertex_array(self.line_program, [(buffer, "3f", "in_position")])
        except Exception as exc:
            self.log(f"Could not cache convex mesh collider wireframe for {component.mesh}: {exc}")
            return None
        line_mesh = RenderLineMesh(vao=vao, buffer=buffer, vertex_count=len(vertices) // 3)
        self._convex_collider_line_cache[cache_key] = line_mesh
        return line_mesh

    def _draw_cached_line_mesh(
        self,
        line_mesh: RenderLineMesh,
        view: list[float],
        projection: list[float],
        color: tuple[float, float, float],
        model: list[float],
    ) -> None:
        try:
            self._set_uniform(self.line_program, "u_model", _mat4_bytes(model), write=True)
            self._set_uniform(self.line_program, "u_view", _mat4_bytes(view), write=True)
            self._set_uniform(self.line_program, "u_projection", _mat4_bytes(projection), write=True)
            self._set_uniform(self.line_program, "u_color", color)
            line_mesh.vao.render(mode=self.moderngl.LINES)
        except Exception as exc:
            self.log(f"Cached line render failed: {exc}")

    def _selection_outline_vao(self, mesh: RenderMesh) -> Any | None:
        if mesh.outline_vao is not None:
            return mesh.outline_vao
        try:
            mesh.outline_vao = self.ctx.vertex_array(
                self.selection_outline_program,
                [(mesh.buffer, MESH_OUTLINE_LAYOUT, "in_position", "in_normal")],
            )
        except Exception as exc:
            self.log(f"Could not bind selection outline mesh: {exc}")
            return None
        return mesh.outline_vao

    def _selection_edge_mesh(self, mesh: RenderMesh, component: MeshRenderer) -> RenderLineMesh | None:
        if mesh.edge_mesh is not None:
            return mesh.edge_mesh
        vertices = _mesh_collider_wire_vertices(self.project, component)
        if not vertices:
            return None
        try:
            import struct

            buffer = self.ctx.buffer(struct.pack(f"{len(vertices)}f", *vertices))
            vao = self.ctx.vertex_array(self.line_program, [(buffer, "3f", "in_position")])
        except Exception as exc:
            self.log(f"Could not cache selection edge mesh for {component.mesh}: {exc}")
            return None
        mesh.edge_mesh = RenderLineMesh(vao=vao, buffer=buffer, vertex_count=len(vertices) // 3)
        return mesh.edge_mesh

    def _mesh_vertex_array(self, program: Any, buffer: Any) -> Any:
        bindings = [(buffer, MESH_VERTEX_LAYOUT, "in_position", "in_uv", "in_normal", "in_color")]
        try:
            return self.ctx.vertex_array(program, bindings, skip_errors=True)
        except TypeError:
            pass
        except Exception as exc:
            if "'in_position'" in str(exc) or "in_position" in str(exc):
                raise
            return self.ctx.vertex_array(program, [(buffer, MESH_POSITION_ONLY_LAYOUT, "in_position")])
        try:
            return self.ctx.vertex_array(program, bindings)
        except Exception as exc:
            if "'in_position'" in str(exc) or "in_position" in str(exc):
                raise
            return self.ctx.vertex_array(program, [(buffer, MESH_POSITION_ONLY_LAYOUT, "in_position")])

    def _program_for(self, shader: str | None) -> Any:
        if not shader:
            return self.program
        shader = normalize_shader_id(shader)
        if shader in self._program_cache:
            return self._program_cache[shader]
        shader_path = self.project.root / shader
        try:
            source = parse_shader(shader_path)
            program = self.ctx.program(vertex_shader=source.vertex, fragment_shader=source.fragment)
        except Exception as exc:
            self.log(f"Shader failed, using internal error shader for {shader}: {exc}")
            program = self.error_program
        self._program_cache[shader] = program
        return program

    def _compile_default_program(self) -> Any:
        shader_path = self.project.root / default_shader_id()
        if shader_path.exists():
            try:
                source = parse_shader(shader_path)
                return self.ctx.program(vertex_shader=source.vertex, fragment_shader=source.fragment)
            except Exception as exc:
                self.log(f"Builtin standard shader failed, using embedded fallback: {exc}")
        return self.ctx.program(vertex_shader=STANDARD_VERTEX_LIT_VERTEX_SHADER, fragment_shader=STANDARD_VERTEX_LIT_FRAGMENT_SHADER)

    def _compile_builtin_program(self, relative: str, fallback_vertex: str, fallback_fragment: str, label: str) -> Any:
        shader_path = self.project.root / relative
        if shader_path.exists():
            try:
                source = parse_shader(shader_path)
                return self.ctx.program(vertex_shader=source.vertex, fragment_shader=source.fragment)
            except Exception as exc:
                self.log(f"Builtin {label} shader failed, using embedded fallback: {_render_exception_text(exc)}")
        return self.ctx.program(vertex_shader=fallback_vertex, fragment_shader=fallback_fragment)

    def _set_uniform(self, program: Any, name: str, value: Any, write: bool = False) -> None:
        try:
            uniform = program[name]
        except KeyError:
            return
        if write:
            uniform.write(value)
        else:
            uniform.value = value

    def _base_color_for(
        self,
        metadata: AssetMetadata,
        obj_mesh: Any,
        material_name: str | None,
        material_asset: MaterialAsset | None = None,
    ) -> tuple[float, float, float]:
        if material_asset is not None and "u_base_color" in material_asset.properties:
            return _color3(material_asset.properties.get("u_base_color"))
        if not material_name:
            return (1.0, 1.0, 1.0)
        material_defs = metadata.settings.get("material_defs", {})
        material = material_defs.get(material_name, {})
        diffuse = material.get("diffuse_color") if isinstance(material, dict) else None
        if diffuse is None and material_name in obj_mesh.material_defs:
            diffuse = obj_mesh.material_defs[material_name].diffuse_color
        return _color3(diffuse)

    def _texture_for(
        self,
        metadata: AssetMetadata,
        material_name: str | None,
        material_asset: MaterialAsset | None = None,
        material_asset_path: Path | None = None,
    ) -> Any:
        if material_asset is not None:
            texture_path = self._material_texture_path(material_asset.textures.get("u_texture"), material_asset_path)
            if texture_path and texture_path.exists():
                return self._load_texture(texture_path)
        material_defs = metadata.settings.get("material_defs", {})
        texture_name = None
        if material_name and material_name in material_defs:
            texture_name = material_defs[material_name].get("diffuse_texture")
        if texture_name:
            path = (self.project.root / metadata.source).parent / str(texture_name)
            if path.exists():
                return self._load_texture(path)
        return self._default_texture()

    def _material_slot_path(self, component: MeshRenderer, metadata: AssetMetadata | None = None, material_name: str | None = None) -> Path | None:
        material_id = None
        source_materials = component.source_materials or (list(metadata.materials) if metadata is not None else [])
        if metadata is not None and material_name and material_name not in source_materials:
            source_materials = list(metadata.materials)
        if material_name and material_name in source_materials:
            index = source_materials.index(material_name)
            if index < len(component.material_slots):
                material_id = component.material_slots[index]
        if not material_id:
            material_id = next((slot for slot in component.material_slots if slot), None)
        if not material_id and metadata is not None and material_name:
            material_assets = metadata.settings.get("material_assets", {})
            if isinstance(material_assets, dict):
                material_id = material_assets.get(material_name)
        if not material_id:
            return None
        return resolve_material_reference(self.project.root, str(material_id))

    def _load_material_asset(self, path: Path | None) -> MaterialAsset | None:
        if path is None or not path.exists():
            return None
        try:
            return MaterialAsset.load(path)
        except Exception as exc:
            self.log(f"Could not load material {path}: {exc}")
            return None

    def _material_texture_path(self, texture_name: str | None, material_asset_path: Path | None) -> Path | None:
        if not texture_name:
            return None
        path = Path(str(texture_name))
        if path.is_absolute():
            return path
        candidates: list[Path] = []
        if str(texture_name).startswith(("assets/", "packages/")):
            candidates.append(self.project.root / texture_name)
        if material_asset_path is not None:
            candidates.append(material_asset_path.parent / texture_name)
            metadata = load_material_metadata(material_asset_path)
            source = metadata.settings.get("source", {}) if metadata else {}
            obj_source = source.get("obj") if isinstance(source, dict) else None
            if obj_source:
                candidates.append((self.project.root / str(obj_source)).parent / texture_name)
        candidates.append(self.project.assets_dir / texture_name)
        return next((candidate for candidate in candidates if candidate.exists()), candidates[0] if candidates else None)

    def _apply_material_properties(self, program: Any, properties: dict[str, Any] | None) -> None:
        if not properties:
            return
        for name, value in properties.items():
            if name == "u_base_color":
                continue
            self._set_uniform(program, name, _uniform_value(value))

    def _load_texture(self, path: Path) -> Any:
        if path in self._texture_cache:
            return self._texture_cache[path]
        try:
            from PIL import Image

            image = Image.open(path).convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            texture = self.ctx.texture(image.size, 4, image.tobytes())
            texture.filter = self._texture_filter()
            self._texture_cache[path] = texture
            return texture
        except Exception as exc:
            self.log(f"Could not load texture {path}: {exc}")
            return self._default_texture()

    def _default_texture(self) -> Any:
        if self._white_texture is None:
            self._white_texture = self.ctx.texture((1, 1), 4, bytes([255, 255, 255, 255]))
            self._white_texture.filter = self._texture_filter()
        return self._white_texture

    def _transparent_default_texture(self) -> Any:
        if self._transparent_texture is None:
            self._transparent_texture = self.ctx.texture((1, 1), 4, bytes([0, 0, 0, 0]))
            self._transparent_texture.filter = self._texture_filter()
        return self._transparent_texture

    def _texture_filter(self) -> tuple[Any, Any]:
        if str(self.project.render_settings.get("texture_filter", "three_point")) == "linear":
            linear = getattr(self.moderngl, "LINEAR", self.moderngl.NEAREST)
            return (linear, linear)
        return (self.moderngl.NEAREST, self.moderngl.NEAREST)


def _color3(value: Any) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (
                max(0.0, min(1.0, float(value[0]))),
                max(0.0, min(1.0, float(value[1]))),
                max(0.0, min(1.0, float(value[2]))),
            )
        except (TypeError, ValueError):
            pass
    return (1.0, 1.0, 1.0)


def _uniform_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _camera_from_entity(camera_entity: Entity | None) -> RenderCamera:
    if camera_entity is None:
        return RenderCamera(position=Vec3(0.0, 2.0, 8.0), rotation=Vec3(-15.0, 0.0, 0.0))
    fov = 60.0
    near = 0.1
    far = 500.0
    for component in camera_entity.components:
        if isinstance(component, Camera):
            fov = component.fov
            near = component.near
            far = component.far
            break
    return RenderCamera(
        world_position(camera_entity),
        world_rotation(camera_entity),
        fov,
        near,
        far,
        world_forward(camera_entity),
        world_right(camera_entity),
        world_up(camera_entity),
    )


def _light_kind_code(light: Light) -> int:
    return {"directional": 0, "point": 1, "spot": 2}.get(light.kind, 0)


def _texture_filter_code(value: str) -> int:
    return {"nearest": 0, "linear": 1, "three_point": 2}.get(value, 2)


def _internal_resolution(settings: dict[str, Any]) -> tuple[int, int]:
    resolution = settings.get("internal_resolution", [320, 240])
    if not isinstance(resolution, (list, tuple)) or len(resolution) < 2:
        return 320, 240
    try:
        return max(1, int(resolution[0])), max(1, int(resolution[1]))
    except (TypeError, ValueError):
        return 320, 240


def _scene_resolution_mode(scene: Scene) -> str:
    canvases = _canvas_entities(scene)
    if any(canvas.resolution_mode == "fixed" for _entity, canvas in canvases):
        return "fixed"
    return "auto"


def _game_render_size(scene: Scene, settings: dict[str, Any]) -> tuple[int, int]:
    for _entity, canvas in _canvas_entities(scene):
        if canvas.resolution_mode == "fixed":
            return _canvas_reference_size(canvas)
    return _internal_resolution(settings)


def _canvas_layout_size(canvas: Canvas, width: int, height: int) -> tuple[int, int]:
    if canvas.resolution_mode == "fixed":
        return _canvas_reference_size(canvas)
    return max(1, int(width)), max(1, int(height))


def _canvas_reference_size(canvas: Canvas) -> tuple[int, int]:
    return (
        max(1, int(canvas.reference_resolution.x)),
        max(1, int(canvas.reference_resolution.y)),
    )


def ui_layout_debug(scene: Scene, width: int, height: int, text_size_getter: Callable[[UIText], tuple[int, int]] | None = None) -> list[UILayoutDebugEntry]:
    entries: list[UILayoutDebugEntry] = []
    canvases = _canvas_entities(scene)
    roots: list[tuple[list[Entity], Canvas]] = (
        [(list(canvas_entity.walk()), canvas) for canvas_entity, canvas in sorted(canvases, key=lambda item: int(item[1].sort_order))]
        if canvases
        else [(list(scene.walk_active()), Canvas())]
    )
    for entities, canvas in roots:
        if not canvas.enabled:
            continue
        layout_width, layout_height = _canvas_layout_size(canvas, width, height)
        rects = _canvas_entity_rects(entities[0] if canvases and entities else None, layout_width, layout_height)
        for entity in entities:
            if not entity_effectively_active(entity):
                continue
            rect = rects.get(entity.id)
            if rect is None:
                continue
            image_rects: list[tuple[float, float, float, float]] = []
            text_rects: list[tuple[float, float, float, float]] = []
            for component in entity.components:
                if isinstance(component, UIImage) and component.enabled:
                    image_rects.append(image_rect_for_fill_mode(component, rect))
                elif isinstance(component, UIText) and component.enabled:
                    texture_size = text_size_getter(component) if text_size_getter is not None else _fallback_text_size(component)
                    text_rects.append(text_rect_with_aspect(component, rect, texture_size))
            entries.append(UILayoutDebugEntry(entity.id, entity.name, rect, tuple(image_rects), tuple(text_rects)))
    return entries


def _world_position(entity: Entity) -> Vec3:
    return world_position(entity)


def _distance_squared(a: Vec3, b: Vec3) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return dx * dx + dy * dy + dz * dz


def flipbook_uv_rect(columns: int, rows: int, fps: float, start: int, end: int, time_value: float) -> tuple[float, float, float, float]:
    columns = max(1, int(columns))
    rows = max(1, int(rows))
    frame_count = columns * rows
    start = max(0, min(frame_count - 1, int(start)))
    end = max(start, min(frame_count - 1, int(end) if int(end) > 0 else frame_count - 1))
    if fps > 0.0 and end > start:
        frame = start + (int(max(0.0, time_value) * fps) % (end - start + 1))
    else:
        frame = start
    col = frame % columns
    row = frame // columns
    u0 = col / columns
    u1 = (col + 1) / columns
    v0 = row / rows
    v1 = (row + 1) / rows
    return u0, v0, u1, v1


def sprite_quad_vertices(entity: Entity, component: SpriteRenderer, camera: RenderCamera, time_value: float = 0.0) -> list[float]:
    center = _world_position(entity)
    _forward, camera_right, camera_up = render_camera_basis(camera)
    if component.billboard == "none":
        matrix = entity.transform.world_matrix(entity)
        right = normalize(Vec3(matrix[0], matrix[4], matrix[8]))
        up = normalize(Vec3(matrix[1], matrix[5], matrix[9]))
    else:
        right = camera_right
        up = camera_up
    scale = world_scale(entity)
    width = max(0.001, float(component.size.x) * max(0.001, float(scale.x)))
    height = max(0.001, float(component.size.y) * max(0.001, float(scale.y)))
    return _quad_vertices_world(center, right, up, width, height, component.pivot, component.color, component.flipbook_columns, component.flipbook_rows, component.flipbook_fps, component.flipbook_start, component.flipbook_end, time_value)


def particle_quad_vertices(entity: Entity, emitter: ParticleEmitter, camera: RenderCamera) -> list[float]:
    _forward, right, up = render_camera_basis(camera)
    origin = _world_position(entity)
    vertices: list[float] = []
    for particle in emitter._runtime_particles:
        position = particle.get("position")
        if not isinstance(position, Vec3):
            continue
        center = Vec3(position.x, position.y, position.z)
        if emitter.local_space:
            center.x += origin.x
            center.y += origin.y
            center.z += origin.z
        age = float(particle.get("age", 0.0))
        lifetime = max(0.001, float(particle.get("lifetime", emitter.lifetime)))
        fade = max(0.0, min(1.0, 1.0 - age / lifetime))
        size = max(0.001, float(particle.get("size", emitter.start_size)))
        color = Vec3(fade, fade, fade)
        vertices.extend(_quad_vertices_world(center, right, up, size, size, Vec3(0.5, 0.5, 0.0), color, emitter.flipbook_columns, emitter.flipbook_rows, emitter.flipbook_fps, emitter.flipbook_start, emitter.flipbook_end, age))
    return vertices


def ui_quad_vertices(component: UIImage, width: int, height: int, time_value: float = 0.0, rect: tuple[float, float, float, float] | None = None) -> list[float]:
    x, y, w, h = image_rect_for_fill_mode(component, rect) if rect is not None else ui_rect(component.anchor, component.offset, component.size, component.pivot, width, height)
    u0, v0, u1, v1 = flipbook_uv_rect(component.flipbook_columns, component.flipbook_rows, component.flipbook_fps, component.flipbook_start, component.flipbook_end, time_value)
    if rect is not None and _image_fill_mode(component.fill_mode) == "fill":
        u0, v0, u1, v1 = image_fill_uv_rect(component, rect, (u0, v0, u1, v1))
    return _quad_vertices_2d(x, y, w, h, component.color, u0, v0, u1, v1)


def ui_text_vertices(component: UIText, width: int, height: int, rect: tuple[float, float, float, float] | None = None, texture_size: tuple[int, int] | None = None) -> list[float]:
    if rect is None:
        measured_width, measured_height = texture_size or _fallback_text_size(component)
        size = Vec3(float(measured_width), float(measured_height), 0.0)
        x, y, w, h = ui_rect(component.anchor, component.offset, size, component.pivot, width, height)
    else:
        x, y, w, h = text_rect_with_aspect(component, rect, texture_size or _fallback_text_size(component))
    return _quad_vertices_2d(x, y, w, h, component.color, 0.0, 0.0, 1.0, 1.0)


def ui_vertices_to_ndc(vertices: list[float], width: int, height: int) -> list[float]:
    if not vertices:
        return []
    safe_width = max(0.001, float(width))
    safe_height = max(0.001, float(height))
    converted = list(vertices)
    for index in range(0, len(converted), 8):
        converted[index] = (converted[index] / safe_width) * 2.0 - 1.0
        converted[index + 1] = 1.0 - (converted[index + 1] / safe_height) * 2.0
        converted[index + 2] = 0.0
    return converted


def ui_rect(anchor: str, offset: Vec3, size: Vec3, pivot: Vec3, width: int, height: int) -> tuple[float, float, float, float]:
    anchors = {
        "top-left": (0.0, 0.0),
        "top": (0.5, 0.0),
        "top-right": (1.0, 0.0),
        "left": (0.0, 0.5),
        "center": (0.5, 0.5),
        "right": (1.0, 0.5),
        "bottom-left": (0.0, 1.0),
        "bottom": (0.5, 1.0),
        "bottom-right": (1.0, 1.0),
    }
    ax, ay = anchors.get(anchor, anchors["center"])
    w = max(0.001, float(size.x))
    h = max(0.001, float(size.y))
    x = float(width) * ax + offset.x - w * pivot.x
    y = float(height) * ay + offset.y - h * pivot.y
    return x, y, w, h


def rect_transform_rect(rect: RectTransform, parent_rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    parent_x, parent_y, parent_w, parent_h = parent_rect
    x, y, w, h = ui_rect(rect.anchor, rect.offset, rect.size, rect.pivot, int(parent_w), int(parent_h))
    return parent_x + x, parent_y + y, w, h


def image_rect_for_fill_mode(component: UIImage, rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    mode = _image_fill_mode(component.fill_mode)
    if mode != "fit":
        return rect
    x, y, box_w, box_h = rect
    box_w = max(0.001, float(box_w))
    box_h = max(0.001, float(box_h))
    image_w = max(0.001, float(component.size.x))
    image_h = max(0.001, float(component.size.y))
    scale = min(box_w / image_w, box_h / image_h)
    w = max(0.001, image_w * scale)
    h = max(0.001, image_h * scale)
    return x + (box_w - w) * 0.5, y + (box_h - h) * 0.5, w, h


def image_fill_uv_rect(component: UIImage, rect: tuple[float, float, float, float], uv: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    _x, _y, box_w, box_h = rect
    box_w = max(0.001, float(box_w))
    box_h = max(0.001, float(box_h))
    image_w = max(0.001, float(component.size.x))
    image_h = max(0.001, float(component.size.y))
    scale = max(box_w / image_w, box_h / image_h)
    visible_w = box_w / max(0.001, image_w * scale)
    visible_h = box_h / max(0.001, image_h * scale)
    u0, v0, u1, v1 = uv
    du = (u1 - u0) * max(0.0, min(1.0, visible_w))
    dv = (v1 - v0) * max(0.0, min(1.0, visible_h))
    u_mid = (u0 + u1) * 0.5
    v_mid = (v0 + v1) * 0.5
    return u_mid - du * 0.5, v_mid - dv * 0.5, u_mid + du * 0.5, v_mid + dv * 0.5


def _image_fill_mode(value: str) -> str:
    mode = str(value or "stretch").lower()
    if mode == "simple":
        return "stretch"
    return mode if mode in {"stretch", "fit", "fill"} else "stretch"


def text_rect_with_aspect(component: UIText, rect: tuple[float, float, float, float], texture_size: tuple[int, int]) -> tuple[float, float, float, float]:
    x, y, w, h = rect
    box_w = max(4.0, float(w))
    box_h = max(4.0, float(h))
    tex_w = max(1.0, float(texture_size[0]))
    tex_h = max(1.0, float(texture_size[1]))
    scale = min(box_w / tex_w, box_h / tex_h)
    text_w = max(1.0, tex_w * scale)
    text_h = max(1.0, tex_h * scale)
    alignment = component.alignment if component.alignment in {"left", "center", "right"} else "center"
    if alignment == "left":
        text_x = x
    elif alignment == "right":
        text_x = x + box_w - text_w
    else:
        text_x = x + (box_w - text_w) * 0.5
    text_y = y + (box_h - text_h) * 0.5
    return text_x, text_y, text_w, text_h


def _fallback_text_size(component: UIText) -> tuple[int, int]:
    font_size = max(1.0, float(component.font_size))
    text = component.text or " "
    return max(1, int(len(text) * font_size * 0.6) + 4), max(1, int(font_size * 1.25) + 4)


def _quad_vertices_world(center: Vec3, right: Vec3, up: Vec3, width: float, height: float, pivot: Vec3, color: Vec3, columns: int, rows: int, fps: float, start: int, end: int, time_value: float) -> list[float]:
    u0, v0, u1, v1 = flipbook_uv_rect(columns, rows, fps, start, end, time_value)
    left = -pivot.x * width
    right_offset = (1.0 - pivot.x) * width
    top = (1.0 - pivot.y) * height
    bottom = -pivot.y * height
    corners = [
        (left, bottom, u0, v1),
        (right_offset, bottom, u1, v1),
        (right_offset, top, u1, v0),
        (left, bottom, u0, v1),
        (right_offset, top, u1, v0),
        (left, top, u0, v0),
    ]
    vertices: list[float] = []
    for x, y, u, v in corners:
        point = Vec3(
            center.x + right.x * x + up.x * y,
            center.y + right.y * x + up.y * y,
            center.z + right.z * x + up.z * y,
        )
        vertices.extend([point.x, point.y, point.z, u, v, color.x, color.y, color.z])
    return vertices


def _quad_vertices_2d(x: float, y: float, width: float, height: float, color: Vec3, u0: float, v0: float, u1: float, v1: float) -> list[float]:
    corners = [
        (x, y + height, u0, v0),
        (x + width, y + height, u1, v0),
        (x + width, y, u1, v1),
        (x, y + height, u0, v0),
        (x + width, y, u1, v1),
        (x, y, u0, v1),
    ]
    vertices: list[float] = []
    for px, py, u, v in corners:
        vertices.extend([px, py, 0.0, u, v, color.x, color.y, color.z])
    return vertices


def _canvas_entities(scene: Scene) -> list[tuple[Entity, Canvas]]:
    canvases: list[tuple[Entity, Canvas]] = []
    for entity in scene.walk_active():
        for component in entity.components:
            if isinstance(component, Canvas) and component.enabled:
                canvases.append((entity, component))
    if canvases:
        return canvases
    return []


def _canvas_entity_rects(canvas_entity: Entity | None, width: int, height: int) -> dict[str, tuple[float, float, float, float]]:
    if canvas_entity is None:
        return {}
    root_rect = (0.0, 0.0, float(max(width, 1)), float(max(height, 1)))
    rects: dict[str, tuple[float, float, float, float]] = {}

    def visit(entity: Entity, parent_rect: tuple[float, float, float, float]) -> None:
        current_rect = rect_transform_rect(entity.rect_transform, parent_rect) if entity.rect_transform is not None else parent_rect
        if entity.rect_transform is not None:
            rects[entity.id] = current_rect
        for child in entity.children:
            visit(child, current_rect)

    visit(canvas_entity, root_rect)
    return rects


def _is_selected_or_child_of_selected(entity: Entity, selected: Entity | None) -> bool:
    current: Entity | None = entity
    while current is not None:
        if current is selected:
            return True
        current = current.parent
    return False


def audio_source_range_radii(source: AudioSource) -> tuple[float, float]:
    min_radius = max(0.0, float(source.min_distance))
    max_radius = max(min_radius, float(source.max_distance))
    return min_radius, max_radius


def camera_frustum_vertices(
    position: Vec3,
    rotation: Vec3,
    fov: float,
    near: float,
    far: float,
    aspect: float,
    basis: tuple[Vec3, Vec3, Vec3] | None = None,
) -> list[float]:
    near = max(0.001, float(near))
    far = max(near + 0.001, float(far))
    aspect = max(0.01, float(aspect))
    forward, right, up = basis or camera_basis(rotation)

    def plane_corners(distance: float) -> list[Vec3]:
        half_height = tan(radians(max(1.0, min(179.0, float(fov)))) * 0.5) * distance
        half_width = half_height * aspect
        center = Vec3(
            position.x + forward.x * distance,
            position.y + forward.y * distance,
            position.z + forward.z * distance,
        )
        return [
            Vec3(center.x - right.x * half_width + up.x * half_height, center.y - right.y * half_width + up.y * half_height, center.z - right.z * half_width + up.z * half_height),
            Vec3(center.x + right.x * half_width + up.x * half_height, center.y + right.y * half_width + up.y * half_height, center.z + right.z * half_width + up.z * half_height),
            Vec3(center.x + right.x * half_width - up.x * half_height, center.y + right.y * half_width - up.y * half_height, center.z + right.z * half_width - up.z * half_height),
            Vec3(center.x - right.x * half_width - up.x * half_height, center.y - right.y * half_width - up.y * half_height, center.z - right.z * half_width - up.z * half_height),
        ]

    near_corners = plane_corners(near)
    far_corners = plane_corners(far)
    vertices: list[float] = []
    for corners in (near_corners, far_corners):
        for start_index, end_index in ((0, 1), (1, 2), (2, 3), (3, 0)):
            vertices.extend(_vec3_values(corners[start_index]))
            vertices.extend(_vec3_values(corners[end_index]))
    for index in range(4):
        vertices.extend(_vec3_values(near_corners[index]))
        vertices.extend(_vec3_values(far_corners[index]))
    return vertices


def cloud_dome_vertices(height: float, radius: float, segments: int = 48) -> list[float]:
    height = max(0.1, float(height))
    segments = max(8, int(segments))
    radius = max(0.1, float(radius))
    horizon_y = height * 0.18
    shoulder_y = height * 0.72
    apex_y = height
    shoulder_radius = radius * 0.46
    apex_radius = radius * 0.08
    vertices: list[float] = []
    for index in range(segments):
        angle0 = 2.0 * pi * (index / segments)
        angle1 = 2.0 * pi * ((index + 1) / segments)
        horizon0 = Vec3(cos(angle0) * radius, horizon_y, sin(angle0) * radius)
        horizon1 = Vec3(cos(angle1) * radius, horizon_y, sin(angle1) * radius)
        shoulder0 = Vec3(cos(angle0) * shoulder_radius, shoulder_y, sin(angle0) * shoulder_radius)
        shoulder1 = Vec3(cos(angle1) * shoulder_radius, shoulder_y, sin(angle1) * shoulder_radius)
        apex0 = Vec3(cos(angle0) * apex_radius, apex_y, sin(angle0) * apex_radius)
        apex1 = Vec3(cos(angle1) * apex_radius, apex_y, sin(angle1) * apex_radius)
        for point in (horizon0, horizon1, shoulder1, horizon0, shoulder1, shoulder0, shoulder0, shoulder1, apex1, shoulder0, apex1, apex0):
            vertices.extend(_vec3_values(point))
    return vertices


def cloud_plane_vertices(camera: RenderCamera, height: float, segments: int = 48) -> list[float]:
    height = max(0.1, float(height))
    radius = max(240.0, height * 8.0, camera.far * 0.75)
    vertices = cloud_dome_vertices(height, radius, segments)
    translated: list[float] = []
    for index, value in enumerate(vertices):
        axis = index % 3
        if axis == 0:
            translated.append(value + camera.position.x)
        elif axis == 1:
            translated.append(value + camera.position.y)
        else:
            translated.append(value + camera.position.z)
    return translated


def _vec3_values(value: Vec3) -> tuple[float, float, float]:
    return (value.x, value.y, value.z)


def grid_line_batches(camera_position: Vec3, settings: dict[str, Any]) -> list[tuple[list[float], tuple[float, float, float]]]:
    spacing = max(0.01, float(settings.get("spacing", 1.0)))
    radius = max(spacing, float(settings.get("radius", 40.0)))
    fade_start = max(0.0, float(settings.get("fade_start", radius * 0.45)))
    fade_end = max(fade_start + spacing, float(settings.get("fade_end", radius)))
    origin_x = round(camera_position.x / spacing) * spacing
    origin_z = round(camera_position.z / spacing) * spacing
    line_count = int(radius / spacing)
    batches: list[tuple[list[float], tuple[float, float, float]]] = []

    for offset in range(-line_count, line_count + 1):
        x = origin_x + offset * spacing
        z = origin_z + offset * spacing
        fade = _grid_fade(abs(offset * spacing), fade_start, fade_end)
        if fade <= 0.02:
            continue
        color = (0.34 * fade, 0.38 * fade, 0.42 * fade)
        batches.append((
            [
                x, 0.0, origin_z - radius,
                x, 0.0, origin_z + radius,
                origin_x - radius, 0.0, z,
                origin_x + radius, 0.0, z,
            ],
            color,
        ))

    batches.append((
        [-radius, 0.002, 0.0, radius, 0.002, 0.0],
        (0.45, 0.18, 0.18),
    ))
    batches.append((
        [0.0, 0.002, -radius, 0.0, 0.002, radius],
        (0.18, 0.25, 0.55),
    ))
    return batches


def _grid_fade(distance: float, fade_start: float, fade_end: float) -> float:
    if distance <= fade_start:
        return 1.0
    if distance >= fade_end:
        return 0.0
    t = (distance - fade_start) / max(fade_end - fade_start, 0.001)
    return max(0.0, min(1.0, 1.0 - t))


def _view_matrix(camera: RenderCamera) -> list[float]:
    forward, right, up = render_camera_basis(camera)
    position = camera.position
    return [
        right.x, right.y, right.z, -dot(right, position),
        up.x, up.y, up.z, -dot(up, position),
        -forward.x, -forward.y, -forward.z, dot(forward, position),
        0.0, 0.0, 0.0, 1.0,
    ]


def camera_basis(rotation: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    pitch = radians(rotation.x)
    yaw = radians(rotation.y)
    forward = normalize(Vec3(sin(yaw) * cos(pitch), sin(pitch), -cos(yaw) * cos(pitch)))
    right = normalize(Vec3(cos(yaw), 0.0, sin(yaw)))
    up = normalize(cross(right, forward))
    return forward, right, up


def render_camera_basis(camera: RenderCamera) -> tuple[Vec3, Vec3, Vec3]:
    if camera.forward is not None and camera.right is not None and camera.up is not None:
        return camera.forward.normalized(), camera.right.normalized(), camera.up.normalized()
    return camera_basis(camera.rotation)


def dot(a: Vec3, b: Vec3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def normalize(v: Vec3) -> Vec3:
    length = max((v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5, 0.000001)
    return Vec3(v.x / length, v.y / length, v.z / length)


def _perspective_matrix(fov_degrees: float, aspect: float, near: float, far: float) -> list[float]:
    from math import radians, tan

    f = 1.0 / tan(radians(fov_degrees) / 2.0)
    return [
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) / (near - far), (2 * far * near) / (near - far),
        0.0, 0.0, -1.0, 0.0,
    ]


def _orthographic_matrix(left: float, right: float, bottom: float, top: float, near: float, far: float) -> list[float]:
    width = max(0.000001, right - left)
    height = max(0.000001, top - bottom)
    depth = max(0.000001, far - near)
    return [
        2.0 / width, 0.0, 0.0, -(right + left) / width,
        0.0, 2.0 / height, 0.0, -(top + bottom) / height,
        0.0, 0.0, -2.0 / depth, -(far + near) / depth,
        0.0, 0.0, 0.0, 1.0,
    ]


def _identity_matrix() -> list[float]:
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _transform_point(matrix: list[float], point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3],
        matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7],
        matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11],
    )


def _project_point(
    point: tuple[float, float, float],
    view: list[float],
    projection: list[float],
    width: int,
    height: int,
) -> tuple[float, float, float] | None:
    view_point = _mul_mat4_vec4(view, (point[0], point[1], point[2], 1.0))
    clip = _mul_mat4_vec4(projection, view_point)
    if abs(clip[3]) < 0.000001:
        return None
    ndc = (clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3])
    if ndc[0] < -1.25 or ndc[0] > 1.25 or ndc[1] < -1.25 or ndc[1] > 1.25:
        return None
    screen_x = (ndc[0] * 0.5 + 0.5) * width
    screen_y = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
    depth = ndc[2] * 0.5 + 0.5
    return screen_x, screen_y, depth


def _screen_ray(camera: RenderCamera, width: int, height: int, screen_x: float, screen_y: float) -> tuple[Vec3, Vec3]:
    width = max(int(width), 1)
    height = max(int(height), 1)
    aspect = max(width / height, 0.01)
    ndc_x = (float(screen_x) / width) * 2.0 - 1.0
    ndc_y = 1.0 - (float(screen_y) / height) * 2.0
    forward, right, up = render_camera_basis(camera)
    half_height = tan(radians(camera.fov) * 0.5)
    direction = normalize(
        Vec3(
            forward.x + right.x * ndc_x * half_height * aspect + up.x * ndc_y * half_height,
            forward.y + right.y * ndc_x * half_height * aspect + up.y * ndc_y * half_height,
            forward.z + right.z * ndc_x * half_height * aspect + up.z * ndc_y * half_height,
        )
    )
    return camera.position, direction


def _mesh_collider_wire_vertices(project: Project, component: MeshRenderer) -> list[float]:
    metadata, mesh = resolve_model_mesh(_metadata_by_id(project), component.mesh, component.submesh)
    wireframe = mesh.get("wireframe") if mesh else None
    vertices = wireframe.get("vertices") if isinstance(wireframe, dict) else None
    if isinstance(vertices, list):
        return [float(value) for value in vertices]
    vertices: list[float] = []
    for triangle in mesh_triangles(project, component):
        for start, end in [(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])]:
            vertices.extend(start)
            vertices.extend(end)
    return vertices


def _convex_collider_wire_vertices(project: Project, component: MeshRenderer) -> list[float]:
    hull = convex_hull(project, component)
    if hull is None:
        return []
    vertices: list[float] = []
    seen: set[tuple[tuple[float, float, float], tuple[float, float, float]]] = set()
    for triangle in hull.triangles:
        for start, end in [(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])]:
            a = (round(start.x, 5), round(start.y, 5), round(start.z, 5))
            b = (round(end.x, 5), round(end.y, 5), round(end.z, 5))
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            vertices.extend([start.x, start.y, start.z, end.x, end.y, end.z])
    return vertices


def _ray_triangle_intersection(origin: Vec3, direction: Vec3, triangle: tuple[Vec3, Vec3, Vec3]) -> float | None:
    epsilon = 0.000001
    v0, v1, v2 = triangle
    edge1 = Vec3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z)
    edge2 = Vec3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z)
    h = cross(direction, edge2)
    a = dot(edge1, h)
    if -epsilon < a < epsilon:
        return None
    f = 1.0 / a
    s = Vec3(origin.x - v0.x, origin.y - v0.y, origin.z - v0.z)
    u = f * dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = cross(s, edge1)
    v = f * dot(direction, q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * dot(edge2, q)
    return t if t > epsilon else None


def _mul_mat4_vec4(matrix: list[float], vector: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = vector
    return (
        matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3] * w,
        matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7] * w,
        matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11] * w,
        matrix[12] * x + matrix[13] * y + matrix[14] * z + matrix[15] * w,
    )


def _mat4_bytes(matrix: list[float]) -> bytes:
    import struct

    # Engine matrices are row-major; OpenGL uniforms are consumed column-major.
    column_major = [matrix[row * 4 + col] for col in range(4) for row in range(4)]
    return struct.pack("16f", *column_major)


def _group_materials(group: Any) -> list[str]:
    materials: list[str] = []
    for face in getattr(group, "faces", []):
        material = getattr(face, "material", None)
        if material and material not in materials:
            materials.append(material)
    return materials


def _mesh_entry_materials(mesh_entry: dict[str, Any] | None) -> list[str]:
    if not mesh_entry:
        return []
    values = mesh_entry.get("material_slots", [])
    return [str(value) for value in values if value]


def _metadata_by_id(project: Project) -> dict[str, AssetMetadata]:
    metadata_by_id: dict[str, AssetMetadata] = {}
    for metadata_path in discover_metadata(project.assets_dir):
        try:
            metadata = AssetMetadata.load(metadata_path)
        except Exception:
            continue
        metadata_by_id[metadata.id] = metadata
    return metadata_by_id
