from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan
from pathlib import Path
from typing import Any

from p64.engine.assets import AssetMetadata, discover_metadata, resolve_model_mesh
from p64.engine.collision import collider_bounds, collider_sphere, controller_bounds
from p64.engine.components import Camera, CharacterController, Collider, Light, MeshRenderer, SpawnPoint
from p64.engine.entity import Entity
from p64.engine.material import MaterialAsset, load_material_metadata, resolve_material_reference
from p64.engine.math import Vec3
from p64.engine.mesh_geometry import clear_convex_hull_cache, convex_hull, mesh_triangles, transform_triangle
from p64.engine.obj import mesh_vertices_for_group, parse_obj
from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.engine.shader import default_shader_id, normalize_shader_id, parse_shader
from p64.renderer.shaders import ERROR_FRAGMENT_SHADER, ERROR_VERTEX_SHADER, STANDARD_VERTEX_LIT_FRAGMENT_SHADER, STANDARD_VERTEX_LIT_VERTEX_SHADER


MAX_SHADER_LIGHTS = 8
MESH_VERTEX_FLOATS = 11
MESH_VERTEX_LAYOUT = "3f 2f 3f 3f"
MESH_POSITION_ONLY_LAYOUT = "3f 32x"
MESH_OUTLINE_LAYOUT = "3f 8x 3f 12x"


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
class RenderCamera:
    position: Vec3
    rotation: Vec3
    fov: float = 60.0
    near: float = 0.1
    far: float = 500.0


class SceneRenderer:
    def __init__(self, ctx: Any, project: Project, log: Any | None = None) -> None:
        import moderngl

        self.ctx = ctx
        self.moderngl = moderngl
        self.project = project
        self.log = log or (lambda message: None)
        self.program = self._compile_default_program()
        self.error_program = ctx.program(vertex_shader=ERROR_VERTEX_SHADER, fragment_shader=ERROR_FRAGMENT_SHADER)
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
        self._program_cache: dict[str | None, Any] = {None: self.program, "__p64_error__": self.error_program}
        self._metadata: dict[str, AssetMetadata] = {}
        self._texture_cache: dict[Path, Any] = {}
        self._mesh_cache: dict[tuple[str, str | None, str | None, str | None, tuple[str, ...], tuple[str | None, ...]], RenderMesh] = {}
        self._mesh_collider_line_cache: dict[tuple[str, str | None], RenderLineMesh] = {}
        self._convex_collider_line_cache: dict[tuple[str, str | None], RenderLineMesh] = {}
        self._white_texture = None
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
        clear_convex_hull_cache(self.project)
        self._program_cache = {None: self.program, "__p64_error__": self.error_program}

    def render(
        self,
        scene: Scene,
        width: int,
        height: int,
        camera: RenderCamera | None = None,
        selected_entity_id: str | None = None,
        show_grid: bool = True,
    ) -> None:
        self.ctx.viewport = (0, 0, max(width, 1), max(height, 1))
        self.ctx.clear(0.22, 0.27, 0.33, 1.0)
        self.ctx.enable(self.moderngl.DEPTH_TEST)
        render_camera = camera or _camera_from_entity(scene.active_camera())
        view = _view_matrix(render_camera)
        projection = _perspective_matrix(render_camera.fov, max(width / max(height, 1), 0.01), render_camera.near, render_camera.far)
        self._current_scene = scene
        self._current_camera = render_camera
        self._current_view = view
        self._current_projection = projection
        submitted = 0
        for entity in scene.walk():
            if not entity.active:
                continue
            for component in entity.components:
                if isinstance(component, MeshRenderer) and component.enabled and component.visible:
                    if self._draw_mesh(entity, component):
                        submitted += 1
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
        for entity in scene.walk():
            if not entity.active:
                continue
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
        self._set_uniform(program, "u_view", _mat4_bytes(view), write=True)
        self._set_uniform(program, "u_projection", _mat4_bytes(projection), write=True)
        self._set_uniform(program, "u_color_levels", float(scene.render_settings.get("color_levels", 32)))
        self._set_uniform(program, "u_texture_filter", _texture_filter_code(str(scene.render_settings.get("texture_filter", "three_point"))))
        self._set_uniform(program, "u_dithering_enabled", bool(scene.render_settings.get("dithering", True)))
        self._apply_light_uniforms(program, scene)
        self._apply_fog_uniforms(program, scene, camera)

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
                direction = camera_basis(entity.transform.rotation)[0]
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
        center = entity.transform.position
        scale = entity.transform.scale
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
                self._apply_material_properties(batch_program, batch.material_properties)
                batch.vao.render()
            return True
        mesh.texture.use(location=0)
        self._set_uniform(program, "u_texture", 0)
        self._set_uniform(program, "u_base_color", mesh.base_color)
        self._apply_material_properties(program, mesh.material_properties)
        mesh.vao.render()
        return True

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
            if not candidate.active:
                continue
            for component in candidate.components:
                if isinstance(component, MeshRenderer) and component.enabled and component.visible:
                    renderers.append((candidate, component))
        return renderers

    def _draw_component_gizmos(self, scene: Scene, view: list[float], projection: list[float], selected: Entity | None = None) -> None:
        for entity in scene.walk():
            if not entity.active:
                continue
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

    def _draw_spawn_marker(self, entity: Entity, view: list[float], projection: list[float]) -> None:
        p = entity.transform.position
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
    return RenderCamera(camera_entity.transform.position, camera_entity.transform.rotation, fov, near, far)


def _light_kind_code(light: Light) -> int:
    return {"directional": 0, "point": 1, "spot": 2}.get(light.kind, 0)


def _texture_filter_code(value: str) -> int:
    return {"nearest": 0, "linear": 1, "three_point": 2}.get(value, 2)


def _world_position(entity: Entity) -> Vec3:
    matrix = entity.transform.world_matrix(entity)
    return Vec3(matrix[3], matrix[7], matrix[11])


def _is_selected_or_child_of_selected(entity: Entity, selected: Entity | None) -> bool:
    current: Entity | None = entity
    while current is not None:
        if current is selected:
            return True
        current = current.parent
    return False


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
    forward, right, up = camera_basis(camera.rotation)
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
    forward, right, up = camera_basis(camera.rotation)
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
