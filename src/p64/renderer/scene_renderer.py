from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path
from typing import Any

from p64.engine.assets import AssetMetadata, discover_metadata
from p64.engine.components import Camera, MeshRenderer
from p64.engine.entity import Entity
from p64.engine.math import Vec3
from p64.engine.obj import mesh_vertices_for_group, parse_obj
from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.engine.shader import parse_shader
from p64.renderer.shaders import N64_FRAGMENT_SHADER, N64_VERTEX_SHADER


@dataclass
class RenderMesh:
    vao: Any
    vertex_count: int
    texture: Any
    model_matrix: list[float]
    shader: str | None


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
        self.program = ctx.program(vertex_shader=N64_VERTEX_SHADER, fragment_shader=N64_FRAGMENT_SHADER)
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
        self._program_cache: dict[str | None, Any] = {None: self.program}
        self._metadata: dict[str, AssetMetadata] = {}
        self._texture_cache: dict[Path, Any] = {}
        self._mesh_cache: dict[tuple[str, str | None, str | None, str | None], RenderMesh] = {}
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
        self._program_cache = {None: self.program}

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
        if show_grid:
            self._draw_grid(view, projection)
        if selected_entity_id:
            selected = scene.find(selected_entity_id)
            if selected:
                self._draw_selection_bounds(selected, view, projection)

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
        view = _view_matrix(render_camera)
        projection = _perspective_matrix(render_camera.fov, max(width / max(height, 1), 0.01), render_camera.near, render_camera.far)
        best_id: str | None = None
        best_distance = 48.0
        for entity in scene.walk():
            if not entity.active:
                continue
            for component in entity.components:
                if not isinstance(component, MeshRenderer) or not component.enabled or not component.visible:
                    continue
                bounds = self._mesh_bounds(component)
                if bounds is None:
                    continue
                mins, maxs = bounds
                center = (
                    (mins[0] + maxs[0]) * 0.5,
                    (mins[1] + maxs[1]) * 0.5,
                    (mins[2] + maxs[2]) * 0.5,
                )
                world = _transform_point(entity.transform.world_matrix(entity), center)
                projected = _project_point(world, view, projection, width, height)
                if projected is None:
                    continue
                px, py, depth = projected
                if depth < 0.0 or depth > 1.0:
                    continue
                distance = ((screen_x - px) ** 2 + (screen_y - py) ** 2) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best_id = entity.id
        return best_id

    def _apply_common_uniforms(self, program: Any, scene: Scene, camera: RenderCamera, view: list[float], projection: list[float]) -> None:
        self._set_uniform(program, "u_view", _mat4_bytes(view), write=True)
        self._set_uniform(program, "u_projection", _mat4_bytes(projection), write=True)
        self._set_uniform(program, "u_color_levels", float(scene.render_settings.get("color_levels", 32)))
        self._apply_fog_uniforms(program, scene, camera)

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
        program = self._program_for(component.shader)
        render_camera = getattr(self, "_current_camera", _camera_from_entity(None))
        view = getattr(self, "_current_view", _view_matrix(render_camera))
        projection = getattr(self, "_current_projection", _perspective_matrix(render_camera.fov, 1.0, render_camera.near, render_camera.far))
        scene = getattr(self, "_current_scene", None)
        if scene is not None:
            self._apply_common_uniforms(program, scene, render_camera, view, projection)
        self._set_uniform(program, "u_model", _mat4_bytes(entity.transform.world_matrix(entity)), write=True)
        mesh.texture.use(location=0)
        self._set_uniform(program, "u_texture", 0)
        mesh.vao.render()
        return True

    def _load_mesh(self, entity: Entity, component: MeshRenderer) -> RenderMesh | None:
        cache_key = (component.mesh, component.submesh, component.material, component.shader)
        if cache_key in self._mesh_cache:
            return self._mesh_cache[cache_key]
        metadata = self._metadata.get(component.mesh)
        if metadata is None:
            self.log(f"Missing mesh metadata: {component.mesh}")
            return None
        obj_path = self.project.root / metadata.source
        try:
            obj_mesh = parse_obj(obj_path)
        except Exception as exc:
            self.log(f"Could not parse {obj_path}: {exc}")
            return None
        group = next((item for item in obj_mesh.groups if item.name == component.submesh), obj_mesh.groups[0] if obj_mesh.groups else None)
        if group is None:
            return None
        vertices = mesh_vertices_for_group(group)
        if not vertices:
            return None
        try:
            import numpy as np

            buffer = self.ctx.buffer(np.array(vertices, dtype="f4").tobytes())
        except Exception as exc:
            self.log(f"Could not build mesh buffer for {entity.name}: {exc}")
            return None
        program = self._program_for(component.shader)
        try:
            vao = self.ctx.vertex_array(
                program,
                [(buffer, "3f 2f 3f", "in_position", "in_uv", "in_normal")],
            )
        except Exception as exc:
            self.log(f"Could not bind mesh attributes for {entity.name}: {exc}")
            return None
        material = component.material or (group.faces[0].material if group.faces else None)
        texture = self._texture_for(metadata, material)
        mesh = RenderMesh(
            vao=vao,
            vertex_count=len(vertices) // 8,
            texture=texture,
            model_matrix=entity.transform.world_matrix(entity),
            shader=component.shader,
        )
        self._mesh_cache[cache_key] = mesh
        return mesh

    def _draw_grid(self, view: list[float], projection: list[float]) -> None:
        vertices: list[float] = []
        extent = 20
        for index in range(-extent, extent + 1):
            vertices.extend([float(index), 0.0, float(-extent), float(index), 0.0, float(extent)])
            vertices.extend([float(-extent), 0.0, float(index), float(extent), 0.0, float(index)])
        self._draw_lines(vertices, view, projection, (0.34, 0.38, 0.42), _identity_matrix())

    def _draw_selection_bounds(self, entity: Entity, view: list[float], projection: list[float]) -> None:
        for component in entity.components:
            if not isinstance(component, MeshRenderer):
                continue
            bounds = self._mesh_bounds(component)
            if bounds is None:
                continue
            mins, maxs = bounds
            x0, y0, z0 = mins
            x1, y1, z1 = maxs
            corners = [
                (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
            ]
            edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
            vertices: list[float] = []
            for a, b in edges:
                vertices.extend(corners[a])
                vertices.extend(corners[b])
            self._draw_lines(vertices, view, projection, (1.0, 0.88, 0.18), entity.transform.world_matrix(entity))

    def _draw_lines(self, vertices: list[float], view: list[float], projection: list[float], color: tuple[float, float, float], model: list[float]) -> None:
        if not vertices:
            return
        try:
            import numpy as np

            buffer = self.ctx.buffer(np.array(vertices, dtype="f4").tobytes())
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

    def _mesh_bounds(self, component: MeshRenderer) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        metadata = self._metadata.get(component.mesh)
        if not metadata:
            return None
        mesh = parse_obj(self.project.root / metadata.source)
        group = next((item for item in mesh.groups if item.name == component.submesh), mesh.groups[0] if mesh.groups else None)
        if not group:
            return None
        positions = [vertex.position for face in group.faces for vertex in face.vertices]
        if not positions:
            return None
        mins = tuple(min(position[index] for position in positions) for index in range(3))
        maxs = tuple(max(position[index] for position in positions) for index in range(3))
        return mins, maxs

    def _program_for(self, shader: str | None) -> Any:
        if not shader:
            return self.program
        if shader in self._program_cache:
            return self._program_cache[shader]
        shader_path = self.project.root / shader
        try:
            source = parse_shader(shader_path)
            program = self.ctx.program(vertex_shader=source.vertex, fragment_shader=source.fragment)
        except Exception as exc:
            self.log(f"Shader failed, using built-in shader for {shader}: {exc}")
            program = self.program
        self._program_cache[shader] = program
        return program

    def _set_uniform(self, program: Any, name: str, value: Any, write: bool = False) -> None:
        try:
            uniform = program[name]
        except KeyError:
            return
        if write:
            uniform.write(value)
        else:
            uniform.value = value

    def _texture_for(self, metadata: AssetMetadata, material_name: str | None) -> Any:
        material_defs = metadata.settings.get("material_defs", {})
        texture_name = None
        if material_name and material_name in material_defs:
            texture_name = material_defs[material_name].get("diffuse_texture")
        if texture_name:
            path = (self.project.root / metadata.source).parent / str(texture_name)
            if path.exists():
                return self._load_texture(path)
        return self._default_texture()

    def _load_texture(self, path: Path) -> Any:
        if path in self._texture_cache:
            return self._texture_cache[path]
        try:
            from PIL import Image

            image = Image.open(path).convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            texture = self.ctx.texture(image.size, 4, image.tobytes())
            texture.filter = (self.moderngl.NEAREST, self.moderngl.NEAREST)
            self._texture_cache[path] = texture
            return texture
        except Exception as exc:
            self.log(f"Could not load texture {path}: {exc}")
            return self._default_texture()

    def _default_texture(self) -> Any:
        if self._white_texture is None:
            self._white_texture = self.ctx.texture((1, 1), 4, bytes([255, 255, 255, 255]))
            self._white_texture.filter = (self.moderngl.NEAREST, self.moderngl.NEAREST)
        return self._white_texture


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
