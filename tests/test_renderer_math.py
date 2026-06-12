import struct
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from p64.engine.math import Vec3
from p64.engine.components import Collider, Light, MeshRenderer
from p64.engine.entity import Entity
from p64.engine.obj import import_obj_to_project
from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.editor.app import _normalize_vec3, _vec3_length
from p64.renderer.scene_renderer import (
    RenderCamera,
    SceneRenderer,
    _convex_collider_wire_vertices,
    _mat4_bytes,
    _mesh_collider_wire_vertices,
    _perspective_matrix,
    _project_point,
    _view_matrix,
    camera_basis,
    grid_line_batches,
)


class RendererMathTests(unittest.TestCase):
    def test_mat4_bytes_transposes_row_major_for_opengl(self):
        row_major_translation = [
            1.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, 3.0,
            0.0, 0.0, 1.0, 4.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        values = struct.unpack("16f", _mat4_bytes(row_major_translation))
        self.assertEqual(values[12:15], (2.0, 3.0, 4.0))

    def test_camera_negative_pitch_looks_down(self):
        forward, _right, _up = camera_basis(Vec3(-18.0, 0.0, 0.0))
        self.assertLess(forward.y, 0.0)
        self.assertLess(forward.z, 0.0)

    def test_project_point_maps_camera_target_to_screen_center(self):
        camera = RenderCamera(position=Vec3(0, 0, 5), rotation=Vec3())
        view = _view_matrix(camera)
        projection = _perspective_matrix(60, 1.0, 0.1, 100)
        projected = _project_point((0, 0, 0), view, projection, 800, 800)
        self.assertIsNotNone(projected)
        self.assertAlmostEqual(projected[0], 400, delta=1)
        self.assertAlmostEqual(projected[1], 400, delta=1)

    def test_broken_shader_uses_internal_error_program(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            shader = project.assets_dir / "shaders" / "broken.shader"
            shader.parent.mkdir(parents=True, exist_ok=True)
            shader.write_text('Shader "Broken"\nVertex { bad glsl }\nFragment { bad glsl }\n', encoding="utf-8")
            ctx = FakeContext()
            previous = sys.modules.get("moderngl")
            sys.modules["moderngl"] = types.SimpleNamespace(DEPTH_TEST=1, LINES=1, NEAREST=1)
            try:
                renderer = SceneRenderer(ctx, project)
            finally:
                if previous is None:
                    sys.modules.pop("moderngl", None)
                else:
                    sys.modules["moderngl"] = previous

            program = renderer._program_for("assets/shaders/broken.shader")

            self.assertIs(program, renderer.error_program)
            self.assertIsNot(program, renderer.program)

    def test_renderer_sends_directional_point_and_spot_lights(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            scene = Scene("Lights")
            sun = Entity("Sun")
            sun.add_component(Light(kind="directional", intensity=1.0))
            point = Entity("Point")
            point.transform.position = Vec3(1, 2, 3)
            point.add_component(Light(kind="point", range=5.0, falloff=1.5))
            spot = Entity("Spot")
            spot.add_component(Light(kind="spot", range=8.0, spot_angle=35.0))
            scene.entities.extend([sun, point, spot])
            renderer = _renderer(project)

            renderer._apply_common_uniforms(
                renderer.program,
                scene,
                RenderCamera(Vec3(), Vec3()),
                _view_matrix(RenderCamera(Vec3(), Vec3())),
                _perspective_matrix(60, 1.0, 0.1, 100),
            )

            uniforms = renderer.program.uniforms
            self.assertEqual(uniforms["u_light_count"].value, 3)
            self.assertEqual(uniforms["u_light_kind"].value[:3], (0, 1, 2))
            self.assertEqual(uniforms["u_light_range[1]"].value, 5.0)
            self.assertEqual(uniforms["u_light_spot_angle[2]"].value, 35.0)
            self.assertEqual(uniforms["u_texture_filter"].value, 2)

    def test_grid_batches_follow_camera_and_fade(self):
        batches = grid_line_batches(Vec3(12.4, 2.0, -7.6), {
            "spacing": 2.0,
            "radius": 12.0,
            "fade_start": 4.0,
            "fade_end": 12.0,
        })
        xs = [batch[0][0] for batch in batches if len(batch[0]) >= 6]
        colors = [batch[1] for batch in batches]

        self.assertTrue(any(abs(x - 12.0) < 0.01 for x in xs))
        self.assertGreater(max(color[0] for color in colors), min(color[0] for color in colors))

    def test_scene_camera_input_normalizes_diagonal_movement(self):
        direction = _normalize_vec3(Vec3(1.0, 0.0, 1.0))
        self.assertAlmostEqual(_vec3_length(direction), 1.0, places=5)

    def test_pick_entity_uses_mesh_triangles_not_pivot(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_pick_mesh(project, "target.obj", x_offset=10.0, z=0.0)
            entity = Entity("OffsetMesh")
            entity.transform.position = Vec3(-10.0, 0.0, 0.0)
            entity.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            scene = Scene("Test", [entity])
            renderer = _renderer(project)

            picked = renderer.pick_entity(scene, 800, 800, 400, 400, RenderCamera(position=Vec3(0, 0, 5), rotation=Vec3()))

            self.assertEqual(picked, entity.id)

    def test_pick_entity_ignores_pivot_when_mesh_is_not_under_cursor(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_pick_mesh(project, "target.obj", x_offset=10.0, z=0.0)
            entity = Entity("OffsetMesh")
            entity.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            scene = Scene("Test", [entity])
            renderer = _renderer(project)

            picked = renderer.pick_entity(scene, 800, 800, 400, 400, RenderCamera(position=Vec3(0, 0, 5), rotation=Vec3()))

            self.assertIsNone(picked)

    def test_pick_entity_returns_nearest_mesh_hit(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            far_metadata = _import_pick_mesh(project, "far.obj", x_offset=0.0, z=-1.0)
            near_metadata = _import_pick_mesh(project, "near.obj", x_offset=0.0, z=1.0)
            far = Entity("Far")
            far.add_component(MeshRenderer(mesh=far_metadata.id, submesh="Body"))
            near = Entity("Near")
            near.add_component(MeshRenderer(mesh=near_metadata.id, submesh="Body"))
            scene = Scene("Test", [far, near])
            renderer = _renderer(project)

            picked = renderer.pick_entity(scene, 800, 800, 400, 400, RenderCamera(position=Vec3(0, 0, 5), rotation=Vec3()))

            self.assertEqual(picked, near.id)

    def test_mesh_collider_wireframe_vertices_are_cached_per_mesh(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_pick_mesh(project, "target.obj", x_offset=0.0, z=0.0)
            component = MeshRenderer(mesh=metadata.id, submesh="Body")
            renderer = _renderer(project)

            vertices = _mesh_collider_wire_vertices(project, component)
            first = renderer._load_mesh_collider_lines(component)
            second = renderer._load_mesh_collider_lines(component)

            self.assertEqual(len(vertices), 18)
            self.assertIs(first, second)
            self.assertEqual(renderer.ctx.buffer_count, 1)

    def test_mesh_collider_gizmo_draws_only_for_selected_object_or_parent(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_pick_mesh(project, "target.obj", x_offset=0.0, z=0.0)
            parent = Entity("Parent")
            child = parent.add_child(Entity("Child"))
            child.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            child.add_component(Collider(shape="mesh"))
            unrelated = Entity("Unrelated")
            scene = Scene("Test", [parent, unrelated])
            renderer = _renderer(project)
            drawn: list[str] = []
            renderer._draw_mesh_collider = lambda entity, _collider, _view, _projection, _color: drawn.append(entity.name)
            view = _view_matrix(RenderCamera(Vec3(), Vec3()))
            projection = _perspective_matrix(60, 1.0, 0.1, 100)

            renderer._draw_component_gizmos(scene, view, projection, selected=None)
            renderer._draw_component_gizmos(scene, view, projection, selected=unrelated)
            self.assertEqual(drawn, [])

            renderer._draw_component_gizmos(scene, view, projection, selected=parent)
            renderer._draw_component_gizmos(scene, view, projection, selected=child)

            self.assertEqual(drawn, ["Child", "Child"])

    def test_convex_mesh_collider_wireframe_uses_hull_cache(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_detailed_cube_mesh(project)
            component = MeshRenderer(mesh=metadata.id, submesh="Body")
            renderer = _renderer(project)

            mesh_vertices = _mesh_collider_wire_vertices(project, component)
            convex_vertices = _convex_collider_wire_vertices(project, component)
            first = renderer._load_convex_collider_lines(component)
            second = renderer._load_convex_collider_lines(component)

            self.assertLess(len(convex_vertices), len(mesh_vertices))
            self.assertIs(first, second)
            self.assertEqual(len(renderer._convex_collider_line_cache), 1)
            renderer.reload_assets()
            self.assertEqual(renderer._convex_collider_line_cache, {})

    def test_selection_outline_uses_visible_child_meshes(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            visible_metadata = _import_pick_mesh(project, "visible.obj", x_offset=0.0, z=0.0)
            hidden_metadata = _import_pick_mesh(project, "hidden.obj", x_offset=0.0, z=0.0)
            parent = Entity("Parent")
            visible = parent.add_child(Entity("Visible"))
            visible.add_component(MeshRenderer(mesh=visible_metadata.id, submesh="Body"))
            invisible = parent.add_child(Entity("Invisible"))
            invisible.add_component(MeshRenderer(mesh=hidden_metadata.id, submesh="Body", visible=False))
            inactive = parent.add_child(Entity("Inactive", active=False))
            inactive.add_component(MeshRenderer(mesh=hidden_metadata.id, submesh="Body"))
            renderer = _renderer(project)

            renderer._draw_selection_outline(parent, _view_matrix(RenderCamera(Vec3(), Vec3())), _perspective_matrix(60, 1.0, 0.1, 100))

            outline_vaos = [vao for vao in renderer.ctx.vaos if vao.program is renderer.selection_outline_program]
            self.assertEqual(len(outline_vaos), 1)
            self.assertEqual(outline_vaos[0].render_count, 1)
            self.assertEqual(renderer.ctx.cull_face, "back")
            self.assertTrue(renderer.ctx.depth_mask)

    def test_selection_outline_vao_is_cached_with_render_mesh_and_cleared_on_reload(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_pick_mesh(project, "target.obj", x_offset=0.0, z=0.0)
            entity = Entity("Target")
            entity.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            renderer = _renderer(project)
            view = _view_matrix(RenderCamera(Vec3(), Vec3()))
            projection = _perspective_matrix(60, 1.0, 0.1, 100)

            renderer._draw_selection_outline(entity, view, projection)
            renderer._draw_selection_outline(entity, view, projection)

            outline_vaos = [vao for vao in renderer.ctx.vaos if vao.program is renderer.selection_outline_program]
            self.assertEqual(len(outline_vaos), 1)
            self.assertEqual(outline_vaos[0].render_count, 2)
            self.assertTrue(any(mesh.outline_vao is not None for mesh in renderer._mesh_cache.values()))
            renderer.reload_assets()
            self.assertEqual(renderer._mesh_cache, {})


class FakeContext:
    def __init__(self):
        self.calls = 0
        self.buffer_count = 0
        self.vertex_array_count = 0
        self.vaos: list[FakeVao] = []
        self.enabled: list[object] = []
        self.disabled: list[object] = []
        self.cull_face = "back"
        self.depth_mask = True

    def program(self, vertex_shader: str, fragment_shader: str):
        self.calls += 1
        if "bad glsl" in vertex_shader or "bad glsl" in fragment_shader:
            raise RuntimeError("compile failed")
        return FakeProgram()

    def buffer(self, data: bytes):
        self.buffer_count += 1
        return FakeBuffer(data)

    def vertex_array(self, program: object, bindings: list[object]):
        self.vertex_array_count += 1
        vao = FakeVao(program)
        self.vaos.append(vao)
        return vao

    def texture(self, size: tuple[int, int], components: int, data: bytes):
        return FakeTexture()

    def enable(self, flag: object) -> None:
        self.enabled.append(flag)

    def disable(self, flag: object) -> None:
        self.disabled.append(flag)


class FakeUniform:
    def __init__(self):
        self.value = None

    def write(self, value: bytes) -> None:
        self.value = value


class FakeProgram:
    def __init__(self):
        self.uniforms: dict[str, FakeUniform] = {}

    def __getitem__(self, name: str) -> FakeUniform:
        if name not in self.uniforms:
            self.uniforms[name] = FakeUniform()
        return self.uniforms[name]


class FakeBuffer:
    def __init__(self, data: bytes):
        self.data = data

    def release(self) -> None:
        pass


class FakeTexture:
    filter = None

    def use(self, location: int = 0) -> None:
        pass


class FakeVao:
    def __init__(self, program: object | None = None):
        self.program = program
        self.render_count = 0

    def render(self, mode: object | None = None) -> None:
        self.render_count += 1

    def release(self) -> None:
        pass


def _renderer(project: Project) -> SceneRenderer:
    previous = sys.modules.get("moderngl")
    sys.modules["moderngl"] = types.SimpleNamespace(DEPTH_TEST=1, LINES=1, NEAREST=1, CULL_FACE=2)
    try:
        return SceneRenderer(FakeContext(), project)
    finally:
        if previous is None:
            sys.modules.pop("moderngl", None)
        else:
            sys.modules["moderngl"] = previous


def _import_pick_mesh(project: Project, name: str, x_offset: float, z: float):
    obj = project.root / name
    obj.write_text(
        "o Body\n"
        f"v {x_offset - 1} -1 {z}\n"
        f"v {x_offset + 1} -1 {z}\n"
        f"v {x_offset} 1 {z}\n"
        "f 1 2 3\n",
        encoding="utf-8",
    )
    return import_obj_to_project(project, obj)


def _import_detailed_cube_mesh(project: Project):
    obj = project.root / "detailed_cube.obj"
    obj.write_text(
        "o Body\n"
        "v -1 -1 -1\n"
        "v 1 -1 -1\n"
        "v 1 1 -1\n"
        "v -1 1 -1\n"
        "v -1 -1 1\n"
        "v 1 -1 1\n"
        "v 1 1 1\n"
        "v -1 1 1\n"
        "v 0 0 0\n"
        "f 1 3 2\n"
        "f 1 4 3\n"
        "f 5 6 7\n"
        "f 5 7 8\n"
        "f 1 2 6\n"
        "f 1 6 5\n"
        "f 2 3 7\n"
        "f 2 7 6\n"
        "f 3 4 8\n"
        "f 3 8 7\n"
        "f 4 1 5\n"
        "f 4 5 8\n"
        "f 1 2 9\n"
        "f 2 3 9\n"
        "f 3 4 9\n"
        "f 4 1 9\n",
        encoding="utf-8",
    )
    return import_obj_to_project(project, obj)


if __name__ == "__main__":
    unittest.main()
