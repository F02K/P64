import struct
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import unittest

from p64.engine.math import Vec3
from p64.engine.assets import AssetMetadata
from p64.engine.components import AudioSource, Camera, Canvas, Collider, Light, MeshRenderer, ModelRenderer, ParticleEmitter, RectTransform, SpriteRenderer, UIImage, UIText
from p64.engine.entity import Entity
from p64.engine.files import find_metadata_for_source
from p64.engine.material import MaterialAsset, save_material_metadata
from p64.engine.obj import import_obj_to_project
from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.editor.app import _normalize_vec3, _vec3_length
from p64.editor.profiler import ProfilerRecorder, aggregate_frames
from p64.renderer.scene_renderer import (
    MESH_OUTLINE_LAYOUT,
    MESH_VERTEX_FLOATS,
    MESH_VERTEX_LAYOUT,
    RenderCamera,
    SceneRenderer,
    TextTexture,
    cloud_dome_vertices,
    flipbook_uv_rect,
    _convex_collider_wire_vertices,
    _canvas_layout_size,
    _camera_from_entity,
    _game_render_size,
    _identity_matrix,
    _mat4_bytes,
    _mesh_collider_wire_vertices,
    particle_quad_vertices,
    _perspective_matrix,
    _project_point,
    _render_geometry_cache_key,
    _view_matrix,
    audio_source_range_radii,
    camera_basis,
    camera_frustum_vertices,
    cloud_plane_vertices,
    grid_line_batches,
    image_fill_uv_rect,
    image_rect_for_fill_mode,
    rect_transform_rect,
    render_camera_basis,
    sprite_quad_vertices,
    text_rect_with_aspect,
    ui_rect,
    ui_layout_debug,
    ui_quad_vertices,
    ui_text_vertices,
    ui_vertices_to_ndc,
)
from p64.renderer.shaders import CLOUD_PLANE_FRAGMENT_SHADER, CLOUD_PLANE_VERTEX_SHADER, SKYBOX_FRAGMENT_SHADER


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

    def test_camera_from_child_entity_uses_world_transform(self):
        parent = Entity("Rig")
        parent.transform.position = Vec3(10.0, 0.0, 0.0)
        parent.transform.rotation = Vec3(0.0, 45.0, 0.0)
        camera_entity = parent.add_child(Entity("Camera"))
        camera_entity.transform.position = Vec3(0.0, 2.0, 0.0)
        camera_entity.transform.rotation = Vec3(0.0, 15.0, 0.0)
        camera_entity.add_component(Camera(fov=75.0, near=0.25, far=123.0))

        camera = _camera_from_entity(camera_entity)

        self.assertEqual(camera.position, Vec3(10.0, 2.0, 0.0))
        self.assertEqual(camera.rotation, Vec3(0.0, 60.0, 0.0))
        self.assertAlmostEqual(camera.forward.x, -0.8660254)
        self.assertAlmostEqual(camera.forward.z, -0.5)
        basis_forward, _basis_right, _basis_up = render_camera_basis(camera)
        self.assertEqual(basis_forward, camera.forward)
        self.assertEqual(camera.fov, 75.0)
        self.assertEqual(camera.near, 0.25)
        self.assertEqual(camera.far, 123.0)

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
            spot.transform.rotation = Vec3(0.0, 90.0, 0.0)
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
            self.assertEqual(uniforms["u_light_kind[0]"].value, 0)
            self.assertEqual(uniforms["u_light_kind[1]"].value, 1)
            self.assertEqual(uniforms["u_light_kind[2]"].value, 2)
            self.assertEqual(uniforms["u_light_position"].value[1], (1, 2, 3))
            self.assertAlmostEqual(uniforms["u_light_direction"].value[2][0], -1.0)
            self.assertAlmostEqual(uniforms["u_light_direction"].value[2][2], 0.0, places=6)
            self.assertEqual(len(uniforms["u_light_position"].value), 8)
            self.assertIsInstance(uniforms["u_light_position"].value[0], tuple)
            self.assertIsInstance(uniforms["u_light_direction"].value[0], tuple)
            self.assertIsInstance(uniforms["u_light_color"].value[0], tuple)
            self.assertEqual(uniforms["u_light_range[1]"].value, 5.0)
            self.assertEqual(uniforms["u_light_spot_angle[2]"].value, 35.0)
            self.assertEqual(uniforms["u_texture_filter"].value, 2)
            self.assertTrue(uniforms["u_dithering_enabled"].value)

    def test_mesh_vao_tolerates_optimized_out_normal_attribute(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_pick_mesh(project, "target.obj", x_offset=0.0, z=0.0)
            shader = project.assets_dir / "shaders" / "position_only.shader"
            shader.parent.mkdir(parents=True, exist_ok=True)
            shader.write_text(
                'Shader "PositionOnly"\n'
                "Vertex { #version 330\n"
                "in vec3 in_position;\n"
                "uniform mat4 u_model;\n"
                "uniform mat4 u_view;\n"
                "uniform mat4 u_projection;\n"
                "void main() { gl_Position = u_projection * u_view * u_model * vec4(in_position, 1.0); } }\n"
                "Fragment { #version 330\n"
                "out vec4 fragColor;\n"
                "void main() { fragColor = vec4(1.0); } }\n",
                encoding="utf-8",
            )
            renderer = _renderer(project)
            entity = Entity("Target")
            component = MeshRenderer(mesh=metadata.id, submesh="Body", shader="assets/shaders/position_only.shader")

            mesh = renderer._load_mesh(entity, component)

            self.assertIsNotNone(mesh)
            self.assertTrue(renderer.ctx.vaos[-1].skip_errors)

    def test_mesh_vertex_array_tolerates_optimized_out_color_attribute(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_pick_mesh(project, "target.obj", x_offset=0.0, z=0.0)
            shader = project.assets_dir / "shaders" / "no_color.shader"
            shader.parent.mkdir(parents=True, exist_ok=True)
            shader.write_text(
                'Shader "NoColor"\n'
                "Vertex { #version 330\n"
                "in vec3 in_position;\n"
                "in vec2 in_uv;\n"
                "in vec3 in_normal;\n"
                "uniform mat4 u_model;\n"
                "uniform mat4 u_view;\n"
                "uniform mat4 u_projection;\n"
                "void main() { gl_Position = u_projection * u_view * u_model * vec4(in_position + in_normal * 0.0 + vec3(in_uv, 0.0) * 0.0, 1.0); } }\n"
                "Fragment { #version 330\n"
                "out vec4 fragColor;\n"
                "void main() { fragColor = vec4(1.0); } }\n",
                encoding="utf-8",
            )
            renderer = _renderer(project)
            entity = Entity("Target")
            component = MeshRenderer(mesh=metadata.id, submesh="Body", shader="assets/shaders/no_color.shader")

            mesh = renderer._load_mesh(entity, component)

            self.assertIsNotNone(mesh)
            self.assertTrue(renderer.ctx.vaos[-1].skip_errors)
            self.assertEqual(renderer.ctx.vaos[-1].bindings[0][1:], (MESH_VERTEX_LAYOUT, "in_position", "in_uv", "in_normal", "in_color"))

    def test_mesh_buffer_includes_default_white_vertex_color(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_pick_mesh(project, "target.obj", x_offset=0.0, z=0.0)
            renderer = _renderer(project)
            entity = Entity("Target")
            component = MeshRenderer(mesh=metadata.id, submesh="Body")

            mesh = renderer._load_mesh(entity, component)

            self.assertIsNotNone(mesh)
            values = struct.unpack(f"{MESH_VERTEX_FLOATS}f", mesh.buffer.data[: MESH_VERTEX_FLOATS * 4])
            self.assertEqual(values[8:11], (1.0, 1.0, 1.0))

    def test_renderer_loads_concrete_model_mesh_entry(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_pick_mesh(project, "target.obj", x_offset=0.0, z=0.0)
            mesh_id = metadata.settings["model"]["meshes"][0]["id"]
            renderer = _renderer(project)
            entity = Entity("Target")
            component = MeshRenderer(mesh=mesh_id)

            mesh = renderer._load_mesh(entity, component)
            vertices = _mesh_collider_wire_vertices(project, component)

            self.assertIsNotNone(mesh)
            self.assertEqual(len(vertices), 18)

    def test_mesh_vao_still_requires_position_attribute(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_pick_mesh(project, "target.obj", x_offset=0.0, z=0.0)
            shader = project.assets_dir / "shaders" / "missing_position.shader"
            shader.parent.mkdir(parents=True, exist_ok=True)
            shader.write_text(
                'Shader "MissingPosition"\n'
                "Vertex { #version 330\n"
                "void main() { gl_Position = vec4(0.0); } }\n"
                "Fragment { #version 330\n"
                "out vec4 fragColor;\n"
                "void main() { fragColor = vec4(1.0); } }\n",
                encoding="utf-8",
            )
            logs: list[str] = []
            renderer = _renderer(project, logs.append)
            entity = Entity("Target")
            component = MeshRenderer(mesh=metadata.id, submesh="Body", shader="assets/shaders/missing_position.shader")

            mesh = renderer._load_mesh(entity, component)

            self.assertIsNone(mesh)
            self.assertTrue(any("Could not bind mesh attributes" in message for message in logs))

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

    def test_audio_source_gizmo_uses_min_and_max_ranges(self):
        source = AudioSource(min_distance=2.0, max_distance=9.0)

        self.assertEqual(audio_source_range_radii(source), (2.0, 9.0))

    def test_camera_frustum_vertices_include_near_far_planes_and_edges(self):
        vertices = camera_frustum_vertices(Vec3(), Vec3(), 90.0, 1.0, 3.0, 1.0)

        self.assertEqual(len(vertices), 72)
        z_values = vertices[2::3]
        self.assertIn(-1.0, [round(value, 4) for value in z_values])
        self.assertIn(-3.0, [round(value, 4) for value in z_values])

    def test_selected_audio_and_camera_gizmos_are_drawn(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            audio_entity = Entity("Audio")
            audio_entity.add_component(AudioSource(min_distance=1.0, max_distance=4.0))
            camera_entity = Entity("Camera")
            camera_entity.add_component(Camera(near=0.2, far=5.0))
            scene = Scene("Test", [audio_entity, camera_entity])
            renderer = _renderer(project)
            drawn: list[str] = []
            renderer._draw_audio_source_ranges = lambda *_args: drawn.append("audio")
            renderer._draw_camera_frustum = lambda *_args: drawn.append("camera")
            view = _view_matrix(RenderCamera(Vec3(), Vec3()))
            projection = _perspective_matrix(60, 1.0, 0.1, 100)

            renderer._draw_component_gizmos(scene, view, projection, selected=audio_entity)
            renderer._draw_component_gizmos(scene, view, projection, selected=camera_entity)

            self.assertEqual(drawn, ["audio", "camera"])

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
            self.assertEqual(outline_vaos[0].bindings[0][1:], (MESH_OUTLINE_LAYOUT, "in_position", "in_normal"))
            self.assertEqual(renderer.ctx.cull_face, "back")
            self.assertTrue(renderer.ctx.depth_mask)

    def test_selection_outline_draws_edge_overlay_for_planar_meshes(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_pick_mesh(project, "target.obj", x_offset=0.0, z=0.0)
            entity = Entity("Target")
            entity.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            renderer = _renderer(project)
            view = _view_matrix(RenderCamera(Vec3(), Vec3()))
            projection = _perspective_matrix(60, 1.0, 0.1, 100)

            renderer._draw_selection_outline(entity, view, projection)

            line_vaos = [vao for vao in renderer.ctx.vaos if vao.program is renderer.line_program]
            self.assertEqual(len(line_vaos), 1)
            self.assertEqual(line_vaos[0].render_count, 1)
            self.assertIsNotNone(next(iter(renderer._mesh_cache.values())).edge_mesh)

    def test_material_diffuse_color_is_sent_as_base_color(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            obj = Path(tmp) / "colored.obj"
            obj.write_text(
                "mtllib colored.mtl\n"
                "o Body\n"
                "v 0 0 0\n"
                "v 1 0 0\n"
                "v 0 1 0\n"
                "vt 0 0\n"
                "vt 1 0\n"
                "vt 0 1\n"
                "vn 0 0 1\n"
                "usemtl Paint\n"
                "f 1/1/1 2/2/1 3/3/1\n",
                encoding="utf-8",
            )
            (Path(tmp) / "colored.mtl").write_text("newmtl Paint\nKd 0.25 0.5 0.75\n", encoding="utf-8")
            metadata = import_obj_to_project(project, obj)
            entity = Entity("Target")
            entity.add_component(MeshRenderer(mesh=metadata.id, submesh="Body", material="Paint"))
            scene = Scene("Test", [entity])
            renderer = _renderer(project)

            renderer.render(scene, 800, 800, camera=RenderCamera(Vec3(0, 0, 5), Vec3()), show_grid=False)

            self.assertEqual(renderer.program.uniforms["u_base_color"].value, (0.25, 0.5, 0.75))

    def test_material_asset_overrides_mtl_base_color(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            obj = Path(tmp) / "colored.obj"
            obj.write_text(
                "mtllib colored.mtl\n"
                "o Body\n"
                "v 0 0 0\n"
                "v 1 0 0\n"
                "v 0 1 0\n"
                "vt 0 0\n"
                "vt 1 0\n"
                "vt 0 1\n"
                "vn 0 0 1\n"
                "usemtl Paint\n"
                "f 1/1/1 2/2/1 3/3/1\n",
                encoding="utf-8",
            )
            (Path(tmp) / "colored.mtl").write_text("newmtl Paint\nKd 0.25 0.5 0.75\n", encoding="utf-8")
            metadata = import_obj_to_project(project, obj)
            metadata_path = find_metadata_for_source(project.root / metadata.source)
            metadata.materials = ["Other", "Paint"]
            metadata.save(metadata_path)
            material_path = project.assets_dir / "materials" / "Paint.material"
            MaterialAsset(properties={"u_base_color": [0.8, 0.1, 0.2], "u_custom_float": 0.6}).save(material_path)
            other_path = project.assets_dir / "materials" / "Other.material"
            MaterialAsset(properties={"u_base_color": [0.0, 1.0, 0.0]}).save(other_path)
            save_material_metadata(project.root, material_path, source={"obj": metadata.source, "mtl": "Paint"})
            entity = Entity("Target")
            entity.add_component(
                MeshRenderer(
                    mesh=metadata.id,
                    submesh="Body",
                    material="Paint",
                    source_materials=["Other", "Paint"],
                    material_slots=["assets/materials/Other.material", "assets/materials/Paint.material"],
                )
            )
            scene = Scene("Test", [entity])
            renderer = _renderer(project)

            renderer.render(scene, 800, 800, camera=RenderCamera(Vec3(0, 0, 5), Vec3()), show_grid=False)

            program = renderer._program_for(next(iter(renderer._mesh_cache.values())).shader)
            self.assertEqual(program.uniforms["u_base_color"].value, (0.8, 0.1, 0.2))
            self.assertEqual(program.uniforms["u_custom_float"].value, 0.6)

    def test_renderer_sets_default_alpha_cutoff_and_material_can_override(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_pick_mesh(project, "target.obj", x_offset=0.0, z=0.0)
            default_entity = Entity("Default")
            default_entity.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            material_path = project.assets_dir / "materials" / "Cutout.material"
            MaterialAsset(properties={"u_alpha_cutoff": 0.42}).save(material_path)
            cutout_entity = Entity("Cutout")
            cutout_entity.add_component(
                MeshRenderer(
                    mesh=metadata.id,
                    submesh="Body",
                    material="Paint",
                    source_materials=["Paint"],
                    material_slots=["assets/materials/Cutout.material"],
                )
            )
            scene = Scene("Test", [default_entity, cutout_entity])
            renderer = _renderer(project)

            renderer.render(scene, 800, 800, camera=RenderCamera(Vec3(0, 0, 5), Vec3()), show_grid=False)

            self.assertEqual(renderer.program.uniforms["u_alpha_cutoff"].value, 0.0)
            program = renderer._program_for(next(mesh for mesh in renderer._mesh_cache.values() if mesh.material_properties).shader)
            self.assertEqual(program.uniforms["u_alpha_cutoff"].value, 0.42)

    def test_renderer_splits_model_mesh_into_material_batches(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            obj = Path(tmp) / "multi_material.obj"
            obj.write_text(
                "mtllib multi_material.mtl\n"
                "o Body\n"
                "v 0 0 0\n"
                "v 1 0 0\n"
                "v 1 1 0\n"
                "v 0 1 0\n"
                "usemtl Stone\n"
                "f 1 2 3\n"
                "usemtl Moss\n"
                "f 1 3 4\n",
                encoding="utf-8",
            )
            (Path(tmp) / "multi_material.mtl").write_text("newmtl Stone\nKd 0.2 0.2 0.2\nnewmtl Moss\nKd 0.1 0.5 0.1\n", encoding="utf-8")
            metadata = import_obj_to_project(project, obj)
            mesh_id = metadata.settings["model"]["meshes"][0]["id"]
            entity = Entity("Target")
            entity.add_component(MeshRenderer(mesh=mesh_id, source_materials=["Stone", "Moss"], material_slots=[None, None]))
            scene = Scene("Test", [entity])
            renderer = _renderer(project)

            renderer.render(scene, 800, 800, camera=RenderCamera(Vec3(0, 0, 5), Vec3()), show_grid=False)

            render_mesh = next(iter(renderer._mesh_cache.values()))
            self.assertIsNotNone(render_mesh.batches)
            self.assertEqual(len(render_mesh.batches or []), 2)
            self.assertEqual(sum(batch.vao.render_count for batch in render_mesh.batches or []), 2)

    def test_model_renderer_uses_persistent_cache_batches(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            obj = Path(tmp) / "static_model.obj"
            obj.write_text(
                "mtllib static_model.mtl\n"
                "o WallA\n"
                "v 0 0 0\n"
                "v 1 0 0\n"
                "v 0 1 0\n"
                "usemtl Stone\n"
                "f 1 2 3\n"
                "o WallB\n"
                "v 2 0 0\n"
                "v 3 0 0\n"
                "v 2 1 0\n"
                "usemtl Stone\n"
                "f 4 5 6\n",
                encoding="utf-8",
            )
            (Path(tmp) / "static_model.mtl").write_text("newmtl Stone\nKd 0.2 0.2 0.2\n", encoding="utf-8")
            metadata = import_obj_to_project(project, obj)
            self.assertEqual(metadata.settings["model_cache"]["batch_count"], 1)
            entity = Entity("Static Model")
            entity.add_component(ModelRenderer(model=metadata.id, source_materials=["Stone"], material_slots=[None]))
            scene = Scene("Test", [entity])
            scene.render_settings["skybox_enabled"] = False
            renderer = _renderer(project)
            recorder = ProfilerRecorder()
            recorder.set_enabled(True)
            renderer.profiler_recorder = recorder

            frame = recorder.begin_frame("Scene")
            renderer.render(scene, 800, 800, camera=RenderCamera(Vec3(0, 0, 5), Vec3()), show_grid=False)
            recorder.end_frame(frame)

            model = next(iter(renderer._model_cache.values()))
            self.assertTrue(model.cache_hit)
            self.assertEqual(len(model.batches), 1)
            self.assertEqual(sum(batch.vao.render_count for batch in model.batches), 1)
            counts = aggregate_frames(recorder.frames()).counts
            self.assertEqual(counts["model renderers"], 1)
            self.assertEqual(counts["dynamic model renderers"], 1)
            self.assertEqual(counts["static model renderers"], 0)
            self.assertEqual(counts["static model batches"], 1)
            self.assertEqual(counts["static cache hits"], 1)
            self.assertEqual(counts["draw submissions"], 1)

    def test_model_renderer_rebuilds_when_persistent_cache_signature_is_stale(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            obj = Path(tmp) / "static_model.obj"
            obj.write_text(
                "o Body\n"
                "v 0 0 0\n"
                "v 1 0 0\n"
                "v 0 1 0\n"
                "f 1 2 3\n",
                encoding="utf-8",
            )
            metadata = import_obj_to_project(project, obj)
            metadata_path = find_metadata_for_source(project.root / metadata.source)
            reloaded = AssetMetadata.load(metadata_path)
            reloaded.settings["model_cache"]["source_signature"] = {"size": -1, "mtime_ns": -1}
            reloaded.save(metadata_path)
            entity = Entity("Static Model")
            entity.add_component(ModelRenderer(model=metadata.id))
            scene = Scene("Test", [entity])
            scene.render_settings["skybox_enabled"] = False
            renderer = _renderer(project)
            recorder = ProfilerRecorder()
            recorder.set_enabled(True)
            renderer.profiler_recorder = recorder

            frame = recorder.begin_frame("Scene")
            renderer.render(scene, 800, 800, camera=RenderCamera(Vec3(0, 0, 5), Vec3()), show_grid=False)
            recorder.end_frame(frame)

            model = next(iter(renderer._model_cache.values()))
            self.assertFalse(model.cache_hit)
            counts = aggregate_frames(recorder.frames()).counts
            self.assertEqual(counts["static model batches"], 1)
            self.assertNotIn("static cache hits", counts)
            self.assertEqual(counts["draw submissions"], 1)

    def test_skybox_pass_renders_before_meshes_when_enabled(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            entity = Entity("Target")
            entity.add_component(MeshRenderer(mesh="missing"))
            scene = Scene("Test", [entity])
            scene.render_settings["skybox_enabled"] = True
            renderer = _renderer(project)
            events: list[str] = []
            renderer._draw_skybox = lambda *_args: events.append("skybox")
            renderer._draw_cloud_plane = lambda *_args: events.append("cloud")
            renderer._draw_mesh = lambda *_args: events.append("mesh") or False

            renderer.render(scene, 320, 240, camera=RenderCamera(Vec3(), Vec3()), show_grid=False)

            self.assertEqual(events, ["skybox", "cloud", "mesh"])

    def test_renderer_reports_profiler_sections_when_enabled(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            renderer = _renderer(project)
            recorder = ProfilerRecorder()
            recorder.set_enabled(True)
            renderer.profiler_recorder = recorder
            scene = Scene("Profile", [
                Entity("Mesh", components=[MeshRenderer(mesh="missing")]),
                Entity("Emitter", components=[ParticleEmitter()]),
                Entity("Label", components=[UIImage(), UIText(text="Ready")]),
            ])

            frame = recorder.begin_frame("Scene")
            renderer.render(scene, 320, 240, RenderCamera(Vec3(), Vec3()))
            recorder.end_frame(frame)
            snapshot = aggregate_frames(recorder.frames())
            names = {section.name for section in snapshot.sections}

            self.assertIn("render total", names)
            self.assertIn("render skybox", names)
            self.assertIn("render clouds", names)
            self.assertIn("render meshes", names)
            self.assertIn("render particles", names)
            self.assertIn("render ui", names)
            self.assertIn("active entities", snapshot.counts)

    def test_renderer_skips_profiler_counts_without_active_profiler(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            renderer = _renderer(project)
            scene = Scene("No Profiler")
            scene.render_settings["skybox_enabled"] = False
            calls: list[str] = []
            renderer._add_render_counts = lambda _active_entities: calls.append("counts")

            renderer.render(scene, 320, 240, RenderCamera(Vec3(), Vec3()), show_grid=False)

            self.assertEqual(calls, [])

    def test_game_view_upscale_quad_is_cached(self):
        source = Path("src/p64/renderer/scene_renderer.py").read_text(encoding="utf-8")
        draw_method = source[source.index("def _draw_upscaled_texture"):source.index("def _upscale_quad_resources")]

        self.assertIn("self._upscale_quad", source)
        self.assertIn("def _upscale_quad_resources", source)
        self.assertIn("def _release_upscale_quad", source)
        self.assertIn("_buffer, vao = self._upscale_quad_resources()", draw_method)
        self.assertNotIn("self.ctx.buffer(struct.pack", draw_method)
        self.assertNotIn("buffer.release()", draw_method)
        self.assertNotIn("vao.release()", draw_method)

    def test_model_renderer_mesh_collider_lines_do_not_require_mesh_attribute(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = AssetMetadata(
                id="model_luigi",
                kind="obj_mesh",
                source="assets/luigi.obj",
                groups=["Body"],
                materials=[],
                settings={
                    "model": {
                        "meshes": [{
                            "id": "mesh_model_luigi_Body",
                            "name": "Body",
                            "source_group": "Body",
                            "wireframe": {"vertices": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]},
                        }],
                    },
                },
            )
            (project.assets_dir / "luigi.obj").write_text("o Body\n", encoding="utf-8")
            metadata.save(project.assets_dir / "luigi.obj.mdp64")
            renderer = _renderer(project)
            component = ModelRenderer(model=metadata.id)

            line_mesh = renderer._load_mesh_collider_lines(component)
            convex_line_mesh = renderer._load_convex_collider_lines(component)

            self.assertIsNotNone(line_mesh)
            self.assertIsNone(convex_line_mesh)
            self.assertEqual(_render_geometry_cache_key(component), ("model", metadata.id, None))

    def test_skybox_pass_skips_when_disabled(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            scene = Scene("Test")
            scene.render_settings["skybox_enabled"] = False
            renderer = _renderer(project)
            calls: list[str] = []
            renderer._draw_skybox = lambda *_args: calls.append("skybox")
            renderer._draw_cloud_plane = lambda *_args: calls.append("cloud")

            renderer.render(scene, 320, 240, camera=RenderCamera(Vec3(), Vec3()), show_grid=False)

            self.assertEqual(calls, [])

    def test_background_state_failures_do_not_abort_mesh_render(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            entity = Entity("Target")
            entity.add_component(MeshRenderer(mesh="missing"))
            scene = Scene("Test", [entity])
            logs: list[str] = []
            renderer = _renderer(project, logs.append)
            renderer.ctx.fail_disable = True
            renderer.ctx.fail_depth_mask = True
            renderer.ctx.fail_blend_func = True
            events: list[str] = []
            renderer._draw_mesh = lambda *_args: events.append("mesh") or False

            renderer.render(scene, 320, 240, camera=RenderCamera(Vec3(), Vec3()), show_grid=False)

            self.assertEqual(events, ["mesh"])
            self.assertTrue(any("Background pass begin/depth_mask failed: RuntimeError" in message for message in logs))
            self.assertTrue(any("Background pass begin/disable_depth failed: RuntimeError" in message for message in logs))
            self.assertFalse(any(message.startswith("Render failed") for message in logs))

    def test_not_implemented_blend_state_does_not_abort_ui_render(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            entity = Entity("Hud")
            entity.add_component(UIImage())
            scene = Scene("UI", [entity])
            scene.render_settings["skybox_enabled"] = False
            logs: list[str] = []
            renderer = _renderer(project, logs.append)
            renderer.ctx.fail_blend_func_not_implemented = True

            renderer.render(scene, 320, 240, camera=RenderCamera(Vec3(), Vec3()), show_grid=False)

            self.assertFalse(any("blend_func failed" in message for message in logs))
            self.assertFalse(any("UI batch render failed" in message for message in logs))
            self.assertGreater(renderer.ctx.vertex_array_count, 0)

    def test_skybox_vao_failure_logs_stage_and_render_continues(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            entity = Entity("Target")
            entity.add_component(MeshRenderer(mesh="missing"))
            scene = Scene("Test", [entity])
            logs: list[str] = []
            renderer = _renderer(project, logs.append)
            events: list[str] = []

            def fail_skybox_vao():
                raise RuntimeError("sky vao boom")

            renderer._skybox_vertex_array = fail_skybox_vao
            renderer._draw_cloud_plane = lambda *_args: events.append("cloud")
            renderer._draw_mesh = lambda *_args: events.append("mesh") or False

            renderer.render(scene, 320, 240, camera=RenderCamera(Vec3(), Vec3()), show_grid=False)

            self.assertEqual(events, ["cloud", "mesh"])
            self.assertTrue(any("Skybox render failed during vao: RuntimeError: RuntimeError('sky vao boom')" in message for message in logs))

    def test_cloud_failures_log_stage_without_raising(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            logs: list[str] = []
            renderer = _renderer(project, logs.append)
            camera = RenderCamera(position=Vec3(0.0, 4.0, 0.0), rotation=Vec3(), far=200.0)
            view = _view_matrix(camera)
            projection = _perspective_matrix(60, 1.0, 0.1, 200.0)

            def fail_cloud_vao(_camera: RenderCamera, _height: float):
                raise RuntimeError("cloud vao boom")

            renderer._cloud_plane_vertex_array = fail_cloud_vao

            renderer._draw_cloud_plane(camera, view, projection, {"skybox_cloud_height": 80.0})

            self.assertTrue(any("Cloud plane render failed during vao: RuntimeError: RuntimeError('cloud vao boom')" in message for message in logs))

    def test_cloud_plane_vertices_form_camera_centered_dome(self):
        camera = RenderCamera(position=Vec3(10.0, 4.0, -6.0), rotation=Vec3(), far=200.0)
        vertices = cloud_plane_vertices(camera, 80.0, segments=12)

        self.assertEqual(len(vertices), 12 * 12 * 3)
        self.assertGreater(len(set(round(value, 4) for value in vertices[1::3])), 2)
        self.assertAlmostEqual((min(vertices[0::3]) + max(vertices[0::3])) / 2.0, 10.0)
        self.assertAlmostEqual((min(vertices[2::3]) + max(vertices[2::3])) / 2.0, -6.0)
        self.assertGreater(max(vertices[1::3]), 4.0 + 79.0)
        self.assertLess(min(vertices[1::3]), 4.0 + 20.0)

    def test_cloud_dome_vertices_are_local_for_cached_camera_follow(self):
        vertices = cloud_dome_vertices(80.0, 640.0, segments=12)

        self.assertEqual(len(vertices), 12 * 12 * 3)
        self.assertAlmostEqual((min(vertices[0::3]) + max(vertices[0::3])) / 2.0, 0.0, delta=0.001)
        self.assertAlmostEqual((min(vertices[2::3]) + max(vertices[2::3])) / 2.0, 0.0, delta=0.001)
        self.assertGreater(len(set(round(value, 4) for value in vertices[1::3])), 2)

    def test_cloud_plane_draw_reuses_cached_geometry(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            renderer = _renderer(project)
            camera = RenderCamera(position=Vec3(0.0, 4.0, 0.0), rotation=Vec3(), far=200.0)
            view = _view_matrix(camera)
            projection = _perspective_matrix(60, 1.0, 0.1, 200.0)
            settings = {
                "skybox_cloud_color": [1.0, 0.96, 0.86],
                "skybox_cloud_coverage": 0.45,
                "skybox_cloud_scale": 3.0,
                "skybox_cloud_height": 80.0,
                "skybox_cloud_softness": 0.08,
                "color_levels": 32,
                "dithering": True,
            }

            renderer._draw_cloud_plane(camera, view, projection, settings)
            first_buffer_count = renderer.ctx.buffer_count
            first_vertex_array_count = renderer.ctx.vertex_array_count
            camera.position.x = 20.0
            renderer._draw_cloud_plane(camera, view, projection, settings)

            self.assertEqual(renderer.ctx.buffer_count, first_buffer_count)
            self.assertEqual(renderer.ctx.vertex_array_count, first_vertex_array_count)
            self.assertIsNotNone(renderer.cloud_plane_vao)
            renderer.reload_assets()
            self.assertIsNone(renderer.cloud_plane_vao)

    def test_flipbook_uv_rect_selects_expected_frame(self):
        self.assertEqual(flipbook_uv_rect(4, 2, 8.0, 1, 6, 0.25), (0.75, 0.0, 1.0, 0.5))
        self.assertEqual(flipbook_uv_rect(1, 1, 0.0, 0, 0, 10.0), (0.0, 0.0, 1.0, 1.0))

    def test_sprite_quad_vertices_billboard_to_camera(self):
        entity = Entity("Sprite")
        entity.transform.position = Vec3(1.0, 2.0, 3.0)
        sprite = SpriteRenderer(size=Vec3(2.0, 4.0, 1.0), pivot=Vec3(0.5, 0.5, 0.0))
        vertices = sprite_quad_vertices(entity, sprite, RenderCamera(Vec3(0, 0, 8), Vec3()))

        self.assertEqual(len(vertices), 6 * 8)
        xs = vertices[0::8]
        ys = vertices[1::8]
        self.assertAlmostEqual(min(xs), 0.0)
        self.assertAlmostEqual(max(xs), 2.0)
        self.assertAlmostEqual(min(ys), 0.0)
        self.assertAlmostEqual(max(ys), 4.0)

    def test_ui_rect_and_quad_vertices_use_pixel_anchors(self):
        image = UIImage(size=Vec3(100.0, 50.0, 0.0), anchor="top-left", offset=Vec3(10.0, 20.0, 0.0), pivot=Vec3(0.0, 0.0, 0.0))

        self.assertEqual(ui_rect(image.anchor, image.offset, image.size, image.pivot, 800, 600), (10.0, 20.0, 100.0, 50.0))
        vertices = ui_quad_vertices(image, 800, 600)

        self.assertEqual(len(vertices), 6 * 8)
        self.assertEqual(vertices[:5], [10.0, 70.0, 0.0, 0.0, 0.0])

    def test_rect_transform_vertices_are_relative_to_parent_rect(self):
        rect = rect_transform_rect(
            RectTransform(anchor="top-left", offset=Vec3(10.0, 20.0, 0.0), size=Vec3(100.0, 50.0, 0.0), pivot=Vec3(0.0, 0.0, 0.0)),
            (50.0, 60.0, 400.0, 300.0),
        )
        image_vertices = ui_quad_vertices(UIImage(), 800, 600, rect=rect)
        text_vertices = ui_text_vertices(UIText(text="Score"), 800, 600, rect=rect, texture_size=(100, 50))

        self.assertEqual(rect, (60.0, 80.0, 100.0, 50.0))
        self.assertEqual(image_vertices[:5], [60.0, 130.0, 0.0, 0.0, 0.0])
        self.assertEqual(text_vertices[:5], [60.0, 130.0, 0.0, 0.0, 0.0])

    def test_rect_transform_center_bounds_define_y_limits(self):
        rect = rect_transform_rect(
            RectTransform(anchor="center", offset=Vec3(0.0, 0.0, 0.0), size=Vec3(160.0, 48.0, 0.0), pivot=Vec3(0.5, 0.5, 0.0)),
            (0.0, 0.0, 800.0, 600.0),
        )

        self.assertEqual(rect, (320.0, 276.0, 160.0, 48.0))
        self.assertEqual((rect[1], rect[1] + rect[3]), (276.0, 324.0))

    def test_ui_layout_debug_reports_rect_and_content_bounds(self):
        canvas = Entity("Canvas", components=[Canvas()])
        child = canvas.add_child(Entity(
            "Card",
            rect_transform=RectTransform(size=Vec3(160.0, 48.0, 0.0)),
            components=[
                UIImage(size=Vec3(128.0, 128.0, 0.0), fill_mode="fit"),
                UIText(text="Text", font_size=24.0),
            ],
        ))
        scene = Scene("UI", [canvas])

        entries = ui_layout_debug(scene, 800, 600, text_size_getter=lambda _component: (120, 30))
        entry = next(item for item in entries if item.entity_id == child.id)

        self.assertEqual(entry.rect, (320.0, 276.0, 160.0, 48.0))
        self.assertEqual(entry.image_rects, ((376.0, 276.0, 48.0, 48.0),))
        self.assertEqual(entry.text_rects, ((320.0, 280.0, 160.0, 40.0),))

    def test_child_rect_transform_is_relative_to_parent_rect(self):
        canvas = Entity("Canvas", components=[Canvas()])
        parent = canvas.add_child(Entity(
            "Panel",
            rect_transform=RectTransform(anchor="top-left", offset=Vec3(100.0, 50.0, 0.0), size=Vec3(300.0, 200.0, 0.0), pivot=Vec3(0.0, 0.0, 0.0)),
        ))
        child = parent.add_child(Entity(
            "Child",
            rect_transform=RectTransform(anchor="bottom-right", offset=Vec3(-10.0, -20.0, 0.0), size=Vec3(80.0, 40.0, 0.0), pivot=Vec3(1.0, 1.0, 0.0)),
            components=[UIImage()],
        ))
        scene = Scene("UI", [canvas])

        entries = ui_layout_debug(scene, 800, 600)
        entry = next(item for item in entries if item.entity_id == child.id)

        self.assertEqual(entry.rect, (310.0, 190.0, 80.0, 40.0))

    def test_canvas_root_without_rect_transform_uses_canvas_layout_as_parent_box(self):
        canvas = Entity("Canvas", components=[Canvas(reference_resolution=Vec3(640.0, 480.0, 0.0), resolution_mode="fixed")])
        child = canvas.add_child(Entity(
            "Child",
            rect_transform=RectTransform(anchor="bottom-right", size=Vec3(100.0, 50.0, 0.0), pivot=Vec3(1.0, 1.0, 0.0)),
            components=[UIImage()],
        ))
        scene = Scene("UI", [canvas])

        entries = ui_layout_debug(scene, 800, 600)
        entry = next(item for item in entries if item.entity_id == child.id)

        self.assertEqual(entry.rect, (540.0, 430.0, 100.0, 50.0))

    def test_ui_image_stretch_fill_mode_uses_full_rect_transform_box(self):
        rect = (10.0, 20.0, 160.0, 48.0)
        image = UIImage(size=Vec3(128.0, 128.0, 0.0), fill_mode="stretch")
        legacy = UIImage(size=Vec3(128.0, 128.0, 0.0), fill_mode="simple")

        self.assertEqual(image_rect_for_fill_mode(image, rect), rect)
        self.assertEqual(image_rect_for_fill_mode(legacy, rect), rect)
        self.assertEqual(ui_quad_vertices(image, 800, 600, rect=rect)[:5], [10.0, 68.0, 0.0, 0.0, 0.0])

    def test_ui_image_fit_fill_mode_preserves_aspect_inside_rect_transform_box(self):
        rect = (10.0, 20.0, 160.0, 48.0)
        image = UIImage(size=Vec3(128.0, 128.0, 0.0), fill_mode="fit")
        vertices = ui_quad_vertices(image, 800, 600, rect=rect)

        self.assertEqual(image_rect_for_fill_mode(image, rect), (66.0, 20.0, 48.0, 48.0))
        self.assertEqual(vertices[:5], [66.0, 68.0, 0.0, 0.0, 0.0])

    def test_ui_vertices_to_ndc_maps_fullscreen_pixel_rect_to_clipspace(self):
        vertices = ui_quad_vertices(UIImage(), 800, 600, rect=(0.0, 0.0, 800.0, 600.0))
        ndc = ui_vertices_to_ndc(vertices, 800, 600)

        self.assertEqual(ndc[:5], [-1.0, -1.0, 0.0, 0.0, 0.0])
        self.assertEqual(ndc[8:13], [1.0, -1.0, 0.0, 1.0, 0.0])
        self.assertEqual(ndc[16:21], [1.0, 1.0, 0.0, 1.0, 1.0])
        self.assertEqual(ndc[24:29], [-1.0, -1.0, 0.0, 0.0, 0.0])
        self.assertEqual(ndc[32:37], [1.0, 1.0, 0.0, 1.0, 1.0])
        self.assertEqual(ndc[40:45], [-1.0, 1.0, 0.0, 0.0, 1.0])

    def test_ui_vertices_to_ndc_maps_center_rect_without_aspect_stretch(self):
        vertices = ui_quad_vertices(UIImage(), 800, 600, rect=(320.0, 276.0, 160.0, 48.0))
        ndc = ui_vertices_to_ndc(vertices, 800, 600)
        xs = ndc[0::8]
        ys = ndc[1::8]

        self.assertAlmostEqual(min(xs), -0.2)
        self.assertAlmostEqual(max(xs), 0.2)
        self.assertAlmostEqual(min(ys), -0.08)
        self.assertAlmostEqual(max(ys), 0.08)

    def test_ui_image_fit_pixel_rect_then_ndc_quad_preserves_shape(self):
        rect = (10.0, 20.0, 160.0, 48.0)
        image = UIImage(size=Vec3(128.0, 128.0, 0.0), fill_mode="fit")
        vertices = ui_quad_vertices(image, 800, 600, rect=rect)
        ndc = ui_vertices_to_ndc(vertices, 800, 600)
        xs = ndc[0::8]
        ys = ndc[1::8]

        self.assertEqual(image_rect_for_fill_mode(image, rect), (66.0, 20.0, 48.0, 48.0))
        self.assertAlmostEqual(max(xs) - min(xs), 48.0 / 800.0 * 2.0)
        self.assertAlmostEqual(max(ys) - min(ys), 48.0 / 600.0 * 2.0)

    def test_ui_text_pixel_rect_then_ndc_quad_preserves_shape(self):
        rect = (10.0, 20.0, 20.0, 200.0)
        vertices = ui_text_vertices(UIText(text="Score"), 800, 600, rect=rect, texture_size=(120, 30))
        ndc = ui_vertices_to_ndc(vertices, 800, 600)
        xs = ndc[0::8]
        ys = ndc[1::8]

        self.assertAlmostEqual(max(xs) - min(xs), 20.0 / 800.0 * 2.0)
        self.assertAlmostEqual(max(ys) - min(ys), 5.0 / 600.0 * 2.0)

    def test_ui_image_fill_fill_mode_crops_uvs_without_stretching_quad(self):
        rect = (10.0, 20.0, 160.0, 48.0)
        image = UIImage(size=Vec3(128.0, 128.0, 0.0), fill_mode="fill")

        self.assertEqual(image_rect_for_fill_mode(image, rect), rect)
        self.assertEqual(image_fill_uv_rect(image, rect, (0.0, 0.0, 1.0, 1.0)), (0.0, 0.35, 1.0, 0.65))
        self.assertEqual(ui_quad_vertices(image, 800, 600, rect=rect)[:5], [10.0, 68.0, 0.0, 0.0, 0.35])

    def test_ui_text_rect_preserves_texture_aspect_in_narrow_layout_box(self):
        component = UIText(text="Score", alignment="center")
        rect = (10.0, 20.0, 20.0, 200.0)
        vertices = ui_text_vertices(component, 800, 600, rect=rect, texture_size=(120, 30))

        xs = vertices[0::8]
        ys = vertices[1::8]
        self.assertAlmostEqual(max(xs) - min(xs), 20.0)
        self.assertAlmostEqual(max(ys) - min(ys), 5.0)
        self.assertAlmostEqual(min(ys), 117.5)

    def test_ui_text_alignment_offsets_aspect_fit_quad_inside_rect(self):
        rect = (10.0, 20.0, 200.0, 40.0)

        left = text_rect_with_aspect(UIText(alignment="left"), rect, (100, 40))
        center = text_rect_with_aspect(UIText(alignment="center"), rect, (100, 40))
        right = text_rect_with_aspect(UIText(alignment="right"), rect, (100, 40))

        self.assertEqual(left, (10.0, 20.0, 100.0, 40.0))
        self.assertEqual(center, (60.0, 20.0, 100.0, 40.0))
        self.assertEqual(right, (110.0, 20.0, 100.0, 40.0))

    def test_particle_emitter_spawns_and_particle_vertices_fade(self):
        emitter = ParticleEmitter(max_particles=2, lifetime=1.0, start_size=1.0)
        emitter.emit(3)
        emitter._runtime_particles[0]["age"] = 0.5
        entity = Entity("Emitter", components=[emitter])
        vertices = particle_quad_vertices(entity, emitter, RenderCamera(Vec3(0, 0, 5), Vec3()))

        self.assertEqual(len(emitter._runtime_particles), 2)
        self.assertEqual(len(vertices), 2 * 6 * 8)
        self.assertIn(0.5, vertices[5::8])

    def test_ui_text_texture_prefers_bitmap_font_reference(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            font = project.assets_dir / "ui.ttf"
            font.write_bytes(b"not a real font")
            renderer = _renderer(project)
            component = UIText(text="Score", font_source="asset", bitmap_font="assets/ui.ttf", font_family="MissingFont")

            class FakeImage:
                def __init__(self, size):
                    self.size = size

                def tobytes(self):
                    return b"\xff" * (self.size[0] * self.size[1] * 4)

                def putalpha(self, _mask):
                    return None

                def transpose(self, _mode):
                    return self

            class FakeDraw:
                def textbbox(self, _position, _text, font=None):
                    return (0, 0, 32, 12)

                def text(self, *_args, **_kwargs):
                    return None

            image_module = type("ImageModule", (), {})()
            image_module.Transpose = type("Transpose", (), {"FLIP_TOP_BOTTOM": object()})
            image_module.new = mock.Mock(side_effect=lambda _mode, size, *_args: FakeImage(size))
            image_module.frombytes = mock.Mock(side_effect=lambda _mode, size, _data: FakeImage(size))
            image_draw_module = type("ImageDrawModule", (), {"Draw": mock.Mock(return_value=FakeDraw())})()
            image_font_module = type("ImageFontModule", (), {})()
            image_font_module.truetype = mock.Mock(return_value=object())
            image_font_module.load_default = mock.Mock(return_value=object())
            pil_module = type("PILModule", (), {})()
            pil_module.Image = image_module
            pil_module.ImageDraw = image_draw_module
            pil_module.ImageFont = image_font_module

            modules = {
                "PIL": pil_module,
                "PIL.Image": image_module,
                "PIL.ImageDraw": image_draw_module,
                "PIL.ImageFont": image_font_module,
            }
            with mock.patch.dict("sys.modules", modules):
                text_texture = renderer._text_texture(component)

            self.assertEqual(image_font_module.truetype.call_args.args[0], str(font))
            self.assertEqual(text_texture.size, (36, 16))

    def test_ui_text_texture_system_font_ignores_bitmap_font_reference(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            font = project.assets_dir / "ui.ttf"
            font.write_bytes(b"not a real font")
            renderer = _renderer(project)
            component = UIText(text="Score", font_source="system", bitmap_font="assets/ui.ttf", font_family="System")

            class FakeImage:
                def __init__(self, size):
                    self.size = size

                def tobytes(self):
                    return b"\xff" * (self.size[0] * self.size[1] * 4)

                def putalpha(self, _mask):
                    return None

                def transpose(self, _mode):
                    return self

            class FakeDraw:
                def textbbox(self, _position, _text, font=None):
                    return (0, 0, 32, 12)

                def text(self, *_args, **_kwargs):
                    return None

            image_module = type("ImageModule", (), {})()
            image_module.Transpose = type("Transpose", (), {"FLIP_TOP_BOTTOM": object()})
            image_module.new = mock.Mock(side_effect=lambda _mode, size, *_args: FakeImage(size))
            image_module.frombytes = mock.Mock(side_effect=lambda _mode, size, _data: FakeImage(size))
            image_draw_module = type("ImageDrawModule", (), {"Draw": mock.Mock(return_value=FakeDraw())})()
            image_font_module = type("ImageFontModule", (), {})()
            image_font_module.truetype = mock.Mock(return_value=object())
            image_font_module.load_default = mock.Mock(return_value=object())
            pil_module = type("PILModule", (), {})()
            pil_module.Image = image_module
            pil_module.ImageDraw = image_draw_module
            pil_module.ImageFont = image_font_module

            modules = {
                "PIL": pil_module,
                "PIL.Image": image_module,
                "PIL.ImageDraw": image_draw_module,
                "PIL.ImageFont": image_font_module,
            }
            with mock.patch.dict("sys.modules", modules):
                renderer._text_texture(component)

            image_font_module.truetype.assert_not_called()
            image_font_module.load_default.assert_called()

    def test_ui_text_texture_uses_alpha_mask_without_opaque_background(self):
        try:
            import PIL  # noqa: F401
        except Exception:
            self.skipTest("PIL is not available")
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            renderer = _renderer(project)

            text_texture = renderer._text_texture(UIText(text="Text", color=Vec3(1.0, 0.0, 0.0), alpha=0.25))

            data = text_texture.texture.data
            alphas = data[3::4]
            self.assertIn(0, alphas)
            self.assertTrue(any(alpha > 0 for alpha in alphas))
            for index in range(0, len(data), 4):
                if data[index + 3] == 0:
                    self.assertEqual(data[index:index + 4], b"\x00\x00\x00\x00")
                    break
            else:
                self.fail("Text atlas did not contain a transparent background pixel")

    def test_ui_text_texture_final_fallback_is_transparent_not_white(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            renderer = _renderer(project)
            blocked_modules = {
                "PIL": None,
                "PIL.Image": None,
                "PIL.ImageDraw": None,
                "PIL.ImageFont": None,
                "PySide6": None,
                "PySide6.QtCore": None,
                "PySide6.QtGui": None,
            }

            with mock.patch.dict("sys.modules", blocked_modules):
                text_texture = renderer._text_texture(UIText(text="Text"))

            self.assertEqual(text_texture.texture.data, b"\x00\x00\x00\x00")

    def test_canvas_fixed_layout_uses_reference_resolution(self):
        auto = Canvas(resolution_mode="auto", reference_resolution=Vec3(640, 480, 0))
        fixed = Canvas(resolution_mode="fixed", reference_resolution=Vec3(640, 480, 0))

        self.assertEqual(_canvas_layout_size(auto, 800, 600), (800, 600))
        self.assertEqual(_canvas_layout_size(fixed, 800, 600), (640, 480))

    def test_fixed_canvas_reference_resolution_drives_game_render_size(self):
        canvas = Entity("Canvas", components=[Canvas(resolution_mode="fixed", reference_resolution=Vec3(1280, 720, 0))])
        scene = Scene("UI", [canvas], render_settings={"internal_resolution": [320, 240]})

        self.assertEqual(_game_render_size(scene, scene.render_settings), (1280, 720))

    def test_game_view_without_active_camera_does_not_use_fallback_camera(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            renderer = _renderer(project)
            scene = Scene("No Camera", [Entity("Mesh", components=[MeshRenderer(mesh="missing")])])

            rendered = renderer.render(scene, 320, 240, game_view=True)

            self.assertFalse(rendered)
            self.assertEqual(renderer.ctx.vertex_array_count, 0)

    def test_game_view_ignores_camera_under_inactive_parent(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            renderer = _renderer(project)
            parent = Entity("Inactive", active=False)
            parent.add_child(Entity("Camera", components=[Camera(active=True)]))
            scene = Scene("Inactive Camera", [parent])

            rendered = renderer.render(scene, 320, 240, game_view=True)

            self.assertFalse(rendered)

    def test_ui_components_draw_in_component_order(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            renderer = _renderer(project)
            image_texture = object()
            text_texture = object()
            calls: list[str] = []
            entity = Entity("Label", components=[UIImage(), UIText(text="Text")])
            scene = Scene("UI", [entity])

            renderer._component_texture = lambda *_args: image_texture
            renderer._text_texture = lambda _component: TextTexture(text_texture, (40, 20))
            renderer._draw_ui_quad_batch = lambda _vertices, _program, texture, *_args: calls.append("text" if texture is text_texture else "image")

            renderer._draw_ui(scene, 320, 240)

            self.assertEqual(calls, ["image", "text"])

    def test_ui_quad_batch_does_not_set_projection_uniform(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            renderer = _renderer(project)
            program = FakeProgram({"in_position", "in_uv", "in_color"})
            vertices = ui_quad_vertices(UIImage(), 320, 240, rect=(0.0, 0.0, 320.0, 240.0))

            renderer._draw_ui_quad_batch(vertices, program, FakeTexture(), 320, 240, Vec3(1.0, 1.0, 1.0), 1.0, None)

            self.assertNotIn("u_projection", program.uniforms)
            self.assertIn("u_texture", program.uniforms)

    def test_textured_quad_batch_still_sets_projection_uniform(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            renderer = _renderer(project)
            program = FakeProgram({"in_position", "in_uv", "in_color"})
            vertices = ui_quad_vertices(UIImage(), 320, 240, rect=(0.0, 0.0, 32.0, 32.0))

            renderer._draw_textured_quad_batch(vertices, program, FakeTexture(), _identity_matrix(), _identity_matrix(), _identity_matrix(), Vec3(1.0, 1.0, 1.0), 1.0, None)

            self.assertIn("u_projection", program.uniforms)

    def test_skybox_gradient_and_cloud_plane_shader_are_split(self):
        self.assertNotIn("value_noise", SKYBOX_FRAGMENT_SHADER)
        self.assertIn("value_noise", CLOUD_PLANE_FRAGMENT_SHADER)
        self.assertIn("fbm", CLOUD_PLANE_FRAGMENT_SHADER)
        self.assertIn("u_cloud_origin", CLOUD_PLANE_VERTEX_SHADER)
        self.assertIn("v_dome_height", CLOUD_PLANE_FRAGMENT_SHADER)
        self.assertIn("horizon_fade", CLOUD_PLANE_FRAGMENT_SHADER)
        self.assertIn("u_skybox_cloud_height", CLOUD_PLANE_FRAGMENT_SHADER)
        self.assertIn("u_skybox_cloud_softness", CLOUD_PLANE_FRAGMENT_SHADER)
        self.assertIn("u_color_levels", CLOUD_PLANE_FRAGMENT_SHADER)
        self.assertIn("u_dithering_enabled", CLOUD_PLANE_FRAGMENT_SHADER)
        self.assertIn("fragColor = vec4", CLOUD_PLANE_FRAGMENT_SHADER)

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
            with mock.patch("p64.renderer.scene_renderer.clear_mesh_geometry_cache") as clear_cache:
                renderer.reload_assets()
            clear_cache.assert_called_once_with(project)
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
        self.fail_enable = False
        self.fail_disable = False
        self.fail_depth_mask = False
        self.fail_blend_func = False
        self.fail_blend_func_not_implemented = False
        self._depth_mask = True
        self._blend_func = None

    @property
    def depth_mask(self):
        return self._depth_mask

    @depth_mask.setter
    def depth_mask(self, value):
        if self.fail_depth_mask:
            raise RuntimeError("depth_mask boom")
        self._depth_mask = value

    @property
    def blend_func(self):
        if self.fail_blend_func_not_implemented:
            raise NotImplementedError()
        return self._blend_func

    @blend_func.setter
    def blend_func(self, value):
        if self.fail_blend_func_not_implemented:
            raise NotImplementedError()
        if self.fail_blend_func:
            raise RuntimeError("blend_func boom")
        self._blend_func = value

    def program(self, vertex_shader: str, fragment_shader: str):
        self.calls += 1
        if "bad glsl" in vertex_shader or "bad glsl" in fragment_shader:
            raise RuntimeError("compile failed")
        return FakeProgram(_shader_attributes(vertex_shader))

    def buffer(self, data: bytes):
        self.buffer_count += 1
        return FakeBuffer(data)

    def vertex_array(self, program: object, bindings: list[object], skip_errors: bool = False):
        attributes = getattr(program, "attributes", None)
        if attributes is not None:
            for _buffer, _layout, *names in bindings:
                for name in names:
                    if name in attributes:
                        continue
                    if skip_errors and name != "in_position":
                        continue
                    raise KeyError(name)
        self.vertex_array_count += 1
        vao = FakeVao(program, bindings=bindings, skip_errors=skip_errors)
        self.vaos.append(vao)
        return vao

    def texture(self, size: tuple[int, int], components: int, data: bytes):
        return FakeTexture(size, components, data)

    def clear(self, red: float, green: float, blue: float, alpha: float) -> None:
        pass

    def enable(self, flag: object) -> None:
        if self.fail_enable:
            raise RuntimeError("enable boom")
        self.enabled.append(flag)

    def disable(self, flag: object) -> None:
        if self.fail_disable:
            raise RuntimeError("disable boom")
        self.disabled.append(flag)


class FakeUniform:
    def __init__(self, name: str):
        self.name = name
        self.value = None

    def write(self, value: bytes) -> None:
        self.value = value


class FakeProgram:
    def __init__(self, attributes: set[str] | None = None):
        self.attributes = attributes or set()
        self.uniforms: dict[str, FakeUniform] = {}

    def __getitem__(self, name: str) -> FakeUniform:
        if name not in self.uniforms:
            self.uniforms[name] = FakeUniform(name)
        return self.uniforms[name]


class FakeBuffer:
    def __init__(self, data: bytes):
        self.data = data

    def release(self) -> None:
        pass


class FakeTexture:
    filter = None

    def __init__(self, size: tuple[int, int] = (1, 1), components: int = 4, data: bytes = b""):
        self.size = size
        self.components = components
        self.data = data

    def use(self, location: int = 0) -> None:
        pass


class FakeVao:
    def __init__(self, program: object | None = None, bindings: list[object] | None = None, skip_errors: bool = False):
        self.program = program
        self.bindings = bindings or []
        self.skip_errors = skip_errors
        self.render_count = 0

    def render(self, mode: object | None = None) -> None:
        self.render_count += 1

    def release(self) -> None:
        pass


def _renderer(project: Project, log: object | None = None) -> SceneRenderer:
    previous = sys.modules.get("moderngl")
    sys.modules["moderngl"] = types.SimpleNamespace(
        DEPTH_TEST=1,
        LINES=1,
        NEAREST=1,
        CULL_FACE=2,
        BLEND=3,
        SRC_ALPHA=4,
        ONE_MINUS_SRC_ALPHA=5,
        ONE=6,
    )
    try:
        return SceneRenderer(FakeContext(), project, log)
    finally:
        if previous is None:
            sys.modules.pop("moderngl", None)
        else:
            sys.modules["moderngl"] = previous


def _shader_attributes(vertex_shader: str) -> set[str]:
    attributes: set[str] = set()
    for line in vertex_shader.splitlines():
        line = line.strip()
        if not line.startswith("in "):
            continue
        parts = line.replace(";", "").split()
        if len(parts) >= 3:
            attributes.add(parts[2])
    return attributes


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
