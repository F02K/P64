from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from p64.engine.assets import discover_assets
from p64.engine.obj import import_obj_to_project, mesh_vertices_for_group, parse_mtl, parse_obj
from p64.engine.project import Project


OBJ_TEXT = """
o Floor
v 0 0 0
v 1 0 0
v 1 0 1
v 0 0 1
f 1 2 3 4
g Door
v 0 0 0
v 0 1 0
v 1 1 0
f 5 6 7
"""


class ObjImportTests(unittest.TestCase):
    def test_parse_obj_preserves_groups_and_triangulates(self):
        with TemporaryDirectory() as tmp:
            obj = Path(tmp) / "scene.obj"
            obj.write_text(OBJ_TEXT, encoding="utf-8")
            mesh = parse_obj(obj)

            self.assertEqual(mesh.group_names, ["Floor", "Door"])
            self.assertEqual(len(mesh.groups[0].faces), 2)
            self.assertEqual(len(mesh.groups[1].faces), 1)

    def test_parse_obj_preserves_optional_vertex_colors(self):
        with TemporaryDirectory() as tmp:
            obj = Path(tmp) / "colored.obj"
            obj.write_text(
                "o Body\n"
                "v 0 0 0 1 0.5 0\n"
                "v 1 0 0 0 1 0.5\n"
                "v 0 1 0 0.25 0 1\n"
                "f 1 2 3\n",
                encoding="utf-8",
            )
            mesh = parse_obj(obj)
            vertices = mesh.groups[0].faces[0].vertices

            self.assertEqual(vertices[0].color, (1.0, 0.5, 0.0))
            self.assertEqual(vertices[1].color, (0.0, 1.0, 0.5))
            self.assertEqual(vertices[2].color, (0.25, 0.0, 1.0))

            packed = mesh_vertices_for_group(mesh.groups[0])
            self.assertEqual(packed[8:11], [1.0, 0.5, 0.0])

    def test_parse_obj_accepts_byte_sized_vertex_colors(self):
        with TemporaryDirectory() as tmp:
            obj = Path(tmp) / "colored.obj"
            obj.write_text(
                "o Body\n"
                "v 0 0 0 255 128 0\n"
                "v 1 0 0 0 255 128\n"
                "v 0 1 0 64 0 255\n"
                "f 1 2 3\n",
                encoding="utf-8",
            )
            mesh = parse_obj(obj)

            self.assertEqual(mesh.groups[0].faces[0].vertices[0].color, (1.0, 128.0 / 255.0, 0.0))

    def test_mesh_vertex_data_defaults_color_to_white(self):
        with TemporaryDirectory() as tmp:
            obj = Path(tmp) / "plain.obj"
            obj.write_text(
                "o Body\n"
                "v 0 0 0\n"
                "v 1 0 0\n"
                "v 0 1 0\n"
                "f 1 2 3\n",
                encoding="utf-8",
            )
            mesh = parse_obj(obj)

            vertices = mesh_vertices_for_group(mesh.groups[0])

            self.assertEqual(len(vertices), 33)
            self.assertEqual(vertices[8:11], [1.0, 1.0, 1.0])

    def test_import_obj_adds_metadata_and_scene_nodes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = root / "scene.obj"
            obj.write_text(OBJ_TEXT, encoding="utf-8")
            project = Project.create(root / "Game")

            metadata = import_obj_to_project(project, obj, add_to_startup_scene=True)
            scene = project.load_startup_scene()
            imported = scene.entities[-1]

            self.assertEqual(metadata.groups, ["Floor", "Door"])
            self.assertTrue((project.assets_dir / "scene.obj.mdp64").exists())
            self.assertTrue(imported.is_game_object)
            self.assertEqual([child.name for child in imported.children], ["Floor", "Door"])
            self.assertTrue(all(child.is_game_object for child in imported.children))

    def test_mtl_parser_resolves_diffuse_texture(self):
        mtl = Path("samples/FirstScene/assets/Ocarina/model.mtl")
        if not mtl.exists():
            self.skipTest("Ocarina sample asset is not present")
        materials = parse_mtl(mtl)
        self.assertEqual(materials["material_0"].diffuse_texture, "0.png")

    def test_ocarina_obj_has_renderable_geometry(self):
        obj = Path("samples/FirstScene/assets/Ocarina/model.obj")
        if not obj.exists():
            self.skipTest("Ocarina sample asset is not present")
        mesh = parse_obj(obj)
        self.assertIn("0", mesh.group_names)
        self.assertGreater(len(mesh.groups[0].faces), 0)
        first_vertex = mesh.groups[0].faces[0].vertices[0]
        self.assertIsNotNone(first_vertex.texcoord)
        self.assertIsNotNone(first_vertex.normal)

    def test_recursive_asset_discovery_sees_ocarina_files(self):
        assets = [path.as_posix() for path in discover_assets(Path("samples/FirstScene/assets"))]
        self.assertIn("samples/FirstScene/assets/Ocarina/model.obj", assets)
        self.assertIn("samples/FirstScene/assets/Ocarina/0.png", assets)


if __name__ == "__main__":
    unittest.main()
