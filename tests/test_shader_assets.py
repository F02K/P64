from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from p64.engine.shader import discover_shaders, parse_shader, shader_asset_id


class ShaderAssetTests(unittest.TestCase):
    def test_parse_unity_style_shader_sections(self):
        shader = parse_shader(Path("samples/FirstScene/assets/shaders/n64_textured.shader"))
        self.assertEqual(shader.name, "P64/N64Textured")
        self.assertIn("in_position", shader.vertex)
        self.assertIn("fragColor", shader.fragment)

    def test_parse_colon_shader_sections(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "simple.shader"
            path.write_text(
                'Shader "P64/Test"\n'
                "vertex:\n"
                "#version 330\n"
                "void main() { gl_Position = vec4(0.0); }\n"
                "fragment:\n"
                "#version 330\n"
                "out vec4 fragColor;\n"
                "void main() { fragColor = vec4(1.0); }\n",
                encoding="utf-8",
            )
            shader = parse_shader(path)
            self.assertEqual(shader.name, "P64/Test")
            self.assertIn("gl_Position", shader.vertex)
            self.assertIn("fragColor", shader.fragment)

    def test_discover_shader_assets_and_ids(self):
        root = Path("samples/FirstScene")
        shaders = discover_shaders(root / "assets")
        self.assertIn(root / "assets/shaders/n64_textured.shader", shaders)
        self.assertEqual(shader_asset_id(root, shaders[0]), "assets/shaders/n64_textured.shader")


if __name__ == "__main__":
    unittest.main()
