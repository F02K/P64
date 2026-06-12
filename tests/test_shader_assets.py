from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from p64.engine.shader import discover_shaders, parse_shader, shader_asset_id


class ShaderAssetTests(unittest.TestCase):
    def test_parse_unity_style_shader_sections(self):
        shader = parse_shader(Path("samples/FirstScene/assets/shaders/standard_textured.shader"))
        self.assertEqual(shader.name, "P64/Standard Textured")
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
        self.assertIn(root / "assets/shaders/standard_textured.shader", shaders)
        self.assertIn(root / "packages/P64Builtin/shaders/standard_vertex_lit.shader", shaders)
        self.assertIn(root / "packages/P64Builtin/shaders/standard_unlit.shader", shaders)
        self.assertIn("assets/shaders/standard_textured.shader", [shader_asset_id(root, shader) for shader in shaders])

    def test_no_console_specific_names_outside_readme_or_generated_assets(self):
        forbidden = ("n" + "64").lower()
        ignored_parts = {
            ".git",
            "build",
            "__pycache__",
            ".pytest_cache",
            "libraries",
        }
        offenders: list[str] = []
        for path in Path(".").rglob("*"):
            if not path.is_file():
                continue
            parts = set(path.parts)
            if parts & ignored_parts:
                continue
            if path.name.lower().startswith("readme"):
                continue
            if path.suffix.lower() not in {".py", ".toml", ".shader", ".p64", ".scenep64", ".mdp64"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if forbidden in text or forbidden in path.name.lower():
                offenders.append(path.as_posix())
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
