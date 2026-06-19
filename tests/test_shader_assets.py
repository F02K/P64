from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from p64.engine.shader import discover_shaders, parse_shader, shader_asset_id
from p64.engine.builtin import standard_unlit_shader_source, standard_vertex_lit_shader_source
from p64.renderer.shaders import (
    STANDARD_UNLIT_FRAGMENT_SHADER,
    STANDARD_UNLIT_VERTEX_SHADER,
    STANDARD_VERTEX_LIT_FRAGMENT_SHADER,
    STANDARD_VERTEX_LIT_VERTEX_SHADER,
)


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

    def test_parse_shader_properties(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "props.shader"
            path.write_text(
                'Shader "P64/Props"\n'
                "{\n"
                "Properties {\n"
                "    Texture u_texture = \"albedo.png\"\n"
                "    Color u_base_color = (0.25, 0.5, 0.75)\n"
                "    Float u_alpha_cutoff = 0.5 Range(0, 1)\n"
                "}\n"
                "Vertex { #version 330\nvoid main() { gl_Position = vec4(0.0); } }\n"
                "Fragment { #version 330\nout vec4 fragColor;\nvoid main() { fragColor = vec4(1.0); } }\n"
                "}\n",
                encoding="utf-8",
            )

            shader = parse_shader(path)

            self.assertEqual([prop.name for prop in shader.properties], ["u_texture", "u_base_color", "u_alpha_cutoff"])
            self.assertEqual(shader.properties[1].default, [0.25, 0.5, 0.75])
            self.assertEqual(shader.properties[2].minimum, 0.0)
            self.assertEqual(shader.properties[2].maximum, 1.0)

    def test_generated_builtin_shaders_declare_material_properties(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            vertex_lit = root / "standard_vertex_lit.shader"
            unlit = root / "standard_unlit.shader"
            vertex_lit.write_text(standard_vertex_lit_shader_source(), encoding="utf-8")
            unlit.write_text(standard_unlit_shader_source(), encoding="utf-8")

            for path in [vertex_lit, unlit]:
                shader = parse_shader(path)
                properties = {prop.name: prop for prop in shader.properties}
                self.assertIn("u_texture", properties)
                self.assertIn("u_base_color", properties)
                self.assertIn("u_alpha_cutoff", properties)
                self.assertEqual(properties["u_texture"].kind, "texture")
                self.assertEqual(properties["u_base_color"].default, [1.0, 1.0, 1.0])
                self.assertEqual(properties["u_alpha_cutoff"].default, 0.0)
                self.assertEqual(properties["u_alpha_cutoff"].minimum, 0.0)
                self.assertEqual(properties["u_alpha_cutoff"].maximum, 1.0)

    def test_discover_shader_assets_and_ids(self):
        root = Path("samples/FirstScene")
        shaders = discover_shaders(root / "assets")
        self.assertIn(root / "assets/shaders/standard_textured.shader", shaders)
        self.assertIn(root / "packages/P64Builtin/shaders/standard_vertex_lit.shader", shaders)
        self.assertIn(root / "packages/P64Builtin/shaders/standard_unlit.shader", shaders)
        self.assertIn("assets/shaders/standard_textured.shader", [shader_asset_id(root, shader) for shader in shaders])

    def test_builtin_shaders_use_base_material_color(self):
        self.assertIn("in vec3 in_color", STANDARD_VERTEX_LIT_VERTEX_SHADER)
        self.assertIn("in vec3 in_color", STANDARD_UNLIT_VERTEX_SHADER)
        self.assertIn("out vec3 v_color", STANDARD_VERTEX_LIT_VERTEX_SHADER)
        self.assertIn("out vec3 v_color", STANDARD_UNLIT_VERTEX_SHADER)
        self.assertIn("in vec3 v_color", STANDARD_VERTEX_LIT_FRAGMENT_SHADER)
        self.assertIn("in vec3 v_color", STANDARD_UNLIT_FRAGMENT_SHADER)
        self.assertIn("uniform vec3 u_base_color", STANDARD_VERTEX_LIT_FRAGMENT_SHADER)
        self.assertIn("texel.rgb * u_base_color * v_color * v_light", STANDARD_VERTEX_LIT_FRAGMENT_SHADER)
        self.assertIn("uniform vec3 u_base_color", STANDARD_UNLIT_FRAGMENT_SHADER)
        self.assertIn("texel.rgb * u_base_color * v_color", STANDARD_UNLIT_FRAGMENT_SHADER)

    def test_builtin_shaders_wire_dithering_into_quantization(self):
        self.assertIn("uniform bool u_dithering_enabled", STANDARD_VERTEX_LIT_FRAGMENT_SHADER)
        self.assertIn("uniform bool u_dithering_enabled", STANDARD_UNLIT_FRAGMENT_SHADER)
        self.assertIn("dither_threshold(gl_FragCoord.xy)", STANDARD_VERTEX_LIT_FRAGMENT_SHADER)
        self.assertIn("dither_threshold(gl_FragCoord.xy)", STANDARD_UNLIT_FRAGMENT_SHADER)
        self.assertIn("quantize_color(lit)", STANDARD_VERTEX_LIT_FRAGMENT_SHADER)
        self.assertIn("quantize_color(texel.rgb * u_base_color * v_color)", STANDARD_UNLIT_FRAGMENT_SHADER)

    def test_builtin_shaders_use_alpha_cutout(self):
        for fragment in [STANDARD_VERTEX_LIT_FRAGMENT_SHADER, STANDARD_UNLIT_FRAGMENT_SHADER]:
            self.assertIn("uniform float u_alpha_cutoff", fragment)
            self.assertIn("if (texel.a < u_alpha_cutoff)", fragment)
            self.assertIn("discard", fragment)

    def test_no_console_specific_names_outside_readme_or_generated_assets(self):
        forbidden = ("n" + "64").lower()
        ignored_parts = {
            ".git",
            "build",
            "__pycache__",
            ".pytest_cache",
            ".venv",
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
