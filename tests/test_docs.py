import ast
import re
import unittest
from pathlib import Path

from p64.engine import scripting
from p64.engine.input import InputState


DOCS = Path("docs")


class DocumentationTests(unittest.TestCase):
    def test_readme_links_to_documentation(self):
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("[documentation index](docs/README.md)", readme)
        self.assertIn("[scripting guide](docs/scripting.md)", readme)

    def test_docs_index_links_exist(self):
        index = DOCS / "README.md"
        self.assertTrue(index.exists())
        for target in _markdown_links(index):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = (index.parent / target.split("#", 1)[0]).resolve()
            self.assertTrue(path.exists(), f"Missing docs link target: {target}")

    def test_all_docs_markdown_links_exist(self):
        for source in DOCS.glob("*.md"):
            for target in _markdown_links(source):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                path = (source.parent / target.split("#", 1)[0]).resolve()
                self.assertTrue(path.exists(), f"{source} links to missing target {target}")

    def test_scripting_python_examples_parse(self):
        text = (DOCS / "scripting.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)

        self.assertGreaterEqual(len(blocks), 8)
        for block in blocks:
            ast.parse(block)

    def test_scripting_docs_match_real_api_names(self):
        text = (DOCS / "scripting.md").read_text(encoding="utf-8")

        self.assertTrue(hasattr(scripting, "GameScript"))
        self.assertFalse(hasattr(scripting, "UserScript"))
        self.assertTrue(hasattr(scripting.GameScript, "persistent"))
        self.assertTrue(hasattr(scripting.GameScript, "move_character"))
        self.assertTrue(hasattr(scripting.GameScript, "forward"))
        self.assertTrue(hasattr(scripting.GameScript, "right"))
        self.assertTrue(hasattr(scripting.GameScript, "up"))
        self.assertTrue(hasattr(scripting.GameScript, "on_start"))
        self.assertTrue(hasattr(scripting.GameScript, "on_update"))
        for name in [
            "self.entity",
            "self.transform",
            "self.scene",
            "self.project",
            "self.scene_manager",
            "self.input",
            "self.time",
            "self.character_controller",
            "self.entity_physics",
        ]:
            self.assertIn(name, text)
        for name in [
            "is_key_down",
            "was_key_pressed",
            "was_key_released",
            "get_axis",
            "is_button_down",
        ]:
            self.assertTrue(hasattr(InputState, name))
            self.assertIn(name, text)


def _markdown_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [
        match.group(1)
        for match in re.finditer(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text)
        if not match.group(1).startswith("<")
    ]


if __name__ == "__main__":
    unittest.main()
