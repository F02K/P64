from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


def _load_build_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "build_tool" / "build_tool.py"
    spec = importlib.util.spec_from_file_location("p64_build_tool", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load build tool")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildToolTests(unittest.TestCase):
    def setUp(self):
        self.tool = _load_build_tool()
        self.root = Path("E:/Repo").resolve()

    def test_no_arguments_launches_gui_mode(self):
        self.assertEqual(self.tool.GUI_TARGET, "ui")
        self.assertEqual(self.tool.CLI_TARGETS, {"app", "hub", "all", "test"})

    def test_app_target_builds_only_p64_app(self):
        args = self.tool._parse_args(["app", "--skip-pyinstaller"])
        steps = self.tool._build_steps(args, self.root)

        self.assertEqual(args.target, "app")
        self.assertEqual([step.title for step in steps], ["Build P64 App"])
        self.assertEqual(steps[0].command[2:], ["p64", "build-hub", "--skip-pyinstaller"])

    def test_hub_target_aliases_app(self):
        args = self.tool._parse_args(["hub", "--out", "dist/P64"])
        steps = self.tool._build_steps(args, self.root)

        self.assertEqual(args.target, "app")
        self.assertEqual(args.raw_target, "hub")
        self.assertEqual([step.title for step in steps], ["Build P64 App"])
        self.assertEqual(steps[0].command[-2:], ["--out", "dist/P64"])

    def test_all_target_runs_tests_then_p64_app(self):
        args = self.tool._parse_args(["all"])
        steps = self.tool._build_steps(args, self.root)

        self.assertEqual([step.title for step in steps], ["Run tests", "Build P64 App"])
        self.assertEqual(steps[0].command[2:], ["unittest", "discover", "-v", "tests"])
        flattened = " ".join(" ".join(step.command) for step in steps)
        self.assertNotIn("samples", flattened)
        self.assertNotIn("FirstScene", flattened)
        self.assertNotIn(" build ", flattened)

    def test_test_target_uses_verbose_test_names(self):
        args = self.tool._parse_args(["test"])
        steps = self.tool._build_steps(args, self.root)

        self.assertEqual([step.title for step in steps], ["Run tests"])
        self.assertEqual(steps[0].command[2:], ["unittest", "discover", "-v", "tests"])

    def test_test_output_parser_recognizes_results(self):
        self.assertEqual(self.tool._parse_test_result_line("test_a (tests.test_x.X.test_a) ... ok"), "passed")
        self.assertEqual(self.tool._parse_test_result_line("test_b (tests.test_x.X.test_b) ... FAIL"), "failed")
        self.assertEqual(self.tool._parse_test_result_line("test_c (tests.test_x.X.test_c) ... ERROR"), "errors")
        self.assertEqual(self.tool._parse_test_result_line("test_d (tests.test_x.X.test_d) ... skipped 'missing dep'"), "skipped")
        self.assertIsNone(self.tool._parse_test_result_line("Ran 10 tests in 1.0s"))

    def test_apply_test_output_counts_results(self):
        progress = self.tool.TestProgress()

        self.tool._apply_test_output(
            progress,
            "\n".join([
                "test_a (tests.test_x.X.test_a) ... ok",
                "ordinary log line",
                "test_b (tests.test_x.X.test_b) ... FAIL",
                "test_c (tests.test_x.X.test_c) ... ERROR",
                "test_d (tests.test_x.X.test_d) ... skipped 'missing dep'",
            ]),
        )

        self.assertEqual(progress.seen, 4)
        self.assertEqual(progress.passed, 1)
        self.assertEqual(progress.failed, 1)
        self.assertEqual(progress.errors, 1)
        self.assertEqual(progress.skipped, 1)

    def test_step_progress_counts_terminal_states(self):
        states = [
            self.tool.StepState("A", status="Passed"),
            self.tool.StepState("B", status="Running"),
            self.tool.StepState("C", status="Cancelled"),
            self.tool.StepState("D", status="Failed"),
        ]

        self.assertEqual(self.tool._step_progress_value(states), (3, 4))

    def test_format_duration_uses_minutes_or_hours(self):
        self.assertEqual(self.tool._format_duration(3.9), "00:03")
        self.assertEqual(self.tool._format_duration(72), "01:12")
        self.assertEqual(self.tool._format_duration(4329), "01:12:09")

    def test_all_target_can_skip_tests(self):
        args = self.tool._parse_args(["all", "--skip-tests"])
        steps = self.tool._build_steps(args, self.root)

        self.assertEqual([step.title for step in steps], ["Build P64 App"])

    def test_game_target_is_explicitly_unsupported_from_root_build(self):
        args = self.tool._parse_args(["game"])

        self.assertIn(args.target, self.tool.UNSUPPORTED_ROOT_TARGETS)
        self.assertEqual(self.tool._build_steps(args, self.root), [])

    def test_gui_args_use_same_step_builder(self):
        args = self.tool._gui_args("app", "dist/P64", skip_tests=False, skip_pyinstaller=True)
        steps = self.tool._build_steps(args, self.root)

        self.assertEqual([step.title for step in steps], ["Build P64 App"])
        self.assertEqual(steps[0].command[-3:], ["--out", "dist/P64", "--skip-pyinstaller"])


if __name__ == "__main__":
    unittest.main()
