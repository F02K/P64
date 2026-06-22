from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


CLI_TARGETS = {"app", "hub", "all", "test"}
GUI_TARGET = "ui"
UNSUPPORTED_ROOT_TARGETS = {"game", "bundle", "quick", "validate"}


@dataclass(frozen=True)
class Step:
    title: str
    command: list[str]


@dataclass
class TestProgress:
    seen: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0


@dataclass
class StepState:
    title: str
    status: str = "Pending"
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    test_progress: TestProgress = field(default_factory=TestProgress)


TEST_RESULT_RE = re.compile(r"^.+\s\([^)]+\)\s\.\.\.\s(?P<result>ok|FAIL|ERROR|skipped\b.*)$")


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        return _launch_gui()

    args = _parse_args(raw_args)
    if args.target == GUI_TARGET:
        return _launch_gui(args)
    if args.target in UNSUPPORTED_ROOT_TARGETS:
        _print_unsupported_target(args.target)
        return 2

    root = _repo_root()
    env = _build_env(root)
    if args.no_pause:
        env["P64_BUILD_TOOL_NO_PAUSE"] = "1"

    steps = _build_steps(args, root)
    if not steps:
        print("Nothing to build.")
        return 0

    _print_header(args.target, root, steps)
    exit_code = _run_steps(steps, root, env)
    if exit_code == 0:
        _print_success(args.target, root, args)
    else:
        print()
        print(f"Build failed with exit code {exit_code}.")
    _pause_if_interactive(env)
    return exit_code


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    return env


def _parse_args(argv: list[str]) -> argparse.Namespace:
    target = "app"
    if argv and argv[0] in CLI_TARGETS | {GUI_TARGET} | UNSUPPORTED_ROOT_TARGETS:
        target = argv.pop(0)
    elif argv and not argv[0].startswith("-"):
        raise SystemExit(f"Unknown build target: {argv[0]}\nRun build.bat --help to see available root build targets.")

    parser = argparse.ArgumentParser(
        prog="build",
        description="Build the P64 tooling app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Targets:\n"
            "  ui        open the Build Center dashboard\n"
            "  app       build the portable P64 App/Hub executable\n"
            "  hub       alias for app\n"
            "  all       run verbose tests, then build the P64 App\n"
            "  test      run the engine/editor test suite with verbose test names\n\n"
            "Project/game builds are handled from the editor/project build settings.\n"
            "Low-level project CLI remains available as: python -m p64 build <project>\n\n"
            "Examples:\n"
            "  build.bat\n"
            "  build.bat app\n"
            "  build.bat all --skip-tests\n"
            "  build.bat hub --skip-pyinstaller\n"
        ),
    )
    parser.add_argument("--out", help="Output directory for the P64 App/Hub build.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip tests for the all target.")
    parser.add_argument("--skip-pyinstaller", action="store_true", help="Create app build files without running PyInstaller.")
    parser.add_argument("--no-pause", action="store_true", help="Do not wait for Enter when launched from a console window.")
    parser.add_argument("--list-targets", action="store_true", help="Print available targets and exit.")

    args = parser.parse_args(argv)
    if args.list_targets:
        print("\n".join(["ui", "app", "hub", "all", "test"]))
        raise SystemExit(0)
    args.target = "app" if target == "hub" else target
    args.raw_target = target
    return args


def _build_steps(args: argparse.Namespace, root: Path) -> list[Step]:
    python = sys.executable
    p64 = [python, "-m", "p64"]
    steps: list[Step] = []

    if args.target in {"test", "all"} and not args.skip_tests:
        steps.append(Step("Run tests", _test_command(python)))

    if args.target in {"app", "all"}:
        command = [*p64, "build-hub"]
        if args.out:
            command.extend(["--out", args.out])
        if args.skip_pyinstaller:
            command.append("--skip-pyinstaller")
        steps.append(Step("Build P64 App", command))

    return steps


def _test_command(python: str) -> list[str]:
    return [python, "-m", "unittest", "discover", "-v", "tests"]


def _run_steps(steps: list[Step], root: Path, env: dict[str, str]) -> int:
    total_start = time.monotonic()
    for index, step in enumerate(steps, start=1):
        step_start = time.monotonic()
        print(f"[{_format_duration(step_start - total_start)}] START {step.title} [{index}/{len(steps)}]")
        print(f"    {_format_command(step.command)}")
        result = subprocess.run(step.command, cwd=root, env=env)
        elapsed = time.monotonic() - step_start
        if result.returncode != 0:
            print(f"[{_format_duration(time.monotonic() - total_start)}] FAIL {step.title} ({_format_duration(elapsed)}, exit {result.returncode})")
            return result.returncode
        print(f"[{_format_duration(time.monotonic() - total_start)}] PASS {step.title} ({_format_duration(elapsed)})")
        print()
    print(f"Total: {_format_duration(time.monotonic() - total_start)}")
    return 0


def _parse_test_result_line(line: str) -> str | None:
    match = TEST_RESULT_RE.match(line.strip())
    if not match:
        return None
    result = match.group("result")
    if result == "ok":
        return "passed"
    if result == "FAIL":
        return "failed"
    if result == "ERROR":
        return "errors"
    if result.startswith("skipped"):
        return "skipped"
    return None


def _apply_test_output(progress: TestProgress, text: str) -> None:
    for line in text.splitlines():
        result = _parse_test_result_line(line)
        if result is None:
            continue
        progress.seen += 1
        if result == "passed":
            progress.passed += 1
        elif result == "failed":
            progress.failed += 1
        elif result == "errors":
            progress.errors += 1
        elif result == "skipped":
            progress.skipped += 1


def _step_progress_value(states: list[StepState]) -> tuple[int, int]:
    total = max(1, len(states))
    done = sum(1 for state in states if state.status in {"Passed", "Failed", "Cancelled"})
    return done, total


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _launch_gui(args: argparse.Namespace | None = None) -> int:
    try:
        from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QFileDialog,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QProgressBar,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        print("PySide6 is required for the Build Center.")
        print("Use build.bat app, build.bat all, or install project dependencies.")
        print(f"Import error: {exc}")
        return 1

    root = _repo_root()

    class BuildCenterWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.root = root
            self.process: QProcess | None = None
            self.steps: list[Step] = []
            self.step_states: list[StepState] = []
            self.step_index = 0
            self.current_target = "app"
            self.build_started_at: float | None = None
            self.setWindowTitle("P64 Build Center")
            self.resize(900, 620)
            self.timer = QTimer(self)
            self.timer.setInterval(1000)
            self.timer.timeout.connect(self.refresh_status)

            panel = QWidget()
            layout = QVBoxLayout(panel)

            title = QLabel("P64 Build Center")
            title.setStyleSheet("font-size: 24px; font-weight: 700;")
            layout.addWidget(title)

            subtitle = QLabel("Build the P64 tooling app. Game builds are handled from the editor/project build settings.")
            subtitle.setWordWrap(True)
            layout.addWidget(subtitle)

            root_label = QLabel(f"Root: {self.root}")
            root_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(root_label)

            cards = QHBoxLayout()
            for label, target, note in [
                ("Build P64 App", "app", "Create the portable Hub/App executable."),
                ("Run Tests", "test", "Run the engine/editor test suite."),
                ("Full Tooling Build", "all", "Run tests, then build the P64 App."),
            ]:
                button = QPushButton(f"{label}\n{note}")
                button.setMinimumHeight(74)
                button.clicked.connect(lambda checked=False, value=target: self.start_target(value))
                cards.addWidget(button)
            layout.addLayout(cards)

            options = QHBoxLayout()
            self.skip_tests = QCheckBox("Skip Tests")
            self.skip_pyinstaller = QCheckBox("Skip PyInstaller")
            self.output = QLineEdit(str(self.root / "build" / "app" / "P64"))
            browse = QPushButton("Browse")
            browse.clicked.connect(self.choose_output)
            options.addWidget(self.skip_tests)
            options.addWidget(self.skip_pyinstaller)
            options.addWidget(QLabel("Output"))
            options.addWidget(self.output, 1)
            options.addWidget(browse)
            layout.addLayout(options)

            controls = QHBoxLayout()
            self.start_app = QPushButton("Start Build P64 App")
            self.start_app.clicked.connect(lambda: self.start_target("app"))
            self.cancel = QPushButton("Cancel")
            self.cancel.setEnabled(False)
            self.cancel.clicked.connect(self.cancel_build)
            self.open_output = QPushButton("Open Output")
            self.open_output.clicked.connect(self.open_output_folder)
            controls.addWidget(self.start_app)
            controls.addWidget(self.cancel)
            controls.addWidget(self.open_output)
            layout.addLayout(controls)

            self.status = QLabel("Ready")
            layout.addWidget(self.status)

            progress_row = QHBoxLayout()
            progress_row.addWidget(QLabel("Overall"))
            self.overall_progress = QProgressBar()
            self.overall_progress.setRange(0, 1)
            self.overall_progress.setValue(0)
            progress_row.addWidget(self.overall_progress, 1)
            layout.addLayout(progress_row)

            current_row = QHBoxLayout()
            current_row.addWidget(QLabel("Current"))
            self.current_progress = QProgressBar()
            self.current_progress.setRange(0, 1)
            self.current_progress.setValue(0)
            current_row.addWidget(self.current_progress, 1)
            layout.addLayout(current_row)

            self.step_list = QListWidget()
            self.step_list.setMaximumHeight(120)
            layout.addWidget(self.step_list)

            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            layout.addWidget(self.log, 1)

            self.setCentralWidget(panel)

        def choose_output(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Choose P64 App Output Folder", self.output.text())
            if path:
                self.output.setText(path)

        def start_target(self, target: str) -> None:
            if self.process is not None:
                QMessageBox.information(self, "Build running", "A build is already running.")
                return
            self.current_target = target
            args = _gui_args(target, self.output.text().strip(), self.skip_tests.isChecked(), self.skip_pyinstaller.isChecked())
            self.steps = _build_steps(args, self.root)
            if not self.steps:
                self.append_log("Nothing to build.")
                return
            self.step_states = [StepState(step.title) for step in self.steps]
            self.step_index = 0
            self.build_started_at = time.monotonic()
            self.log.clear()
            self.cancel.setEnabled(True)
            self.timer.start()
            self.status.setText(f"Starting {target}...")
            self.reset_progress()
            self.refresh_step_list()
            self.append_event("INFO", f"Target: {target}")
            self.append_event("INFO", f"Root: {self.root}")
            self.run_next_step()

        def run_next_step(self) -> None:
            if self.step_index >= len(self.steps):
                self.timer.stop()
                self.status.setText(f"Build complete in {self.elapsed_text()}.")
                self.cancel.setEnabled(False)
                self.process = None
                self.update_overall_progress()
                self.current_progress.setRange(0, 1)
                self.current_progress.setValue(1)
                self.append_log("")
                self.append_event("PASS", f"Build complete in {self.elapsed_text()}.")
                self.append_output_hint()
                return
            step = self.steps[self.step_index]
            state = self.step_states[self.step_index]
            state.status = "Running"
            state.started_at = time.monotonic()
            state.finished_at = None
            state.exit_code = None
            state.test_progress = TestProgress()
            self.status.setText(f"[{self.step_index + 1}/{len(self.steps)}] {step.title} | elapsed {self.elapsed_text()}")
            self.update_overall_progress()
            self.configure_current_progress(step, state)
            self.refresh_step_list()
            self.append_log("")
            self.append_event("START", f"{step.title} [{self.step_index + 1}/{len(self.steps)}]")
            self.append_log(f"  $ {_format_command(step.command)}")

            process = QProcess(self)
            process.setWorkingDirectory(str(self.root))
            env = _build_env(self.root)
            process_environment = QProcessEnvironment.systemEnvironment()
            for key, value in env.items():
                process_environment.insert(key, value)
            process.setProcessEnvironment(process_environment)
            process.readyReadStandardOutput.connect(lambda: self.append_process_output(process.readAllStandardOutput(), "stdout"))
            process.readyReadStandardError.connect(lambda: self.append_process_output(process.readAllStandardError(), "stderr"))
            process.finished.connect(self.step_finished)
            self.process = process
            process.start(step.command[0], step.command[1:])

        def step_finished(self, exit_code: int, _exit_status: object) -> None:
            state = self.step_states[self.step_index]
            state.finished_at = time.monotonic()
            state.exit_code = int(exit_code)
            if exit_code != 0:
                state.status = "Failed"
                self.timer.stop()
                self.status.setText(f"Build failed with exit code {exit_code} after {self.elapsed_text()}.")
                self.append_log("")
                self.append_event("FAIL", f"{state.title} ({self.step_duration_text(state)}, exit {exit_code})")
                self.cancel.setEnabled(False)
                self.process = None
                self.update_overall_progress()
                self.current_progress.setRange(0, 1)
                self.current_progress.setValue(1)
                self.refresh_step_list()
                return
            state.status = "Passed"
            self.append_event("PASS", f"{state.title} ({self.step_duration_text(state)})")
            self.step_index += 1
            self.process = None
            self.update_overall_progress()
            self.refresh_step_list()
            self.run_next_step()

        def cancel_build(self) -> None:
            if self.process is None:
                return
            self.append_log("")
            self.append_event("CANCEL", "Cancel requested.")
            if 0 <= self.step_index < len(self.step_states):
                state = self.step_states[self.step_index]
                state.status = "Cancelled"
                state.finished_at = time.monotonic()
                self.refresh_step_list()
            self.process.kill()
            self.process = None
            self.cancel.setEnabled(False)
            self.timer.stop()
            self.status.setText(f"Cancelled after {self.elapsed_text()}.")
            self.update_overall_progress()
            self.current_progress.setRange(0, 1)
            self.current_progress.setValue(1)

        def append_process_output(self, data: object, stream: str) -> None:
            text = bytes(data).decode(errors="replace")
            if not text:
                return
            if 0 <= self.step_index < len(self.steps) and self.steps[self.step_index].title == "Run tests":
                state = self.step_states[self.step_index]
                _apply_test_output(state.test_progress, text)
                self.update_test_progress(state)
                self.refresh_step_list()
            prefix = "  !" if stream == "stderr" else "  |"
            for line in text.rstrip().splitlines():
                self.append_log(f"{prefix} {line}")

        def append_log(self, text: str) -> None:
            self.log.appendPlainText(text)
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

        def append_event(self, label: str, message: str) -> None:
            self.append_log(f"[{self.elapsed_text()}] {label} {message}")

        def reset_progress(self) -> None:
            self.overall_progress.setRange(0, max(1, len(self.steps)))
            self.overall_progress.setValue(0)
            self.current_progress.setRange(0, 1)
            self.current_progress.setValue(0)

        def update_overall_progress(self) -> None:
            done, total = _step_progress_value(self.step_states)
            self.overall_progress.setRange(0, total)
            self.overall_progress.setValue(done)

        def configure_current_progress(self, step: Step, state: StepState) -> None:
            if step.title == "Run tests":
                self.current_progress.setRange(0, 0)
                self.current_progress.setFormat("Discovering tests...")
                return
            self.current_progress.setRange(0, 0)
            self.current_progress.setFormat(f"{step.title} running...")

        def update_test_progress(self, state: StepState) -> None:
            progress = state.test_progress
            if progress.seen <= 0:
                self.current_progress.setRange(0, 0)
                self.current_progress.setFormat("Discovering tests...")
                return
            self.current_progress.setRange(0, progress.seen)
            self.current_progress.setValue(progress.seen)
            self.current_progress.setFormat(self.test_progress_text(progress))

        def test_progress_text(self, progress: TestProgress) -> str:
            return f"{progress.seen} tests | ok {progress.passed} | fail {progress.failed} | error {progress.errors} | skipped {progress.skipped}"

        def refresh_status(self) -> None:
            if self.process is None or not (0 <= self.step_index < len(self.step_states)):
                return
            state = self.step_states[self.step_index]
            self.status.setText(f"[{self.step_index + 1}/{len(self.steps)}] {state.title} | elapsed {self.elapsed_text()} | step {self.step_duration_text(state)}")
            self.refresh_step_list()

        def refresh_step_list(self) -> None:
            self.step_list.clear()
            for index, state in enumerate(self.step_states, start=1):
                item = QListWidgetItem(f"{index}. {state.title} - {state.status} - {self.step_duration_text(state)}")
                self.step_list.addItem(item)

        def elapsed_text(self) -> str:
            if self.build_started_at is None:
                return "00:00"
            return _format_duration(time.monotonic() - self.build_started_at)

        def step_duration_text(self, state: StepState) -> str:
            if state.started_at is None:
                return "00:00"
            end = state.finished_at if state.finished_at is not None else time.monotonic()
            return _format_duration(end - state.started_at)

        def append_output_hint(self) -> None:
            if self.current_target in {"app", "all"}:
                self.append_event("INFO", f"Output: {self.output_path()}")

        def output_path(self) -> Path:
            text = self.output.text().strip()
            return Path(text).resolve() if text else self.root / "build" / "app"

        def open_output_folder(self) -> None:
            path = self.output_path()
            path.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    app = QApplication.instance() or QApplication([])
    window = BuildCenterWindow()
    if args is not None and getattr(args, "out", None):
        window.output.setText(str(Path(args.out).resolve()))
    window.show()
    app.exec()
    return 0


def _gui_args(target: str, out: str, skip_tests: bool, skip_pyinstaller: bool) -> argparse.Namespace:
    return argparse.Namespace(
        target="app" if target == "hub" else target,
        raw_target=target,
        out=out or None,
        skip_tests=skip_tests,
        skip_pyinstaller=skip_pyinstaller,
        no_pause=True,
    )


def _print_header(target: str, root: Path, steps: list[Step]) -> None:
    print("P64 Build")
    print()
    print(f"Target: {target}")
    print(f"Root  : {root}")
    print(f"Steps : {len(steps)}")
    print()


def _print_success(target: str, root: Path, args: argparse.Namespace) -> None:
    print("Build complete.")
    print()
    if target in {"app", "all"}:
        print("P64 App:")
        print(f"  {Path(args.out).resolve() if args.out else root / 'build' / 'app' / 'P64'}")
        print()
    if target in {"app", "all"} and not args.skip_pyinstaller:
        print("Keep P64Hub.exe together with its generated support folders.")


def _print_unsupported_target(target: str) -> None:
    print(f"Root build target '{target}' is no longer supported.")
    print("Game builds are handled from the editor/project build settings.")
    print("Low-level project CLI remains available as: python -m p64 build <project>")


def _pause_if_interactive(env: dict[str, str]) -> None:
    if sys.stdin.isatty() and env.get("P64_BUILD_TOOL_NO_PAUSE") != "1":
        print()
        input("Press Enter to close this build window...")


def _format_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


if __name__ == "__main__":
    raise SystemExit(main())
