from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from p64.engine.project import Project


def open_build_settings_dialog(
    parent: object,
    project: Project,
    on_saved: Callable[[], None],
    on_build_bundle: Callable[[], None] | None = None,
    on_build_executable: Callable[[], None] | None = None,
) -> None:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFileDialog,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover - editor dependency
        raise RuntimeError("Install PySide6 to use the P64 editor.") from exc

    from p64.build.pipeline import describe_build_python

    project.apply_default_settings()
    dialog = QDialog(parent)
    dialog.setWindowTitle("Build Settings")
    dialog.resize(620, 360)
    layout = QVBoxLayout(dialog)
    form = QFormLayout()
    layout.addLayout(form)

    exe_name = QLineEdit(str(project.build_settings.get("executable_name", project.name)))
    output_folder = QLineEdit(str(project.build_settings.get("output_folder", "build/game")))
    build_mode = QComboBox()
    build_mode.addItems(["executable", "bundle"])
    build_mode.setCurrentText(str(project.build_settings.get("default_build_mode", "executable")))
    windowed = QCheckBox()
    windowed.setChecked(bool(project.build_settings.get("windowed", True)))
    auto_install = QCheckBox()
    auto_install.setChecked(bool(project.build_settings.get("auto_install_build_dependencies", True)))

    icon_path = QLineEdit(str(project.build_settings.get("icon_path", "")))
    python_path = QLineEdit(str(project.build_settings.get("python_executable", "")))
    pipeline_path = QLineEdit(str(project.build_settings.get("build_pipeline_path", "libraries/P64Build")))

    detected_python = QLabel(describe_build_python(project))
    detected_python.setTextInteractionFlags(Qt.TextSelectableByMouse)

    form.addRow("Game Name", exe_name)
    form.addRow("Output Folder", output_folder)
    form.addRow("Default Build Mode", build_mode)
    form.addRow("Windowed", windowed)
    form.addRow("Auto Install Build Dependencies", auto_install)
    form.addRow("Detected Python", detected_python)
    form.addRow("Python Executable", _browse_row(QWidget, QHBoxLayout, QPushButton, python_path, lambda: _choose_python(dialog, QFileDialog, python_path)))
    form.addRow("BuildPipeline Path", _browse_row(QWidget, QHBoxLayout, QPushButton, pipeline_path, lambda: _choose_folder(dialog, QFileDialog, pipeline_path)))
    form.addRow("Icon", _browse_row(QWidget, QHBoxLayout, QPushButton, icon_path, lambda: _choose_icon(dialog, QFileDialog, icon_path)))

    reset_pipeline = QPushButton("Reset BuildPipeline Path")
    reset_pipeline.clicked.connect(lambda: pipeline_path.setText("libraries/P64Build"))
    layout.addWidget(reset_pipeline)

    build_actions = QHBoxLayout()
    if on_build_bundle is not None:
        build_bundle = QPushButton("Build Bundle")
        build_bundle.clicked.connect(lambda: save_and_run(on_build_bundle))
        build_actions.addWidget(build_bundle)
    if on_build_executable is not None:
        build_executable = QPushButton("Build")
        build_executable.clicked.connect(lambda: save_and_run(on_build_executable))
        build_actions.addWidget(build_executable)
    if build_actions.count():
        layout.addLayout(build_actions)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
    layout.addWidget(buttons)

    def persist_settings() -> bool:
        try:
            project.build_settings.update({
                "executable_name": exe_name.text().strip() or project.name,
                "output_folder": output_folder.text().strip() or "build/game",
                "default_build_mode": build_mode.currentText(),
                "windowed": windowed.isChecked(),
                "auto_install_build_dependencies": auto_install.isChecked(),
                "icon_path": _project_relative(project, icon_path.text().strip()),
                "python_executable": python_path.text().strip(),
                "build_pipeline_path": pipeline_path.text().strip() or "libraries/P64Build",
            })
            project.save()
            on_saved()
            return True
        except ValueError as exc:
            QMessageBox.critical(dialog, "Invalid build settings", str(exc))
            return False

    def save_settings() -> None:
        if persist_settings():
            dialog.accept()

    def save_and_run(callback: Callable[[], None]) -> None:
        if persist_settings():
            dialog.accept()
            callback()

    buttons.accepted.connect(save_settings)
    buttons.rejected.connect(dialog.reject)
    dialog.exec()


def _browse_row(widget_cls: type, layout_cls: type, button_cls: type, edit: object, callback: Callable[[], None]) -> object:
    row = widget_cls()
    layout = layout_cls(row)
    layout.setContentsMargins(0, 0, 0, 0)
    button = button_cls("Browse")
    button.clicked.connect(callback)
    layout.addWidget(edit, 1)
    layout.addWidget(button)
    return row


def _choose_python(parent: object, file_dialog: object, edit: object) -> None:
    path, _filter = file_dialog.getOpenFileName(parent, "Choose Python Executable", "", "Python Executable (python.exe py.exe);;Executable (*.exe);;All Files (*)")
    if path:
        edit.setText(path)


def _choose_folder(parent: object, file_dialog: object, edit: object) -> None:
    path = file_dialog.getExistingDirectory(parent, "Choose BuildPipeline Folder")
    if path:
        edit.setText(path)


def _choose_icon(parent: object, file_dialog: object, edit: object) -> None:
    path, _filter = file_dialog.getOpenFileName(parent, "Choose Game Icon", "", "Icons (*.ico *.png);;All Files (*)")
    if path:
        edit.setText(path)


def _project_relative(project: Project, value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        return value.replace("\\", "/")
    try:
        return path.resolve().relative_to(project.root.resolve()).as_posix()
    except ValueError:
        return str(path)
