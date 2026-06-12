from __future__ import annotations

from collections.abc import Callable

from p64.engine.project import Project


def open_project_settings_dialog(parent: object, project: Project, scenes: list[str], on_saved: Callable[[], None]) -> None:
    try:
        from PySide6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QLineEdit,
            QMessageBox,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover - editor dependency
        raise RuntimeError("Install PySide6 to use the P64 editor.") from exc

    project.apply_default_settings()
    dialog = QDialog(parent)
    dialog.setWindowTitle("Project Settings")
    dialog.resize(520, 420)
    layout = QVBoxLayout(dialog)
    tabs = QTabWidget()
    layout.addWidget(tabs)

    general = QWidget()
    general_form = QFormLayout(general)
    name_edit = QLineEdit(project.name)
    startup_combo = QComboBox()
    startup_combo.addItems(scenes)
    startup_combo.setCurrentText(project.startup_scene)
    general_form.addRow("Project Name", name_edit)
    general_form.addRow("Startup Scene", startup_combo)
    tabs.addTab(general, "General")

    render = QWidget()
    render_form = QFormLayout(render)
    resolution = project.render_settings.get("internal_resolution", [320, 240])
    width_edit = QLineEdit(str(resolution[0]))
    height_edit = QLineEdit(str(resolution[1]))
    color_levels = QLineEdit(str(project.render_settings.get("color_levels", 32)))
    dithering = QCheckBox()
    dithering.setChecked(bool(project.render_settings.get("dithering", True)))
    fog = QCheckBox()
    fog.setChecked(bool(project.render_settings.get("fog", True)))
    texture_filter = QComboBox()
    filter_labels = {"Three Point": "three_point", "Nearest": "nearest", "Linear": "linear"}
    texture_filter.addItems(list(filter_labels))
    current_filter = str(project.render_settings.get("texture_filter", "three_point"))
    texture_filter.setCurrentText(next((label for label, value in filter_labels.items() if value == current_filter), "Three Point"))
    render_form.addRow("Internal Width", width_edit)
    render_form.addRow("Internal Height", height_edit)
    render_form.addRow("Color Levels", color_levels)
    render_form.addRow("Dithering", dithering)
    render_form.addRow("Fog", fog)
    render_form.addRow("Texture Filter", texture_filter)
    tabs.addTab(render, "Render")

    scene_view = QWidget()
    scene_form = QFormLayout(scene_view)
    grid = project.editor_settings.get("scene_grid", {})
    grid_enabled = QCheckBox()
    grid_enabled.setChecked(bool(grid.get("enabled", True)))
    grid_spacing = QLineEdit(str(grid.get("spacing", 1.0)))
    grid_radius = QLineEdit(str(grid.get("radius", 40.0)))
    grid_fade_start = QLineEdit(str(grid.get("fade_start", 18.0)))
    grid_fade_end = QLineEdit(str(grid.get("fade_end", 40.0)))
    scene_form.addRow("Grid Enabled", grid_enabled)
    scene_form.addRow("Grid Spacing", grid_spacing)
    scene_form.addRow("Grid Radius", grid_radius)
    scene_form.addRow("Fade Start", grid_fade_start)
    scene_form.addRow("Fade End", grid_fade_end)
    tabs.addTab(scene_view, "Scene View")

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
    layout.addWidget(buttons)

    def save_settings() -> None:
        try:
            project.name = name_edit.text().strip() or project.name
            if startup_combo.currentText():
                project.startup_scene = startup_combo.currentText()
            project.render_settings.update({
                "internal_resolution": [int(width_edit.text()), int(height_edit.text())],
                "color_levels": int(color_levels.text()),
                "dithering": dithering.isChecked(),
                "fog": fog.isChecked(),
                "texture_filter": filter_labels.get(texture_filter.currentText(), "three_point"),
            })
            project.editor_settings["scene_grid"] = {
                "enabled": grid_enabled.isChecked(),
                "spacing": float(grid_spacing.text()),
                "radius": float(grid_radius.text()),
                "fade_start": float(grid_fade_start.text()),
                "fade_end": float(grid_fade_end.text()),
            }
            project.save()
            on_saved()
            dialog.accept()
        except ValueError as exc:
            QMessageBox.critical(dialog, "Invalid settings", str(exc))

    buttons.accepted.connect(save_settings)
    buttons.rejected.connect(dialog.reject)
    dialog.exec()
