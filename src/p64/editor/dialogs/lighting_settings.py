from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from p64.engine.lighting import clamp_lighting_settings, default_lighting_settings, lighting_path_for_scene


def apply_lighting_settings(scene: Any, values: dict[str, Any]) -> dict[str, Any]:
    scene.lighting_settings = clamp_lighting_settings({
        **default_lighting_settings(),
        **dict(getattr(scene, "lighting_settings", {})),
        **values,
    })
    return scene.lighting_settings


def open_lighting_settings_dialog(
    parent: object,
    scene: Any,
    scene_path: Path,
    on_changed: Callable[[], None],
) -> None:
    try:
        from PySide6.QtWidgets import (
            QCheckBox,
            QColorDialog,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover - editor dependency
        raise RuntimeError("Install PySide6 to use the P64 editor.") from exc

    settings = apply_lighting_settings(scene, {})
    lighting_path = lighting_path_for_scene(scene_path)
    dialog = QDialog(parent)
    dialog.setWindowTitle(f"Lighting Settings — {scene_path.name}")
    dialog.resize(480, 430)
    layout = QVBoxLayout(dialog)
    scene_label = QLabel(f"Active Scene: {scene_path.name}", dialog)
    scene_label.setStyleSheet("font-weight: 600;")
    asset_label = QLabel(f"Asset: {lighting_path}", dialog)
    asset_label.setWordWrap(True)
    layout.addWidget(scene_label)
    layout.addWidget(asset_label)
    tabs = QTabWidget(dialog)
    layout.addWidget(tabs)

    sky_page = QWidget(tabs)
    sky_form = QFormLayout(sky_page)
    fog_page = QWidget(tabs)
    fog_form = QFormLayout(fog_page)
    tabs.addTab(sky_page, "Sky & Clouds")
    tabs.addTab(fog_page, "Fog")

    def update(values: dict[str, Any]) -> None:
        apply_lighting_settings(scene, values)
        on_changed()

    def float_editor(form: Any, label: str, key: str) -> None:
        edit = QLineEdit(str(settings[key]), dialog)

        def commit() -> None:
            try:
                update({key: float(edit.text())})
                edit.setText(str(scene.lighting_settings[key]))
            except ValueError:
                edit.setText(str(scene.lighting_settings[key]))

        edit.editingFinished.connect(commit)
        form.addRow(label, edit)

    skybox_enabled = QCheckBox(dialog)
    skybox_enabled.setChecked(bool(settings["skybox_enabled"]))
    skybox_enabled.toggled.connect(lambda checked: update({"skybox_enabled": checked}))
    sky_form.addRow("Skybox Enabled", skybox_enabled)
    sky_form.addRow("Sky Top", _color_editor(dialog, settings["skybox_top_color"], lambda values: update({"skybox_top_color": values})))
    sky_form.addRow("Sky Horizon", _color_editor(dialog, settings["skybox_horizon_color"], lambda values: update({"skybox_horizon_color": values})))
    sky_form.addRow("Cloud Color", _color_editor(dialog, settings["skybox_cloud_color"], lambda values: update({"skybox_cloud_color": values})))
    float_editor(sky_form, "Cloud Coverage", "skybox_cloud_coverage")
    float_editor(sky_form, "Cloud Scale", "skybox_cloud_scale")
    float_editor(sky_form, "Cloud Height", "skybox_cloud_height")
    float_editor(sky_form, "Cloud Softness", "skybox_cloud_softness")

    fog_enabled = QCheckBox(dialog)
    fog_enabled.setChecked(bool(settings["fog_enabled"]))
    fog_enabled.toggled.connect(lambda checked: update({"fog_enabled": checked}))
    fog_form.addRow("Fog Enabled", fog_enabled)
    fog_form.addRow("Fog Color", _color_editor(dialog, settings["fog_color"], lambda values: update({"fog_color": values})))
    float_editor(fog_form, "Near", "fog_near")
    float_editor(fog_form, "Far", "fog_far")
    float_editor(fog_form, "Density", "fog_density")

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()


def _color_editor(parent: object, value: Any, on_changed: Callable[[list[float]], None]) -> object:
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

    values = _color_values(value)
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    button = QPushButton(_color_tooltip(values), row)

    def refresh() -> None:
        button.setText(_color_tooltip(values))
        r, g, b = [round(item * 255) for item in values]
        button.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #111; min-height: 22px;")

    def choose() -> None:
        nonlocal values
        selected = QColorDialog.getColor(QColor.fromRgbF(*values), row, "Pick Color")
        if not selected.isValid():
            return
        values = [selected.redF(), selected.greenF(), selected.blueF()]
        refresh()
        on_changed(values)

    button.clicked.connect(choose)
    layout.addWidget(button)
    refresh()
    return row


def _color_values(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [max(0.0, min(1.0, float(value[0]))), max(0.0, min(1.0, float(value[1]))), max(0.0, min(1.0, float(value[2])))]
        except (TypeError, ValueError):
            pass
    return [1.0, 1.0, 1.0]


def _color_tooltip(values: list[float]) -> str:
    r, g, b = [round(max(0.0, min(1.0, item)) * 255) for item in values]
    return f"#{r:02X}{g:02X}{b:02X}"
