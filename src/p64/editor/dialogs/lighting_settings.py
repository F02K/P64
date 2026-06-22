from __future__ import annotations

from collections.abc import Callable
from typing import Any

from p64.engine.render_settings import clamp_render_settings, default_render_settings


def apply_lighting_settings(scene: Any, values: dict[str, Any]) -> dict[str, Any]:
    scene.render_settings = clamp_render_settings({
        **default_render_settings(),
        **dict(getattr(scene, "render_settings", {})),
        **values,
    })
    return scene.render_settings


def open_lighting_settings_dialog(parent: object, scene: Any, on_changed: Callable[[], None]) -> None:
    try:
        from PySide6.QtWidgets import (
            QCheckBox,
            QColorDialog,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QHBoxLayout,
            QLineEdit,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover - editor dependency
        raise RuntimeError("Install PySide6 to use the P64 editor.") from exc

    settings = apply_lighting_settings(scene, {})
    dialog = QDialog(parent)
    dialog.setWindowTitle("Lighting Settings")
    dialog.resize(420, 360)
    layout = QVBoxLayout(dialog)
    form = QFormLayout()
    layout.addLayout(form)

    skybox_enabled = QCheckBox(dialog)
    skybox_enabled.setChecked(bool(settings.get("skybox_enabled", True)))
    fog_enabled = QCheckBox(dialog)
    fog_enabled.setChecked(bool(settings.get("fog", True)))
    coverage = QLineEdit(str(settings.get("skybox_cloud_coverage", 0.45)), dialog)
    scale = QLineEdit(str(settings.get("skybox_cloud_scale", 3.0)), dialog)
    height = QLineEdit(str(settings.get("skybox_cloud_height", 80.0)), dialog)
    softness = QLineEdit(str(settings.get("skybox_cloud_softness", 0.08)), dialog)

    def update(values: dict[str, Any]) -> None:
        apply_lighting_settings(scene, values)
        refresh_numeric_fields()
        on_changed()

    def update_float(edit: QLineEdit, key: str) -> None:
        try:
            update({key: float(edit.text())})
        except ValueError:
            edit.setText(str(settings.get(key, "")))

    def refresh_numeric_fields() -> None:
        current = scene.render_settings
        coverage.setText(str(current.get("skybox_cloud_coverage", 0.45)))
        scale.setText(str(current.get("skybox_cloud_scale", 3.0)))
        height.setText(str(current.get("skybox_cloud_height", 80.0)))
        softness.setText(str(current.get("skybox_cloud_softness", 0.08)))

    skybox_enabled.toggled.connect(lambda checked: update({"skybox_enabled": checked}))
    fog_enabled.toggled.connect(lambda checked: update({"fog": checked}))
    coverage.editingFinished.connect(lambda: update_float(coverage, "skybox_cloud_coverage"))
    scale.editingFinished.connect(lambda: update_float(scale, "skybox_cloud_scale"))
    height.editingFinished.connect(lambda: update_float(height, "skybox_cloud_height"))
    softness.editingFinished.connect(lambda: update_float(softness, "skybox_cloud_softness"))

    form.addRow("Skybox Enabled", skybox_enabled)
    form.addRow("Fog Enabled", fog_enabled)
    form.addRow("Sky Top", _color_editor(dialog, settings.get("skybox_top_color"), lambda values: update({"skybox_top_color": values})))
    form.addRow("Sky Horizon", _color_editor(dialog, settings.get("skybox_horizon_color"), lambda values: update({"skybox_horizon_color": values})))
    form.addRow("Cloud Color", _color_editor(dialog, settings.get("skybox_cloud_color"), lambda values: update({"skybox_cloud_color": values})))
    form.addRow("Cloud Coverage", coverage)
    form.addRow("Cloud Scale", scale)
    form.addRow("Cloud Height", height)
    form.addRow("Cloud Softness", softness)

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
        initial = QColor.fromRgbF(values[0], values[1], values[2])
        selected = QColorDialog.getColor(initial, row, "Pick Color")
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
