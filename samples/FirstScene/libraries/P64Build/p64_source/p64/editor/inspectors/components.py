from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from p64.editor.ops import (
    add_component,
    import_audio_asset,
    extract_materials_for_obj,
    move_component as move_component_in_entity,
    move_script_entry as move_script_entry_in_component,
    reset_material_asset,
    split_mesh_renderer_into_children,
)
from p64.editor.panels.assets import is_preview_image
from p64.editor.panels.inspector import missing_reference_summary
from p64.editor.utils.ui import make_widget_compact
from p64.engine.assets import AssetMetadata, discover_metadata, model_meshes, resolve_model_mesh
from p64.engine.audio import audio_info
from p64.engine.collision import apply_mesh_primitive_defaults
from p64.engine.components import AudioListener, AudioSource, Camera, CharacterController, Collider, EntityPhysics, Fog, Light, MeshRenderer, ScriptComponent, ScriptEntry, SpawnPoint
from p64.engine.entity import ENTITY, GAME_OBJECT, Entity, set_object_type_recursive
from p64.engine.files import is_metadata_file
from p64.engine.files import find_metadata_for_source
from p64.engine.material import MaterialAsset, is_material_file, load_material_metadata, material_asset_id, resolve_material_reference
from p64.engine.math import Vec3
from p64.engine.render_settings import clamp_render_settings, default_render_settings
from p64.engine.shader import discover_shaders, normalize_shader_id, parse_shader, shader_asset_id
from p64.engine.validation import entity_reference_errors


AVAILABLE_COMPONENTS = (
    "MeshRenderer",
    "AudioSource",
    "AudioListener",
    "Camera",
    "Light",
    "Fog",
    "SpawnPoint",
    "Collider",
    "CharacterController",
    "EntityPhysics",
    "ScriptComponent",
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def component_summary(component: object) -> str:
    if isinstance(component, MeshRenderer):
        return f"MeshRenderer: {component.mesh}"
    if isinstance(component, Camera):
        return f"Camera: fov={component.fov} active={component.active}"
    if isinstance(component, Light):
        return f"Light: {component.kind} intensity={component.intensity}"
    if isinstance(component, AudioSource):
        return f"AudioSource: {component.clip}"
    if isinstance(component, AudioListener):
        return f"AudioListener: active={component.active}"
    if isinstance(component, Fog):
        return f"Fog: near={component.near} far={component.far}"
    return type(component).__name__


def _color_values(value: Any) -> list[float]:
    if isinstance(value, Vec3):
        values = [value.x, value.y, value.z]
    elif isinstance(value, (list, tuple)) and len(value) >= 3:
        values = [float(value[0]), float(value[1]), float(value[2])]
    else:
        values = [1.0, 1.0, 1.0]
    return [max(0.0, min(1.0, float(item))) for item in values[:3]]


def _color_tooltip(values: list[float]) -> str:
    r, g, b = [round(max(0.0, min(1.0, item)) * 255) for item in values]
    return f"#{r:02X}{g:02X}{b:02X}"


def _color_button_style(values: list[float]) -> str:
    r, g, b = [round(max(0.0, min(1.0, item)) * 255) for item in values]
    return f"background-color: rgb({r}, {g}, {b}); border: 1px solid #111; min-height: 20px;"


def texture_image_paths(project: Any) -> list[Path]:
    roots = [project.assets_dir, project.packages_dir]
    images: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        images.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(images, key=lambda path: path.relative_to(project.root).as_posix().lower())


def project_texture_reference(project: Any, path: Path) -> str:
    try:
        return path.resolve().relative_to(project.root.resolve()).as_posix()
    except ValueError:
        return str(path)


def create_inspector_mixin(
    QCheckBox: Any,
    QColorDialog: Any,
    QComboBox: Any,
    QCompleter: Any,
    QFormLayout: Any,
    QGroupBox: Any,
    QHBoxLayout: Any,
    QLabel: Any,
    QLineEdit: Any,
    QMenu: Any,
    QMessageBox: Any,
    QFileDialog: Any,
    QPixmap: Any,
    QPushButton: Any,
    QSizePolicy: Any,
    Qt: Any,
    QVBoxLayout: Any,
    QWidget: Any,
) -> type:
    class InspectorMixin:
        def _close_inspector_popups(self) -> None:
            if not hasattr(self, "inspector"):
                return
            for combo in self.inspector.findChildren(QComboBox):
                combo.hidePopup()
                completer = combo.completer()
                if completer and completer.popup():
                    completer.popup().hide()

        def _add_component_by_name(self, component_name: str) -> None:
            if not self.selected:
                return
            try:
                add_component(self.selected, component_name, self.project)
                self._mark_dirty()
                self._populate_inspector()
                self.viewport.update()
            except Exception as exc:
                self._log(str(exc))

        def _add_component_from_menu(self, component_name: str) -> None:
            if component_name == "MeshRenderer":
                self._add_mesh_renderer()
            elif component_name == "AudioSource":
                self._add_audio_source()
            elif component_name == "Fog":
                self._add_fog()
            elif component_name == "ScriptComponent":
                self._add_script_component()
            else:
                self._add_component_by_name(component_name)

        def _populate_inspector(self) -> None:
            self._close_inspector_popups()
            while self.inspector_layout.count():
                child = self.inspector_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            if self.selected_asset:
                self._populate_asset_inspector(self.selected_asset)
                return
            if not self.selected:
                self._add_scene_render_settings_editor()
                return

            self._add_entity_header()
            self._add_reference_warnings()
            self._add_transform_editor()
            for component in self.selected.components:
                if isinstance(component, ScriptComponent):
                    self._add_script_component_editor(component)
                elif isinstance(component, MeshRenderer):
                    self._add_mesh_renderer_editor(component)
                elif isinstance(component, Fog):
                    self._add_fog_editor(component)
                elif isinstance(component, Camera):
                    self._add_camera_editor(component)
                elif isinstance(component, Light):
                    self._add_light_editor(component)
                elif isinstance(component, AudioSource):
                    self._add_audio_source_editor(component)
                elif isinstance(component, AudioListener):
                    self._add_audio_listener_editor(component)
                elif isinstance(component, SpawnPoint):
                    self._add_spawn_point_editor(component)
                elif isinstance(component, Collider):
                    self._add_collider_editor(component)
                elif isinstance(component, CharacterController):
                    self._add_character_controller_editor(component)
                elif isinstance(component, EntityPhysics):
                    self._add_entity_physics_editor(component)
                else:
                    self.inspector_layout.addWidget(QLabel(component_summary(component), self.inspector))
            self._add_component_controls()
            for component in self.selected.components:
                if isinstance(component, MeshRenderer):
                    self._add_material_slots_editor(component)
            self.inspector_layout.addStretch(1)

        def _add_scene_render_settings_editor(self) -> None:
            scene = getattr(self, "scene", None)
            if scene is None:
                self.inspector_layout.addWidget(QLabel("No SceneObject selected", self.inspector))
                return
            settings = self._scene_render_settings()
            content = QWidget(self.inspector)
            form = QFormLayout(content)
            form.setContentsMargins(8, 6, 8, 8)
            enabled = QCheckBox(content)
            enabled.setChecked(bool(settings.get("skybox_enabled", True)))
            enabled.toggled.connect(lambda checked: self._set_scene_render_bool("skybox_enabled", checked))
            coverage = QLineEdit(str(settings.get("skybox_cloud_coverage", 0.45)), content)
            scale = QLineEdit(str(settings.get("skybox_cloud_scale", 3.0)), content)
            height = QLineEdit(str(settings.get("skybox_cloud_height", 80.0)), content)
            softness = QLineEdit(str(settings.get("skybox_cloud_softness", 0.08)), content)
            coverage.editingFinished.connect(lambda: self._set_scene_render_float(coverage, "skybox_cloud_coverage", 0.0, 1.0))
            scale.editingFinished.connect(lambda: self._set_scene_render_float(scale, "skybox_cloud_scale", 0.1, 24.0))
            height.editingFinished.connect(lambda: self._set_scene_render_float(height, "skybox_cloud_height", 0.1, 10000.0))
            softness.editingFinished.connect(lambda: self._set_scene_render_float(softness, "skybox_cloud_softness", 0.0, 1.0))
            form.addRow("Skybox Enabled", enabled)
            form.addRow("Sky Top", self._color_editor(settings.get("skybox_top_color"), lambda values: self._set_scene_render_color("skybox_top_color", values)))
            form.addRow("Sky Horizon", self._color_editor(settings.get("skybox_horizon_color"), lambda values: self._set_scene_render_color("skybox_horizon_color", values)))
            form.addRow("Cloud Color", self._color_editor(settings.get("skybox_cloud_color"), lambda values: self._set_scene_render_color("skybox_cloud_color", values)))
            form.addRow("Cloud Coverage", coverage)
            form.addRow("Cloud Scale", scale)
            form.addRow("Cloud Height", height)
            form.addRow("Cloud Softness", softness)
            self.inspector_layout.addWidget(self._foldout_panel("Scene Render Settings", "scene:render_settings", content))
            self.inspector_layout.addStretch(1)

        def _populate_asset_inspector(self, path: Path) -> None:
            title = QLabel(path.name, self.inspector)
            title.setWordWrap(True)
            self.inspector_layout.addWidget(title)
            form = QFormLayout()
            form.addRow("Path", QLabel(str(path.relative_to(self.project.root)) if self.project else str(path), self.inspector))
            form.addRow("Type", QLabel("Folder" if path.is_dir() else path.suffix or "File", self.inspector))
            box = QGroupBox("Asset", self.inspector)
            box.setLayout(form)
            self.inspector_layout.addWidget(box)

            actions = QHBoxLayout()
            action_widget = QWidget(self.inspector)
            action_widget.setLayout(actions)
            open_button = QPushButton("Open", action_widget)
            open_button.clicked.connect(lambda: self._open_path(path))
            reveal_button = QPushButton("Reveal", action_widget)
            reveal_button.clicked.connect(lambda: self._reveal_path(path))
            actions.addWidget(open_button)
            actions.addWidget(reveal_button)
            if path.suffix.lower() == ".obj" or is_metadata_file(path):
                import_button = QPushButton("Import", action_widget)
                import_button.clicked.connect(lambda: self._import_asset_obj(path))
                actions.addWidget(import_button)
            if path.suffix.lower() == ".wav":
                refresh_audio = QPushButton("Refresh Audio", action_widget)
                refresh_audio.clicked.connect(lambda: self._refresh_audio_asset(path))
                actions.addWidget(refresh_audio)
            self.inspector_layout.addWidget(action_widget)

            if is_preview_image(path):
                pixmap = QPixmap(str(path))
                if not pixmap.isNull():
                    preview = QLabel(self.inspector)
                    preview.setAlignment(Qt.AlignCenter)
                    preview.setPixmap(pixmap.scaled(220, 220, Qt.KeepAspectRatio, Qt.FastTransformation))
                    self.inspector_layout.addWidget(preview)
            elif is_metadata_file(path):
                try:
                    metadata = AssetMetadata.load(path)
                    meta = QFormLayout()
                    meta.addRow("ID", QLabel(metadata.id, self.inspector))
                    meta.addRow("Kind", QLabel(metadata.kind, self.inspector))
                    meta.addRow("Source", QLabel(metadata.source, self.inspector))
                    meta.addRow("Groups", QLabel(", ".join(metadata.groups), self.inspector))
                    meta.addRow("Materials", QLabel(", ".join(metadata.materials), self.inspector))
                    meta_box = QGroupBox("Metadata", self.inspector)
                    meta_box.setLayout(meta)
                    self.inspector_layout.addWidget(meta_box)
                except Exception as exc:
                    self.inspector_layout.addWidget(QLabel(f"Metadata error: {exc}", self.inspector))
            elif path.suffix.lower() == ".shader":
                try:
                    shader = parse_shader(path)
                    shader_form = QFormLayout()
                    shader_form.addRow("Name", QLabel(shader.name, self.inspector))
                    shader_form.addRow("Vertex", QLabel(f"{len(shader.vertex.splitlines())} lines", self.inspector))
                    shader_form.addRow("Fragment", QLabel(f"{len(shader.fragment.splitlines())} lines", self.inspector))
                    shader_form.addRow("Properties", QLabel(str(len(shader.properties)), self.inspector))
                    shader_box = QGroupBox("Shader", self.inspector)
                    shader_box.setLayout(shader_form)
                    self.inspector_layout.addWidget(shader_box)
                except Exception as exc:
                    self.inspector_layout.addWidget(QLabel(f"Shader parse error: {exc}", self.inspector))
            elif path.suffix.lower() == ".obj":
                self._add_obj_asset_inspector(path)
            elif path.suffix.lower() == ".wav":
                self._add_audio_asset_inspector(path)
            elif is_material_file(path):
                self._add_material_asset_inspector(path)
            self.inspector_layout.addStretch(1)

        def _add_entity_header(self) -> None:
            if not self.selected:
                return
            header = QWidget(self.inspector)
            header.setObjectName("EntityHeader")
            header.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
            header.setStyleSheet("#EntityHeader { background: #2b2d30; border: 1px solid #4a4d52; }")
            row = QHBoxLayout(header)
            row.setContentsMargins(8, 5, 8, 5)
            row.setSpacing(8)

            active = QCheckBox(self.inspector)
            active.setChecked(self.selected.active)
            active.toggled.connect(lambda checked: self._set_selected_active(checked))
            name = QLineEdit(self.selected.name, self.inspector)
            name.editingFinished.connect(lambda: self._rename_selected(name.text()))
            persist = QCheckBox("Persistent", self.inspector)
            persist.setChecked(self.selected.persistent)
            persist.toggled.connect(lambda checked: self._set_selected_persistent(checked))
            object_type = QComboBox(self.inspector)
            object_type.addItems(["GameObject", "Entity"])
            object_type.setCurrentText(self.selected.object_type_label)
            object_type.currentTextChanged.connect(lambda text: self._set_selected_object_type(text))

            row.addWidget(active)
            row.addWidget(name, 1)
            row.addWidget(persist)
            row.addWidget(object_type)
            self.inspector_layout.addWidget(header)

        def _add_reference_warnings(self) -> None:
            if not self.project or not self.selected:
                return
            errors = entity_reference_errors(self.project, self.selected)
            if not errors:
                return
            strip = QWidget(self.inspector)
            strip.setObjectName("MissingReferenceStrip")
            strip.setStyleSheet(
                "#MissingReferenceStrip { background: #3a1f1f; border: 1px solid #9a3a3a; }"
                "#MissingReferenceStrip QLabel { color: #f0b6b6; }"
            )
            strip.setMaximumHeight(72)
            row = QHBoxLayout(strip)
            row.setContentsMargins(8, 4, 8, 4)
            message = QLabel(missing_reference_summary(errors), strip)
            message.setWordWrap(True)
            message.setToolTip("\n".join(errors))
            details = QPushButton("Log", strip)
            details.setMaximumWidth(48)
            details.clicked.connect(lambda: self._log("Missing references:\n" + "\n".join(errors)))
            row.addWidget(message, 1)
            row.addWidget(details)
            self.inspector_layout.addWidget(strip)

        def _add_transform_editor(self) -> None:
            if not self.selected:
                return
            transform = self.selected.transform
            content = QWidget(self.inspector)
            form = QFormLayout(content)
            form.setContentsMargins(8, 6, 8, 8)
            form.addRow("Position", self._vec3_editor(transform.position))
            form.addRow("Rotation", self._vec3_editor(transform.rotation))
            form.addRow("Scale", self._vec3_editor(transform.scale))
            reset = QPushButton("Reset Transform", content)
            reset.clicked.connect(self._reset_transform)
            form.addRow("Reset", reset)
            self.inspector_layout.addWidget(self._foldout_panel("Transform", f"{self.selected.id}:Transform", content))

        def _add_script_component_editor(self, component: ScriptComponent) -> None:
            content = QWidget(self.inspector)
            layout = QVBoxLayout(content)
            layout.setContentsMargins(8, 6, 8, 8)

            for index, entry in enumerate(component.scripts):
                layout.addLayout(self._script_row(component, entry, index))

            add_script = QPushButton("Add Script", content)
            add_script.clicked.connect(lambda: self._add_script_entry(component))
            layout.addWidget(add_script)
            self.inspector_layout.addWidget(self._component_panel(component, "ScriptComponent", content))

        def _add_mesh_renderer_editor(self, component: MeshRenderer) -> None:
            content, form = self._component_content_widget()
            visible = QCheckBox(content)
            visible.setChecked(component.visible)
            visible.toggled.connect(lambda checked: self._set_mesh_visible(component, checked))

            mesh_choices = self._mesh_choice_items()
            mesh_combo = self._search_combo([label for label, _metadata, _mesh_entry in mesh_choices])
            mesh_id_to_label = {str(mesh_entry.get("id") or metadata.id): label for label, metadata, mesh_entry in mesh_choices}
            label_to_choice = {label: (metadata, mesh_entry) for label, metadata, mesh_entry in mesh_choices}
            mesh_combo.setCurrentText(mesh_id_to_label.get(component.mesh, component.mesh))
            material_combo = QComboBox(content)
            material_combo.setEditable(True)
            self._populate_mesh_dependent_combos(component, material_combo)

            mesh_combo.currentTextChanged.connect(
                lambda text: self._set_mesh_from_label(component, text, label_to_choice, material_combo)
            )
            material_combo.currentTextChanged.connect(lambda text: self._set_mesh_material(component, text))

            form.addRow("Visible", visible)
            form.addRow("Mesh", mesh_combo)
            form.addRow("OBJ Material", material_combo)
            source_materials = self._source_materials_for_component(component)
            source_label = QLabel(", ".join(source_materials) or "None", content)
            source_label.setWordWrap(True)
            form.addRow("Source Materials", source_label)
            texture_label = QLabel(self._texture_summary(component), content)
            texture_label.setWordWrap(True)
            form.addRow("Texture", texture_label)
            pixmap = self._texture_pixmap(component)
            if pixmap:
                preview = QLabel(content)
                preview.setPixmap(pixmap)
                form.addRow("Preview", preview)
            split_button = QPushButton("Split Into Child GameObjects", content)
            split_button.clicked.connect(lambda: self._split_mesh_renderer(component))
            form.addRow("Sub-objects", split_button)
            self.inspector_layout.addWidget(self._component_panel(component, "MeshRenderer", content))

        def _add_obj_asset_inspector(self, path: Path) -> None:
            if not self.project:
                return
            metadata_path = find_metadata_for_source(path)
            if not metadata_path or not metadata_path.exists():
                return
            try:
                metadata = AssetMetadata.load(metadata_path)
            except Exception as exc:
                self.inspector_layout.addWidget(QLabel(f"OBJ metadata error: {exc}", self.inspector))
                return
            material_defs = metadata.settings.get("material_defs", {})
            material_assets = metadata.settings.get("material_assets", {})
            meshes = model_meshes(metadata)
            form = QFormLayout()
            form.addRow("Model Source", QLabel(metadata.source, self.inspector))
            form.addRow("Meshes", QLabel(str(len(meshes) or len(metadata.groups)), self.inspector))
            form.addRow("Materials", QLabel(", ".join(metadata.materials) or "None", self.inspector))
            extracted = sum(1 for name in metadata.materials if isinstance(material_assets, dict) and material_assets.get(name))
            form.addRow("Extracted", QLabel(f"{extracted}/{len(metadata.materials)}", self.inspector))
            texture_names = []
            for name in metadata.materials:
                material = material_defs.get(name, {})
                if isinstance(material, dict) and material.get("diffuse_texture"):
                    texture_names.append(f"{name}: {material.get('diffuse_texture')}")
            texture_label = QLabel("\n".join(texture_names) or "None", self.inspector)
            texture_label.setWordWrap(True)
            form.addRow("Textures", texture_label)
            extract = QPushButton("Extract Materials", self.inspector)
            extract.clicked.connect(lambda: self._extract_materials(path))
            form.addRow("Materials", extract)
            box = QGroupBox("OBJ Import", self.inspector)
            box.setLayout(form)
            self.inspector_layout.addWidget(box)
            if meshes:
                mesh_box = QGroupBox("Model Meshes", self.inspector)
                mesh_layout = QVBoxLayout(mesh_box)
                for mesh in meshes:
                    bounds = mesh.get("bounds", {})
                    material_slots = mesh.get("material_slots", [])
                    label = QLabel(
                        f"{mesh.get('node_path', mesh.get('name', 'Mesh'))} | "
                        f"{mesh.get('triangle_count', 0)} tris | "
                        f"{len(material_slots) if isinstance(material_slots, list) else 0} material(s) | "
                        f"bounds {bounds.get('min', '?')} -> {bounds.get('max', '?') if isinstance(bounds, dict) else '?'}",
                        mesh_box,
                    )
                    label.setWordWrap(True)
                    mesh_layout.addWidget(label)
                self.inspector_layout.addWidget(mesh_box)
                preview = self._model_wireframe_preview(meshes)
                if preview is not None:
                    self.inspector_layout.addWidget(preview)

        def _model_wireframe_preview(self, meshes: list[dict[str, Any]]) -> Any | None:
            try:
                from PySide6.QtGui import QColor, QPainter, QPen
            except Exception:
                return None
            points: list[tuple[float, float, float]] = []
            for mesh in meshes:
                wireframe = mesh.get("wireframe", {})
                values = wireframe.get("vertices", []) if isinstance(wireframe, dict) else []
                if not isinstance(values, list):
                    continue
                for index in range(0, len(values) - 2, 3):
                    points.append((float(values[index]), float(values[index + 1]), float(values[index + 2])))
            if len(points) < 2:
                return None
            min_x = min(point[0] for point in points)
            max_x = max(point[0] for point in points)
            min_y = min(point[1] for point in points)
            max_y = max(point[1] for point in points)
            width, height = 220, 150
            scale = min((width - 24) / max(max_x - min_x, 0.001), (height - 24) / max(max_y - min_y, 0.001))

            def project(point: tuple[float, float, float]) -> tuple[int, int]:
                x = 12 + (point[0] - min_x) * scale
                y = height - (12 + (point[1] - min_y) * scale)
                return int(x), int(y)

            pixmap = QPixmap(width, height)
            pixmap.fill(QColor(28, 31, 36))
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor(120, 190, 255), 1))
            for index in range(0, len(points) - 1, 2):
                start = project(points[index])
                end = project(points[index + 1])
                painter.drawLine(start[0], start[1], end[0], end[1])
            painter.end()
            label = QLabel(self.inspector)
            label.setAlignment(Qt.AlignCenter)
            label.setPixmap(pixmap)
            return label

        def _add_audio_asset_inspector(self, path: Path) -> None:
            metadata_path = find_metadata_for_source(path)
            form = QFormLayout()
            if metadata_path and metadata_path.exists():
                try:
                    metadata = AssetMetadata.load(metadata_path)
                    info = audio_info(metadata) or {}
                    form.addRow("Clip ID", QLabel(metadata.id, self.inspector))
                    form.addRow("Original", QLabel(f"{info.get('original_channels', '?')} ch @ {info.get('original_sample_rate', '?')} Hz", self.inspector))
                    form.addRow("Imported", QLabel(f"mono @ {info.get('imported_sample_rate', '?')} Hz", self.inspector))
                    form.addRow("Duration", QLabel(f"{float(info.get('duration', 0.0)):.2f}s", self.inspector))
                    form.addRow("Samples", QLabel(str(info.get("sample_count", "?")), self.inspector))
                    form.addRow("Generated", QLabel(str(info.get("generated_path", "")), self.inspector))
                except Exception as exc:
                    form.addRow("Audio", QLabel(f"Metadata error: {exc}", self.inspector))
            else:
                form.addRow("Audio", QLabel("Not imported", self.inspector))
            box = QGroupBox("AudioClip", self.inspector)
            box.setLayout(form)
            self.inspector_layout.addWidget(box)

        def _add_material_asset_inspector(self, path: Path) -> None:
            if not self.project:
                return
            try:
                material = MaterialAsset.load(path)
            except Exception as exc:
                self.inspector_layout.addWidget(QLabel(f"Material error: {exc}", self.inspector))
                return
            content = QWidget(self.inspector)
            layout = QVBoxLayout(content)
            layout.setContentsMargins(8, 6, 8, 8)
            self._add_material_editor_fields(layout, path, material)
            reset = QPushButton("Reset From MTL Defaults", content)
            reset.clicked.connect(lambda: self._reset_material(path))
            layout.addWidget(reset)
            metadata = load_material_metadata(path)
            usage = metadata.settings.get("usage_cache", []) if metadata else []
            if usage:
                used_by = QLabel("\n".join(f"{item.get('scene', '')}: {item.get('entity', '')}" for item in usage), content)
                used_by.setWordWrap(True)
                layout.addWidget(used_by)
            self.inspector_layout.addWidget(self._foldout_panel("Material", f"asset:{path}:material", content))

        def _add_material_editor_fields(self, layout: Any, path: Path, material: MaterialAsset | None = None) -> None:
            if not self.project:
                return
            try:
                material = material or MaterialAsset.load(path)
            except Exception as exc:
                layout.addWidget(QLabel(f"Material error: {exc}", self.inspector))
                return
            parent = self.inspector
            form = QFormLayout()
            shader_choices = self._shader_choices()
            shader_combo = self._search_combo([label for label, _path in shader_choices])
            shader_label_by_id = {shader_id: label for label, shader_id in shader_choices}
            shader_label_to_id = dict(shader_choices)
            shader_combo.setCurrentText(shader_label_by_id.get(material.shader, material.shader))
            shader_combo.currentTextChanged.connect(lambda text: self._set_material_shader(path, material, text, shader_label_to_id))
            form.addRow("Shader", shader_combo)
            layout.addLayout(form)

            prop_form = QFormLayout()
            shader_path = self.project.root / material.shader
            try:
                shader = parse_shader(shader_path)
                properties = shader.properties
            except Exception:
                properties = []
            for prop in properties:
                if prop.kind == "texture":
                    value = material.textures.get(prop.name, str(prop.default or ""))
                    prop_form.addRow(prop.name, self._texture_editor(path, material, prop.name, str(value)))
                else:
                    value = material.properties.get(prop.name, prop.default)
                    if prop.kind == "color":
                        prop_form.addRow(prop.name, self._color_editor(value, lambda values, name=prop.name: self._apply_material_color_property(path, material, name, values)))
                    else:
                        edit = QLineEdit(json.dumps(value) if isinstance(value, (list, dict)) else str(value), parent)
                        edit.editingFinished.connect(lambda edit=edit, name=prop.name: self._apply_material_property(path, material, name, edit.text()))
                        prop_form.addRow(prop.name, edit)
            if prop_form.rowCount() == 0:
                prop_form.addRow("Properties", QLabel("No shader properties", parent))
            prop_box = QGroupBox("Properties", parent)
            prop_box.setLayout(prop_form)
            layout.addWidget(prop_box)

        def _add_material_slots_editor(self, component: MeshRenderer) -> None:
            content = QWidget(self.inspector)
            layout = QVBoxLayout(content)
            layout.setContentsMargins(8, 6, 8, 8)
            source_materials = self._source_materials_for_component(component)
            slot_count = max(1, len(component.material_slots), len(source_materials))
            while len(component.material_slots) < slot_count:
                component.material_slots.append(None)
            choices = self._material_choices()
            labels = ["None"] + [label for label, _path in choices]
            label_by_id = {material_id: label for label, material_id in choices}
            label_to_id = dict(choices)
            for index in range(slot_count):
                row_box = QGroupBox(source_materials[index] if index < len(source_materials) else f"Slot {index}", content)
                row_layout = QVBoxLayout(row_box)
                row_layout.setContentsMargins(8, 6, 8, 8)
                row_form = QFormLayout()
                combo = self._search_combo(labels)
                current = component.material_slots[index]
                combo.setCurrentText(label_by_id.get(current or "", current or "None"))
                combo.currentTextChanged.connect(lambda text, index=index: self._set_material_slot(component, index, text, label_to_id))
                row_form.addRow("Material", combo)
                row_layout.addLayout(row_form)
                material_path = resolve_material_reference(self.project.root, current) if self.project else None
                if material_path and material_path.exists():
                    self._add_material_editor_fields(row_layout, material_path)
                else:
                    fallback = QLabel("Using MTL Defaults / Standard VertexLit", row_box)
                    fallback.setWordWrap(True)
                    row_layout.addWidget(fallback)
                layout.addWidget(row_box)
            self.inspector_layout.addWidget(self._foldout_panel("Materials", f"{id(component)}:Materials", content))

        def _add_fog_editor(self, component: Fog) -> None:
            content, form = self._component_content_widget()
            near = QLineEdit(str(component.near), content)
            far = QLineEdit(str(component.far), content)
            density = QLineEdit(str(component.density), content)
            near.editingFinished.connect(lambda: self._apply_float(near, component, "near"))
            far.editingFinished.connect(lambda: self._apply_float(far, component, "far"))
            density.editingFinished.connect(lambda: self._apply_float(density, component, "density"))
            form.addRow("Color", self._color_editor(component.color, lambda values: self._set_vec3_color(component.color, values)))
            form.addRow("Size", self._vec3_editor(component.size))
            form.addRow("Near", near)
            form.addRow("Far", far)
            form.addRow("Density", density)
            self.inspector_layout.addWidget(self._component_panel(component, "Fog Volume", content))

        def _add_camera_editor(self, component: Camera) -> None:
            content, form = self._component_content_widget()
            active = QCheckBox(content)
            active.setChecked(component.active)
            active.toggled.connect(lambda checked: self._set_camera_active(component, checked))
            fov = QLineEdit(str(component.fov), content)
            near = QLineEdit(str(component.near), content)
            far = QLineEdit(str(component.far), content)
            fov.editingFinished.connect(lambda: self._apply_float(fov, component, "fov"))
            near.editingFinished.connect(lambda: self._apply_float(near, component, "near"))
            far.editingFinished.connect(lambda: self._apply_float(far, component, "far"))
            form.addRow("Active", active)
            form.addRow("FOV", fov)
            form.addRow("Near", near)
            form.addRow("Far", far)
            self.inspector_layout.addWidget(self._component_panel(component, "Camera", content))

        def _add_light_editor(self, component: Light) -> None:
            content, form = self._component_content_widget()
            kind = QComboBox(content)
            kind.addItems(["directional", "point", "spot"])
            kind.setCurrentText(component.kind)
            kind.currentTextChanged.connect(lambda text: self._set_light_kind(component, text))
            intensity = QLineEdit(str(component.intensity), content)
            intensity.editingFinished.connect(lambda: self._apply_float(intensity, component, "intensity"))
            form.addRow("Kind", kind)
            form.addRow("Color", self._color_editor(component.color, lambda values: self._set_vec3_color(component.color, values)))
            form.addRow("Intensity", intensity)
            if component.kind in {"point", "spot"}:
                range_edit = QLineEdit(str(component.range), content)
                range_edit.editingFinished.connect(lambda: self._apply_float(range_edit, component, "range"))
                falloff = QLineEdit(str(component.falloff), content)
                falloff.editingFinished.connect(lambda: self._apply_float(falloff, component, "falloff"))
                form.addRow("Range", range_edit)
                form.addRow("Falloff", falloff)
            if component.kind == "spot":
                spot_angle = QLineEdit(str(component.spot_angle), content)
                spot_angle.editingFinished.connect(lambda: self._apply_float(spot_angle, component, "spot_angle"))
                form.addRow("Spot Angle", spot_angle)
            self.inspector_layout.addWidget(self._component_panel(component, "Light", content))

        def _add_audio_source_editor(self, component: AudioSource) -> None:
            content, form = self._component_content_widget()
            clip_choices = self._audio_clip_choices()
            labels = ["None"] + [label for label, _metadata in clip_choices]
            label_by_id = {metadata.id: label for label, metadata in clip_choices}
            label_to_id = {label: metadata.id for label, metadata in clip_choices}
            clip = self._search_combo(labels)
            clip.setCurrentText(label_by_id.get(component.clip, component.clip or "None"))
            clip.currentTextChanged.connect(lambda text: self._set_audio_clip(component, text, label_to_id))
            loop = QCheckBox(content)
            loop.setChecked(component.loop)
            loop.toggled.connect(lambda checked: self._set_audio_bool(component, "loop", checked))
            play_on_awake = QCheckBox(content)
            play_on_awake.setChecked(component.play_on_awake)
            play_on_awake.toggled.connect(lambda checked: self._set_audio_bool(component, "play_on_awake", checked))
            spatial = QCheckBox(content)
            spatial.setChecked(component.spatial)
            spatial.toggled.connect(lambda checked: self._set_audio_bool(component, "spatial", checked))
            volume = QLineEdit(str(component.volume), content)
            pitch = QLineEdit(str(component.pitch), content)
            min_distance = QLineEdit(str(component.min_distance), content)
            max_distance = QLineEdit(str(component.max_distance), content)
            for edit, attr in [
                (volume, "volume"),
                (pitch, "pitch"),
                (min_distance, "min_distance"),
                (max_distance, "max_distance"),
            ]:
                edit.editingFinished.connect(lambda edit=edit, attr=attr: self._apply_float(edit, component, attr))
            form.addRow("Clip", clip)
            form.addRow("Volume", volume)
            form.addRow("Pitch", pitch)
            form.addRow("Loop", loop)
            form.addRow("Play On Awake", play_on_awake)
            form.addRow("Spatial", spatial)
            form.addRow("Min Distance", min_distance)
            form.addRow("Max Distance", max_distance)
            self.inspector_layout.addWidget(self._component_panel(component, "AudioSource", content))

        def _add_audio_listener_editor(self, component: AudioListener) -> None:
            content, form = self._component_content_widget()
            active = QCheckBox(content)
            active.setChecked(component.active)
            active.toggled.connect(lambda checked: self._set_audio_listener_active(component, checked))
            form.addRow("Active", active)
            self.inspector_layout.addWidget(self._component_panel(component, "AudioListener", content))

        def _add_spawn_point_editor(self, component: SpawnPoint) -> None:
            content, form = self._component_content_widget()
            spawn_id = QLineEdit(component.spawn_id, content)
            from_scene = QLineEdit(component.from_scene, content)
            is_default = QCheckBox(content)
            is_default.setChecked(component.is_default)
            spawn_id.editingFinished.connect(lambda: self._apply_text(spawn_id, component, "spawn_id"))
            from_scene.editingFinished.connect(lambda: self._apply_text(from_scene, component, "from_scene"))
            is_default.toggled.connect(lambda checked: self._set_spawn_default(component, checked))
            form.addRow("Spawn ID", spawn_id)
            form.addRow("From Scene", from_scene)
            form.addRow("Default", is_default)
            self.inspector_layout.addWidget(self._component_panel(component, "SpawnPoint", content))

        def _add_collider_editor(self, component: Collider) -> None:
            content, form = self._component_content_widget()
            shape = QComboBox(content)
            shape.addItems(["box", "sphere", "mesh"])
            shape.setCurrentText(component.shape)
            shape.currentTextChanged.connect(lambda text: self._set_collider_shape(component, text))
            trigger = QCheckBox(content)
            trigger.setChecked(component.is_trigger)
            trigger.toggled.connect(lambda checked: self._set_collider_trigger(component, checked))
            layer = QLineEdit(component.layer, content)
            mask = QLineEdit(component.mask, content)
            layer.editingFinished.connect(lambda: self._apply_text(layer, component, "layer"))
            mask.editingFinished.connect(lambda: self._apply_text(mask, component, "mask"))
            form.addRow("Shape", shape)
            form.addRow("Mesh Source", QLabel(self._collider_mesh_source(), content))
            if component.shape == "box":
                fit_to_mesh = QCheckBox(content)
                fit_to_mesh.setChecked(component.fit_to_mesh)
                fit_to_mesh.toggled.connect(lambda checked: self._set_collider_fit_to_mesh(component, checked))
                form.addRow("Center", self._vec3_editor(component.center))
                form.addRow("Size", self._vec3_editor(component.size))
                form.addRow("Fit To Mesh", fit_to_mesh)
            elif component.shape == "sphere":
                radius = QLineEdit(str(component.radius), content)
                radius.editingFinished.connect(lambda: self._apply_float(radius, component, "radius"))
                fit_to_mesh = QCheckBox(content)
                fit_to_mesh.setChecked(component.fit_to_mesh)
                fit_to_mesh.toggled.connect(lambda checked: self._set_collider_fit_to_mesh(component, checked))
                form.addRow("Center", self._vec3_editor(component.center))
                form.addRow("Radius", radius)
                form.addRow("Fit To Mesh", fit_to_mesh)
            elif component.shape == "mesh":
                convex = QCheckBox(content)
                convex.setChecked(component.convex)
                convex.toggled.connect(lambda checked: self._set_collider_convex(component, checked))
                form.addRow("Convex", convex)
            form.addRow("Trigger", trigger)
            form.addRow("Layer", layer)
            form.addRow("Mask", mask)
            self.inspector_layout.addWidget(self._component_panel(component, "Collider", content))

        def _add_character_controller_editor(self, component: CharacterController) -> None:
            content, form = self._component_content_widget()
            height = QLineEdit(str(component.height), content)
            radius = QLineEdit(str(component.radius), content)
            skin_width = QLineEdit(str(component.skin_width), content)
            slope_limit = QLineEdit(str(component.slope_limit), content)
            gravity = QLineEdit(str(component.gravity), content)
            for edit, attr in [
                (height, "height"),
                (radius, "radius"),
                (skin_width, "skin_width"),
                (slope_limit, "slope_limit"),
                (gravity, "gravity"),
            ]:
                edit.editingFinished.connect(lambda edit=edit, attr=attr: self._apply_float(edit, component, attr))
            grounded = QLabel("Yes" if component.grounded else "No", content)
            form.addRow("Height", height)
            form.addRow("Radius", radius)
            form.addRow("Skin Width", skin_width)
            form.addRow("Slope Limit", slope_limit)
            form.addRow("Gravity", gravity)
            form.addRow("Velocity", self._vec3_editor(component.velocity))
            form.addRow("Grounded", grounded)
            self.inspector_layout.addWidget(self._component_panel(component, "CharacterController", content))

        def _add_entity_physics_editor(self, component: EntityPhysics) -> None:
            content, form = self._component_content_widget()
            mass = QLineEdit(str(component.mass), content)
            drag = QLineEdit(str(component.drag), content)
            angular_drag = QLineEdit(str(component.angular_drag), content)
            for edit, attr in [
                (mass, "mass"),
                (drag, "drag"),
                (angular_drag, "angular_drag"),
            ]:
                edit.editingFinished.connect(lambda edit=edit, attr=attr: self._apply_float(edit, component, attr))
            gravity = QCheckBox(content)
            gravity.setChecked(component.use_gravity)
            gravity.toggled.connect(lambda checked: self._set_entity_physics_bool(component, "use_gravity", checked))
            kinematic = QCheckBox(content)
            kinematic.setChecked(component.is_kinematic)
            kinematic.toggled.connect(lambda checked: self._set_entity_physics_bool(component, "is_kinematic", checked))
            form.addRow("Mass", mass)
            form.addRow("Gravity", gravity)
            form.addRow("Drag", drag)
            form.addRow("Angular Drag", angular_drag)
            form.addRow("Kinematic", kinematic)
            form.addRow("Velocity", self._vec3_editor(component.velocity))
            form.addRow("Angular Velocity", self._vec3_editor(component.angular_velocity))
            form.addRow("Freeze Position", self._vec3_editor(component.freeze_position))
            form.addRow("Freeze Rotation", self._vec3_editor(component.freeze_rotation))
            self.inspector_layout.addWidget(self._component_panel(component, "EntityPhysics", content))

        def _script_row(self, component: ScriptComponent, entry: ScriptEntry, index: int) -> Any:
            row = QHBoxLayout()
            script_combo = QComboBox(self.inspector)
            scripts = self._script_files()
            script_combo.addItems(scripts)
            if entry.script and entry.script not in scripts:
                script_combo.addItem(entry.script)
            script_combo.setCurrentText(entry.script)

            class_combo = QComboBox(self.inspector)
            class_combo.setEditable(True)
            self._populate_class_combo(class_combo, script_combo.currentText(), entry.class_name)

            enabled = QCheckBox(self.inspector)
            enabled.setChecked(entry.enabled)
            actions = QPushButton("...", self.inspector)
            actions.setMaximumWidth(36)
            actions_menu = QMenu(actions)

            script_combo.currentTextChanged.connect(
                lambda text: self._update_script_entry(component, entry, script=text, class_combo=class_combo)
            )
            class_combo.currentTextChanged.connect(lambda text: self._update_script_entry(component, entry, class_name=text))
            enabled.toggled.connect(lambda checked: self._update_script_entry(component, entry, enabled=checked))
            actions_menu.addAction("Move Up", lambda: self._move_script_entry(component, index, -1))
            actions_menu.addAction("Move Down", lambda: self._move_script_entry(component, index, 1))
            actions_menu.addAction(
                "Reload Classes",
                lambda: self._populate_class_combo(class_combo, script_combo.currentText(), entry.class_name),
            )
            actions_menu.addSeparator()
            actions_menu.addAction("Remove", lambda: self._remove_script_entry(component, index))
            actions.setMenu(actions_menu)

            row.addWidget(enabled)
            row.addWidget(script_combo)
            row.addWidget(class_combo)
            row.addWidget(actions)
            return row

        def _add_component_controls(self) -> None:
            row = QHBoxLayout()
            add_component = QPushButton("Add Component", self.inspector)
            add_component.setEnabled(self.selected is not None)
            add_component.clicked.connect(lambda: self._show_add_component_menu(add_component.mapToGlobal(add_component.rect().bottomLeft())))
            row.addWidget(add_component)
            row.addStretch(1)
            box = QGroupBox("Add Component", self.inspector)
            box.setLayout(row)
            self._make_inspector_box_compact(box)
            self.inspector_layout.addWidget(box)

        def _show_add_component_menu(self, global_pos: Any) -> None:
            if not self.selected:
                return
            menu = QMenu(self)
            for component_name in AVAILABLE_COMPONENTS:
                menu.addAction(component_name, lambda checked=False, name=component_name: self._add_component_from_menu(name))
            menu.exec(global_pos)

        def _show_inspector_context_menu(self, pos: Any) -> None:
            if not self.selected:
                return
            widget = self.inspector.childAt(pos)
            while widget is not None and widget is not self.inspector:
                if widget.objectName() in {"ComponentPanel", "ComponentHeader"}:
                    return
                widget = widget.parentWidget()
            self._show_add_component_menu(self.inspector.mapToGlobal(pos))

        def _make_inspector_box_compact(self, widget: Any) -> None:
            make_widget_compact(widget, QSizePolicy)

        def _component_panel(self, component: Any, title: str, content: Any) -> Any:
            key = self._component_key(component)
            collapsed = self.collapsed_components.get(key, False)
            panel = QWidget(self.inspector)
            panel.setObjectName("ComponentPanel")
            panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
            panel.setContextMenuPolicy(Qt.CustomContextMenu)
            panel.customContextMenuRequested.connect(lambda pos: self._show_component_context_menu(component, panel.mapToGlobal(pos)))
            panel.setStyleSheet(
                "#ComponentPanel { background: #2b2d30; border: 1px solid #4a4d52; }"
                "#ComponentHeader { background: #34373b; }"
            )
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            header = QWidget(panel)
            header.setObjectName("ComponentHeader")
            header.setContextMenuPolicy(Qt.CustomContextMenu)
            header.customContextMenuRequested.connect(lambda pos: self._show_component_context_menu(component, header.mapToGlobal(pos)))
            header_row = QHBoxLayout(header)
            header_row.setContentsMargins(6, 3, 6, 3)
            header_row.setSpacing(6)

            foldout = QPushButton(">" if collapsed else "v", header)
            foldout.setMaximumWidth(24)
            foldout.clicked.connect(lambda: self._toggle_component_collapsed(component))
            enabled = QCheckBox(header)
            enabled.setChecked(bool(getattr(component, "enabled", True)))
            enabled.toggled.connect(lambda checked: self._set_component_enabled(component, checked))
            name = QLabel(title, header)
            name.setStyleSheet("font-weight: 600;")
            menu = QPushButton("...", header)
            menu.setMaximumWidth(28)
            menu.clicked.connect(lambda: self._show_component_context_menu(component, menu.mapToGlobal(menu.rect().bottomLeft())))
            header_row.addWidget(foldout)
            header_row.addWidget(enabled)
            header_row.addWidget(name, 1)
            header_row.addWidget(menu)

            content.setVisible(not collapsed)
            layout.addWidget(header)
            layout.addWidget(content)
            return panel

        def _component_content_widget(self) -> tuple[Any, Any]:
            widget = QWidget(self.inspector)
            form = QFormLayout(widget)
            form.setContentsMargins(8, 6, 8, 8)
            return widget, form

        def _foldout_panel(self, title: str, key: str, content: Any) -> Any:
            collapsed = self.collapsed_components.get(key, False)
            panel = QWidget(self.inspector)
            panel.setObjectName("ComponentPanel")
            panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
            panel.setStyleSheet(
                "#ComponentPanel { background: #2b2d30; border: 1px solid #4a4d52; }"
                "#ComponentHeader { background: #34373b; }"
            )
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            header = QWidget(panel)
            header.setObjectName("ComponentHeader")
            row = QHBoxLayout(header)
            row.setContentsMargins(6, 3, 6, 3)
            foldout = QPushButton(">" if collapsed else "v", header)
            foldout.setMaximumWidth(24)
            foldout.clicked.connect(lambda: self._toggle_foldout(key))
            name = QLabel(title, header)
            name.setStyleSheet("font-weight: 600;")
            row.addWidget(foldout)
            row.addWidget(name, 1)
            content.setVisible(not collapsed)
            layout.addWidget(header)
            layout.addWidget(content)
            return panel

        def _component_key(self, component: Any) -> str:
            if not self.selected:
                return f"component:{id(component)}"
            try:
                index = self.selected.components.index(component)
            except ValueError:
                index = -1
            return f"{self.selected.id}:{index}:{type(component).__name__}"

        def _toggle_component_collapsed(self, component: Any) -> None:
            key = self._component_key(component)
            self._toggle_foldout(key)

        def _toggle_foldout(self, key: str) -> None:
            self.collapsed_components[key] = not self.collapsed_components.get(key, False)
            self._populate_inspector()

        def _show_component_context_menu(self, component: Any, global_pos: Any) -> None:
            menu = QMenu(self)
            menu.addAction("Move Up", lambda: self._move_component(component, -1))
            menu.addAction("Move Down", lambda: self._move_component(component, 1))
            menu.addSeparator()
            menu.addAction("Copy Component", lambda: self._copy_component(component))
            menu.addAction("Paste Values", lambda: self._paste_component_over(component))
            menu.addAction("Reset", lambda: self._reset_component(component))
            menu.addSeparator()
            menu.addAction("Remove Component", lambda: self._remove_component(component))
            menu.exec(global_pos)

        def _component_buttons(self, component: Any) -> Any:
            row = QHBoxLayout()
            widget = QWidget(self.inspector)
            widget.setLayout(row)
            up_button = QPushButton("Up", widget)
            down_button = QPushButton("Down", widget)
            copy_button = QPushButton("Copy", widget)
            paste_button = QPushButton("Paste", widget)
            reset_button = QPushButton("Reset", widget)
            remove_button = QPushButton("Remove", widget)
            up_button.clicked.connect(lambda: self._move_component(component, -1))
            down_button.clicked.connect(lambda: self._move_component(component, 1))
            copy_button.clicked.connect(lambda: self._copy_component(component))
            paste_button.clicked.connect(lambda: self._paste_component_over(component))
            reset_button.clicked.connect(lambda: self._reset_component(component))
            remove_button.clicked.connect(lambda: self._remove_component(component))
            row.addWidget(up_button)
            row.addWidget(down_button)
            row.addWidget(copy_button)
            row.addWidget(paste_button)
            row.addWidget(reset_button)
            row.addWidget(remove_button)
            return widget

        def _copy_component(self, component: Any) -> None:
            if hasattr(component, "to_dict"):
                self.copied_component = component.to_dict()
                self._log(f"Copied {type(component).__name__}.")

        def _paste_component_over(self, component: Any) -> None:
            if not self.copied_component or not self.selected:
                return
            from p64.engine.components import component_from_dict

            replacement = component_from_dict(self.copied_component)
            for index, existing in enumerate(self.selected.components):
                if existing is component:
                    self.selected.components[index] = replacement
                    self._mark_dirty()
                    self._populate_inspector()
                    self.viewport.reload_assets()
                    return

        def _move_component(self, component: Any, direction: int) -> None:
            if not self.selected or not move_component_in_entity(self.selected, component, direction):
                return
            self._mark_dirty()
            self._populate_inspector()
            self.viewport.reload_assets()

        def _reset_component(self, component: Any) -> None:
            if not self.selected or component not in self.selected.components:
                return
            if isinstance(component, MeshRenderer):
                replacement = MeshRenderer()
            elif isinstance(component, Camera):
                replacement = Camera()
            elif isinstance(component, Light):
                replacement = Light()
            elif isinstance(component, AudioSource):
                replacement = AudioSource()
            elif isinstance(component, AudioListener):
                replacement = AudioListener()
            elif isinstance(component, Fog):
                replacement = Fog()
            elif isinstance(component, ScriptComponent):
                replacement = ScriptComponent()
            elif isinstance(component, SpawnPoint):
                replacement = SpawnPoint()
            elif isinstance(component, Collider):
                replacement = Collider()
            elif isinstance(component, CharacterController):
                replacement = CharacterController()
            elif isinstance(component, EntityPhysics):
                replacement = EntityPhysics()
            else:
                return
            index = self.selected.components.index(component)
            self.selected.components[index] = replacement
            self._mark_dirty()
            self._populate_inspector()
            self.viewport.reload_assets()

        def _remove_component(self, component: Any) -> None:
            if self.selected and component in self.selected.components:
                self.selected.components.remove(component)
                self._mark_dirty()
                self._populate_inspector()
                self.viewport.reload_assets()

        def _reset_transform(self) -> None:
            if not self.selected:
                return
            from p64.engine.components import Transform

            self.selected.transform = Transform()
            self._mark_dirty()
            self._populate_inspector()
            self.viewport.update()

        def _rename_selected(self, value: str) -> None:
            if self.selected:
                self.selected.name = value or self.selected.name
                self._mark_dirty()
                self._populate_hierarchy()
                self.viewport.update()

        def _set_selected_active(self, checked: bool) -> None:
            if self.selected:
                self.selected.active = checked
                self._mark_dirty()
                self.viewport.update()

        def _set_selected_persistent(self, checked: bool) -> None:
            if self.selected:
                self.selected.persistent = checked
                self._mark_dirty()

        def _set_selected_object_type(self, label: str) -> None:
            if not self.selected:
                return
            set_object_type_recursive(self.selected, GAME_OBJECT if label == "GameObject" else ENTITY)
            self._mark_dirty()
            self._populate_hierarchy()
            self._populate_inspector()
            self.viewport.update()

        def _set_component_enabled(self, component: Any, checked: bool) -> None:
            component.enabled = checked
            self._mark_dirty()
            self.viewport.update()

        def _set_mesh_visible(self, component: MeshRenderer, checked: bool) -> None:
            component.visible = checked
            self._mark_dirty()
            self.viewport.update()

        def _set_camera_active(self, component: Camera, checked: bool) -> None:
            component.active = checked
            self._mark_dirty()
            self.viewport.update()

        def _set_light_kind(self, component: Light, value: str) -> None:
            component.kind = value
            self._mark_dirty()
            self._populate_inspector()
            self.viewport.update()

        def _set_spawn_default(self, component: SpawnPoint, checked: bool) -> None:
            component.is_default = checked
            self._mark_dirty()
            self.viewport.update()

        def _set_collider_shape(self, component: Collider, value: str) -> None:
            component.shape = value
            if value in {"box", "sphere"} and self.selected:
                apply_mesh_primitive_defaults(self.project, self.selected, component, value)
            self._mark_dirty()
            self._populate_inspector()
            self.viewport.update()

        def _set_collider_trigger(self, component: Collider, checked: bool) -> None:
            component.is_trigger = checked
            self._mark_dirty()
            self.viewport.update()

        def _set_collider_fit_to_mesh(self, component: Collider, checked: bool) -> None:
            component.fit_to_mesh = checked
            self._mark_dirty()
            self.viewport.update()

        def _set_collider_convex(self, component: Collider, checked: bool) -> None:
            component.convex = checked
            self._mark_dirty()
            self.viewport.update()

        def _set_entity_physics_bool(self, component: EntityPhysics, name: str, checked: bool) -> None:
            setattr(component, name, checked)
            self._mark_dirty()
            self.viewport.update()

        def _set_audio_bool(self, component: AudioSource, name: str, checked: bool) -> None:
            setattr(component, name, checked)
            self._mark_dirty()
            self.viewport.update()

        def _set_audio_listener_active(self, component: AudioListener, checked: bool) -> None:
            component.active = checked
            self._mark_dirty()
            self.viewport.update()

        def _scene_render_settings(self) -> dict[str, Any]:
            scene = getattr(self, "scene", None)
            if scene is None:
                return default_render_settings()
            scene.render_settings = clamp_render_settings({**default_render_settings(), **scene.render_settings})
            return scene.render_settings

        def _set_scene_render_bool(self, key: str, value: bool) -> None:
            settings = self._scene_render_settings()
            settings[key] = bool(value)
            self._mark_dirty("Edit Scene Render Settings")
            self.viewport.update()

        def _set_scene_render_color(self, key: str, values: list[float]) -> None:
            settings = self._scene_render_settings()
            settings[key] = [float(values[0]), float(values[1]), float(values[2])]
            clamp_render_settings(settings)
            self._mark_dirty("Edit Scene Render Settings")
            self.viewport.update()

        def _set_scene_render_float(self, edit: Any, key: str, minimum: float, maximum: float) -> None:
            try:
                value = float(edit.text())
            except ValueError:
                self._log(f"Invalid number: {edit.text()}")
                return
            settings = self._scene_render_settings()
            settings[key] = max(minimum, min(maximum, value))
            edit.setText(str(settings[key]))
            self._mark_dirty("Edit Scene Render Settings")
            self.viewport.update()

        def _color_editor(self, value: Any, apply_callback: Any) -> Any:
            row = QHBoxLayout()
            widget = QWidget(self.inspector)
            widget.setLayout(row)
            values = _color_values(value)
            swatch = QPushButton("", widget)
            swatch.setMinimumWidth(48)
            swatch.setToolTip(_color_tooltip(values))
            swatch.setStyleSheet(_color_button_style(values))
            pick = QPushButton("Pick", widget)

            def choose_color() -> None:
                try:
                    from PySide6.QtGui import QColor
                except Exception:
                    return
                initial = QColor.fromRgbF(values[0], values[1], values[2])
                color = QColorDialog.getColor(initial, self, "Choose Color")
                if not color.isValid():
                    return
                apply_callback([float(color.redF()), float(color.greenF()), float(color.blueF())])
                self._populate_inspector()
                self.viewport.reload_assets()

            swatch.clicked.connect(choose_color)
            pick.clicked.connect(choose_color)
            row.addWidget(swatch)
            row.addWidget(pick)
            row.addStretch(1)
            return widget

        def _set_vec3_color(self, vec: Vec3, values: list[float]) -> None:
            vec.x, vec.y, vec.z = values
            self._mark_dirty()
            self.viewport.update()

        def _vec3_editor(self, vec: Vec3) -> Any:
            row = QHBoxLayout()
            widget = QWidget(self.inspector)
            widget.setLayout(row)
            for label, attr in [("X", "x"), ("Y", "y"), ("Z", "z")]:
                edit = QLineEdit(str(getattr(vec, attr)), widget)
                edit.setPlaceholderText(label)
                edit.editingFinished.connect(lambda edit=edit, attr=attr: self._apply_vec3_part(edit, vec, attr))
                row.addWidget(QLabel(label, widget))
                row.addWidget(edit)
            return widget

        def _apply_vec3_part(self, edit: Any, vec: Vec3, attr: str) -> None:
            try:
                value = float(edit.text())
            except ValueError:
                self._log(f"Invalid number: {edit.text()}")
                return
            setattr(vec, attr, value)
            self._mark_dirty()
            self.viewport.update()

        def _apply_float(self, edit: Any, target: Any, name: str) -> None:
            try:
                value = float(edit.text())
            except ValueError:
                self._log(f"Invalid number: {edit.text()}")
                return
            setattr(target, name, value)
            self._mark_dirty()
            self.viewport.update()

        def _apply_text(self, edit: Any, target: Any, name: str) -> None:
            setattr(target, name, edit.text())
            self._mark_dirty()
            self.viewport.update()

        def _add_mesh_renderer(self) -> None:
            if not self.selected:
                return
            choices = self._mesh_choice_items()
            component = MeshRenderer()
            if choices:
                _label, metadata, mesh_entry = choices[0]
                component.mesh = str(mesh_entry.get("id") or metadata.id)
                component.submesh = None
                source_materials = [str(item) for item in mesh_entry.get("material_slots", [])]
                component.material = source_materials[0] if source_materials else None
                self._sync_mesh_materials(component, metadata)
            self.selected.add_component(component)
            self._mark_dirty()
            self._populate_inspector()
            self.viewport.reload_assets()

        def _add_fog(self) -> None:
            if self.selected:
                self.selected.add_component(Fog())
                self._mark_dirty()
                self._populate_inspector()
                self.viewport.update()

        def _add_audio_source(self) -> None:
            if not self.selected:
                return
            component = AudioSource()
            choices = self._audio_clip_choices()
            if choices:
                _label, metadata = choices[0]
                component.clip = metadata.id
            self.selected.add_component(component)
            self._mark_dirty()
            self._populate_inspector()
            self.viewport.reload_assets()

        def _add_script_component(self) -> None:
            if not self.selected:
                return
            component = ScriptComponent()
            self.selected.add_component(component)
            self._mark_dirty()
            self._add_script_entry(component)

        def _add_script_entry(self, component: ScriptComponent) -> None:
            script = self._script_files()[0] if self._script_files() else ""
            classes = self._classes_for_script(script)
            component.scripts.append(ScriptEntry(script=script, class_name=classes[0] if len(classes) == 1 else ""))
            self._mark_dirty()
            self._populate_inspector()

        def _remove_script_entry(self, component: ScriptComponent, index: int) -> None:
            if 0 <= index < len(component.scripts):
                component.scripts.pop(index)
                self._mark_dirty()
                self._populate_inspector()

        def _move_script_entry(self, component: ScriptComponent, index: int, direction: int) -> None:
            if not move_script_entry_in_component(component, index, direction):
                return
            self._mark_dirty()
            self._populate_inspector()

        def _update_script_entry(
            self,
            component: ScriptComponent,
            entry: ScriptEntry,
            script: str | None = None,
            class_name: str | None = None,
            enabled: bool | None = None,
            class_combo: Any | None = None,
        ) -> None:
            if script is not None:
                entry.script = script
                classes = self._classes_for_script(script)
                if len(classes) == 1:
                    entry.class_name = classes[0]
                if class_combo is not None:
                    self._populate_class_combo(class_combo, script, entry.class_name)
            if class_name is not None:
                entry.class_name = class_name
            if enabled is not None:
                entry.enabled = enabled
            self._mark_dirty()
            self.viewport.update()

        def _populate_class_combo(self, combo: Any, script: str, selected: str) -> None:
            combo.blockSignals(True)
            combo.clear()
            classes = self._classes_for_script(script)
            combo.addItems(classes)
            if selected and selected not in classes:
                combo.addItem(selected)
            combo.setCurrentText(selected or (classes[0] if len(classes) == 1 else ""))
            combo.blockSignals(False)

        def _script_files(self) -> list[str]:
            if not self.project or not self.project.scripts_dir.exists():
                return []
            return sorted(
                path.relative_to(self.project.scripts_dir).as_posix()
                for path in self.project.scripts_dir.rglob("*.py")
                if path.name != "p64_project_api.py"
            )

        def _scene_files(self) -> list[str]:
            if not self.project:
                return []
            scenes: list[str] = []
            for folder in [self.project.scenes_dir, self.project.legacy_scenes_dir]:
                if not folder.exists():
                    continue
                for path in folder.rglob("*.scenep64"):
                    scenes.append(path.relative_to(self.project.root).as_posix())
            return sorted(set(scenes))

        def _classes_for_script(self, script: str) -> list[str]:
            if not self.project or not script:
                return []
            path = self.project.scripts_dir / script
            if not path.exists():
                return []
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                return []
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == "GameScript":
                            classes.append(node.name)
                        elif isinstance(base, ast.Attribute) and base.attr == "GameScript":
                            classes.append(node.name)
            return sorted(classes)

        def _mesh_choices(self) -> list[tuple[str, AssetMetadata]]:
            if not self.project or not self.project.assets_dir.exists():
                return []
            choices: list[tuple[str, AssetMetadata]] = []
            for metadata_path in discover_metadata(self.project.assets_dir):
                try:
                    metadata = AssetMetadata.load(metadata_path)
                except Exception:
                    continue
                if metadata.kind == "obj_mesh":
                    choices.append((f"{metadata.id}  ({metadata.source})", metadata))
            return choices

        def _mesh_choice_items(self) -> list[tuple[str, AssetMetadata, dict[str, Any]]]:
            choices: list[tuple[str, AssetMetadata, dict[str, Any]]] = []
            for label, metadata in self._mesh_choices():
                meshes = model_meshes(metadata)
                if not meshes:
                    for group in metadata.groups:
                        choices.append((f"{group}  ({metadata.source})", metadata, {"id": metadata.id, "name": group, "legacy_submesh": group, "material_slots": list(metadata.materials)}))
                    continue
                for mesh in meshes:
                    choices.append((f"{mesh.get('node_path', mesh.get('name', 'Mesh'))}  ({metadata.source})", metadata, mesh))
            return choices

        def _search_combo(self, items: list[str]) -> Any:
            combo = QComboBox(self.inspector)
            combo.setEditable(True)
            combo.addItems(items)
            completer = QCompleter(items, combo)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setFilterMode(Qt.MatchContains)
            combo.setCompleter(completer)
            return combo

        def _metadata_for_mesh(self, mesh_id: str) -> AssetMetadata | None:
            metadata_by_id = {metadata.id: metadata for _label, metadata in self._mesh_choices()}
            metadata, _mesh = resolve_model_mesh(metadata_by_id, mesh_id)
            return metadata

        def _mesh_entry_for_component(self, component: MeshRenderer) -> dict[str, Any] | None:
            metadata_by_id = {metadata.id: metadata for _label, metadata in self._mesh_choices()}
            _metadata, mesh = resolve_model_mesh(metadata_by_id, component.mesh, component.submesh)
            return mesh

        def _source_materials_for_component(self, component: MeshRenderer) -> list[str]:
            if component.source_materials:
                return list(component.source_materials)
            mesh_entry = self._mesh_entry_for_component(component)
            if mesh_entry:
                return [str(item) for item in mesh_entry.get("material_slots", [])]
            metadata = self._metadata_for_mesh(component.mesh)
            if metadata:
                return list(metadata.materials)
            return []

        def _sync_mesh_materials(self, component: MeshRenderer, metadata: AssetMetadata | None = None) -> None:
            metadata = metadata or self._metadata_for_mesh(component.mesh)
            mesh_entry = self._mesh_entry_for_component(component)
            component.source_materials = [str(item) for item in mesh_entry.get("material_slots", [])] if mesh_entry else (list(metadata.materials) if metadata else [])
            material_assets = metadata.settings.get("material_assets", {}) if metadata else {}
            slots: list[str | None] = []
            for index, material in enumerate(component.source_materials):
                existing = component.material_slots[index] if index < len(component.material_slots) else None
                mapped = material_assets.get(material) if isinstance(material_assets, dict) else None
                slots.append(existing or (str(mapped) if mapped else None))
            component.material_slots = slots

        def _populate_mesh_dependent_combos(self, component: MeshRenderer, material_combo: Any) -> None:
            metadata = self._metadata_for_mesh(component.mesh)
            material_combo.blockSignals(True)
            material_combo.clear()
            source_materials = self._source_materials_for_component(component)
            material_combo.addItems(source_materials)
            if component.material and (metadata is None or component.material not in metadata.materials):
                material_combo.addItem(component.material)
            material_combo.setCurrentText(component.material or "")
            material_combo.blockSignals(False)

        def _set_mesh_from_label(
            self,
            component: MeshRenderer,
            label: str,
            label_to_choice: dict[str, tuple[AssetMetadata, dict[str, Any]]],
            material_combo: Any,
        ) -> None:
            choice = label_to_choice.get(label)
            if choice is None:
                choice = next(((metadata, mesh) for _label, metadata, mesh in self._mesh_choice_items() if mesh.get("id") == label), None)
            if choice is None:
                return
            metadata, mesh_entry = choice
            component.mesh = str(mesh_entry.get("id") or metadata.id)
            component.submesh = mesh_entry.get("legacy_submesh")
            source_materials = [str(item) for item in mesh_entry.get("material_slots", [])]
            component.material = source_materials[0] if source_materials else None
            self._sync_mesh_materials(component, metadata)
            self._populate_mesh_dependent_combos(component, material_combo)
            self._mark_dirty()
            self.viewport.reload_assets()

        def _set_mesh_submesh(self, component: MeshRenderer, value: str) -> None:
            component.submesh = value or None
            self._mark_dirty()
            self.viewport.update()

        def _set_mesh_material(self, component: MeshRenderer, value: str) -> None:
            component.material = value or None
            metadata = self._metadata_for_mesh(component.mesh)
            if metadata and not component.source_materials:
                self._sync_mesh_materials(component, metadata)
            self._mark_dirty()
            self.viewport.reload_assets()

        def _set_audio_clip(self, component: AudioSource, label: str, label_to_id: dict[str, str]) -> None:
            component.clip = "" if label == "None" else label_to_id.get(label, label)
            self._mark_dirty()
            self.viewport.reload_assets()

        def _set_mesh_shader(self, component: MeshRenderer, label: str, label_to_id: dict[str, str]) -> None:
            component.shader = normalize_shader_id(label_to_id.get(label))
            self._mark_dirty()
            self.viewport.reload_assets()

        def _set_material_slot(self, component: MeshRenderer, index: int, label: str, label_to_id: dict[str, str]) -> None:
            while len(component.material_slots) <= index:
                component.material_slots.append(None)
            component.material_slots[index] = label_to_id.get(label) if label != "None" else None
            if component.material_slots[index] is None and label not in {"", "None"}:
                component.material_slots[index] = label
            self._mark_dirty()
            self._populate_inspector()
            self.viewport.reload_assets()

        def _set_material_shader(self, path: Path, material: MaterialAsset, label: str, label_to_id: dict[str, str]) -> None:
            shader = normalize_shader_id(label_to_id.get(label) or label)
            if not shader:
                return
            material.shader = shader
            material.save(path)
            self._populate_inspector()
            self.viewport.reload_assets()

        def _apply_material_property(self, path: Path, material: MaterialAsset, name: str, text: str) -> None:
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                try:
                    value = float(text)
                except ValueError:
                    value = text
            material.properties[name] = value
            material.save(path)
            self.viewport.reload_assets()

        def _apply_material_color_property(self, path: Path, material: MaterialAsset, name: str, values: list[float]) -> None:
            material.properties[name] = [float(values[0]), float(values[1]), float(values[2])]
            material.save(path)
            self.viewport.reload_assets()

        def _apply_material_texture(self, path: Path, material: MaterialAsset, name: str, text: str) -> None:
            material.textures[name] = text
            material.save(path)
            self.viewport.reload_assets()

        def _texture_editor(self, path: Path, material: MaterialAsset, name: str, value: str) -> Any:
            row = QWidget(self.inspector)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            preview = QLabel(row)
            preview.setFixedSize(34, 34)
            preview.setStyleSheet("border: 1px solid #111; background: #202124;")
            edit = QLineEdit(value, row)
            pick = QPushButton("Pick", row)

            def refresh_preview() -> None:
                pixmap = self._texture_pixmap_for_reference(edit.text(), 32)
                if pixmap is None:
                    preview.clear()
                    preview.setText("")
                else:
                    preview.setPixmap(pixmap)

            def apply_value() -> None:
                self._apply_material_texture(path, material, name, edit.text())
                refresh_preview()

            def choose() -> None:
                selected = self._choose_texture_reference(edit.text())
                if selected is None:
                    return
                edit.setText(selected)
                apply_value()

            edit.editingFinished.connect(apply_value)
            pick.clicked.connect(choose)
            layout.addWidget(preview)
            layout.addWidget(edit, 1)
            layout.addWidget(pick)
            refresh_preview()
            return row

        def _texture_pixmap_for_reference(self, reference: str, size: int = 64) -> Any | None:
            if not self.project or not reference:
                return None
            path = self._resolve_texture_reference(reference)
            if path is None or not path.exists() or path.suffix.lower() not in IMAGE_SUFFIXES:
                return None
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                return None
            return pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.FastTransformation)

        def _resolve_texture_reference(self, reference: str) -> Path | None:
            if not self.project:
                return None
            path = Path(reference)
            if path.is_absolute():
                return path
            candidates = [self.project.root / reference, self.project.assets_dir / reference, self.project.packages_dir / reference]
            return next((candidate for candidate in candidates if candidate.exists()), candidates[0] if candidates else None)

        def _choose_texture_reference(self, current: str = "") -> str | None:
            if not self.project:
                return None
            try:
                from PySide6.QtCore import QSize
                from PySide6.QtGui import QIcon
                from PySide6.QtWidgets import QDialog, QDialogButtonBox, QListView, QListWidget, QListWidgetItem, QVBoxLayout
            except Exception as exc:
                self._log(f"Texture picker unavailable: {exc}")
                return None

            dialog = QDialog(self)
            dialog.setWindowTitle("Pick Texture")
            layout = QVBoxLayout(dialog)
            grid = QListWidget(dialog)
            grid.setViewMode(QListView.ViewMode.IconMode)
            grid.setResizeMode(QListView.ResizeMode.Adjust)
            grid.setMovement(QListView.Movement.Static)
            grid.setIconSize(QSize(72, 72))
            grid.setGridSize(QSize(120, 104))
            grid.setUniformItemSizes(True)
            current_path = self._resolve_texture_reference(current)
            for image_path in texture_image_paths(self.project):
                reference = project_texture_reference(self.project, image_path)
                item = QListWidgetItem(QIcon(str(image_path)), image_path.name)
                item.setToolTip(reference)
                item.setData(Qt.UserRole, reference)
                grid.addItem(item)
                if current_path and image_path.resolve() == current_path.resolve():
                    grid.setCurrentItem(item)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            grid.itemDoubleClicked.connect(lambda _item: dialog.accept())
            layout.addWidget(grid)
            layout.addWidget(buttons)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            item = grid.currentItem()
            return str(item.data(Qt.UserRole)) if item else None

        def _reset_material(self, path: Path) -> None:
            if not self.project:
                return
            reset_material_asset(self.project, path)
            self._populate_inspector()
            self.viewport.reload_assets()

        def _extract_materials(self, path: Path) -> None:
            if not self.project:
                return
            start = self._material_extract_start_folder(path)
            folder = QFileDialog.getExistingDirectory(self, "Extract Materials To", str(start))
            if not folder:
                return
            output_dir = Path(folder)
            try:
                output_dir.resolve().relative_to(self.project.assets_dir.resolve())
            except ValueError:
                QMessageBox.warning(
                    self,
                    "External Material Folder",
                    "This folder is outside Assets. Materials will work by absolute path, but they will not appear in the Asset Browser.",
                )
            try:
                materials = extract_materials_for_obj(self.project, path, output_dir)
                self._refresh_assets_from_watcher()
                self._populate_inspector()
                self._log(f"Extracted {len(materials)} material(s) from {path.name}.")
            except Exception as exc:
                self._log(f"Extract materials failed: {exc}")

        def _refresh_audio_asset(self, path: Path) -> None:
            if not self.project:
                return
            try:
                metadata = import_audio_asset(self.project, path)
                self._refresh_assets_from_watcher()
                self._populate_inspector()
                self._log(f"Imported audio clip: {metadata.id}")
            except Exception as exc:
                self._log(f"Audio import failed: {exc}")

        def _material_extract_start_folder(self, path: Path) -> Path:
            if not self.project:
                return path.parent
            metadata_path = find_metadata_for_source(path)
            if metadata_path and metadata_path.exists():
                try:
                    metadata = AssetMetadata.load(metadata_path)
                    folder = metadata.settings.get("material_extract_folder")
                    if folder:
                        candidate = Path(str(folder))
                        return candidate if candidate.is_absolute() else self.project.root / candidate
                except Exception:
                    pass
            return self.project.assets_dir / "materials" / path.stem

        def _split_mesh_renderer(self, component: MeshRenderer) -> None:
            if not self.selected:
                return
            metadata = self._metadata_for_mesh(component.mesh)
            if not metadata:
                self._log("No mesh metadata found for split.")
                return
            created = split_mesh_renderer_into_children(self.selected, metadata)
            self._mark_dirty()
            self._refresh_all()
            self._log(f"Created {len(created)} child GameObjects from mesh groups.")

        def _shader_choices(self) -> list[tuple[str, str]]:
            if not self.project:
                return []
            choices: list[tuple[str, str]] = []
            for shader_path in discover_shaders(self.project.assets_dir):
                shader_id = shader_asset_id(self.project.root, shader_path)
                try:
                    source = parse_shader(shader_path)
                    label = f"{source.name}  ({shader_id})"
                except Exception:
                    label = f"{shader_path.stem}  ({shader_id})"
                choices.append((label, shader_id))
            return choices

        def _material_choices(self) -> list[tuple[str, str]]:
            if not self.project or not self.project.assets_dir.exists():
                return []
            choices: list[tuple[str, str]] = []
            for path in self.project.assets_dir.rglob("*.material"):
                material_id = material_asset_id(self.project.root, path)
                choices.append((f"{path.stem}  ({material_id})", material_id))
            return sorted(choices)

        def _audio_clip_choices(self) -> list[tuple[str, AssetMetadata]]:
            if not self.project or not self.project.assets_dir.exists():
                return []
            choices: list[tuple[str, AssetMetadata]] = []
            for metadata_path in discover_metadata(self.project.assets_dir):
                try:
                    metadata = AssetMetadata.load(metadata_path)
                except Exception:
                    continue
                if metadata.kind != "audio_clip":
                    continue
                info = audio_info(metadata) or {}
                label = f"{Path(metadata.source).stem}  ({metadata.id}, {info.get('imported_sample_rate', '?')} Hz)"
                choices.append((label, metadata))
            return sorted(choices, key=lambda item: item[0].lower())

        def _texture_summary(self, component: MeshRenderer) -> str:
            texture = self._texture_path_for(component)
            return str(texture.relative_to(self.project.root)) if texture and self.project else "No diffuse texture"

        def _texture_pixmap(self, component: MeshRenderer) -> Any | None:
            texture = self._texture_path_for(component)
            if not texture or not texture.exists():
                return None
            pixmap = QPixmap(str(texture))
            if pixmap.isNull():
                return None
            return pixmap.scaled(96, 96, Qt.KeepAspectRatio, Qt.FastTransformation)

        def _texture_path_for(self, component: MeshRenderer) -> Path | None:
            if not self.project:
                return None
            metadata = self._metadata_for_mesh(component.mesh)
            if not metadata:
                return None
            material = component.material
            material_defs = metadata.settings.get("material_defs", {})
            texture_name = material_defs.get(material, {}).get("diffuse_texture") if material else None
            if not texture_name:
                return None
            return (self.project.root / metadata.source).parent / str(texture_name)

        def _collider_mesh_source(self) -> str:
            if not self.selected:
                return "No SceneObject"
            for component in self.selected.components:
                if isinstance(component, MeshRenderer) and component.mesh:
                    return component.mesh
            return "No MeshRenderer"

    return InspectorMixin
