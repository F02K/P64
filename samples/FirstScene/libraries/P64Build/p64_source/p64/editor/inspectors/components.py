from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from p64.editor.ops import (
    add_component,
    move_component as move_component_in_entity,
    move_script_entry as move_script_entry_in_component,
    split_mesh_renderer_into_children,
)
from p64.editor.panels.assets import is_preview_image
from p64.editor.panels.inspector import missing_reference_summary
from p64.editor.utils.ui import make_widget_compact
from p64.engine.assets import AssetMetadata, discover_metadata
from p64.engine.collision import apply_mesh_primitive_defaults
from p64.engine.components import Camera, CharacterController, Collider, EntityPhysics, Fog, Light, MeshRenderer, ScriptComponent, ScriptEntry, SpawnPoint
from p64.engine.entity import ENTITY, GAME_OBJECT, Entity, set_object_type_recursive
from p64.engine.files import is_metadata_file
from p64.engine.math import Vec3
from p64.engine.shader import discover_shaders, normalize_shader_id, parse_shader, shader_asset_id
from p64.engine.validation import entity_reference_errors


AVAILABLE_COMPONENTS = (
    "MeshRenderer",
    "Camera",
    "Light",
    "Fog",
    "SpawnPoint",
    "Collider",
    "CharacterController",
    "EntityPhysics",
    "ScriptComponent",
)


def component_summary(component: object) -> str:
    if isinstance(component, MeshRenderer):
        return f"MeshRenderer: {component.mesh} / {component.submesh or '*'}"
    if isinstance(component, Camera):
        return f"Camera: fov={component.fov} active={component.active}"
    if isinstance(component, Light):
        return f"Light: {component.kind} intensity={component.intensity}"
    if isinstance(component, Fog):
        return f"Fog: near={component.near} far={component.far}"
    return type(component).__name__


def create_inspector_mixin(
    QCheckBox: Any,
    QComboBox: Any,
    QCompleter: Any,
    QFormLayout: Any,
    QGroupBox: Any,
    QHBoxLayout: Any,
    QLabel: Any,
    QLineEdit: Any,
    QMenu: Any,
    QMessageBox: Any,
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
                self.inspector_layout.addWidget(QLabel("No SceneObject selected", self.inspector))
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
                    shader_box = QGroupBox("Shader", self.inspector)
                    shader_box.setLayout(shader_form)
                    self.inspector_layout.addWidget(shader_box)
                except Exception as exc:
                    self.inspector_layout.addWidget(QLabel(f"Shader parse error: {exc}", self.inspector))
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

            mesh_combo = self._search_combo([label for label, _metadata in self._mesh_choices()])
            mesh_id_to_label = {metadata.id: label for label, metadata in self._mesh_choices()}
            label_to_metadata = dict(self._mesh_choices())
            mesh_combo.setCurrentText(mesh_id_to_label.get(component.mesh, component.mesh))
            submesh_combo = QComboBox(content)
            material_combo = QComboBox(content)
            material_combo.setEditable(True)
            self._populate_mesh_dependent_combos(component, submesh_combo, material_combo)

            mesh_combo.currentTextChanged.connect(
                lambda text: self._set_mesh_from_label(component, text, label_to_metadata, submesh_combo, material_combo)
            )
            submesh_combo.currentTextChanged.connect(lambda text: self._set_mesh_submesh(component, text))
            material_combo.currentTextChanged.connect(lambda text: self._set_mesh_material(component, text))

            form.addRow("Visible", visible)
            form.addRow("Mesh", mesh_combo)
            form.addRow("Submesh", submesh_combo)
            form.addRow("Material", material_combo)
            shader_combo = self._search_combo([label for label, _path in self._shader_choices()])
            shader_label_by_id = {shader_id: label for label, shader_id in self._shader_choices()}
            shader_label_to_id = dict(self._shader_choices())
            shader_combo.addItem("Standard VertexLit")
            component.shader = normalize_shader_id(component.shader)
            shader_combo.setCurrentText(shader_label_by_id.get(component.shader or "", "Standard VertexLit"))
            shader_combo.currentTextChanged.connect(lambda text: self._set_mesh_shader(component, text, shader_label_to_id))
            form.addRow("Shader", shader_combo)
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

        def _add_fog_editor(self, component: Fog) -> None:
            content, form = self._component_content_widget()
            near = QLineEdit(str(component.near), content)
            far = QLineEdit(str(component.far), content)
            density = QLineEdit(str(component.density), content)
            near.editingFinished.connect(lambda: self._apply_float(near, component, "near"))
            far.editingFinished.connect(lambda: self._apply_float(far, component, "far"))
            density.editingFinished.connect(lambda: self._apply_float(density, component, "density"))
            form.addRow("Color", self._vec3_editor(component.color))
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
            range_edit = QLineEdit(str(component.range), content)
            range_edit.editingFinished.connect(lambda: self._apply_float(range_edit, component, "range"))
            falloff = QLineEdit(str(component.falloff), content)
            falloff.editingFinished.connect(lambda: self._apply_float(falloff, component, "falloff"))
            spot_angle = QLineEdit(str(component.spot_angle), content)
            spot_angle.editingFinished.connect(lambda: self._apply_float(spot_angle, component, "spot_angle"))
            form.addRow("Kind", kind)
            form.addRow("Color", self._vec3_editor(component.color))
            form.addRow("Intensity", intensity)
            if component.kind in {"point", "spot"}:
                form.addRow("Range", range_edit)
                form.addRow("Falloff", falloff)
            if component.kind == "spot":
                form.addRow("Spot Angle", spot_angle)
            self.inspector_layout.addWidget(self._component_panel(component, "Light", content))

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
            choices = self._mesh_choices()
            component = MeshRenderer()
            if choices:
                _label, metadata = choices[0]
                component.mesh = metadata.id
                component.submesh = metadata.groups[0] if metadata.groups else None
                component.material = metadata.materials[0] if metadata.materials else None
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
            return sorted(path.relative_to(self.project.scripts_dir).as_posix() for path in self.project.scripts_dir.rglob("*.py"))

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
                        if isinstance(base, ast.Name) and base.id == "UserScript":
                            classes.append(node.name)
                        elif isinstance(base, ast.Attribute) and base.attr == "UserScript":
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
            for _label, metadata in self._mesh_choices():
                if metadata.id == mesh_id:
                    return metadata
            return None

        def _populate_mesh_dependent_combos(self, component: MeshRenderer, submesh_combo: Any, material_combo: Any) -> None:
            metadata = self._metadata_for_mesh(component.mesh)
            submesh_combo.blockSignals(True)
            material_combo.blockSignals(True)
            submesh_combo.clear()
            material_combo.clear()
            submesh_combo.addItems(metadata.groups if metadata else [])
            material_combo.addItems(metadata.materials if metadata else [])
            if component.submesh and (metadata is None or component.submesh not in metadata.groups):
                submesh_combo.addItem(component.submesh)
            if component.material and (metadata is None or component.material not in metadata.materials):
                material_combo.addItem(component.material)
            submesh_combo.setCurrentText(component.submesh or "")
            material_combo.setCurrentText(component.material or "")
            submesh_combo.blockSignals(False)
            material_combo.blockSignals(False)

        def _set_mesh_from_label(
            self,
            component: MeshRenderer,
            label: str,
            label_to_metadata: dict[str, AssetMetadata],
            submesh_combo: Any,
            material_combo: Any,
        ) -> None:
            metadata = label_to_metadata.get(label)
            if metadata is None:
                for _label, item in self._mesh_choices():
                    if item.id == label:
                        metadata = item
                        break
            if metadata is None:
                return
            component.mesh = metadata.id
            component.submesh = metadata.groups[0] if metadata.groups else None
            component.material = metadata.materials[0] if metadata.materials else None
            self._populate_mesh_dependent_combos(component, submesh_combo, material_combo)
            self._mark_dirty()
            self.viewport.reload_assets()

        def _set_mesh_submesh(self, component: MeshRenderer, value: str) -> None:
            component.submesh = value or None
            self._mark_dirty()
            self.viewport.update()

        def _set_mesh_material(self, component: MeshRenderer, value: str) -> None:
            component.material = value or None
            self._mark_dirty()
            self.viewport.reload_assets()

        def _set_mesh_shader(self, component: MeshRenderer, label: str, label_to_id: dict[str, str]) -> None:
            component.shader = normalize_shader_id(label_to_id.get(label))
            self._mark_dirty()
            self.viewport.reload_assets()

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
                    return f"{component.mesh} / {component.submesh or '*'}"
            return "No MeshRenderer"

    return InspectorMixin
