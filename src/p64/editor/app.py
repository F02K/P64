from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Callable

from p64.engine.assets import AssetMetadata, discover_metadata
from p64.engine.components import Camera, Fog, Light, MeshRenderer, ScriptComponent, ScriptEntry
from p64.engine.entity import Entity
from p64.engine.files import is_metadata_file
from p64.engine.math import Vec3
from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.engine.shader import discover_shaders, parse_shader, shader_asset_id
from p64.editor.ops import (
    DirtyTracker,
    add_component,
    create_script_template,
    create_shader_template,
    delete_entity,
    duplicate_entity,
    find_parent,
    insert_obj_scene_entity,
    split_mesh_renderer_into_children,
)
from p64.renderer.scene_renderer import RenderCamera, camera_basis


def launch_editor(project_path: Path | None = None) -> None:
    try:
        from PySide6.QtCore import QFileSystemWatcher, QPoint, Qt, QTimer, QUrl
        from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QPixmap, QShortcut, QSurfaceFormat
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QCompleter,
            QFileDialog,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QInputDialog,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMenu,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QSplitter,
            QTabBar,
            QTabWidget,
            QTreeWidget,
            QTreeWidgetItem,
            QVBoxLayout,
            QWidget,
        )
        try:
            from PySide6.QtOpenGLWidgets import QOpenGLWidget
        except ImportError:
            QOpenGLWidget = None
    except ImportError as exc:  # pragma: no cover - depends on optional GUI package
        raise RuntimeError("Install PySide6 to use the P64 editor.") from exc

    surface_format = QSurfaceFormat()
    surface_format.setVersion(3, 3)
    surface_format.setProfile(QSurfaceFormat.CoreProfile)
    surface_format.setDepthBufferSize(24)
    surface_format.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(surface_format)

    if QOpenGLWidget is not None:
        class Viewport(QOpenGLWidget):  # type: ignore[misc, valid-type]
            def __init__(
                self,
                project_getter: Callable[[], Project | None],
                scene_getter: Callable[[], Scene | None],
                selected_getter: Callable[[], Entity | None],
                selection_setter: Callable[[str | None], None],
                logger: Callable[[str], None],
            ) -> None:
                super().__init__()
                self.ctx = None
                self.renderer = None
                self.renderer_project: Project | None = None
                self.qt_framebuffer = None
                self.logged_framebuffer = False
                self.view_mode = "Scene"
                self.scene_camera = RenderCamera(position=Vec3(0.0, 2.5, 8.0), rotation=Vec3(-15.0, 0.0, 0.0), fov=60.0)
                self.keys_down: set[int] = set()
                self.mouse_look = False
                self.last_mouse_pos: QPoint | None = None
                self.move_speed = 5.0
                self.project_getter = project_getter
                self.scene_getter = scene_getter
                self.selected_getter = selected_getter
                self.selection_setter = selection_setter
                self.logger = logger
                self.setFocusPolicy(Qt.StrongFocus)
                self.setMouseTracking(True)

            def initializeGL(self) -> None:
                try:
                    import moderngl

                    self.ctx = moderngl.create_context(require=330)
                    self.ctx.enable(moderngl.DEPTH_TEST)
                    self.logger("ModernGL viewport initialized.")
                except Exception as exc:
                    self.ctx = None
                    self.logger(f"P64 viewport could not initialize ModernGL: {exc}")

            def resizeGL(self, width: int, height: int) -> None:
                if self.ctx:
                    self._bind_qt_framebuffer()
                    self.ctx.viewport = (0, 0, width, height)

            def paintGL(self) -> None:
                if not self.ctx:
                    return
                self._bind_qt_framebuffer()
                project = self.project_getter()
                scene = self.scene_getter()
                if not project or not scene:
                    self.ctx.clear(0.16, 0.18, 0.21, 1.0)
                    return
                try:
                    if self.renderer is None or self.renderer_project != project:
                        from p64.renderer.scene_renderer import SceneRenderer

                        self.renderer = SceneRenderer(self.ctx, project, self.logger)
                        self.renderer_project = project
                    camera = self.scene_camera if self.view_mode == "Scene" else None
                    selected = self.selected_getter()
                    self.renderer.render(
                        scene,
                        self.width(),
                        self.height(),
                        camera=camera,
                        selected_entity_id=selected.id if selected else None,
                        show_grid=self.view_mode == "Scene",
                    )
                except Exception as exc:
                    self.ctx.clear(0.16, 0.18, 0.21, 1.0)
                    self.logger(f"Render failed: {exc}")

            def reload_assets(self) -> None:
                if self.renderer:
                    self.renderer.reload_assets()
                self.update()

            def set_view_mode(self, mode: str) -> None:
                self.view_mode = mode
                self.setFocus()
                self.update()

            def tick(self, dt: float) -> None:
                if self.view_mode == "Scene" and self.mouse_look:
                    self._move_scene_camera(dt)
                self.update()

            def keyPressEvent(self, event: Any) -> None:
                if not event.isAutoRepeat():
                    self.keys_down.add(event.key())
                super().keyPressEvent(event)

            def keyReleaseEvent(self, event: Any) -> None:
                if not event.isAutoRepeat():
                    self.keys_down.discard(event.key())
                super().keyReleaseEvent(event)

            def mousePressEvent(self, event: Any) -> None:
                if event.button() == Qt.RightButton:
                    self.mouse_look = True
                    self.last_mouse_pos = event.position().toPoint()
                    self.setCursor(Qt.BlankCursor)
                    self.setFocus()
                elif event.button() == Qt.LeftButton:
                    self._pick_scene_object(event.position().x(), event.position().y())
                super().mousePressEvent(event)

            def mouseReleaseEvent(self, event: Any) -> None:
                if event.button() == Qt.RightButton:
                    self.mouse_look = False
                    self.last_mouse_pos = None
                    self.unsetCursor()
                super().mouseReleaseEvent(event)

            def mouseMoveEvent(self, event: Any) -> None:
                if self.mouse_look and self.last_mouse_pos is not None:
                    current = event.position().toPoint()
                    delta = current - self.last_mouse_pos
                    self.last_mouse_pos = current
                    self.scene_camera.rotation.y += delta.x() * 0.15
                    self.scene_camera.rotation.x = max(-89.0, min(89.0, self.scene_camera.rotation.x - delta.y() * 0.15))
                    self.update()
                super().mouseMoveEvent(event)

            def wheelEvent(self, event: Any) -> None:
                if self.view_mode == "Scene":
                    self.move_speed = max(0.5, min(40.0, self.move_speed + event.angleDelta().y() / 240.0))
                    self.logger(f"Scene camera speed: {self.move_speed:.1f}")
                super().wheelEvent(event)

            def _move_scene_camera(self, dt: float) -> None:
                forward, right, up = camera_basis(self.scene_camera.rotation)
                speed = self.move_speed * (3.0 if Qt.Key_Shift in self.keys_down else 1.0)
                movement = Vec3()
                if Qt.Key_W in self.keys_down:
                    movement = _add_vec3(movement, forward)
                if Qt.Key_S in self.keys_down:
                    movement = _sub_vec3(movement, forward)
                if Qt.Key_D in self.keys_down:
                    movement = _add_vec3(movement, right)
                if Qt.Key_A in self.keys_down:
                    movement = _sub_vec3(movement, right)
                if Qt.Key_E in self.keys_down:
                    movement = _add_vec3(movement, up)
                if Qt.Key_Q in self.keys_down:
                    movement = _sub_vec3(movement, up)
                self.scene_camera.position = _add_vec3(self.scene_camera.position, _scale_vec3(movement, speed * dt))

            def _bind_qt_framebuffer(self) -> None:
                if not self.ctx:
                    return
                framebuffer_id = self.defaultFramebufferObject()
                if self.qt_framebuffer is None or getattr(self.qt_framebuffer, "glo", None) != framebuffer_id:
                    self.qt_framebuffer = self.ctx.detect_framebuffer(framebuffer_id)
                    if not self.logged_framebuffer:
                        self.logger(f"Viewport framebuffer bound: {framebuffer_id}")
                        self.logged_framebuffer = True
                self.qt_framebuffer.use()

            def _pick_scene_object(self, x: float, y: float) -> None:
                if self.view_mode != "Scene" or not self.renderer:
                    return
                scene = self.scene_getter()
                if not scene:
                    return
                selected_id = self.renderer.pick_entity(scene, self.width(), self.height(), x, y, camera=self.scene_camera)
                self.selection_setter(selected_id)
    else:
        class Viewport(QWidget):  # type: ignore[no-redef]
            def __init__(
                self,
                project_getter: Callable[[], Project | None],
                scene_getter: Callable[[], Scene | None],
                selected_getter: Callable[[], Entity | None],
                selection_setter: Callable[[str | None], None],
                logger: Callable[[str], None],
            ) -> None:
                super().__init__()
                layout = QVBoxLayout(self)
                label = QLabel("P64 Viewport\nInstall PySide6 OpenGL widgets for accelerated rendering.")
                label.setAlignment(Qt.AlignCenter)
                layout.addWidget(label)

            def reload_assets(self) -> None:
                pass

            def set_view_mode(self, mode: str) -> None:
                pass

            def tick(self, dt: float) -> None:
                pass

    class MainWindow(QMainWindow):
        def __init__(self, project: Project | None) -> None:
            super().__init__()
            self.setWindowTitle("P64 Editor")
            self.resize(1280, 760)
            self.project = project
            self.scene = project.load_startup_scene() if project else None
            self.selected: Entity | None = None
            self.selected_asset: Path | None = None
            self.dirty = DirtyTracker()
            self.copied_component: dict[str, Any] | None = None
            self.asset_watcher = QFileSystemWatcher(self)

            self.hierarchy = QTreeWidget()
            self.hierarchy.setHeaderLabel("Hierarchy")
            self.hierarchy.itemSelectionChanged.connect(self._select_from_tree)
            self.hierarchy.setContextMenuPolicy(Qt.CustomContextMenu)
            self.hierarchy.customContextMenuRequested.connect(self._show_hierarchy_menu)

            self.inspector = QWidget()
            self.inspector_layout = QVBoxLayout(self.inspector)

            self.assets = QTreeWidget()
            self.assets.setHeaderLabel("Assets")
            self.assets.setContextMenuPolicy(Qt.CustomContextMenu)
            self.assets.customContextMenuRequested.connect(self._show_asset_menu)
            self.assets.itemDoubleClicked.connect(self._asset_double_clicked)
            self.assets.itemSelectionChanged.connect(self._asset_selection_changed)

            self.console = QPlainTextEdit()
            self.console.setReadOnly(True)

            self.viewport = Viewport(lambda: self.project, lambda: self.scene, lambda: self.selected, self._select_entity_by_id, self._log)
            self.repaint_timer = QTimer(self)
            self.repaint_timer.timeout.connect(lambda: self.viewport.tick(1 / 30))
            self.repaint_timer.start(33)

            self.view_tabs = QTabBar()
            self.view_tabs.addTab("Scene")
            self.view_tabs.addTab("Game")
            self.view_tabs.currentChanged.connect(self._view_tab_changed)
            self.viewport_status = QLabel("Scene | Speed 5.0 | No selection")
            viewport_panel = QWidget()
            viewport_layout = QVBoxLayout(viewport_panel)
            viewport_layout.setContentsMargins(0, 0, 0, 0)
            viewport_layout.addWidget(self.view_tabs)
            viewport_layout.addWidget(self.viewport, 1)
            viewport_layout.addWidget(self.viewport_status)

            center = QSplitter(Qt.Horizontal)
            center.addWidget(self.hierarchy)
            center.addWidget(viewport_panel)
            center.addWidget(self.inspector)
            center.setSizes([260, 720, 300])

            bottom = QSplitter(Qt.Vertical)
            bottom.addWidget(center)
            bottom_tabs = QTabWidget()
            bottom_tabs.addTab(self.assets, "Assets")
            bottom_tabs.addTab(self.console, "Console")
            bottom.addWidget(bottom_tabs)
            bottom.setSizes([600, 160])
            self.setCentralWidget(bottom)

            open_button = QPushButton("Open Project")
            open_button.clicked.connect(self._open_project)
            new_button = QPushButton("New Entity")
            new_button.clicked.connect(self._create_entity)
            duplicate_button = QPushButton("Duplicate")
            duplicate_button.clicked.connect(self._duplicate_selected)
            delete_button = QPushButton("Delete")
            delete_button.clicked.connect(self._delete_selected)
            frame_button = QPushButton("Frame")
            frame_button.clicked.connect(self._frame_selected)
            run_button = QPushButton("Run")
            run_button.clicked.connect(self._run_project)
            save_button = QPushButton("Save Scene")
            save_button.clicked.connect(self._save_scene)
            bundle_button = QPushButton("Build Bundle")
            bundle_button.clicked.connect(self._build_bundle)
            build_button = QPushButton("Build")
            build_button.clicked.connect(self._build_project)
            toolbar = self.addToolBar("Project")
            toolbar.addWidget(open_button)
            toolbar.addWidget(new_button)
            toolbar.addWidget(duplicate_button)
            toolbar.addWidget(delete_button)
            toolbar.addWidget(frame_button)
            toolbar.addWidget(run_button)
            toolbar.addWidget(save_button)
            toolbar.addWidget(bundle_button)
            toolbar.addWidget(build_button)

            self.asset_watcher.directoryChanged.connect(lambda _path: self._refresh_assets_from_watcher())
            self.asset_watcher.fileChanged.connect(lambda _path: self._refresh_assets_from_watcher())
            self._install_shortcuts()
            self._refresh_all()
            self._update_window_title()

        def _open_project(self) -> None:
            if not self._confirm_discard_changes():
                return
            folder = QFileDialog.getExistingDirectory(self, "Open P64 Project")
            if not folder:
                return
            try:
                self.project = Project.load(Path(folder))
                self.scene = self.project.load_startup_scene()
                self.selected = None
                self.selected_asset = None
                self.dirty.mark_saved()
                self.viewport.reload_assets()
                self._log(f"Opened {self.project.root}")
                self._refresh_all()
                self._update_window_title()
            except Exception as exc:
                QMessageBox.critical(self, "Open failed", str(exc))

        def _save_scene(self) -> None:
            if self.project and self.scene:
                self.project.save_startup_scene(self.scene)
                self.dirty.mark_saved()
                self._update_window_title()
                self._log("Scene saved.")

        def closeEvent(self, event: Any) -> None:
            if self._confirm_discard_changes():
                event.accept()
            else:
                event.ignore()

        def _build_project(self) -> None:
            if not self.project:
                self._log("No project open.")
                return
            self._save_scene()
            try:
                from p64.build.pipeline import build_executable

                output = build_executable(self.project.root, run_pyinstaller=True)
                self._log(f"Build complete: {output}")
            except Exception as exc:
                self._log(f"Build failed: {exc}")

        def _build_bundle(self) -> None:
            if not self.project:
                self._log("No project open.")
                return
            self._save_scene()
            try:
                from p64.build.pipeline import create_runtime_bundle

                output = create_runtime_bundle(self.project.root)
                self._log(f"Bundle complete: {output}")
            except Exception as exc:
                self._log(f"Bundle failed: {exc}")

        def _run_project(self) -> None:
            if not self.project:
                self._log("No project open.")
                return
            self._save_scene()
            try:
                from p64.engine.runtime import run_project

                self._log("Running project...")
                run_project(self.project.root)
            except Exception as exc:
                self._log(f"Run failed: {exc}")

        def _install_shortcuts(self) -> None:
            shortcuts = [
                ("Ctrl+S", self._save_scene),
                ("Delete", self._delete_selected),
                ("F", self._frame_selected),
                ("Ctrl+D", self._duplicate_selected),
                ("F2", self._rename_selected_dialog),
            ]
            for key, callback in shortcuts:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(callback)

        def _mark_dirty(self) -> None:
            self.dirty.mark_dirty()
            self._update_window_title()

        def _update_window_title(self) -> None:
            name = self.project.name if self.project else "No Project"
            mark = "*" if self.dirty.dirty else ""
            self.setWindowTitle(f"P64 Editor - {name}{mark}")

        def _confirm_discard_changes(self) -> bool:
            if not self.dirty.dirty:
                return True
            result = QMessageBox.question(
                self,
                "Unsaved changes",
                "Save changes before continuing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if result == QMessageBox.Save:
                self._save_scene()
                return not self.dirty.dirty
            return result == QMessageBox.Discard

        def _view_tab_changed(self, index: int) -> None:
            mode = "Scene" if index == 0 else "Game"
            self.viewport.set_view_mode(mode)
            self._update_viewport_status()
            self._log(f"{mode} view active.")

        def _refresh_all(self) -> None:
            self._populate_hierarchy()
            self._populate_assets()
            self._populate_inspector()
            self.viewport.update()

        def _refresh_assets_from_watcher(self) -> None:
            self._populate_assets()
            self.viewport.reload_assets()

        def _populate_hierarchy(self) -> None:
            self.hierarchy.clear()
            if not self.scene:
                return
            for entity in self.scene.entities:
                self.hierarchy.addTopLevelItem(self._entity_item(entity))
            self.hierarchy.expandAll()

        def _entity_item(self, entity: Entity) -> Any:
            item = QTreeWidgetItem([entity.name])
            item.setData(0, Qt.UserRole, entity.id)
            for child in entity.children:
                item.addChild(self._entity_item(child))
            for label in self._virtual_submesh_labels(entity):
                virtual = QTreeWidgetItem([label])
                virtual.setFlags(virtual.flags() & ~Qt.ItemIsSelectable)
                item.addChild(virtual)
            return item

        def _populate_assets(self) -> None:
            self.assets.clear()
            self._reset_asset_watcher()
            if not self.project or not self.project.assets_dir.exists():
                return
            root_item = QTreeWidgetItem([self.project.assets_dir.name])
            root_item.setData(0, Qt.UserRole, str(self.project.assets_dir))
            self.assets.addTopLevelItem(root_item)
            self._add_asset_children(root_item, self.project.assets_dir)
            self.assets.expandAll()

        def _add_asset_children(self, item: Any, folder: Path) -> None:
            for path in sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                child = QTreeWidgetItem([path.name])
                child.setData(0, Qt.UserRole, str(path))
                item.addChild(child)
                if path.is_dir():
                    self._add_asset_children(child, path)

        def _reset_asset_watcher(self) -> None:
            if self.asset_watcher.directories():
                self.asset_watcher.removePaths(self.asset_watcher.directories())
            if self.asset_watcher.files():
                self.asset_watcher.removePaths(self.asset_watcher.files())
            if not self.project or not self.project.assets_dir.exists():
                return
            paths = [str(path) for path in self.project.assets_dir.rglob("*")]
            paths.append(str(self.project.assets_dir))
            if paths:
                self.asset_watcher.addPaths(paths)

        def _select_from_tree(self) -> None:
            if not self.scene:
                return
            items = self.hierarchy.selectedItems()
            self.selected = self.scene.find(items[0].data(0, Qt.UserRole)) if items else None
            self.selected_asset = None
            self._populate_inspector()
            self._update_viewport_status()

        def _select_entity_by_id(self, entity_id: str | None) -> None:
            if not self.scene or not entity_id:
                return
            self.selected = self.scene.find(entity_id)
            self.selected_asset = None
            self._select_hierarchy_item(entity_id)
            self._populate_inspector()
            self._update_viewport_status()
            self.viewport.update()

        def _select_hierarchy_item(self, entity_id: str) -> None:
            def visit(item: Any) -> bool:
                if item.data(0, Qt.UserRole) == entity_id:
                    self.hierarchy.setCurrentItem(item)
                    return True
                for index in range(item.childCount()):
                    if visit(item.child(index)):
                        return True
                return False

            for index in range(self.hierarchy.topLevelItemCount()):
                if visit(self.hierarchy.topLevelItem(index)):
                    return

        def _update_viewport_status(self) -> None:
            mode = getattr(self.viewport, "view_mode", "Scene")
            speed = getattr(self.viewport, "move_speed", 0.0)
            if self.selected:
                selection = self.selected.name
            elif self.selected_asset:
                selection = f"Asset: {self.selected_asset.name}"
            else:
                selection = "No selection"
            self.viewport_status.setText(f"{mode} | Speed {speed:.1f} | {selection} | RMB+WASD/QE, Shift speed, F frame")

        def _create_entity(self) -> None:
            if not self.scene:
                return
            entity = Entity("Entity")
            self.scene.add_entity(entity)
            self.selected = entity
            self._mark_dirty()
            self._refresh_all()

        def _add_child_entity(self) -> None:
            if not self.selected:
                self._create_entity()
                return
            child = Entity("Entity")
            self.selected.add_child(child)
            self.selected = child
            self._mark_dirty()
            self._refresh_all()

        def _duplicate_selected(self) -> None:
            if not self.scene or not self.selected:
                return
            duplicate = duplicate_entity(self.selected)
            parent = find_parent(self.scene, self.selected.id)
            if parent:
                parent.add_child(duplicate)
            else:
                self.scene.add_entity(duplicate)
            self.selected = duplicate
            self._mark_dirty()
            self._refresh_all()

        def _delete_selected(self) -> None:
            if not self.scene or not self.selected:
                return
            deleted = delete_entity(self.scene, self.selected.id)
            if deleted:
                self.selected = None
                self._mark_dirty()
                self._refresh_all()

        def _rename_selected_dialog(self) -> None:
            if not self.selected:
                return
            value, ok = QInputDialog.getText(self, "Rename Entity", "Name:", text=self.selected.name)
            if ok and value:
                self.selected.name = value
                self._mark_dirty()
                self._refresh_all()

        def _frame_selected(self) -> None:
            if not self.selected:
                return
            position = self.selected.transform.position
            self.viewport.scene_camera.position = Vec3(position.x, position.y + 2.0, position.z + 8.0)
            self.viewport.scene_camera.rotation = Vec3(-15.0, 0.0, 0.0)
            self.viewport.set_view_mode("Scene")
            self.view_tabs.setCurrentIndex(0)
            self._update_viewport_status()

        def _show_hierarchy_menu(self, pos: Any) -> None:
            menu = QMenu(self)
            menu.addAction("Create Empty", self._create_entity)
            menu.addAction("Add Child", self._add_child_entity)
            menu.addSeparator()
            menu.addAction("Duplicate", self._duplicate_selected)
            menu.addAction("Delete", self._delete_selected)
            menu.addAction("Rename", self._rename_selected_dialog)
            menu.addAction("Frame Selected", self._frame_selected)
            menu.addSeparator()
            for component_name in ["MeshRenderer", "Camera", "Light", "Fog", "ScriptComponent"]:
                menu.addAction(f"Add {component_name}", lambda checked=False, name=component_name: self._add_component_by_name(name))
            menu.exec(self.hierarchy.viewport().mapToGlobal(pos))

        def _show_asset_menu(self, pos: Any) -> None:
            path = self._selected_asset_path()
            menu = QMenu(self)
            if path and (path.suffix.lower() == ".obj" or is_metadata_file(path)):
                menu.addAction("Import OBJ into Scene", lambda: self._import_asset_obj(path))
            menu.addAction("Create Shader", self._create_shader_asset)
            menu.addAction("Create Script", self._create_script_asset)
            menu.addAction("Refresh Assets", self._refresh_assets_from_watcher)
            if path:
                menu.addSeparator()
                menu.addAction("Open", lambda: self._open_path(path))
                menu.addAction("Reveal in Explorer", lambda: self._reveal_path(path))
            menu.exec(self.assets.viewport().mapToGlobal(pos))

        def _asset_double_clicked(self, item: Any) -> None:
            path = Path(item.data(0, Qt.UserRole))
            if path.is_dir():
                return
            if path.suffix.lower() == ".obj" or is_metadata_file(path):
                self._import_asset_obj(path)
            elif path.suffix.lower() in {".shader", ".py"}:
                self._open_path(path)

        def _asset_selection_changed(self) -> None:
            path = self._selected_asset_path()
            if not path or not path.exists():
                self.selected_asset = None
                self._populate_inspector()
                return
            self.selected_asset = path
            self.selected = None
            self.hierarchy.blockSignals(True)
            self.hierarchy.clearSelection()
            self.hierarchy.blockSignals(False)
            self._populate_inspector()
            self._update_viewport_status()

        def _selected_asset_path(self) -> Path | None:
            items = self.assets.selectedItems()
            if not items:
                return None
            data = items[0].data(0, Qt.UserRole)
            return Path(data) if data else None

        def _import_asset_obj(self, path: Path) -> None:
            if not self.project or not self.scene:
                return
            try:
                entity = insert_obj_scene_entity(self.project, self.scene, path)
                self.selected = entity
                self._mark_dirty()
                self._refresh_all()
                self._log(f"Imported OBJ into scene: {path}")
            except Exception as exc:
                self._log(f"Import failed: {exc}")

        def _create_shader_asset(self) -> None:
            if not self.project:
                return
            path = create_shader_template(self.project.assets_dir)
            self._refresh_assets_from_watcher()
            self._log(f"Created shader: {path}")
            self._open_path(path)

        def _create_script_asset(self) -> None:
            if not self.project:
                return
            path = create_script_template(self.project.scripts_dir)
            self._refresh_assets_from_watcher()
            self._log(f"Created script: {path}")
            self._open_path(path)

        def _open_path(self, path: Path) -> None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

        def _reveal_path(self, path: Path) -> None:
            try:
                os.startfile(str(path.parent.resolve()))
            except Exception as exc:
                self._log(f"Reveal failed: {exc}")

        def _add_component_by_name(self, component_name: str) -> None:
            if not self.selected:
                return
            try:
                add_component(self.selected, component_name)
                self._mark_dirty()
                self._populate_inspector()
                self.viewport.update()
            except Exception as exc:
                self._log(str(exc))

        def _populate_inspector(self) -> None:
            while self.inspector_layout.count():
                child = self.inspector_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            if self.selected_asset:
                self._populate_asset_inspector(self.selected_asset)
                return
            if not self.selected:
                self.inspector_layout.addWidget(QLabel("No entity selected"))
                return

            self._add_entity_header()
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
                else:
                    self.inspector_layout.addWidget(QLabel(component_summary(component)))
            self._add_component_controls()
            self.inspector_layout.addStretch(1)

        def _populate_asset_inspector(self, path: Path) -> None:
            title = QLabel(path.name)
            title.setWordWrap(True)
            self.inspector_layout.addWidget(title)
            form = QFormLayout()
            form.addRow("Path", QLabel(str(path.relative_to(self.project.root)) if self.project else str(path)))
            form.addRow("Type", QLabel("Folder" if path.is_dir() else path.suffix or "File"))
            box = QGroupBox("Asset")
            box.setLayout(form)
            self.inspector_layout.addWidget(box)

            actions = QHBoxLayout()
            action_widget = QWidget()
            action_widget.setLayout(actions)
            open_button = QPushButton("Open")
            open_button.clicked.connect(lambda: self._open_path(path))
            reveal_button = QPushButton("Reveal")
            reveal_button.clicked.connect(lambda: self._reveal_path(path))
            actions.addWidget(open_button)
            actions.addWidget(reveal_button)
            if path.suffix.lower() == ".obj" or is_metadata_file(path):
                import_button = QPushButton("Import")
                import_button.clicked.connect(lambda: self._import_asset_obj(path))
                actions.addWidget(import_button)
            self.inspector_layout.addWidget(action_widget)

            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                pixmap = QPixmap(str(path))
                if not pixmap.isNull():
                    preview = QLabel()
                    preview.setAlignment(Qt.AlignCenter)
                    preview.setPixmap(pixmap.scaled(220, 220, Qt.KeepAspectRatio, Qt.FastTransformation))
                    self.inspector_layout.addWidget(preview)
            elif is_metadata_file(path):
                try:
                    metadata = AssetMetadata.load(path)
                    meta = QFormLayout()
                    meta.addRow("ID", QLabel(metadata.id))
                    meta.addRow("Kind", QLabel(metadata.kind))
                    meta.addRow("Source", QLabel(metadata.source))
                    meta.addRow("Groups", QLabel(", ".join(metadata.groups)))
                    meta.addRow("Materials", QLabel(", ".join(metadata.materials)))
                    meta_box = QGroupBox("Metadata")
                    meta_box.setLayout(meta)
                    self.inspector_layout.addWidget(meta_box)
                except Exception as exc:
                    self.inspector_layout.addWidget(QLabel(f"Metadata error: {exc}"))
            elif path.suffix.lower() == ".shader":
                try:
                    shader = parse_shader(path)
                    shader_form = QFormLayout()
                    shader_form.addRow("Name", QLabel(shader.name))
                    shader_form.addRow("Vertex", QLabel(f"{len(shader.vertex.splitlines())} lines"))
                    shader_form.addRow("Fragment", QLabel(f"{len(shader.fragment.splitlines())} lines"))
                    shader_box = QGroupBox("Shader")
                    shader_box.setLayout(shader_form)
                    self.inspector_layout.addWidget(shader_box)
                except Exception as exc:
                    self.inspector_layout.addWidget(QLabel(f"Shader parse error: {exc}"))
            self.inspector_layout.addStretch(1)

        def _add_entity_header(self) -> None:
            if not self.selected:
                return
            form = QFormLayout()
            name = QLineEdit(self.selected.name)
            name.editingFinished.connect(lambda: self._rename_selected(name.text()))
            active = QCheckBox()
            active.setChecked(self.selected.active)
            active.toggled.connect(lambda checked: self._set_selected_active(checked))
            form.addRow("Name", name)
            form.addRow("Active", active)
            box = QGroupBox("Entity")
            box.setLayout(form)
            self.inspector_layout.addWidget(box)

        def _add_transform_editor(self) -> None:
            if not self.selected:
                return
            transform = self.selected.transform
            form = QFormLayout()
            form.addRow("Position", self._vec3_editor(transform.position))
            form.addRow("Rotation", self._vec3_editor(transform.rotation))
            form.addRow("Scale", self._vec3_editor(transform.scale))
            box = QGroupBox("Transform")
            box.setLayout(form)
            self.inspector_layout.addWidget(box)

        def _add_script_component_editor(self, component: ScriptComponent) -> None:
            box = QGroupBox("ScriptComponent")
            layout = QVBoxLayout(box)
            layout.addLayout(self._component_buttons(component))
            enabled = QCheckBox("Enabled")
            enabled.setChecked(component.enabled)
            enabled.toggled.connect(lambda checked: self._set_component_enabled(component, checked))
            layout.addWidget(enabled)

            for index, entry in enumerate(component.scripts):
                layout.addLayout(self._script_row(component, entry, index))

            add_script = QPushButton("Add Script")
            add_script.clicked.connect(lambda: self._add_script_entry(component))
            layout.addWidget(add_script)
            self.inspector_layout.addWidget(box)

        def _add_mesh_renderer_editor(self, component: MeshRenderer) -> None:
            box = QGroupBox("MeshRenderer")
            form = QFormLayout(box)
            form.addRow("Actions", self._component_buttons(component))
            enabled = QCheckBox()
            enabled.setChecked(component.enabled)
            enabled.toggled.connect(lambda checked: self._set_component_enabled(component, checked))
            visible = QCheckBox()
            visible.setChecked(component.visible)
            visible.toggled.connect(lambda checked: self._set_mesh_visible(component, checked))

            mesh_combo = self._search_combo([label for label, _metadata in self._mesh_choices()])
            mesh_id_to_label = {metadata.id: label for label, metadata in self._mesh_choices()}
            label_to_metadata = dict(self._mesh_choices())
            mesh_combo.setCurrentText(mesh_id_to_label.get(component.mesh, component.mesh))
            submesh_combo = QComboBox()
            material_combo = QComboBox()
            material_combo.setEditable(True)
            self._populate_mesh_dependent_combos(component, submesh_combo, material_combo)

            mesh_combo.currentTextChanged.connect(
                lambda text: self._set_mesh_from_label(component, text, label_to_metadata, submesh_combo, material_combo)
            )
            submesh_combo.currentTextChanged.connect(lambda text: self._set_mesh_submesh(component, text))
            material_combo.currentTextChanged.connect(lambda text: self._set_mesh_material(component, text))

            form.addRow("Enabled", enabled)
            form.addRow("Visible", visible)
            form.addRow("Mesh", mesh_combo)
            form.addRow("Submesh", submesh_combo)
            form.addRow("Material", material_combo)
            shader_combo = self._search_combo([label for label, _path in self._shader_choices()])
            shader_label_by_id = {shader_id: label for label, shader_id in self._shader_choices()}
            shader_label_to_id = dict(self._shader_choices())
            shader_combo.addItem("<Built-in N64>")
            shader_combo.setCurrentText(shader_label_by_id.get(component.shader or "", "<Built-in N64>"))
            shader_combo.currentTextChanged.connect(lambda text: self._set_mesh_shader(component, text, shader_label_to_id))
            form.addRow("Shader", shader_combo)
            texture_label = QLabel(self._texture_summary(component))
            texture_label.setWordWrap(True)
            form.addRow("Texture", texture_label)
            pixmap = self._texture_pixmap(component)
            if pixmap:
                preview = QLabel()
                preview.setPixmap(pixmap)
                form.addRow("Preview", preview)
            split_button = QPushButton("Split Into Child Entities")
            split_button.clicked.connect(lambda: self._split_mesh_renderer(component))
            form.addRow("Sub-objects", split_button)
            self.inspector_layout.addWidget(box)

        def _add_fog_editor(self, component: Fog) -> None:
            box = QGroupBox("Fog Volume")
            form = QFormLayout(box)
            form.addRow("Actions", self._component_buttons(component))
            enabled = QCheckBox()
            enabled.setChecked(component.enabled)
            enabled.toggled.connect(lambda checked: self._set_component_enabled(component, checked))
            near = QLineEdit(str(component.near))
            far = QLineEdit(str(component.far))
            density = QLineEdit(str(component.density))
            near.editingFinished.connect(lambda: self._apply_float(near, component, "near"))
            far.editingFinished.connect(lambda: self._apply_float(far, component, "far"))
            density.editingFinished.connect(lambda: self._apply_float(density, component, "density"))
            form.addRow("Enabled", enabled)
            form.addRow("Color", self._vec3_editor(component.color))
            form.addRow("Size", self._vec3_editor(component.size))
            form.addRow("Near", near)
            form.addRow("Far", far)
            form.addRow("Density", density)
            self.inspector_layout.addWidget(box)

        def _add_camera_editor(self, component: Camera) -> None:
            box = QGroupBox("Camera")
            form = QFormLayout(box)
            form.addRow("Actions", self._component_buttons(component))
            enabled = QCheckBox()
            enabled.setChecked(component.enabled)
            enabled.toggled.connect(lambda checked: self._set_component_enabled(component, checked))
            active = QCheckBox()
            active.setChecked(component.active)
            active.toggled.connect(lambda checked: self._set_camera_active(component, checked))
            fov = QLineEdit(str(component.fov))
            near = QLineEdit(str(component.near))
            far = QLineEdit(str(component.far))
            fov.editingFinished.connect(lambda: self._apply_float(fov, component, "fov"))
            near.editingFinished.connect(lambda: self._apply_float(near, component, "near"))
            far.editingFinished.connect(lambda: self._apply_float(far, component, "far"))
            form.addRow("Enabled", enabled)
            form.addRow("Active", active)
            form.addRow("FOV", fov)
            form.addRow("Near", near)
            form.addRow("Far", far)
            self.inspector_layout.addWidget(box)

        def _add_light_editor(self, component: Light) -> None:
            box = QGroupBox("Light")
            form = QFormLayout(box)
            form.addRow("Actions", self._component_buttons(component))
            enabled = QCheckBox()
            enabled.setChecked(component.enabled)
            enabled.toggled.connect(lambda checked: self._set_component_enabled(component, checked))
            kind = QComboBox()
            kind.addItems(["directional", "point"])
            kind.setCurrentText(component.kind)
            kind.currentTextChanged.connect(lambda text: self._set_light_kind(component, text))
            intensity = QLineEdit(str(component.intensity))
            intensity.editingFinished.connect(lambda: self._apply_float(intensity, component, "intensity"))
            form.addRow("Enabled", enabled)
            form.addRow("Kind", kind)
            form.addRow("Color", self._vec3_editor(component.color))
            form.addRow("Intensity", intensity)
            self.inspector_layout.addWidget(box)

        def _script_row(self, component: ScriptComponent, entry: ScriptEntry, index: int) -> Any:
            row = QHBoxLayout()
            script_combo = QComboBox()
            scripts = self._script_files()
            script_combo.addItems(scripts)
            if entry.script and entry.script not in scripts:
                script_combo.addItem(entry.script)
            script_combo.setCurrentText(entry.script)

            class_combo = QComboBox()
            class_combo.setEditable(True)
            self._populate_class_combo(class_combo, script_combo.currentText(), entry.class_name)

            enabled = QCheckBox()
            enabled.setChecked(entry.enabled)
            remove = QPushButton("Remove")

            script_combo.currentTextChanged.connect(
                lambda text: self._update_script_entry(component, entry, script=text, class_combo=class_combo)
            )
            class_combo.currentTextChanged.connect(lambda text: self._update_script_entry(component, entry, class_name=text))
            enabled.toggled.connect(lambda checked: self._update_script_entry(component, entry, enabled=checked))
            remove.clicked.connect(lambda: self._remove_script_entry(component, index))

            row.addWidget(enabled)
            row.addWidget(script_combo)
            row.addWidget(class_combo)
            row.addWidget(remove)
            return row

        def _add_component_controls(self) -> None:
            row = QHBoxLayout()
            add_script = QPushButton("Add ScriptComponent")
            add_script.clicked.connect(self._add_script_component)
            add_mesh = QPushButton("Add MeshRenderer")
            add_mesh.clicked.connect(self._add_mesh_renderer)
            add_camera = QPushButton("Add Camera")
            add_camera.clicked.connect(lambda: self._add_component_by_name("Camera"))
            add_light = QPushButton("Add Light")
            add_light.clicked.connect(lambda: self._add_component_by_name("Light"))
            add_fog = QPushButton("Add Fog")
            add_fog.clicked.connect(self._add_fog)
            row.addWidget(add_mesh)
            row.addWidget(add_camera)
            row.addWidget(add_light)
            row.addWidget(add_fog)
            row.addWidget(add_script)
            box = QGroupBox("Add Component")
            box.setLayout(row)
            self.inspector_layout.addWidget(box)

        def _component_buttons(self, component: Any) -> Any:
            row = QHBoxLayout()
            widget = QWidget()
            widget.setLayout(row)
            copy_button = QPushButton("Copy")
            paste_button = QPushButton("Paste")
            remove_button = QPushButton("Remove")
            copy_button.clicked.connect(lambda: self._copy_component(component))
            paste_button.clicked.connect(lambda: self._paste_component_over(component))
            remove_button.clicked.connect(lambda: self._remove_component(component))
            row.addWidget(copy_button)
            row.addWidget(paste_button)
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

        def _remove_component(self, component: Any) -> None:
            if self.selected and component in self.selected.components:
                self.selected.components.remove(component)
                self._mark_dirty()
                self._populate_inspector()
                self.viewport.reload_assets()

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
            self.viewport.update()

        def _vec3_editor(self, vec: Vec3) -> Any:
            row = QHBoxLayout()
            widget = QWidget()
            widget.setLayout(row)
            for label, attr in [("X", "x"), ("Y", "y"), ("Z", "z")]:
                edit = QLineEdit(str(getattr(vec, attr)))
                edit.setPlaceholderText(label)
                edit.editingFinished.connect(lambda edit=edit, attr=attr: self._apply_vec3_part(edit, vec, attr))
                row.addWidget(QLabel(label))
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
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(items)
            completer = QCompleter(items)
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
            component.shader = label_to_id.get(label)
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
            self._log(f"Created {len(created)} child entities from mesh groups.")

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

        def _virtual_submesh_labels(self, entity: Entity) -> list[str]:
            labels: list[str] = []
            child_names = {child.name for child in entity.children}
            for component in entity.components:
                if not isinstance(component, MeshRenderer):
                    continue
                metadata = self._metadata_for_mesh(component.mesh)
                if not metadata:
                    continue
                for group in metadata.groups:
                    if group not in child_names:
                        labels.append(f"{group}  [submesh]")
            return labels

        def _log(self, text: str) -> None:
            self.console.appendPlainText(text)

    app = QApplication.instance() or QApplication([])
    project = Project.load(project_path) if project_path else None
    window = MainWindow(project)
    window.show()
    app.exec()


def launch_runtime_window(project: Project, scene: Scene) -> None:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install PySide6 to run P64 projects.") from exc

    app = QApplication.instance() or QApplication([])
    window = QWidget()
    window.setWindowTitle(project.name)
    layout = QVBoxLayout(window)
    label = QLabel(f"{project.name}\nScene: {scene.name}\nRenderer: ModernGL N64-style path")
    label.setAlignment(Qt.AlignCenter)
    layout.addWidget(label)
    window.resize(960, 720)
    window.show()
    app.exec()


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


def _add_vec3(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z)


def _sub_vec3(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z)


def _scale_vec3(v: Vec3, scale: float) -> Vec3:
    return Vec3(v.x * scale, v.y * scale, v.z * scale)
