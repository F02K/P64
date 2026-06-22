from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from p64.engine.entity import Entity
from p64.engine.audio import ensure_audio_clips_for_assets
from p64.engine.project import Project
from p64.engine.runtime_session import RuntimeSession
from p64.engine.scene import Scene
from p64.engine.vscode import setup_vscode_project
from p64.editor.dialogs.build_settings import open_build_settings_dialog
from p64.editor.dialogs.lighting_settings import open_lighting_settings_dialog
from p64.editor.dialogs.project_settings import open_project_settings_dialog
from p64.editor.inspectors.components import create_inspector_mixin
from p64.editor.ops import DirtyTracker, update_material_usage_cache
from p64.editor.panels.analysis import create_analysis_panel
from p64.editor.panels.assets import create_asset_browser_mixin
from p64.editor.panels.console import create_console_panel
from p64.editor.panels.hierarchy import create_hierarchy_mixin
from p64.editor.profiler import ProfilerRecorder
from p64.editor.undo import UndoManager, UndoState
from p64.editor.viewport import create_viewport_class


def launch_editor(project_path: Path | None = None) -> None:
    try:
        from PySide6.QtCore import QElapsedTimer, QFileSystemWatcher, QPoint, QSize, Qt, QTimer, QUrl
        from PySide6.QtGui import QAction, QBrush, QColor, QDesktopServices, QIcon, QKeySequence, QPixmap, QShortcut, QSurfaceFormat
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QColorDialog,
            QComboBox,
            QCompleter,
            QDialog,
            QDialogButtonBox,
            QFileDialog,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QInputDialog,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMenu,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSplitter,
            QStyle,
            QTabBar,
            QTabWidget,
            QTableWidget,
            QTableWidgetItem,
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
    surface_format.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(surface_format)

    Viewport = create_viewport_class(QOpenGLWidget, QWidget, QLabel, QVBoxLayout, Qt)
    AssetBrowserMixin = create_asset_browser_mixin(
        QTreeWidgetItem, QListWidgetItem, QIcon, QStyle, Qt, QMenu, QInputDialog, QMessageBox, QDesktopServices, QUrl
    )
    HierarchyMixin = create_hierarchy_mixin(QTreeWidgetItem, QBrush, QColor, Qt, QMenu, QInputDialog)
    InspectorMixin = create_inspector_mixin(
        QCheckBox, QColorDialog, QComboBox, QCompleter, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QMenu, QMessageBox, QFileDialog, QPixmap, QPushButton, QSizePolicy, Qt, QVBoxLayout, QWidget
    )
    ConsolePanel = create_console_panel(
        QApplication, QBrush, QCheckBox, QColor, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
        QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, Qt
    )
    AnalysisPanel = create_analysis_panel(QLabel, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QTimer, Qt)

    class MainWindow(AssetBrowserMixin, HierarchyMixin, InspectorMixin, QMainWindow):
        def __init__(self, project: Project | None) -> None:
            super().__init__()
            self.setWindowTitle("P64 Editor")
            self.resize(1280, 760)
            self.project = project
            self.scene = project.load_startup_scene() if project else None
            self.current_scene_path = project.resolve_scene_path(project.startup_scene) if project else None
            self.selected: Entity | None = None
            self.selected_asset: Path | None = None
            self.current_asset_folder: Path | None = project.assets_dir if project else None
            self.dirty = DirtyTracker()
            self.undo = UndoManager()
            self._restoring_history = False
            self.copied_component: dict[str, Any] | None = None
            self.collapsed_components: dict[str, bool] = {}
            self.current_transform_tool = "move"
            self.play_session: RuntimeSession | None = None
            self.asset_watcher = QFileSystemWatcher(self)
            self._updating_asset_grid = False
            self.profiler_recorder = ProfilerRecorder()
            self.analysis_window: QWidget | None = None

            self.hierarchy = QTreeWidget()
            self.hierarchy.setHeaderLabel("Hierarchy")
            self.hierarchy.itemSelectionChanged.connect(self._select_from_tree)
            self.hierarchy.setContextMenuPolicy(Qt.CustomContextMenu)
            self.hierarchy.customContextMenuRequested.connect(self._show_hierarchy_menu)

            self.inspector = QWidget()
            self.inspector.setContextMenuPolicy(Qt.CustomContextMenu)
            self.inspector.customContextMenuRequested.connect(self._show_inspector_context_menu)
            self.inspector_layout = QVBoxLayout(self.inspector)
            self.inspector_layout.setContentsMargins(6, 6, 6, 6)
            self.inspector_scroll = QScrollArea()
            self.inspector_scroll.setWidgetResizable(True)
            self.inspector_scroll.setWidget(self.inspector)

            self.asset_folders = QTreeWidget()
            self.asset_folders.setHeaderLabel("Project")
            self.asset_folders.itemSelectionChanged.connect(self._asset_folder_selection_changed)
            self.asset_folders.setContextMenuPolicy(Qt.CustomContextMenu)
            self.asset_folders.customContextMenuRequested.connect(self._show_asset_folder_menu)
            self.assets = QListWidget()
            self.assets.setViewMode(QListWidget.IconMode)
            self.assets.setIconSize(QSize(36, 36))
            self.assets.setResizeMode(QListWidget.Adjust)
            self.assets.setGridSize(QSize(112, 82))
            self.assets.setMovement(QListWidget.Static)
            self.assets.setContextMenuPolicy(Qt.CustomContextMenu)
            self.assets.customContextMenuRequested.connect(self._show_asset_menu)
            self.assets.itemDoubleClicked.connect(self._asset_double_clicked)
            self.assets.itemSelectionChanged.connect(self._asset_selection_changed)
            self.assets.itemChanged.connect(self._asset_item_changed)
            asset_browser = QSplitter(Qt.Horizontal)
            asset_browser.addWidget(self.asset_folders)
            asset_browser.addWidget(self.assets)
            asset_browser.setSizes([260, 640])

            self.console = ConsolePanel(self)

            self.viewport = Viewport(
                lambda: self.project,
                self._viewport_scene,
                lambda: self.selected,
                self._select_entity_by_id,
                self._log,
                lambda: self.play_session.input if self.play_session else None,
                self._scene_changed_live,
                self._begin_scene_edit,
                self._commit_scene_edit,
                lambda: self.profiler_recorder if self.profiler_recorder.enabled else None,
            )
            self.frame_clock = QElapsedTimer()
            self.frame_clock.start()
            self.repaint_timer = QTimer(self)
            self.repaint_timer.setTimerType(Qt.PreciseTimer)
            self.repaint_timer.timeout.connect(self._tick_viewport)
            self.repaint_timer.start(16)

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
            center.addWidget(self.inspector_scroll)
            center.setSizes([260, 720, 300])

            bottom = QSplitter(Qt.Vertical)
            bottom.addWidget(center)
            bottom_tabs = QTabWidget()
            bottom_tabs.addTab(asset_browser, "Assets")
            bottom_tabs.addTab(self.console, "Console")
            bottom.addWidget(bottom_tabs)
            bottom.setSizes([600, 160])
            self.setCentralWidget(bottom)

            open_button = QPushButton("Open Project")
            open_button.clicked.connect(self._open_project)
            project_button = QPushButton("Project")
            project_menu = QMenu(project_button)
            project_menu.addAction("Save Scene", self._save_scene)
            project_menu.addAction("Setup VSCode", self._setup_vscode)
            project_menu.addAction("Project Settings", self._open_project_settings)
            project_menu.addAction("Build Settings", self._open_build_settings)
            project_button.setMenu(project_menu)
            entity_button = QPushButton("Entity")
            entity_menu = QMenu(entity_button)
            entity_menu.addAction("New Entity", self._create_entity)
            entity_menu.addAction("Duplicate", self._duplicate_selected)
            entity_menu.addAction("Delete", self._delete_selected)
            entity_button.setMenu(entity_menu)
            frame_button = QPushButton("Frame")
            frame_button.clicked.connect(self._frame_selected)
            self.tool_buttons: dict[str, QPushButton] = {}
            move_button = QPushButton("Move")
            rotate_button = QPushButton("Rotate")
            scale_button = QPushButton("Scale")
            for tool, button in [("move", move_button), ("rotate", rotate_button), ("scale", scale_button)]:
                button.setCheckable(True)
                button.clicked.connect(lambda checked=False, tool=tool: self._set_transform_tool(tool))
                self.tool_buttons[tool] = button
            self.play_button = QPushButton("Play")
            self.play_button.clicked.connect(self._toggle_playmode)
            toolbar = self.addToolBar("Project")
            toolbar.addWidget(open_button)
            toolbar.addWidget(project_button)
            toolbar.addWidget(entity_button)
            toolbar.addWidget(frame_button)
            toolbar.addSeparator()
            toolbar.addWidget(move_button)
            toolbar.addWidget(rotate_button)
            toolbar.addWidget(scale_button)
            toolbar.addSeparator()
            toolbar.addWidget(self.play_button)
            window_menu = self.menuBar().addMenu("Window")
            window_menu.addAction("Lighting Settings", self._open_lighting_settings)
            window_menu.addAction("Analysis", self._open_analysis)

            self.asset_watcher.directoryChanged.connect(lambda _path: self._refresh_assets_from_watcher())
            self.asset_watcher.fileChanged.connect(lambda _path: self._refresh_assets_from_watcher())
            self._install_shortcuts()
            self._set_transform_tool("move")
            self._reset_undo_history()
            self._refresh_all()
            self._update_window_title()

        def _tick_viewport(self) -> None:
            elapsed_ms = self.frame_clock.restart()
            dt = max(0.001, elapsed_ms / 1000.0)
            profiler = self.profiler_recorder if self.profiler_recorder.enabled else None
            frame = None
            if profiler is not None:
                frame = profiler.begin_frame("Editor")
            try:
                with _profiler_section(profiler, "editor tick"):
                    if self.play_session:
                        with _profiler_section(profiler, "playmode tick"):
                            self.play_session.profiler_recorder = profiler
                            for error in self.play_session.tick(dt):
                                self._log(f"Playmode script error: {error}")
                    with _profiler_section(profiler, "viewport tick"):
                        self.viewport.tick(dt)
            finally:
                if profiler is not None:
                    profiler.end_frame(frame)

        def _viewport_scene(self) -> Scene | None:
            return self.play_session.scene if self.play_session else self.scene

        def _open_project(self) -> None:
            self._stop_playmode()
            if not self._confirm_discard_changes():
                return
            folder = QFileDialog.getExistingDirectory(self, "Open P64 Project")
            if not folder:
                return
            try:
                self.project = Project.load(Path(folder))
                self.scene = self.project.load_startup_scene()
                self.current_scene_path = self.project.resolve_scene_path(self.project.startup_scene)
                self.current_asset_folder = self.project.assets_dir
                self.selected = None
                self.selected_asset = None
                self.dirty.mark_saved()
                self._reset_undo_history()
                self.viewport.reload_assets()
                self._log(f"Opened {self.project.root}")
                self._refresh_all()
                self._update_window_title()
            except Exception as exc:
                QMessageBox.critical(self, "Open failed", str(exc))

        def _save_scene(self) -> None:
            if self.project and self.scene:
                path = self.current_scene_path or self.project.resolve_scene_path(self.project.startup_scene)
                update_material_usage_cache(self.project, self.scene, path)
                self.scene.save(path)
                self.undo.mark_saved()
                self.dirty.mark_saved()
                self._update_window_title()
                self._log(f"Scene saved: {path}")

        def closeEvent(self, event: Any) -> None:
            if self._confirm_discard_changes():
                event.accept()
            else:
                event.ignore()

        def _build_project(self) -> None:
            if not self.project:
                self._log("No project open.")
                return
            self._stop_playmode()
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
            self._stop_playmode()
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

        def _toggle_playmode(self) -> None:
            if self.play_session:
                self._stop_playmode()
            else:
                self._start_playmode()

        def _start_playmode(self) -> None:
            if not self.project or not self.scene:
                self._log("No project open.")
                return
            self._save_scene()
            runtime_scene = Scene.from_dict(self.scene.to_dict())
            self.play_session = RuntimeSession(self.project, runtime_scene)
            self.play_session.profiler_recorder = self.profiler_recorder if self.profiler_recorder.enabled else None
            self.play_button.setText("Stop")
            self.hierarchy.setEnabled(False)
            self.inspector.setEnabled(False)
            self.view_tabs.setCurrentIndex(1)
            self.viewport.set_view_mode("Game")
            for error in self.play_session.start():
                self._log(f"Playmode script error: {error}")
            self._update_viewport_status()
            self.viewport.reload_assets()
            self._log("Playmode started.")

        def _stop_playmode(self) -> None:
            if not self.play_session:
                return
            self.play_session.input.set_cursor_mode("normal")
            self.play_session.stop()
            self.viewport.reset_runtime_cursor()
            self.play_session = None
            self.play_button.setText("Play")
            self.hierarchy.setEnabled(True)
            self.inspector.setEnabled(True)
            self._refresh_all()
            self._update_viewport_status()
            self._log("Playmode stopped.")

        def _open_project_settings(self) -> None:
            if not self.project:
                return
            def on_saved() -> None:
                if self.scene:
                    self.scene.render_settings = dict(self.project.render_settings)
                self.viewport.reload_assets()
                self.viewport.update()
                self._update_window_title()
                self._log("Project settings saved.")

            open_project_settings_dialog(self, self.project, self._scene_files(), on_saved)

        def _open_build_settings(self) -> None:
            if not self.project:
                return
            def on_saved() -> None:
                self._update_window_title()
                self._log("Build settings saved.")

            open_build_settings_dialog(self, self.project, on_saved, self._build_bundle, self._build_project)

        def _open_lighting_settings(self) -> None:
            if not self.scene:
                return

            def on_changed() -> None:
                self._mark_dirty("Edit Lighting Settings")
                self.viewport.reload_assets()
                self.viewport.update()

            open_lighting_settings_dialog(self, self.scene, on_changed)

        def _open_analysis(self) -> None:
            if self.analysis_window is None:
                self.analysis_window = AnalysisPanel(self.profiler_recorder, self)
                self.analysis_window.destroyed.connect(lambda *_args: setattr(self, "analysis_window", None))
            self.analysis_window.show()
            self.analysis_window.raise_()
            self.analysis_window.activateWindow()

        def _setup_vscode(self) -> None:
            if not self.project:
                self._log("No project open.")
                return
            try:
                setup_vscode_project(self.project)
                self._refresh_assets_from_watcher()
                self._log("VSCode setup refreshed.")
            except Exception as exc:
                QMessageBox.critical(self, "VSCode setup failed", str(exc))

        def _install_shortcuts(self) -> None:
            shortcuts = [
                ("Ctrl+S", self._save_scene),
                ("Ctrl+Z", self._undo_scene_edit),
                ("Ctrl+Y", self._redo_scene_edit),
                ("Delete", self._delete_selected),
                ("F", self._frame_selected),
                ("Ctrl+D", self._duplicate_selected),
                ("F2", self._rename_selected_dialog),
            ]
            for key, callback in shortcuts:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(callback)

        def _mark_dirty(self, label: str = "Edit Scene") -> None:
            if self._restoring_history:
                return
            if self.scene:
                self.undo.record(label, self.scene, self.selected.id if self.selected else None)
                self.dirty.dirty = self.undo.is_dirty
            else:
                self.dirty.mark_dirty()
            self._update_window_title()

        def _reset_undo_history(self) -> None:
            self.undo.reset(self.scene, self.selected.id if self.selected else None)
            self.dirty.mark_saved()

        def _begin_scene_edit(self, label: str) -> None:
            if self.scene:
                self.undo.begin(label, self.scene, self.selected.id if self.selected else None)

        def _commit_scene_edit(self) -> None:
            if self.scene:
                self.undo.commit(self.scene, self.selected.id if self.selected else None)
                self.dirty.dirty = self.undo.is_dirty
                self._populate_inspector()
                self._update_viewport_status()
                self._update_window_title()

        def _scene_changed_live(self) -> None:
            self.dirty.mark_dirty()
            self._populate_inspector()
            self._update_viewport_status()
            self._update_window_title()

        def _undo_scene_edit(self) -> None:
            self._restore_undo_state(self.undo.undo(), "Undo")

        def _redo_scene_edit(self) -> None:
            self._restore_undo_state(self.undo.redo(), "Redo")

        def _restore_undo_state(self, state: UndoState | None, action: str) -> None:
            if state is None or self.play_session:
                return
            self._restoring_history = True
            try:
                self.scene = Scene.from_dict(state.scene_data)
                self.selected = self.scene.find(state.selection_id) if state.selection_id else None
                self.selected_asset = None
                self.dirty.dirty = self.undo.is_dirty
                self._refresh_all()
                if self.selected:
                    self._select_hierarchy_item(self.selected.id)
                self._update_viewport_status()
                self._update_window_title()
                self._log(f"{action}: {state.label}")
            finally:
                self._restoring_history = False

        def _set_transform_tool(self, tool: str) -> None:
            self.current_transform_tool = tool if tool in {"move", "rotate", "scale"} else "move"
            for name, button in getattr(self, "tool_buttons", {}).items():
                button.setChecked(name == self.current_transform_tool)
            if hasattr(self.viewport, "set_transform_tool"):
                self.viewport.set_transform_tool(self.current_transform_tool)
            self._update_viewport_status()

        def _update_window_title(self) -> None:
            name = self.project.name if self.project else "No Project"
            scene_name = self.current_scene_path.name if self.current_scene_path else "No Scene"
            mark = "*" if self.dirty.dirty else ""
            self.setWindowTitle(f"P64 Editor - {name} - {scene_name}{mark}")

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
            if self.project:
                for metadata in ensure_audio_clips_for_assets(self.project):
                    self._log(f"Imported audio clip: {metadata.id}")
            self._populate_hierarchy()
            self._populate_assets()
            self._populate_inspector()
            self.viewport.update()

        def _refresh_assets_from_watcher(self) -> None:
            if self.project:
                for metadata in ensure_audio_clips_for_assets(self.project):
                    self._log(f"Imported audio clip: {metadata.id}")
            self._populate_assets()
            self.viewport.reload_assets()

        def _log(self, text: str) -> None:
            self.console.add_log(text)

    app = QApplication.instance() or QApplication([])
    project = Project.load(project_path) if project_path else None
    window = MainWindow(project)
    window.show()
    app.exec()


def _profiler_section(profiler: Any | None, name: str) -> Any:
    if profiler is None:
        return nullcontext()
    try:
        return profiler.section(name)
    except Exception:
        return nullcontext()
