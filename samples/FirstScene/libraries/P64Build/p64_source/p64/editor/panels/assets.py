from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from p64.editor.ops import create_script_template, create_shader_template, insert_obj_scene_entity
from p64.engine.assets import AssetMetadata
from p64.engine.files import is_metadata_file, is_scene_file
from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.engine.shader import parse_shader


def asset_roots(project: Project) -> list[tuple[str, Path]]:
    return [("Assets", project.assets_dir), ("Packages", project.packages_dir)]


def is_preview_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}


def create_asset_browser_mixin(
    QTreeWidgetItem: Any,
    QListWidgetItem: Any,
    QIcon: Any,
    QStyle: Any,
    Qt: Any,
    QMenu: Any,
    QInputDialog: Any,
    QMessageBox: Any,
    QDesktopServices: Any,
    QUrl: Any,
) -> type:
    class AssetBrowserMixin:
        def _populate_assets(self) -> None:
            self.asset_folders.clear()
            self.assets.clear()
            self._reset_asset_watcher()
            if not self.project:
                return
            first_item = None
            for label, path in asset_roots(self.project):
                if not path.exists():
                    continue
                root_item = QTreeWidgetItem([label])
                root_item.setData(0, Qt.UserRole, str(path))
                self.asset_folders.addTopLevelItem(root_item)
                self._add_asset_folder_children(root_item, path)
                first_item = first_item or root_item
            self.asset_folders.expandAll()
            if self.current_asset_folder is None or not self.current_asset_folder.exists():
                self.current_asset_folder = self.project.assets_dir
            if first_item and not self.asset_folders.selectedItems():
                self._select_asset_folder_item(self.current_asset_folder)
            self._populate_asset_grid()

        def _add_asset_folder_children(self, item: Any, folder: Path) -> None:
            for path in sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if not path.is_dir():
                    continue
                child = QTreeWidgetItem([path.name])
                child.setData(0, Qt.UserRole, str(path))
                item.addChild(child)
                self._add_asset_folder_children(child, path)

        def _populate_asset_grid(self) -> None:
            self.assets.clear()
            folder = self.current_asset_folder
            if not folder or not folder.exists():
                return
            for path in sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                item = QListWidgetItem(self._icon_for_asset(path), path.name)
                item.setData(Qt.UserRole, str(path))
                self.assets.addItem(item)

        def _icon_for_asset(self, path: Path) -> Any:
            if path.is_dir():
                icon_name = "folder_full.png" if any(path.iterdir()) else "folder_empty.png"
                icon_path = Path(__file__).parent / "resources" / "icons" / icon_name
                if icon_path.exists():
                    return QIcon(str(icon_path))
                return self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
            return self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)

        def _asset_folder_selection_changed(self) -> None:
            items = self.asset_folders.selectedItems()
            if not items:
                return
            data = items[0].data(0, Qt.UserRole)
            if data:
                self.current_asset_folder = Path(data)
                self._populate_asset_grid()

        def _select_asset_folder_item(self, folder: Path) -> None:
            def visit(item: Any) -> bool:
                if item.data(0, Qt.UserRole) == str(folder):
                    self.asset_folders.setCurrentItem(item)
                    return True
                for index in range(item.childCount()):
                    if visit(item.child(index)):
                        return True
                return False

            for index in range(self.asset_folders.topLevelItemCount()):
                if visit(self.asset_folders.topLevelItem(index)):
                    return

        def _reset_asset_watcher(self) -> None:
            if self.asset_watcher.directories():
                self.asset_watcher.removePaths(self.asset_watcher.directories())
            if self.asset_watcher.files():
                self.asset_watcher.removePaths(self.asset_watcher.files())
            if not self.project:
                return
            watched_roots = [self.project.assets_dir, self.project.packages_dir]
            paths = []
            for root in watched_roots:
                if root.exists():
                    paths.extend(str(path) for path in root.rglob("*"))
                    paths.append(str(root))
            if paths:
                self.asset_watcher.addPaths(paths)

        def _show_asset_menu(self, pos: Any) -> None:
            path = self._selected_asset_path()
            menu = QMenu(self)
            if path and (path.suffix.lower() == ".obj" or is_metadata_file(path)):
                menu.addAction("Import OBJ into Scene", lambda: self._import_asset_obj(path))
            if path and is_scene_file(path):
                menu.addAction("Open Scene", lambda: self._open_scene_asset(path))
                menu.addAction("Set As Startup Scene", lambda: self._set_startup_scene(path))
            menu.addAction("Create Scene", self._create_scene_asset)
            menu.addAction("Create Shader", self._create_shader_asset)
            menu.addAction("Create Script", self._create_script_asset)
            menu.addAction("Refresh Assets", self._refresh_assets_from_watcher)
            if path:
                menu.addSeparator()
                menu.addAction("Open", lambda: self._open_path(path))
                menu.addAction("Reveal in Explorer", lambda: self._reveal_path(path))
            menu.exec(self.assets.viewport().mapToGlobal(pos))

        def _asset_double_clicked(self, item: Any) -> None:
            path = Path(item.data(Qt.UserRole))
            if path.is_dir():
                self.current_asset_folder = path
                self._select_asset_folder_item(path)
                self._populate_asset_grid()
                return
            if path.suffix.lower() == ".obj" or is_metadata_file(path):
                self._import_asset_obj(path)
            elif is_scene_file(path):
                self._open_scene_asset(path)
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
            data = items[0].data(Qt.UserRole)
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

        def _create_scene_asset(self) -> None:
            if not self.project:
                return
            name, ok = QInputDialog.getText(self, "Create Scene", "Scene name:", text="new_scene")
            if not ok or not name.strip():
                return
            scene_dir = self.project.scenes_dir
            scene_dir.mkdir(parents=True, exist_ok=True)
            path = scene_dir / f"{name.strip()}.scenep64"
            index = 1
            while path.exists():
                path = scene_dir / f"{name.strip()}_{index}.scenep64"
                index += 1
            Scene(name.strip()).save(path)
            self.current_asset_folder = scene_dir
            self._refresh_assets_from_watcher()
            self._log(f"Created scene: {path}")

        def _open_scene_asset(self, path: Path) -> None:
            if not self.project:
                return
            if not self._confirm_discard_changes():
                return
            try:
                self.scene = Scene.load(path)
                self.current_scene_path = path
                self.selected = None
                self.selected_asset = None
                self.dirty.mark_saved()
                self._refresh_all()
                self._update_window_title()
                self._log(f"Opened scene: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Open scene failed", str(exc))

        def _set_startup_scene(self, path: Path) -> None:
            if not self.project:
                return
            try:
                self.project.startup_scene = path.resolve().relative_to(self.project.root.resolve()).as_posix()
                self.project.save()
                self._log(f"Startup scene set: {self.project.startup_scene}")
            except Exception as exc:
                QMessageBox.critical(self, "Startup scene failed", str(exc))

        def _open_path(self, path: Path) -> None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

        def _reveal_path(self, path: Path) -> None:
            try:
                os.startfile(str(path.parent.resolve()))
            except Exception as exc:
                self._log(f"Reveal failed: {exc}")

    return AssetBrowserMixin
