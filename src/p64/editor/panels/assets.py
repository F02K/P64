from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from p64.editor.ops import (
    AssetOperationError,
    asset_path_is_editable,
    create_asset_folder,
    create_blank_asset_file,
    create_script_template,
    create_shader_template,
    delete_asset_path,
    duplicate_scene_asset,
    import_audio_asset,
    insert_obj_scene_entity,
    is_project_startup_scene,
    open_script_in_vscode_project,
    rename_asset_path,
    update_startup_scene_after_asset_rename,
)
from p64.engine.assets import AssetMetadata
from p64.engine.files import is_lighting_file, is_metadata_file, is_scene_file
from p64.engine.lighting import scene_path_for_lighting
from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.engine.shader import parse_shader


def asset_roots(project: Project) -> list[tuple[str, Path]]:
    return [("Assets", project.assets_dir), ("Packages", project.packages_dir)]


def is_preview_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}


def visible_asset_paths(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [
        path
        for path in sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        if not (path.is_file() and is_metadata_file(path))
    ]


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
            self._updating_asset_grid = True
            self.assets.clear()
            folder = self.current_asset_folder
            try:
                if not folder or not folder.exists():
                    return
                for path in visible_asset_paths(folder):
                    item = QListWidgetItem(self._icon_for_asset(path), path.name)
                    item.setData(Qt.UserRole, str(path))
                    if self.project and asset_path_is_editable(self.project, path) and not is_lighting_file(path):
                        item.setFlags(item.flags() | Qt.ItemIsEditable)
                    self.assets.addItem(item)
            finally:
                self._updating_asset_grid = False

        def _icon_for_asset(self, path: Path) -> Any:
            if path.is_dir():
                icon_name = "folder_full.png" if any(path.iterdir()) else "folder_empty.png"
                icon_path = Path(__file__).parent / "resources" / "icons" / icon_name
                if icon_path.exists():
                    return QIcon(str(icon_path))
                return self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
            if is_preview_image(path):
                try:
                    from PySide6.QtGui import QPixmap

                    pixmap = QPixmap(str(path))
                    if not pixmap.isNull():
                        size = self.assets.iconSize()
                        width = max(size.width(), 64)
                        height = max(size.height(), 64)
                        return QIcon(pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.FastTransformation))
                except Exception:
                    pass
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
            can_create = self._asset_folder_is_editable(self.current_asset_folder)
            can_modify = path is not None and self._asset_path_can_be_modified(path)
            if can_create:
                menu.addAction("New Folder", self._create_asset_folder)
                menu.addAction("New File", self._create_blank_asset_file)
                menu.addSeparator()
            if path and path.suffix.lower() == ".obj":
                menu.addAction("Import OBJ into Scene", lambda: self._import_asset_obj(path))
            if path and path.suffix.lower() == ".wav":
                menu.addAction("Refresh Audio Import", lambda: self._refresh_audio_asset(path))
            if path and is_scene_file(path):
                menu.addAction("Open Scene", lambda: self._open_scene_asset(path))
                menu.addAction("Set As Startup Scene", lambda: self._set_startup_scene(path))
                menu.addAction("Duplicate Scene", lambda: self._duplicate_scene_asset(path))
            if path and is_lighting_file(path):
                menu.addAction("Open Lighting Settings", lambda: self._open_lighting_asset(path))
            menu.addAction("Create Scene", self._create_scene_asset)
            menu.addAction("Create Shader", self._create_shader_asset)
            menu.addAction("Create Script", self._create_script_asset)
            menu.addAction("Refresh Assets", self._refresh_assets_from_watcher)
            if path:
                menu.addSeparator()
                menu.addAction("Open", lambda: self._open_path(path))
                menu.addAction("Reveal in Explorer", lambda: self._reveal_path(path))
                if can_modify:
                    menu.addSeparator()
                    menu.addAction("Rename", lambda: self._begin_asset_rename(path))
                    menu.addAction("Delete", lambda: self._delete_asset_path(path))
            menu.exec(self.assets.viewport().mapToGlobal(pos))

        def _show_asset_folder_menu(self, pos: Any) -> None:
            item = self.asset_folders.itemAt(pos)
            folder = Path(item.data(0, Qt.UserRole)) if item and item.data(0, Qt.UserRole) else self.current_asset_folder
            menu = QMenu(self)
            if self._asset_folder_is_editable(folder):
                menu.addAction("New Folder", lambda: self._create_asset_folder(folder))
                menu.addAction("New File", lambda: self._create_blank_asset_file(folder))
            menu.addAction("Refresh Assets", self._refresh_assets_from_watcher)
            if folder:
                menu.addSeparator()
                menu.addAction("Reveal in Explorer", lambda: self._reveal_path(folder))
            menu.exec(self.asset_folders.viewport().mapToGlobal(pos))

        def _asset_double_clicked(self, item: Any) -> None:
            path = Path(item.data(Qt.UserRole))
            if path.is_dir():
                self.current_asset_folder = path
                self._select_asset_folder_item(path)
                self._populate_asset_grid()
                return
            if path.suffix.lower() == ".obj":
                self.selected_asset = path
                self.selected = None
                self._populate_inspector()
                self._update_viewport_status()
            elif is_scene_file(path):
                self._open_scene_asset(path)
            elif is_lighting_file(path):
                self._open_lighting_asset(path)
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

        def _asset_item_changed(self, item: Any) -> None:
            if getattr(self, "_updating_asset_grid", False):
                return
            data = item.data(Qt.UserRole)
            if not data:
                return
            path = Path(data)
            if item.text() == path.name:
                return
            try:
                new_path = self._rename_asset_path(path, item.text())
                item.setData(Qt.UserRole, str(new_path))
            except Exception as exc:
                self._log(f"Rename failed: {exc}")
                self._updating_asset_grid = True
                try:
                    item.setText(path.name)
                finally:
                    self._updating_asset_grid = False

        def _asset_folder_is_editable(self, path: Path | None) -> bool:
            return bool(self.project and path and path.exists() and path.is_dir() and asset_path_is_editable(self.project, path))

        def _asset_path_can_be_modified(self, path: Path) -> bool:
            if not self.project or not asset_path_is_editable(self.project, path):
                return False
            if is_lighting_file(path):
                return False
            return path.resolve() != self.project.assets_dir.resolve()

        def _create_asset_folder(self, folder: Path | None = None) -> None:
            if not self.project:
                return
            try:
                path = create_asset_folder(self.project, folder or self.current_asset_folder or self.project.assets_dir)
                self.current_asset_folder = path.parent
                self._refresh_assets_from_watcher()
                self._select_asset_path(path)
                self._begin_asset_rename(path)
                self._log(f"Created folder: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Create folder failed", str(exc))

        def _create_blank_asset_file(self, folder: Path | None = None) -> None:
            if not self.project:
                return
            try:
                path = create_blank_asset_file(self.project, folder or self.current_asset_folder or self.project.assets_dir)
                self.current_asset_folder = path.parent
                self._refresh_assets_from_watcher()
                self._select_asset_path(path)
                self._begin_asset_rename(path)
                self._log(f"Created file: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Create file failed", str(exc))

        def _begin_asset_rename(self, path: Path) -> None:
            item = self._select_asset_path(path)
            if item is not None:
                self.assets.editItem(item)

        def _select_asset_path(self, path: Path) -> Any | None:
            self._select_asset_folder_item(path.parent)
            if self.current_asset_folder != path.parent:
                self.current_asset_folder = path.parent
                self._populate_asset_grid()
            for index in range(self.assets.count()):
                item = self.assets.item(index)
                if item.data(Qt.UserRole) == str(path):
                    self.assets.setCurrentItem(item)
                    return item
            return None

        def _rename_asset_path(self, path: Path, new_name: str) -> Path:
            if not self.project:
                raise AssetOperationError("No project open.")
            old_current_scene = self.current_scene_path
            new_path = rename_asset_path(self.project, path, new_name)
            if old_current_scene and old_current_scene.resolve() == path.resolve():
                self.current_scene_path = new_path
                self._update_window_title()
            if update_startup_scene_after_asset_rename(self.project, path, new_path):
                self._log(f"Startup scene updated: {self.project.startup_scene}")
            self.selected_asset = new_path
            self.current_asset_folder = new_path.parent
            self._refresh_assets_from_watcher()
            self._select_asset_path(new_path)
            self._populate_inspector()
            self._log(f"Renamed asset: {path.name} -> {new_path.name}")
            return new_path

        def _delete_asset_path(self, path: Path) -> None:
            if not self.project:
                return
            if self.current_scene_path and self._asset_path_contains(path, self.current_scene_path):
                QMessageBox.warning(self, "Delete blocked", "Open another scene before deleting the current scene.")
                return
            startup_scene = self.project.resolve_scene_path(self.project.startup_scene)
            if is_project_startup_scene(self.project, path) or self._asset_path_contains(path, startup_scene):
                QMessageBox.warning(self, "Delete blocked", "Choose another startup scene before deleting this scene.")
                return
            result = QMessageBox.question(
                self,
                "Delete Asset",
                f"Delete {path.name}?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return
            try:
                delete_asset_path(self.project, path)
                if self.selected_asset and self.selected_asset.resolve() == path.resolve():
                    self.selected_asset = None
                if self.current_asset_folder and not self.current_asset_folder.exists():
                    self.current_asset_folder = self.project.assets_dir
                self._refresh_assets_from_watcher()
                self._populate_inspector()
                self._log(f"Deleted asset: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Delete failed", str(exc))

        def _asset_path_contains(self, path: Path, child: Path) -> bool:
            resolved = path.resolve()
            candidate = child.resolve()
            if resolved == candidate:
                return True
            if not path.is_dir():
                return False
            try:
                candidate.relative_to(resolved)
                return True
            except ValueError:
                return False

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

        def _duplicate_scene_asset(self, path: Path) -> None:
            if not self.project:
                return
            try:
                duplicated, _lighting = duplicate_scene_asset(self.project, path)
                self.current_asset_folder = duplicated.parent
                self._refresh_assets_from_watcher()
                self._select_asset_path(duplicated)
                self._log(f"Duplicated scene: {duplicated}")
            except Exception as exc:
                QMessageBox.critical(self, "Duplicate scene failed", str(exc))

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

        def _open_lighting_asset(self, path: Path) -> None:
            scene_path = scene_path_for_lighting(path)
            if not scene_path.exists():
                QMessageBox.warning(self, "Lighting asset", f"Coupled Scene is missing: {scene_path.name}")
                return
            if not self.current_scene_path or self.current_scene_path.resolve() != scene_path.resolve():
                self._open_scene_asset(scene_path)
                if not self.current_scene_path or self.current_scene_path.resolve() != scene_path.resolve():
                    return
            self._open_lighting_settings()

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
            if is_lighting_file(path):
                self._open_lighting_asset(path)
                return
            if self.project and path.suffix.lower() == ".py":
                message = open_script_in_vscode_project(
                    self.project,
                    path,
                    lambda folder: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve()))),
                )
                if message:
                    self._log(message)
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

        def _reveal_path(self, path: Path) -> None:
            try:
                os.startfile(str(path.parent.resolve()))
            except Exception as exc:
                self._log(f"Reveal failed: {exc}")

    return AssetBrowserMixin
