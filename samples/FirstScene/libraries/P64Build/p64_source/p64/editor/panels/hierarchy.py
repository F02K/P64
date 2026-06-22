from __future__ import annotations

from collections.abc import Callable
from typing import Any

from p64.editor.ops import delete_entity, duplicate_entity, find_parent
from p64.engine.assets import AssetMetadata, model_meshes
from p64.engine.components import Canvas, MeshRenderer, RectTransform
from p64.engine.entity import Entity, entity_effectively_active, entity_under_canvas
from p64.engine.math import Vec3
from p64.engine.transforms import world_position
from p64.engine.validation import asset_metadata_by_id, entity_reference_errors, has_reference_errors


def virtual_submesh_labels(entity: Entity, metadata_for_mesh: Callable[[str], AssetMetadata | None]) -> list[str]:
    labels: list[str] = []
    for component in entity.components:
        if not isinstance(component, MeshRenderer) or not component.mesh:
            continue
        metadata = metadata_for_mesh(component.mesh)
        if metadata is None:
            continue
        meshes = model_meshes(metadata)
        if meshes:
            for index, mesh in enumerate(meshes):
                mesh_id = str(mesh.get("id") or "")
                if component.mesh not in {metadata.id, mesh_id}:
                    continue
                if component.submesh and component.submesh not in {
                    str(mesh.get("name") or ""),
                    str(mesh.get("source_group") or ""),
                    str(mesh.get("node_path") or ""),
                    str(mesh.get("legacy_submesh") or ""),
                }:
                    continue
                name = str(mesh.get("name") or mesh.get("source_group") or mesh.get("node_path") or f"Mesh {index + 1}")
                labels.append(f"Mesh: {name}")
        elif component.submesh:
            labels.append(f"Mesh: {component.submesh}")
        else:
            labels.append(f"Mesh: {metadata.source or metadata.id}")
    return labels


def create_hierarchy_mixin(
    QTreeWidgetItem: Any, QBrush: Any, QColor: Any, Qt: Any, QMenu: Any, QInputDialog: Any
) -> type:
    class HierarchyMixin:
        def _populate_hierarchy(self) -> None:
            selected_id = self.selected.id if self.selected else None
            had_items = self.hierarchy.topLevelItemCount() > 0
            expanded_ids = self._expanded_hierarchy_ids()
            self.hierarchy.blockSignals(True)
            self.hierarchy.clear()
            try:
                if not self.scene:
                    return
                metadata = asset_metadata_by_id(self.project) if self.project else {}
                for entity in self.scene.entities:
                    self.hierarchy.addTopLevelItem(self._entity_item(entity, metadata))
                if had_items:
                    self._restore_hierarchy_expanded_ids(expanded_ids)
                else:
                    self.hierarchy.expandAll()
                if selected_id and self.scene.find(selected_id):
                    self._select_hierarchy_item(selected_id)
            finally:
                self.hierarchy.blockSignals(False)

        def _entity_item(self, entity: Entity, metadata: dict[str, AssetMetadata] | None = None) -> Any:
            tags = [entity.object_type_label]
            if not entity.active:
                tags.append("Inactive")
            elif not entity_effectively_active(entity):
                tags.append("Inherited Inactive")
            if entity.persistent:
                tags.append("Persistent")
            item = QTreeWidgetItem([f"{entity.name}  ({', '.join(tags)})"])
            item.setData(0, Qt.UserRole, entity.id)
            if self.project and has_reference_errors(self.project, entity, metadata):
                color = QColor(210, 70, 70) if entity_reference_errors(self.project, entity, metadata) else QColor(170, 110, 110)
                item.setForeground(0, QBrush(color))
            elif not entity.active:
                item.setForeground(0, QBrush(QColor(135, 135, 135)))
            elif not entity_effectively_active(entity):
                item.setForeground(0, QBrush(QColor(115, 115, 115)))
            elif entity.persistent:
                item.setForeground(0, QBrush(QColor(80, 120, 170)))
            for child in entity.children:
                item.addChild(self._entity_item(child, metadata))
            for label in self._virtual_submesh_labels(entity):
                virtual = QTreeWidgetItem([label])
                virtual.setFlags(virtual.flags() & ~Qt.ItemIsSelectable)
                virtual.setForeground(0, QBrush(QColor(115, 115, 115)))
                item.addChild(virtual)
            return item

        def _expanded_hierarchy_ids(self) -> set[str]:
            expanded: set[str] = set()

            def visit(item: Any) -> None:
                entity_id = item.data(0, Qt.UserRole)
                if entity_id and item.isExpanded():
                    expanded.add(str(entity_id))
                for index in range(item.childCount()):
                    visit(item.child(index))

            for index in range(self.hierarchy.topLevelItemCount()):
                visit(self.hierarchy.topLevelItem(index))
            return expanded

        def _restore_hierarchy_expanded_ids(self, expanded_ids: set[str]) -> None:
            def visit(item: Any) -> None:
                entity_id = item.data(0, Qt.UserRole)
                item.setExpanded(bool(entity_id and str(entity_id) in expanded_ids))
                for index in range(item.childCount()):
                    visit(item.child(index))

            for index in range(self.hierarchy.topLevelItemCount()):
                visit(self.hierarchy.topLevelItem(index))

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
            tool = getattr(self, "current_transform_tool", getattr(self.viewport, "transform_tool", "move")).title()
            if self.selected:
                selection = self.selected.name
            elif self.selected_asset:
                selection = f"Asset: {self.selected_asset.name}"
            else:
                selection = "No selection"
            self.viewport_status.setText(f"{mode} | Tool {tool} | Speed {speed:.1f} | {selection} | RMB+WASD/QE, Shift speed, F frame")

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
            if _entity_has_canvas(self.selected) or entity_under_canvas(self.selected):
                child.rect_transform = RectTransform()
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
            position = world_position(self.selected)
            self.viewport.scene_camera.position = Vec3(position.x, position.y + 2.0, position.z + 8.0)
            self.viewport.scene_camera.rotation = Vec3(-15.0, 0.0, 0.0)
            self.viewport.set_view_mode("Scene")
            self.view_tabs.setCurrentIndex(0)
            self._update_viewport_status()

        def _show_hierarchy_menu(self, pos: Any) -> None:
            item = self.hierarchy.itemAt(pos)
            if item is not None:
                entity_id = item.data(0, Qt.UserRole)
                if self.scene and entity_id and self.scene.find(entity_id):
                    self.hierarchy.setCurrentItem(item)
                else:
                    self.hierarchy.clearSelection()
                    self.selected = None
                    self._populate_inspector()
                    self._update_viewport_status()
            has_entity = bool(self.scene and self.selected and self.scene.find(self.selected.id))
            menu = QMenu(self)
            menu.addAction("Create Empty", self._create_entity)
            add_child = menu.addAction("Add Child", self._add_child_entity)
            add_child.setEnabled(has_entity)
            menu.addSeparator()
            duplicate = menu.addAction("Duplicate", self._duplicate_selected)
            duplicate.setEnabled(has_entity)
            delete = menu.addAction("Delete", self._delete_selected)
            delete.setEnabled(has_entity)
            rename = menu.addAction("Rename", self._rename_selected_dialog)
            rename.setEnabled(has_entity)
            frame = menu.addAction("Frame Selected", self._frame_selected)
            frame.setEnabled(has_entity)
            menu.addSeparator()
            for component_name in [
                "MeshRenderer",
                "Camera",
                "Light",
                "Fog",
                "SpawnPoint",
                "Collider",
                "CharacterController",
                "EntityPhysics",
                "ScriptComponent",
            ]:
                action = menu.addAction(f"Add {component_name}", lambda checked=False, name=component_name: self._add_component_by_name(name))
                action.setEnabled(has_entity)
            menu.exec(self.hierarchy.viewport().mapToGlobal(pos))

        def _virtual_submesh_labels(self, entity: Entity) -> list[str]:
            return virtual_submesh_labels(entity, self._metadata_for_mesh)

    return HierarchyMixin


def _entity_has_canvas(entity: Entity) -> bool:
    return any(isinstance(component, Canvas) for component in entity.components)
