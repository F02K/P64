from __future__ import annotations

from collections.abc import Callable
from typing import Any

from p64.editor.ops import delete_entity, duplicate_entity, find_parent
from p64.engine.assets import AssetMetadata
from p64.engine.components import MeshRenderer
from p64.engine.entity import Entity
from p64.engine.math import Vec3
from p64.engine.validation import asset_metadata_by_id, entity_reference_errors, has_reference_errors


def virtual_submesh_labels(entity: Entity, metadata_for_mesh: Callable[[str], AssetMetadata | None]) -> list[str]:
    child_names = {child.name for child in entity.children}
    labels: list[str] = []
    for component in entity.components:
        if not isinstance(component, MeshRenderer):
            continue
        metadata = metadata_for_mesh(component.mesh)
        if not metadata:
            continue
        for group in metadata.groups:
            if group not in child_names:
                labels.append(f"{group}  [submesh]")
    return labels


def create_hierarchy_mixin(
    QTreeWidgetItem: Any, QBrush: Any, QColor: Any, Qt: Any, QMenu: Any, QInputDialog: Any
) -> type:
    class HierarchyMixin:
        def _populate_hierarchy(self) -> None:
            self.hierarchy.clear()
            if not self.scene:
                return
            metadata = asset_metadata_by_id(self.project) if self.project else {}
            for entity in self.scene.entities:
                self.hierarchy.addTopLevelItem(self._entity_item(entity, metadata))
            self.hierarchy.expandAll()

        def _entity_item(self, entity: Entity, metadata: dict[str, AssetMetadata] | None = None) -> Any:
            item = QTreeWidgetItem([f"{entity.name}  [{entity.object_type_label}]"])
            item.setData(0, Qt.UserRole, entity.id)
            if self.project and has_reference_errors(self.project, entity, metadata):
                color = QColor(210, 70, 70) if entity_reference_errors(self.project, entity, metadata) else QColor(170, 110, 110)
                item.setForeground(0, QBrush(color))
            for child in entity.children:
                item.addChild(self._entity_item(child, metadata))
            for label in self._virtual_submesh_labels(entity):
                virtual = QTreeWidgetItem([label])
                virtual.setFlags(virtual.flags() & ~Qt.ItemIsSelectable)
                item.addChild(virtual)
            return item

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
                menu.addAction(f"Add {component_name}", lambda checked=False, name=component_name: self._add_component_by_name(name))
            menu.exec(self.hierarchy.viewport().mapToGlobal(pos))

        def _virtual_submesh_labels(self, entity: Entity) -> list[str]:
            return virtual_submesh_labels(entity, self._metadata_for_mesh)

    return HierarchyMixin
