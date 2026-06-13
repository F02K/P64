import unittest
from pathlib import Path


class EditorImportTests(unittest.TestCase):
    def test_editor_app_imports_without_starting_gui(self):
        import p64.editor.app as app

        self.assertTrue(callable(app.launch_editor))
        self.assertTrue(callable(app.launch_runtime_window))
        self.assertTrue(callable(app.component_summary))
        self.assertTrue(callable(app._normalize_vec3))
        self.assertTrue(callable(app._vec3_length))

    def test_runtime_window_uses_game_viewport(self):
        source = Path("src/p64/editor/runtime_window.py").read_text(encoding="utf-8")

        self.assertIn("create_viewport_class", source)
        self.assertIn("RuntimeSession", source)
        self.assertIn("QElapsedTimer", source)
        self.assertIn("QTimer", source)
        self.assertIn('viewport.set_view_mode("Game")', source)
        self.assertNotIn("Renderer: ModernGL " + "N" + "64-style path", source)

    def test_inspector_header_exposes_scene_object_type_dropdown(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")

        self.assertIn('QCheckBox("Persistent"', source)
        self.assertIn('object_type.addItems(["GameObject", "Entity"])', source)
        self.assertIn("_set_selected_object_type", source)
        self.assertIn("set_object_type_recursive", source)

    def test_mesh_collider_gizmo_uses_wireframe_not_bounds(self):
        source = Path("src/p64/renderer/scene_renderer.py").read_text(encoding="utf-8")

        self.assertIn('if component.shape == "mesh":', source)
        self.assertIn("_draw_mesh_collider", source)
        self.assertIn("mesh_triangles", source)
        self.assertNotIn('if component.shape == "mesh":\n                        self._draw_world_bounds', source)

    def test_selection_uses_mesh_outline_not_bounds(self):
        source = Path("src/p64/renderer/scene_renderer.py").read_text(encoding="utf-8")

        self.assertIn("selection_outline_program", source)
        self.assertIn("_draw_selection_outline(selected", source)
        self.assertIn("u_outline_width", source)
        self.assertIn("CULL_FACE", source)
        self.assertIn("depth_mask", source)
        self.assertNotIn("_draw_selection_bounds", source)

    def test_collider_shape_switch_only_prefills_box_and_sphere(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")

        self.assertIn('if value in {"box", "sphere"}', source)
        self.assertIn("apply_mesh_primitive_defaults", source)

    def test_collider_shape_specific_fields_are_created_inside_shape_branches(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        collider_editor = source[source.index("def _add_collider_editor"):source.index("def _add_character_controller_editor")]
        before_box_branch = collider_editor.split('if component.shape == "box":', 1)[0]

        self.assertNotIn("radius = QLineEdit", before_box_branch)
        self.assertNotIn("fit_to_mesh = QCheckBox", before_box_branch)

    def test_light_kind_specific_fields_are_created_inside_kind_branches(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        light_editor = source[source.index("def _add_light_editor"):source.index("def _add_spawn_point_editor")]
        before_point_branch = light_editor.split('if component.kind in {"point", "spot"}:', 1)[0]
        point_branch = light_editor.split('if component.kind in {"point", "spot"}:', 1)[1].split('if component.kind == "spot":', 1)[0]
        spot_branch = light_editor.split('if component.kind == "spot":', 1)[1]

        self.assertNotIn("range_edit = QLineEdit", before_point_branch)
        self.assertNotIn("falloff = QLineEdit", before_point_branch)
        self.assertNotIn("spot_angle = QLineEdit", before_point_branch)
        self.assertIn("range_edit = QLineEdit", point_branch)
        self.assertIn("falloff = QLineEdit", point_branch)
        self.assertIn("spot_angle = QLineEdit", spot_branch)

    def test_mesh_collider_shows_convex_without_primitive_fields(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        collider_editor = source[source.index("def _add_collider_editor"):source.index("def _add_character_controller_editor")]
        mesh_branch = collider_editor.split('elif component.shape == "mesh":', 1)[1].split('form.addRow("Trigger"', 1)[0]

        self.assertIn('form.addRow("Convex"', mesh_branch)
        self.assertIn("_set_collider_convex", source)
        self.assertNotIn('form.addRow("Center"', mesh_branch)
        self.assertNotIn('form.addRow("Size"', mesh_branch)
        self.assertNotIn('form.addRow("Radius"', mesh_branch)
        self.assertNotIn('form.addRow("Fit To Mesh"', mesh_branch)

    def test_entity_physics_inspector_fields_exist(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")

        self.assertIn('"EntityPhysics"', source)
        self.assertIn("def _add_entity_physics_editor", source)
        self.assertIn('form.addRow("Mass"', source)
        self.assertIn('form.addRow("Gravity"', source)
        self.assertIn('form.addRow("Velocity"', source)
        self.assertIn('form.addRow("Freeze Position"', source)

    def test_script_component_row_uses_action_menu(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        script_row = source[source.index("def _script_row"):source.index("def _add_component_controls")]

        self.assertIn("actions.setMenu", script_row)
        self.assertIn('"Move Up"', script_row)
        self.assertIn('"Move Down"', script_row)
        self.assertIn('"Reload Classes"', script_row)
        self.assertIn('"Remove"', script_row)
        self.assertNotIn('QPushButton("Up"', script_row)
        self.assertNotIn('QPushButton("Down"', script_row)
        self.assertNotIn('QPushButton("Reload"', script_row)
        self.assertNotIn('QPushButton("Remove"', script_row)

    def test_asset_browser_exposes_file_operations_for_editable_assets(self):
        source = Path("src/p64/editor/panels/assets.py").read_text(encoding="utf-8")

        self.assertIn('"New Folder"', source)
        self.assertIn('"New File"', source)
        self.assertIn('"Rename"', source)
        self.assertIn('"Delete"', source)
        self.assertIn("_begin_asset_rename", source)
        self.assertIn("self.assets.editItem", source)
        self.assertIn("create_asset_folder", source)
        self.assertIn("create_blank_asset_file", source)
        self.assertIn("delete_asset_path", source)

    def test_asset_browser_keeps_packages_read_only(self):
        source = Path("src/p64/editor/panels/assets.py").read_text(encoding="utf-8")
        main_window = Path("src/p64/editor/main_window.py").read_text(encoding="utf-8")

        self.assertIn("asset_path_is_editable", source)
        self.assertIn("_asset_folder_is_editable", source)
        self.assertIn("_asset_path_can_be_modified", source)
        self.assertIn("item.setFlags(item.flags() | Qt.ItemIsEditable)", source)
        self.assertIn("self.assets.itemChanged.connect(self._asset_item_changed)", main_window)
        self.assertIn("self.asset_folders.customContextMenuRequested.connect(self._show_asset_folder_menu)", main_window)

    def test_asset_browser_blocks_current_and_startup_scene_deletion(self):
        source = Path("src/p64/editor/panels/assets.py").read_text(encoding="utf-8")
        delete_method = source[source.index("def _delete_asset_path"):source.index("def _open_scene_asset")]

        self.assertIn("Open another scene before deleting the current scene", delete_method)
        self.assertIn("Choose another startup scene before deleting this scene", delete_method)
        self.assertIn("is_project_startup_scene", delete_method)
        self.assertIn("_asset_path_contains", delete_method)
