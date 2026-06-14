import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from p64.editor.inspectors.components import project_texture_reference, texture_image_paths
from p64.engine.project import Project


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

    def test_obj_double_click_selects_model_asset_instead_of_importing(self):
        source = Path("src/p64/editor/panels/assets.py").read_text(encoding="utf-8")
        method = source[source.index("def _asset_double_clicked"):source.index("def _asset_selection_changed")]

        self.assertIn('if path.suffix.lower() == ".obj":', method)
        self.assertIn("self.selected_asset = path", method)
        self.assertNotIn("self._import_asset_obj(path)", method)

    def test_model_inspector_exposes_meshes_and_wireframe_preview(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")

        self.assertIn('"Model Meshes"', source)
        self.assertIn("_model_wireframe_preview", source)
        self.assertIn('form.addRow("Meshes"', source)

    def test_mesh_renderer_inspector_no_longer_exposes_submesh_row(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        mesh_editor = source[source.index("def _add_mesh_renderer_editor"):source.index("def _add_obj_asset_inspector")]

        self.assertNotIn('form.addRow("Submesh"', mesh_editor)
        self.assertNotIn("submesh_combo", mesh_editor)

    def test_audio_source_editor_hooks_are_available(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")

        self.assertIn('"AudioSource"', source)
        self.assertIn("def _add_audio_source_editor", source)
        self.assertIn("def _audio_clip_choices", source)
        self.assertIn("def _add_audio_asset_inspector", source)

    def test_audio_listener_editor_hooks_are_available(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        ops = Path("src/p64/editor/ops.py").read_text(encoding="utf-8")

        self.assertIn('"AudioListener"', source)
        self.assertIn("def _add_audio_listener_editor", source)
        self.assertIn("AudioListener: active=", source)
        self.assertIn("AudioListener()", ops)

    def test_wav_asset_browser_exposes_audio_refresh(self):
        source = Path("src/p64/editor/panels/assets.py").read_text(encoding="utf-8")

        self.assertIn('path.suffix.lower() == ".wav"', source)
        self.assertIn("Refresh Audio Import", source)

    def test_editor_refresh_auto_imports_audio_before_populating_assets(self):
        source = Path("src/p64/editor/main_window.py").read_text(encoding="utf-8")
        refresh_all = source[source.index("def _refresh_all"):source.index("def _refresh_assets_from_watcher")]
        refresh_assets = source[source.index("def _refresh_assets_from_watcher"):source.index("def _log")]

        self.assertIn("ensure_audio_clips_for_assets", refresh_all)
        self.assertLess(refresh_all.index("ensure_audio_clips_for_assets"), refresh_all.index("self._populate_assets()"))
        self.assertIn("ensure_audio_clips_for_assets", refresh_assets)
        self.assertLess(refresh_assets.index("ensure_audio_clips_for_assets"), refresh_assets.index("self._populate_assets()"))

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

    def test_editor_transform_toolbar_shortcuts_and_undo_are_wired(self):
        source = Path("src/p64/editor/main_window.py").read_text(encoding="utf-8")
        viewport = Path("src/p64/editor/viewport.py").read_text(encoding="utf-8")

        self.assertIn('QPushButton("Move"', source)
        self.assertIn('QPushButton("Rotate"', source)
        self.assertIn('QPushButton("Scale"', source)
        self.assertIn('"Ctrl+Z"', source)
        self.assertIn('"Ctrl+Y"', source)
        self.assertIn("UndoManager", source)
        shortcuts = source[source.index("def _install_shortcuts"):source.index("def _mark_dirty")]
        self.assertNotIn('("W"', shortcuts)
        self.assertNotIn('("E"', shortcuts)
        self.assertNotIn('("R"', shortcuts)
        self.assertIn("def _handle_tool_shortcut", viewport)
        self.assertIn("self.mouse_look", viewport)
        self.assertIn("Qt.Key_W", viewport)
        self.assertIn("Qt.Key_E", viewport)
        self.assertIn("Qt.Key_R", viewport)
        self.assertIn("hit_test_gizmo", viewport)
        self.assertIn("apply_gizmo_drag", viewport)

    def test_color_fields_use_color_picker_not_vec3_editor(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        fog_editor = source[source.index("def _add_fog_editor"):source.index("def _add_camera_editor")]
        light_editor = source[source.index("def _add_light_editor"):source.index("def _add_audio_source_editor")]
        material_fields = source[source.index("def _add_material_editor_fields"):source.index("def _add_material_slots_editor")]

        self.assertIn("QColorDialog.getColor", source)
        self.assertIn("def _color_editor", source)
        self.assertIn('form.addRow("Color", self._color_editor', fog_editor)
        self.assertIn('form.addRow("Color", self._color_editor', light_editor)
        self.assertIn('prop.kind == "color"', material_fields)
        self.assertNotIn("_vec3_editor(component.color)", fog_editor)
        self.assertNotIn("_vec3_editor(component.color)", light_editor)

    def test_empty_selection_inspector_exposes_scene_skybox_settings(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        scene_editor = source[source.index("def _add_scene_render_settings_editor"):source.index("def _add_entity_header")]

        self.assertIn('"Scene Render Settings"', scene_editor)
        self.assertIn('"Skybox Enabled"', scene_editor)
        self.assertIn('"Sky Top"', scene_editor)
        self.assertIn('"Sky Horizon"', scene_editor)
        self.assertIn('"Cloud Color"', scene_editor)
        self.assertIn('"Cloud Height"', scene_editor)
        self.assertIn('"Cloud Softness"', scene_editor)
        self.assertIn("self._color_editor", scene_editor)
        self.assertNotIn("_vec3_editor", scene_editor)
        self.assertIn('self._mark_dirty("Edit Scene Render Settings")', source)

    def test_texture_picker_finds_only_image_assets_and_packages(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            image = project.assets_dir / "albedo.png"
            image.write_bytes(b"not really a png")
            package_image = project.packages_dir / "Builtin" / "checker.bmp"
            package_image.parent.mkdir(parents=True)
            package_image.write_bytes(b"bmp")
            (project.assets_dir / "readme.txt").write_text("skip", encoding="utf-8")
            (project.assets_dir / "albedo.png.mdp64").write_text("skip", encoding="utf-8")

            paths = texture_image_paths(project)

            self.assertEqual([path.name for path in paths], ["albedo.png", "checker.bmp"])
            self.assertEqual(project_texture_reference(project, image), "assets/albedo.png")

    def test_shader_texture_properties_use_texture_picker(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        material_fields = source[source.index("def _add_material_editor_fields"):source.index("def _add_material_slots_editor")]

        self.assertIn('prop.kind == "texture"', material_fields)
        self.assertIn("self._texture_editor", material_fields)
        self.assertIn("def _choose_texture_reference", source)
        self.assertIn("QListView.ViewMode.IconMode", source)

    def test_asset_browser_uses_image_thumbnails(self):
        source = Path("src/p64/editor/panels/assets.py").read_text(encoding="utf-8")
        icon_method = source[source.index("def _icon_for_asset"):source.index("def _asset_folder_selection_changed")]

        self.assertIn("is_preview_image(path)", icon_method)
        self.assertIn("QPixmap", icon_method)
        self.assertIn("Qt.KeepAspectRatio", icon_method)

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
