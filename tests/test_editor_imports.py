import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from p64.editor.inspectors.components import font_asset_paths, project_texture_reference, texture_image_paths
from p64.editor.dialogs.lighting_settings import apply_lighting_settings
from p64.editor.panels.console import ConsoleLogModel, infer_console_level
from p64.engine.project import Project
from p64.engine.scene import Scene


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

    def test_font_picker_finds_font_assets_and_packages(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            font = project.assets_dir / "ui.ttf"
            font.write_bytes(b"not really a font")
            package_font = project.packages_dir / "Fonts" / "hud.otf"
            package_font.parent.mkdir(parents=True)
            package_font.write_bytes(b"font")
            (project.assets_dir / "note.txt").write_text("skip", encoding="utf-8")

            paths = font_asset_paths(project)

            self.assertEqual([path.name for path in paths], ["ui.ttf", "hud.otf"])

    def test_shader_texture_properties_use_texture_picker(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        material_fields = source[source.index("def _add_material_editor_fields"):source.index("def _add_material_slots_editor")]

        self.assertIn('prop.kind == "texture"', material_fields)
        self.assertIn("self._texture_editor", material_fields)
        self.assertIn("def _choose_texture_reference", source)
        self.assertIn("QListView.ViewMode.IconMode", source)

    def test_sprite_ui_particle_editors_use_asset_pickers_and_material_combo(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        sprite_editor = source[source.index("def _add_sprite_renderer_editor"):source.index("def _add_canvas_editor")]
        ui_editor = source[source.index("def _add_ui_image_editor"):source.index("def _add_ui_text_editor")]
        text_editor = source[source.index("def _add_ui_text_editor"):source.index("def _add_particle_emitter_editor")]
        particle_editor = source[source.index("def _add_particle_emitter_editor"):source.index("def _add_flipbook_rows")]
        reset = source[source.index("def _reset_component"):source.index("def _remove_component")]

        for editor in [sprite_editor, ui_editor, particle_editor]:
            self.assertIn("_component_texture_editor", editor)
            self.assertIn("_component_material_editor", editor)
            self.assertNotIn("material = QLineEdit", editor)
        self.assertIn('"Fill Mode"', ui_editor)
        self.assertIn('"Stretch"', ui_editor)
        self.assertIn('"Fit"', ui_editor)
        self.assertIn('"Fill"', ui_editor)
        self.assertIn('"fill_mode"', ui_editor)
        self.assertIn("_font_asset_editor", text_editor)
        self.assertIn('"Font Source"', text_editor)
        self.assertIn('"System Font"', text_editor)
        self.assertIn('"Font Asset"', text_editor)
        self.assertIn("font_family_editor.setEnabled", text_editor)
        self.assertIn("font_asset_editor.setEnabled", text_editor)
        self.assertIn("self.project.packages_dir", source[source.index("def _material_choices"):source.index("def _audio_clip_choices")])
        self.assertIn("SpriteRenderer(material=SPRITE_MATERIAL_RELATIVE)", reset)
        self.assertIn("UIImage(material=UI_IMAGE_MATERIAL_RELATIVE)", reset)
        self.assertIn("ParticleEmitter(material=PARTICLE_MATERIAL_RELATIVE)", reset)

    def test_canvas_editor_exposes_resolution_mode(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        canvas_editor = source[source.index("def _add_canvas_editor"):source.index("def _add_ui_image_editor")]

        self.assertIn('"Resolution Mode"', canvas_editor)
        self.assertIn('"auto"', canvas_editor)
        self.assertIn('"fixed"', canvas_editor)
        self.assertIn('"resolution_mode"', canvas_editor)

    def test_rect_transform_editor_explains_outer_ui_bounds(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        rect_editor = source[source.index("def _add_rect_transform_editor"):source.index("def _add_script_component_editor")]

        self.assertIn("outer UI bounds", rect_editor)
        self.assertIn('"Bounds"', rect_editor)

    def test_viewport_draws_selected_ui_bounds_overlay_in_game_view(self):
        source = Path("src/p64/editor/viewport.py").read_text(encoding="utf-8")

        self.assertIn("def _draw_ui_bounds_overlay", source)
        self.assertIn("ui_layout_debug", source)
        self.assertIn("_draw_rect_outline", source)
        self.assertIn("self._draw_ui_bounds_overlay(scene, selected)", source)

    def test_ui_text_font_family_uses_system_font_picker(self):
        source = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        text_editor = source[source.index("def _add_ui_text_editor"):source.index("def _add_particle_emitter_editor")]
        font_family_editor = source[source.index("def _font_family_editor"):source.index("def _texture_editor")]

        self.assertIn("self._font_family_editor(component)", text_editor)
        self.assertNotIn("font = QLineEdit(component.font_family", text_editor)
        self.assertIn("QFontDatabase.families()", font_family_editor)
        self.assertIn("combo.setEditable(True)", font_family_editor)
        self.assertIn('"System"', font_family_editor)

    def test_console_log_model_collapses_exact_duplicate_messages(self):
        model = ConsoleLogModel()

        model.add("UI batch render failed: boom", 1.0)
        model.add("UI batch render failed: boom", 2.0)

        rows = model.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].count, 2)
        self.assertEqual(rows[0].first_seen, 1.0)
        self.assertEqual(rows[0].last_seen, 2.0)
        self.assertEqual(rows[0].level, "Error")

    def test_console_log_model_uses_exact_message_for_duplicates(self):
        model = ConsoleLogModel()

        model.add("Missing texture: assets/a.png", 1.0)
        model.add("Missing texture: assets/b.png", 2.0)

        rows = model.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual([row.count for row in rows], [1, 1])
        self.assertEqual(infer_console_level("Missing texture"), "Warning")

    def test_editor_log_routes_to_console_panel(self):
        source = Path("src/p64/editor/main_window.py").read_text(encoding="utf-8")
        log_method = source[source.index("def _log"):source.index("app = QApplication")]

        self.assertIn("create_console_panel", source)
        self.assertIn("ConsolePanel", source)
        self.assertIn("self.console.add_log(text)", log_method)
        self.assertNotIn("appendPlainText", log_method)

    def test_window_menu_exposes_lighting_and_analysis(self):
        source = Path("src/p64/editor/main_window.py").read_text(encoding="utf-8")

        self.assertIn('addMenu("Window")', source)
        self.assertIn('"Lighting Settings"', source)
        self.assertIn('"Analysis"', source)
        self.assertIn("open_lighting_settings_dialog", source)
        self.assertIn("AnalysisPanel", source)
        self.assertIn("ProfilerRecorder", source)

    def test_analysis_panel_is_top_level_window(self):
        source = Path("src/p64/editor/panels/analysis.py").read_text(encoding="utf-8")

        self.assertIn("super().__init__(parent, Qt.Window)", source)
        self.assertIn("QTabWidget", source)
        self.assertIn('addTab(self.overview, "Overview")', source)
        self.assertIn('addTab(self.runtime, "Runtime")', source)
        self.assertIn('addTab(self.render, "Render")', source)
        self.assertIn('addTab(self.counts, "Counts")', source)
        main_window = Path("src/p64/editor/main_window.py").read_text(encoding="utf-8")
        self.assertIn("QTabWidget", main_window)

    def test_viewport_accepts_profiler_getter(self):
        source = Path("src/p64/editor/viewport.py").read_text(encoding="utf-8")

        self.assertIn("profiler_getter", source)
        self.assertIn("profiler.begin_frame", source)
        self.assertIn("renderer.profiler_recorder", source)

    def test_editor_uses_precise_timer_and_unthrottled_swap(self):
        main_window = Path("src/p64/editor/main_window.py").read_text(encoding="utf-8")
        runtime_window = Path("src/p64/editor/runtime_window.py").read_text(encoding="utf-8")

        self.assertIn("surface_format.setSwapInterval(0)", main_window)
        self.assertIn("self.repaint_timer.setTimerType(Qt.PreciseTimer)", main_window)
        self.assertIn("surface_format.setSwapInterval(0)", runtime_window)
        self.assertIn("timer.setTimerType(Qt.PreciseTimer)", runtime_window)

    def test_profiler_tick_and_paint_frames_are_separated(self):
        main_window = Path("src/p64/editor/main_window.py").read_text(encoding="utf-8")
        tick_method = main_window[main_window.index("def _tick_viewport"):main_window.index("def _viewport_scene")]
        viewport = Path("src/p64/editor/viewport.py").read_text(encoding="utf-8")
        paint_method = viewport[viewport.index("def paintGL"):viewport.index("def reload_assets")]

        self.assertIn('frame = profiler.begin_frame("Editor")', tick_method)
        self.assertIn("finally:", tick_method)
        self.assertIn("profiler.end_frame(frame)", tick_method)
        self.assertNotIn("end_current_frame", tick_method)
        self.assertIn("owns_frame", paint_method)
        self.assertIn("paint_profiler", paint_method)
        self.assertIn("if profiler.current_frame() is None:", paint_method)
        self.assertIn("self.renderer.profiler_recorder = paint_profiler", paint_method)
        self.assertIn('with _profiler_section(paint_profiler, "viewport paint")', paint_method)
        self.assertIn("if owns_frame and profiler is not None:", paint_method)

    def test_game_view_does_not_send_selection_outline_to_renderer(self):
        source = Path("src/p64/editor/viewport.py").read_text(encoding="utf-8")

        self.assertIn('selected_entity_id=selected.id if selected and self.view_mode != "Game" else None', source)
        self.assertIn("self._draw_ui_bounds_overlay(scene, selected)", source)

    def test_lighting_settings_helper_updates_scene_render_settings(self):
        scene = Scene("Lighting")

        settings = apply_lighting_settings(scene, {
            "skybox_enabled": False,
            "fog": False,
            "skybox_cloud_coverage": 2.0,
            "skybox_cloud_softness": -1.0,
        })

        self.assertFalse(settings["skybox_enabled"])
        self.assertFalse(settings["fog"])
        self.assertEqual(settings["skybox_cloud_coverage"], 1.0)
        self.assertEqual(settings["skybox_cloud_softness"], 0.0)
        self.assertIs(scene.render_settings, settings)

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

    def test_hierarchy_preserves_state_and_disables_entity_actions_without_selection(self):
        source = Path("src/p64/editor/panels/hierarchy.py").read_text(encoding="utf-8")
        populate = source[source.index("def _populate_hierarchy"):source.index("def _entity_item")]
        menu = source[source.index("def _show_hierarchy_menu"):source.index("def _virtual_submesh_labels")]

        self.assertIn("_expanded_hierarchy_ids", populate)
        self.assertIn("_restore_hierarchy_expanded_ids", populate)
        self.assertIn("blockSignals(True)", populate)
        self.assertIn("has_entity", menu)
        self.assertIn("setEnabled(has_entity)", menu)
        self.assertIn("virtual_submesh_labels", source)

    def test_hierarchy_and_inspector_expose_inherited_inactive_and_rect_transform(self):
        hierarchy = Path("src/p64/editor/panels/hierarchy.py").read_text(encoding="utf-8")
        inspector = Path("src/p64/editor/inspectors/components.py").read_text(encoding="utf-8")
        transform_editor = inspector[inspector.index("def _add_transform_editor"):inspector.index("def _add_script_component_editor")]

        self.assertIn("entity_effectively_active", hierarchy)
        self.assertIn("Inherited Inactive", hierarchy)
        self.assertIn("Rect Transform", transform_editor)
        self.assertIn("self.selected.rect_transform is not None", transform_editor)
