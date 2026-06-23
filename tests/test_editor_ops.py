from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import unittest

from p64.build.pipeline import validate_project
from p64.editor.ops import (
    AssetOperationError,
    DirtyTracker,
    add_component,
    create_asset_folder,
    create_blank_asset_file,
    create_script_template,
    create_shader_template,
    delete_asset_path,
    delete_entity,
    duplicate_scene_asset,
    duplicate_entity,
    extract_materials_for_obj,
    insert_obj_scene_entity,
    move_component,
    move_script_entry,
    open_script_in_vscode_project,
    rename_asset_path,
    reset_material_asset,
    split_mesh_renderer_into_children,
    update_startup_scene_after_asset_rename,
    update_material_usage_cache,
)
from p64.editor.panels.assets import visible_asset_paths
from p64.editor.panels.hierarchy import virtual_submesh_labels
from p64.engine.assets import AssetMetadata
from p64.engine.builtin import PARTICLE_MATERIAL_RELATIVE, SPRITE_MATERIAL_RELATIVE, UI_IMAGE_MATERIAL_RELATIVE
from p64.engine.components import AudioListener, Camera, Canvas, Collider, EntityPhysics, MeshRenderer, ModelRenderer, ParticleEmitter, RectTransform, ScriptComponent, ScriptEntry, SpriteRenderer, UIImage, UIText
from p64.engine.files import find_metadata_for_source
from p64.engine.lighting import lighting_path_for_scene
from p64.engine.material import MaterialAsset, load_material_metadata
from p64.engine.entity import ENTITY, GAME_OBJECT, Entity
from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.engine.validation import entity_reference_errors


OBJ_TEXT = """
mtllib model.mtl
o Body
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
usemtl Mat
f 1/1/1 2/2/1 3/3/1
"""

MTL_TEXT = """
newmtl Mat
map_Kd missing.png
"""


class EditorOpsTests(unittest.TestCase):
    def test_duplicate_entity_regenerates_ids(self):
        root = Entity("Root")
        child = Entity("Child")
        root.add_child(child)
        duplicate = duplicate_entity(root)

        self.assertNotEqual(root.id, duplicate.id)
        self.assertNotEqual(root.children[0].id, duplicate.children[0].id)
        self.assertEqual(duplicate.name, "Root Copy")

    def test_delete_entity_removes_nested_child(self):
        parent = Entity("Parent")
        child = Entity("Child")
        parent.add_child(child)
        scene = Scene("Test", [parent])

        self.assertTrue(delete_entity(scene, child.id))
        self.assertEqual(parent.children, [])

    def test_dirty_tracker_marks_and_clears(self):
        tracker = DirtyTracker()
        tracker.mark_dirty()
        self.assertTrue(tracker.dirty)
        tracker.mark_saved()
        self.assertFalse(tracker.dirty)

    def test_create_shader_and_script_templates(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shader = create_shader_template(root / "assets")
            script = create_script_template(root / "scripts", "Mover")
            self.assertIn("Vertex", shader.read_text(encoding="utf-8"))
            script_text = script.read_text(encoding="utf-8")
            self.assertIn("from p64.engine.scripting import GameScript", script_text)
            self.assertIn("class Mover(GameScript)", script_text)
            self.assertIn("def on_update(self, dt: float) -> None", script_text)

    def test_open_script_in_vscode_project_opens_workspace_and_file(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            script = project.scripts_dir / "move.py"
            script.write_text("", encoding="utf-8")

            with (
                mock.patch("p64.editor.ops.shutil.which", return_value="code"),
                mock.patch("p64.editor.ops.subprocess.Popen") as popen,
            ):
                message = open_script_in_vscode_project(project, script)

            self.assertIsNone(message)
            command = popen.call_args.args[0]
            self.assertEqual(command[:2], ["code", "-r"])
            self.assertEqual(command[2], str(project.root.resolve()))
            self.assertEqual(command[3], "--goto")
            self.assertEqual(command[4], f"{script.resolve()}:1:1")

    def test_open_script_in_vscode_project_falls_back_to_project_folder(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            script = project.scripts_dir / "move.py"
            script.write_text("", encoding="utf-8")
            opened: list[Path] = []

            with mock.patch("p64.editor.ops.shutil.which", return_value=None):
                message = open_script_in_vscode_project(project, script, opened.append)

            self.assertIn("code", message or "")
            self.assertEqual(opened, [project.root])

    def test_asset_file_operations_create_unique_folder_and_file_under_assets(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            first_folder = create_asset_folder(project, project.assets_dir)
            second_folder = create_asset_folder(project, project.assets_dir)
            first_file = create_blank_asset_file(project, project.assets_dir)
            second_file = create_blank_asset_file(project, project.assets_dir)

            self.assertEqual(first_folder.name, "New Folder")
            self.assertEqual(second_folder.name, "New Folder_1")
            self.assertEqual(first_file.name, "new_file.txt")
            self.assertEqual(second_file.name, "new_file_1.txt")
            self.assertTrue(first_folder.is_dir())
            self.assertEqual(first_file.read_text(encoding="utf-8"), "")

    def test_asset_file_operations_refuse_packages_and_outside_project(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            outside = Path(tmp) / "outside"
            outside.mkdir()

            with self.assertRaises(AssetOperationError):
                create_blank_asset_file(project, project.packages_dir)
            with self.assertRaises(AssetOperationError):
                create_asset_folder(project, outside)

            package_file = project.packages_dir / "readonly.txt"
            package_file.write_text("keep", encoding="utf-8")
            with self.assertRaises(AssetOperationError):
                rename_asset_path(project, package_file, "renamed.txt")
            with self.assertRaises(AssetOperationError):
                delete_asset_path(project, package_file)

    def test_asset_rename_rejects_empty_separator_and_collision_names(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            source = create_blank_asset_file(project, project.assets_dir, "source.txt")
            create_blank_asset_file(project, project.assets_dir, "target.txt")

            for name in ["", "../escape.txt", "folder/name.txt", "target.txt"]:
                with self.subTest(name=name):
                    with self.assertRaises(AssetOperationError):
                        rename_asset_path(project, source, name)

    def test_asset_source_rename_moves_sidecar_and_updates_metadata_source(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            source = project.assets_dir / "model.obj"
            source.write_text(OBJ_TEXT, encoding="utf-8")
            metadata = AssetMetadata(id="mesh_model", kind="obj_mesh", source="assets/model.obj")
            metadata.save(project.assets_dir / "model.obj.mdp64")

            renamed = rename_asset_path(project, source, "renamed.obj")

            self.assertFalse(source.exists())
            self.assertFalse((project.assets_dir / "model.obj.mdp64").exists())
            self.assertTrue(renamed.exists())
            moved_metadata = project.assets_dir / "renamed.obj.mdp64"
            self.assertTrue(moved_metadata.exists())
            self.assertEqual(AssetMetadata.load(moved_metadata).source, "assets/renamed.obj")

    def test_asset_folder_rename_updates_nested_metadata_sources(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            folder = project.assets_dir / "Models"
            folder.mkdir()
            source = folder / "model.obj"
            source.write_text(OBJ_TEXT, encoding="utf-8")
            AssetMetadata(id="mesh_model", kind="obj_mesh", source="assets/Models/model.obj").save(folder / "model.obj.mdp64")

            renamed = rename_asset_path(project, folder, "Renamed")

            self.assertEqual(renamed.name, "Renamed")
            self.assertEqual(AssetMetadata.load(renamed / "model.obj.mdp64").source, "assets/Renamed/model.obj")

    def test_asset_delete_removes_source_sidecar(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            source = project.assets_dir / "model.obj"
            source.write_text(OBJ_TEXT, encoding="utf-8")
            sidecar = project.assets_dir / "model.obj.mdp64"
            AssetMetadata(id="mesh_model", kind="obj_mesh", source="assets/model.obj").save(sidecar)

            delete_asset_path(project, source)

            self.assertFalse(source.exists())
            self.assertFalse(sidecar.exists())

    def test_startup_scene_updates_after_asset_rename(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            old_scene = project.resolve_scene_path(project.startup_scene)
            new_scene = rename_asset_path(project, old_scene, "renamed.scenep64")

            changed = update_startup_scene_after_asset_rename(project, old_scene, new_scene)

            self.assertTrue(changed)
            self.assertEqual(project.startup_scene, "assets/scenes/renamed.scenep64")
            self.assertEqual(Project.load(project.root).startup_scene, "assets/scenes/renamed.scenep64")
            self.assertTrue(lighting_path_for_scene(new_scene).exists())
            self.assertFalse(lighting_path_for_scene(old_scene).exists())

    def test_scene_lighting_asset_lifecycle_is_strictly_coupled(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            scene = project.resolve_scene_path(project.startup_scene)
            lighting = lighting_path_for_scene(scene)

            duplicate, duplicate_lighting = duplicate_scene_asset(project, scene)
            self.assertTrue(duplicate.exists())
            self.assertTrue(duplicate_lighting.exists())
            with self.assertRaises(AssetOperationError):
                rename_asset_path(project, lighting, "renamed.lightingp64")
            with self.assertRaises(AssetOperationError):
                delete_asset_path(project, lighting)

            delete_asset_path(project, duplicate)
            self.assertFalse(duplicate.exists())
            self.assertFalse(duplicate_lighting.exists())

    def test_validation_reports_missing_or_invalid_lighting_asset(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            scene = project.resolve_scene_path(project.startup_scene)
            lighting = lighting_path_for_scene(scene)
            lighting.unlink()

            missing = validate_project(project.root)
            self.assertTrue(any("Lighting asset missing" in error for error in missing.errors))

            lighting.write_text("[]\n", encoding="utf-8")
            invalid = validate_project(project.root)
            self.assertTrue(any("Invalid lighting asset" in error for error in invalid.errors))

    def test_component_and_script_entry_ordering(self):
        entity = Entity("Root")
        mesh = entity.add_component(MeshRenderer())
        camera = entity.add_component(Camera())
        self.assertTrue(move_component(entity, camera, -1))
        self.assertIs(entity.components[0], camera)
        self.assertIs(entity.components[1], mesh)

        scripts = ScriptComponent(scripts=[
            ScriptEntry(script="a.py", class_name="A"),
            ScriptEntry(script="b.py", class_name="B"),
        ])
        self.assertTrue(move_script_entry(scripts, 1, -1))
        self.assertEqual([entry.script for entry in scripts.scripts], ["b.py", "a.py"])

    def test_add_collider_prefills_mesh_bounds_when_mesh_renderer_exists(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.obj"
            source.write_text(OBJ_TEXT, encoding="utf-8")
            (root / "model.mtl").write_text(MTL_TEXT, encoding="utf-8")
            project = Project.create(root / "Game")
            metadata = AssetMetadata(
                id="mesh_body",
                kind="obj_mesh",
                source="assets/model.obj",
                groups=["Body"],
                materials=["Mat"],
            )
            project.assets_dir.mkdir(parents=True, exist_ok=True)
            (project.assets_dir / "model.obj").write_text(OBJ_TEXT, encoding="utf-8")
            entity = Entity("Body")
            entity.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            metadata.save(project.assets_dir / "model.obj.mdp64")

            collider = add_component(entity, "Collider", project)

            self.assertIsInstance(collider, Collider)
            self.assertEqual(collider.shape, "box")
            self.assertEqual(collider.center.to_list(), [0.5, 0.5, 0.0])
            self.assertEqual(collider.size.to_list(), [1.0, 1.0, 0.001])
            self.assertEqual(collider.radius, 0.5)
            self.assertFalse(collider.fit_to_mesh)

    def test_add_collider_without_mesh_keeps_fallback_defaults(self):
        entity = Entity("Empty")

        collider = add_component(entity, "Collider")

        self.assertEqual(collider.center.to_list(), [0.0, 0.0, 0.0])
        self.assertEqual(collider.size.to_list(), [1.0, 1.0, 1.0])
        self.assertEqual(collider.radius, 0.5)

    def test_add_entity_physics_switches_game_object_to_entity(self):
        entity = Entity("Crate", object_type=GAME_OBJECT)

        physics = add_component(entity, "EntityPhysics")

        self.assertIsInstance(physics, EntityPhysics)
        self.assertTrue(entity.is_entity)

    def test_add_audio_listener_component(self):
        entity = Entity("Camera")

        listener = add_component(entity, "AudioListener")

        self.assertIsInstance(listener, AudioListener)
        self.assertTrue(listener.active)

    def test_add_sprite_ui_and_particle_components_use_builtin_materials(self):
        entity = Entity("Visuals")

        sprite = add_component(entity, "SpriteRenderer")
        image = add_component(entity, "UIImage")
        particle = add_component(entity, "ParticleEmitter")

        self.assertIsInstance(sprite, SpriteRenderer)
        self.assertEqual(sprite.material, SPRITE_MATERIAL_RELATIVE)
        self.assertIsInstance(image, UIImage)
        self.assertEqual(image.material, UI_IMAGE_MATERIAL_RELATIVE)
        self.assertIsInstance(particle, ParticleEmitter)
        self.assertEqual(particle.material, PARTICLE_MATERIAL_RELATIVE)

    def test_add_ui_components_under_canvas_creates_rect_transform(self):
        canvas = Entity("Canvas")
        canvas.add_component(Canvas())
        child = canvas.add_child(Entity("Label"))

        text = add_component(child, "UIText")

        self.assertIsInstance(text, UIText)
        self.assertIsInstance(child.rect_transform, RectTransform)
        self.assertEqual(child.rect_transform.size.to_list(), [240.0, 40.0, 0.0])

    def test_add_ui_image_under_canvas_creates_square_rect_transform(self):
        canvas = Entity("Canvas")
        canvas.add_component(Canvas())
        child = canvas.add_child(Entity("Image"))

        image = add_component(child, "UIImage")

        self.assertIsInstance(image, UIImage)
        self.assertIsInstance(child.rect_transform, RectTransform)
        self.assertEqual(child.rect_transform.size.to_list(), [128.0, 128.0, 0.0])

    def test_validation_reports_entity_physics_on_game_object(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            entity = Entity("Static Crate", object_type=GAME_OBJECT)
            entity.add_component(EntityPhysics())

            errors = entity_reference_errors(project, entity)

            self.assertIn("EntityPhysics requires an Entity", errors)

    def test_insert_obj_scene_entity_creates_submesh_children(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.obj"
            source.write_text(OBJ_TEXT, encoding="utf-8")
            (root / "model.mtl").write_text(MTL_TEXT, encoding="utf-8")
            project = Project.create(root / "Game")
            scene = project.load_startup_scene()

            entity = insert_obj_scene_entity(project, scene, source)
            self.assertTrue(entity.is_game_object)
            self.assertEqual(entity.children, [])
            renderer = entity.components[0]
            self.assertIsInstance(renderer, ModelRenderer)
            self.assertTrue(renderer.model)
            self.assertEqual(renderer.source_materials, ["Mat"])
            self.assertEqual(renderer.material_slots, [None])

    def test_split_mesh_renderer_into_children_creates_mesh_children(self):
        parent = Entity("Model", object_type=GAME_OBJECT)
        metadata = AssetMetadata(
            id="model_robot",
            kind="obj_mesh",
            source="assets/robot.obj",
            settings={
                "model": {
                    "meshes": [
                        {"id": "mesh_robot_body", "name": "Body", "legacy_submesh": "Body", "material_slots": ["Paint"]},
                        {"id": "mesh_robot_head", "name": "Head", "legacy_submesh": "Head", "material_slots": ["Paint"]},
                    ]
                },
                "material_assets": {"Paint": "assets/materials/Paint.material"},
            },
        )

        created = split_mesh_renderer_into_children(parent, metadata)

        self.assertEqual([child.name for child in created], ["Body", "Head"])
        self.assertEqual([child.object_type for child in created], [GAME_OBJECT, GAME_OBJECT])
        renderer = created[0].components[0]
        self.assertIsInstance(renderer, MeshRenderer)
        self.assertEqual(renderer.mesh, "mesh_robot_body")
        self.assertEqual(renderer.submesh, "Body")
        self.assertEqual(renderer.source_materials, ["Paint"])
        self.assertEqual(renderer.material_slots, ["assets/materials/Paint.material"])

    def test_dynamic_components_require_entity_and_editor_switches_type(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            static = Entity("Static", object_type=GAME_OBJECT)
            static.add_component(ScriptComponent(scripts=[ScriptEntry(script="move.py", class_name="Move")]))

            self.assertIn("ScriptComponent requires an Entity", entity_reference_errors(project, static))

            converted = Entity("Converted", object_type=GAME_OBJECT)
            add_component(converted, "ScriptComponent")
            self.assertEqual(converted.object_type, ENTITY)

    def test_add_model_renderer_does_not_change_scene_object_type(self):
        game_object = Entity("Static", object_type=GAME_OBJECT)
        entity = Entity("Dynamic", object_type=ENTITY)

        static_renderer = add_component(game_object, "ModelRenderer")
        dynamic_renderer = add_component(entity, "ModelRenderer")

        self.assertIsInstance(static_renderer, ModelRenderer)
        self.assertIsInstance(dynamic_renderer, ModelRenderer)
        self.assertEqual(game_object.object_type, GAME_OBJECT)
        self.assertEqual(entity.object_type, ENTITY)

    def test_virtual_submesh_labels_show_mesh_renderer_metadata(self):
        entity = Entity("Model")
        entity.add_component(MeshRenderer(mesh="model_robot"))
        metadata = AssetMetadata(
            id="model_robot",
            kind="obj_mesh",
            source="assets/robot.obj",
            settings={
                "model": {
                    "meshes": [
                        {"id": "mesh_robot_body", "name": "Body"},
                        {"id": "mesh_robot_head", "source_group": "Head"},
                    ]
                }
            },
        )

        labels = virtual_submesh_labels(entity, lambda mesh_id: metadata if mesh_id == metadata.id else None)

        self.assertEqual(labels, ["Mesh: Body", "Mesh: Head"])

    def test_asset_browser_hides_metadata_sidecars(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            visible = folder / "model.obj"
            hidden = folder / "model.obj.mdp64"
            visible.write_text("o Body\n", encoding="utf-8")
            hidden.write_text("{}", encoding="utf-8")

            paths = visible_asset_paths(folder)

            self.assertIn(visible, paths)
            self.assertNotIn(hidden, paths)

    def test_extract_materials_creates_material_and_sidecar_without_overwrite(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.obj"
            source.write_text(OBJ_TEXT, encoding="utf-8")
            (root / "model.mtl").write_text("newmtl Mat\nKd 0.25 0.5 0.75\nmap_Kd albedo.png\n", encoding="utf-8")
            project = Project.create(root / "Game")
            metadata = extract_materials_for_obj(project, source)
            material_path = metadata[0]
            material = MaterialAsset.load(material_path)
            material.properties["u_base_color"] = [0.9, 0.8, 0.7]
            material.save(material_path)

            imported_obj = project.assets_dir / root.name / "model.obj"
            second = extract_materials_for_obj(project, imported_obj)

            self.assertEqual(second, [material_path])
            self.assertEqual(MaterialAsset.load(material_path).properties["u_base_color"], [0.9, 0.8, 0.7])
            sidecar = load_material_metadata(material_path)
            self.assertEqual(sidecar.settings["defaults"]["diffuse_color"], [0.25, 0.5, 0.75])
            obj_metadata = AssetMetadata.load(find_metadata_for_source(imported_obj))
            self.assertEqual(obj_metadata.settings["material_assets"]["Mat"], "assets/materials/model/Mat.material")
            self.assertEqual(obj_metadata.settings["material_extract_folder"], "assets/materials/model")

    def test_extract_materials_uses_chosen_output_folder(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.obj"
            source.write_text(OBJ_TEXT, encoding="utf-8")
            (root / "model.mtl").write_text("newmtl Mat\nKd 0.2 0.3 0.4\n", encoding="utf-8")
            project = Project.create(root / "Game")
            output_dir = project.assets_dir / "Art" / "Materials"

            materials = extract_materials_for_obj(project, source, output_dir)

            self.assertEqual(materials[0], output_dir / "Mat.material")
            imported_obj = project.assets_dir / root.name / "model.obj"
            obj_metadata = AssetMetadata.load(find_metadata_for_source(imported_obj))
            self.assertEqual(obj_metadata.settings["material_assets"]["Mat"], "assets/Art/Materials/Mat.material")
            self.assertEqual(obj_metadata.settings["material_extract_folder"], "assets/Art/Materials")

    def test_external_material_reference_warns_without_validation_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.obj"
            source.write_text(OBJ_TEXT, encoding="utf-8")
            (root / "model.mtl").write_text("newmtl Mat\nKd 0.2 0.3 0.4\n", encoding="utf-8")
            project = Project.create(root / "Game")
            external_dir = root / "ExternalMaterials"
            material_path = extract_materials_for_obj(project, source, external_dir)[0]
            scene = project.load_startup_scene()
            entity = Entity("Body")
            entity.add_component(MeshRenderer(mesh="mesh_body", source_materials=["Mat"], material_slots=[str(material_path.resolve())]))
            scene.add_entity(entity)
            project.save_startup_scene(scene)

            report = validate_project(project.root)

            self.assertFalse(any("Missing material asset" in error for error in report.errors))
            self.assertTrue(any("external material" in warning for warning in report.warnings))

    def test_material_reset_uses_sidecar_defaults(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.obj"
            source.write_text(OBJ_TEXT, encoding="utf-8")
            (root / "model.mtl").write_text("newmtl Mat\nKd 0.1 0.2 0.3\n", encoding="utf-8")
            project = Project.create(root / "Game")
            material_path = extract_materials_for_obj(project, source)[0]
            material = MaterialAsset.load(material_path)
            material.properties["u_base_color"] = [1.0, 1.0, 1.0]
            material.save(material_path)

            reset_material_asset(project, material_path)

            self.assertEqual(MaterialAsset.load(material_path).properties["u_base_color"], [0.1, 0.2, 0.3])

    def test_material_usage_cache_is_computed_from_scene(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            material_path = project.assets_dir / "materials" / "Paint.material"
            material_path.parent.mkdir(parents=True)
            MaterialAsset().save(material_path)
            entity = Entity("Body")
            entity.add_component(MeshRenderer(mesh="mesh_body", submesh="Body", material_slots=["assets/materials/Paint.material"]))
            scene = Scene("Scene", [entity])

            update_material_usage_cache(project, scene, project.assets_dir / "scenes" / "main.scenep64")

            usage = load_material_metadata(material_path).settings["usage_cache"]
            self.assertEqual(usage[0]["entity"], "Body")
            self.assertEqual(usage[0]["submesh"], "Body")

    def test_validation_reports_missing_texture(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = AssetMetadata(
                id="mesh_missing_texture",
                kind="obj_mesh",
                source="assets/model.obj",
                groups=["Body"],
                materials=["Mat"],
                settings={"material_defs": {"Mat": {"diffuse_texture": "missing.png"}}},
            )
            (project.assets_dir / "model.obj").write_text("o Body\n", encoding="utf-8")
            metadata.save(project.assets_dir / "model.obj.mdp64")
            scene = project.load_startup_scene()
            entity = Entity("Body")
            entity.add_component(MeshRenderer(mesh=metadata.id, submesh="Body", material="Mat"))
            scene.add_entity(entity)
            project.save_startup_scene(scene)

            report = validate_project(project.root)
            self.assertFalse(any("missing texture" in error for error in report.errors))
            self.assertTrue(any("missing texture" in warning for warning in report.warnings))

    def test_validation_warns_for_missing_material_texture_without_blocking_build(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = AssetMetadata(
                id="model_luigi",
                kind="obj_mesh",
                source="assets/luigi.obj",
                groups=["Body"],
                materials=[],
                settings={"model": {"meshes": [{"id": "mesh_model_luigi_Body", "name": "Body"}]}},
            )
            (project.assets_dir / "luigi.obj").write_text("o Body\n", encoding="utf-8")
            metadata.save(project.assets_dir / "luigi.obj.mdp64")
            material_path = project.assets_dir / "materials" / "Broken.material"
            material_path.parent.mkdir(parents=True)
            MaterialAsset(textures={"u_texture": "missing.png"}).save(material_path)
            scene = project.load_startup_scene()
            entity = Entity("Luigi")
            entity.add_component(ModelRenderer(model="model_luigi", material_slots=["assets/materials/Broken.material"]))
            scene.add_entity(entity)
            project.save_startup_scene(scene)

            report = validate_project(project.root)

            self.assertFalse(any("missing material texture" in error for error in report.errors))
            self.assertTrue(any("missing material texture" in warning for warning in report.warnings))

    def test_reference_validation_reports_missing_mesh(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            entity = Entity("Broken")
            entity.add_component(MeshRenderer(mesh="missing_mesh"))

            errors = entity_reference_errors(project, entity)

            self.assertTrue(any("Missing mesh asset" in error for error in errors))

    def test_empty_script_entry_reports_validation_error_without_crashing(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            entity = Entity("Empty Script")
            entity.add_component(ScriptComponent(scripts=[ScriptEntry(script="", class_name="")]))

            errors = entity_reference_errors(project, entity)

            self.assertIn("Script entry has no script file", errors)


if __name__ == "__main__":
    unittest.main()
