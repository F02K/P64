from tempfile import TemporaryDirectory
from pathlib import Path
from math import cos, sin
from unittest import mock
import unittest

import p64.engine.collision as collision
import p64.engine.mesh_geometry as mesh_geometry
from p64.engine.assets import AssetMetadata
from p64.engine.collision import CollisionWorld, apply_mesh_primitive_defaults, collider_bounds, collider_sphere
from p64.engine.components import CharacterController, Collider, EntityPhysics, MeshRenderer, ScriptComponent, ScriptEntry, component_from_dict
from p64.engine.entity import GAME_OBJECT, Entity
from p64.engine.files import find_metadata_for_source
from p64.engine.mesh_geometry import clear_mesh_geometry_cache, convex_hull, mesh_triangles
from p64.engine.obj import import_obj_to_project
from p64.engine.math import Vec3
from p64.engine.project import Project
from p64.engine.runtime_session import RuntimeSession
from p64.engine.scene import Scene


class CollisionTests(unittest.TestCase):
    def test_collider_serializes_and_world_bounds_include_transform(self):
        entity = Entity("Box")
        entity.transform.position = Vec3(2, 3, 4)
        collider = Collider(size=Vec3(4, 2, 6), center=Vec3(1, 0, -1))
        entity.add_component(collider)

        loaded = Entity.from_dict(entity.to_dict())
        bounds = collider_bounds(loaded, loaded.components[0])

        self.assertEqual(bounds.min.to_list(), [1.0, 2.0, 0.0])
        self.assertEqual(bounds.max.to_list(), [5.0, 4.0, 6.0])

    def test_collision_world_reports_overlaps_and_ignores_triggers_for_blocking(self):
        actor = Entity("Actor")
        actor.add_component(Collider(size=Vec3(1, 1, 1)))
        blocker = Entity("Blocker")
        blocker.transform.position = Vec3(0.5, 0, 0)
        blocker.add_component(Collider(size=Vec3(1, 1, 1)))
        trigger = Entity("Trigger")
        trigger.transform.position = Vec3(0.5, 0, 0)
        trigger.add_component(Collider(size=Vec3(1, 1, 1), is_trigger=True))
        scene = Scene("Test", [actor, blocker, trigger])

        world = CollisionWorld(scene)
        all_hits = world.overlaps(actor, actor.components[0], include_triggers=True)
        blocking_hits = world.overlaps(actor, actor.components[0], include_triggers=False)

        self.assertEqual({hit.entity.name for hit in all_hits}, {"Blocker", "Trigger"})
        self.assertEqual([hit.entity.name for hit in blocking_hits], ["Blocker"])

    def test_character_controller_blocks_wall_and_lands_on_floor(self):
        player = Entity("Player")
        controller = CharacterController(height=1.8, radius=0.35, gravity=9.0)
        player.add_component(controller)
        floor = Entity("Floor")
        floor.transform.position = Vec3(0, -0.5, 0)
        floor.add_component(Collider(size=Vec3(8, 1, 8)))
        wall = Entity("Wall")
        wall.transform.position = Vec3(1, 0.5, 0)
        wall.add_component(Collider(size=Vec3(1, 2, 2)))
        scene = Scene("Test", [player, floor, wall])

        world = CollisionWorld(scene)
        moved = world.move_character(player, controller, Vec3(1, 0, 0), 0.0)
        world.move_character(player, controller, Vec3(0, 0, 0), 0.1)

        self.assertEqual(moved.x, 0.0)
        self.assertEqual(player.transform.position.x, 0.0)
        self.assertTrue(controller.grounded)
        self.assertEqual(controller.velocity.y, 0.0)

    def test_script_can_move_character_controller(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "move_player.py").write_text(
                "from p64.engine.math import Vec3\n"
                "from p64.engine.scripting import GameScript\n"
                "class MovePlayer(GameScript):\n"
                "    def on_update(self, dt):\n"
                "        self.character_controller.move(Vec3(0.25, 0.0, 0.0), dt)\n",
                encoding="utf-8",
            )
            scene = project.load_startup_scene()
            player = Entity("Player")
            player.add_component(CharacterController(gravity=0.0))
            player.add_component(ScriptComponent(scripts=[ScriptEntry(script="move_player.py", class_name="MovePlayer")]))
            scene.add_entity(player)

            errors = scene.run_scripts_once(project.root, dt=1.0)

            self.assertEqual(errors, [])
            self.assertEqual(player.transform.position.x, 0.25)

    def test_runtime_binds_character_controller_to_reused_collision_world(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            (project.scripts_dir / "idle.py").write_text(
                "from p64.engine.scripting import GameScript\n"
                "class Idle(GameScript):\n"
                "    pass\n",
                encoding="utf-8",
            )
            player = Entity("Player")
            controller = CharacterController(gravity=0.0)
            player.add_component(controller)
            player.add_component(ScriptComponent(scripts=[ScriptEntry(script="idle.py", class_name="Idle")]))
            scene = Scene("Test", [player])

            session = RuntimeSession(project, scene)
            errors = session.start()

            self.assertEqual(errors, [])
            self.assertIs(controller._runtime_collision_world, session.collision_world)

    def test_mesh_fit_box_and_sphere_use_mesh_bounds(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_test_mesh(project)
            static = Entity("Rock", object_type=GAME_OBJECT)
            static.transform.position = Vec3(1, 2, 3)
            static.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))

            box = Collider(shape="box", fit_to_mesh=True)
            sphere = Collider(shape="sphere", fit_to_mesh=True)

            bounds = collider_bounds(static, box, project)
            center, radius = collider_sphere(static, sphere, project)

            self.assertEqual(bounds.min.to_list(), [1.0, 2.0, 3.0])
            self.assertEqual(bounds.max.to_list(), [3.0, 4.0, 3.0])
            self.assertEqual(center.to_list(), [2.0, 3.0, 3.0])
            self.assertEqual(radius, 1.0)

    def test_mesh_primitive_defaults_apply_on_shape_switch(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_test_mesh(project)
            static = Entity("Rock", object_type=GAME_OBJECT)
            static.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            collider = Collider(shape="mesh", center=Vec3(9, 9, 9), size=Vec3(9, 9, 9), radius=9)

            self.assertTrue(apply_mesh_primitive_defaults(project, static, collider, "sphere"))
            self.assertEqual(collider.shape, "sphere")
            self.assertEqual(collider.center.to_list(), [1.0, 1.0, 0.0])
            self.assertEqual(collider.radius, 1.0)

            self.assertTrue(apply_mesh_primitive_defaults(project, static, collider, "box"))
            self.assertEqual(collider.shape, "box")
            self.assertEqual(collider.size.to_list(), [2.0, 2.0, 0.001])

    def test_mesh_collider_serializes_convex_and_validates_static_only_when_non_convex(self):
        collider = Collider(shape="mesh", fit_to_mesh=True, convex=True)
        loaded = component_from_dict(collider.to_dict())
        self.assertEqual(loaded.shape, "mesh")
        self.assertTrue(loaded.fit_to_mesh)
        self.assertTrue(loaded.convex)

        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_test_mesh(project)
            dynamic = Entity("Moving Door")
            dynamic.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            dynamic.add_component(Collider(shape="mesh"))
            convex_dynamic = Entity("Convex Door")
            convex_dynamic.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            convex_dynamic.add_component(Collider(shape="mesh", convex=True))

            from p64.engine.validation import entity_reference_errors

            errors = entity_reference_errors(project, dynamic)
            convex_errors = entity_reference_errors(project, convex_dynamic)
            self.assertIn("Non-convex MeshCollider requires a GameObject", errors)
            self.assertNotIn("Non-convex MeshCollider requires a GameObject", convex_errors)

    def test_convex_hull_uses_outer_points_and_ignores_inner_mesh_detail(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_cube_with_inner_detail(project)
            renderer = MeshRenderer(mesh=metadata.id, submesh="Body")

            hull = convex_hull(project, renderer)

            self.assertIsNotNone(hull)
            self.assertLess(len(hull.triangles), len(mesh_triangles(project, renderer)))
            self.assertEqual(hull.bounds[0].to_list(), [-1.0, -1.0, -1.0])
            self.assertEqual(hull.bounds[1].to_list(), [1.0, 1.0, 1.0])

    def test_convex_hull_reduces_large_mesh_point_cloud_before_exact_hull(self):
        from p64.engine.mesh_geometry import build_convex_hull

        points = [
            Vec3(cos(index * 0.37) * (1.0 + (index % 7) * 0.03), sin(index * 0.23), cos(index * 0.11))
            for index in range(240)
        ]

        hull = build_convex_hull(points)

        self.assertIsNotNone(hull)
        self.assertLessEqual(len(hull.vertices), 100)
        self.assertLess(len(hull.triangles), 400)

    def test_convex_mesh_collider_fills_concavity_as_simplified_hull(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_cube_mesh(project)
            convex = Entity("Convex")
            convex.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            convex.add_component(Collider(shape="mesh", convex=True))
            non_convex = Entity("NonConvex", object_type=GAME_OBJECT)
            non_convex.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            non_convex.add_component(Collider(shape="mesh"))
            actor = Entity("Actor")
            actor_collider = Collider(size=Vec3(0.2, 0.2, 0.2))
            actor.add_component(actor_collider)

            convex_hits = CollisionWorld(Scene("Test", [actor, convex]), project).overlaps(actor, actor_collider)
            non_convex_hits = CollisionWorld(Scene("Test", [actor, non_convex]), project).overlaps(actor, actor_collider)

            self.assertEqual([hit.entity.name for hit in convex_hits], ["Convex"])
            self.assertEqual(non_convex_hits, [])

    def test_obj_import_does_not_bake_collision_without_mesh_collider(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_test_mesh(project)

            reloaded = AssetMetadata.load(find_metadata_for_source(project.root / metadata.source))

            self.assertNotIn("collision", reloaded.settings)

    def test_static_world_ignores_objects_without_enabled_colliders(self):
        static = Entity("Static", object_type=GAME_OBJECT)
        static.add_component(Collider(enabled=False))
        plain = Entity("Plain", object_type=GAME_OBJECT)
        scene = Scene("Test", [static, plain])

        world = CollisionWorld(scene)

        self.assertEqual(world.static_colliders, [])

    def test_primitive_static_colliders_do_not_bake_mesh_collision_data(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_test_mesh(project)
            static = Entity("Static", object_type=GAME_OBJECT)
            static.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            static.add_component(Collider(shape="box"))
            actor = Entity("Actor")
            actor_collider = Collider(size=Vec3(0.2, 0.2, 0.2))
            actor.add_component(actor_collider)
            scene = Scene("Test", [actor, static])

            hits = CollisionWorld(scene, project).overlaps(actor, actor_collider)
            reloaded = AssetMetadata.load(find_metadata_for_source(project.root / metadata.source))

            self.assertEqual([hit.entity.name for hit in hits], ["Static"])
            self.assertNotIn("collision", reloaded.settings)

    def test_mesh_collider_bakes_and_reuses_mdp64_collision_data(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_floor_mesh(project)
            floor = Entity("Floor", object_type=GAME_OBJECT)
            floor.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            floor.add_component(Collider(shape="mesh"))
            actor = Entity("Actor")
            actor_collider = Collider(size=Vec3(0.2, 0.2, 0.2))
            actor.add_component(actor_collider)
            scene = Scene("Test", [actor, floor])

            hits = CollisionWorld(scene, project).overlaps(actor, actor_collider)
            reloaded = AssetMetadata.load(find_metadata_for_source(project.root / metadata.source))

            self.assertEqual({hit.entity.name for hit in hits}, {"Floor"})
            collision_data = reloaded.settings.get("collision", {})
            self.assertEqual(collision_data.get("version"), 1)
            baked_meshes = collision_data.get("meshes", {})
            self.assertIn(metadata.settings["model"]["meshes"][0]["id"], baked_meshes)
            self.assertGreater(len(next(iter(baked_meshes.values()))["triangles"]), 0)

            clear_mesh_geometry_cache(project)
            with mock.patch("p64.engine.mesh_geometry.parse_obj") as parse:
                triangles = mesh_triangles(project, MeshRenderer(mesh=metadata.id, submesh="Body"))

            self.assertGreater(len(triangles), 0)
            parse.assert_not_called()

    def test_character_controller_blocks_against_static_mesh_collider(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_floor_mesh(project)
            floor = Entity("Floor", object_type=GAME_OBJECT)
            floor.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            floor.add_component(Collider(shape="mesh"))
            player = Entity("Player")
            controller = CharacterController(height=1.8, radius=0.35, gravity=9.0)
            player.add_component(controller)
            scene = Scene("Test", [player, floor])

            world = CollisionWorld(scene, project)
            world.move_character(player, controller, Vec3(), 0.1)

            self.assertEqual(player.transform.position.y, 0.0)
            self.assertTrue(controller.grounded)

    def test_mesh_collider_uses_triangle_shape_not_triangle_bounds(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_test_mesh(project)
            mesh = Entity("Triangle", object_type=GAME_OBJECT)
            mesh.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            mesh.add_component(Collider(shape="mesh"))
            actor = Entity("Actor")
            actor_collider = Collider(center=Vec3(1.8, 1.8, 0.0), size=Vec3(0.2, 0.2, 0.2))
            actor.add_component(actor_collider)
            scene = Scene("Test", [actor, mesh])

            hits = CollisionWorld(scene, project).overlaps(actor, actor_collider)

            self.assertEqual(hits, [])

    def test_sphere_mesh_collider_uses_triangle_shape_not_bounds(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_test_mesh(project)
            mesh = Entity("Triangle", object_type=GAME_OBJECT)
            mesh.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            mesh.add_component(Collider(shape="mesh"))
            actor = Entity("Actor")
            actor_collider = Collider(shape="sphere", center=Vec3(1.8, 1.8, 0.0), radius=0.1)
            actor.add_component(actor_collider)
            scene = Scene("Test", [actor, mesh])

            hits = CollisionWorld(scene, project).overlaps(actor, actor_collider)

            self.assertEqual(hits, [])

    def test_entity_physics_gravity_force_impulse_and_drag_move_body(self):
        actor = Entity("Actor")
        collider = Collider(size=Vec3(0.5, 0.5, 0.5))
        physics = EntityPhysics(use_gravity=True, drag=0.5)
        actor.add_component(collider)
        actor.add_component(physics)
        scene = Scene("Test", [actor])

        physics.add_impulse(Vec3(2.0, 0.0, 0.0))
        physics.add_force(Vec3(2.0, 0.0, 0.0))
        CollisionWorld(scene).step_physics(1.0)

        self.assertGreater(actor.transform.position.x, 1.9)
        self.assertLess(actor.transform.position.y, 0.0)
        self.assertLess(physics.velocity.x, 4.0)
        self.assertEqual(physics._force.to_list(), [0.0, 0.0, 0.0])

    def test_entity_physics_kinematic_and_freeze_axes(self):
        kinematic = Entity("Kinematic")
        kinematic_physics = EntityPhysics(is_kinematic=True, velocity=Vec3(10, 0, 0), use_gravity=False)
        kinematic.add_component(kinematic_physics)
        frozen = Entity("Frozen")
        frozen_physics = EntityPhysics(use_gravity=False, velocity=Vec3(1, 2, 3), freeze_position=Vec3(1, 0, 1))
        frozen.add_component(frozen_physics)
        scene = Scene("Test", [kinematic, frozen])

        CollisionWorld(scene).step_physics(1.0)

        self.assertEqual(kinematic.transform.position.to_list(), [0.0, 0.0, 0.0])
        self.assertEqual(frozen.transform.position.to_list(), [0.0, 2.0, 0.0])
        self.assertEqual(frozen_physics.velocity.to_list(), [0.0, 2.0, 0.0])

    def test_step_physics_skips_collision_cache_without_enabled_physics(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_test_mesh(project)
            actor = Entity("Actor")
            actor.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            actor.add_component(Collider(shape="mesh", convex=True))
            actor.add_component(EntityPhysics(enabled=False, use_gravity=False))
            world = CollisionWorld(Scene("Test", [actor]), project)

            with mock.patch("p64.engine.collision.collider_bounds", wraps=collision.collider_bounds) as bounds:
                world.step_physics(1.0)

            self.assertEqual(bounds.call_count, 0)
            self.assertEqual(world.dynamic_colliders, [])

    def test_step_physics_skips_collision_cache_for_only_kinematic_physics(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_test_mesh(project)
            actor = Entity("Actor")
            actor.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            actor.add_component(Collider(shape="mesh", convex=True))
            physics = EntityPhysics(is_kinematic=True, use_gravity=False)
            physics.add_force(Vec3(5.0, 0.0, 0.0))
            actor.add_component(physics)
            world = CollisionWorld(Scene("Test", [actor]), project)

            with mock.patch("p64.engine.collision.collider_bounds", wraps=collision.collider_bounds) as bounds:
                world.step_physics(1.0)

            self.assertEqual(bounds.call_count, 0)
            self.assertEqual(actor.transform.position.to_list(), [0.0, 0.0, 0.0])
            self.assertEqual(physics._force.to_list(), [0.0, 0.0, 0.0])

    def test_entity_physics_angular_velocity_and_torque_rotate_body(self):
        actor = Entity("Actor")
        physics = EntityPhysics(use_gravity=False, angular_velocity=Vec3(0.0, 5.0, 0.0), freeze_rotation=Vec3(1, 0, 0))
        actor.add_component(physics)
        scene = Scene("Test", [actor])

        physics.add_torque(Vec3(10.0, 5.0, 0.0))
        CollisionWorld(scene).step_physics(1.0)

        self.assertEqual(actor.transform.rotation.x, 0.0)
        self.assertGreater(actor.transform.rotation.y, 5.0)

    def test_entity_physics_stops_against_box_collider(self):
        actor = Entity("Actor")
        physics = EntityPhysics(use_gravity=False, velocity=Vec3(2.0, 0.0, 0.0))
        actor.add_component(Collider(size=Vec3(1, 1, 1)))
        actor.add_component(physics)
        wall = Entity("Wall", object_type=GAME_OBJECT)
        wall.transform.position = Vec3(1.0, 0.0, 0.0)
        wall.add_component(Collider(size=Vec3(1, 2, 2)))
        scene = Scene("Test", [actor, wall])

        CollisionWorld(scene).step_physics(1.0)

        self.assertEqual(actor.transform.position.x, 0.0)
        self.assertEqual(physics.velocity.x, 0.0)

    def test_entity_physics_uses_child_collider_as_body_shape(self):
        actor = Entity("Actor")
        physics = EntityPhysics(use_gravity=False, velocity=Vec3(2.0, 0.0, 0.0))
        actor.add_component(physics)
        child = actor.add_child(Entity("Body"))
        child.add_component(Collider(size=Vec3(1, 1, 1)))
        wall = Entity("Wall", object_type=GAME_OBJECT)
        wall.transform.position = Vec3(1.0, 0.0, 0.0)
        wall.add_component(Collider(size=Vec3(1, 2, 2)))
        scene = Scene("Test", [actor, wall])

        CollisionWorld(scene).step_physics(1.0)

        self.assertEqual(actor.transform.position.x, 0.0)
        self.assertEqual(physics.velocity.x, 0.0)

    def test_entity_physics_checks_all_child_colliders(self):
        actor = Entity("Actor")
        physics = EntityPhysics(use_gravity=False, velocity=Vec3(2.0, 0.0, 0.0))
        actor.add_component(physics)
        rear = actor.add_child(Entity("Rear"))
        rear.transform.position = Vec3(-5.0, 0.0, 0.0)
        rear.add_component(Collider(size=Vec3(1, 1, 1)))
        front = actor.add_child(Entity("Front"))
        front.add_component(Collider(size=Vec3(1, 1, 1)))
        wall = Entity("Wall", object_type=GAME_OBJECT)
        wall.transform.position = Vec3(1.0, 0.0, 0.0)
        wall.add_component(Collider(size=Vec3(1, 2, 2)))
        scene = Scene("Test", [actor, wall])

        CollisionWorld(scene).step_physics(1.0)

        self.assertEqual(actor.transform.position.x, 0.0)
        self.assertEqual(physics.velocity.x, 0.0)

    def test_entity_physics_compound_broadphase_skips_distant_colliders(self):
        actor = Entity("Actor")
        actor.add_component(EntityPhysics(use_gravity=False, velocity=Vec3(1.0, 0.0, 0.0)))
        for index in range(30):
            child = actor.add_child(Entity(f"Collider {index}"))
            child.transform.position = Vec3(index * 2.0, 0.0, 0.0)
            child.add_component(Collider(size=Vec3(1, 1, 1)))
        blockers: list[Entity] = []
        for index in range(30):
            blocker = Entity(f"Blocker {index}", object_type=GAME_OBJECT)
            blocker.transform.position = Vec3(1000.0 + index * 2.0, 0.0, 0.0)
            blocker.add_component(Collider(size=Vec3(1, 1, 1)))
            blockers.append(blocker)
        scene = Scene("Test", [actor, *blockers])

        with mock.patch("p64.engine.collision.collider_bounds", wraps=collision.collider_bounds) as bounds:
            CollisionWorld(scene).step_physics(1.0)

        self.assertEqual(actor.transform.position.x, 1.0)
        self.assertLess(bounds.call_count, 100)

    def test_entity_physics_reuses_static_bounds_during_step(self):
        actor = Entity("Actor")
        actor.add_component(Collider(size=Vec3(1, 1, 1)))
        actor.add_component(EntityPhysics(use_gravity=False, velocity=Vec3(1.0, 0.0, 0.0)))
        blockers: list[Entity] = []
        for index in range(40):
            blocker = Entity(f"Blocker {index}", object_type=GAME_OBJECT)
            blocker.transform.position = Vec3(100.0 + index * 4.0, 0.0, 0.0)
            blocker.add_component(Collider(size=Vec3(1, 1, 1)))
            blockers.append(blocker)
        world = CollisionWorld(Scene("Test", [actor, *blockers]))

        with mock.patch("p64.engine.collision.collider_bounds", wraps=collision.collider_bounds) as bounds:
            world.step_physics(1.0)

        self.assertEqual(actor.transform.position.x, 1.0)
        self.assertLessEqual(bounds.call_count, 2)

    def test_spatial_hash_skips_distant_static_narrowphase_candidates(self):
        actor = Entity("Actor")
        actor.add_component(Collider(size=Vec3(1, 1, 1)))
        actor.add_component(EntityPhysics(use_gravity=False, velocity=Vec3(1.0, 0.0, 0.0)))
        blockers: list[Entity] = []
        for index in range(50):
            blocker = Entity(f"Blocker {index}", object_type=GAME_OBJECT)
            blocker.transform.position = Vec3(1000.0 + index * 4.0, 0.0, 0.0)
            blocker.add_component(Collider(size=Vec3(1, 1, 1)))
            blockers.append(blocker)
        world = CollisionWorld(Scene("Test", [actor, *blockers]))

        with mock.patch("p64.engine.collision._separation", wraps=collision._separation) as separation:
            world.step_physics(1.0)

        self.assertEqual(actor.transform.position.x, 1.0)
        separation.assert_not_called()

    def test_entity_physics_ignores_colliders_inside_same_body_subtree(self):
        actor = Entity("Actor")
        physics = EntityPhysics(use_gravity=False, velocity=Vec3(1.0, 0.0, 0.0))
        actor.add_component(Collider(size=Vec3(1, 1, 1)))
        actor.add_component(physics)
        child = actor.add_child(Entity("Child"))
        child.add_component(Collider(size=Vec3(1, 1, 1)))
        scene = Scene("Test", [actor])

        CollisionWorld(scene).step_physics(1.0)

        self.assertEqual(actor.transform.position.x, 1.0)
        self.assertEqual(physics.velocity.x, 1.0)

    def test_active_physics_collides_with_passive_dynamic_collider(self):
        actor = Entity("Actor")
        physics = EntityPhysics(use_gravity=False, velocity=Vec3(2.0, 0.0, 0.0))
        actor.add_component(Collider(size=Vec3(1, 1, 1)))
        actor.add_component(physics)
        passive = Entity("Passive")
        passive.transform.position = Vec3(1.0, 0.0, 0.0)
        passive.add_component(Collider(size=Vec3(1, 1, 1)))
        scene = Scene("Test", [actor, passive])

        CollisionWorld(scene).step_physics(1.0)

        self.assertEqual(actor.transform.position.x, 0.0)
        self.assertEqual(physics.velocity.x, 0.0)

    def test_entity_physics_parent_does_not_absorb_child_physics_body(self):
        parent = Entity("Parent")
        parent.add_component(EntityPhysics(use_gravity=False, velocity=Vec3(1.0, 0.0, 0.0)))
        child = parent.add_child(Entity("Child Body"))
        child.add_component(Collider(size=Vec3(1, 1, 1)))
        child.add_component(EntityPhysics(use_gravity=False))
        wall = Entity("Wall", object_type=GAME_OBJECT)
        wall.transform.position = Vec3(1.0, 0.0, 0.0)
        wall.add_component(Collider(size=Vec3(1, 2, 2)))
        scene = Scene("Test", [parent, wall])

        CollisionWorld(scene).step_physics(1.0)

        self.assertEqual(parent.transform.position.x, 1.0)

    def test_child_primitive_collider_bounds_include_parent_transform(self):
        parent = Entity("Parent")
        parent.transform.position = Vec3(5.0, 0.0, 0.0)
        child = parent.add_child(Entity("Child"))
        child.transform.position = Vec3(1.0, 0.0, 0.0)
        collider = Collider(size=Vec3(2, 2, 2))
        child.add_component(collider)

        bounds = collider_bounds(child, collider)

        self.assertEqual(bounds.min.to_list(), [5.0, -1.0, -1.0])
        self.assertEqual(bounds.max.to_list(), [7.0, 1.0, 1.0])

    def test_entity_physics_stops_against_static_mesh_collider(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_floor_mesh(project)
            floor = Entity("Floor", object_type=GAME_OBJECT)
            floor.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            floor.add_component(Collider(shape="mesh"))
            actor = Entity("Actor")
            actor.add_component(Collider(size=Vec3(0.5, 0.5, 0.5)))
            physics = EntityPhysics(use_gravity=True)
            actor.add_component(physics)
            scene = Scene("Test", [actor, floor])

            CollisionWorld(scene, project).step_physics(0.1)

            self.assertEqual(actor.transform.position.y, 0.0)
            self.assertEqual(physics.velocity.y, 0.0)

    def test_entity_physics_with_convex_mesh_collider_stops_against_box(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_cube_mesh(project)
            actor = Entity("Actor")
            actor.add_component(MeshRenderer(mesh=metadata.id, submesh="Body"))
            actor.add_component(Collider(shape="mesh", convex=True))
            physics = EntityPhysics(use_gravity=False, velocity=Vec3(2.0, 0.0, 0.0))
            actor.add_component(physics)
            wall = Entity("Wall", object_type=GAME_OBJECT)
            wall.transform.position = Vec3(1.0, 0.0, 0.0)
            wall.add_component(Collider(size=Vec3(1.0, 2.0, 2.0)))
            scene = Scene("Test", [actor, wall])

            CollisionWorld(scene, project).step_physics(1.0)

            self.assertEqual(actor.transform.position.x, 0.0)
            self.assertEqual(physics.velocity.x, 0.0)

    def test_entity_physics_with_convex_mesh_collider_stops_against_static_mesh(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            cube = _import_cube_mesh(project)
            floor_mesh = _import_floor_mesh(project)
            actor = Entity("Actor")
            actor.add_component(MeshRenderer(mesh=cube.id, submesh="Body"))
            actor.add_component(Collider(shape="mesh", convex=True))
            physics = EntityPhysics(use_gravity=True)
            actor.add_component(physics)
            floor = Entity("Floor", object_type=GAME_OBJECT)
            floor.add_component(MeshRenderer(mesh=floor_mesh.id, submesh="Body"))
            floor.add_component(Collider(shape="mesh"))
            scene = Scene("Test", [actor, floor])

            CollisionWorld(scene, project).step_physics(0.1)

            self.assertEqual(actor.transform.position.y, 0.0)
            self.assertEqual(physics.velocity.y, 0.0)

    def test_mesh_metadata_cache_reuses_and_clears_project_metadata(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            metadata = _import_test_mesh(project)
            renderer = MeshRenderer(mesh=metadata.id, submesh="Body")
            clear_mesh_geometry_cache(project)

            with mock.patch("p64.engine.mesh_geometry.discover_metadata", wraps=mesh_geometry.discover_metadata) as discover:
                self.assertIsNotNone(mesh_geometry.mesh_bounds(project, renderer))
                self.assertIsNotNone(mesh_geometry.mesh_bounds(project, renderer))
                self.assertEqual(discover.call_count, 1)
                clear_mesh_geometry_cache(project)
                self.assertIsNotNone(mesh_geometry.mesh_bounds(project, renderer))
                self.assertEqual(discover.call_count, 2)


def _import_test_mesh(project: Project):
    obj = project.root / "test_mesh.obj"
    obj.write_text(
        "o Body\n"
        "v 0 0 0\n"
        "v 2 0 0\n"
        "v 0 2 0\n"
        "f 1 2 3\n",
        encoding="utf-8",
    )
    return import_obj_to_project(project, obj)


def _import_floor_mesh(project: Project):
    obj = project.root / "floor_mesh.obj"
    obj.write_text(
        "o Body\n"
        "v -2 0 -2\n"
        "v 2 0 -2\n"
        "v 2 0 2\n"
        "v -2 0 2\n"
        "f 1 2 3 4\n",
        encoding="utf-8",
    )
    return import_obj_to_project(project, obj)


def _import_cube_mesh(project: Project):
    obj = project.root / "cube_mesh.obj"
    obj.write_text(
        "o Body\n"
        "v -1 -1 -1\n"
        "v 1 -1 -1\n"
        "v 1 1 -1\n"
        "v -1 1 -1\n"
        "v -1 -1 1\n"
        "v 1 -1 1\n"
        "v 1 1 1\n"
        "v -1 1 1\n"
        "f 1 3 2\n"
        "f 1 4 3\n"
        "f 5 6 7\n"
        "f 5 7 8\n"
        "f 1 2 6\n"
        "f 1 6 5\n"
        "f 2 3 7\n"
        "f 2 7 6\n"
        "f 3 4 8\n"
        "f 3 8 7\n"
        "f 4 1 5\n"
        "f 4 5 8\n",
        encoding="utf-8",
    )
    return import_obj_to_project(project, obj)


def _import_cube_with_inner_detail(project: Project):
    obj = project.root / "detailed_cube_mesh.obj"
    obj.write_text(
        "o Body\n"
        "v -1 -1 -1\n"
        "v 1 -1 -1\n"
        "v 1 1 -1\n"
        "v -1 1 -1\n"
        "v -1 -1 1\n"
        "v 1 -1 1\n"
        "v 1 1 1\n"
        "v -1 1 1\n"
        "v 0 0 0\n"
        "f 1 3 2\n"
        "f 1 4 3\n"
        "f 5 6 7\n"
        "f 5 7 8\n"
        "f 1 2 6\n"
        "f 1 6 5\n"
        "f 2 3 7\n"
        "f 2 7 6\n"
        "f 3 4 8\n"
        "f 3 8 7\n"
        "f 4 1 5\n"
        "f 4 5 8\n"
        "f 1 2 9\n"
        "f 2 3 9\n"
        "f 3 4 9\n"
        "f 4 1 9\n"
        "f 5 6 9\n"
        "f 6 7 9\n"
        "f 7 8 9\n"
        "f 8 5 9\n",
        encoding="utf-8",
    )
    return import_obj_to_project(project, obj)


if __name__ == "__main__":
    unittest.main()
