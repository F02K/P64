from __future__ import annotations

from pathlib import Path

from p64.engine.components import SpawnPoint
from p64.engine.entity import Entity
from p64.engine.math import Vec3
from p64.engine.project import Project
from p64.engine.scene import Scene


class SceneManager:
    def __init__(self, project: Project, scene: Scene | None = None) -> None:
        self.project = project
        self.current_scene = scene or project.load_startup_scene()
        self.current_scene_path = project.resolve_scene_path(project.startup_scene)
        self.previous_scene_path: Path | None = None
        self._queued_scene: Path | None = None
        self._queued_spawn_id: str | None = None
        self._persistent_entities: dict[str, Entity] = {}

    def load_scene(self, scene_path: str | Path, spawn_id: str | None = None) -> None:
        self._queued_scene = self.project.resolve_scene_path(scene_path)
        self._queued_spawn_id = spawn_id

    def load_scene_by_name(self, name: str, spawn_id: str | None = None) -> None:
        scene_path = self.project.scene_path_by_name(name)
        if scene_path is None:
            raise FileNotFoundError(f"Scene not found: {name}")
        self._queued_scene = scene_path
        self._queued_spawn_id = spawn_id

    def apply_queued_scene(self) -> bool:
        if self._queued_scene is None:
            return False
        self._remember_persistent_entities()
        scene_path = self._queued_scene
        spawn_id = self._queued_spawn_id
        self._queued_scene = None
        self._queued_spawn_id = None
        next_scene = Scene.load(scene_path)
        existing = {entity.id for entity in next_scene.entities}
        for entity in self._persistent_entities.values():
            if entity.id not in existing:
                entity.parent = None
                next_scene.entities.insert(0, entity)
        spawn_entity = self._select_spawn_point(next_scene, spawn_id)
        if spawn_entity is not None:
            self._apply_spawn_point(spawn_entity)
        self.previous_scene_path = self.current_scene_path
        self.current_scene = next_scene
        self.current_scene_path = scene_path
        return True

    def _remember_persistent_entities(self) -> None:
        for entity in self.current_scene.entities:
            if entity.persistent:
                self._persistent_entities[entity.id] = entity

    def _select_spawn_point(self, scene: Scene, spawn_id: str | None) -> Entity | None:
        spawns: list[tuple[Entity, SpawnPoint]] = []
        for entity in scene.walk_active():
            for component in entity.components:
                if isinstance(component, SpawnPoint) and component.enabled:
                    spawns.append((entity, component))
        if not spawns:
            return None
        if spawn_id:
            for entity, component in spawns:
                if component.spawn_id == spawn_id:
                    return entity
        previous_names = self._previous_scene_names()
        for entity, component in spawns:
            if component.from_scene and component.from_scene in previous_names:
                return entity
        for entity, component in spawns:
            if component.is_default:
                return entity
        return spawns[0][0]

    def _previous_scene_names(self) -> set[str]:
        if self.current_scene_path is None:
            return set()
        try:
            relative = self.current_scene_path.resolve().relative_to(self.project.root.resolve()).as_posix()
        except ValueError:
            relative = self.current_scene_path.as_posix()
        return {
            self.current_scene_path.as_posix(),
            str(self.current_scene_path),
            relative,
            self.current_scene_path.name,
            self.current_scene_path.stem,
            self.current_scene.name,
        }

    def _apply_spawn_point(self, spawn_entity: Entity) -> None:
        target = self._first_persistent_entity()
        if target is None:
            return
        target.transform.position = Vec3.from_value(spawn_entity.transform.position)
        target.transform.local_quaternion = spawn_entity.transform.local_quaternion

    def _first_persistent_entity(self) -> Entity | None:
        for entity in self._persistent_entities.values():
            return entity
        return None
