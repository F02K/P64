from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from p64.engine.components import AudioListener, Camera, Light, ScriptComponent
from p64.engine.entity import Entity, entity_effectively_active
from p64.engine.lighting import clamp_lighting_settings, default_lighting_settings, lighting_path_for_scene, load_lighting_settings, save_lighting_settings
from p64.engine.render_settings import clamp_render_settings, default_render_settings


@dataclass
class Scene:
    name: str
    entities: list[Entity] = field(default_factory=list)
    render_settings: dict[str, Any] = field(default_factory=dict)
    lighting_settings: dict[str, Any] = field(default_factory=default_lighting_settings)

    def add_entity(self, entity: Entity) -> Entity:
        self.entities.append(entity)
        return entity

    def walk(self) -> Iterable[Entity]:
        for entity in self.entities:
            yield from entity.walk()

    def walk_active(self) -> Iterable[Entity]:
        for entity in self.walk():
            if entity_effectively_active(entity):
                yield entity

    def find(self, entity_id: str) -> Entity | None:
        for entity in self.entities:
            found = entity.find(entity_id)
            if found:
                return found
        return None

    def find_scene_object(self, name_or_id: str) -> Entity | None:
        return self._find_by_name_or_id(name_or_id)

    def find_game_object(self, name_or_id: str) -> Entity | None:
        found = self._find_by_name_or_id(name_or_id)
        return found if found and found.is_game_object else None

    def find_entity(self, name_or_id: str) -> Entity | None:
        found = self._find_by_name_or_id(name_or_id)
        return found if found and found.is_entity else None

    def _find_by_name_or_id(self, name_or_id: str) -> Entity | None:
        for entity in self.walk():
            if entity.id == name_or_id or entity.name == name_or_id:
                return entity
        return None

    def active_camera(self) -> Entity | None:
        fallback = None
        for entity in self.walk_active():
            for component in entity.components:
                if isinstance(component, Camera):
                    fallback = fallback or entity
                    if component.active:
                        return entity
        return fallback

    def active_audio_listener(self) -> Entity | None:
        for entity in self.walk_active():
            for component in entity.components:
                if isinstance(component, AudioListener) and component.enabled and component.active:
                    return entity
        return None

    def lights(self) -> list[tuple[Entity, Light]]:
        return [
            (entity, component)
            for entity in self.walk_active()
            for component in entity.components
            if isinstance(component, Light) and component.enabled
        ]

    def script_components(self) -> list[tuple[Entity, ScriptComponent]]:
        return [
            (entity, component)
            for entity in self.walk_active()
            for component in entity.components
            if entity.is_entity and isinstance(component, ScriptComponent) and component.enabled
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entities": [entity.to_dict() for entity in self.entities],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        legacy_lighting = dict(data.get("render_settings", {}))
        if "fog" in legacy_lighting and "fog_enabled" not in legacy_lighting:
            legacy_lighting["fog_enabled"] = legacy_lighting["fog"]
        scene = cls(
            name=str(data.get("name", "Scene")),
            render_settings={},
            lighting_settings=clamp_lighting_settings({**default_lighting_settings(), **legacy_lighting}),
            entities=[Entity.from_dict(_without_legacy_fog(item)) for item in data.get("entities", [])],
        )
        _remove_legacy_fog_components(scene.entities)
        return scene

    @classmethod
    def load(cls, path: Path) -> "Scene":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        scene = cls.from_dict(data)
        lighting_path = lighting_path_for_scene(path)
        scene.lighting_settings = load_lighting_settings(lighting_path, dict(data.get("render_settings", {})))
        if not lighting_path.exists():
            save_lighting_settings(lighting_path, scene.lighting_settings)
        return scene

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        save_lighting_settings(lighting_path_for_scene(path), self.lighting_settings)

    def run_scripts_once(self, project_root: Path, dt: float = 1 / 60, scene_manager: object | None = None) -> list[str]:
        from p64.engine.project import Project
        from p64.engine.runtime_session import RuntimeSession

        project = Project.load(project_root)
        session = RuntimeSession(project, self)
        if scene_manager is not None:
            session.scene_manager = scene_manager
            session.scene_manager.current_scene = self
        return session.tick(dt)


def _remove_legacy_fog_components(entities: list[Entity]) -> None:
    for entity in entities:
        entity.components = [component for component in entity.components if component.type_name != "Fog"]
        _remove_legacy_fog_components(entity.children)


def _without_legacy_fog(data: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(data)
    cleaned["components"] = [
        component
        for component in data.get("components", [])
        if not isinstance(component, dict) or component.get("type") != "Fog"
    ]
    cleaned["children"] = [
        _without_legacy_fog(child)
        for child in data.get("children", [])
        if isinstance(child, dict)
    ]
    return cleaned
