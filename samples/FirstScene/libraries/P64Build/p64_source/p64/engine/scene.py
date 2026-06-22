from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from p64.engine.components import AudioListener, Camera, Fog, Light, ScriptComponent
from p64.engine.entity import Entity, entity_effectively_active
from p64.engine.render_settings import clamp_render_settings, default_render_settings


@dataclass
class Scene:
    name: str
    entities: list[Entity] = field(default_factory=list)
    render_settings: dict[str, Any] = field(default_factory=dict)

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

    def fog(self) -> Fog | None:
        volume = self.fog_volume()
        return volume[1] if volume else None

    def fog_volume(self) -> tuple[Entity, Fog] | None:
        for entity in self.walk_active():
            for component in entity.components:
                if isinstance(component, Fog) and component.enabled:
                    return entity, component
        return None

    def script_components(self) -> list[tuple[Entity, ScriptComponent]]:
        return [
            (entity, component)
            for entity in self.walk_active()
            for component in entity.components
            if isinstance(component, ScriptComponent) and component.enabled
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "render_settings": self.render_settings,
            "entities": [entity.to_dict() for entity in self.entities],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        return cls(
            name=str(data.get("name", "Scene")),
            render_settings=clamp_render_settings({**default_render_settings(), **dict(data.get("render_settings", {}))}),
            entities=[Entity.from_dict(item) for item in data.get("entities", [])],
        )

    @classmethod
    def load(cls, path: Path) -> "Scene":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    def run_scripts_once(self, project_root: Path, dt: float = 1 / 60, scene_manager: object | None = None) -> list[str]:
        from p64.engine.project import Project
        from p64.engine.runtime_session import RuntimeSession

        project = Project.load(project_root)
        session = RuntimeSession(project, self)
        if scene_manager is not None:
            session.scene_manager = scene_manager
            session.scene_manager.current_scene = self
        return session.tick(dt)
