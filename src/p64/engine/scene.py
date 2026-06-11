from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from p64.engine.components import Camera, Fog, Light, ScriptComponent
from p64.engine.entity import Entity
from p64.engine.scripting import ScriptContext, ScriptManager


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

    def find(self, entity_id: str) -> Entity | None:
        for entity in self.entities:
            found = entity.find(entity_id)
            if found:
                return found
        return None

    def active_camera(self) -> Entity | None:
        fallback = None
        for entity in self.walk():
            if not entity.active:
                continue
            for component in entity.components:
                if isinstance(component, Camera):
                    fallback = fallback or entity
                    if component.active:
                        return entity
        return fallback

    def lights(self) -> list[tuple[Entity, Light]]:
        return [
            (entity, component)
            for entity in self.walk()
            for component in entity.components
            if entity.active and isinstance(component, Light) and component.enabled
        ]

    def fog(self) -> Fog | None:
        volume = self.fog_volume()
        return volume[1] if volume else None

    def fog_volume(self) -> tuple[Entity, Fog] | None:
        for entity in self.walk():
            if not entity.active:
                continue
            for component in entity.components:
                if isinstance(component, Fog) and component.enabled:
                    return entity, component
        return None

    def script_components(self) -> list[tuple[Entity, ScriptComponent]]:
        return [
            (entity, component)
            for entity in self.walk()
            for component in entity.components
            if entity.active and isinstance(component, ScriptComponent) and component.enabled
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
            render_settings=dict(data.get("render_settings", {})),
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

    def run_scripts_once(self, project_root: Path, dt: float = 1 / 60) -> list[str]:
        manager = ScriptManager(project_root / "scripts")
        context = ScriptContext(scene=self, time=dt)
        errors: list[str] = []
        for entity, component in self.script_components():
            for entry, instance, error in manager.instantiate_component(entity, component, context):
                if error:
                    errors.append(error)
                    continue
                if hasattr(instance, "on_start"):
                    try:
                        instance.on_start()
                    except Exception as exc:  # pragma: no cover - exact user code is unknowable
                        errors.append(f"{entry.script}:{entry.class_name}.on_start failed: {exc}")
                if hasattr(instance, "on_update"):
                    try:
                        instance.on_update(dt)
                    except Exception as exc:  # pragma: no cover
                        errors.append(f"{entry.script}:{entry.class_name}.on_update failed: {exc}")
        return errors
