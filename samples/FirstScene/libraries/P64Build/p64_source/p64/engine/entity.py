from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
from uuid import uuid4

from p64.engine.components import Component, Transform, component_from_dict

SCENE_OBJECT = "scene_object"
GAME_OBJECT = "game_object"
ENTITY = "entity"
OBJECT_TYPES = {GAME_OBJECT, ENTITY}


@dataclass
class Entity:
    name: str
    id: str = field(default_factory=lambda: uuid4().hex)
    object_type: str = ENTITY
    active: bool = True
    persistent: bool = False
    transform: Transform = field(default_factory=Transform)
    components: list[Component] = field(default_factory=list)
    children: list["Entity"] = field(default_factory=list)
    parent: "Entity | None" = field(default=None, repr=False, compare=False)

    def add_child(self, child: "Entity") -> "Entity":
        child.parent = self
        set_object_type_recursive(child, self.object_type)
        self.children.append(child)
        return child

    def add_component(self, component: Component) -> Component:
        self.components.append(component)
        return component

    @property
    def is_game_object(self) -> bool:
        return self.object_type == GAME_OBJECT

    @property
    def is_entity(self) -> bool:
        return self.object_type == ENTITY

    @property
    def object_type_label(self) -> str:
        return "GameObject" if self.is_game_object else "Entity"

    def walk(self) -> Iterable["Entity"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, entity_id: str) -> "Entity | None":
        for entity in self.walk():
            if entity.id == entity_id:
                return entity
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "object_type": self.object_type if self.object_type in OBJECT_TYPES else ENTITY,
            "active": self.active,
            "persistent": self.persistent,
            "transform": self.transform.to_dict(),
            "components": [component.to_dict() for component in self.components],
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        entity = cls(
            id=str(data.get("id") or uuid4().hex),
            name=str(data.get("name", "Entity")),
            object_type=_object_type(data.get("object_type", ENTITY)),
            active=bool(data.get("active", True)),
            persistent=bool(data.get("persistent", False)),
            transform=Transform.from_dict(data.get("transform")),
            components=[component_from_dict(item) for item in data.get("components", [])],
        )
        for child_data in data.get("children", []):
            entity.add_child(cls.from_dict(child_data))
        return entity


def _object_type(value: object) -> str:
    text = str(value or ENTITY)
    return text if text in OBJECT_TYPES else ENTITY


def set_object_type_recursive(entity: Entity, object_type: str) -> None:
    entity.object_type = _object_type(object_type)
    for child in entity.children:
        set_object_type_recursive(child, entity.object_type)


SceneObject = Entity
