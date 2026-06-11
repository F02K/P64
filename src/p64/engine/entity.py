from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
from uuid import uuid4

from p64.engine.components import Component, Transform, component_from_dict


@dataclass
class Entity:
    name: str
    id: str = field(default_factory=lambda: uuid4().hex)
    active: bool = True
    transform: Transform = field(default_factory=Transform)
    components: list[Component] = field(default_factory=list)
    children: list["Entity"] = field(default_factory=list)
    parent: "Entity | None" = field(default=None, repr=False, compare=False)

    def add_child(self, child: "Entity") -> "Entity":
        child.parent = self
        self.children.append(child)
        return child

    def add_component(self, component: Component) -> Component:
        self.components.append(component)
        return component

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
            "active": self.active,
            "transform": self.transform.to_dict(),
            "components": [component.to_dict() for component in self.components],
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        entity = cls(
            id=str(data.get("id") or uuid4().hex),
            name=str(data.get("name", "Entity")),
            active=bool(data.get("active", True)),
            transform=Transform.from_dict(data.get("transform")),
            components=[component_from_dict(item) for item in data.get("components", [])],
        )
        for child_data in data.get("children", []):
            entity.add_child(cls.from_dict(child_data))
        return entity
