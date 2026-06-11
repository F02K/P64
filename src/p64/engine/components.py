from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from p64.engine.math import Vec3, compose_transform, multiply


@dataclass(slots=True)
class Transform:
    position: Vec3 = field(default_factory=Vec3)
    rotation: Vec3 = field(default_factory=Vec3)
    scale: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))

    def local_matrix(self) -> list[float]:
        return compose_transform(self.position, self.rotation, self.scale)

    def world_matrix(self, entity: Any) -> list[float]:
        local = self.local_matrix()
        if entity.parent is None:
            return local
        return multiply(entity.parent.transform.world_matrix(entity.parent), local)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": self.position.to_list(),
            "rotation": self.rotation.to_list(),
            "scale": self.scale.to_list(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Transform":
        data = data or {}
        return cls(
            position=Vec3.from_value(data.get("position", [0.0, 0.0, 0.0])),
            rotation=Vec3.from_value(data.get("rotation", [0.0, 0.0, 0.0])),
            scale=Vec3.from_value(data.get("scale", [1.0, 1.0, 1.0])),
        )


@dataclass(slots=True)
class Component:
    enabled: bool = True

    type_name: str = "Component"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type_name, "enabled": self.enabled}


@dataclass(slots=True)
class MeshRenderer(Component):
    mesh: str = ""
    submesh: str | None = None
    material: str | None = None
    shader: str | None = None
    visible: bool = True
    type_name: str = "MeshRenderer"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "mesh": self.mesh,
            "submesh": self.submesh,
            "material": self.material,
            "shader": self.shader,
            "visible": self.visible,
        })
        return data


@dataclass(slots=True)
class Camera(Component):
    fov: float = 60.0
    near: float = 0.1
    far: float = 500.0
    active: bool = False
    type_name: str = "Camera"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({"fov": self.fov, "near": self.near, "far": self.far, "active": self.active})
        return data


@dataclass(slots=True)
class Light(Component):
    kind: str = "directional"
    color: Vec3 = field(default_factory=lambda: Vec3(1.0, 0.96, 0.84))
    intensity: float = 1.0
    type_name: str = "Light"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({"kind": self.kind, "color": self.color.to_list(), "intensity": self.intensity})
        return data


@dataclass(slots=True)
class Fog(Component):
    color: Vec3 = field(default_factory=lambda: Vec3(0.46, 0.58, 0.72))
    size: Vec3 = field(default_factory=lambda: Vec3(60.0, 30.0, 60.0))
    near: float = 20.0
    far: float = 120.0
    density: float = 0.0
    type_name: str = "Fog"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "color": self.color.to_list(),
            "size": self.size.to_list(),
            "near": self.near,
            "far": self.far,
            "density": self.density,
        })
        return data


@dataclass(slots=True)
class ScriptEntry:
    script: str = ""
    class_name: str = ""
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"script": self.script, "class_name": self.class_name, "enabled": self.enabled}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScriptEntry":
        return cls(
            script=str(data.get("script", "")),
            class_name=str(data.get("class_name", "")),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(slots=True)
class ScriptComponent(Component):
    scripts: list[ScriptEntry] = field(default_factory=list)
    type_name: str = "ScriptComponent"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({"scripts": [entry.to_dict() for entry in self.scripts]})
        return data


def component_from_dict(data: dict[str, Any]) -> Component:
    kind = data.get("type")
    enabled = bool(data.get("enabled", True))
    if kind == "MeshRenderer":
        return MeshRenderer(
            enabled=enabled,
            mesh=str(data.get("mesh", "")),
            submesh=data.get("submesh"),
            material=data.get("material"),
            shader=data.get("shader"),
            visible=bool(data.get("visible", True)),
        )
    if kind == "Camera":
        return Camera(
            enabled=enabled,
            fov=float(data.get("fov", 60.0)),
            near=float(data.get("near", 0.1)),
            far=float(data.get("far", 500.0)),
            active=bool(data.get("active", False)),
        )
    if kind == "Light":
        return Light(
            enabled=enabled,
            kind=str(data.get("kind", "directional")),
            color=Vec3.from_value(data.get("color", [1.0, 0.96, 0.84])),
            intensity=float(data.get("intensity", 1.0)),
        )
    if kind == "Fog":
        return Fog(
            enabled=enabled,
            color=Vec3.from_value(data.get("color", [0.46, 0.58, 0.72])),
            size=Vec3.from_value(data.get("size", [60.0, 30.0, 60.0])),
            near=float(data.get("near", 20.0)),
            far=float(data.get("far", 120.0)),
            density=float(data.get("density", 0.0)),
        )
    if kind == "ScriptComponent":
        if "scripts" in data:
            scripts = [ScriptEntry.from_dict(item) for item in data.get("scripts", [])]
        else:
            scripts = [
                ScriptEntry(
                    script=str(data.get("script", "")),
                    class_name=str(data.get("class_name", "")),
                    enabled=enabled,
                )
            ]
        return ScriptComponent(
            enabled=enabled,
            scripts=scripts,
        )
    return Component(enabled=enabled)
