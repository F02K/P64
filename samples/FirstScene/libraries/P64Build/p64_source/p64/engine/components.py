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
    source_materials: list[str] = field(default_factory=list)
    material_slots: list[str | None] = field(default_factory=list)
    visible: bool = True
    type_name: str = "MeshRenderer"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "mesh": self.mesh,
            "submesh": self.submesh,
            "material": self.material,
            "shader": self.shader,
            "source_materials": self.source_materials,
            "material_slots": self.material_slots,
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
    range: float = 12.0
    spot_angle: float = 45.0
    falloff: float = 2.0
    type_name: str = "Light"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "kind": self.kind,
            "color": self.color.to_list(),
            "intensity": self.intensity,
            "range": self.range,
            "spot_angle": self.spot_angle,
            "falloff": self.falloff,
        })
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
class SpawnPoint(Component):
    spawn_id: str = "default"
    from_scene: str = ""
    is_default: bool = False
    type_name: str = "SpawnPoint"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "spawn_id": self.spawn_id,
            "from_scene": self.from_scene,
            "is_default": self.is_default,
        })
        return data


@dataclass(slots=True)
class Collider(Component):
    shape: str = "box"
    size: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    radius: float = 0.5
    center: Vec3 = field(default_factory=Vec3)
    fit_to_mesh: bool = False
    convex: bool = False
    is_trigger: bool = False
    layer: str = "Default"
    mask: str = "*"
    type_name: str = "Collider"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "shape": self.shape,
            "size": self.size.to_list(),
            "radius": self.radius,
            "center": self.center.to_list(),
            "fit_to_mesh": self.fit_to_mesh,
            "convex": self.convex,
            "is_trigger": self.is_trigger,
            "layer": self.layer,
            "mask": self.mask,
        })
        return data


@dataclass(slots=True)
class CharacterController(Component):
    height: float = 1.8
    radius: float = 0.35
    skin_width: float = 0.05
    slope_limit: float = 45.0
    gravity: float = 18.0
    velocity: Vec3 = field(default_factory=Vec3)
    grounded: bool = False
    _runtime_entity: Any | None = field(default=None, repr=False, compare=False)
    _runtime_scene: Any | None = field(default=None, repr=False, compare=False)
    _runtime_project: Any | None = field(default=None, repr=False, compare=False)
    type_name: str = "CharacterController"

    @property
    def is_grounded(self) -> bool:
        return self.grounded

    def bind_runtime(self, entity: Any, scene: Any, project: Any | None = None) -> None:
        self._runtime_entity = entity
        self._runtime_scene = scene
        self._runtime_project = project

    def move(self, motion: Vec3, dt: float) -> Vec3:
        if self._runtime_entity is None or self._runtime_scene is None:
            return Vec3()
        from p64.engine.collision import CollisionWorld

        return CollisionWorld(self._runtime_scene, self._runtime_project).move_character(self._runtime_entity, self, motion, dt)

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "height": self.height,
            "radius": self.radius,
            "skin_width": self.skin_width,
            "slope_limit": self.slope_limit,
            "gravity": self.gravity,
            "velocity": self.velocity.to_list(),
            "grounded": self.grounded,
        })
        return data


@dataclass(slots=True)
class EntityPhysics(Component):
    mass: float = 1.0
    use_gravity: bool = True
    drag: float = 0.0
    angular_drag: float = 0.0
    is_kinematic: bool = False
    velocity: Vec3 = field(default_factory=Vec3)
    angular_velocity: Vec3 = field(default_factory=Vec3)
    freeze_position: Vec3 = field(default_factory=Vec3)
    freeze_rotation: Vec3 = field(default_factory=Vec3)
    _force: Vec3 = field(default_factory=Vec3, repr=False, compare=False)
    _torque: Vec3 = field(default_factory=Vec3, repr=False, compare=False)
    type_name: str = "EntityPhysics"

    @property
    def inverse_mass(self) -> float:
        return 1.0 / max(float(self.mass), 0.001)

    def add_force(self, force: Vec3) -> None:
        self._force.x += force.x
        self._force.y += force.y
        self._force.z += force.z

    def add_impulse(self, impulse: Vec3) -> None:
        inverse_mass = self.inverse_mass
        self.velocity.x += impulse.x * inverse_mass
        self.velocity.y += impulse.y * inverse_mass
        self.velocity.z += impulse.z * inverse_mass

    def add_torque(self, torque: Vec3) -> None:
        self._torque.x += torque.x
        self._torque.y += torque.y
        self._torque.z += torque.z

    def add_angular_impulse(self, impulse: Vec3) -> None:
        inverse_mass = self.inverse_mass
        self.angular_velocity.x += impulse.x * inverse_mass
        self.angular_velocity.y += impulse.y * inverse_mass
        self.angular_velocity.z += impulse.z * inverse_mass

    def clear_accumulators(self) -> None:
        self._force = Vec3()
        self._torque = Vec3()

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "mass": self.mass,
            "use_gravity": self.use_gravity,
            "drag": self.drag,
            "angular_drag": self.angular_drag,
            "is_kinematic": self.is_kinematic,
            "velocity": self.velocity.to_list(),
            "angular_velocity": self.angular_velocity.to_list(),
            "freeze_position": self.freeze_position.to_list(),
            "freeze_rotation": self.freeze_rotation.to_list(),
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
        from p64.engine.shader import normalize_shader_id

        return MeshRenderer(
            enabled=enabled,
            mesh=str(data.get("mesh", "")),
            submesh=data.get("submesh"),
            material=data.get("material"),
            shader=normalize_shader_id(data.get("shader")),
            source_materials=list(data.get("source_materials", [])),
            material_slots=list(data.get("material_slots", [])),
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
            range=float(data.get("range", 12.0)),
            spot_angle=float(data.get("spot_angle", 45.0)),
            falloff=float(data.get("falloff", 2.0)),
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
    if kind == "SpawnPoint":
        return SpawnPoint(
            enabled=enabled,
            spawn_id=str(data.get("spawn_id", "default")),
            from_scene=str(data.get("from_scene", "")),
            is_default=bool(data.get("is_default", False)),
        )
    if kind == "Collider":
        return Collider(
            enabled=enabled,
            shape=str(data.get("shape", "box")),
            size=Vec3.from_value(data.get("size", [1.0, 1.0, 1.0])),
            radius=float(data.get("radius", 0.5)),
            center=Vec3.from_value(data.get("center", [0.0, 0.0, 0.0])),
            fit_to_mesh=bool(data.get("fit_to_mesh", False)),
            convex=bool(data.get("convex", False)),
            is_trigger=bool(data.get("is_trigger", False)),
            layer=str(data.get("layer", "Default")),
            mask=str(data.get("mask", "*")),
        )
    if kind == "CharacterController":
        return CharacterController(
            enabled=enabled,
            height=float(data.get("height", 1.8)),
            radius=float(data.get("radius", 0.35)),
            skin_width=float(data.get("skin_width", 0.05)),
            slope_limit=float(data.get("slope_limit", 45.0)),
            gravity=float(data.get("gravity", 18.0)),
            velocity=Vec3.from_value(data.get("velocity", [0.0, 0.0, 0.0])),
            grounded=bool(data.get("grounded", False)),
        )
    if kind == "EntityPhysics":
        return EntityPhysics(
            enabled=enabled,
            mass=float(data.get("mass", 1.0)),
            use_gravity=bool(data.get("use_gravity", True)),
            drag=float(data.get("drag", 0.0)),
            angular_drag=float(data.get("angular_drag", 0.0)),
            is_kinematic=bool(data.get("is_kinematic", False)),
            velocity=Vec3.from_value(data.get("velocity", [0.0, 0.0, 0.0])),
            angular_velocity=Vec3.from_value(data.get("angular_velocity", [0.0, 0.0, 0.0])),
            freeze_position=Vec3.from_value(data.get("freeze_position", [0.0, 0.0, 0.0])),
            freeze_rotation=Vec3.from_value(data.get("freeze_rotation", [0.0, 0.0, 0.0])),
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
