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
class RectTransform:
    anchor: str = "center"
    offset: Vec3 = field(default_factory=Vec3)
    size: Vec3 = field(default_factory=lambda: Vec3(160.0, 48.0, 0.0))
    pivot: Vec3 = field(default_factory=lambda: Vec3(0.5, 0.5, 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor": self.anchor,
            "offset": self.offset.to_list(),
            "size": self.size.to_list(),
            "pivot": self.pivot.to_list(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RectTransform":
        data = data or {}
        return cls(
            anchor=str(data.get("anchor", "center")),
            offset=Vec3.from_value(data.get("offset", [0.0, 0.0, 0.0])),
            size=Vec3.from_value(data.get("size", [160.0, 48.0, 0.0])),
            pivot=Vec3.from_value(data.get("pivot", [0.5, 0.5, 0.0])),
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
class SpriteRenderer(Component):
    texture: str = ""
    material: str | None = None
    color: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    alpha: float = 1.0
    size: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    pivot: Vec3 = field(default_factory=lambda: Vec3(0.5, 0.5, 0.0))
    billboard: str = "camera"
    sorting_layer: str = "Default"
    sorting_order: int = 0
    flipbook_columns: int = 1
    flipbook_rows: int = 1
    flipbook_fps: float = 0.0
    flipbook_start: int = 0
    flipbook_end: int = 0
    type_name: str = "SpriteRenderer"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "texture": self.texture,
            "material": self.material,
            "color": self.color.to_list(),
            "alpha": self.alpha,
            "size": self.size.to_list(),
            "pivot": self.pivot.to_list(),
            "billboard": self.billboard,
            "sorting_layer": self.sorting_layer,
            "sorting_order": self.sorting_order,
            "flipbook_columns": self.flipbook_columns,
            "flipbook_rows": self.flipbook_rows,
            "flipbook_fps": self.flipbook_fps,
            "flipbook_start": self.flipbook_start,
            "flipbook_end": self.flipbook_end,
        })
        return data


@dataclass(slots=True)
class Canvas(Component):
    sort_order: int = 0
    reference_resolution: Vec3 = field(default_factory=lambda: Vec3(1280.0, 720.0, 0.0))
    resolution_mode: str = "auto"
    type_name: str = "Canvas"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "sort_order": self.sort_order,
            "reference_resolution": self.reference_resolution.to_list(),
            "resolution_mode": self.resolution_mode,
        })
        return data


@dataclass(slots=True)
class UIImage(Component):
    texture: str = ""
    material: str | None = None
    color: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    alpha: float = 1.0
    size: Vec3 = field(default_factory=lambda: Vec3(128.0, 128.0, 0.0))
    anchor: str = "center"
    offset: Vec3 = field(default_factory=Vec3)
    pivot: Vec3 = field(default_factory=lambda: Vec3(0.5, 0.5, 0.0))
    fill_mode: str = "simple"
    flipbook_columns: int = 1
    flipbook_rows: int = 1
    flipbook_fps: float = 0.0
    flipbook_start: int = 0
    flipbook_end: int = 0
    type_name: str = "UIImage"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "texture": self.texture,
            "material": self.material,
            "color": self.color.to_list(),
            "alpha": self.alpha,
            "size": self.size.to_list(),
            "anchor": self.anchor,
            "offset": self.offset.to_list(),
            "pivot": self.pivot.to_list(),
            "fill_mode": self.fill_mode,
            "flipbook_columns": self.flipbook_columns,
            "flipbook_rows": self.flipbook_rows,
            "flipbook_fps": self.flipbook_fps,
            "flipbook_start": self.flipbook_start,
            "flipbook_end": self.flipbook_end,
        })
        return data


@dataclass(slots=True)
class UIText(Component):
    text: str = "Text"
    font_source: str = "system"
    font_family: str = "System"
    bitmap_font: str = ""
    font_size: float = 24.0
    color: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    alpha: float = 1.0
    alignment: str = "center"
    anchor: str = "center"
    offset: Vec3 = field(default_factory=Vec3)
    pivot: Vec3 = field(default_factory=lambda: Vec3(0.5, 0.5, 0.0))
    type_name: str = "UIText"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "text": self.text,
            "font_source": self.font_source,
            "font_family": self.font_family,
            "bitmap_font": self.bitmap_font,
            "font_size": self.font_size,
            "color": self.color.to_list(),
            "alpha": self.alpha,
            "alignment": self.alignment,
            "anchor": self.anchor,
            "offset": self.offset.to_list(),
            "pivot": self.pivot.to_list(),
        })
        return data


@dataclass(slots=True)
class ParticleEmitter(Component):
    material: str | None = None
    texture: str = ""
    max_particles: int = 128
    rate: float = 12.0
    burst: int = 0
    lifetime: float = 1.0
    start_size: float = 0.25
    start_color: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    start_alpha: float = 1.0
    start_velocity: Vec3 = field(default_factory=lambda: Vec3(0.0, 1.0, 0.0))
    gravity: Vec3 = field(default_factory=Vec3)
    local_space: bool = False
    looping: bool = True
    play_on_awake: bool = True
    blend_mode: str = "alpha"
    flipbook_columns: int = 1
    flipbook_rows: int = 1
    flipbook_fps: float = 0.0
    flipbook_start: int = 0
    flipbook_end: int = 0
    _runtime_playing: bool = field(default=True, repr=False, compare=False)
    _runtime_particles: list[dict[str, Any]] = field(default_factory=list, repr=False, compare=False)
    _runtime_accumulator: float = field(default=0.0, repr=False, compare=False)
    _runtime_burst_done: bool = field(default=False, repr=False, compare=False)
    type_name: str = "ParticleEmitter"

    def play(self) -> None:
        self._runtime_playing = True

    def stop(self) -> None:
        self._runtime_playing = False
        self._runtime_accumulator = 0.0

    def emit(self, count: int) -> None:
        for _index in range(max(0, int(count))):
            if len(self._runtime_particles) >= max(0, int(self.max_particles)):
                return
            self._runtime_particles.append({
                "age": 0.0,
                "lifetime": max(0.001, float(self.lifetime)),
                "position": Vec3(),
                "velocity": Vec3(self.start_velocity.x, self.start_velocity.y, self.start_velocity.z),
                "size": max(0.001, float(self.start_size)),
            })

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "material": self.material,
            "texture": self.texture,
            "max_particles": self.max_particles,
            "rate": self.rate,
            "burst": self.burst,
            "lifetime": self.lifetime,
            "start_size": self.start_size,
            "start_color": self.start_color.to_list(),
            "start_alpha": self.start_alpha,
            "start_velocity": self.start_velocity.to_list(),
            "gravity": self.gravity.to_list(),
            "local_space": self.local_space,
            "looping": self.looping,
            "play_on_awake": self.play_on_awake,
            "blend_mode": self.blend_mode,
            "flipbook_columns": self.flipbook_columns,
            "flipbook_rows": self.flipbook_rows,
            "flipbook_fps": self.flipbook_fps,
            "flipbook_start": self.flipbook_start,
            "flipbook_end": self.flipbook_end,
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
class AudioListener(Component):
    active: bool = True
    type_name: str = "AudioListener"

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({"active": self.active})
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
class AudioSource(Component):
    clip: str = ""
    volume: float = 1.0
    pitch: float = 1.0
    loop: bool = False
    play_on_awake: bool = True
    spatial: bool = True
    min_distance: float = 1.0
    max_distance: float = 25.0
    _runtime_audio: Any | None = field(default=None, repr=False, compare=False)
    _runtime_entity: Any | None = field(default=None, repr=False, compare=False)
    type_name: str = "AudioSource"

    def bind_runtime(self, audio: Any, entity: Any) -> None:
        self._runtime_audio = audio
        self._runtime_entity = entity

    def play(self) -> bool:
        if self._runtime_audio is None or self._runtime_entity is None:
            return False
        return bool(self._runtime_audio.play(self._runtime_entity, self))

    def stop(self) -> None:
        if self._runtime_audio is not None:
            self._runtime_audio.stop(self)

    def pause(self) -> None:
        if self._runtime_audio is not None:
            self._runtime_audio.pause(self)

    def resume(self) -> None:
        if self._runtime_audio is not None:
            self._runtime_audio.resume(self)

    def to_dict(self) -> dict[str, Any]:
        data = Component.to_dict(self)
        data.update({
            "clip": self.clip,
            "volume": self.volume,
            "pitch": self.pitch,
            "loop": self.loop,
            "play_on_awake": self.play_on_awake,
            "spatial": self.spatial,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
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
    _runtime_collision_world: Any | None = field(default=None, repr=False, compare=False)
    type_name: str = "CharacterController"

    @property
    def is_grounded(self) -> bool:
        return self.grounded

    def bind_runtime(self, entity: Any, scene: Any, project: Any | None = None, collision_world: Any | None = None) -> None:
        self._runtime_entity = entity
        self._runtime_scene = scene
        self._runtime_project = project
        self._runtime_collision_world = collision_world

    def move(self, motion: Vec3, dt: float) -> Vec3:
        if self._runtime_entity is None or self._runtime_scene is None:
            return Vec3()
        from p64.engine.collision import CollisionWorld

        world = self._runtime_collision_world or CollisionWorld(self._runtime_scene, self._runtime_project)
        return world.move_character(self._runtime_entity, self, motion, dt)

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
    if kind == "SpriteRenderer":
        return SpriteRenderer(
            enabled=enabled,
            texture=str(data.get("texture", "")),
            material=data.get("material"),
            color=Vec3.from_value(data.get("color", [1.0, 1.0, 1.0])),
            alpha=float(data.get("alpha", 1.0)),
            size=Vec3.from_value(data.get("size", [1.0, 1.0, 1.0])),
            pivot=Vec3.from_value(data.get("pivot", [0.5, 0.5, 0.0])),
            billboard=str(data.get("billboard", "camera")),
            sorting_layer=str(data.get("sorting_layer", "Default")),
            sorting_order=int(data.get("sorting_order", 0)),
            flipbook_columns=int(data.get("flipbook_columns", 1)),
            flipbook_rows=int(data.get("flipbook_rows", 1)),
            flipbook_fps=float(data.get("flipbook_fps", 0.0)),
            flipbook_start=int(data.get("flipbook_start", 0)),
            flipbook_end=int(data.get("flipbook_end", 0)),
        )
    if kind == "Canvas":
        return Canvas(
            enabled=enabled,
            sort_order=int(data.get("sort_order", 0)),
            reference_resolution=Vec3.from_value(data.get("reference_resolution", [1280.0, 720.0, 0.0])),
            resolution_mode=_choice(data.get("resolution_mode", "auto"), {"auto", "fixed"}, "auto"),
        )
    if kind == "UIImage":
        return UIImage(
            enabled=enabled,
            texture=str(data.get("texture", "")),
            material=data.get("material"),
            color=Vec3.from_value(data.get("color", [1.0, 1.0, 1.0])),
            alpha=float(data.get("alpha", 1.0)),
            size=Vec3.from_value(data.get("size", [128.0, 128.0, 0.0])),
            anchor=str(data.get("anchor", "center")),
            offset=Vec3.from_value(data.get("offset", [0.0, 0.0, 0.0])),
            pivot=Vec3.from_value(data.get("pivot", [0.5, 0.5, 0.0])),
            fill_mode=str(data.get("fill_mode", "simple")),
            flipbook_columns=int(data.get("flipbook_columns", 1)),
            flipbook_rows=int(data.get("flipbook_rows", 1)),
            flipbook_fps=float(data.get("flipbook_fps", 0.0)),
            flipbook_start=int(data.get("flipbook_start", 0)),
            flipbook_end=int(data.get("flipbook_end", 0)),
        )
    if kind == "UIText":
        return UIText(
            enabled=enabled,
            text=str(data.get("text", "Text")),
            font_source=_choice(data.get("font_source", "system"), {"system", "asset"}, "system"),
            font_family=str(data.get("font_family", "System")),
            bitmap_font=str(data.get("bitmap_font", "")),
            font_size=float(data.get("font_size", 24.0)),
            color=Vec3.from_value(data.get("color", [1.0, 1.0, 1.0])),
            alpha=float(data.get("alpha", 1.0)),
            alignment=str(data.get("alignment", "center")),
            anchor=str(data.get("anchor", "center")),
            offset=Vec3.from_value(data.get("offset", [0.0, 0.0, 0.0])),
            pivot=Vec3.from_value(data.get("pivot", [0.5, 0.5, 0.0])),
        )
    if kind == "ParticleEmitter":
        emitter = ParticleEmitter(
            enabled=enabled,
            material=data.get("material"),
            texture=str(data.get("texture", "")),
            max_particles=int(data.get("max_particles", 128)),
            rate=float(data.get("rate", 12.0)),
            burst=int(data.get("burst", 0)),
            lifetime=float(data.get("lifetime", 1.0)),
            start_size=float(data.get("start_size", 0.25)),
            start_color=Vec3.from_value(data.get("start_color", [1.0, 1.0, 1.0])),
            start_alpha=float(data.get("start_alpha", 1.0)),
            start_velocity=Vec3.from_value(data.get("start_velocity", [0.0, 1.0, 0.0])),
            gravity=Vec3.from_value(data.get("gravity", [0.0, 0.0, 0.0])),
            local_space=bool(data.get("local_space", False)),
            looping=bool(data.get("looping", True)),
            play_on_awake=bool(data.get("play_on_awake", True)),
            blend_mode=str(data.get("blend_mode", "alpha")),
            flipbook_columns=int(data.get("flipbook_columns", 1)),
            flipbook_rows=int(data.get("flipbook_rows", 1)),
            flipbook_fps=float(data.get("flipbook_fps", 0.0)),
            flipbook_start=int(data.get("flipbook_start", 0)),
            flipbook_end=int(data.get("flipbook_end", 0)),
        )
        emitter._runtime_playing = emitter.play_on_awake
        return emitter
    if kind == "Camera":
        return Camera(
            enabled=enabled,
            fov=float(data.get("fov", 60.0)),
            near=float(data.get("near", 0.1)),
            far=float(data.get("far", 500.0)),
            active=bool(data.get("active", False)),
        )
    if kind == "AudioListener":
        return AudioListener(
            enabled=enabled,
            active=bool(data.get("active", True)),
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
    if kind == "AudioSource":
        return AudioSource(
            enabled=enabled,
            clip=str(data.get("clip", "")),
            volume=float(data.get("volume", 1.0)),
            pitch=float(data.get("pitch", 1.0)),
            loop=bool(data.get("loop", False)),
            play_on_awake=bool(data.get("play_on_awake", True)),
            spatial=bool(data.get("spatial", True)),
            min_distance=float(data.get("min_distance", 1.0)),
            max_distance=float(data.get("max_distance", 25.0)),
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


def _choice(value: object, allowed: set[str], default: str) -> str:
    text = str(value or default)
    return text if text in allowed else default
