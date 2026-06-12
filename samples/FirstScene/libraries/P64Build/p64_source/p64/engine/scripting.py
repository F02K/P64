from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from dataclasses import dataclass, field
from typing import Any

from p64.engine.collision import CollisionWorld
from p64.engine.components import CharacterController, EntityPhysics, ScriptComponent, ScriptEntry
from p64.engine.input import InputState
from p64.engine.math import Vec3


@dataclass
class ScriptContext:
    scene: Any
    project: Any | None = None
    scene_manager: Any | None = None
    time: float = 0.0
    input: InputState = field(default_factory=InputState)


class UserScript:
    entity: Any
    transform: Any

    def __init__(self, entity: Any, context: ScriptContext | None = None, **properties: Any) -> None:
        self.entity = entity
        self.transform = entity.transform
        self.scene = context.scene if context else None
        self.project = context.project if context else None
        self.scene_manager = context.scene_manager if context else None
        self.input = context.input if context else InputState()
        self.time = context.time if context else 0.0
        self.character_controller = self._find_character_controller()
        self.entity_physics = self._find_entity_physics()
        if self.character_controller is not None and context is not None:
            self.character_controller.bind_runtime(entity, context.scene, context.project)
        for key, value in properties.items():
            setattr(self, key, value)

    def persistent(self) -> None:
        self.entity.persistent = True

    def move_character(self, motion: Vec3, dt: float) -> Vec3:
        if self.scene is None or self.character_controller is None:
            return Vec3()
        return CollisionWorld(self.scene, self.project).move_character(self.entity, self.character_controller, motion, dt)

    def _find_character_controller(self) -> CharacterController | None:
        for component in getattr(self.entity, "components", []):
            if isinstance(component, CharacterController):
                return component
        return None

    def _find_entity_physics(self) -> EntityPhysics | None:
        for component in getattr(self.entity, "components", []):
            if isinstance(component, EntityPhysics):
                return component
        return None


class ScriptManager:
    def __init__(self, scripts_dir: Path | list[Path]) -> None:
        self.scripts_dirs = scripts_dir if isinstance(scripts_dir, list) else [scripts_dir]
        self._modules: dict[str, ModuleType] = {}

    def instantiate(
        self,
        entity: Any,
        entry: ScriptEntry,
        context: ScriptContext | None = None,
    ) -> tuple[Any | None, str | None]:
        module, error = self._load_module(entry.script)
        if error:
            return None, error
        if module is None:
            return None, f"Script module {entry.script!r} could not be loaded."
        cls = getattr(module, entry.class_name, None)
        if cls is None:
            return None, f"Class {entry.class_name!r} not found in {entry.script!r}."
        try:
            return cls(entity=entity, context=context), None
        except Exception as exc:  # pragma: no cover - user constructors vary
            return None, f"Could not instantiate {entry.class_name!r}: {exc}"

    def instantiate_component(
        self,
        entity: Any,
        component: ScriptComponent,
        context: ScriptContext | None = None,
    ) -> list[tuple[ScriptEntry, Any | None, str | None]]:
        instances: list[tuple[ScriptEntry, Any | None, str | None]] = []
        for entry in component.scripts:
            if not entry.enabled:
                continue
            instances.append((entry, *self.instantiate(entity, entry, context)))
        return instances

    def _load_module(self, script: str) -> tuple[ModuleType | None, str | None]:
        script_path = self._resolve_script_path(script)
        if script_path is None:
            if not script.strip():
                return None, "Script entry has no script file"
            searched = ", ".join(str(_script_candidate(folder, script)) for folder in self.scripts_dirs)
            return None, f"Script file not found: {searched}"
        cache_key = str(script_path)
        if cache_key in self._modules:
            return self._modules[cache_key], None

        module_name = f"p64_user_{script_path.stem}_{abs(hash(cache_key))}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            return None, f"Could not create import spec for {script_path}"
        module = importlib.util.module_from_spec(spec)
        module.UserScript = UserScript
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            return None, f"Could not load {script_path}: {exc}"
        self._modules[cache_key] = module
        return module, None

    def _resolve_script_path(self, script: str) -> Path | None:
        if not script.strip():
            return None
        relative = Path(script)
        if relative.name == "":
            return None
        if relative.suffix != ".py":
            relative = relative.with_suffix(".py")
        for folder in self.scripts_dirs:
            path = (folder / relative).resolve()
            if path.exists():
                return path
        return None


def _script_candidate(folder: Path, script: str) -> Path:
    relative = Path(script)
    if relative.name == "":
        return folder
    if relative.suffix != ".py":
        relative = relative.with_suffix(".py")
    return folder / relative
