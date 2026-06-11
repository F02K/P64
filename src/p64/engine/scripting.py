from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from dataclasses import dataclass, field
from typing import Any

from p64.engine.components import ScriptComponent, ScriptEntry


@dataclass
class InputState:
    keys_down: set[str] = field(default_factory=set)

    def is_key_down(self, key: str) -> bool:
        return key in self.keys_down


@dataclass
class ScriptContext:
    scene: Any
    time: float = 0.0
    input: InputState = field(default_factory=InputState)


class UserScript:
    entity: Any
    transform: Any

    def __init__(self, entity: Any, context: ScriptContext | None = None, **properties: Any) -> None:
        self.entity = entity
        self.transform = entity.transform
        self.scene = context.scene if context else None
        self.input = context.input if context else InputState()
        self.time = context.time if context else 0.0
        for key, value in properties.items():
            setattr(self, key, value)


class ScriptManager:
    def __init__(self, scripts_dir: Path) -> None:
        self.scripts_dir = scripts_dir
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
        script_path = (self.scripts_dir / script).resolve()
        if script_path.suffix != ".py":
            script_path = script_path.with_suffix(".py")
        if not script_path.exists():
            return None, f"Script file not found: {script_path}"
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
