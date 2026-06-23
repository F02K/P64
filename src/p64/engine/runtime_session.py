from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.engine.scene_manager import SceneManager
from p64.engine.audio import AudioSystem, ensure_audio_clips_for_assets
from p64.engine.collision import CollisionWorld
from p64.engine.entity import entity_effectively_active
from p64.engine.input import InputState
from p64.engine.scripting import ScriptContext, ScriptManager
from p64.engine.ui import UIEventSystem


MAX_PHYSICS_DT = 0.05


@dataclass
class RuntimeScript:
    entry_name: str
    instance: Any


class RuntimeSession:
    def __init__(self, project: Project, scene: Scene | None = None) -> None:
        self.project = project
        self.errors: list[str] = []
        self._pending_errors: list[str] = []
        try:
            ensure_audio_clips_for_assets(project)
        except Exception as exc:
            self._record_runtime_error(f"Audio import failed: {exc}")
        self.scene_manager = SceneManager(project, scene)
        self.collision_world = CollisionWorld(self.scene_manager.current_scene, project)
        self.script_manager = ScriptManager(
            [project.scripts_dir, project.root / "scripts"],
            import_dirs=[project.project_api_dir],
        )
        self.input = InputState()
        self.ui = UIEventSystem(self._dispatch_ui_callback)
        self.audio = AudioSystem(project, logger=self._record_runtime_error)
        self.time = 0.0
        self._scripts: list[RuntimeScript] = []
        self._started = False
        self.profiler_recorder: Any | None = None

    @property
    def scene(self) -> Scene:
        return self.scene_manager.current_scene

    def start(self) -> list[str]:
        errors = self._drain_pending_errors()
        if self._started:
            return errors
        self._started = True
        self.audio.start_scene(self.scene)
        errors.extend(self._drain_pending_errors())
        errors.extend(self._instantiate_scene_scripts())
        return errors

    def stop(self) -> None:
        self.audio.stop_all()
        self._scripts = []
        self.ui.reset(self.scene)
        self._started = False

    def tick(self, dt: float) -> list[str]:
        dt = max(0.0, dt)
        self.input.begin_frame()
        profiler = self.profiler_recorder
        with _profiler_section(profiler, "runtime start"):
            errors = self.start()
        try:
            with _profiler_section(profiler, "runtime total"):
                self.time += dt
                with _profiler_section(profiler, "runtime ui"):
                    errors.extend(self.ui.process(self.scene, self.input, dt))
                _profiler_add_count(profiler, "runtime scripts", len(self._scripts))
                with _profiler_section(profiler, "runtime scripts"):
                    for runtime_script in list(self._scripts):
                        instance = runtime_script.instance
                        instance.scene = self.scene
                        instance.project = self.project
                        instance.scene_manager = self.scene_manager
                        instance.collision_world = self.collision_world
                        instance.input = self.input
                        instance.time = self.time
                        if not entity_effectively_active(instance.entity):
                            continue
                        if hasattr(instance, "on_update"):
                            try:
                                if _profiler_sample_details(profiler):
                                    with _profiler_section(profiler, "script", runtime_script.entry_name):
                                        instance.on_update(dt)
                                else:
                                    instance.on_update(dt)
                            except Exception as exc:  # pragma: no cover - exact user code is unknowable
                                errors.append(f"{runtime_script.entry_name}.on_update failed: {exc}")
                with _profiler_section(profiler, "runtime scene switch"):
                    if self.scene_manager.apply_queued_scene():
                        self.audio.stop_all()
                        self.ui.reset()
                        self.collision_world = CollisionWorld(self.scene, self.project)
                        self.audio.start_scene(self.scene)
                        errors.extend(self._drain_pending_errors())
                        errors.extend(self._instantiate_scene_scripts())
                _profiler_add_count(profiler, "physics bodies", self.collision_world.physics_body_count())
                with _profiler_section(profiler, "runtime physics"):
                    self.collision_world.step_physics(min(dt, MAX_PHYSICS_DT))
                with _profiler_section(profiler, "runtime audio"):
                    self.audio.tick(self.scene, dt)
                _profiler_add_count(profiler, "audio sources", self.audio.source_count())
                _profiler_add_count(profiler, "audio channels", self.audio.channel_count())
            errors.extend(self._drain_pending_errors())
            self.errors.extend(errors)
            return errors
        finally:
            self.input.end_frame()

    def _instantiate_scene_scripts(self) -> list[str]:
        self._scripts = []
        errors: list[str] = []
        context = ScriptContext(
            scene=self.scene,
            project=self.project,
            scene_manager=self.scene_manager,
            collision_world=self.collision_world,
            time=self.time,
            input=self.input,
        )
        for _entity, component in self.scene.script_components():
            for entry, instance, error in self.script_manager.instantiate_component(_entity, component, context):
                entry_name = f"{entry.script}:{entry.class_name}"
                if error:
                    errors.append(error)
                    continue
                if hasattr(instance, "on_start"):
                    try:
                        instance.on_start()
                    except Exception as exc:  # pragma: no cover - exact user code is unknowable
                        errors.append(f"{entry_name}.on_start failed: {exc}")
                self._scripts.append(RuntimeScript(entry_name=entry_name, instance=instance))
        return errors

    def _dispatch_ui_callback(self, entity: Any, method: str, args: tuple[Any, ...]) -> list[str]:
        errors: list[str] = []
        for runtime_script in self._scripts:
            instance = runtime_script.instance
            if instance.entity is not entity or not hasattr(instance, method):
                continue
            try:
                getattr(instance, method)(*args)
            except Exception as exc:  # pragma: no cover - exact user code is unknowable
                errors.append(f"{runtime_script.entry_name}.{method} failed: {exc}")
        return errors

    def _record_runtime_error(self, message: str) -> None:
        self._pending_errors.append(message)

    def _drain_pending_errors(self) -> list[str]:
        errors = list(self._pending_errors)
        self._pending_errors.clear()
        return errors


def _profiler_section(profiler: Any | None, name: str, detail: str = "") -> Any:
    if profiler is None:
        return nullcontext()
    try:
        return profiler.section(name, str(detail or ""))
    except Exception:
        return nullcontext()


def _profiler_sample_details(profiler: Any | None) -> bool:
    if profiler is None:
        return False
    try:
        return bool(profiler.sample_details())
    except Exception:
        return False


def _profiler_add_count(profiler: Any | None, name: str, amount: int) -> None:
    if profiler is None:
        return
    try:
        profiler.add_count(name, amount)
    except Exception:
        return
