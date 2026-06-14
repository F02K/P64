from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.engine.scene_manager import SceneManager
from p64.engine.audio import AudioSystem, ensure_audio_clips_for_assets
from p64.engine.collision import CollisionWorld
from p64.engine.input import InputState
from p64.engine.scripting import ScriptContext, ScriptManager


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
        self.script_manager = ScriptManager(
            [project.scripts_dir, project.root / "scripts"],
            import_dirs=[project.project_api_dir],
        )
        self.input = InputState()
        self.audio = AudioSystem(project, logger=self._record_runtime_error)
        self.time = 0.0
        self._scripts: list[RuntimeScript] = []
        self._started = False

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
        self._started = False

    def tick(self, dt: float) -> list[str]:
        dt = max(0.0, dt)
        self.input.begin_frame()
        errors = self.start()
        try:
            self.time += dt
            for runtime_script in list(self._scripts):
                instance = runtime_script.instance
                instance.scene = self.scene
                instance.project = self.project
                instance.scene_manager = self.scene_manager
                instance.input = self.input
                instance.time = self.time
                if hasattr(instance, "on_update"):
                    try:
                        instance.on_update(dt)
                    except Exception as exc:  # pragma: no cover - exact user code is unknowable
                        errors.append(f"{runtime_script.entry_name}.on_update failed: {exc}")
            if self.scene_manager.apply_queued_scene():
                self.audio.stop_all()
                self.audio.start_scene(self.scene)
                errors.extend(self._drain_pending_errors())
                errors.extend(self._instantiate_scene_scripts())
            CollisionWorld(self.scene, self.project).step_physics(dt)
            self.audio.tick(self.scene, dt)
            errors.extend(self._drain_pending_errors())
            self.errors.extend(errors)
            return errors
        finally:
            self.input.end_frame()

    def _instantiate_scene_scripts(self) -> list[str]:
        self._scripts = []
        errors: list[str] = []
        context = ScriptContext(scene=self.scene, project=self.project, scene_manager=self.scene_manager, time=self.time, input=self.input)
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

    def _record_runtime_error(self, message: str) -> None:
        self._pending_errors.append(message)

    def _drain_pending_errors(self) -> list[str]:
        errors = list(self._pending_errors)
        self._pending_errors.clear()
        return errors
