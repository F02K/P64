from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LIGHTING_SUFFIX = ".lightingp64"

SKYBOX_TOP_COLOR = [0.22, 0.48, 0.86]
SKYBOX_HORIZON_COLOR = [0.66, 0.82, 0.95]
SKYBOX_CLOUD_COLOR = [1.0, 0.96, 0.86]
FOG_COLOR = [0.46, 0.58, 0.72]


def default_lighting_settings() -> dict[str, Any]:
    return {
        "skybox_enabled": True,
        "skybox_top_color": list(SKYBOX_TOP_COLOR),
        "skybox_horizon_color": list(SKYBOX_HORIZON_COLOR),
        "skybox_cloud_color": list(SKYBOX_CLOUD_COLOR),
        "skybox_cloud_coverage": 0.45,
        "skybox_cloud_scale": 3.0,
        "skybox_cloud_height": 80.0,
        "skybox_cloud_softness": 0.08,
        "fog_enabled": True,
        "fog_color": list(FOG_COLOR),
        "fog_near": 18.0,
        "fog_far": 85.0,
        "fog_density": 0.0,
    }


def clamp_lighting_settings(settings: dict[str, Any]) -> dict[str, Any]:
    settings["skybox_enabled"] = bool(settings.get("skybox_enabled", True))
    settings["skybox_top_color"] = _color3(settings.get("skybox_top_color"), SKYBOX_TOP_COLOR)
    settings["skybox_horizon_color"] = _color3(settings.get("skybox_horizon_color"), SKYBOX_HORIZON_COLOR)
    settings["skybox_cloud_color"] = _color3(settings.get("skybox_cloud_color"), SKYBOX_CLOUD_COLOR)
    settings["skybox_cloud_coverage"] = max(0.0, min(1.0, float(settings.get("skybox_cloud_coverage", 0.45))))
    settings["skybox_cloud_scale"] = max(0.1, min(24.0, float(settings.get("skybox_cloud_scale", 3.0))))
    settings["skybox_cloud_height"] = max(0.1, float(settings.get("skybox_cloud_height", 80.0)))
    settings["skybox_cloud_softness"] = max(0.0, min(1.0, float(settings.get("skybox_cloud_softness", 0.08))))
    settings["fog_enabled"] = bool(settings.get("fog_enabled", settings.get("fog", True)))
    settings["fog_color"] = _color3(settings.get("fog_color"), FOG_COLOR)
    settings["fog_near"] = max(0.0, float(settings.get("fog_near", 18.0)))
    settings["fog_far"] = max(settings["fog_near"] + 0.001, float(settings.get("fog_far", 85.0)))
    settings["fog_density"] = max(0.0, min(1.0, float(settings.get("fog_density", 0.0))))
    return {key: settings[key] for key in default_lighting_settings()}


def lighting_path_for_scene(scene_path: Path) -> Path:
    return scene_path.with_suffix(LIGHTING_SUFFIX)


def scene_path_for_lighting(lighting_path: Path) -> Path:
    return lighting_path.with_suffix(".scenep64")


def load_lighting_settings(path: Path, legacy: dict[str, Any] | None = None) -> dict[str, Any]:
    values = {**default_lighting_settings(), **_legacy_lighting_values(legacy or {})}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                values.update(loaded)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return clamp_lighting_settings(values)


def save_lighting_settings(path: Path, settings: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clamp_lighting_settings(dict(settings)), indent=2) + "\n", encoding="utf-8")


def _legacy_lighting_values(settings: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: settings[key]
        for key in default_lighting_settings()
        if key in settings
    }
    if "fog" in settings and "fog_enabled" not in result:
        result["fog_enabled"] = settings["fog"]
    return result


def _color3(value: Any, default: list[float]) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        value = default
    try:
        return [max(0.0, min(1.0, float(value[0]))), max(0.0, min(1.0, float(value[1]))), max(0.0, min(1.0, float(value[2])))]
    except (TypeError, ValueError):
        return list(default)
