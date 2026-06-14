from __future__ import annotations

from typing import Any


SKYBOX_TOP_COLOR = [0.22, 0.48, 0.86]
SKYBOX_HORIZON_COLOR = [0.66, 0.82, 0.95]
SKYBOX_CLOUD_COLOR = [1.0, 0.96, 0.86]


def default_render_settings() -> dict[str, Any]:
    return {
        "internal_resolution": [320, 240],
        "texture_filter": "three_point",
        "color_levels": 32,
        "dithering": True,
        "fog": True,
        "skybox_enabled": True,
        "skybox_top_color": list(SKYBOX_TOP_COLOR),
        "skybox_horizon_color": list(SKYBOX_HORIZON_COLOR),
        "skybox_cloud_color": list(SKYBOX_CLOUD_COLOR),
        "skybox_cloud_coverage": 0.45,
        "skybox_cloud_scale": 3.0,
        "skybox_cloud_height": 80.0,
        "skybox_cloud_softness": 0.08,
    }


def clamp_render_settings(settings: dict[str, Any]) -> dict[str, Any]:
    resolution = list(settings.get("internal_resolution", [320, 240]))
    if len(resolution) < 2:
        resolution = [320, 240]
    settings["internal_resolution"] = [max(1, int(resolution[0])), max(1, int(resolution[1]))]
    settings["color_levels"] = max(2, int(settings.get("color_levels", 32)))
    filter_value = str(settings.get("texture_filter", "three_point"))
    settings["texture_filter"] = filter_value if filter_value in {"nearest", "linear", "three_point"} else "three_point"
    settings["dithering"] = bool(settings.get("dithering", True))
    settings["fog"] = bool(settings.get("fog", True))
    settings["skybox_enabled"] = bool(settings.get("skybox_enabled", True))
    settings["skybox_top_color"] = _color3(settings.get("skybox_top_color"), SKYBOX_TOP_COLOR)
    settings["skybox_horizon_color"] = _color3(settings.get("skybox_horizon_color"), SKYBOX_HORIZON_COLOR)
    settings["skybox_cloud_color"] = _color3(settings.get("skybox_cloud_color"), SKYBOX_CLOUD_COLOR)
    settings["skybox_cloud_coverage"] = max(0.0, min(1.0, float(settings.get("skybox_cloud_coverage", 0.45))))
    settings["skybox_cloud_scale"] = max(0.1, min(24.0, float(settings.get("skybox_cloud_scale", 3.0))))
    settings["skybox_cloud_height"] = max(0.1, float(settings.get("skybox_cloud_height", 80.0)))
    settings["skybox_cloud_softness"] = max(0.0, min(1.0, float(settings.get("skybox_cloud_softness", 0.08))))
    return settings


def _color3(value: Any, default: list[float]) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        value = default
    try:
        return [max(0.0, min(1.0, float(value[0]))), max(0.0, min(1.0, float(value[1]))), max(0.0, min(1.0, float(value[2])))]
    except (TypeError, ValueError):
        return list(default)
