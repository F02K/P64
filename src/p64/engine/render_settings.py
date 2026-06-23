from __future__ import annotations

from typing import Any


def default_render_settings() -> dict[str, Any]:
    return {
        "internal_resolution": [320, 240],
        "texture_filter": "three_point",
        "color_levels": 32,
        "dithering": True,
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
    return {key: settings[key] for key in default_render_settings()}
