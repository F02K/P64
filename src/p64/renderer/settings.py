from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RenderSettings:
    internal_resolution: tuple[int, int] = (320, 240)
    texture_filter: str = "three_point"
    color_levels: int = 32
    dithering: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RenderSettings":
        data = data or {}
        resolution = data.get("internal_resolution", [320, 240])
        return cls(
            internal_resolution=(int(resolution[0]), int(resolution[1])),
            texture_filter=str(data.get("texture_filter", "three_point")),
            color_levels=int(data.get("color_levels", 32)),
            dithering=bool(data.get("dithering", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "internal_resolution": list(self.internal_resolution),
            "texture_filter": self.texture_filter,
            "color_levels": self.color_levels,
            "dithering": self.dithering,
        }
