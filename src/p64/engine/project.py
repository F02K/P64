from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from p64.engine.components import Camera, Fog, Light
from p64.engine.entity import Entity
from p64.engine.files import DEFAULT_SCENE, PROJECT_FILE, alternate_scene_path, normalize_scene_path, project_file_for, project_root_from_path
from p64.engine.math import Vec3
from p64.engine.scene import Scene


@dataclass
class Project:
    root: Path
    name: str
    startup_scene: str = DEFAULT_SCENE
    render_settings: dict[str, Any] = field(default_factory=dict)

    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def scenes_dir(self) -> Path:
        return self.root / "scenes"

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def build_dir(self) -> Path:
        return self.root / "build"

    @classmethod
    def create(cls, root: Path, name: str | None = None) -> "Project":
        root = project_root_from_path(root)
        project = cls(
            root=root,
            name=name or root.name,
            render_settings={
                "internal_resolution": [320, 240],
                "texture_filter": "nearest",
                "color_levels": 32,
                "dithering": True,
                "fog": True,
            },
        )
        project.ensure_layout()
        scene = default_scene("main")
        scene.render_settings = dict(project.render_settings)
        scene.save(root / project.startup_scene)
        project.save()
        return project

    @classmethod
    def load(cls, root: Path) -> "Project":
        root = project_root_from_path(root)
        project_path = project_file_for(root)
        with project_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            root=root,
            name=str(data.get("name", root.name)),
            startup_scene=str(data.get("startup_scene", DEFAULT_SCENE)),
            render_settings=dict(data.get("render_settings", {})),
        )

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(exist_ok=True)
        self.scenes_dir.mkdir(exist_ok=True)
        self.scripts_dir.mkdir(exist_ok=True)
        self.build_dir.mkdir(exist_ok=True)

    def save(self) -> None:
        self.ensure_layout()
        self.startup_scene = normalize_scene_path(self.startup_scene)
        with self.project_file.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "name": self.name,
                    "startup_scene": normalize_scene_path(self.startup_scene),
                    "render_settings": self.render_settings,
                },
                handle,
                indent=2,
            )
            handle.write("\n")

    def load_startup_scene(self) -> Scene:
        scene_path = self.root / self.startup_scene
        if not scene_path.exists():
            alternate = self.root / alternate_scene_path(Path(self.startup_scene))
            if alternate.exists():
                scene_path = alternate
        return Scene.load(scene_path)

    def save_startup_scene(self, scene: Scene) -> None:
        self.startup_scene = normalize_scene_path(self.startup_scene)
        scene.save(self.root / self.startup_scene)


def default_scene(name: str) -> Scene:
    scene = Scene(name=name)

    camera = Entity("Camera")
    camera.transform.position = Vec3(0.0, 3.0, 8.0)
    camera.transform.rotation = Vec3(-18.0, 0.0, 0.0)
    camera.add_component(Camera(active=True))
    scene.add_entity(camera)

    sun = Entity("Sun")
    sun.transform.rotation = Vec3(-45.0, 35.0, 0.0)
    sun.add_component(Light(kind="directional", intensity=1.25))
    scene.add_entity(sun)

    fog = Entity("Fog")
    fog.add_component(Fog(near=18.0, far=85.0))
    scene.add_entity(fog)
    return scene
