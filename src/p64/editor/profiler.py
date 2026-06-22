from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from threading import Event, Lock, Thread, local
from time import perf_counter, time
from typing import Any


@dataclass(frozen=True)
class ProfilerSection:
    name: str
    duration_ms: float
    detail: str = ""


@dataclass
class ProfilerFrame:
    view_mode: str
    started_at: float
    duration_ms: float = 0.0
    sections: list[ProfilerSection] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    _started_perf: float = field(default=0.0, repr=False, compare=False)


@dataclass(frozen=True)
class ProfilerStat:
    name: str
    last_ms: float
    average_ms: float
    min_ms: float
    max_ms: float
    samples: int


@dataclass(frozen=True)
class ProfilerSnapshot:
    frames: int = 0
    scene_fps: float = 0.0
    game_fps: float = 0.0
    frame_ms: float = 0.0
    sections: tuple[ProfilerStat, ...] = ()
    counts: dict[str, int] = field(default_factory=dict)


OVERVIEW_SECTIONS = (
    "editor tick",
    "playmode tick",
    "viewport tick",
    "viewport paint",
    "runtime total",
    "render total",
)

RUNTIME_SECTION_PREFIXES = ("runtime", "script:")
RENDER_SECTION_PREFIXES = ("viewport paint", "viewport overlays", "render", "component:")
COUNT_ORDER = (
    "runtime scripts",
    "physics bodies",
    "audio sources",
    "audio channels",
    "active entities",
    "lights",
    "mesh renderers",
    "model renderers",
    "static model renderers",
    "dynamic model renderers",
    "sprites",
    "particle emitters",
    "particles",
    "ui elements",
    "draw submissions",
    "static model batches",
    "static cache hits",
)


class ProfilerRecorder:
    def __init__(self, capacity: int = 240, detail_interval: int = 30) -> None:
        self.enabled = False
        self._frames: deque[ProfilerFrame] = deque(maxlen=max(1, int(capacity)))
        self._lock = Lock()
        self._local = local()
        self._frame_index = 0
        self._detail_interval = max(1, int(detail_interval))

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not enabled:
            self._local.frame = None

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def begin_frame(self, view_mode: str) -> ProfilerFrame | None:
        if not self.enabled:
            return None
        if getattr(self._local, "frame", None) is not None:
            return self._local.frame
        self._frame_index += 1
        frame = ProfilerFrame(view_mode=str(view_mode or "Scene"), started_at=time(), _started_perf=perf_counter())
        self._local.frame = frame
        return frame

    def end_frame(self, frame: ProfilerFrame | None) -> None:
        if frame is None or not self.enabled:
            return
        frame.duration_ms = (perf_counter() - frame._started_perf) * 1000.0
        with self._lock:
            self._frames.append(frame)
        if getattr(self._local, "frame", None) is frame:
            self._local.frame = None

    def current_frame(self) -> ProfilerFrame | None:
        if not self.enabled:
            return None
        return getattr(self._local, "frame", None)

    def end_current_frame(self) -> None:
        self.end_frame(self.current_frame())

    def section(self, name: str, detail: str = "") -> AbstractContextManager[None]:
        if not self.enabled:
            return _NoOpSection()
        frame = getattr(self._local, "frame", None)
        if frame is None:
            return _NoOpSection()
        return _SectionTimer(frame, str(name), str(detail or ""))

    def add_count(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        frame = getattr(self._local, "frame", None)
        if frame is None:
            return
        key = str(name)
        frame.counts[key] = frame.counts.get(key, 0) + int(amount)

    def sample_details(self) -> bool:
        return self.enabled and self._frame_index % self._detail_interval == 0

    def frames(self) -> list[ProfilerFrame]:
        with self._lock:
            return list(self._frames)


class ProfilerAggregator:
    def __init__(self, recorder: ProfilerRecorder, interval: float = 0.25) -> None:
        self.recorder = recorder
        self.interval = max(0.05, float(interval))
        self._snapshot = ProfilerSnapshot()
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="P64ProfilerAggregator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def snapshot(self) -> ProfilerSnapshot:
        with self._lock:
            return self._snapshot

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            snapshot = aggregate_frames(self.recorder.frames())
            with self._lock:
                self._snapshot = snapshot


def aggregate_frames(frames: list[ProfilerFrame]) -> ProfilerSnapshot:
    if not frames:
        return ProfilerSnapshot()
    section_values: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for frame in frames:
        for section in frame.sections:
            label = section.name if not section.detail else f"{section.name}: {section.detail}"
            section_values.setdefault(label, []).append(section.duration_ms)
        for key, value in frame.counts.items():
            counts[key] = value
    stats = []
    for name, values in section_values.items():
        stats.append(ProfilerStat(
            name=name,
            last_ms=values[-1],
            average_ms=sum(values) / len(values),
            min_ms=min(values),
            max_ms=max(values),
            samples=len(values),
        ))
    stats.sort(key=lambda item: item.average_ms, reverse=True)
    scene_fps = _fps_for_view(frames, "Scene")
    game_fps = _fps_for_view(frames, "Game")
    last_render_frame = _last_render_frame(frames)
    return ProfilerSnapshot(
        frames=len(frames),
        scene_fps=scene_fps,
        game_fps=game_fps,
        frame_ms=(last_render_frame or frames[-1]).duration_ms,
        sections=tuple(stats),
        counts=counts,
    )


def profiler_sections_by_group(snapshot: ProfilerSnapshot, group: str) -> tuple[ProfilerStat, ...]:
    if group == "overview":
        by_name = {section.name: section for section in snapshot.sections}
        return tuple(by_name.get(name, _empty_stat(name)) for name in OVERVIEW_SECTIONS)
    if group == "runtime":
        return tuple(section for section in snapshot.sections if _matches_prefix(section.name, RUNTIME_SECTION_PREFIXES))
    if group == "render":
        return tuple(section for section in snapshot.sections if _matches_prefix(section.name, RENDER_SECTION_PREFIXES))
    return snapshot.sections


def profiler_counts_for_display(snapshot: ProfilerSnapshot) -> tuple[tuple[str, int], ...]:
    items: list[tuple[str, int]] = []
    seen: set[str] = set()
    for key in COUNT_ORDER:
        if key in snapshot.counts:
            items.append((key, snapshot.counts[key]))
            seen.add(key)
    for key in sorted(snapshot.counts):
        if key not in seen:
            items.append((key, snapshot.counts[key]))
    return tuple(items)


def _empty_stat(name: str) -> ProfilerStat:
    return ProfilerStat(name=name, last_ms=0.0, average_ms=0.0, min_ms=0.0, max_ms=0.0, samples=0)


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix) for prefix in prefixes)


class _NoOpSection(AbstractContextManager[None]):
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


class _SectionTimer(AbstractContextManager[None]):
    def __init__(self, frame: ProfilerFrame, name: str, detail: str) -> None:
        self.frame = frame
        self.name = name
        self.detail = detail
        self.started = 0.0

    def __enter__(self) -> None:
        self.started = perf_counter()
        return None

    def __exit__(self, *args: Any) -> None:
        self.frame.sections.append(ProfilerSection(
            name=self.name,
            detail=self.detail,
            duration_ms=(perf_counter() - self.started) * 1000.0,
        ))


def _fps_for_view(frames: list[ProfilerFrame], view_mode: str) -> float:
    view_frames = [frame for frame in frames if frame.view_mode == view_mode]
    if len(view_frames) < 2:
        return 0.0
    elapsed = view_frames[-1].started_at - view_frames[0].started_at
    if elapsed <= 0.0001:
        return 0.0
    return (len(view_frames) - 1) / elapsed


def _last_render_frame(frames: list[ProfilerFrame]) -> ProfilerFrame | None:
    for frame in reversed(frames):
        if frame.view_mode in {"Scene", "Game"}:
            return frame
    return None
