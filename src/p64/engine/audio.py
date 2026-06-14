from __future__ import annotations

import math
import re
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from p64.engine.assets import AssetMetadata, discover_metadata
from p64.engine.components import AudioSource
from p64.engine.files import find_metadata_for_source, metadata_path_for_source
from p64.engine.math import Vec3
from p64.engine.project import Project
from p64.engine.scene import Scene


MAX_AUDIO_SAMPLE_RATE = 22050


@dataclass(slots=True)
class AudioImportInfo:
    original_sample_rate: int
    original_channels: int
    original_sample_width: int
    imported_sample_rate: int
    sample_count: int
    duration: float


def import_audio_clip(project: Project, wav_path: Path) -> AssetMetadata:
    source = wav_path.resolve()
    if source.suffix.lower() != ".wav":
        raise ValueError("P64 audio import supports WAV files only.")
    try:
        source.relative_to(project.assets_dir.resolve())
    except ValueError as exc:
        raise ValueError("Audio clips must live under the project's assets folder.") from exc

    metadata_path = find_metadata_for_source(source) or metadata_path_for_source(source)
    existing = AssetMetadata.load(metadata_path) if metadata_path.exists() else None
    audio_id = existing.id if existing else safe_audio_id(source.stem)
    generated = project.generated_audio_dir / f"{audio_id}.wav"
    generated.parent.mkdir(parents=True, exist_ok=True)

    samples, info = _load_wav_as_mono_int16(source)
    _write_mono_wav(generated, samples, info.imported_sample_rate)

    metadata = AssetMetadata(
        id=audio_id,
        kind="audio_clip",
        source=source.relative_to(project.root.resolve()).as_posix(),
        settings={
            "audio": {
                "import_version": 1,
                "original_sample_rate": info.original_sample_rate,
                "original_channels": info.original_channels,
                "original_sample_width": info.original_sample_width,
                "imported_sample_rate": info.imported_sample_rate,
                "channels": 1,
                "sample_width": 2,
                "sample_count": info.sample_count,
                "duration": info.duration,
                "loopable": True,
                "generated_path": generated.relative_to(project.root.resolve()).as_posix(),
            }
        },
    )
    metadata.save(metadata_path)
    return metadata


def ensure_audio_clips_for_assets(project: Project, force: bool = False) -> list[AssetMetadata]:
    if not project.assets_dir.exists():
        return []
    imported: list[AssetMetadata] = []
    for wav_path in sorted(project.assets_dir.rglob("*.wav")):
        if not wav_path.is_file():
            continue
        if force or _audio_import_is_stale(project, wav_path):
            imported.append(import_audio_clip(project, wav_path))
    return imported


def audio_info(metadata: AssetMetadata) -> dict[str, Any] | None:
    value = metadata.settings.get("audio")
    return value if isinstance(value, dict) else None


def resolve_audio_clip(metadata_by_id: dict[str, AssetMetadata], clip: str) -> AssetMetadata | None:
    if clip in metadata_by_id:
        return metadata_by_id[clip]
    for metadata in metadata_by_id.values():
        if metadata.source == clip:
            return metadata
    return None


def safe_audio_id(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower() or "audio"
    return f"audio_{safe}"


def _audio_import_is_stale(project: Project, wav_path: Path) -> bool:
    metadata_path = find_metadata_for_source(wav_path)
    if metadata_path is None or not metadata_path.exists():
        return True
    try:
        metadata = AssetMetadata.load(metadata_path)
    except Exception:
        return True
    if metadata.kind != "audio_clip":
        return True
    info = audio_info(metadata)
    generated = info.get("generated_path") if info else None
    if not generated:
        return True
    generated_path = project.root / str(generated)
    if not generated_path.exists():
        return True
    source_mtime = wav_path.stat().st_mtime
    return source_mtime > metadata_path.stat().st_mtime or source_mtime > generated_path.stat().st_mtime


class AudioSystem:
    def __init__(self, project: Project, logger: Callable[[str], None] | None = None) -> None:
        self.project = project
        self.logger = logger
        self._pygame: Any | None = None
        self._sound_cache: dict[str, Any] = {}
        self._channels: dict[int, Any] = {}
        self._metadata = self._load_metadata()
        self._available: bool | None = None
        self._scene: Scene | None = None

    def bind_scene(self, scene: Scene) -> None:
        self._scene = scene
        for entity in scene.walk():
            for component in entity.components:
                if isinstance(component, AudioSource):
                    component.bind_runtime(self, entity)

    def start_scene(self, scene: Scene) -> None:
        self.bind_scene(scene)
        for entity in scene.walk():
            if not entity.active:
                continue
            for component in entity.components:
                if isinstance(component, AudioSource) and component.enabled and component.play_on_awake:
                    self.play(entity, component)

    def tick(self, scene: Scene, _dt: float) -> None:
        listener = _listener_position(scene)
        for entity in scene.walk():
            for component in entity.components:
                if not isinstance(component, AudioSource):
                    continue
                component.bind_runtime(self, entity)
                channel = self._channels.get(id(component))
                if channel is None:
                    continue
                if not entity.active or not component.enabled:
                    self.stop(component)
                    continue
                left, right = spatial_gains(component, _world_position(entity), listener)
                channel.set_volume(left, right)

    def stop_all(self) -> None:
        for channel in list(self._channels.values()):
            channel.stop()
        self._channels.clear()

    def play(self, entity: Any, source: AudioSource) -> bool:
        if not source.enabled or not source.clip:
            return False
        if not self._ensure_mixer():
            return False
        sound = self._sound_for(source.clip)
        if sound is None:
            return False
        self.stop(source)
        loops = -1 if source.loop else 0
        channel = sound.play(loops=loops)
        if channel is None:
            return False
        listener = _listener_position(self._scene) if self._scene else Vec3()
        left, right = spatial_gains(source, _world_position(entity), listener)
        channel.set_volume(left, right)
        self._channels[id(source)] = channel
        return True

    def stop(self, source: AudioSource) -> None:
        channel = self._channels.pop(id(source), None)
        if channel is not None:
            channel.stop()

    def pause(self, source: AudioSource) -> None:
        channel = self._channels.get(id(source))
        if channel is not None:
            channel.pause()

    def resume(self, source: AudioSource) -> None:
        channel = self._channels.get(id(source))
        if channel is not None:
            channel.unpause()

    def _load_metadata(self) -> dict[str, AssetMetadata]:
        metadata: dict[str, AssetMetadata] = {}
        for path in discover_metadata(self.project.assets_dir):
            try:
                item = AssetMetadata.load(path)
            except Exception:
                continue
            if item.kind == "audio_clip":
                metadata[item.id] = item
        return metadata

    def _sound_for(self, clip: str) -> Any | None:
        metadata = resolve_audio_clip(self._metadata, clip)
        if metadata is None:
            self._log(f"Audio clip not found: {clip}")
            return None
        info = audio_info(metadata)
        generated = info.get("generated_path") if info else None
        if not generated:
            self._log(f"Audio clip has no generated WAV: {clip}")
            return None
        path = self.project.root / str(generated)
        if not path.exists():
            self._log(f"Generated audio missing: {path}")
            return None
        cache_key = str(path.resolve())
        if cache_key not in self._sound_cache:
            self._sound_cache[cache_key] = self._pygame.mixer.Sound(str(path))
        return self._sound_cache[cache_key]

    def _ensure_mixer(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=MAX_AUDIO_SAMPLE_RATE, size=-16, channels=2)
            self._pygame = pygame
            self._available = True
        except Exception as exc:  # pragma: no cover - depends on local audio device
            self._log(f"Audio unavailable: {exc}")
            self._available = False
        return bool(self._available)

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)


def spatial_gains(source: AudioSource, position: Vec3, listener: Vec3 | None = None) -> tuple[float, float]:
    volume = max(0.0, min(1.0, float(source.volume)))
    if not source.spatial:
        return volume, volume
    listener = listener or Vec3()
    dx = position.x - listener.x
    dy = position.y - listener.y
    dz = position.z - listener.z
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    min_distance = max(0.0, float(source.min_distance))
    max_distance = max(min_distance + 0.001, float(source.max_distance))
    if distance >= max_distance:
        attenuated = 0.0
    elif distance <= min_distance:
        attenuated = volume
    else:
        attenuated = volume * (1.0 - ((distance - min_distance) / (max_distance - min_distance)))
    pan = max(-1.0, min(1.0, dx / max_distance))
    if pan >= 0.0:
        return attenuated * (1.0 - pan), attenuated
    return attenuated, attenuated * (1.0 + pan)


def _load_wav_as_mono_int16(path: Path) -> tuple[list[int], AudioImportInfo]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            frames = handle.readframes(frame_count)
    except wave.Error as exc:
        if "unknown format: 3" not in str(exc):
            raise
        samples, channels, sample_width, sample_rate = _load_ieee_float_wav(path)
        return _finish_mono_import(samples, channels, sample_width, sample_rate)
    if channels < 1:
        raise ValueError("WAV file has no audio channels.")
    samples = _decode_pcm(frames, sample_width)
    return _finish_mono_import(samples, channels, sample_width, sample_rate)


def _finish_mono_import(samples: list[float], channels: int, sample_width: int, sample_rate: int) -> tuple[list[int], AudioImportInfo]:
    if channels < 1:
        raise ValueError("WAV file has no audio channels.")
    if channels > 1:
        samples = [sum(samples[index:index + channels]) / channels for index in range(0, len(samples), channels)]
    target_rate = min(MAX_AUDIO_SAMPLE_RATE, int(sample_rate))
    if target_rate != sample_rate and len(samples) > 0:
        target_count = max(1, int(round(len(samples) * (target_rate / sample_rate))))
        samples = _resample_linear(samples, target_count)
    output = [max(-32768, min(32767, int(round(sample)))) for sample in samples]
    info = AudioImportInfo(
        original_sample_rate=int(sample_rate),
        original_channels=int(channels),
        original_sample_width=int(sample_width),
        imported_sample_rate=int(target_rate),
        sample_count=int(len(output)),
        duration=(len(output) / target_rate) if target_rate else 0.0,
    )
    return output, info


def _load_ieee_float_wav(path: Path) -> tuple[list[float], int, int, int]:
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a WAV file.")
    offset = 12
    format_tag = channels = sample_rate = bits_per_sample = None
    audio_data = b""
    while offset + 8 <= len(data):
        chunk_id = data[offset:offset + 4]
        chunk_size = struct.unpack("<I", data[offset + 4:offset + 8])[0]
        chunk = data[offset + 8:offset + 8 + chunk_size]
        if chunk_id == b"fmt ":
            format_tag, channels, sample_rate, _byte_rate, _block_align, bits_per_sample = struct.unpack("<HHIIHH", chunk[:16])
        elif chunk_id == b"data":
            audio_data = chunk
        offset += 8 + chunk_size + (chunk_size % 2)
    if format_tag != 3 or channels is None or sample_rate is None or bits_per_sample is None:
        raise ValueError("Unsupported WAV encoding.")
    if bits_per_sample == 32:
        count = len(audio_data) // 4
        samples = [max(-1.0, min(1.0, value)) * 32767.0 for value in struct.unpack(f"<{count}f", audio_data)]
        return samples, int(channels), 4, int(sample_rate)
    if bits_per_sample == 64:
        count = len(audio_data) // 8
        samples = [max(-1.0, min(1.0, value)) * 32767.0 for value in struct.unpack(f"<{count}d", audio_data)]
        return samples, int(channels), 8, int(sample_rate)
    raise ValueError(f"Unsupported float WAV sample width: {bits_per_sample}")


def _decode_pcm(frames: bytes, sample_width: int) -> list[float]:
    if sample_width == 1:
        return [(value - 128.0) * 256.0 for value in frames]
    if sample_width == 2:
        count = len(frames) // 2
        return [float(value) for value in struct.unpack(f"<{count}h", frames)]
    if sample_width == 3:
        values = []
        for index in range(0, len(frames) - 2, 3):
            value = frames[index] | (frames[index + 1] << 8) | (frames[index + 2] << 16)
            if value & 0x800000:
                value -= 0x1000000
            values.append(value / 256.0)
        return values
    if sample_width == 4:
        count = len(frames) // 4
        return [float(value) / 65536.0 for value in struct.unpack(f"<{count}i", frames)]
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def _resample_linear(samples: list[float], target_count: int) -> list[float]:
    if target_count <= 0:
        return []
    if len(samples) == 1:
        return [samples[0]] * target_count
    if target_count == 1:
        return [samples[0]]
    scale = (len(samples) - 1) / (target_count - 1)
    output: list[float] = []
    for index in range(target_count):
        position = index * scale
        left = int(math.floor(position))
        right = min(left + 1, len(samples) - 1)
        blend = position - left
        output.append(samples[left] * (1.0 - blend) + samples[right] * blend)
    return output


def _write_mono_wav(path: Path, samples: list[int], sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _listener_position(scene: Scene) -> Vec3:
    camera = scene.active_camera()
    return _world_position(camera) if camera else Vec3()


def _world_position(entity: Any | None) -> Vec3:
    if entity is None:
        return Vec3()
    matrix = entity.transform.world_matrix(entity)
    return Vec3(matrix[3], matrix[7], matrix[11])
