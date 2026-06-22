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
from p64.engine.entity import entity_effectively_active
from p64.engine.files import find_metadata_for_source, metadata_path_for_source
from p64.engine.math import Vec3
from p64.engine.project import Project
from p64.engine.scene import Scene
from p64.engine.transforms import world_position as transform_world_position, world_rotation as transform_world_rotation


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
        self._paused: set[int] = set()
        self._metadata = self._load_metadata()
        self._available: bool | None = None
        self._scene: Scene | None = None
        self._missing_listener_logged = False

    def bind_scene(self, scene: Scene) -> None:
        self._scene = scene
        self._missing_listener_logged = False
        for entity in scene.walk_active():
            for component in entity.components:
                if isinstance(component, AudioSource):
                    component.bind_runtime(self, entity)

    def start_scene(self, scene: Scene) -> None:
        self.bind_scene(scene)
        for entity in scene.walk():
            for component in entity.components:
                if isinstance(component, AudioSource) and component.enabled and component.play_on_awake:
                    self.play(entity, component)

    def tick(self, scene: Scene, _dt: float) -> None:
        listener = _listener_pose(scene)
        if listener is None:
            if self._channels:
                self._log_missing_listener()
                self.stop_all()
            return
        listener_position, listener_rotation = listener
        for entity in scene.walk():
            for component in entity.components:
                if not isinstance(component, AudioSource):
                    continue
                component.bind_runtime(self, entity)
                component_id = id(component)
                channel = self._channels.get(component_id)
                if channel is None:
                    continue
                if not entity_effectively_active(entity) or not component.enabled:
                    self.stop(component)
                    continue
                try:
                    if not component.loop and component_id not in self._paused and hasattr(channel, "get_busy") and not channel.get_busy():
                        self._channels.pop(component_id, None)
                        continue
                    left, right = spatial_gains(component, _world_position(entity), listener_position, listener_rotation)
                    channel.set_volume(left, right)
                except Exception as exc:
                    self._channels.pop(component_id, None)
                    self._paused.discard(component_id)
                    self._log(f"Audio channel update failed: {exc}")

    def stop_all(self) -> None:
        for channel in list(self._channels.values()):
            channel.stop()
        self._channels.clear()
        self._paused.clear()

    def play(self, entity: Any, source: AudioSource) -> bool:
        if not entity_effectively_active(entity) or not source.enabled or not source.clip:
            return False
        listener = _listener_pose(self._scene) if self._scene else None
        if listener is None:
            self._log_missing_listener()
            return False
        if not self._ensure_mixer():
            return False
        sound = self._sound_for(source.clip, source.pitch)
        if sound is None:
            return False
        self.stop(source)
        loops = -1 if source.loop else 0
        try:
            channel = sound.play(loops=loops)
        except Exception as exc:
            self._log(f"Audio playback failed: {exc}")
            return False
        if channel is None:
            return False
        listener_position, listener_rotation = listener
        left, right = spatial_gains(source, _world_position(entity), listener_position, listener_rotation)
        channel.set_volume(left, right)
        self._channels[id(source)] = channel
        return True

    def stop(self, source: AudioSource) -> None:
        channel = self._channels.pop(id(source), None)
        self._paused.discard(id(source))
        if channel is not None:
            channel.stop()

    def pause(self, source: AudioSource) -> None:
        channel = self._channels.get(id(source))
        if channel is not None:
            channel.pause()
            self._paused.add(id(source))

    def resume(self, source: AudioSource) -> None:
        channel = self._channels.get(id(source))
        if channel is not None:
            channel.unpause()
            self._paused.discard(id(source))

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

    def _sound_for(self, clip: str, pitch: float = 1.0) -> Any | None:
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
        pitch = max(0.001, float(pitch))
        cache_key = f"{path.resolve()}|pitch={pitch:.6f}"
        try:
            if cache_key not in self._sound_cache:
                if _is_default_pitch(pitch):
                    self._sound_cache[cache_key] = self._pygame.mixer.Sound(str(path))
                else:
                    self._sound_cache[cache_key] = self._pitched_sound(path, pitch)
        except Exception as exc:
            self._log(f"Audio clip could not be loaded: {clip}: {exc}")
            return None
        return self._sound_cache[cache_key]

    def _pitched_sound(self, path: Path, pitch: float) -> Any:
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - dependency is declared in pyproject
            np = None

        samples, _sample_rate = _read_mono_wav_int16(path)
        pitched = _pitch_shift_mono_samples(samples, pitch)
        if np is None:
            stereo = [(sample, sample) for sample in pitched]
        else:
            mono = np.asarray(pitched, dtype=np.int16)
            stereo = np.column_stack((mono, mono)).copy()
        return self._pygame.sndarray.make_sound(stereo)

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

    def _log_missing_listener(self) -> None:
        if self._missing_listener_logged:
            return
        self._missing_listener_logged = True
        self._log("Audio listener missing: add an AudioListener component to the camera or another active entity.")


def spatial_gains(source: AudioSource, position: Vec3, listener: Vec3 | None = None, listener_rotation: Vec3 | None = None) -> tuple[float, float]:
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
    right_x, right_z = _listener_right(listener_rotation or Vec3())
    lateral = dx * right_x + dz * right_z
    pan = max(-1.0, min(1.0, lateral / max_distance))
    if pan >= 0.0:
        return attenuated * (1.0 - pan), attenuated
    return attenuated, attenuated * (1.0 + pan)


def _is_default_pitch(pitch: float) -> bool:
    return abs(float(pitch) - 1.0) < 0.000001


def _read_mono_wav_int16(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        frames = handle.readframes(frame_count)
    if channels != 1 or sample_width != 2:
        raise ValueError("Generated audio must be mono 16-bit WAV.")
    return [int(value) for value in struct.unpack(f"<{len(frames) // 2}h", frames)], int(sample_rate)


def _pitch_shift_mono_samples(samples: list[int], pitch: float) -> list[int]:
    pitch = max(0.001, float(pitch))
    if not samples or _is_default_pitch(pitch):
        return list(samples)
    target_count = max(1, int(round(len(samples) / pitch)))
    shifted = _resample_linear([float(sample) for sample in samples], target_count)
    return [max(-32768, min(32767, int(round(sample)))) for sample in shifted]


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


def _listener_pose(scene: Scene | None) -> tuple[Vec3, Vec3] | None:
    if scene is None:
        return None
    listener = scene.active_audio_listener()
    if listener is None:
        return None
    return _world_position(listener), _world_rotation(listener)


def _listener_right(rotation: Vec3) -> tuple[float, float]:
    yaw = math.radians(rotation.y)
    return math.cos(yaw), math.sin(yaw)


def _world_position(entity: Any | None) -> Vec3:
    return transform_world_position(entity)


def _world_rotation(entity: Any | None) -> Vec3:
    return transform_world_rotation(entity)
