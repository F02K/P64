from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import math
import struct
import time
import wave
import unittest

from p64.build.pipeline import create_runtime_bundle, validate_project
from p64.engine.audio import AudioSystem, ensure_audio_clips_for_assets, import_audio_clip, spatial_gains, _pitch_shift_mono_samples
from p64.engine.components import AudioListener, AudioSource, Camera
from p64.engine.entity import Entity
from p64.engine.math import Vec3
from p64.engine.project import Project
from p64.engine.runtime import run_project
from p64.engine.runtime_session import RuntimeSession
from p64.engine.scene import Scene


class AudioSourceTests(unittest.TestCase):
    def test_audio_listener_serializes(self):
        listener = AudioListener(active=False)
        scene = Scene("Audio", [Entity("Listener", components=[listener])])

        loaded = Scene.from_dict(scene.to_dict())
        component = loaded.entities[0].components[0]

        self.assertIsInstance(component, AudioListener)
        self.assertFalse(component.active)

    def test_audio_source_serializes(self):
        source = AudioSource(
            clip="audio_beep",
            volume=0.5,
            pitch=1.25,
            loop=True,
            play_on_awake=False,
            spatial=False,
            min_distance=2.0,
            max_distance=12.0,
        )
        scene = Scene("Audio", [Entity("Speaker", components=[source])])

        loaded = Scene.from_dict(scene.to_dict())
        component = loaded.entities[0].components[0]

        self.assertIsInstance(component, AudioSource)
        self.assertEqual(component.clip, "audio_beep")
        self.assertEqual(component.volume, 0.5)
        self.assertEqual(component.pitch, 1.25)
        self.assertTrue(component.loop)
        self.assertFalse(component.play_on_awake)
        self.assertFalse(component.spatial)
        self.assertEqual(component.min_distance, 2.0)
        self.assertEqual(component.max_distance, 12.0)

    def test_wav_import_creates_mono_limited_generated_audio(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            wav_path = project.assets_dir / "beep.wav"
            _write_stereo_wav(wav_path, sample_rate=44100)

            metadata = import_audio_clip(project, wav_path)

            info = metadata.settings["audio"]
            generated = project.root / info["generated_path"]
            self.assertEqual(metadata.kind, "audio_clip")
            self.assertEqual(info["original_sample_rate"], 44100)
            self.assertEqual(info["original_channels"], 2)
            self.assertEqual(info["imported_sample_rate"], 22050)
            self.assertTrue(generated.exists())
            self.assertTrue((wav_path.with_suffix(".wav.mdp64")).exists())
            with wave.open(str(generated), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getframerate(), 22050)

    def test_float_wav_import_converts_to_pcm_mono(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            wav_path = project.assets_dir / "float.wav"
            _write_float_wav(wav_path, sample_rate=44100)

            metadata = import_audio_clip(project, wav_path)

            generated = project.root / metadata.settings["audio"]["generated_path"]
            with wave.open(str(generated), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getframerate(), 22050)

    def test_ensure_audio_clips_imports_wav_assets_once(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            wav_path = project.assets_dir / "beep.wav"
            _write_stereo_wav(wav_path, sample_rate=44100)

            imported = ensure_audio_clips_for_assets(project)
            second = ensure_audio_clips_for_assets(project)

            info = imported[0].settings["audio"]
            generated = project.root / info["generated_path"]
            self.assertEqual(len(imported), 1)
            self.assertEqual(second, [])
            self.assertTrue(wav_path.with_suffix(".wav.mdp64").exists())
            self.assertTrue(generated.exists())
            self.assertEqual(imported[0].id, "audio_beep")

    def test_ensure_audio_clips_regenerates_missing_generated_wav(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            wav_path = project.assets_dir / "beep.wav"
            _write_stereo_wav(wav_path, sample_rate=44100)
            metadata = ensure_audio_clips_for_assets(project)[0]
            generated = project.root / metadata.settings["audio"]["generated_path"]
            generated.unlink()

            imported = ensure_audio_clips_for_assets(project)

            self.assertEqual(imported[0].id, metadata.id)
            self.assertTrue(generated.exists())

    def test_ensure_audio_clips_reimports_when_source_changes(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            wav_path = project.assets_dir / "beep.wav"
            _write_stereo_wav(wav_path, sample_rate=44100)
            first = ensure_audio_clips_for_assets(project)[0]
            time.sleep(1.1)
            _write_stereo_wav(wav_path, sample_rate=16000)

            imported = ensure_audio_clips_for_assets(project)

            self.assertEqual(imported[0].id, first.id)
            self.assertEqual(imported[0].settings["audio"]["imported_sample_rate"], 16000)

    def test_audio_reimport_preserves_metadata_id(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            wav_path = project.assets_dir / "beep.wav"
            _write_stereo_wav(wav_path, sample_rate=44100)
            first = import_audio_clip(project, wav_path)
            _write_stereo_wav(wav_path, sample_rate=16000)

            second = import_audio_clip(project, wav_path)

            self.assertEqual(second.id, first.id)
            self.assertEqual(second.settings["audio"]["imported_sample_rate"], 16000)

    def test_spatial_gains_pan_and_attenuate(self):
        source = AudioSource(volume=1.0, min_distance=1.0, max_distance=10.0)

        center = spatial_gains(source, Vec3(), Vec3())
        right = spatial_gains(source, Vec3(5, 0, 0), Vec3())
        far = spatial_gains(source, Vec3(20, 0, 0), Vec3())

        self.assertEqual(center, (1.0, 1.0))
        self.assertLess(right[0], right[1])
        self.assertEqual(far, (0.0, 0.0))

    def test_spatial_gains_use_listener_rotation_for_panning(self):
        source = AudioSource(volume=1.0, min_distance=1.0, max_distance=10.0)

        right_at_yaw_zero = spatial_gains(source, Vec3(5, 0, 0), Vec3(), Vec3(0, 0, 0))
        centered_after_turn = spatial_gains(source, Vec3(5, 0, 0), Vec3(), Vec3(0, 90, 0))

        self.assertLess(right_at_yaw_zero[0], right_at_yaw_zero[1])
        self.assertAlmostEqual(centered_after_turn[0], centered_after_turn[1])

    def test_pitch_shift_resamples_mono_samples(self):
        samples = [0, 1000, 2000, 3000]

        higher = _pitch_shift_mono_samples(samples, 2.0)
        lower = _pitch_shift_mono_samples(samples, 0.5)

        self.assertEqual(len(higher), 2)
        self.assertEqual(len(lower), 8)
        self.assertEqual(_pitch_shift_mono_samples(samples, 1.0), samples)

    def test_sound_cache_distinguishes_pitch_values(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            wav_path = project.assets_dir / "beep.wav"
            _write_stereo_wav(wav_path, sample_rate=22050)
            metadata = import_audio_clip(project, wav_path)
            pygame = _FakePygame()
            audio = AudioSystem(project)
            audio._pygame = pygame
            audio._available = True

            normal = audio._sound_for(metadata.id, 1.0)
            high = audio._sound_for(metadata.id, 2.0)
            high_again = audio._sound_for(metadata.id, 2.0)

            self.assertIsNot(normal, high)
            self.assertIs(high, high_again)
            self.assertEqual(len(pygame.mixer.sound_paths), 1)
            self.assertEqual(len(pygame.sndarray.arrays), 1)
            self.assertEqual(len(audio._sound_cache), 2)

    def test_runtime_session_imports_audio_before_metadata_load(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            wav_path = project.assets_dir / "beep.wav"
            _write_stereo_wav(wav_path, sample_rate=44100)
            scene = Scene("Audio", [Entity("Speaker", components=[AudioSource(clip="audio_beep")])])

            session = RuntimeSession(project, scene)

            self.assertTrue(wav_path.with_suffix(".wav.mdp64").exists())
            self.assertIn("audio_beep", session.audio._metadata)

    def test_audio_system_refuses_play_without_listener(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            speaker = Entity("Speaker", components=[AudioSource(clip="audio_beep")])
            scene = Scene("Audio", [speaker])
            messages: list[str] = []
            audio = AudioSystem(project, logger=messages.append)
            audio.bind_scene(scene)

            self.assertFalse(audio.play(speaker, speaker.components[0]))
            self.assertEqual(audio._channels, {})
            self.assertEqual(messages, ["Audio listener missing: add an AudioListener component to the camera or another active entity."])

    def test_tick_stops_channels_when_listener_is_missing(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            source = AudioSource(loop=True)
            speaker = Entity("Speaker", components=[source])
            scene = Scene("Audio", [speaker])
            audio = AudioSystem(project)
            channel = _FakeChannel()
            audio._channels[id(source)] = channel

            audio.tick(scene, 1 / 60)

            self.assertTrue(channel.stopped)
            self.assertEqual(audio._channels, {})

    def test_run_project_stops_preflight_session_before_window_launch(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")

            with mock.patch("p64.engine.runtime.RuntimeSession.stop") as stop:
                with mock.patch("p64.editor.app.launch_runtime_window") as launch:
                    run_project(project.root)

            stop.assert_called_once()
            launch.assert_called_once()

    def test_tick_removes_finished_one_shot_channels(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            source = AudioSource(loop=False)
            speaker = Entity("Speaker", components=[source])
            listener = Entity("Listener", components=[AudioListener()])
            scene = Scene("Audio", [listener, speaker])
            audio = AudioSystem(project)
            channel = _FakeChannel(busy=False)
            audio._channels[id(source)] = channel

            audio.tick(scene, 1 / 60)

            self.assertNotIn(id(source), audio._channels)
            self.assertEqual(channel.volume_calls, [])

    def test_runtime_play_on_awake_and_script_methods_use_audio_system(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            speaker = Entity("Speaker")
            component = speaker.add_component(AudioSource(clip="audio_beep"))
            scene = Scene("Audio", [speaker])
            session = RuntimeSession(project, scene)

            with mock.patch.object(AudioSystem, "play", return_value=True) as play:
                session.start()
                self.assertTrue(component.play())
                component.pause()
                component.resume()
                component.stop()

            self.assertEqual(play.call_count, 2)

    def test_validation_and_bundle_include_generated_audio(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            wav_path = project.assets_dir / "beep.wav"
            _write_stereo_wav(wav_path, sample_rate=44100)
            metadata = import_audio_clip(project, wav_path)
            scene = project.load_startup_scene()
            speaker = Entity("Speaker")
            speaker.add_component(AudioSource(clip=metadata.id))
            scene.add_entity(speaker)
            project.save_startup_scene(scene)

            report = validate_project(project.root)
            bundle = create_runtime_bundle(project.root)

            self.assertTrue(report.ok)
            self.assertTrue((bundle / "packages" / "P64Generated" / "audio" / f"{metadata.id}.wav").exists())

    def test_validation_and_bundle_auto_import_wav_assets(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            wav_path = project.assets_dir / "beep.wav"
            _write_stereo_wav(wav_path, sample_rate=44100)

            report = validate_project(project.root)
            bundle = create_runtime_bundle(project.root)

            self.assertTrue(report.ok)
            self.assertTrue(wav_path.with_suffix(".wav.mdp64").exists())
            self.assertTrue((bundle / "packages" / "P64Generated" / "audio" / "audio_beep.wav").exists())


def _write_stereo_wav(path: Path, sample_rate: int) -> None:
    frames = max(1, math.floor(sample_rate / 10))
    values: list[int] = []
    for index in range(frames):
        t = index / sample_rate
        values.append(int(math.sin(2.0 * math.pi * 440.0 * t) * 20000))
        values.append(int(math.sin(2.0 * math.pi * 660.0 * t) * 16000))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{len(values)}h", *values))


def _write_float_wav(path: Path, sample_rate: int) -> None:
    frames = max(1, math.floor(sample_rate / 10))
    values: list[float] = []
    for index in range(frames):
        t = index / sample_rate
        values.append(math.sin(2.0 * math.pi * 440.0 * t) * 0.5)
        values.append(math.sin(2.0 * math.pi * 660.0 * t) * 0.4)
    payload = struct.pack(f"<{len(values)}f", *values)
    block_align = 2 * 4
    byte_rate = sample_rate * block_align
    with path.open("wb") as handle:
        handle.write(b"RIFF")
        handle.write(struct.pack("<I", 4 + (8 + 16) + (8 + len(payload))))
        handle.write(b"WAVE")
        handle.write(b"fmt ")
        handle.write(struct.pack("<IHHIIHH", 16, 3, 2, sample_rate, byte_rate, block_align, 32))
        handle.write(b"data")
        handle.write(struct.pack("<I", len(payload)))
        handle.write(payload)


class _FakeSound:
    def play(self, loops: int = 0):
        return _FakeChannel()


class _FakeMixer:
    def __init__(self) -> None:
        self.sound_paths: list[str] = []

    def Sound(self, path: str):
        self.sound_paths.append(path)
        return _FakeSound()


class _FakeSndArray:
    def __init__(self) -> None:
        self.arrays: list[object] = []

    def make_sound(self, array):
        self.arrays.append(array)
        return _FakeSound()


class _FakePygame:
    def __init__(self) -> None:
        self.mixer = _FakeMixer()
        self.sndarray = _FakeSndArray()


class _FakeChannel:
    def __init__(self, busy: bool = True) -> None:
        self.busy = busy
        self.volume_calls: list[tuple[float, float]] = []
        self.stopped = False
        self.paused = False

    def get_busy(self) -> bool:
        return self.busy

    def set_volume(self, left: float, right: float) -> None:
        self.volume_calls.append((left, right))

    def stop(self) -> None:
        self.stopped = True

    def pause(self) -> None:
        self.paused = True

    def unpause(self) -> None:
        self.paused = False


if __name__ == "__main__":
    unittest.main()
