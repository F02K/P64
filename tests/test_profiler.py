import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from p64.editor.profiler import ProfilerFrame, ProfilerRecorder, aggregate_frames, profiler_counts_for_display, profiler_sections_by_group
from p64.engine.components import AudioListener, AudioSource, EntityPhysics
from p64.engine.entity import Entity
from p64.engine.project import Project
from p64.engine.runtime_session import RuntimeSession
from p64.engine.scene import Scene


class ProfilerTests(unittest.TestCase):
    def test_disabled_profiler_records_no_samples(self):
        recorder = ProfilerRecorder()

        frame = recorder.begin_frame("Scene")
        with recorder.section("render"):
            pass
        recorder.add_count("draws", 1)
        recorder.end_frame(frame)

        self.assertEqual(recorder.frames(), [])

    def test_enabled_profiler_aggregates_sections_and_counts(self):
        recorder = ProfilerRecorder()
        recorder.set_enabled(True)
        frame = recorder.begin_frame("Scene")
        self.assertIsNotNone(frame)

        with recorder.section("render skybox"):
            pass
        recorder.add_count("active entities", 3)
        recorder.end_frame(frame)

        snapshot = aggregate_frames(recorder.frames())

        self.assertEqual(snapshot.frames, 1)
        self.assertEqual(snapshot.counts["active entities"], 3)
        self.assertEqual(snapshot.sections[0].name, "render skybox")
        self.assertGreaterEqual(snapshot.sections[0].last_ms, 0.0)

    def test_editor_frames_do_not_affect_render_fps_or_last_frame_ms(self):
        frames = [
            ProfilerFrame("Game", started_at=1.000, duration_ms=16.0),
            ProfilerFrame("Editor", started_at=1.004, duration_ms=1.5),
            ProfilerFrame("Game", started_at=1.016, duration_ms=17.0),
            ProfilerFrame("Editor", started_at=1.020, duration_ms=2.5),
        ]

        snapshot = aggregate_frames(frames)

        self.assertAlmostEqual(snapshot.game_fps, 62.5, places=1)
        self.assertEqual(snapshot.scene_fps, 0.0)
        self.assertEqual(snapshot.frame_ms, 17.0)

    def test_runtime_tick_reports_subsystem_sections_and_counts(self):
        with TemporaryDirectory() as tmp:
            project = Project.create(Path(tmp) / "Game")
            speaker = Entity("Speaker", components=[AudioSource(clip="")])
            listener = Entity("Listener", components=[AudioListener()])
            body = Entity("Body", components=[EntityPhysics(use_gravity=False)])
            session = RuntimeSession(project, Scene("Runtime", [listener, speaker, body]))
            recorder = ProfilerRecorder()
            recorder.set_enabled(True)
            session.profiler_recorder = recorder

            frame = recorder.begin_frame("Game")
            session.tick(1 / 60)
            recorder.end_frame(frame)
            snapshot = aggregate_frames(recorder.frames())
            names = {section.name for section in snapshot.sections}

            self.assertIn("runtime total", names)
            self.assertIn("runtime start", names)
            self.assertIn("runtime scripts", names)
            self.assertIn("runtime scene switch", names)
            self.assertIn("runtime physics", names)
            self.assertIn("runtime audio", names)
            self.assertEqual(snapshot.counts["runtime scripts"], 0)
            self.assertEqual(snapshot.counts["audio sources"], 1)
            self.assertEqual(snapshot.counts["audio channels"], 0)
            self.assertEqual(snapshot.counts["physics bodies"], 1)

    def test_profiler_groups_sections_and_counts_for_analysis_tabs(self):
        recorder = ProfilerRecorder()
        recorder.set_enabled(True)
        frame = recorder.begin_frame("Game")

        with recorder.section("editor tick"):
            pass
        with recorder.section("playmode tick"):
            pass
        with recorder.section("runtime physics"):
            pass
        with recorder.section("script", "spin.py:Spin"):
            pass
        with recorder.section("render meshes"):
            pass
        with recorder.section("viewport paint"):
            pass
        recorder.add_count("draw submissions", 3)
        recorder.add_count("physics bodies", 1)
        recorder.add_count("audio sources", 2)
        recorder.end_frame(frame)

        snapshot = aggregate_frames(recorder.frames())
        overview_names = [section.name for section in profiler_sections_by_group(snapshot, "overview")]
        runtime_names = [section.name for section in profiler_sections_by_group(snapshot, "runtime")]
        render_names = [section.name for section in profiler_sections_by_group(snapshot, "render")]
        counts = profiler_counts_for_display(snapshot)

        self.assertEqual(overview_names, [
            "editor tick",
            "playmode tick",
            "viewport tick",
            "viewport paint",
            "runtime total",
            "render total",
        ])
        self.assertIn("runtime physics", runtime_names)
        self.assertIn("script: spin.py:Spin", runtime_names)
        self.assertIn("render meshes", render_names)
        self.assertIn("viewport paint", render_names)
        self.assertEqual(counts[:3], (("physics bodies", 1), ("audio sources", 2), ("draw submissions", 3)))


if __name__ == "__main__":
    unittest.main()
