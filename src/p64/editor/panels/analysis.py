from __future__ import annotations

from typing import Any

from p64.editor.profiler import ProfilerAggregator, ProfilerRecorder, ProfilerStat, profiler_counts_for_display, profiler_sections_by_group


def create_analysis_panel(
    QLabel: Any,
    QTabWidget: Any,
    QTableWidget: Any,
    QTableWidgetItem: Any,
    QVBoxLayout: Any,
    QWidget: Any,
    QTimer: Any,
    Qt: Any,
) -> type:
    class AnalysisPanel(QWidget):  # type: ignore[misc, valid-type]
        def __init__(self, recorder: ProfilerRecorder, parent: object | None = None) -> None:
            super().__init__(parent, Qt.Window)
            self.recorder = recorder
            self.aggregator = ProfilerAggregator(recorder)
            self.setWindowTitle("Analysis")
            self.resize(620, 480)
            layout = QVBoxLayout(self)
            self.summary = QLabel("Profiler inactive", self)
            self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(self.summary)
            self.tabs = QTabWidget(self)
            self.overview = _section_table(QTableWidget, self)
            self.runtime = _section_table(QTableWidget, self)
            self.render = _section_table(QTableWidget, self)
            self.counts = QTableWidget(0, 2, self)
            self.counts.setHorizontalHeaderLabels(["Metric", "Value"])
            self.counts.setSortingEnabled(False)
            self.tabs.addTab(self.overview, "Overview")
            self.tabs.addTab(self.runtime, "Runtime")
            self.tabs.addTab(self.render, "Render")
            self.tabs.addTab(self.counts, "Counts")
            layout.addWidget(self.tabs, 1)
            self.refresh_timer = QTimer(self)
            self.refresh_timer.timeout.connect(self.refresh)

        def showEvent(self, event: object) -> None:
            self.recorder.clear()
            self.recorder.set_enabled(True)
            self.aggregator.start()
            self.refresh_timer.start(250)
            super().showEvent(event)

        def closeEvent(self, event: object) -> None:
            self.refresh_timer.stop()
            self.aggregator.stop()
            self.recorder.set_enabled(False)
            super().closeEvent(event)

        def refresh(self) -> None:
            snapshot = self.aggregator.snapshot()
            self.summary.setText(
                f"Frames {snapshot.frames} | Scene {snapshot.scene_fps:.1f} FPS | "
                f"Game {snapshot.game_fps:.1f} FPS | Last frame {snapshot.frame_ms:.2f} ms"
            )
            _populate_section_table(self.overview, profiler_sections_by_group(snapshot, "overview"), QTableWidgetItem)
            _populate_section_table(self.runtime, profiler_sections_by_group(snapshot, "runtime"), QTableWidgetItem)
            _populate_section_table(self.render, profiler_sections_by_group(snapshot, "render"), QTableWidgetItem)
            _populate_counts_table(self.counts, profiler_counts_for_display(snapshot), QTableWidgetItem)

    return AnalysisPanel


def _section_table(QTableWidget: Any, parent: Any) -> Any:
    table = QTableWidget(0, 6, parent)
    table.setHorizontalHeaderLabels(["Section", "Last ms", "Avg ms", "Min ms", "Max ms", "Samples"])
    table.setSortingEnabled(False)
    return table


def _populate_section_table(table: Any, rows: tuple[ProfilerStat, ...], QTableWidgetItem: Any) -> None:
    table.setRowCount(len(rows))
    for row, stat in enumerate(rows):
        values = [
            stat.name,
            f"{stat.last_ms:.3f}",
            f"{stat.average_ms:.3f}",
            f"{stat.min_ms:.3f}",
            f"{stat.max_ms:.3f}",
            str(stat.samples),
        ]
        for column, value in enumerate(values):
            table.setItem(row, column, QTableWidgetItem(value))
    table.resizeColumnsToContents()


def _populate_counts_table(table: Any, rows: tuple[tuple[str, int], ...], QTableWidgetItem: Any) -> None:
    table.setRowCount(len(rows))
    for row, (name, value) in enumerate(rows):
        table.setItem(row, 0, QTableWidgetItem(name))
        table.setItem(row, 1, QTableWidgetItem(str(value)))
    table.resizeColumnsToContents()
