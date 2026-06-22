from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ConsoleEvent:
    text: str
    level: str
    timestamp: float


@dataclass
class ConsoleEntry:
    text: str
    level: str
    count: int
    first_seen: float
    last_seen: float


def infer_console_level(text: str) -> str:
    normalized = text.lower()
    if any(token in normalized for token in ["error", "failed", "exception", "traceback", "notimplementederror"]):
        return "Error"
    if any(token in normalized for token in ["warning", "warn", "missing", "invalid"]):
        return "Warning"
    return "Info"


class ConsoleLogModel:
    def __init__(self) -> None:
        self._entries: list[ConsoleEntry] = []
        self._entry_by_text: dict[str, ConsoleEntry] = {}
        self._events: list[ConsoleEvent] = []

    def add(self, text: str, timestamp: float) -> ConsoleEntry:
        message = str(text)
        level = infer_console_level(message)
        self._events.append(ConsoleEvent(message, level, timestamp))
        entry = self._entry_by_text.get(message)
        if entry is None:
            entry = ConsoleEntry(message, level, 1, timestamp, timestamp)
            self._entry_by_text[message] = entry
            self._entries.append(entry)
            return entry
        entry.count += 1
        entry.last_seen = timestamp
        return entry

    def clear(self) -> None:
        self._entries.clear()
        self._entry_by_text.clear()
        self._events.clear()

    def rows(self, collapse: bool = True) -> list[ConsoleEntry]:
        if collapse:
            return list(self._entries)
        return [ConsoleEntry(event.text, event.level, 1, event.timestamp, event.timestamp) for event in self._events]

    def filtered(self, query: str = "", levels: set[str] | None = None, collapse: bool = True) -> list[ConsoleEntry]:
        normalized_query = query.strip().lower()
        accepted_levels = levels or {"Info", "Warning", "Error"}
        rows: list[ConsoleEntry] = []
        for entry in self.rows(collapse):
            if entry.level not in accepted_levels:
                continue
            if normalized_query and normalized_query not in entry.text.lower():
                continue
            rows.append(entry)
        return rows


def create_console_panel(
    QApplication: Any,
    QBrush: Any,
    QCheckBox: Any,
    QColor: Any,
    QHBoxLayout: Any,
    QLabel: Any,
    QLineEdit: Any,
    QPlainTextEdit: Any,
    QPushButton: Any,
    QTreeWidget: Any,
    QTreeWidgetItem: Any,
    QVBoxLayout: Any,
    QWidget: Any,
    Qt: Any,
) -> type:
    class ConsolePanel(QWidget):
        def __init__(self, parent: Any = None) -> None:
            super().__init__(parent)
            self.model = ConsoleLogModel()
            self._build_ui()

        def _build_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)
            toolbar = QHBoxLayout()
            toolbar.setContentsMargins(0, 0, 0, 0)
            toolbar.setSpacing(6)

            self.search = QLineEdit(self)
            self.search.setPlaceholderText("Search")
            self.collapse = QCheckBox("Collapse", self)
            self.collapse.setChecked(True)
            self.info_filter = QCheckBox("Info", self)
            self.info_filter.setChecked(True)
            self.warning_filter = QCheckBox("Warnings", self)
            self.warning_filter.setChecked(True)
            self.error_filter = QCheckBox("Errors", self)
            self.error_filter.setChecked(True)
            self.copy_selected = QPushButton("Copy Selected", self)
            self.copy_details = QPushButton("Copy Details", self)
            self.clear_button = QPushButton("Clear", self)

            toolbar.addWidget(QLabel("Console", self))
            toolbar.addWidget(self.search, 1)
            toolbar.addWidget(self.collapse)
            toolbar.addWidget(self.info_filter)
            toolbar.addWidget(self.warning_filter)
            toolbar.addWidget(self.error_filter)
            toolbar.addWidget(self.copy_selected)
            toolbar.addWidget(self.copy_details)
            toolbar.addWidget(self.clear_button)

            self.tree = QTreeWidget(self)
            self.tree.setColumnCount(4)
            self.tree.setHeaderLabels(["Level", "Message", "Count", "Last"])
            self.tree.setRootIsDecorated(False)
            self.tree.setAlternatingRowColors(True)
            self.tree.setUniformRowHeights(True)
            self.tree.setColumnWidth(0, 84)
            self.tree.setColumnWidth(2, 64)
            self.details = QPlainTextEdit(self)
            self.details.setReadOnly(True)
            self.details.setMaximumHeight(90)

            layout.addLayout(toolbar)
            layout.addWidget(self.tree, 1)
            layout.addWidget(self.details)

            self.search.textChanged.connect(lambda _text: self.refresh())
            self.collapse.toggled.connect(lambda _checked: self.refresh())
            self.info_filter.toggled.connect(lambda _checked: self.refresh())
            self.warning_filter.toggled.connect(lambda _checked: self.refresh())
            self.error_filter.toggled.connect(lambda _checked: self.refresh())
            self.clear_button.clicked.connect(self.clear)
            self.copy_selected.clicked.connect(self._copy_selected)
            self.copy_details.clicked.connect(self._copy_details)
            self.tree.itemSelectionChanged.connect(self._selection_changed)

        def add_log(self, text: str) -> None:
            import time

            scrollbar = self.tree.verticalScrollBar()
            was_at_bottom = scrollbar.value() >= scrollbar.maximum()
            self.model.add(text, time.time())
            self.refresh()
            if was_at_bottom:
                self.tree.scrollToBottom()

        def clear(self) -> None:
            self.model.clear()
            self.refresh()
            self.details.clear()

        def refresh(self) -> None:
            selected_text = self._selected_text()
            self.tree.clear()
            for entry in self.model.filtered(self.search.text(), self._enabled_levels(), self.collapse.isChecked()):
                item = QTreeWidgetItem([entry.level, self._first_line(entry.text), str(entry.count), self._format_timestamp(entry.last_seen)])
                item.setToolTip(1, entry.text)
                item.setData(0, Qt.UserRole, entry.text)
                item.setForeground(0, self._level_brush(entry.level))
                item.setForeground(1, self._level_brush(entry.level))
                self.tree.addTopLevelItem(item)
                if selected_text and selected_text == entry.text:
                    self.tree.setCurrentItem(item)
            if selected_text and not self.tree.currentItem():
                self.details.clear()
            self.tree.resizeColumnToContents(0)
            self.tree.resizeColumnToContents(2)
            self.tree.resizeColumnToContents(3)

        def _enabled_levels(self) -> set[str]:
            levels: set[str] = set()
            if self.info_filter.isChecked():
                levels.add("Info")
            if self.warning_filter.isChecked():
                levels.add("Warning")
            if self.error_filter.isChecked():
                levels.add("Error")
            return levels

        def _selection_changed(self) -> None:
            self.details.setPlainText(self._selected_text())

        def _selected_text(self) -> str:
            item = self.tree.currentItem()
            return str(item.data(0, Qt.UserRole)) if item else ""

        def _copy_selected(self) -> None:
            text = self._selected_text()
            if text:
                QApplication.clipboard().setText(text)

        def _copy_details(self) -> None:
            text = self.details.toPlainText()
            if text:
                QApplication.clipboard().setText(text)

        def _level_brush(self, level: str) -> Any:
            if level == "Error":
                return QBrush(QColor("#ff6b6b"))
            if level == "Warning":
                return QBrush(QColor("#f2c94c"))
            return QBrush(QColor("#cfd4dc"))

        @staticmethod
        def _first_line(text: str) -> str:
            return text.splitlines()[0] if text.splitlines() else text

        @staticmethod
        def _format_timestamp(timestamp: float) -> str:
            import time

            return time.strftime("%H:%M:%S", time.localtime(timestamp))

    return ConsolePanel
