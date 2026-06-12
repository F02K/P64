from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from p64.engine.files import PROJECT_FILE, is_project_root, project_root_from_path
from p64.hub.registry import ProjectRegistry, file_association_command, project_file_path


def launch_hub(open_path: Path | None = None) -> None:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtWidgets import (
            QApplication,
            QFileDialog,
            QHBoxLayout,
            QInputDialog,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover - optional GUI package
        raise RuntimeError("Install PySide6 to use the P64 Hub.") from exc

    class HubWindow(QMainWindow):
        def __init__(self, initial_path: Path | None = None) -> None:
            super().__init__()
            self.registry = ProjectRegistry.load()
            self.setWindowTitle("P64 Hub")
            self.resize(880, 620)

            root = QWidget()
            layout = QVBoxLayout(root)

            title = QLabel("P64 Hub")
            title.setStyleSheet("font-size: 24px; font-weight: 700;")
            layout.addWidget(title)

            actions = QHBoxLayout()
            add_file = QPushButton("Add Project File")
            add_file.clicked.connect(self._add_project_file)
            add_folder = QPushButton("Add Project Folder")
            add_folder.clicked.connect(self._add_project_folder)
            create = QPushButton("Create Project")
            create.clicked.connect(self._create_project)
            sample = QPushButton("Open Sample Project")
            sample.clicked.connect(self._open_sample)
            refresh = QPushButton("Refresh")
            refresh.clicked.connect(self._refresh)
            associate = QPushButton("File Association")
            associate.clicked.connect(self._show_file_association)
            for button in [add_file, add_folder, create, sample, refresh, associate]:
                actions.addWidget(button)
            layout.addLayout(actions)

            self.scroll = QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.list_widget = QWidget()
            self.list_layout = QVBoxLayout(self.list_widget)
            self.scroll.setWidget(self.list_widget)
            layout.addWidget(self.scroll, 1)

            self.status = QLabel("")
            layout.addWidget(self.status)
            self.setCentralWidget(root)

            if initial_path:
                self._add_and_open(initial_path)
            self._refresh()

        def _refresh(self) -> None:
            self.registry = ProjectRegistry.load()
            while self.list_layout.count():
                child = self.list_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            if not self.registry.projects:
                empty = QLabel("No projects yet. Add an existing project or create a new one.")
                empty.setAlignment(Qt.AlignCenter)
                self.list_layout.addWidget(empty)
            for project in self.registry.projects:
                self.list_layout.addWidget(self._project_row(project))
            self.list_layout.addStretch(1)

        def _project_row(self, project: Any) -> QWidget:
            row = QWidget()
            layout = QHBoxLayout(row)
            open_button = QPushButton(project.name)
            open_button.setMinimumHeight(44)
            open_button.clicked.connect(lambda checked=False, path=project.path: self._open_project(path))
            if not project.exists:
                open_button.setText(f"{project.name}  (missing)")
                open_button.setEnabled(False)
            details = QLabel(str(project.path))
            details.setTextInteractionFlags(Qt.TextSelectableByMouse)
            remove = QPushButton("Remove")
            remove.clicked.connect(lambda checked=False, path=project.path: self._remove_project(path))
            delete = QPushButton("Delete")
            delete.clicked.connect(lambda checked=False, path=project.path: self._delete_project(path))
            layout.addWidget(open_button, 2)
            layout.addWidget(details, 3)
            layout.addWidget(remove)
            layout.addWidget(delete)
            return row

        def _add_project_file(self) -> None:
            path, _filter = QFileDialog.getOpenFileName(self, "Add P64 Project", "", "P64 Project (*.p64);;Legacy P64 Project (*.json)")
            if path:
                self._add_project(Path(path))

        def _add_project_folder(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Add P64 Project Folder")
            if path:
                self._add_project(Path(path))

        def _create_project(self) -> None:
            folder = QFileDialog.getExistingDirectory(self, "Choose Project Parent Folder")
            if not folder:
                return
            name, ok = QInputDialog.getText(self, "Create Project", "Project name:")
            if not ok or not name.strip():
                return
            try:
                entry = self.registry.create_project(Path(folder) / name.strip(), name=name.strip())
                self.status.setText(f"Created {entry.path}")
                self._refresh()
            except Exception as exc:
                QMessageBox.critical(self, "Create failed", str(exc))

        def _open_sample(self) -> None:
            sample = Path.cwd() / "samples" / "FirstScene"
            self._add_and_open(sample)

        def _add_project(self, path: Path) -> None:
            try:
                entry = self.registry.add(path)
                self.status.setText(f"Added {entry.path}")
                self._refresh()
            except Exception as exc:
                QMessageBox.critical(self, "Add failed", str(exc))

        def _add_and_open(self, path: Path) -> None:
            try:
                root = project_root_from_path(path)
                self.registry.mark_opened(root)
                self._open_project(root)
            except Exception as exc:
                QMessageBox.critical(self, "Open failed", str(exc))

        def _open_project(self, path: Path) -> None:
            try:
                root = project_root_from_path(path)
                if not is_project_root(root):
                    raise ValueError(f"Missing {PROJECT_FILE}: {root}")
                self.registry.mark_opened(root)
                command = _editor_command(project_file_path(root))
                subprocess.Popen(command, cwd=str(root))
                self.status.setText(f"Opened {root}")
            except Exception as exc:
                QMessageBox.critical(self, "Open failed", str(exc))

        def _remove_project(self, path: Path) -> None:
            if self.registry.remove(path):
                self.status.setText(f"Removed {path}")
                self._refresh()

        def _delete_project(self, path: Path) -> None:
            result = QMessageBox.warning(
                self,
                "Delete project",
                f"This will permanently delete:\n{path}\n\nType Delete in the next dialog to confirm.",
                QMessageBox.Ok | QMessageBox.Cancel,
            )
            if result != QMessageBox.Ok:
                return
            text, ok = QInputDialog.getText(self, "Confirm Delete", 'Type "Delete" to delete this project:')
            if not ok or text != "Delete":
                return
            try:
                self.registry.delete_project(path)
                self.status.setText(f"Deleted {path}")
                self._refresh()
            except Exception as exc:
                QMessageBox.critical(self, "Delete failed", str(exc))

        def _show_file_association(self) -> None:
            executable = Path(sys.executable)
            command = file_association_command(executable)
            QGuiApplication.clipboard().setText(command)
            QMessageBox.information(
                self,
                "File association",
                "Run this in a Windows Command Prompt to associate .p64 files with this Hub. The command was copied to the clipboard:\n\n"
                + command,
            )

    app = QApplication.instance() or QApplication([])
    window = HubWindow(open_path)
    window.show()
    app.exec()


def _editor_command(project_file: Path) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--editor", str(project_file)]
    return [sys.executable, "-m", "p64", "editor", str(project_file)]
