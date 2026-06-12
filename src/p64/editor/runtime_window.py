from __future__ import annotations

from p64.engine.project import Project
from p64.engine.runtime_session import RuntimeSession
from p64.engine.scene import Scene
from p64.editor.viewport import create_viewport_class


def launch_runtime_window(project: Project, scene: Scene) -> None:
    try:
        from PySide6.QtCore import QElapsedTimer, QTimer, Qt
        from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
        try:
            from PySide6.QtOpenGLWidgets import QOpenGLWidget
        except ImportError:
            QOpenGLWidget = None
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install PySide6 to run P64 projects.") from exc

    app = QApplication.instance() or QApplication([])
    window = QWidget()
    window.setWindowTitle(project.name)
    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)

    if QOpenGLWidget is None:
        label = QLabel(f"{project.name}\nScene: {scene.name}\nInstall PySide6 OpenGL widgets for accelerated rendering.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        window.resize(960, 720)
        window.show()
        app.exec()
        return

    session = RuntimeSession(project, scene)

    def log(message: str) -> None:
        print(message)

    Viewport = create_viewport_class(QOpenGLWidget, QWidget, QLabel, QVBoxLayout, Qt)
    viewport = Viewport(
        lambda: project,
        lambda: session.scene,
        lambda: None,
        lambda _entity_id: None,
        log,
        lambda: session.input,
    )
    viewport.set_view_mode("Game")
    layout.addWidget(viewport, 1)

    clock = QElapsedTimer()
    clock.start()
    timer = QTimer(window)

    def tick() -> None:
        elapsed_ms = clock.restart()
        dt = max(0.001, elapsed_ms / 1000.0)
        for error in session.tick(dt):
            log(f"Runtime script error: {error}")
        viewport.tick(dt)

    timer.timeout.connect(tick)
    timer.start(16)
    window.resize(960, 720)
    window.show()
    app.exec()
