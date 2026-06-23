from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from p64.engine.scene import Scene


@dataclass
class UndoState:
    label: str
    scene_data: dict[str, Any]
    lighting_settings: dict[str, Any]
    selection_id: str | None


class UndoManager:
    def __init__(self, limit: int = 100) -> None:
        self.limit = max(1, limit)
        self._states: list[UndoState] = []
        self._index = -1
        self._saved_index = -1
        self._pending: tuple[str, UndoState] | None = None

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._index < len(self._states) - 1

    @property
    def is_dirty(self) -> bool:
        return self._index != self._saved_index

    @property
    def history_length(self) -> int:
        return len(self._states)

    def clear(self) -> None:
        self._states = []
        self._index = -1
        self._saved_index = -1
        self._pending = None

    def reset(self, scene: Scene | None, selection_id: str | None = None) -> None:
        self.clear()
        if scene is None:
            return
        self._states = [self._state("Initial", scene, selection_id)]
        self._index = 0
        self._saved_index = 0

    def mark_saved(self) -> None:
        self._saved_index = self._index

    def begin(self, label: str, scene: Scene | None, selection_id: str | None = None) -> None:
        if scene is None or self._pending is not None:
            return
        self._pending = (label, self._state(label, scene, selection_id))

    def commit(self, scene: Scene | None, selection_id: str | None = None) -> bool:
        if scene is None:
            self._pending = None
            return False
        label = self._pending[0] if self._pending else "Edit Scene"
        self._pending = None
        return self.record(label, scene, selection_id)

    def record(self, label: str, scene: Scene | None, selection_id: str | None = None) -> bool:
        if scene is None:
            return False
        state = self._state(label, scene, selection_id)
        if self._index >= 0 and self._states[self._index].scene_data == state.scene_data and self._states[self._index].selection_id == state.selection_id:
            return False
        del self._states[self._index + 1 :]
        self._states.append(state)
        if len(self._states) > self.limit:
            self._states.pop(0)
            self._saved_index = max(-1, self._saved_index - 1)
        self._index = len(self._states) - 1
        return True

    def undo(self) -> UndoState | None:
        if not self.can_undo:
            return None
        self._pending = None
        self._index -= 1
        return self._states[self._index]

    def redo(self) -> UndoState | None:
        if not self.can_redo:
            return None
        self._pending = None
        self._index += 1
        return self._states[self._index]

    def _state(self, label: str, scene: Scene, selection_id: str | None) -> UndoState:
        return UndoState(
            label=label,
            scene_data=scene.to_dict(),
            lighting_settings=dict(scene.lighting_settings),
            selection_id=selection_id,
        )
