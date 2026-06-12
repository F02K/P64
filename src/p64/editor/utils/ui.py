from __future__ import annotations

from typing import Any


def make_widget_compact(widget: Any, size_policy: Any) -> None:
    widget.setSizePolicy(size_policy.Policy.Preferred, size_policy.Policy.Maximum)
