"""Événements temps réel pour le moniteur Web (journal + états par job)."""

from __future__ import annotations

import time
from typing import Any

MAX_PIPELINE_ACTIVITY = 200


def push_pipeline_event(
    stats: dict[str, Any],
    *,
    kind: str,
    file: str,
    **extra: Any,
) -> None:
    activity = stats.get("activity")
    lock = stats.get("lock")
    if activity is None or lock is None:
        return
    row: dict[str, Any] = {"t": time.time(), "kind": kind, "file": file or "", **extra}
    with lock:
        activity.append(row)
        while len(activity) > MAX_PIPELINE_ACTIVITY:
            activity.pop(0)
