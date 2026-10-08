from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_time(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER.search(str(value))
    return float(match.group()) if match else None


@dataclass(frozen=True)
class SceneWindow:
    window_id: str
    start_time_sec: float
    end_time_sec: float
    objects: tuple[dict[str, Any], ...]
    source_dir: Path


def _window_bounds(window_dir: Path) -> tuple[float, float]:
    context_path = window_dir / "context_after_stages.json"
    if context_path.exists():
        context = json.loads(context_path.read_text(encoding="utf-8"))
        metadata = context.get("input", {}).get("segment_metadata", {})
        start = parse_time(metadata.get("start_time_sec"))
        end = parse_time(metadata.get("end_time_sec"))
        if start is not None and end is not None:
            return start, end
    frames_path = window_dir / "frames.json"
    if frames_path.exists():
        payload = json.loads(frames_path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("frames", [])
        times = [parse_time(item.get("time_sec")) for item in records if isinstance(item, dict)]
        times = [value for value in times if value is not None]
        if times:
            return min(times), max(times)
    raise ValueError(f"cannot determine time range for {window_dir}")


def _selected_objects(scene: dict[str, Any], source: str) -> list[dict[str, Any]]:
    objects = [item for item in scene.get("objects", []) if isinstance(item, dict)]
    if source == "objects":
        return objects
    if source != "interaction_objects":
        raise ValueError("object_source must be objects or interaction_objects")
    wanted = {
        str(item.get("object_id"))
        for item in scene.get("interaction_objects", [])
        if isinstance(item, dict) and item.get("object_id")
    }
    return [item for item in objects if str(item.get("object_id")) in wanted]


def load_scene_windows(
    scene_output_dir: str | Path,
    object_source: str,
    excluded_categories: set[str],
) -> list[SceneWindow]:
    root = Path(scene_output_dir)
    windows: list[SceneWindow] = []
    for scene_path in sorted(root.glob("window_*/scene.json")):
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        start, end = _window_bounds(scene_path.parent)
        selected = []
        for item in _selected_objects(scene, object_source):
            category = str(item.get("category") or item.get("category_id") or "")
            if category in excluded_categories:
                continue
            if not item.get("object_id"):
                continue
            selected.append(item)
        windows.append(
            SceneWindow(
                window_id=scene_path.parent.name,
                start_time_sec=start,
                end_time_sec=end,
                objects=tuple(selected),
                source_dir=scene_path.parent,
            )
        )
    if not windows:
        raise FileNotFoundError(f"no window_*/scene.json found under {root}")
    return windows


def object_time_range(obj: dict[str, Any], window: SceneWindow) -> tuple[float, float]:
    first = parse_time(obj.get("first_view_time"))
    final = parse_time(obj.get("final_view_time"))
    if final is None:
        final = parse_time(obj.get("final_review_time"))
    start = max(window.start_time_sec, first if first is not None else window.start_time_sec)
    end = min(window.end_time_sec, final if final is not None else window.end_time_sec)
    if end < start:
        end = start
    return start, end

