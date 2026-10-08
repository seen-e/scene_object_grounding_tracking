from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import cv2

from .scene_index import SceneWindow, load_scene_windows, object_time_range
from .tracking import CoTrackerBBoxPropagator, OpticalFlowBBoxPropagator, build_bbox_propagator, read_frame, read_frame_range, video_info
from .vlm_bbox import VLMBBoxAnnotator, bbox_pixel_to_1000


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "object"


def _anchor_frames(start_sec: float, end_sec: float, interval_sec: float, fps: float, frame_count: int) -> list[int]:
    if interval_sec <= 0:
        raise ValueError("anchor_interval_sec must be positive")
    start_frame = max(0, min(frame_count - 1, round(start_sec * fps)))
    # Scene windows are [start, end). Keep the next window's first frame out of this track.
    last_frame = max(start_frame, min(frame_count - 1, math.ceil(end_sec * fps) - 1))
    times = [start_sec]
    cursor = start_sec + interval_sec
    while cursor < end_sec - 1e-6:
        times.append(cursor)
        cursor += interval_sec
    indices = {max(start_frame, min(last_frame, round(value * fps))) for value in times}
    indices.add(last_frame)
    return sorted(indices)


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ObjectBBoxPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_config = config["model"]
        self.settings = config["bbox_tracking"]
        self.video_path = Path(str(self.settings["video"]))
        self.scene_output_dir = Path(str(self.settings["scene_output_dir"]))
        self.output_dir = Path(str(self.settings["output_dir"]))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video = video_info(self.video_path)
        self.windows = load_scene_windows(
            self.scene_output_dir,
            str(self.settings.get("object_source", "objects")),
            {str(value) for value in self.settings.get("exclude_categories", [])},
        )
        self.annotator: VLMBBoxAnnotator | None = None
        self.propagator: CoTrackerBBoxPropagator | OpticalFlowBBoxPropagator | None = None

    def build_plan(self, window_limit: int | None = None, object_limit: int | None = None) -> dict[str, Any]:
        windows = self.windows[:window_limit] if window_limit is not None else self.windows
        payload: dict[str, Any] = {
            "video": {"path": str(self.video_path), **self.video},
            "scene_output_dir": str(self.scene_output_dir),
            "object_source": self.settings.get("object_source", "objects"),
            "anchor_interval_sec": float(self.settings.get("anchor_interval_sec", 1.0)),
            "identity_scope": "window_local",
            "windows": [],
        }
        for window in windows:
            objects = list(window.objects[:object_limit] if object_limit is not None else window.objects)
            object_plans = []
            for obj in objects:
                start, end = object_time_range(obj, window)
                anchors = _anchor_frames(
                    start,
                    end,
                    float(self.settings.get("anchor_interval_sec", 1.0)),
                    float(self.video["fps"]),
                    int(self.video["frame_count"]),
                )
                object_plans.append(
                    {
                        "track_id": f"{window.window_id}/{obj['object_id']}",
                        "object": obj,
                        "start_time_sec": start,
                        "end_time_sec": end,
                        "anchor_frames": anchors,
                        "anchor_times_sec": [round(index / self.video["fps"], 6) for index in anchors],
                    }
                )
            payload["windows"].append(
                {
                    "window_id": window.window_id,
                    "start_time_sec": window.start_time_sec,
                    "end_time_sec": window.end_time_sec,
                    "objects": object_plans,
                }
            )
        return payload

    def run(
        self,
        window_limit: int | None = None,
        object_limit: int | None = None,
        skip_existing: bool = False,
        detect_only: bool = False,
        render: bool = True,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        plan = self.build_plan(window_limit, object_limit)
        _json_write(self.output_dir / "tracking_plan.json", plan)
        if dry_run:
            return plan

        self.annotator = VLMBBoxAnnotator(self.model_config, self.settings)
        results = {
            "video": plan["video"],
            "scene_output_dir": plan["scene_output_dir"],
            "identity_scope": "window_local",
            "windows": [],
        }
        for window_plan in plan["windows"]:
            print(f"window={window_plan['window_id']} objects={len(window_plan['objects'])}", flush=True)
            window_result = self._run_window(window_plan, skip_existing, detect_only)
            results["windows"].append(window_result)
            window_dir = self.output_dir / window_plan["window_id"]
            _json_write(window_dir / "tracking_results.json", window_result)
            if render and not detect_only:
                self._render_window(window_plan, window_result, window_dir / "bbox_tracking.mp4")
        _json_write(self.output_dir / "tracking_results.json", results)
        return results

    def _run_window(self, window_plan: dict[str, Any], skip_existing: bool, detect_only: bool) -> dict[str, Any]:
        tracks = []
        for order, object_plan in enumerate(window_plan["objects"], start=1):
            print(
                f"  [{order}/{len(window_plan['objects'])}] object={object_plan['object']['object_id']} "
                f"anchors={len(object_plan['anchor_frames'])}",
                flush=True,
            )
            tracks.append(self._run_object(window_plan["window_id"], object_plan, skip_existing, detect_only))
        return {
            "window_id": window_plan["window_id"],
            "start_time_sec": window_plan["start_time_sec"],
            "end_time_sec": window_plan["end_time_sec"],
            "tracks": tracks,
        }

    def _run_object(
        self,
        window_id: str,
        object_plan: dict[str, Any],
        skip_existing: bool,
        detect_only: bool,
    ) -> dict[str, Any]:
        assert self.annotator is not None
        object_id = str(object_plan["object"]["object_id"])
        object_dir = self.output_dir / window_id / "objects" / _safe_name(object_id)
        anchors: dict[int, dict[str, Any]] = {}
        for frame_index in object_plan["anchor_frames"]:
            anchor_dir = object_dir / "anchors" / f"frame_{frame_index:06d}"
            final_path = anchor_dir / "final_bbox.json"
            if skip_existing and final_path.exists():
                result = json.loads(final_path.read_text(encoding="utf-8"))
            else:
                frame = read_frame(self.video_path, frame_index)
                result = self.annotator.annotate(
                    frame,
                    object_plan["object"],
                    {
                        "frame_index": frame_index,
                        "time_sec": round(frame_index / self.video["fps"], 6),
                        "scene_first_view_time": object_plan["object"].get("first_view_time"),
                        "window_id": window_id,
                    },
                    anchor_dir,
                )
            result = {
                **result,
                "frame_index": frame_index,
                "time_sec": round(frame_index / self.video["fps"], 6),
            }
            anchors[frame_index] = result

        if detect_only:
            frames = [
                {**record, "source": record.get("source", "vlm_anchor")}
                for _, record in sorted(anchors.items())
            ]
        else:
            if self.propagator is None:
                self.propagator = build_bbox_propagator(self.settings)
                print(f"bbox propagation device={self.propagator.device}", flush=True)
            frames = self._propagate_anchors(anchors)

        result = {
            "track_id": object_plan["track_id"],
            "object_id": object_id,
            "object": object_plan["object"],
            "start_time_sec": object_plan["start_time_sec"],
            "end_time_sec": object_plan["end_time_sec"],
            "anchors": [anchors[index] for index in sorted(anchors)],
            "frames": frames,
        }
        _json_write(object_dir / "track.json", result)
        return result

    def _propagate_anchors(self, anchors: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        assert self.propagator is not None
        indices = sorted(anchors)
        frame_records: dict[int, dict[str, Any]] = {}
        for index in indices:
            anchor = anchors[index]
            frame_records[index] = {
                "frame_index": index,
                "time_sec": round(index / self.video["fps"], 6),
                "bbox": anchor.get("bbox"),
                "bbox_1000": anchor.get("bbox_1000"),
                "visible": bool(anchor.get("visible", False)),
                "confidence": anchor.get("round1", {}).get("correction_confidence") if isinstance(anchor.get("round1"), dict) else None,
                "source": "vlm_anchor",
            }
        for start, end in zip(indices, indices[1:]):
            start_record = anchors[start]
            if not start_record.get("visible") or start_record.get("bbox") is None or end <= start:
                continue
            segment_frames = read_frame_range(self.video_path, start, end)
            propagated = self.propagator.propagate(segment_frames, start_record["bbox"])
            for item in propagated[1:]:
                frame_index = start + int(item["local_frame_index"])
                if frame_index == end and bool(anchors[end].get("visible")):
                    continue
                box = item["bbox"]
                frame_records[frame_index] = {
                    "frame_index": frame_index,
                    "time_sec": round(frame_index / self.video["fps"], 6),
                    "bbox": box,
                    "bbox_1000": bbox_pixel_to_1000(box, self.video["width"], self.video["height"]) if box else None,
                    "visible": box is not None,
                    "confidence": item["tracking_confidence"],
                    "visible_points": item["visible_points"],
                    "source": "cotracker",
                }
        return [frame_records[index] for index in sorted(frame_records)]

    def _render_window(self, window_plan: dict[str, Any], result: dict[str, Any], output_path: Path) -> None:
        fps = float(self.video["fps"])
        start = max(0, round(window_plan["start_time_sec"] * fps))
        end = min(int(self.video["frame_count"]) - 1, max(start, math.ceil(window_plan["end_time_sec"] * fps) - 1))
        by_frame: dict[int, list[tuple[str, dict[str, Any]]]] = {}
        for track in result["tracks"]:
            for record in track["frames"]:
                if record.get("visible") and record.get("bbox"):
                    by_frame.setdefault(int(record["frame_index"]), []).append((track["object_id"], record))
        frames = read_frame_range(self.video_path, start, end)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (int(self.video["width"]), int(self.video["height"])),
        )
        colors = [(40, 220, 80), (70, 150, 255), (255, 120, 40), (220, 80, 220), (80, 220, 220)]
        for offset, frame in enumerate(frames):
            frame_index = start + offset
            for order, (object_id, record) in enumerate(by_frame.get(frame_index, [])):
                x1, y1, x2, y2 = record["bbox"]
                color = colors[order % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, object_id, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
            cv2.putText(frame, f"{frame_index / fps:.3f}s", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            writer.write(frame)
        writer.release()
