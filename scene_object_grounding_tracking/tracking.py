from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_frame(video_path: Path, frame_index: int) -> Any:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_index} from {video_path}")
    return frame


def read_frame_range(video_path: Path, start_frame: int, end_frame: int) -> list[Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    expected = end_frame - start_frame + 1
    if len(frames) != expected:
        raise RuntimeError(f"decoded {len(frames)}/{expected} frames for {start_frame}-{end_frame}")
    return frames


def video_info(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    result = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    if result["fps"] <= 0 or result["frame_count"] <= 0:
        raise ValueError(f"invalid video metadata: {result}")
    return result


def sample_bbox_points(box: list[int], count: int, inset_ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box]
    inset_x = (x2 - x1) * inset_ratio
    inset_y = (y2 - y1) * inset_ratio
    x1, x2 = x1 + inset_x, x2 - inset_x
    y1, y2 = y1 + inset_y, y2 - inset_y
    side = max(2, int(np.ceil(np.sqrt(count))))
    xs = np.linspace(x1, x2, side, dtype=np.float32)
    ys = np.linspace(y1, y2, side, dtype=np.float32)
    points = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return points[:count]


def _bbox_from_tracks(
    source_points: np.ndarray,
    target_points: np.ndarray,
    visible: np.ndarray,
    source_box: list[int],
    width: int,
    height: int,
    min_visible_points: int,
) -> tuple[list[int] | None, float]:
    valid = visible.astype(bool) & np.isfinite(target_points).all(axis=1)
    valid &= (target_points[:, 0] >= 0) & (target_points[:, 0] < width)
    valid &= (target_points[:, 1] >= 0) & (target_points[:, 1] < height)
    visible_count = int(valid.sum())
    confidence = visible_count / max(1, len(source_points))
    if visible_count < min_visible_points:
        return None, confidence
    src, dst = source_points[valid], target_points[valid]
    matrix = None
    if len(src) >= 3:
        matrix, _inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1000,
            confidence=0.99,
        )
    x1, y1, x2, y2 = source_box
    corners = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    if matrix is not None:
        transformed = cv2.transform(corners[None], matrix)[0]
    else:
        displacement = np.median(dst - src, axis=0)
        transformed = corners + displacement[None]
    left = int(round(np.clip(transformed[:, 0].min(), 0, width - 1)))
    top = int(round(np.clip(transformed[:, 1].min(), 0, height - 1)))
    right = int(round(np.clip(transformed[:, 0].max(), left + 1, width)))
    bottom = int(round(np.clip(transformed[:, 1].max(), top + 1, height)))
    return [left, top, right, bottom], confidence


class CoTrackerBBoxPropagator:
    def __init__(self, config: dict[str, Any]) -> None:
        source = Path(str(config["cotracker_source"]))
        if not source.exists():
            raise FileNotFoundError(source)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        import torch
        from cotracker.predictor import CoTrackerPredictor

        requested = str(config.get("device", "cuda"))
        self.device = torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.model = CoTrackerPredictor(
            checkpoint=str(config["checkpoint"]),
            offline=True,
            window_len=int(config.get("cotracker_window_len", 60)),
        ).to(self.device)
        self.model.eval()
        self.points_per_object = int(config.get("points_per_object", 64))
        self.point_batch_size = int(config.get("point_batch_size", 256))
        self.inset_ratio = float(config.get("sample_inset_ratio", 0.08))
        self.min_visible_points = int(config.get("min_visible_points", 6))

    def _run(self, frames: list[Any], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        video = self.torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2)[None].float().to(self.device)
        tracks_all, visibility_all = [], []
        for start in range(0, len(points), self.point_batch_size):
            end = min(start + self.point_batch_size, len(points))
            query = np.zeros((1, end - start, 3), dtype=np.float32)
            query[0, :, 1:] = points[start:end]
            query_tensor = self.torch.from_numpy(query).to(self.device)
            with self.torch.inference_mode():
                tracks, visibility = self.model(video, queries=query_tensor)
            tracks_all.append(tracks.detach().cpu())
            visibility_all.append(visibility.detach().cpu())
        tracks_np = self.torch.cat(tracks_all, dim=2)[0].numpy()
        visibility_np = self.torch.cat(visibility_all, dim=2)[0].numpy().astype(bool)
        del video
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
        return tracks_np, visibility_np

    def propagate(
        self,
        frames: list[Any],
        start_box: list[int],
    ) -> list[dict[str, Any]]:
        height, width = frames[0].shape[:2]
        points = sample_bbox_points(start_box, self.points_per_object, self.inset_ratio)
        tracks, visibility = self._run(frames, points)
        results = []
        for local_index in range(len(frames)):
            box, confidence = _bbox_from_tracks(
                points,
                tracks[local_index],
                visibility[local_index],
                start_box,
                width,
                height,
                self.min_visible_points,
            )
            results.append(
                {
                    "local_frame_index": local_index,
                    "bbox": box,
                    "tracking_confidence": round(confidence, 6),
                    "visible_points": int(visibility[local_index].sum()),
                }
            )
        return results


class OpticalFlowBBoxPropagator:
    """CPU fallback with the same output contract as CoTrackerBBoxPropagator."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.points_per_object = int(config.get("points_per_object", 64))
        self.inset_ratio = float(config.get("sample_inset_ratio", 0.08))
        self.min_visible_points = int(config.get("min_visible_points", 6))
        self.device = "cpu-optical-flow"

    def propagate(self, frames: list[Any], start_box: list[int]) -> list[dict[str, Any]]:
        height, width = frames[0].shape[:2]
        gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray0)
        x1, y1, x2, y2 = start_box
        inset_x = round((x2 - x1) * self.inset_ratio)
        inset_y = round((y2 - y1) * self.inset_ratio)
        mask[max(0, y1 + inset_y) : min(height, y2 - inset_y), max(0, x1 + inset_x) : min(width, x2 - inset_x)] = 255
        features = cv2.goodFeaturesToTrack(
            gray0,
            maxCorners=self.points_per_object,
            qualityLevel=0.01,
            minDistance=3,
            mask=mask,
        )
        if features is None or len(features) < self.min_visible_points:
            source = sample_bbox_points(start_box, self.points_per_object, self.inset_ratio)
        else:
            source = features.reshape(-1, 2).astype(np.float32)
        current = source.copy()
        alive = np.ones((len(source),), dtype=bool)
        previous_gray = gray0
        results = [
            {
                "local_frame_index": 0,
                "bbox": list(start_box),
                "tracking_confidence": 1.0,
                "visible_points": int(len(source)),
            }
        ]
        for local_index, frame in enumerate(frames[1:], start=1):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            next_points, status, _error = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                gray,
                current.reshape(-1, 1, 2),
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if next_points is None or status is None:
                alive[:] = False
            else:
                current = next_points.reshape(-1, 2)
                alive &= status.reshape(-1).astype(bool)
            box, confidence = _bbox_from_tracks(
                source,
                current,
                alive,
                start_box,
                width,
                height,
                self.min_visible_points,
            )
            results.append(
                {
                    "local_frame_index": local_index,
                    "bbox": box,
                    "tracking_confidence": round(confidence, 6),
                    "visible_points": int(alive.sum()),
                }
            )
            previous_gray = gray
        return results


def build_bbox_propagator(config: dict[str, Any]) -> CoTrackerBBoxPropagator | OpticalFlowBBoxPropagator:
    backend = str(config.get("tracking_backend", "auto"))
    if backend not in {"auto", "cotracker", "optical_flow"}:
        raise ValueError("tracking_backend must be auto, cotracker, or optical_flow")
    if backend in {"auto", "cotracker"}:
        checkpoint = Path(str(config.get("checkpoint", "")))
        source = Path(str(config.get("cotracker_source", "")))
        try:
            cotracker_available = checkpoint.is_file() and source.is_dir()
        except OSError:
            cotracker_available = False
        if cotracker_available:
            return CoTrackerBBoxPropagator(config)
        if backend == "cotracker":
            raise FileNotFoundError(f"CoTracker unavailable: checkpoint={checkpoint}, source={source}")
        print("CoTracker files unavailable; falling back to OpenCV optical flow", flush=True)
    return OpticalFlowBBoxPropagator(config)
