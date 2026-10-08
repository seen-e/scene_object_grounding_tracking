from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import cv2
from openai import OpenAI

from .prompts.grounding_bbox import (
    GROUNDING_BBOX_DETECT_PROMPT,
    GROUNDING_BBOX_REFINE_PROMPT,
    GROUNDING_BBOX_SYSTEM_PROMPT,
)


EDGE_NAMES = ("left", "top", "right", "bottom")
EDGE_STATUSES = {"accurate", "too_inside", "too_outside", "uncertain"}


def _data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.S).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.S)
    if fenced:
        cleaned = fenced.group(1)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response contains no JSON object")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model response must be a JSON object")
    return payload


def _render(template: str, **values: Any) -> str:
    rendered = template
    for key, value in values.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        rendered = rendered.replace("{{" + key + "}}", text)
    unresolved = re.findall(r"\{\{[^{}]+\}\}", rendered)
    if unresolved:
        raise ValueError(f"unresolved prompt placeholders: {unresolved}")
    return rendered


def validate_bbox_1000(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"invalid bbox_1000: {value!r}")
    x1, y1, x2, y2 = [float(item) for item in value]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    box = [round(max(0, min(1000, item))) for item in (x1, y1, x2, y2)]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"zero-area bbox_1000: {value!r}")
    return box


def bbox_1000_to_pixel(box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    px1 = max(0, min(width - 1, round(x1 * width / 1000)))
    py1 = max(0, min(height - 1, round(y1 * height / 1000)))
    px2 = max(px1 + 1, min(width, round(x2 * width / 1000)))
    py2 = max(py1 + 1, min(height, round(y2 * height / 1000)))
    return [px1, py1, px2, py2]


def bbox_pixel_to_1000(box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        round(x1 * 1000 / width),
        round(y1 * 1000 / height),
        round(x2 * 1000 / width),
        round(y2 * 1000 / height),
    ]


def draw_grid(frame: Any, step: int) -> Any:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    for x in range(0, width, step):
        cv2.line(overlay, (x, 0), (x, height - 1), (80, 80, 80), 1)
        cv2.putText(overlay, str(round(x / width * 1000)), (x + 1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (255, 255, 0), 1)
    for y in range(0, height, step):
        cv2.line(overlay, (0, y), (width - 1, y), (80, 80, 80), 1)
        cv2.putText(overlay, str(round(y / height * 1000)), (1, max(10, y + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (255, 255, 0), 1)
    return cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)


def draw_bbox_overlay(frame: Any, object_id: str, box: list[int]) -> Any:
    canvas = frame.copy()
    x1, y1, x2, y2 = box
    color = (40, 220, 80)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.putText(canvas, object_id, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(canvas, "L", (x1 + 2, (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(canvas, "T", ((x1 + x2) // 2, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(canvas, "R", (max(x1, x2 - 10), (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(canvas, "B", ((x1 + x2) // 2, max(y1 + 12, y2 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return canvas


def local_crop_bounds(
    box: list[int],
    width: int,
    height: int,
    padding_ratio: float,
    min_size: int,
) -> list[int]:
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    crop_width = max(float(min_size), box_width * (1.0 + 2.0 * padding_ratio))
    crop_height = max(float(min_size), box_height * (1.0 + 2.0 * padding_ratio))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = max(0, int(round(center_x - crop_width / 2.0)))
    top = max(0, int(round(center_y - crop_height / 2.0)))
    right = min(width, int(round(center_x + crop_width / 2.0)))
    bottom = min(height, int(round(center_y + crop_height / 2.0)))
    if right - left < min_size:
        left = max(0, min(left, width - min_size))
        right = min(width, left + min_size)
    if bottom - top < min_size:
        top = max(0, min(top, height - min_size))
        bottom = min(height, top + min_size)
    return [left, top, right, bottom]


def resize_long_side(frame: Any, target_long_side: int) -> Any:
    height, width = frame.shape[:2]
    scale = target_long_side / max(height, width)
    if scale <= 1.0:
        return frame.copy()
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


class VLMBBoxAnnotator:
    def __init__(self, model_config: dict[str, Any], tracking_config: dict[str, Any]) -> None:
        self.client = OpenAI(
            base_url=model_config["base_url"],
            api_key=model_config["api_key"],
            timeout=float(model_config.get("timeout", 1200)),
            max_retries=int(model_config.get("max_retries", 2)),
        )
        self.model = str(model_config["model"])
        self.temperature = float(model_config.get("temperature", 0.0))
        self.top_p = float(model_config.get("top_p", 0.95))
        self.max_tokens = int(model_config.get("max_tokens", 4096))
        self.extra_body = dict(model_config.get("extra_body") or {})
        top_k = int(model_config.get("top_k", 0) or 0)
        if top_k > 0:
            self.extra_body["top_k"] = top_k
        self.grid_step = int(tracking_config.get("grid_step_px", 14))
        self.max_delta = int(tracking_config.get("max_refinement_delta_1000", 120))
        self.max_refinement_rounds = int(tracking_config.get("max_refinement_rounds", 3))
        self.convergence_delta = int(tracking_config.get("refinement_convergence_delta_1000", 3))
        self.required_accurate_rounds = int(tracking_config.get("required_consecutive_accurate_rounds", 2))
        self.crop_padding_ratio = float(tracking_config.get("local_crop_padding_ratio", 1.0))
        self.crop_min_size = int(tracking_config.get("local_crop_min_size_px", 64))
        self.crop_long_side = int(tracking_config.get("local_crop_long_side_px", 768))
        if self.max_refinement_rounds < 1:
            raise ValueError("max_refinement_rounds must be at least 1")
        if not 1 <= self.required_accurate_rounds <= self.max_refinement_rounds:
            raise ValueError("required_consecutive_accurate_rounds must be between 1 and max_refinement_rounds")

    def _call(self, prompt: str, image_paths: list[Path], raw_path: Path) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            extra_body=self.extra_body,
            messages=[
                {"role": "system", "content": GROUNDING_BBOX_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                    + [{"type": "image_url", "image_url": {"url": _data_url(path)}} for path in image_paths],
                },
            ],
        )
        text = response.choices[0].message.content or ""
        raw_path.write_text(text, encoding="utf-8")
        return _parse_json(text)

    def annotate(
        self,
        frame: Any,
        object_record: dict[str, Any],
        frame_metadata: dict[str, Any],
        output_dir: Path,
        context_frames: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        original_path = output_dir / "target_frame.jpg"
        grid_path = output_dir / "target_frame_grid.jpg"
        cv2.imwrite(str(original_path), frame)
        cv2.imwrite(str(grid_path), draw_grid(frame, self.grid_step))

        detect_images = [original_path, grid_path]
        context_lines = []
        for context_index, context in enumerate(context_frames or [], start=1):
            context_frame = context.get("frame")
            if context_frame is None:
                raise ValueError("context frame record is missing frame")
            time_sec = float(context.get("time_sec", 0.0))
            context_path = output_dir / f"context_{context_index:02d}_t_{time_sec:.3f}.jpg"
            cv2.imwrite(str(context_path), context_frame)
            detect_images.append(context_path)
            context_lines.append(
                f"Image {context_index + 2}：身份上下文帧，绝对时间 {time_sec:.3f}s，"
                "只用于判断哪个实例是目标，不得在该帧输出 bbox。"
            )
        context_description = "\n".join(context_lines) if context_lines else "本次没有额外身份上下文帧。"
        detect_prompt = _render(
            GROUNDING_BBOX_DETECT_PROMPT,
            OBJECT_JSON=object_record,
            FRAME_JSON={**frame_metadata, "image_width": width, "image_height": height},
            CONTEXT_DESCRIPTION=context_description,
        )
        detect_payload = self._call(detect_prompt, detect_images, output_dir / "round0.raw.txt")
        (output_dir / "round0.json").write_text(json.dumps(detect_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        item = detect_payload.get("object") if isinstance(detect_payload.get("object"), dict) else {}
        if str(item.get("object_id")) != str(object_record.get("object_id")):
            raise ValueError("Round0 object_id does not match Scene object_id")
        if not bool(item.get("visible", False)) or item.get("bbox_1000") is None:
            result = {
                "object_id": object_record["object_id"],
                "visible": False,
                "valid": False,
                "bbox_1000": None,
                "bbox": None,
                "invalid_reason": "round0_not_visible_or_unresolved",
                "source": "vlm_round0",
                "round0": detect_payload,
                "round1": None,
            }
            (output_dir / "final_bbox.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result

        round0_box = validate_bbox_1000(item.get("bbox_1000"))
        round0_pixel = bbox_1000_to_pixel(round0_box, width, height)
        crop_box = local_crop_bounds(
            round0_pixel,
            width,
            height,
            self.crop_padding_ratio,
            self.crop_min_size,
        )
        crop_left, crop_top, crop_right, crop_bottom = crop_box
        crop_source = frame[crop_top:crop_bottom, crop_left:crop_right]
        crop_frame = resize_long_side(crop_source, self.crop_long_side)
        crop_height, crop_width = crop_frame.shape[:2]
        crop_source_height, crop_source_width = crop_source.shape[:2]
        crop_path = output_dir / "local_crop.jpg"
        cv2.imwrite(str(crop_path), crop_frame)
        crop_metadata = {
            **frame_metadata,
            "coordinate_space": "local_crop",
            "crop_bbox_in_full_frame": crop_box,
            "crop_image_width": crop_width,
            "crop_image_height": crop_height,
            "full_image_width": width,
            "full_image_height": height,
            "instruction": "本轮所有 bbox_1000 均相对于局部裁剪图，而不是完整原图。",
        }
        local_grid_path = output_dir / "local_crop_grid.jpg"
        cv2.imwrite(str(local_grid_path), draw_grid(crop_frame, max(32, round(self.grid_step * crop_width / width))))
        local_detect_prompt = _render(
            GROUNDING_BBOX_DETECT_PROMPT,
            OBJECT_JSON=object_record,
            FRAME_JSON=crop_metadata,
            CONTEXT_DESCRIPTION=context_description,
        )
        local_detect_payload = self._call(
            local_detect_prompt,
            [crop_path, local_grid_path, *detect_images[2:]],
            output_dir / "local_round0.raw.txt",
        )
        (output_dir / "local_round0.json").write_text(
            json.dumps(local_detect_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        local_item = local_detect_payload.get("object") if isinstance(local_detect_payload.get("object"), dict) else {}
        if (
            str(local_item.get("object_id")) != str(object_record.get("object_id"))
            or not bool(local_item.get("visible", False))
            or local_item.get("bbox_1000") is None
        ):
            result = {
                "object_id": object_record["object_id"],
                "visible": bool(local_item.get("visible", False)),
                "valid": False,
                "bbox_1000": None,
                "bbox": None,
                "candidate_bbox_1000": round0_box,
                "candidate_bbox": round0_pixel,
                "local_crop_bbox": crop_box,
                "invalid_reason": "local_redetect_not_visible_or_unresolved",
                "source": "vlm_local_round0",
                "round0": detect_payload,
                "local_round0": local_detect_payload,
                "round1": None,
                "refinement_rounds": [],
            }
            (output_dir / "final_bbox.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return result

        current_box = validate_bbox_1000(local_item.get("bbox_1000"))
        seen_boxes = {tuple(current_box)}
        refinement_rounds: list[dict[str, Any]] = []
        status = "max_rounds_reached"
        converged = False
        consecutive_accurate = 0
        for round_index in range(1, self.max_refinement_rounds + 1):
            current_pixel = bbox_1000_to_pixel(current_box, crop_width, crop_height)
            overlay_path = output_dir / f"round{round_index - 1}_bbox_overlay.jpg"
            cv2.imwrite(
                str(overlay_path),
                draw_bbox_overlay(crop_frame, str(object_record["object_id"]), current_pixel),
            )
            refine_prompt = _render(
                GROUNDING_BBOX_REFINE_PROMPT,
                OBJECT_JSON=object_record,
                FRAME_JSON=crop_metadata,
                CURRENT_BBOX_JSON={"object_id": object_record["object_id"], "bbox_1000": current_box},
            )
            refine_payload = self._call(
                refine_prompt,
                [crop_path, overlay_path],
                output_dir / f"round{round_index}.raw.txt",
            )
            (output_dir / f"round{round_index}.json").write_text(
                json.dumps(refine_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            next_box, status = self._apply_refinement(
                current_box,
                refine_payload,
                str(object_record["object_id"]),
            )
            refinement_rounds.append(
                {
                    "round": round_index,
                    "input_bbox_1000": current_box,
                    "output_bbox_1000": next_box,
                    "status": status,
                    "payload": refine_payload,
                }
            )
            if status == "unchanged":
                current_box = next_box
                consecutive_accurate += 1
                if consecutive_accurate >= self.required_accurate_rounds:
                    converged = True
                    break
                continue
            consecutive_accurate = 0
            if status != "corrected":
                current_box = next_box
                break
            if tuple(next_box) in seen_boxes:
                current_box = next_box
                status = "stalled_or_oscillating"
                refinement_rounds[-1]["status"] = status
                break
            current_box = next_box
            seen_boxes.add(tuple(current_box))
        else:
            status = "max_rounds_reached"

        final_local_box = current_box
        final_local_pixel = bbox_1000_to_pixel(final_local_box, crop_width, crop_height)
        final_pixel = [
            crop_left + round(final_local_pixel[0] * crop_source_width / crop_width),
            crop_top + round(final_local_pixel[1] * crop_source_height / crop_height),
            crop_left + round(final_local_pixel[2] * crop_source_width / crop_width),
            crop_top + round(final_local_pixel[3] * crop_source_height / crop_height),
        ]
        final_pixel = [
            max(0, min(width - 1, final_pixel[0])),
            max(0, min(height - 1, final_pixel[1])),
            max(1, min(width, final_pixel[2])),
            max(1, min(height, final_pixel[3])),
        ]
        final_full_box = bbox_pixel_to_1000(final_pixel, width, height)
        cv2.imwrite(
            str(output_dir / "candidate_bbox.jpg"),
            draw_bbox_overlay(frame, str(object_record["object_id"]), final_pixel),
        )
        (output_dir / "refinement_history.json").write_text(
            json.dumps(refinement_rounds, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result = {
            "object_id": object_record["object_id"],
            "visible": True,
            "valid": converged,
            "bbox_1000": final_full_box if converged else None,
            "bbox": final_pixel if converged else None,
            "candidate_bbox_1000": final_full_box,
            "candidate_bbox": final_pixel,
            "local_crop_bbox": crop_box,
            "local_candidate_bbox_1000": final_local_box,
            "invalid_reason": None if converged else status,
            "source": f"vlm_round{len(refinement_rounds)}",
            "refinement_status": status,
            "refinement_converged": converged,
            "refinement_round_count": len(refinement_rounds),
            "round0": detect_payload,
            "local_round0": local_detect_payload,
            "round1": refinement_rounds[0]["payload"] if refinement_rounds else None,
            "refinement_rounds": refinement_rounds,
        }
        visualization_box = final_pixel if converged else round0_pixel
        cv2.imwrite(
            str(output_dir / "final_bbox.jpg"),
            draw_bbox_overlay(frame, str(object_record["object_id"]), visualization_box),
        )
        (output_dir / "final_bbox.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def _apply_refinement(self, original: list[int], payload: dict[str, Any], object_id: str) -> tuple[list[int], str]:
        if str(payload.get("object_id")) != object_id:
            return original, "object_id_mismatch"
        supplied = payload.get("input_bbox_1000")
        try:
            if validate_bbox_1000(supplied) != original:
                return original, "input_bbox_mismatch"
        except (TypeError, ValueError):
            return original, "input_bbox_invalid"
        diagnosis = payload.get("edge_diagnosis")
        if not isinstance(diagnosis, dict):
            return original, "missing_diagnosis"
        deltas: list[int] = []
        statuses: list[str] = []
        for edge in EDGE_NAMES:
            item = diagnosis.get(edge)
            if not isinstance(item, dict) or str(item.get("status")) not in EDGE_STATUSES:
                return original, "invalid_diagnosis"
            edge_status = str(item.get("status"))
            if edge_status == "uncertain":
                return original, "uncertain"
            statuses.append(edge_status)
            try:
                deltas.append(round(float(item.get("delta_1000", 0))))
            except (TypeError, ValueError):
                return original, "invalid_delta"
        if max(abs(value) for value in deltas) > self.max_delta:
            return original, "redetect_required"
        try:
            corrected = validate_bbox_1000([value + delta for value, delta in zip(original, deltas)])
        except ValueError:
            return original, "invalid_corrected_bbox"
        converged = all(value == "accurate" for value in statuses) and max(abs(value) for value in deltas) <= self.convergence_delta
        return corrected, "unchanged" if converged else "corrected"
