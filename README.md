# Scene Object Grounding Tracking

This standalone pipeline binds objects produced by a Scene stage to video bounding boxes, refines the boxes with a VLM, and propagates them across frames.

## Inputs

The pipeline requires one video and a Scene output directory with this layout:

```text
<scene_output_dir>/
  window_0000/
    scene.json
    context_after_stages.json  # or frames.json
```

Each Scene object must contain `object_id`. Descriptive fields such as `category`, `color`, `texture`, `shape`, `description`, `first_view_time`, and `final_view_time` improve identity grounding and determine its active time range.

Set `bbox_tracking.object_source` to:

- `objects` to process every object in `scene.json`.
- `interaction_objects` to process only object IDs listed by the Scene interaction set.

Window bounds are read from `context_after_stages.json`. If it is absent, the loader uses timestamps from `frames.json`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default `auto` tracking backend uses CoTracker when its source directory and checkpoint are available. Otherwise it falls back to OpenCV optical flow. Install PyTorch separately when using CoTracker.

## Configuration

Copy `configs/default.yaml`, set the video, Scene output, output paths, and model name, then export the credentials referenced by the file:

```bash
export VLM_BASE_URL='https://example.com/v1'
export VLM_API_KEY='...'
```

Direct string values are also accepted in custom YAML files. Environment references use `${NAME}` syntax and must be defined before startup.

## Run

```bash
python run.py --config configs/default.yaml
```

Useful options:

```text
--window-limit N
--object-limit N
--skip-existing
--detect-only
--no-render
--dry-run
```

`--detect-only` runs VLM anchor localization without temporal propagation. `--dry-run` validates inputs and writes only the tracking plan.

## Outputs

```text
tracking_plan.json
tracking_results.json
window_xxxx/tracking_results.json
window_xxxx/bbox_tracking.mp4
window_xxxx/objects/<object_id>/track.json
window_xxxx/objects/<object_id>/anchors/frame_xxxxxx/final_bbox.json
```

Pixel bboxes use `[left, top, right, bottom]` in the original full-frame coordinate system. `bbox_1000` uses the same coordinate system normalized to `0-1000`.
