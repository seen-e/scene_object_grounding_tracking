#!/usr/bin/env python3
from __future__ import annotations

import argparse

from scene_object_grounding_tracking import ObjectBBoxPipeline
from scene_object_grounding_tracking.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ground Scene objects to video bboxes and propagate them across frames"
    )
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file")
    parser.add_argument("--window-limit", type=int)
    parser.add_argument("--object-limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = ObjectBBoxPipeline(load_config(args.config))
    result = pipeline.run(
        window_limit=args.window_limit,
        object_limit=args.object_limit,
        skip_existing=args.skip_existing,
        detect_only=args.detect_only,
        render=not args.no_render,
        dry_run=args.dry_run,
    )
    windows = result.get("windows", [])
    object_count = sum(len(item.get("objects", item.get("tracks", []))) for item in windows)
    print(f"done windows={len(windows)} objects={object_count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
