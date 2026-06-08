#!/usr/bin/env python3
"""Downsample frame-level prompt-weight JSON for lightweight graph previews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def downsample_weights(input_path: Path, output_path: Path, target_fps: float) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    source_fps = float(data.get("frame_rate", target_fps))
    if target_fps <= 0:
        raise ValueError("target fps must be positive")
    stride = max(1, round(source_fps / target_fps))
    frames = []
    for index, frame in enumerate(data["frames"][::stride]):
        next_frame = dict(frame)
        next_frame["frame"] = index
        frames.append(next_frame)
    data["schema"] = "mrt-cpp-prompt-weight-frames-preview-v1"
    data["source_weights"] = str(input_path)
    data["frame_rate"] = target_fps
    data["frames"] = frames
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    downsample_weights(args.input, args.output, args.fps)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
