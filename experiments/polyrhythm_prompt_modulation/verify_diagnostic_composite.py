#!/usr/bin/env python3
"""Verify that a diagnostic composite follows the saved prompt weights."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "polyrhythm_prompt_modulation"
RUNWAY_DIR = OUTPUT_DIR / "runway"
DEFAULT_MODULATION = Path(__file__).resolve().parent / "data" / "diagnostic_weight_switch_8s.json"
DEFAULT_COMPOSITE = OUTPUT_DIR / "diagnostic_weight_switch_8s.mp4"
DEFAULT_REPORT = OUTPUT_DIR / "diagnostic_weight_switch_8s.verify.json"
SLOT_IDS = ["tension_chords", "glassy_arps", "broken_beat", "granular_texture"]
SAMPLE_TIMES = [0.72, 2.72, 4.72, 6.72]


def default_video_paths() -> list[Path]:
    return [
        RUNWAY_DIR / "slot_01_tension_chords.mp4",
        RUNWAY_DIR / "slot_02_glassy_arps.mp4",
        RUNWAY_DIR / "slot_03_broken_beat.mp4",
        RUNWAY_DIR / "slot_04_granular_texture.mp4",
    ]


def extract_frame(video: Path, time_seconds: float, output: Path, loop: bool = False) -> np.ndarray:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if loop:
        command += ["-stream_loop", "-1"]
    command += [
        "-i",
        str(video),
        "-ss",
        f"{time_seconds:.4f}",
        "-frames:v",
        "1",
        "-vf",
        "scale=320:180,format=rgb24",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        str(output),
    ]
    subprocess.run(command, check=True)
    data = output.read_bytes()
    if len(data) != 320 * 180 * 3:
        raise RuntimeError(f"Unexpected frame size from {video}: {len(data)} bytes")
    return np.frombuffer(data, dtype=np.uint8).reshape((180, 320, 3)).astype(np.float32)


def frame_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def modulation_at(modulation: dict[str, Any], time_seconds: float) -> dict[str, Any]:
    fps = int(modulation["metadata"]["fps"])
    index = min(len(modulation["frames"]) - 1, max(0, int(round(time_seconds * fps))))
    return modulation["frames"][index]


def verify(modulation: dict[str, Any], composite: Path, videos: list[Path]) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    passed = True
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for sample_index, time_seconds in enumerate(SAMPLE_TIMES):
            composite_frame = extract_frame(composite, time_seconds, tmpdir / f"composite_{sample_index}.raw")
            row = modulation_at(modulation, time_seconds)
            expected_slot = row["dominant_slot"]
            expected_index = SLOT_IDS.index(expected_slot)
            distances = []
            for video_index, video in enumerate(videos):
                source_frame = extract_frame(video, time_seconds, tmpdir / f"source_{sample_index}_{video_index}.raw", loop=True)
                distances.append(frame_mae(composite_frame, source_frame))
            best_index = min(range(len(distances)), key=lambda index: distances[index])
            sample_passed = best_index == expected_index
            passed = passed and sample_passed
            samples.append(
                {
                    "time_seconds": time_seconds,
                    "expected_slot": expected_slot,
                    "expected_weight": row["weights"][expected_slot],
                    "best_matching_slot": SLOT_IDS[best_index],
                    "mae_to_sources": {
                        slot_id: round(distances[index], 4)
                        for index, slot_id in enumerate(SLOT_IDS)
                    },
                    "passed": sample_passed,
                }
            )

    sums_ok = all(abs(sum(frame["weights_array"]) - 1.0) < 0.00001 for frame in modulation["frames"])
    solo_ok = all(sample["expected_weight"] >= 0.99 for sample in samples)
    return {
        "schema": "mrt-diagnostic-composite-verification-v1",
        "composite": str(composite),
        "modulation": str(DEFAULT_MODULATION),
        "videos": [str(video) for video in videos],
        "weight_sums_ok": sums_ok,
        "solo_sample_weights_ok": solo_ok,
        "source_match_ok": passed,
        "passed": bool(sums_ok and solo_ok and passed),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulation", type=Path, default=DEFAULT_MODULATION)
    parser.add_argument("--composite", type=Path, default=DEFAULT_COMPOSITE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--video", action="append", type=Path)
    args = parser.parse_args()

    videos = args.video or default_video_paths()
    modulation = json.loads(args.modulation.read_text(encoding="utf-8"))
    report = verify(modulation, args.composite, videos)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.report)
    if not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
