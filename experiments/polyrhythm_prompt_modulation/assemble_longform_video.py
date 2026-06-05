#!/usr/bin/env python3
"""Assemble segmented Runway clips with the continuous 180 second WAV."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "polyrhythm_prompt_modulation" / "longform"
DEFAULT_RUNWAY_DIR = DEFAULT_OUTPUT_DIR / "runway_segments"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_RUNWAY_DIR / "longform_runway_manifest.json")
    parser.add_argument("--audio", type=Path, default=DEFAULT_OUTPUT_DIR / "magenta_longform_polyrhythm_180s.wav")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "magenta_longform_polyrhythm_video_180s.mp4")
    parser.add_argument("--duration", type=float, default=180.0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    clips = [Path(segment["clip_path"]) for segment in sorted(manifest["segments"], key=lambda item: item["index"])]
    if not clips:
        raise SystemExit("No clips found in manifest.")
    for clip in clips:
        if not clip.exists():
            raise FileNotFoundError(clip)
    if not args.audio.exists():
        raise FileNotFoundError(args.audio)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    concat_path = args.output.with_suffix(".concat.txt")
    concat_path.write_text("".join(f"file '{clip.as_posix()}'\n" for clip in clips), encoding="utf-8")

    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-i",
            str(args.audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            f"{args.duration:.3f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(args.output),
        ]
    )

    report = {
        "schema": "mrt-longform-video-assembly-report-v1",
        "output": str(args.output),
        "audio": str(args.audio),
        "manifest": str(args.manifest),
        "clip_count": len(clips),
        "probe": ffprobe(args.output),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
