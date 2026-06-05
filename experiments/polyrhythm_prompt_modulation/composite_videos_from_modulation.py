#!/usr/bin/env python3
"""Blend four videos using the saved prompt modulation data."""

from __future__ import annotations

import argparse
import json
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DEFAULT_MODULATION = EXPERIMENT / "data" / "modulation_10s_104bpm.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "polyrhythm_prompt_modulation"
DEFAULT_RUNWAY_DIR = DEFAULT_OUTPUT_DIR / "runway"


def run_ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def reader(video: Path, width: int, height: int, fps: int) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-stream_loop",
            "-1",
            "-i",
            str(video),
            "-vf",
            f"scale={width}:{height},fps={fps},format=rgb24",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )


def writer(output: Path, width: int, height: int, fps: int, duration: float, audio: Path | None) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
    ]
    if audio:
        command += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    command += [
        "-t",
        f"{duration:.4f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if audio:
        command += ["-c:a", "aac", "-b:a", "192k"]
    command.append(str(output))
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def default_video_paths() -> list[Path]:
    return [
        DEFAULT_RUNWAY_DIR / "slot_01_tension_chords.mp4",
        DEFAULT_RUNWAY_DIR / "slot_02_glassy_arps.mp4",
        DEFAULT_RUNWAY_DIR / "slot_03_broken_beat.mp4",
        DEFAULT_RUNWAY_DIR / "slot_04_granular_texture.mp4",
    ]


def build_video(modulation: dict[str, Any], videos: list[Path], output: Path, audio: Path | None, width: int, height: int) -> None:
    metadata = modulation["metadata"]
    fps = int(metadata["fps"])
    duration = float(metadata["duration_seconds"])
    frame_count = int(metadata["frame_count"])
    frame_size = width * height * 3

    for video in videos:
        if not video.exists():
            raise FileNotFoundError(f"Missing video input: {video}")

    output.parent.mkdir(parents=True, exist_ok=True)
    readers = [reader(video, width, height, fps) for video in videos]
    out = writer(output, width, height, fps, duration, audio)

    try:
        for index in range(frame_count):
            weights = np.array(modulation["frames"][index]["weights_array"], dtype=np.float32)
            mixed = np.zeros((height, width, 3), dtype=np.float32)
            for source, weight in zip(readers, weights):
                assert source.stdout is not None
                data = source.stdout.read(frame_size)
                if len(data) != frame_size:
                    raise RuntimeError(f"Could not read frame {index}; got {len(data)} bytes")
                frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3)).astype(np.float32)
                mixed += frame * weight
            assert out.stdin is not None
            out.stdin.write(np.clip(mixed, 0, 255).astype(np.uint8).tobytes())
    finally:
        if out.stdin:
            out.stdin.close()
        out.wait()
        for source in readers:
            source.terminate()
            with suppress(subprocess.TimeoutExpired):
                source.wait(timeout=2)
            if source.poll() is None:
                source.kill()
                source.wait(timeout=2)

    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg writer failed with code {out.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulation", type=Path, default=DEFAULT_MODULATION)
    parser.add_argument("--video", action="append", type=Path, help="Four input videos, in prompt slot order.")
    parser.add_argument("--audio", type=Path, default=DEFAULT_OUTPUT_DIR / "magenta_polyrhythm_prompt_mod_10s.wav")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "polyrhythm_prompt_video_composite.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    videos = args.video or default_video_paths()
    if len(videos) != 4:
        raise SystemExit("Exactly four videos are required.")
    modulation = json.loads(args.modulation.read_text(encoding="utf-8"))
    audio = args.audio if args.audio and args.audio.exists() else None
    build_video(modulation, videos, args.output, audio, args.width, args.height)

    report = {
        "schema": "mrt-polyrhythm-video-composite-report-v1",
        "output": str(args.output),
        "duration_seconds": run_ffprobe_duration(args.output),
        "modulation": str(args.modulation),
        "videos": [str(video) for video in videos],
        "audio": str(audio) if audio else None,
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
