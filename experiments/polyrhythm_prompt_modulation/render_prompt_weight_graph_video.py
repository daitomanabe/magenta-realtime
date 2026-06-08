#!/usr/bin/env python3
"""Render a prompt-weight JSON file as an animated graph video."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = [
    (255, 90, 90),
    (80, 190, 255),
    (255, 205, 70),
    (170, 120, 255),
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_frame(data: dict, frame_index: int, width: int, height: int) -> Image.Image:
    frames = data["frames"]
    slots = data["slots"]
    duration = float(data["duration_seconds"])
    bpm = float(data.get("bpm", 120))
    weight_mode = data.get("weight_mode", "weights")
    current = frames[frame_index]
    weights = current["weights"]

    image = Image.new("RGB", (width, height), (12, 14, 18))
    draw = ImageDraw.Draw(image)
    title_font = load_font(34)
    label_font = load_font(22)
    small_font = load_font(16)
    value_font = load_font(26)

    margin = 72
    graph_left = margin
    graph_top = 92
    graph_right = width - 340
    graph_bottom = height - 86
    graph_width = graph_right - graph_left
    graph_height = graph_bottom - graph_top
    bar_left = graph_right + 54
    bar_right = width - 70

    draw.text((margin, 28),
              f"Prompt Weight Modulation: {weight_mode} / {duration:.2f}s / {bpm:g} BPM",
              fill=(235, 238, 242), font=title_font)
    draw.text((margin, 62), data.get("profile", "sustained_synth_textures"),
              fill=(150, 156, 168), font=small_font)

    draw.rectangle((graph_left, graph_top, graph_right, graph_bottom),
                   outline=(76, 82, 94), width=2)
    for step in range(5):
        y = graph_bottom - int(graph_height * step / 4)
        value = step / 4
        draw.line((graph_left, y, graph_right, y), fill=(38, 43, 52), width=1)
        draw.text((graph_left - 44, y - 10), f"{value:.2f}",
                  fill=(120, 126, 138), font=small_font)
    for sec in range(0, int(duration) + 1, 2):
        x = graph_left + int(graph_width * sec / duration)
        draw.line((x, graph_top, x, graph_bottom), fill=(30, 35, 43), width=1)
        draw.text((x - 10, graph_bottom + 12), f"{sec}",
                  fill=(120, 126, 138), font=small_font)

    def point(sample_index: int, slot_index: int) -> tuple[int, int]:
        t = frames[sample_index]["time_seconds"]
        w = frames[sample_index]["weights"][slot_index]
        x = graph_left + int(graph_width * t / duration)
        y = graph_bottom - int(graph_height * w)
        return x, y

    for slot_index, color in enumerate(COLORS):
        points = [point(i, slot_index) for i in range(len(frames))]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)

    playhead_x = graph_left + int(graph_width * current["time_seconds"] / duration)
    draw.line((playhead_x, graph_top - 10, playhead_x, graph_bottom + 10),
              fill=(245, 245, 245), width=3)
    draw.text((playhead_x + 8, graph_top - 34), f"{current['time_seconds']:.2f}s",
              fill=(235, 238, 242), font=small_font)

    draw.text((bar_left, graph_top), "Current Weights", fill=(235, 238, 242), font=label_font)
    bar_top = graph_top + 52
    row_h = 104
    for slot_index, slot in enumerate(slots):
        y = bar_top + slot_index * row_h
        color = COLORS[slot_index]
        label = slot.get("label") or slot.get("id") or f"Slot {slot_index + 1}"
        draw.text((bar_left, y), f"{slot_index + 1}. {label}",
                  fill=(225, 228, 235), font=small_font)
        draw.rectangle((bar_left, y + 30, bar_right, y + 58),
                       outline=(70, 76, 88), width=1)
        fill_w = int((bar_right - bar_left) * weights[slot_index])
        draw.rectangle((bar_left, y + 30, bar_left + fill_w, y + 58),
                       fill=color)
        draw.text((bar_left, y + 64), f"{weights[slot_index]:.3f}",
                  fill=color, font=value_font)

    draw.text((margin, height - 42),
              "Weight curves are normalized each frame. Video is silent and intended for modulation QA.",
              fill=(135, 142, 154), font=small_font)
    return image


def render_video(weights_path: Path, output_path: Path, width: int, height: int) -> None:
    data = json.loads(weights_path.read_text())
    fps = int(round(float(data.get("frame_rate", 25))))
    frames = data["frames"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
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
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    with subprocess.Popen(cmd, cwd=weights_path.parent, stdin=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        for frame_index in range(len(frames)):
            image = draw_frame(data, frame_index, width, height)
            proc.stdin.write(image.tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    render_video(args.weights, args.output, args.width, args.height)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
