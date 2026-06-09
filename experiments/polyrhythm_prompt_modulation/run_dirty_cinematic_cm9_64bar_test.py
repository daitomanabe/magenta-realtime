#!/usr/bin/env python3
"""Render one 64-bar dirty cinematic ambient Cm9 prompt-modulation test."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import run_sustained_synth_sources as sustained


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT / "build/examples/midi_prompt_diagnostic/mrt2_midi_prompt_diagnostic"
OUTPUT_DIR = ROOT / "outputs/polyrhythm_prompt_modulation/dirty_cinematic_cm9_64bar"
GRAPH_SCRIPT = ROOT / "experiments/polyrhythm_prompt_modulation/render_prompt_weight_graph_video.py"
DOWNSAMPLE_SCRIPT = ROOT / "experiments/polyrhythm_prompt_modulation/downsample_prompt_weight_frames.py"
BPM = 120
BARS = 64
BEATS_PER_BAR = 4
DURATION_SECONDS = BARS * BEATS_PER_BAR * 60 / BPM


def vlq(value: int) -> bytes:
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= ((value & 0x7F) | 0x80)
        value >>= 7
    out = bytearray()
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def write_cm9_hold_midi(path: Path) -> None:
    ppq = 480
    tempo_us = round(60_000_000 / BPM)
    end_ticks = BARS * BEATS_PER_BAR * ppq
    chord = [36, 43, 48, 51, 55, 58, 62, 67]
    track = bytearray()

    def add(delta: int, event: bytes) -> None:
        track.extend(vlq(delta))
        track.extend(event)

    name = b"Dirty Cinematic Cm9 64 bars BPM120 hold"
    add(0, b"\xff\x03" + vlq(len(name)) + name)
    add(0, b"\xff\x51\x03" + tempo_us.to_bytes(3, "big"))
    add(0, b"\xff\x58\x04" + bytes([4, 2, 24, 8]))
    for note in chord:
        add(0, bytes([0x90, note, 84]))
    for index, note in enumerate(chord):
        add(end_ticks if index == 0 else 0, bytes([0x80, note, 0]))
    add(0, b"\xff\x2f\x00")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"MThd")
        f.write((6).to_bytes(4, "big"))
        f.write((0).to_bytes(2, "big"))
        f.write((1).to_bytes(2, "big"))
        f.write(ppq.to_bytes(2, "big"))
        f.write(b"MTrk")
        f.write(len(track).to_bytes(4, "big"))
        f.write(track)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    if not args.binary.exists():
        raise SystemExit(f"Missing binary: {args.binary}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    midi = args.output_dir / "dirty_cinematic_cm9_64bars_bpm120_hold.mid"
    wav = args.output_dir / "dirty_cinematic_cm9_64bars_bpm120_meter_macro_sine.wav"
    report = args.output_dir / "dirty_cinematic_cm9_64bars_bpm120_meter_macro_sine.report.json"
    weights = args.output_dir / "dirty_cinematic_cm9_64bars_bpm120_meter_macro_sine.weights.json"
    audio_check_path = args.output_dir / "dirty_cinematic_cm9_64bars_bpm120_meter_macro_sine.audio_check.json"
    preview_weights = args.output_dir / "dirty_cinematic_cm9_64bars_bpm120_meter_macro_sine.weights.preview_10fps.json"
    graph = args.output_dir / "dirty_cinematic_cm9_64bars_bpm120_meter_macro_sine.graph_10fps.mp4"
    graph_audio = args.output_dir / "dirty_cinematic_cm9_64bars_bpm120_meter_macro_sine.graph_audio_10fps.mp4"
    manifest = args.output_dir / "dirty_cinematic_cm9_64bars_bpm120_meter_macro_sine.manifest.json"

    write_cm9_hold_midi(midi)

    if not args.skip_render:
        run([
            str(args.binary),
            "--midi",
            str(midi),
            "--profile",
            "dirty_cinematic_ambient_cm9",
            "--duration",
            f"{DURATION_SECONDS:.3f}",
            "--weight-mode",
            "meter_macro_sine",
            "--control-mode",
            "ambient_crescendo",
            "--macro-reference-seconds",
            "128.000",
            "--macro-phase-offset",
            "0.000",
            "--batch-variant",
            "0",
            "--transition",
            "0.000",
            "--stabilize-window-rms",
            "--min-window-rms",
            "0.012",
            "--max-window-gain",
            "1024",
            "--output",
            str(wav),
            "--report",
            str(report),
            "--weights-output",
            str(weights),
            "--text-prompts",
        ])

    probe = sustained.ffprobe(wav)
    report_data = json.loads(report.read_text(encoding="utf-8"))
    weights_data = json.loads(weights.read_text(encoding="utf-8"))
    audio_check = sustained.analyze_float_wav(wav.resolve())
    audio_check_path.write_text(json.dumps(audio_check, indent=2) + "\n", encoding="utf-8")

    if abs(float(probe["format"]["duration"]) - DURATION_SECONDS) > 0.05:
        raise RuntimeError(f"Duration mismatch: {probe['format']['duration']}")
    if not report_data.get("non_silent", False):
        raise RuntimeError("Rendered WAV is silent")
    if not audio_check["continuous_after_post_start"]:
        raise RuntimeError(f"Quiet windows found: {audio_check['quiet_windows'][:5]}")
    if weights_data.get("frame_rate") != 60:
        raise RuntimeError(f"Unexpected weight frame rate: {weights_data.get('frame_rate')}")
    expected_weight_frames = int(round(DURATION_SECONDS * 60))
    if len(weights_data.get("frames", [])) != expected_weight_frames:
        raise RuntimeError("Weight frame count mismatch")

    if not args.skip_video:
        run([
            "python3",
            str(DOWNSAMPLE_SCRIPT),
            "--input",
            str(weights),
            "--output",
            str(preview_weights),
            "--fps",
            "10",
        ])
        run([
            "python3",
            str(GRAPH_SCRIPT),
            "--weights",
            str(preview_weights),
            "--output",
            str(graph),
        ])
        run([
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(graph),
            "-i",
            str(wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(graph_audio),
        ])

    manifest.write_text(
        json.dumps(
            {
                "schema": "mrt-dirty-cinematic-cm9-64bar-test-v1",
                "bpm": BPM,
                "bars": BARS,
                "duration_seconds": DURATION_SECONDS,
                "profile": "dirty_cinematic_ambient_cm9",
                "weight_mode": "meter_macro_sine",
                "control_mode": "ambient_crescendo",
                "macro_reference_seconds": 128.0,
                "macro_phase_offset": 0.0,
                "batch_variant": 0,
                "window_rms_stabilization": {
                    "enabled": True,
                    "min_window_rms": 0.012,
                    "max_window_gain": 1024,
                },
                "control_roles": {
                    "primary": "prompt embedding mix with BPM-synced meter_sine multiplied by 128-second macro_sine",
                    "secondary": "CFG weight rises toward the second half",
                    "expression": "temperature moves slowly in a restrained ambient range",
                    "exploration": "top_k moves slowly in a restrained ambient range",
                    "stability": "fixed 25 Hz frames / 1920 sample chunks",
                },
                "midi": str(midi.relative_to(ROOT)),
                "wav": str(wav.relative_to(ROOT)),
                "report": str(report.relative_to(ROOT)),
                "weights": str(weights.relative_to(ROOT)),
                "audio_check": str(audio_check_path.relative_to(ROOT)),
                "preview_weights": str(preview_weights.relative_to(ROOT)) if preview_weights.exists() else None,
                "graph_video": str(graph.relative_to(ROOT)) if graph.exists() else None,
                "graph_audio_video": str(graph_audio.relative_to(ROOT)) if graph_audio.exists() else None,
                "ffprobe": probe,
                "audio_activity": {
                    "continuous_after_3s": audio_check["continuous_after_post_start"],
                    "quiet_window_count": audio_check["quiet_window_count"],
                    "min_post_start_rms": audio_check["min_post_start_rms"],
                    "last_8s_to_first_3s_rms_ratio": audio_check["last_8s_to_first_3s_rms_ratio"],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
