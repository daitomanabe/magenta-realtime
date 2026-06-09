#!/usr/bin/env python3
"""Render 50 five-minute dirty cinematic Cm9 meter+macro prompt-mix takes."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
from pathlib import Path

import run_sustained_synth_sources as sustained

try:
    import numpy as np
except ImportError:  # pragma: no cover - slow fallback for minimal Python envs.
    np = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT / "build/examples/midi_prompt_diagnostic/mrt2_midi_prompt_diagnostic"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/polyrhythm_prompt_modulation/dirty_cinematic_cm9_5min_batch"
BPM = 120
BEATS_PER_BAR = 4
DURATION_SECONDS = 300.0
MACRO_REFERENCE_SECONDS = 128.0
SAMPLE_RATE = 48000
CHANNELS = 2
BYTES_PER_SAMPLE = 4
WEIGHT_FRAME_RATE = 60
CHORD = [36, 43, 48, 51, 55, 58, 62, 67]


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


def write_cm9_hold_midi(path: Path, duration_seconds: float) -> None:
    ppq = 480
    tempo_us = round(60_000_000 / BPM)
    beats = int(round(duration_seconds * BPM / 60.0))
    end_ticks = beats * ppq
    track = bytearray()

    def add(delta: int, event: bytes) -> None:
        track.extend(vlq(delta))
        track.extend(event)

    name = f"Dirty Cinematic Cm9 {duration_seconds:.0f}s BPM120 hold".encode("ascii")
    add(0, b"\xff\x03" + vlq(len(name)) + name)
    add(0, b"\xff\x51\x03" + tempo_us.to_bytes(3, "big"))
    add(0, b"\xff\x58\x04" + bytes([4, 2, 24, 8]))
    for note in CHORD:
        add(0, bytes([0x90, note, 84]))
    for index, note in enumerate(CHORD):
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


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(f"Render failed with exit code {proc.returncode}: {log_path}\n{tail}")


def find_float_wav_data(path: Path) -> tuple[dict[str, int], int, int]:
    with path.open("rb") as f:
        header = f.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise RuntimeError(f"{path} is not a RIFF/WAVE file")
        fmt: dict[str, int] | None = None
        data_offset = -1
        data_size = -1
        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[:4]
            chunk_size = int.from_bytes(chunk_header[4:8], "little")
            chunk_start = f.tell()
            if chunk_id == b"fmt ":
                chunk = f.read(chunk_size)
                audio_format, channels, sample_rate, _, block_align, bits = struct.unpack_from(
                    "<HHIIHH", chunk, 0
                )
                fmt = {
                    "audio_format": audio_format,
                    "channels": channels,
                    "sample_rate": sample_rate,
                    "block_align": block_align,
                    "bits": bits,
                }
                if chunk_size & 1:
                    f.seek(1, 1)
            elif chunk_id == b"data":
                data_offset = chunk_start
                data_size = chunk_size
                f.seek(chunk_size + (chunk_size & 1), 1)
            else:
                f.seek(chunk_size + (chunk_size & 1), 1)
        if fmt is None or data_offset < 0:
            raise RuntimeError(f"{path} is missing fmt or data chunk")
        if fmt["audio_format"] != 3 or fmt["bits"] != 32:
            raise RuntimeError(f"{path} must be 32-bit float WAV, got {fmt}")
        return fmt, data_offset, data_size


def analyze_float_wav_stream(path: Path, window_seconds: float = 1.0) -> dict:
    fmt, data_offset, data_size = find_float_wav_data(path)
    channels = fmt["channels"]
    sample_rate = fmt["sample_rate"]
    bytes_per_frame = channels * 4
    total_frames = data_size // bytes_per_frame
    window_frames = max(1, int(round(sample_rate * window_seconds)))
    windows = []

    if np is not None:
        samples = np.memmap(
            path,
            dtype="<f4",
            mode="r",
            offset=data_offset,
            shape=(total_frames, channels),
        )
        frame_start = 0
        while frame_start < total_frames:
            frames_to_read = min(window_frames, total_frames - frame_start)
            mono = samples[frame_start:frame_start + frames_to_read].mean(axis=1)
            rms = float(np.sqrt(np.mean(np.square(mono)))) if frames_to_read else 0.0
            peak = float(np.max(np.abs(mono))) if frames_to_read else 0.0
            windows.append({
                "start_seconds": frame_start / sample_rate,
                "end_seconds": (frame_start + frames_to_read) / sample_rate,
                "rms": rms,
                "peak": peak,
            })
            frame_start += frames_to_read
        del samples
    else:
        with path.open("rb") as f:
            f.seek(data_offset)
            frame_start = 0
            while frame_start < total_frames:
                frames_to_read = min(window_frames, total_frames - frame_start)
                chunk = f.read(frames_to_read * bytes_per_frame)
                values = struct.unpack("<" + "f" * (len(chunk) // 4), chunk)
                sum_sq = 0.0
                peak = 0.0
                count = 0
                for frame in range(frames_to_read):
                    base = frame * channels
                    mono = sum(values[base + channel] for channel in range(channels)) / channels
                    sum_sq += mono * mono
                    peak = max(peak, abs(mono))
                    count += 1
                rms = math.sqrt(sum_sq / count) if count else 0.0
                windows.append({
                    "start_seconds": frame_start / sample_rate,
                    "end_seconds": (frame_start + frames_to_read) / sample_rate,
                    "rms": rms,
                    "peak": peak,
                })
                frame_start += frames_to_read

    post_start_windows = [window for window in windows if window["start_seconds"] >= 3.0]
    min_post_start_rms = min((window["rms"] for window in post_start_windows), default=0.0)
    first_3s_windows = [window for window in windows if window["start_seconds"] < 3.0]
    last_8s_start = max(0.0, (total_frames / sample_rate) - 8.0)
    last_8s_windows = [window for window in windows if window["start_seconds"] >= last_8s_start]
    first_3s_rms = (
        sum(window["rms"] for window in first_3s_windows) / len(first_3s_windows)
        if first_3s_windows else 0.0
    )
    last_8s_median_rms = (
        sorted(window["rms"] for window in last_8s_windows)[len(last_8s_windows) // 2]
        if last_8s_windows else 0.0
    )
    quiet_windows = [
        window for window in post_start_windows
        if window["rms"] < 0.0001 or window["peak"] < 0.001
    ]
    return {
        "schema": "mrt-audio-activity-check-v1",
        "path": str(path.relative_to(ROOT)),
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": total_frames / sample_rate,
        "window_seconds": window_seconds,
        "post_start_seconds": 3.0,
        "rms_threshold": 0.0001,
        "peak_threshold": 0.001,
        "first_3s_rms": first_3s_rms,
        "last_8s_median_rms": last_8s_median_rms,
        "min_to_first_3s_rms_ratio": (
            min_post_start_rms / first_3s_rms if first_3s_rms > 0.0 else 0.0
        ),
        "last_8s_to_first_3s_rms_ratio": (
            last_8s_median_rms / first_3s_rms if first_3s_rms > 0.0 else 0.0
        ),
        "min_post_start_rms": min_post_start_rms,
        "quiet_window_count": len(quiet_windows),
        "quiet_windows": quiet_windows[:20],
        "windows": windows,
        "continuous_after_post_start": len(quiet_windows) == 0,
    }


def control_summary(frames: list[dict]) -> dict:
    keys = ["cfg_musiccoca", "cfg_notes", "temperature", "top_k"]
    summary = {}
    for key in keys:
        values = [frame[key] for frame in frames]
        summary[key] = {
            "min": min(values),
            "max": max(values),
            "first": values[0],
            "last": values[-1],
        }
    slot_values = [[frame["weights"][slot] for frame in frames] for slot in range(4)]
    summary["prompt_weights"] = [
        {
            "slot": slot,
            "min": min(values),
            "max": max(values),
            "span": max(values) - min(values),
        }
        for slot, values in enumerate(slot_values)
    ]
    summary["weight_sum_max_error"] = max(
        abs(sum(frame["weights"]) - 1.0) for frame in frames
    )
    return summary


def macro_phase_offset_for_take(take_index: int) -> float:
    return ((take_index - 1) * 0.13750352374993502) % 1.0


def paths_for_take(output_dir: Path, take_index: int) -> dict[str, Path]:
    take_dir = output_dir / f"take_{take_index:03d}"
    stem = f"dirty_cinematic_cm9_take_{take_index:03d}_meter_macro_sine_5min"
    return {
        "dir": take_dir,
        "wav": take_dir / f"{stem}.wav",
        "report": take_dir / f"{stem}.report.json",
        "weights": take_dir / f"{stem}.weights.json",
        "audio_check": take_dir / f"{stem}.audio_check.json",
        "log": take_dir / f"{stem}.render.log",
    }


def validate_take(paths: dict[str, Path], duration_seconds: float) -> dict:
    wav = paths["wav"]
    report = paths["report"]
    weights = paths["weights"]
    audio_check_path = paths["audio_check"]
    probe = sustained.ffprobe(wav)
    report_data = json.loads(report.read_text(encoding="utf-8"))
    weights_data = json.loads(weights.read_text(encoding="utf-8"))
    audio_check = analyze_float_wav_stream(wav.resolve())
    audio_check_path.write_text(json.dumps(audio_check, indent=2) + "\n", encoding="utf-8")

    duration = float(probe["format"]["duration"])
    if abs(duration - duration_seconds) > 0.05:
        raise RuntimeError(f"Duration mismatch: {duration}")
    if report_data.get("weight_mode") != "meter_macro_sine":
        raise RuntimeError(f"Unexpected weight_mode: {report_data.get('weight_mode')}")
    if weights_data.get("frame_rate") != WEIGHT_FRAME_RATE:
        raise RuntimeError(f"Unexpected weight frame rate: {weights_data.get('frame_rate')}")
    expected_weight_frames = int(round(duration_seconds * WEIGHT_FRAME_RATE))
    frames = weights_data.get("frames", [])
    if len(frames) != expected_weight_frames:
        raise RuntimeError(f"Weight frame count mismatch: {len(frames)}")
    if "meter_sine" not in weights_data or "macro_sine" not in weights_data:
        raise RuntimeError("Weight JSON is missing meter_sine or macro_sine metadata")
    if not report_data.get("non_silent", False):
        raise RuntimeError("Rendered WAV is silent")
    if not audio_check["continuous_after_post_start"]:
        raise RuntimeError(f"Quiet windows found: {audio_check['quiet_windows'][:5]}")

    controls = control_summary(frames)
    for slot in controls["prompt_weights"]:
        if slot["span"] < 0.10:
            raise RuntimeError(f"Prompt slot {slot['slot']} modulation span is too small: {slot}")

    return {
        "duration_seconds": duration,
        "wav": str(wav.relative_to(ROOT)),
        "report": str(report.relative_to(ROOT)),
        "weights": str(weights.relative_to(ROOT)),
        "audio_check": str(audio_check_path.relative_to(ROOT)),
        "log": str(paths["log"].relative_to(ROOT)),
        "ffprobe": probe,
        "peak": report_data["peak"],
        "rms": report_data["rms"],
        "non_silent": report_data["non_silent"],
        "audio_activity": {
            "continuous_after_3s": audio_check["continuous_after_post_start"],
            "quiet_window_count": audio_check["quiet_window_count"],
            "min_post_3s_rms": audio_check["min_post_start_rms"],
            "last_8s_to_first_3s_rms_ratio": audio_check["last_8s_to_first_3s_rms_ratio"],
        },
        "meter_sine": weights_data["meter_sine"],
        "macro_sine": weights_data["macro_sine"],
        "controls": controls,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--duration", type=float, default=DURATION_SECONDS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    if not args.binary.exists():
        raise SystemExit(f"Missing binary: {args.binary}")
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    midi = output_dir / "dirty_cinematic_cm9_5min_bpm120_hold.mid"
    write_cm9_hold_midi(midi, args.duration)

    free_bytes = shutil.disk_usage(output_dir).free
    estimated_wav_bytes = args.count * args.duration * SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE
    estimated_weight_bytes = args.count * args.duration * WEIGHT_FRAME_RATE * 700
    print(
        "Storage preflight: "
        f"free={free_bytes / (1024 ** 3):.2f} GiB, "
        f"estimated_batch={((estimated_wav_bytes + estimated_weight_bytes) / (1024 ** 3)):.2f} GiB",
        flush=True,
    )

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema": "mrt-dirty-cinematic-cm9-5min-batch-v1",
        "profile": "dirty_cinematic_ambient_cm9",
        "weight_mode": "meter_macro_sine",
        "control_mode": "ambient_crescendo",
        "bpm": BPM,
        "bars": args.duration * BPM / 60.0 / BEATS_PER_BAR,
        "duration_seconds": args.duration,
        "count_requested": args.count,
        "start_index": args.start_index,
        "macro_reference_seconds": MACRO_REFERENCE_SECONDS,
        "midi": str(midi.relative_to(ROOT)),
        "control_targets": {
            "cfg_musiccoca": [5.8, 7.8],
            "cfg_notes": [5.6, 6.4],
            "temperature": [0.6075, 0.7205],
            "top_k": [51, 103],
            "buffer_chunk": "fixed 25 Hz frames / 1920 sample chunks",
            "window_rms_stabilization": {
                "enabled": True,
                "min_window_rms": 0.012,
                "max_window_gain": 1024,
            },
        },
        "mix_modulation": {
            "micro": "BPM120 meter_sine for 4/4, 3/4, 6/8, 4/4 phase offset 0.25",
            "macro": "128-second macro sine cycles multiplied into the prompt mix",
        },
        "takes": [],
    }

    for take_index in range(args.start_index, args.start_index + args.count):
        paths = paths_for_take(output_dir, take_index)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        phase_offset = macro_phase_offset_for_take(take_index)
        print(f"take {take_index:03d}: macro_phase_offset={phase_offset:.6f}", flush=True)

        existing = all(paths[key].exists() for key in ["wav", "report", "weights"])
        if args.force:
            existing = False
        if not existing:
            if args.skip_render:
                raise RuntimeError(f"Missing rendered files for take {take_index:03d}")
            run_logged(
                [
                    str(args.binary),
                    "--midi",
                    str(midi),
                    "--profile",
                    "dirty_cinematic_ambient_cm9",
                    "--duration",
                    f"{args.duration:.3f}",
                    "--weight-mode",
                    "meter_macro_sine",
                    "--control-mode",
                    "ambient_crescendo",
                    "--macro-reference-seconds",
                    f"{MACRO_REFERENCE_SECONDS:.3f}",
                    "--macro-phase-offset",
                    f"{phase_offset:.9f}",
                    "--batch-variant",
                    str(take_index),
                    "--transition",
                    "0.000",
                    "--stabilize-window-rms",
                    "--min-window-rms",
                    "0.012",
                    "--max-window-gain",
                    "1024",
                    "--output",
                    str(paths["wav"]),
                    "--report",
                    str(paths["report"]),
                    "--weights-output",
                    str(paths["weights"]),
                    "--text-prompts",
                ],
                paths["log"],
            )
        else:
            print(f"take {take_index:03d}: existing render found, validating", flush=True)

        summary = validate_take(paths, args.duration)
        summary.update({
            "take_index": take_index,
            "macro_phase_offset": phase_offset,
            "batch_variant": take_index,
        })
        manifest["takes"].append(summary)
        manifest["completed_count"] = len(manifest["takes"])
        write_manifest(manifest_path, manifest)
        print(
            f"take {take_index:03d}: ok "
            f"rms={summary['rms']:.6f} "
            f"quiet={summary['audio_activity']['quiet_window_count']}",
            flush=True,
        )

    write_manifest(manifest_path, manifest)
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
