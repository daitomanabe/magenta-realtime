#!/usr/bin/env python3
"""Render four sustained synth prompt sources and one prompt-weight modulation."""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT / "build/examples/midi_prompt_diagnostic/mrt2_midi_prompt_diagnostic"
DEFAULT_MIDI = ROOT / "assets/Am-Minor Prog 01 (i-VI-v-iv).mid"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/polyrhythm_prompt_modulation/sustained_synth_prompt_sources"
GRAPH_VIDEO_SCRIPT = ROOT / "experiments/polyrhythm_prompt_modulation/render_prompt_weight_graph_video.py"
CM9_32_BARS_BPM = 130
CM9_32_BARS = 32
CM9_32_DURATION = CM9_32_BARS * 4 * 60 / CM9_32_BARS_BPM


EXPERIMENTS = [
    {
        "stage": "sources",
        "name": "01_low_voltage_fog_source_10s",
        "duration": 10.0,
        "weight_mode": "solo",
        "solo_slot": 1,
    },
    {
        "stage": "sources",
        "name": "02_spectral_glass_bloom_source_10s",
        "duration": 10.0,
        "weight_mode": "solo",
        "solo_slot": 2,
    },
    {
        "stage": "sources",
        "name": "03_carbon_cinema_noise_source_10s",
        "duration": 10.0,
        "weight_mode": "solo",
        "solo_slot": 3,
    },
    {
        "stage": "sources",
        "name": "04_buffer_freeze_dust_source_10s",
        "duration": 10.0,
        "weight_mode": "solo",
        "solo_slot": 4,
    },
    {
        "stage": "modulated",
        "name": "05_sustained_synth_prompt_weight_modulation_30s",
        "duration": 30.0,
        "weight_mode": "modulated",
        "solo_slot": 1,
        "bpm": 120,
    },
    {
        "stage": "loop_16s",
        "name": "06_sustained_synth_loop_sine_weight_modulation_16s",
        "duration": 16.0,
        "weight_mode": "loop_sine",
        "solo_slot": 1,
        "bpm": 120,
    },
    {
        "stage": "cm9_32bars_130",
        "name": "07_cm9_32bars_bpm130_meter_sine_prompt_weight_modulation",
        "duration": CM9_32_DURATION,
        "weight_mode": "meter_sine",
        "solo_slot": 1,
        "bpm": CM9_32_BARS_BPM,
        "bars": CM9_32_BARS,
        "generated_midi": "cm9_32bars_bpm130",
        "require_continuous_audio": True,
    },
    {
        "stage": "no_decay_trials",
        "name": "08_no_decay_unbroken_synth_pad_32bars",
        "profile": "sustained_no_decay_trials",
        "duration": CM9_32_DURATION,
        "weight_mode": "solo",
        "solo_slot": 1,
        "bpm": CM9_32_BARS_BPM,
        "bars": CM9_32_BARS,
        "generated_midi": "cm9_32bars_bpm130_hold",
    },
    {
        "stage": "no_decay_trials",
        "name": "09_no_decay_frozen_synth_sheet_32bars",
        "profile": "sustained_no_decay_trials",
        "duration": CM9_32_DURATION,
        "weight_mode": "solo",
        "solo_slot": 2,
        "bpm": CM9_32_BARS_BPM,
        "bars": CM9_32_BARS,
        "generated_midi": "cm9_32bars_bpm130_hold",
    },
    {
        "stage": "no_decay_trials",
        "name": "10_no_decay_string_machine_hold_32bars",
        "profile": "sustained_no_decay_trials",
        "duration": CM9_32_DURATION,
        "weight_mode": "solo",
        "solo_slot": 3,
        "bpm": CM9_32_BARS_BPM,
        "bars": CM9_32_BARS,
        "generated_midi": "cm9_32bars_bpm130_hold",
    },
    {
        "stage": "no_decay_trials",
        "name": "11_no_decay_sine_stack_sustain_32bars",
        "profile": "sustained_no_decay_trials",
        "duration": CM9_32_DURATION,
        "weight_mode": "solo",
        "solo_slot": 4,
        "bpm": CM9_32_BARS_BPM,
        "bars": CM9_32_BARS,
        "generated_midi": "cm9_32bars_bpm130_hold",
    },
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, text=True, check=True)


def ffprobe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,width,height,duration",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def analyze_float_wav(path: Path, window_seconds: float = 1.0) -> dict:
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RuntimeError(f"{path} is not a RIFF/WAVE file")

    offset = 12
    fmt: dict[str, int] | None = None
    audio_data: bytes | None = None
    while offset + 8 <= len(data):
        chunk_id = data[offset:offset + 4]
        chunk_size = int.from_bytes(data[offset + 4:offset + 8], "little")
        offset += 8
        chunk = data[offset:offset + chunk_size]
        if chunk_id == b"fmt ":
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
        elif chunk_id == b"data":
            audio_data = chunk
        offset += chunk_size + (chunk_size & 1)

    if fmt is None or audio_data is None:
        raise RuntimeError(f"{path} is missing fmt or data chunk")
    if fmt["audio_format"] != 3 or fmt["bits"] != 32:
        raise RuntimeError(f"{path} must be 32-bit float WAV, got {fmt}")

    channels = fmt["channels"]
    sample_rate = fmt["sample_rate"]
    total_frames = len(audio_data) // (4 * channels)
    values = struct.unpack("<" + "f" * (len(audio_data) // 4), audio_data)
    window_frames = max(1, int(round(sample_rate * window_seconds)))
    windows = []
    for start in range(0, total_frames, window_frames):
        end = min(total_frames, start + window_frames)
        sum_sq = 0.0
        peak = 0.0
        count = 0
        for frame in range(start, end):
            base = frame * channels
            mono = sum(values[base + channel] for channel in range(channels)) / channels
            sum_sq += mono * mono
            peak = max(peak, abs(mono))
            count += 1
        rms = math.sqrt(sum_sq / count) if count else 0.0
        windows.append({
            "start_seconds": start / sample_rate,
            "end_seconds": end / sample_rate,
            "rms": rms,
            "peak": peak,
        })

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
    min_to_first_3s_rms_ratio = (
        min_post_start_rms / first_3s_rms if first_3s_rms > 0.0 else 0.0
    )
    last_8s_to_first_3s_rms_ratio = (
        last_8s_median_rms / first_3s_rms if first_3s_rms > 0.0 else 0.0
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
        "min_to_first_3s_rms_ratio": min_to_first_3s_rms_ratio,
        "last_8s_to_first_3s_rms_ratio": last_8s_to_first_3s_rms_ratio,
        "min_post_start_rms": min_post_start_rms,
        "quiet_window_count": len(quiet_windows),
        "quiet_windows": quiet_windows[:20],
        "windows": windows,
        "continuous_after_post_start": len(quiet_windows) == 0,
    }


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


def write_cm9_32bar_midi(path: Path) -> None:
    ppq = 480
    bar_ticks = 4 * ppq
    note_ticks = bar_ticks - 1
    tempo_us = round(60_000_000 / CM9_32_BARS_BPM)
    chord = [36, 43, 48, 51, 55, 58, 62, 67]  # C2, G2, C3, Eb3, G3, Bb3, D4, G4
    track = bytearray()

    def add(delta: int, event: bytes) -> None:
        track.extend(vlq(delta))
        track.extend(event)

    name = b"Cm9 32 bars BPM130"
    add(0, b"\xff\x03" + vlq(len(name)) + name)
    add(0, b"\xff\x51\x03" + tempo_us.to_bytes(3, "big"))
    add(0, b"\xff\x58\x04" + bytes([4, 2, 24, 8]))

    pending_delta = 0
    for _bar in range(CM9_32_BARS):
        for index, note in enumerate(chord):
            add(pending_delta if index == 0 else 0, bytes([0x90, note, 88]))
            pending_delta = 0
        for index, note in enumerate(chord):
            add(note_ticks if index == 0 else 0, bytes([0x80, note, 0]))
        pending_delta = 1
    add(pending_delta, b"\xff\x2f\x00")

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


def write_cm9_32bar_hold_midi(path: Path) -> None:
    ppq = 480
    end_ticks = CM9_32_BARS * 4 * ppq
    tempo_us = round(60_000_000 / CM9_32_BARS_BPM)
    chord = [36, 43, 48, 51, 55, 58, 62, 67]  # C2, G2, C3, Eb3, G3, Bb3, D4, G4
    track = bytearray()

    def add(delta: int, event: bytes) -> None:
        track.extend(vlq(delta))
        track.extend(event)

    name = b"Cm9 32 bars BPM130 hold"
    add(0, b"\xff\x03" + vlq(len(name)) + name)
    add(0, b"\xff\x51\x03" + tempo_us.to_bytes(3, "big"))
    add(0, b"\xff\x58\x04" + bytes([4, 2, 24, 8]))
    for note in chord:
        add(0, bytes([0x90, note, 88]))
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--midi", type=Path, default=DEFAULT_MIDI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "sources",
            "modulated",
            "loop_16s",
            "cm9_32bars_130",
            "no_decay_trials",
        ],
        default="all",
    )
    parser.add_argument("--audio-prompts", action="store_true")
    args = parser.parse_args()

    if not args.binary.exists():
        raise SystemExit(f"Missing binary: {args.binary}")
    if not args.midi.exists():
        raise SystemExit(f"Missing MIDI: {args.midi}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [item for item in EXPERIMENTS if args.stage == "all" or item["stage"] == args.stage]
    manifest_path = args.output_dir / "manifest.json"
    existing_by_name: dict[str, dict] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for experiment in existing.get("experiments", []):
            name = experiment.get("name")
            if name:
                existing_by_name[name] = experiment

    manifest = {
        "schema": "mrt-sustained-synth-prompt-sources-v1",
        "profile": "varies_by_experiment",
        "midi": str(args.midi.relative_to(ROOT)),
        "bpm": 120,
        "embedding_source": "audio" if args.audio_prompts else "text",
        "experiments": [],
    }

    for item in selected:
        midi_path = args.midi
        if item.get("generated_midi") == "cm9_32bars_bpm130":
            midi_path = args.output_dir / "cm9_32bars_bpm130.mid"
            write_cm9_32bar_midi(midi_path)
        elif item.get("generated_midi") == "cm9_32bars_bpm130_hold":
            midi_path = args.output_dir / "cm9_32bars_bpm130_hold.mid"
            write_cm9_32bar_hold_midi(midi_path)

        wav = args.output_dir / f"{item['name']}.wav"
        report = args.output_dir / f"{item['name']}.report.json"
        weights = args.output_dir / f"{item['name']}.weights.json"
        audio_check_path = args.output_dir / f"{item['name']}.audio_check.json"
        audio_prompt_dir = args.output_dir / f"{item['name']}_audio_prompts"
        cmd = [
            str(args.binary),
            "--midi",
            str(midi_path),
            "--profile",
            item.get("profile", "sustained_synth_textures"),
            "--duration",
            f"{item['duration']:.3f}",
            "--transition",
            "0.000",
            "--weight-mode",
            item["weight_mode"],
            "--solo-slot",
            str(item["solo_slot"]),
            "--output",
            str(wav),
            "--report",
            str(report),
            "--weights-output",
            str(weights),
            "--audio-prompt-dir",
            str(audio_prompt_dir),
        ]
        if not args.audio_prompts:
            cmd.append("--text-prompts")
        run(cmd)

        report_data = json.loads(report.read_text())
        weights_data = json.loads(weights.read_text())
        probe = ffprobe(wav)
        audio_check = analyze_float_wav(wav)
        audio_check_path.write_text(json.dumps(audio_check, indent=2) + "\n")
        duration = float(probe["format"]["duration"])
        weight_frame_rate = float(weights_data.get("frame_rate", 25))
        expected_frames = int(round(item["duration"] * weight_frame_rate))
        if abs(duration - item["duration"]) > 0.05:
            raise RuntimeError(f"{item['name']} duration mismatch: {duration}")
        if not report_data.get("non_silent", False):
            raise RuntimeError(f"{item['name']} is silent")
        if item.get("require_continuous_audio") and not audio_check["continuous_after_post_start"]:
            raise RuntimeError(
                f"{item['name']} has quiet audio windows after 3s: "
                f"{audio_check['quiet_windows'][:5]}"
            )
        if len(weights_data.get("frames", [])) != expected_frames:
            raise RuntimeError(f"{item['name']} weight frame count mismatch")

        graph_video = None
        graph_probe = None
        if item["weight_mode"] != "solo":
            graph_video = args.output_dir / f"{item['name']}.graph.mp4"
            run([
                "python3",
                str(GRAPH_VIDEO_SCRIPT),
                "--weights",
                str(weights),
                "--output",
                str(graph_video),
            ])
            graph_probe = ffprobe(graph_video)

        slot_index = max(0, min(3, item["solo_slot"] - 1))
        if item["weight_mode"] == "solo":
            slot_id = report_data["segments"][slot_index]["slot_id"]
            slot_label = report_data["segments"][slot_index]["slot_label"]
            prompt = report_data["segments"][slot_index]["prompt"]
        else:
            slot_id = "all_sustained_synth_slots"
            slot_label = "All Sustained Synth Slots"
            prompt = "Prompt-weight modulation across Low Voltage Fog, Spectral Glass Bloom, Carbon Cinema Noise, and Buffer Freeze Dust."
        existing_by_name[item["name"]] = {
            "stage": item["stage"],
            "name": item["name"],
            "profile": item.get("profile", "sustained_synth_textures"),
            "bpm": item.get("bpm", 120),
            "bars": item.get("bars"),
            "weight_mode": item["weight_mode"],
            "solo_slot": item["solo_slot"],
            "slot_id": slot_id,
            "slot_label": slot_label,
            "prompt": prompt,
            "duration_seconds": duration,
            "midi": str(midi_path.relative_to(ROOT)),
            "wav": str(wav.relative_to(ROOT)),
            "report": str(report.relative_to(ROOT)),
            "weights": str(weights.relative_to(ROOT)),
            "audio_check": str(audio_check_path.relative_to(ROOT)),
            "graph_video": str(graph_video.relative_to(ROOT)) if graph_video else None,
            "peak": report_data["peak"],
            "rms": report_data["rms"],
            "non_silent": report_data["non_silent"],
            "continuous_after_3s": audio_check["continuous_after_post_start"],
            "min_post_3s_rms": audio_check["min_post_start_rms"],
            "last_8s_to_first_3s_rms_ratio": audio_check["last_8s_to_first_3s_rms_ratio"],
            "min_to_first_3s_rms_ratio": audio_check["min_to_first_3s_rms_ratio"],
            "ffprobe": probe,
            "audio_activity": audio_check,
            "graph_ffprobe": graph_probe,
        }

    manifest["experiments"] = [
        existing_by_name[item["name"]]
        for item in EXPERIMENTS
        if item["name"] in existing_by_name
    ]

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
