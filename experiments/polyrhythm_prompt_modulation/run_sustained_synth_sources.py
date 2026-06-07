#!/usr/bin/env python3
"""Render four sustained synth prompt sources and one prompt-weight modulation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT / "build/examples/midi_prompt_diagnostic/mrt2_midi_prompt_diagnostic"
DEFAULT_MIDI = ROOT / "assets/Am-Minor Prog 01 (i-VI-v-iv).mid"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/polyrhythm_prompt_modulation/sustained_synth_prompt_sources"
GRAPH_VIDEO_SCRIPT = ROOT / "experiments/polyrhythm_prompt_modulation/render_prompt_weight_graph_video.py"


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--midi", type=Path, default=DEFAULT_MIDI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=["all", "sources", "modulated", "loop_16s"], default="all")
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
        "profile": "sustained_synth_textures",
        "midi": str(args.midi.relative_to(ROOT)),
        "bpm": 120,
        "embedding_source": "audio" if args.audio_prompts else "text",
        "experiments": [],
    }

    for item in selected:
        wav = args.output_dir / f"{item['name']}.wav"
        report = args.output_dir / f"{item['name']}.report.json"
        weights = args.output_dir / f"{item['name']}.weights.json"
        audio_prompt_dir = args.output_dir / f"{item['name']}_audio_prompts"
        cmd = [
            str(args.binary),
            "--midi",
            str(args.midi),
            "--profile",
            "sustained_synth_textures",
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
        duration = float(probe["format"]["duration"])
        expected_frames = int(round(item["duration"] * 25))
        if abs(duration - item["duration"]) > 0.05:
            raise RuntimeError(f"{item['name']} duration mismatch: {duration}")
        if not report_data.get("non_silent", False):
            raise RuntimeError(f"{item['name']} is silent")
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
            "bpm": item.get("bpm", 120),
            "weight_mode": item["weight_mode"],
            "solo_slot": item["solo_slot"],
            "slot_id": slot_id,
            "slot_label": slot_label,
            "prompt": prompt,
            "duration_seconds": duration,
            "wav": str(wav.relative_to(ROOT)),
            "report": str(report.relative_to(ROOT)),
            "weights": str(weights.relative_to(ROOT)),
            "graph_video": str(graph_video.relative_to(ROOT)) if graph_video else None,
            "peak": report_data["peak"],
            "rms": report_data["rms"],
            "non_silent": report_data["non_silent"],
            "ffprobe": probe,
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
