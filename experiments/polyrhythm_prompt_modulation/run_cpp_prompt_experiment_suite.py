#!/usr/bin/env python3
"""Run clear C++ Magenta MIDI prompt modulation tests and longer experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT / "build/examples/midi_prompt_diagnostic/mrt2_midi_prompt_diagnostic"
DEFAULT_MIDI = ROOT / "assets/Am-Minor Prog 01 (i-VI-v-iv).mid"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/polyrhythm_prompt_modulation/cpp_prompt_experiments"


EXPERIMENTS = [
    {
        "stage": "clear_10s",
        "name": "10s_clear_extreme_audio",
        "profile": "clear_extreme",
        "duration": 10.0,
        "transition": 0.18,
    },
    {
        "stage": "clear_10s",
        "name": "10s_microcinematic_footwork_audio",
        "profile": "microcinematic_footwork",
        "duration": 10.0,
        "transition": 0.18,
    },
    {
        "stage": "clear_10s",
        "name": "10s_glass_trap_pressure_audio",
        "profile": "glass_trap_pressure",
        "duration": 10.0,
        "transition": 0.18,
    },
    {
        "stage": "long_30s",
        "name": "30s_microcinematic_footwork_audio",
        "profile": "microcinematic_footwork",
        "duration": 30.0,
        "transition": 0.65,
    },
    {
        "stage": "long_30s",
        "name": "30s_negative_space_club_audio",
        "profile": "negative_space_club",
        "duration": 30.0,
        "transition": 0.65,
    },
    {
        "stage": "long_30s",
        "name": "30s_metallic_ambient_bounce_audio",
        "profile": "metallic_ambient_bounce",
        "duration": 30.0,
        "transition": 0.65,
    },
]


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True, check=True)


def ffprobe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,duration",
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
    parser.add_argument("--stage", choices=["all", "clear_10s", "long_30s"], default="all")
    parser.add_argument("--text-prompts", action="store_true")
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
        "schema": "mrt-cpp-prompt-experiment-suite-v1",
        "midi": str(args.midi.relative_to(ROOT)),
        "embedding_source": "text" if args.text_prompts else "audio",
        "experiments": [],
    }

    for item in selected:
        wav = args.output_dir / f"{item['name']}.wav"
        report = args.output_dir / f"{item['name']}.report.json"
        audio_prompt_dir = args.output_dir / f"{item['name']}_audio_prompts"
        cmd = [
            str(args.binary),
            "--midi",
            str(args.midi),
            "--profile",
            item["profile"],
            "--duration",
            f"{item['duration']:.3f}",
            "--transition",
            f"{item['transition']:.3f}",
            "--output",
            str(wav),
            "--report",
            str(report),
            "--audio-prompt-dir",
            str(audio_prompt_dir),
        ]
        if args.text_prompts:
            cmd.append("--text-prompts")
        run(cmd, ROOT)

        report_data = json.loads(report.read_text())
        probe = ffprobe(wav)
        duration = float(probe["format"]["duration"])
        if abs(duration - item["duration"]) > 0.05:
            raise RuntimeError(f"{item['name']} duration mismatch: {duration}")
        if not report_data.get("non_silent", False):
            raise RuntimeError(f"{item['name']} is silent")

        existing_by_name[item["name"]] = {
            "stage": item["stage"],
            "name": item["name"],
            "profile": item["profile"],
            "duration_seconds": duration,
            "wav": str(wav.relative_to(ROOT)),
            "report": str(report.relative_to(ROOT)),
            "peak": report_data["peak"],
            "rms": report_data["rms"],
            "non_silent": report_data["non_silent"],
            "ffprobe": probe,
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
