#!/usr/bin/env python3
"""Render 64 beatless Cm9 drone prompts, 64 bars each."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import run_dirty_cinematic_cm9_5min_batch as dirty


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT / "build/examples/midi_prompt_diagnostic/mrt2_midi_prompt_diagnostic"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/polyrhythm_prompt_modulation/beatless_drone_cm9_64bar_64patterns"
BPM = 120
BARS = 64
BEATS_PER_BAR = 4
DURATION_SECONDS = BARS * BEATS_PER_BAR * 60.0 / BPM
COUNT = 64
SAMPLE_RATE = 48000
CHANNELS = 2
BYTES_PER_SAMPLE = 4
WEIGHT_FRAME_RATE = 60

MATERIALS = [
    "Obsidian",
    "Basalt",
    "Graphite",
    "Slate",
    "Mercury",
    "Chrome",
    "Frost",
    "Ash",
    "Carbon",
    "Resin",
    "Violet",
    "Granite",
    "Smoke",
    "Onyx",
    "Charcoal",
    "Glass",
]

STATES = [
    "Undertow",
    "Continuum",
    "Horizon",
    "Cathedral",
    "Weather",
    "Reservoir",
    "Corridor",
    "Tide",
]

SOURCES = [
    "low modular oscillator stack",
    "warm organ-like sine drone",
    "dark wavetable harmonic cloud",
    "smooth additive sine cluster",
    "tape-loop synth pad",
    "feedback resonator bank below self-oscillation",
    "soft FM sine partial cloud",
    "analog string-machine sustain",
]

MOTIONS = [
    "non-periodic phase drift",
    "random voltage detune that never forms an LFO pattern",
    "very slow low-pass color creep",
    "spectral blur that thaws over minutes",
    "resonance breathing without a pulse",
    "slow stereo air drift",
    "sub-harmonic fog swelling without meter",
    "tape wow and filter shade moving irregularly",
]

EFFECTS = [
    "dark ladder filtering, soft tape saturation, and a stable reverb field",
    "muted chorus, spectral smear, and a distant room tail",
    "convolution-like space, low shelf warmth, and gentle high-cut damping",
    "soft wavefolder color, resonant notch shading, and wide reverb",
    "plate reverb, warm compression, and blurred high-frequency air",
    "slow formant color, tape haze, and a large connected tail",
    "smooth ensemble thickening, dark filtering, and stable stereo widening",
    "diffuse reverb, low-pass bloom, and soft analog saturation",
]

REGISTERS = [
    "low-register",
    "low-mid",
    "full-range dark",
    "sub-heavy",
    "midrange warm",
    "wide low-register",
    "hazy high-mid over low foundation",
    "deep cinematic",
]

SPACES = [
    "a huge but stable concrete room",
    "a wide suspended stereo field",
    "a distant dark hall",
    "a close center with a large tail",
    "a slow expanding ambient space",
    "a low-ceiling machine room blurred into reverb",
    "a soft horizon-like stereo field",
    "a deep cinematic void",
]

AVOID = (
    "Avoid drums, percussion, kick, snare, hats, shakers, claps, clicks, ticks, "
    "impacts, bursts, accents, attacks, stutters, grains, arpeggios, sequencer patterns, "
    "ostinatos, bass grooves, rhythmic pulses, rhythmic LFOs, audible modulation cycles, "
    "tremolo, gated motion, sidechain pumping, delay taps, vocals, lead melody, chord stabs, "
    "drops, fade-outs, sudden cuts, and silence."
)


def prompt_for_take(take_index: int) -> dict[str, str | float | int]:
    idx = take_index - 1
    material = MATERIALS[idx % len(MATERIALS)]
    state = STATES[(idx // len(MATERIALS)) % len(STATES)]
    source = SOURCES[(idx * 3) % len(SOURCES)]
    motion = MOTIONS[(idx * 5) % len(MOTIONS)]
    effects = EFFECTS[(idx * 7) % len(EFFECTS)]
    register = REGISTERS[(idx * 11) % len(REGISTERS)]
    space = SPACES[(idx * 13) % len(SPACES)]
    name = f"{material} {state}"
    slug = f"{material.lower()}_{state.lower()}".replace(" ", "_")
    prompt = (
        f"Create an original sustained synthesizer texture called {name}. "
        f"A continuous Cm9 {register} {source} held as one uninterrupted drone for exactly "
        f"64 bars at 120 BPM, with no repeated attacks and no beat grid. "
        f"The only movement is {motion}, spread through {space}. "
        f"Add {effects}. Keep the sound ambient, suspended, cinematic, smooth, and legato from "
        f"the first second to the final bar, as one long sustained tone field with a steady envelope. {AVOID}"
    )
    return {
        "id": f"take{take_index:03d}_{slug}",
        "label": name,
        "category": "beatless_drone",
        "prompt": prompt,
        "guide_kind": f"beatless_drone_{take_index:03d}",
        "temperature": 0.22 + 0.004 * (idx % 5),
        "top_k": 4 + (idx % 3),
        "cfg_musiccoca": 8.0 + 0.04 * (idx % 4),
        "cfg_notes": 7.0 + 0.04 * (idx % 4),
        "cfg_drums": 0.0,
    }


def paths_for_take(output_dir: Path, take_index: int) -> dict[str, Path]:
    take_dir = output_dir / f"take_{take_index:03d}"
    stem = f"cm9_beatless_drone_take_{take_index:03d}_64bars"
    return {
        "dir": take_dir,
        "prompts": take_dir / f"{stem}.prompts.tsv",
        "wav": take_dir / f"{stem}.wav",
        "report": take_dir / f"{stem}.report.json",
        "weights": take_dir / f"{stem}.weights.json",
        "audio_check": take_dir / f"{stem}.audio_check.json",
        "log": take_dir / f"{stem}.render.log",
    }


def short_spike_count(wav: Path) -> int | None:
    if dirty.np is None:
        return None
    fmt, data_offset, data_size = dirty.find_float_wav_data(wav)
    channels = fmt["channels"]
    sample_rate = fmt["sample_rate"]
    total_frames = data_size // (channels * 4)
    samples = dirty.np.memmap(
        wav,
        dtype="<f4",
        mode="r",
        offset=data_offset,
        shape=(total_frames, channels),
    )
    mono = samples.mean(axis=1)
    win = max(1, int(round(sample_rate * 0.1)))
    values = []
    for start in range(int(round(sample_rate * 3.0)), total_frames, win):
        chunk = mono[start:start + win]
        if len(chunk):
            values.append(float(dirty.np.sqrt(dirty.np.mean(dirty.np.square(chunk)))))
    del samples
    if not values:
        return 0
    arr = dirty.np.array(values, dtype=dirty.np.float64)
    median = float(dirty.np.median(arr))
    mad = float(dirty.np.median(dirty.np.abs(arr - median)))
    threshold = median + max(0.006, 6.0 * mad)
    return int(dirty.np.sum(arr > threshold))


def validate_take(paths: dict[str, Path], duration_seconds: float, max_spike_count: int) -> dict:
    probe = dirty.sustained.ffprobe(paths["wav"])
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    weights = json.loads(paths["weights"].read_text(encoding="utf-8"))
    audio_check = dirty.analyze_float_wav_stream(paths["wav"].resolve(), window_seconds=2.0)
    low_rms_windows = [
        window for window in audio_check["windows"]
        if window["start_seconds"] >= 3.0 and window["rms"] < 0.003
    ]
    audio_check["low_rms_window_count_lt_0_003"] = len(low_rms_windows)
    audio_check["low_rms_windows_lt_0_003"] = low_rms_windows[:20]
    audio_check["rms_100ms_spike_count"] = short_spike_count(paths["wav"].resolve())
    paths["audio_check"].write_text(json.dumps(audio_check, indent=2) + "\n", encoding="utf-8")

    duration = float(probe["format"]["duration"])
    if abs(duration - duration_seconds) > 0.05:
        raise RuntimeError(f"Duration mismatch: {duration}")
    if report.get("embedding_source") != "audio":
        raise RuntimeError(f"Unexpected embedding source: {report.get('embedding_source')}")
    if report.get("midi_mode") != "initial_latch":
        raise RuntimeError(f"Unexpected MIDI mode: {report.get('midi_mode')}")
    if report.get("weight_mode") != "solo":
        raise RuntimeError(f"Unexpected weight mode: {report.get('weight_mode')}")
    if weights.get("frame_rate") != WEIGHT_FRAME_RATE:
        raise RuntimeError(f"Unexpected weight frame rate: {weights.get('frame_rate')}")
    if not report.get("non_silent", False):
        raise RuntimeError("Rendered WAV is silent")
    if not audio_check["continuous_after_post_start"]:
        raise RuntimeError(f"Quiet windows found: {audio_check['quiet_windows'][:5]}")
    if low_rms_windows:
        raise RuntimeError(f"Collapsed low-RMS windows found: {low_rms_windows[:5]}")
    spike_count = audio_check["rms_100ms_spike_count"]
    if spike_count is not None and spike_count > max_spike_count:
        raise RuntimeError(f"Too many 100ms RMS spikes: {spike_count} > {max_spike_count}")

    return {
        "duration_seconds": duration,
        "prompt_library": str(paths["prompts"].relative_to(ROOT)),
        "wav": str(paths["wav"].relative_to(ROOT)),
        "report": str(paths["report"].relative_to(ROOT)),
        "weights": str(paths["weights"].relative_to(ROOT)),
        "audio_check": str(paths["audio_check"].relative_to(ROOT)),
        "log": str(paths["log"].relative_to(ROOT)),
        "ffprobe": probe,
        "peak": report["peak"],
        "rms": report["rms"],
        "non_silent": report["non_silent"],
        "audio_activity": {
            "continuous_after_3s": audio_check["continuous_after_post_start"],
            "quiet_window_count": audio_check["quiet_window_count"],
            "low_rms_window_count_lt_0_003": len(low_rms_windows),
            "rms_100ms_spike_count": audio_check["rms_100ms_spike_count"],
            "min_post_3s_rms": audio_check["min_post_start_rms"],
        },
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--duration", type=float, default=DURATION_SECONDS)
    parser.add_argument("--max-spike-count", type=int, default=24)
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
    midi = output_dir / "cm9_beatless_drone_64bars_bpm120_hold.mid"
    dirty.write_cm9_hold_midi(midi, args.duration)

    free_bytes = shutil.disk_usage(output_dir).free
    estimated_wav_bytes = args.count * args.duration * SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE
    estimated_weight_bytes = args.count * args.duration * WEIGHT_FRAME_RATE * 450
    print(
        "Storage preflight: "
        f"free={free_bytes / (1024 ** 3):.2f} GiB, "
        f"estimated_batch={((estimated_wav_bytes + estimated_weight_bytes) / (1024 ** 3)):.2f} GiB",
        flush=True,
    )

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema": "mrt-beatless-drone-cm9-64bar-batch-v1",
        "profile": "beatless_drone_ambient_cm9",
        "bpm": BPM,
        "bars": BARS,
        "duration_seconds": args.duration,
        "count_requested": args.count,
        "start_index": args.start_index,
        "midi": str(midi.relative_to(ROOT)),
        "midi_mode": "initial_latch",
        "midi_refresh_seconds": 0.0,
        "embedding_source": "audio",
        "weight_mode": "solo",
        "control_mode": "slot_blend",
        "prompt_variant_mode": "one_unique_beatless_prompt_per_take",
        "takes": [],
        "completed_count": 0,
    }

    for offset in range(args.count):
        take_index = args.start_index + offset
        paths = paths_for_take(output_dir, take_index)
        prompt = prompt_for_take(take_index)
        dirty.write_prompt_library(paths["prompts"], [prompt])
        prompt_summary = {
            "id": str(prompt["id"]),
            "label": str(prompt["label"]),
            "category": str(prompt["category"]),
            "prompt": str(prompt["prompt"]),
        }

        if paths["wav"].exists() and not args.force and not args.skip_render:
            print(f"take {take_index:03d}: exists, validating", flush=True)
        elif not args.skip_render:
            cmd = [
                str(args.binary),
                "--midi", str(midi),
                "--profile", "beatless_drone_ambient_cm9",
                "--prompt-library", str(paths["prompts"]),
                "--prompt-library-audio-prompts",
                "--no-write-audio-prompt-guides",
                "--prompt-page-seconds", f"{args.duration + 1.0:.3f}",
                "--weight-mode", "solo",
                "--solo-slot", "1",
                "--control-mode", "slot_blend",
                "--duration", f"{args.duration:.3f}",
                "--midi-mode", "initial_latch",
                "--midi-refresh-seconds", "0",
                "--transition", "0.000",
                "--target-rms", "0.045",
                "--output", str(paths["wav"]),
                "--report", str(paths["report"]),
                "--weights-output", str(paths["weights"]),
            ]
            dirty.run_logged(cmd, paths["log"])

        summary = validate_take(paths, args.duration, args.max_spike_count)
        summary["take_index"] = take_index
        summary["prompt"] = prompt_summary
        manifest["takes"].append(summary)
        manifest["completed_count"] = len(manifest["takes"])
        write_manifest(manifest_path, manifest)
        print(
            f"take {take_index:03d}: ok "
            f"rms={summary['rms']:.6f} "
            f"min_post_3s={summary['audio_activity']['min_post_3s_rms']:.6f} "
            f"spikes={summary['audio_activity']['rms_100ms_spike_count']}",
            flush=True,
        )

    write_manifest(manifest_path, manifest)
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
