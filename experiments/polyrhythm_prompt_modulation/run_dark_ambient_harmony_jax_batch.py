#!/usr/bin/env python3
"""Render the dark ambient harmony MIDI batch with the JAX backend.

This is the Linux / CUDA companion to ``run_dark_ambient_harmony_midi_batch.py``.
The C++ renderer uses MLX/Metal and is macOS-only; this script keeps the same
prompt sets, MIDI latch concept, and meter+macro prompt-weight modulation, then
drives ``MagentaRT2Jax`` in stateful chunks so it can run on raytrek4090.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_dark_ambient_harmony_midi_batch as dark_batch  # noqa: E402
import run_dirty_cinematic_cm9_5min_batch as batch  # noqa: E402


DEFAULT_SOURCE_DIR = ROOT / "assets/dark_ambient_harmonies_128bars"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs/polyrhythm_prompt_modulation/"
    "dark_ambient_harmonies_128bars_prompt2_jax_cuda_16"
)
BPM = 120.0
BARS = 128
DURATION_SECONDS = 256.0
MODEL = "mrt2_small"
WEIGHT_FRAME_RATE = 60
GENERATION_FPS = 25
TARGET_RMS = 0.045
PEAK_CEILING = 0.85

METER_QUARTER_NOTE_PERIODS = [4.0, 3.0, 3.0, 4.0]
METER_PHASE_OFFSETS = [0.0, 0.0, 0.0, 0.25]
MACRO_CYCLES_PER_REFERENCE = [1.0, 1.5, 2.5, 3.5]
MACRO_PHASE_OFFSETS = [0.0, 0.23, 0.47, 0.71]


def parse_take_indices(value: str) -> set[int]:
    indices: set[int] = set()
    if not value.strip():
        return indices
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid take range: {item}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(item))
    if any(index < 1 for index in indices):
        raise argparse.ArgumentTypeError("--take-indices are 1-based")
    return indices


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def smoothstep(value: float) -> float:
    x = clamp(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 1e-9 or not math.isfinite(total):
        return [1.0, 0.0, 0.0, 0.0]
    return [weight / total for weight in weights]


def meter_sine_raw(time_seconds: float, modulation: dict) -> list[float]:
    beat_position = time_seconds * BPM / 60.0
    speed_scale = max(0.001, float(modulation["meter_speed_scale"]))
    depth = clamp(float(modulation["meter_depth"]), 0.0, 2.0)
    weights = []
    for period, offset in zip(METER_QUARTER_NOTE_PERIODS, METER_PHASE_OFFSETS):
        phase = 2.0 * math.pi * (speed_scale * beat_position / period + offset)
        lfo = 0.5 + 0.5 * math.sin(phase)
        shaped = clamp(0.5 + (lfo**3.0 - 0.5) * depth, 0.0, 1.35)
        weights.append(0.015 + shaped)
    return weights


def macro_sine_raw(time_seconds: float, modulation: dict, batch_variant: int, phase: float) -> list[float]:
    reference = max(0.001, float(modulation["macro_reference_seconds"]))
    normalized_time = time_seconds / reference
    variant_phase = phase + 0.017 * float(batch_variant)
    depth = clamp(float(modulation["macro_depth"]), 0.0, 2.0)
    weights = []
    for slot, (cycles, base_phase) in enumerate(zip(MACRO_CYCLES_PER_REFERENCE, MACRO_PHASE_OFFSETS)):
        slot_phase = base_phase + variant_phase * float(slot + 1)
        lfo_phase = 2.0 * math.pi * (normalized_time * cycles + slot_phase)
        lfo = 0.5 + 0.5 * math.sin(lfo_phase)
        shaped = clamp(0.5 + (lfo**1.8 - 0.5) * depth, 0.0, 1.35)
        weights.append(0.04 + shaped)
    return weights


def prompt_weights(time_seconds: float, modulation: dict, batch_variant: int, phase: float) -> list[float]:
    meter = meter_sine_raw(time_seconds, modulation)
    macro = macro_sine_raw(time_seconds, modulation, batch_variant, phase)
    macro_mix = clamp(float(modulation["macro_mix"]), 0.0, 1.5)
    mixed = []
    for micro_raw, slow_raw in zip(meter, macro):
        micro = max(0.0001, micro_raw) ** 0.72
        slow = max(0.0001, slow_raw) ** 1.05
        slow_mix = (1.0 - macro_mix) + macro_mix * slow
        mixed.append(0.020 + 0.82 * micro * slow_mix + 0.10 * micro + 0.08 * macro_mix * slow)
    return normalize(mixed)


def control_values(time_seconds: float, duration: float) -> dict:
    progress = clamp(time_seconds / max(0.001, duration), 0.0, 1.0)
    rise = smoothstep(progress)
    temp_wobble = 0.12 * progress * (1.0 - progress) * math.sin(2.0 * math.pi * (1.37 * progress + 0.13))
    topk_wobble = 0.12 * progress * (1.0 - progress) * math.sin(2.0 * math.pi * (1.11 * progress + 0.31))
    temp_shape = clamp(smoothstep(progress) + temp_wobble, 0.0, 1.0)
    topk_shape = clamp(smoothstep(progress) + topk_wobble, 0.0, 1.0)
    return {
        "cfg_musiccoca": 3.0 + 2.0 * rise,
        "cfg_notes": 0.1 + 0.9 * rise,
        "cfg_drums": 0.0,
        "temperature": 0.85 + (1.05 - 0.85) * temp_shape,
        "top_k": int(round(40.0 + (80.0 - 40.0) * topk_shape)),
    }


def build_weight_frames(duration: float, modulation: dict, batch_variant: int, phase: float, slots: list[dict]) -> list[dict]:
    frame_count = int(round(duration * WEIGHT_FRAME_RATE))
    labels = [slot["label"] for slot in slots]
    ids = [slot["id"] for slot in slots]
    frames = []
    for frame in range(frame_count):
        time_seconds = frame / WEIGHT_FRAME_RATE
        controls = control_values(time_seconds, duration)
        frames.append({
            "frame": frame,
            "time_seconds": round(time_seconds, 6),
            "prompt_page": 0,
            "active_slot_ids": ids,
            "active_slot_labels": labels,
            "weights": [round(value, 6) for value in prompt_weights(time_seconds, modulation, batch_variant, phase)],
            **{key: round(value, 6) if isinstance(value, float) else value for key, value in controls.items()},
        })
    return frames


def write_weight_json(path: Path, duration: float, modulation: dict, batch_variant: int, phase: float, slots: list[dict], frames: list[dict]) -> None:
    path.write_text(json.dumps({
        "schema": "mrt-jax-prompt-weight-frames-v1",
        "backend": "jax_cuda",
        "profile": "beatless_drone_ambient_cm9",
        "weight_mode": "meter_macro_sine",
        "control_mode": "jax_stable_ambient_crescendo",
        "bpm": BPM,
        "duration_seconds": duration,
        "frame_rate": WEIGHT_FRAME_RATE,
        "prompt_library_size": len(slots),
        "batch_variant": batch_variant,
        "macro_phase_offset": phase,
        "macro_reference_seconds": modulation["macro_reference_seconds"],
        "meter_speed_scale": modulation["meter_speed_scale"],
        "meter_depth": modulation["meter_depth"],
        "macro_depth": modulation["macro_depth"],
        "macro_mix": modulation["macro_mix"],
        "midi_mode": "initial_latch",
        "midi_refresh_seconds": 0.0,
        "meter_sine": [
            {"meter": "4/4", "quarter_note_period": 4.0, "phase_offset": 0.0},
            {"meter": "3/4", "quarter_note_period": 3.0, "phase_offset": 0.0},
            {"meter": "6/8", "quarter_note_period": 3.0, "phase_offset": 0.0},
            {"meter": "4/4", "quarter_note_period": 4.0, "phase_offset": 0.25},
        ],
        "macro_sine": [
            {"slot": index, "cycles_per_reference": cycles, "base_phase_offset": offset, "reference_seconds": modulation["macro_reference_seconds"]}
            for index, (cycles, offset) in enumerate(zip(MACRO_CYCLES_PER_REFERENCE, MACRO_PHASE_OFFSETS))
        ],
        "slots": [{"index": index, "id": slot["id"], "label": slot["label"]} for index, slot in enumerate(slots)],
        "prompt_pages": [{
            "page": 0,
            "start_seconds": 0.0,
            "end_seconds": duration,
            "prompt_indices": [0, 1, 2, 3],
            "ids": [slot["id"] for slot in slots],
            "labels": [slot["label"] for slot in slots],
        }],
        "frames": frames,
    }, indent=2) + "\n", encoding="utf-8")


def note_conditioning(pitches: list[int], first_chunk: bool) -> list[int]:
    notes = [-1] * 128
    value = 1 if first_chunk else 2
    for pitch in pitches:
        if 0 <= pitch < 128:
            notes[pitch] = value
    return notes


def gain_match(samples: np.ndarray, target_rms: float, peak_ceiling: float) -> np.ndarray:
    mono = samples.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if rms <= 1e-9 or peak <= 1e-9 or target_rms <= 0.0:
        return samples
    gain = target_rms / rms
    if peak * gain > peak_ceiling:
        gain = peak_ceiling / peak
    return np.clip(samples * gain, -1.0, 1.0).astype(np.float32)


def add_noise_floor(samples: np.ndarray, floor_rms: float, seed: int) -> np.ndarray:
    if floor_rms <= 0.0 or samples.size == 0:
        return samples
    rng = np.random.default_rng(seed)
    floor = rng.standard_normal(samples.shape[0]).astype(np.float32)
    rms = float(np.sqrt(np.mean(np.square(floor))))
    if rms <= 1e-9:
        return samples
    floor *= floor_rms / rms
    return np.clip(samples + floor[:, None], -1.0, 1.0).astype(np.float32)


def paths_for(output_dir: Path, take_index: int, midi_meta: dict, prompt_set: dict) -> dict[str, Path]:
    take_dir = output_dir / f"take_{take_index:03d}"
    stem = f"dark_ambient_{midi_meta['name']}_{prompt_set['short']}_128bars_jax"
    return {
        "dir": take_dir,
        "prompts": take_dir / f"{stem}.prompts.tsv",
        "wav": take_dir / f"{stem}.wav",
        "report": take_dir / f"{stem}.report.json",
        "weights": take_dir / f"{stem}.weights.json",
        "audio_check": take_dir / f"{stem}.audio_check.json",
        "log": take_dir / f"{stem}.render.log",
    }


def relative_string(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def modulation_for_take(take_index: int, fixed_modulation_index: int) -> dict:
    if fixed_modulation_index:
        return dark_batch.MODULATION_VARIANTS[fixed_modulation_index - 1]
    return dark_batch.MODULATION_VARIANTS[(take_index - 1) % len(dark_batch.MODULATION_VARIANTS)]


def render_take(mrt, task: dict, args: argparse.Namespace, device_summary: list[str]) -> dict:
    source = task["source"]
    take_index = task["take_index"]
    prompt_set = task["prompt_set"]
    paths = paths_for(args.output_dir, take_index, source, prompt_set)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    slots = dark_batch.prompt_slots(prompt_set, source, take_index)
    batch.write_prompt_library(paths["prompts"], slots)
    modulation = modulation_for_take(take_index, args.fixed_modulation_index)
    batch_variant = take_index
    phase = (batch.macro_phase_offset_for_take(batch_variant) + modulation["phase_offset"]) % 1.0
    frames = build_weight_frames(args.duration, modulation, batch_variant, phase, slots)
    write_weight_json(paths["weights"], args.duration, modulation, batch_variant, phase, slots, frames)

    if args.dry_run:
        return {
            "take_index": take_index,
            "status": "dry_run",
            "weights": relative_string(paths["weights"]),
            "prompts": relative_string(paths["prompts"]),
            "source_midi": relative_string(Path(source["path"])),
            "latch_midi": relative_string(task["latch_midi"]),
            "chord_name": source["name"],
            "prompt_set": prompt_set,
            "modulation": {**modulation, "batch_variant": batch_variant, "macro_phase_offset": phase},
        }

    import soundfile as sf  # noqa: WPS433

    prompt_embeddings = []
    for slot in slots:
        prompt_embeddings.append(np.asarray(mrt.embed_style(slot["prompt"], use_mapper=True), dtype=np.float32))

    total_frames = int(round(args.duration * GENERATION_FPS))
    chunk_frames = max(1, int(round(args.chunk_seconds * GENERATION_FPS)))
    generated = []
    state = None
    started = time.time()
    with paths["log"].open("w", encoding="utf-8") as log:
        log.write(f"backend=jax_cuda model={args.model} take={take_index}\n")
        log.write(f"devices={device_summary}\n")
        frame_cursor = 0
        while frame_cursor < total_frames:
            frames_this_chunk = min(chunk_frames, total_frames - frame_cursor)
            time_mid = (frame_cursor + frames_this_chunk * 0.5) / GENERATION_FPS
            weights = np.asarray(prompt_weights(time_mid, modulation, batch_variant, phase), dtype=np.float32)
            mixed_embedding = np.sum(np.stack(prompt_embeddings, axis=0) * weights[:, None], axis=0)
            controls = control_values(time_mid, args.duration)
            waveform, state = mrt.generate(
                style=mixed_embedding,
                notes=None
                if args.notes_mode == "none"
                else note_conditioning(source["pitches"], first_chunk=(frame_cursor == 0)),
                drums=[0],
                cfg_musiccoca=float(controls["cfg_musiccoca"]),
                cfg_notes=float(controls["cfg_notes"]),
                cfg_drums=0.0,
                temperature=float(controls["temperature"]),
                top_k=int(controls["top_k"]),
                frames=frames_this_chunk,
                state=state,
            )
            generated.append(waveform.samples)
            frame_cursor += frames_this_chunk
            if frame_cursor % max(chunk_frames * 8, 1) == 0 or frame_cursor >= total_frames:
                log.write(f"frames={frame_cursor}/{total_frames} time={frame_cursor / GENERATION_FPS:.2f}s\n")
                log.flush()

    samples = np.concatenate(generated, axis=0).astype(np.float32)
    samples = gain_match(samples, args.target_rms, args.peak_ceiling)
    samples = add_noise_floor(samples, args.noise_floor_rms, seed=1000 + take_index)
    sf.write(paths["wav"], samples, 48000, subtype="FLOAT")
    audio_check = batch.analyze_float_wav_stream(paths["wav"].resolve())
    paths["audio_check"].write_text(json.dumps(audio_check, indent=2) + "\n", encoding="utf-8")
    mono = samples.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    controls = batch.control_summary(frames)
    report = {
        "schema": "mrt-jax-dark-ambient-harmony-take-v1",
        "backend": "jax_cuda",
        "model": args.model,
        "devices": device_summary,
        "wav": relative_string(paths["wav"]),
        "weights": relative_string(paths["weights"]),
        "prompts": relative_string(paths["prompts"]),
        "source_midi": relative_string(Path(source["path"])),
        "latch_midi": relative_string(task["latch_midi"]),
        "duration_seconds": samples.shape[0] / 48000.0,
        "sample_rate": 48000,
        "channels": int(samples.shape[1]),
        "rms": rms,
        "peak": peak,
        "non_silent": rms > 0.0001 and peak > 0.001,
        "midi_mode": "initial_latch",
        "notes_mode": args.notes_mode,
        "midi": {
            "notes": len(source["pitches"]),
            "note_on_events": len(source["pitches"]),
            "note_off_events": 0,
            "inferred_held_notes": len(source["pitches"]),
            "conditioned": args.notes_mode != "none",
        },
        "embedding_source": "text_embedding_mix",
        "weight_mode": "meter_macro_sine",
        "control_mode": "jax_stable_ambient_crescendo",
        "chunk_seconds": args.chunk_seconds,
        "noise_floor_rms": args.noise_floor_rms,
        "controls": controls,
        "audio_activity": {
            "quiet_window_count": audio_check["quiet_window_count"],
            "min_post_3s_rms": audio_check["min_post_start_rms"],
            "continuous_after_3s": audio_check["continuous_after_post_start"],
        },
        "elapsed_seconds": time.time() - started,
    }
    paths["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["non_silent"]:
        raise RuntimeError(f"silent render: {paths['wav']}")
    if not audio_check["continuous_after_post_start"]:
        raise RuntimeError(f"quiet windows in render: {paths['wav']}")
    return {
        "take_index": take_index,
        "wav": relative_string(paths["wav"]),
        "report": relative_string(paths["report"]),
        "weights": relative_string(paths["weights"]),
        "prompts": relative_string(paths["prompts"]),
        "audio_check": relative_string(paths["audio_check"]),
        "duration_seconds": report["duration_seconds"],
        "rms": rms,
        "peak": peak,
        "source_midi": relative_string(Path(source["path"])),
        "latch_midi": relative_string(task["latch_midi"]),
        "chord_name": source["name"],
        "pitches": source["pitches"],
        "prompt_set": prompt_set,
        "prompt_slots": batch.prompt_manifest(slots),
        "modulation": {**modulation, "batch_variant": batch_variant, "macro_phase_offset": phase},
        "controls": controls,
        "audio_activity": report["audio_activity"],
        "midi": report["midi"],
        "log": relative_string(paths["log"]),
    }


def load_sources(source_dir: Path, output_dir: Path, duration: float) -> list[dict]:
    midis = sorted(source_dir.glob("*.mid"))
    if len(midis) != 8:
        raise SystemExit(f"Expected 8 MIDI files, found {len(midis)} in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    latch_dir = output_dir / "midi_latch"
    sources = []
    for midi in midis:
        source = dark_batch.parse_source_midi(midi)
        if abs(source["bpm"] - BPM) > 0.01 or abs(source["duration_seconds"] - duration) > 0.01:
            raise SystemExit(
                f"Unexpected MIDI metadata for {midi}: bpm={source['bpm']} duration={source['duration_seconds']}"
            )
        latch_midi = latch_dir / f"{midi.stem}_noteon_latch.mid"
        dark_batch.write_latch_midi(source, latch_midi)
        latch_source = dark_batch.parse_source_midi(latch_midi)
        if latch_source["note_off_count_source"] != 0:
            raise SystemExit(f"Latch MIDI contains note-off events: {latch_midi}")
        source["latch_midi"] = latch_midi
        sources.append(source)
    return sources


def selected_prompt_sets(prompt_set_short: str) -> list[dict]:
    if prompt_set_short == "all":
        return dark_batch.PROMPT_SETS
    matches = [prompt_set for prompt_set in dark_batch.PROMPT_SETS if prompt_set["short"] == prompt_set_short]
    if not matches:
        valid = ", ".join(["all"] + [prompt_set["short"] for prompt_set in dark_batch.PROMPT_SETS])
        raise SystemExit(f"Unknown --prompt-set-short {prompt_set_short!r}; valid values: {valid}")
    return matches


def build_tasks(sources: list[dict], prompt_set_short: str) -> list[dict]:
    tasks = []
    take_index = 1
    prompt_sets = selected_prompt_sets(prompt_set_short)
    for source in sources:
        for prompt_set in prompt_sets:
            tasks.append({
                "take_index": take_index,
                "source": source,
                "prompt_set": prompt_set,
                "latch_midi": source["latch_midi"],
            })
            take_index += 1
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--duration", type=float, default=DURATION_SECONDS)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument("--target-rms", type=float, default=TARGET_RMS)
    parser.add_argument("--peak-ceiling", type=float, default=PEAK_CEILING)
    parser.add_argument("--noise-floor-rms", type=float, default=0.0012)
    parser.add_argument("--limit", type=int, default=0, help="Limit take count for smoke tests.")
    parser.add_argument(
        "--prompt-set-short",
        default="all",
        help="Prompt set short name to render, or 'all' for the full 8 MIDI x 2 prompt batch.",
    )
    parser.add_argument(
        "--fixed-modulation-index",
        type=int,
        default=0,
        help="1-based modulation variant to use for every take. 0 keeps the per-take variant rotation.",
    )
    parser.add_argument(
        "--notes-mode",
        choices=("chord", "none"),
        default="chord",
        help="Use chord note conditioning, or disable notes for prompt-only fallback.",
    )
    parser.add_argument(
        "--take-indices",
        type=parse_take_indices,
        default=set(),
        help="Comma-separated 1-based take indices or ranges, e.g. 1,3,5-8.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Manifest filename under output dir. Parallel workers should use unique names.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write prompts/weights/manifest without loading JAX.")
    args = parser.parse_args()

    args.source_dir = args.source_dir if args.source_dir.is_absolute() else ROOT / args.source_dir
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.chunk_seconds <= 0.0:
        raise SystemExit("--chunk-seconds must be positive")
    if args.noise_floor_rms < 0.0:
        raise SystemExit("--noise-floor-rms must be non-negative")
    if args.fixed_modulation_index < 0 or args.fixed_modulation_index > len(dark_batch.MODULATION_VARIANTS):
        raise SystemExit(
            f"--fixed-modulation-index must be 0 or 1-{len(dark_batch.MODULATION_VARIANTS)}"
        )

    sources = load_sources(args.source_dir, args.output_dir, args.duration)
    prompt_sets = selected_prompt_sets(args.prompt_set_short)
    tasks = build_tasks(sources, args.prompt_set_short)
    if args.take_indices:
        all_indices = {task["take_index"] for task in tasks}
        missing = sorted(args.take_indices - all_indices)
        if missing:
            raise SystemExit(f"Unknown take indices: {missing}")
        tasks = [task for task in tasks if task["take_index"] in args.take_indices]
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise SystemExit("No takes selected")

    device_summary: list[str] = []
    mrt = None
    if not args.dry_run:
        import jax  # noqa: WPS433
        from magenta_rt import MagentaRT2Jax  # noqa: WPS433

        device_summary = [f"{device.platform}:{device.device_kind}" for device in jax.devices()]
        if not any(device.platform == "gpu" for device in jax.devices()):
            raise SystemExit(f"JAX GPU device not available: {device_summary}")
        mrt = MagentaRT2Jax(
            size=args.model,
            temperature=0.85,
            top_k=40,
            cfg_musiccoca=3.0,
            cfg_notes=0.1,
            cfg_drums=0.0,
        )

    manifest = {
        "schema": "mrt-dark-ambient-harmonies-128bars-prompt2-jax-batch-v1",
        "backend": "jax_cuda",
        "source_dir": relative_string(args.source_dir),
        "output_dir": relative_string(args.output_dir),
        "embedding_source": "text_embedding_mix",
        "bpm": BPM,
        "bars": BARS,
        "duration_seconds": args.duration,
        "chunk_seconds": args.chunk_seconds,
        "noise_floor_rms": args.noise_floor_rms,
        "midi_mode": "initial_latch",
        "notes_mode": args.notes_mode,
        "midi_note_output": "source MIDI pitches copied to Note-On-only latch MIDI; no Note Off events in render MIDI",
        "weight_mode": "meter_macro_sine",
        "control_mode": "jax_stable_ambient_crescendo",
        "prompt_set_short": args.prompt_set_short,
        "prompt_sets": prompt_sets,
        "fixed_modulation_index": args.fixed_modulation_index,
        "fixed_modulation": None
        if not args.fixed_modulation_index
        else dark_batch.MODULATION_VARIANTS[args.fixed_modulation_index - 1],
        "model": args.model,
        "devices": device_summary,
        "dry_run": args.dry_run,
        "selected_take_indices": [task["take_index"] for task in tasks],
        "manifest_name": args.manifest_name,
        "takes": [],
    }
    manifest_path = args.output_dir / args.manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    completed = []
    for task in tasks:
        result = render_take(mrt, task, args, device_summary)
        completed.append(result)
        manifest["takes"] = completed
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"completed {len(completed)}/{len(tasks)} take={result['take_index']:03d}", flush=True)
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
