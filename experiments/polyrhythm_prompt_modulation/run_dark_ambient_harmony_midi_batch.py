#!/usr/bin/env python3
"""Render the 8 dark ambient harmony MIDIs with two sustained prompt sets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys

import run_dirty_cinematic_cm9_5min_batch as batch
import run_sustained_synth_sources as sustained


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = ROOT / "assets/dark_ambient_harmonies_128bars"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs/polyrhythm_prompt_modulation/dark_ambient_harmonies_128bars_prompt2_audio_16"
)
DEFAULT_BINARY = ROOT / "build/examples/midi_prompt_diagnostic/mrt2_midi_prompt_diagnostic"
BPM = 120.0
BARS = 128
DURATION_SECONDS = 256.0
WEIGHT_FRAME_RATE = 60

PROMPT_SETS = [
    {
        "id": "abyssal_voltage_drone",
        "label": "Abyssal Voltage Drone",
        "short": "abyssal",
        "description": "low dirty analog drone, pressure, feedback, and cinematic fog",
    },
    {
        "id": "corroded_spectral_static",
        "label": "Corroded Spectral Static",
        "short": "corroded",
        "description": "rough spectral glass, frozen glitch dust, metallic air, and damaged reverb",
    },
]

MODULATION_VARIANTS = [
    {"macro_reference_seconds": 224.0, "meter_speed_scale": 0.500, "meter_depth": 0.82, "macro_depth": 1.30, "macro_mix": 1.16, "phase_offset": 0.03, "label": "ultra-slow wide macro"},
    {"macro_reference_seconds": 192.0, "meter_speed_scale": 0.667, "meter_depth": 0.94, "macro_depth": 1.18, "macro_mix": 1.02, "phase_offset": 0.11, "label": "slow three-over-two drift"},
    {"macro_reference_seconds": 160.0, "meter_speed_scale": 0.875, "meter_depth": 1.08, "macro_depth": 1.05, "macro_mix": 0.90, "phase_offset": 0.19, "label": "medium deep texture weave"},
    {"macro_reference_seconds": 136.0, "meter_speed_scale": 1.000, "meter_depth": 1.18, "macro_depth": 0.96, "macro_mix": 0.82, "phase_offset": 0.27, "label": "balanced meter macro"},
    {"macro_reference_seconds": 208.0, "meter_speed_scale": 0.750, "meter_depth": 0.76, "macro_depth": 1.42, "macro_mix": 1.28, "phase_offset": 0.35, "label": "deep macro pressure"},
    {"macro_reference_seconds": 148.0, "meter_speed_scale": 1.250, "meter_depth": 1.24, "macro_depth": 0.88, "macro_mix": 0.74, "phase_offset": 0.43, "label": "faster gritty polyrhythm"},
    {"macro_reference_seconds": 176.0, "meter_speed_scale": 1.125, "meter_depth": 0.98, "macro_depth": 1.22, "macro_mix": 1.08, "phase_offset": 0.51, "label": "off-grid macro bloom"},
    {"macro_reference_seconds": 120.0, "meter_speed_scale": 1.500, "meter_depth": 1.34, "macro_depth": 0.84, "macro_mix": 0.66, "phase_offset": 0.59, "label": "dense meter rotation"},
]


def read_vlq(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7f)
        if not byte & 0x80:
            return value, offset


def parse_source_midi(path: Path) -> dict:
    data = path.read_bytes()
    offset = 0
    if data[offset:offset + 4] != b"MThd":
        raise RuntimeError(f"Not a MIDI file: {path}")
    offset += 4
    header_len = int.from_bytes(data[offset:offset + 4], "big")
    offset += 4
    midi_format = int.from_bytes(data[offset:offset + 2], "big")
    offset += 2
    tracks = int.from_bytes(data[offset:offset + 2], "big")
    offset += 2
    ppq = int.from_bytes(data[offset:offset + 2], "big")
    offset = 8 + header_len

    tempo_us = round(60_000_000 / BPM)
    time_sig = (4, 2, 24, 8)
    end_tick = 0
    note_ons: list[tuple[int, int, int, int]] = []
    note_off_count = 0

    for _ in range(tracks):
        if data[offset:offset + 4] != b"MTrk":
            raise RuntimeError(f"Bad MIDI track in {path}")
        offset += 4
        track_size = int.from_bytes(data[offset:offset + 4], "big")
        offset += 4
        track = data[offset:offset + track_size]
        offset += track_size

        pos = 0
        tick = 0
        running_status: int | None = None
        while pos < len(track):
            delta, pos = read_vlq(track, pos)
            tick += delta
            end_tick = max(end_tick, tick)
            if pos >= len(track):
                break
            status = track[pos]
            if status < 0x80:
                if running_status is None:
                    raise RuntimeError("Running status without previous status")
                status = running_status
            else:
                pos += 1
                if status < 0xf0:
                    running_status = status
            if status == 0xff:
                meta_type = track[pos]
                pos += 1
                length, pos = read_vlq(track, pos)
                payload = track[pos:pos + length]
                pos += length
                if meta_type == 0x51 and length == 3:
                    tempo_us = int.from_bytes(payload, "big")
                elif meta_type == 0x58 and length >= 4:
                    time_sig = (payload[0], payload[1], payload[2], payload[3])
                elif meta_type == 0x2f:
                    break
                continue
            if status in (0xf0, 0xf7):
                length, pos = read_vlq(track, pos)
                pos += length
                continue

            event_type = status & 0xf0
            channel = status & 0x0f
            if event_type in (0x80, 0x90):
                pitch = track[pos]
                velocity = track[pos + 1]
                pos += 2
                if event_type == 0x90 and velocity > 0:
                    note_ons.append((tick, pitch, velocity, channel))
                else:
                    note_off_count += 1
            elif event_type in (0xa0, 0xb0, 0xe0):
                pos += 2
            elif event_type in (0xc0, 0xd0):
                pos += 1
            else:
                raise RuntimeError(f"Unsupported MIDI status {status:x} in {path}")

    duration_seconds = end_tick / ppq * tempo_us / 1_000_000
    pitches = sorted({pitch for tick, pitch, _, _ in note_ons if tick == 0})
    return {
        "path": str(path),
        "name": path.stem,
        "format": midi_format,
        "tracks": tracks,
        "ppq": ppq,
        "tempo_us": tempo_us,
        "bpm": 60_000_000 / tempo_us,
        "time_sig": time_sig,
        "end_tick": end_tick,
        "duration_seconds": duration_seconds,
        "note_ons": note_ons,
        "note_off_count_source": note_off_count,
        "pitches": pitches,
    }


def write_latch_midi(source: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    track = bytearray()

    def add(delta: int, event: bytes) -> None:
        track.extend(batch.vlq(delta))
        track.extend(event)

    name = f"{source['name']} note-on latch 128bars".encode("ascii", errors="ignore")
    add(0, b"\xff\x03" + batch.vlq(len(name)) + name)
    add(0, b"\xff\x51\x03" + int(source["tempo_us"]).to_bytes(3, "big"))
    ts = source["time_sig"]
    add(0, b"\xff\x58\x04" + bytes([ts[0], ts[1], ts[2], ts[3]]))
    last_tick = 0
    for tick, pitch, velocity, channel in sorted(source["note_ons"], key=lambda item: (item[0], item[1])):
        add(tick - last_tick, bytes([0x90 | channel, pitch, max(1, velocity)]))
        last_tick = tick
    add(int(source["end_tick"]) - last_tick, b"\xff\x2f\x00")

    with output_path.open("wb") as file:
        file.write(b"MThd")
        file.write((6).to_bytes(4, "big"))
        file.write((0).to_bytes(2, "big"))
        file.write((1).to_bytes(2, "big"))
        file.write(int(source["ppq"]).to_bytes(2, "big"))
        file.write(b"MTrk")
        file.write(len(track).to_bytes(4, "big"))
        file.write(track)


def note_names(pitches: list[int]) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return ", ".join(f"{names[pitch % 12]}{pitch // 12 - 1}" for pitch in pitches)


def prompt_slots(prompt_set: dict, midi_meta: dict, take_index: int) -> list[dict[str, str | float | int]]:
    chord = midi_meta["name"].split("_", 1)[1] if "_" in midi_meta["name"] else midi_meta["name"]
    pitches = note_names(midi_meta["pitches"])
    shared = (
        f"Use the supplied MIDI chord {chord} ({pitches}) as the harmonic source, "
        "held continuously for 128 bars at 120 BPM. The result must be one long beatless "
        "dark ambient drone with no re-attacks, no rhythmic loop, no arpeggio, no percussion, "
        "no click pattern, no tremolo pulse, no delay taps as rhythm, no vocals, no lead melody, "
        "no drops, no fade-out, and no silence. Let the prompt-weight modulation change texture "
        "and post effects only, while the MIDI harmony remains sustained."
    )
    if prompt_set["id"] == "abyssal_voltage_drone":
        return [
            {
                "id": f"take{take_index:03d}_{midi_meta['name']}_abyssal_voltage",
                "label": f"{chord} Abyssal Voltage",
                "category": "analog_drone",
                "prompt": (
                    f"Create an original sustained synthesizer texture called {chord} Abyssal Voltage. "
                    "A low-register modular VCO and organ-like drone, dirty and heavy, with sub pressure, "
                    "slow random-voltage pitch drift, dark ladder low-pass movement, and burnt transformer grain. "
                    f"Add tape compression, soft wavefolder saturation, and a deep black-room reverb tail. {shared}"
                ),
                "guide_kind": "beatless_drone_abyssal_voltage",
                "temperature": 0.54,
                "top_k": 34,
                "cfg_musiccoca": 6.9,
                "cfg_notes": 6.5,
                "cfg_drums": 0.0,
            },
            {
                "id": f"take{take_index:03d}_{midi_meta['name']}_pressure_fog",
                "label": f"{chord} Pressure Fog",
                "category": "cinema_noise",
                "prompt": (
                    f"Create an original sustained synthesizer texture called {chord} Pressure Fog. "
                    "A continuous filtered noise mass wrapped around the held MIDI chord, with soot-like "
                    "broadband air, low sub-harmonic pressure, very slow band-pass color drift, and distant "
                    f"concrete diffusion. Add dark saturation, spectral blur, and wide stable stereo depth. {shared}"
                ),
                "guide_kind": "beatless_drone_pressure_fog",
                "temperature": 0.58,
                "top_k": 42,
                "cfg_musiccoca": 7.0,
                "cfg_notes": 6.4,
                "cfg_drums": 0.0,
            },
            {
                "id": f"take{take_index:03d}_{midi_meta['name']}_feedback_hold",
                "label": f"{chord} Feedback Hold",
                "category": "feedback_drone",
                "prompt": (
                    f"Create an original sustained synthesizer texture called {chord} Feedback Hold. "
                    "A modular feedback network held just below self-oscillation, fused to the MIDI chord, "
                    "with slow resonance breathing, wavefolder warmth, unstable analog edge, and no audible "
                    f"beat grid. Add BBD smear, low-pass damping, and a long connected reverb field. {shared}"
                ),
                "guide_kind": "beatless_drone_feedback_hold",
                "temperature": 0.57,
                "top_k": 38,
                "cfg_musiccoca": 6.8,
                "cfg_notes": 6.6,
                "cfg_drums": 0.0,
            },
            {
                "id": f"take{take_index:03d}_{midi_meta['name']}_carbon_organ",
                "label": f"{chord} Carbon Organ",
                "category": "spectral_pad",
                "prompt": (
                    f"Create an original sustained synthesizer texture called {chord} Carbon Organ. "
                    "A dense dark spectral pad and electronic organ cloud, matte and cinematic, with slow phase "
                    f"smear, muted chorus, tape haze, and a wide black glass harmonic tail. {shared}"
                ),
                "guide_kind": "beatless_drone_carbon_organ",
                "temperature": 0.52,
                "top_k": 32,
                "cfg_musiccoca": 7.1,
                "cfg_notes": 6.7,
                "cfg_drums": 0.0,
            },
        ]

    return [
        {
            "id": f"take{take_index:03d}_{midi_meta['name']}_corroded_glass",
            "label": f"{chord} Corroded Glass",
            "category": "spectral_pad",
            "prompt": (
                f"Create an original sustained synthesizer texture called {chord} Corroded Glass. "
                "A rough spectral glass pad following the supplied MIDI harmony, with brittle wavetable drift, "
                "muted high-frequency air, slow resonant notch movement, and dirty chorus. Add oxidized plate "
                f"reverb and soft tape haze, keeping the surface unbroken and non-melodic. {shared}"
            ),
            "guide_kind": "beatless_drone_corroded_glass",
            "temperature": 0.60,
            "top_k": 52,
            "cfg_musiccoca": 6.7,
            "cfg_notes": 6.3,
            "cfg_drums": 0.0,
        },
        {
            "id": f"take{take_index:03d}_{midi_meta['name']}_frozen_static",
            "label": f"{chord} Frozen Static",
            "category": "granular_glitch",
            "prompt": (
                f"Create an original sustained synthesizer texture called {chord} Frozen Static. "
                "A glitch-damaged continuous synth sheet made of frozen buffers, bit-depth erosion, codec dust, "
                "and granular smear fused into one sustained layer. The digital damage must be texture only, "
                f"not stutters or clicks. Add diffuse reverb and dark spectral softening. {shared}"
            ),
            "guide_kind": "beatless_drone_frozen_static",
            "temperature": 0.66,
            "top_k": 64,
            "cfg_musiccoca": 6.6,
            "cfg_notes": 6.2,
            "cfg_drums": 0.0,
        },
        {
            "id": f"take{take_index:03d}_{midi_meta['name']}_metal_air",
            "label": f"{chord} Metal Air",
            "category": "metallic_harmonic_cloud",
            "prompt": (
                f"Create an original sustained synthesizer texture called {chord} Metal Air. "
                "A dark metallic harmonic cloud built from FM partials and ring-modulated sine stacks, tuned to "
                "the held MIDI chord, with slow resonator drift and rough oxidized air. Add large plate reverb, "
                f"filtered shimmer, and no mallet attacks. {shared}"
            ),
            "guide_kind": "beatless_drone_metal_air",
            "temperature": 0.62,
            "top_k": 58,
            "cfg_musiccoca": 6.8,
            "cfg_notes": 6.4,
            "cfg_drums": 0.0,
        },
        {
            "id": f"take{take_index:03d}_{midi_meta['name']}_dust_horizon",
            "label": f"{chord} Dust Horizon",
            "category": "air_dust_layer",
            "prompt": (
                f"Create an original sustained synthesizer texture called {chord} Dust Horizon. "
                "A wide dark ambient dust layer around the MIDI chord, made from filtered tape hiss, high-pass air, "
                f"slow stereo drift, and distant reverb reflections. Add soft saturation and spectral freeze-thaw motion. {shared}"
            ),
            "guide_kind": "beatless_drone_dust_horizon",
            "temperature": 0.56,
            "top_k": 44,
            "cfg_musiccoca": 6.9,
            "cfg_notes": 6.5,
            "cfg_drums": 0.0,
        },
    ]


def relative_string(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def paths_for(output_dir: Path, take_index: int, midi_meta: dict, prompt_set: dict) -> dict[str, Path]:
    take_dir = output_dir / f"take_{take_index:03d}"
    stem = f"dark_ambient_{midi_meta['name']}_{prompt_set['short']}_128bars"
    return {
        "dir": take_dir,
        "prompts": take_dir / f"{stem}.prompts.tsv",
        "wav": take_dir / f"{stem}.wav",
        "report": take_dir / f"{stem}.report.json",
        "weights": take_dir / f"{stem}.weights.json",
        "audio_check": take_dir / f"{stem}.audio_check.json",
        "log": take_dir / f"{stem}.render.log",
    }


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed {proc.returncode}; see {log_path}")


def validate(paths: dict[str, Path], source: dict, duration: float, embedding_source: str) -> dict:
    probe = sustained.ffprobe(paths["wav"])
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    weights = json.loads(paths["weights"].read_text(encoding="utf-8"))
    check = batch.analyze_float_wav_stream(paths["wav"].resolve())
    paths["audio_check"].write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")

    rendered_duration = float(probe["format"]["duration"])
    if abs(rendered_duration - duration) > 0.05:
        raise RuntimeError(f"duration mismatch {rendered_duration}")
    if not report.get("non_silent"):
        raise RuntimeError("silent render")
    if report.get("midi_mode") != "initial_latch":
        raise RuntimeError(f"bad midi_mode {report.get('midi_mode')}")
    if report.get("embedding_source") != embedding_source:
        raise RuntimeError(f"bad embedding_source {report.get('embedding_source')}")
    midi_info = report.get("midi", {})
    if midi_info.get("note_off_events") != 0:
        raise RuntimeError(f"latch MIDI still has note_off_events={midi_info.get('note_off_events')}")
    if midi_info.get("inferred_held_notes") != len(source["pitches"]):
        raise RuntimeError(
            f"held note count mismatch {midi_info.get('inferred_held_notes')} expected {len(source['pitches'])}"
        )
    if weights.get("frame_rate") != WEIGHT_FRAME_RATE:
        raise RuntimeError(f"bad weight frame rate {weights.get('frame_rate')}")
    if len(weights.get("frames", [])) != int(round(duration * WEIGHT_FRAME_RATE)):
        raise RuntimeError(f"bad weight frame count {len(weights.get('frames', []))}")
    if not check["continuous_after_post_start"]:
        raise RuntimeError(f"quiet windows {check['quiet_windows'][:5]}")
    low_rms = [
        window for window in check["windows"]
        if window["start_seconds"] >= 3.0 and window["rms"] < 0.003
    ]
    if low_rms:
        raise RuntimeError(f"collapsed low-RMS windows {low_rms[:5]}")
    controls = batch.control_summary(weights.get("frames", []))
    for slot in controls["prompt_weights"]:
        if slot["span"] < 0.10:
            raise RuntimeError(f"prompt slot span too small {slot}")

    return {
        "wav": relative_string(paths["wav"]),
        "report": relative_string(paths["report"]),
        "weights": relative_string(paths["weights"]),
        "prompts": relative_string(paths["prompts"]),
        "audio_check": relative_string(paths["audio_check"]),
        "duration_seconds": rendered_duration,
        "rms": report["rms"],
        "peak": report["peak"],
        "midi": midi_info,
        "audio_activity": {
            "quiet_window_count": check["quiet_window_count"],
            "min_post_3s_rms": check["min_post_start_rms"],
            "continuous_after_3s": check["continuous_after_post_start"],
        },
        "controls": controls,
    }


def render_task(task: dict, args: argparse.Namespace) -> dict:
    take_index = task["take_index"]
    source = task["source"]
    prompt_set = task["prompt_set"]
    paths = paths_for(args.output_dir, take_index, source, prompt_set)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    slots = prompt_slots(prompt_set, source, take_index)
    batch.write_prompt_library(paths["prompts"], slots)
    modulation = MODULATION_VARIANTS[(take_index - 1) % len(MODULATION_VARIANTS)]
    summary = None
    used_modulation = None

    for attempt in range(1, args.max_attempts + 1):
        batch_variant = take_index + (attempt - 1) * 1000
        phase = (batch.macro_phase_offset_for_take(batch_variant) + modulation["phase_offset"]) % 1.0
        print(
            f"take {take_index:03d}: {source['name']} {prompt_set['short']} "
            f"attempt={attempt} mod={modulation['label']} phase={phase:.6f}",
            flush=True,
        )
        cmd = [
            str(args.binary),
            "--midi", str(task["latch_midi"]),
            "--profile", "beatless_drone_ambient_cm9",
            "--duration", f"{args.duration:.3f}",
            "--weight-mode", "meter_macro_sine",
            "--control-mode", "ambient_crescendo",
            "--macro-reference-seconds", f"{modulation['macro_reference_seconds']:.3f}",
            "--macro-phase-offset", f"{phase:.9f}",
            "--meter-speed-scale", f"{modulation['meter_speed_scale']:.6f}",
            "--meter-depth", f"{modulation['meter_depth']:.6f}",
            "--macro-depth", f"{modulation['macro_depth']:.6f}",
            "--macro-mix", f"{modulation['macro_mix']:.6f}",
            "--batch-variant", str(batch_variant),
            "--midi-mode", "initial_latch",
            "--midi-refresh-seconds", "0.000",
            "--prompt-library", str(paths["prompts"]),
            "--prompt-page-seconds", f"{args.duration + 1.0:.3f}",
            "--transition", "0.000",
            "--output", str(paths["wav"]),
            "--report", str(paths["report"]),
            "--weights-output", str(paths["weights"]),
        ]
        if args.embedding_source == "audio":
            cmd.append("--prompt-library-audio-prompts")
        else:
            cmd.append("--text-prompts")
        try:
            run_logged(cmd, paths["log"])
            summary = validate(paths, source, args.duration, args.embedding_source)
            used_modulation = {
                **modulation,
                "attempt": attempt,
                "batch_variant": batch_variant,
                "macro_phase_offset": phase,
            }
            break
        except Exception as exc:
            print(f"take {take_index:03d}: validation failed attempt={attempt}: {exc}", flush=True)
            if attempt == args.max_attempts:
                raise

    assert summary is not None and used_modulation is not None
    summary.update({
        "take_index": take_index,
        "source_midi": relative_string(Path(source["path"])),
        "latch_midi": relative_string(task["latch_midi"]),
        "chord_name": source["name"],
        "pitches": source["pitches"],
        "prompt_set": prompt_set,
        "prompt_slots": batch.prompt_manifest(slots),
        "modulation": used_modulation,
        "log": relative_string(paths["log"]),
    })
    print(
        f"take {take_index:03d}: ok rms={summary['rms']:.6f} "
        f"peak={summary['peak']:.3f} off={summary['midi']['note_off_events']}",
        flush=True,
    )
    return summary


def json_source(source: dict) -> dict:
    result = {}
    for key, value in source.items():
        if key == "note_ons":
            continue
        if isinstance(value, Path):
            result[key] = relative_string(value)
        else:
            result[key] = value
    result["path"] = relative_string(Path(source["path"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--duration", type=float, default=DURATION_SECONDS)
    parser.add_argument("--embedding-source", choices=["audio", "text"], default="audio")
    args = parser.parse_args()

    if not args.binary.exists():
        raise SystemExit(f"Missing binary: {args.binary}")
    midis = sorted(args.source_dir.glob("*.mid"))
    if len(midis) != 8:
        raise SystemExit(f"Expected 8 MIDI files, found {len(midis)}")
    if args.jobs <= 0 or args.max_attempts <= 0:
        raise SystemExit("--jobs and --max-attempts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latch_dir = args.output_dir / "midi_latch"
    sources = []
    for midi in midis:
        source = parse_source_midi(midi)
        if abs(source["bpm"] - BPM) > 0.01 or abs(source["duration_seconds"] - args.duration) > 0.01:
            raise SystemExit(
                f"Unexpected MIDI metadata for {midi}: bpm={source['bpm']} "
                f"duration={source['duration_seconds']}"
            )
        latch_midi = latch_dir / f"{midi.stem}_noteon_latch.mid"
        write_latch_midi(source, latch_midi)
        latch_source = parse_source_midi(latch_midi)
        if latch_source["note_off_count_source"] != 0:
            raise SystemExit(f"Latch MIDI contains note off events: {latch_midi}")
        source["latch_midi"] = latch_midi
        sources.append(source)

    free_bytes = shutil.disk_usage(args.output_dir).free
    estimated_wav_bytes = len(midis) * len(PROMPT_SETS) * args.duration * 48000 * 2 * 4
    print(
        "Storage preflight: "
        f"free={free_bytes / (1024 ** 3):.2f} GiB "
        f"estimated_wav={estimated_wav_bytes / (1024 ** 3):.2f} GiB",
        flush=True,
    )

    tasks = []
    take_index = 1
    for source in sources:
        for prompt_set in PROMPT_SETS:
            tasks.append({
                "take_index": take_index,
                "source": source,
                "prompt_set": prompt_set,
                "latch_midi": source["latch_midi"],
            })
            take_index += 1

    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "schema": "mrt-dark-ambient-harmonies-128bars-prompt2-batch-v1",
        "source_dir": relative_string(args.source_dir),
        "output_dir": relative_string(args.output_dir),
        "embedding_source": args.embedding_source,
        "bpm": BPM,
        "bars": BARS,
        "duration_seconds": args.duration,
        "midi_mode": "initial_latch",
        "midi_note_output": "source MIDI pitches copied to Note-On-only latch MIDI; no Note Off events in render MIDI",
        "prompt_sets": PROMPT_SETS,
        "jobs": args.jobs,
        "max_attempts": args.max_attempts,
        "sources": [json_source(source) for source in sources],
        "takes": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Running {len(tasks)} renders with jobs={args.jobs}", flush=True)
    completed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(render_task, task, args) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            manifest["takes"] = sorted(completed, key=lambda item: item["take_index"])
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"manifest updated completed={len(completed)}/{len(tasks)}", flush=True)

    manifest["takes"] = sorted(completed, key=lambda item: item["take_index"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
