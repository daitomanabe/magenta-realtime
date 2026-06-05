#!/usr/bin/env python3
"""Generate 180 second longform prompt modulation and video segment prompts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EXPERIMENT = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT / "data"
PLAN = EXPERIMENT / "longform_plan.json"


SLOT_IDS = ["tension_chords", "glassy_arps", "broken_beat", "granular_texture"]
BASE_CYCLES = [3.0, 5.0, 7.0, 11.0]
LATE_CYCLES = [5.0, 8.0, 13.0, 21.0]
PHASE_DEGREES = [0.0, 76.0, 151.0, 223.0]
GAMMA = [1.25, 1.65, 2.05, 1.45]


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - (2.0 * value))


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def interpolate(a: float, b: float, amount: float) -> float:
    return a + ((b - a) * amount)


def section_at(plan: dict[str, Any], time_seconds: float) -> dict[str, Any]:
    for section in plan["sections"]:
        if section["start"] <= time_seconds < section["end"]:
            return section
    return plan["sections"][-1]


def section_progress(section: dict[str, Any], time_seconds: float) -> float:
    return smoothstep((time_seconds - section["start"]) / (section["end"] - section["start"]))


def section_energy(section: dict[str, Any], time_seconds: float) -> float:
    progress = section_progress(section, time_seconds)
    start, end = section["energy"]
    return interpolate(float(start), float(end), progress)


def lfo(time_seconds: float, frequency: float, phase_degrees: float, phase_drift: float) -> float:
    phase = math.radians(phase_degrees + phase_drift)
    return 0.5 + 0.5 * math.sin((2.0 * math.pi * frequency * time_seconds) + phase)


def beat_pulse(beat_position: float, division: float, phase_degrees: float) -> float:
    phase = math.radians(phase_degrees)
    return math.exp(2.9 * (math.cos((2.0 * math.pi * division * beat_position) + phase) - 1.0))


def build_frames(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bpm = float(plan["bpm"])
    fps = int(plan["fps"])
    duration = float(plan["duration_seconds"])
    frames = int(round(duration * fps))
    beat_hz = bpm / 60.0
    two_bar_seconds = (8.0 * 60.0) / bpm

    rows: list[dict[str, Any]] = []
    for frame in range(frames):
        time_seconds = frame / fps
        beat_position = time_seconds * beat_hz
        bar_position = beat_position / 4.0
        late = smoothstep((time_seconds - 70.0) / 95.0)
        section = section_at(plan, time_seconds)
        energy = section_energy(section, time_seconds)
        focus = [float(section["focus"][slot_id]) for slot_id in SLOT_IDS]

        raw: list[float] = []
        for index, slot_id in enumerate(SLOT_IDS):
            cycles = interpolate(BASE_CYCLES[index], LATE_CYCLES[index], late)
            frequency = cycles / two_bar_seconds
            phase_drift = late * (38.0 + (index * 27.0)) + math.sin(time_seconds * 0.031) * 19.0
            periodic = lfo(time_seconds, frequency, PHASE_DEGREES[index], phase_drift) ** GAMMA[index]
            rhythmic = beat_pulse(beat_position, 0.5 + (index * 0.25) + late * 0.5, PHASE_DEGREES[index] * 0.4)
            value = 0.025 + (0.52 * focus[index]) + (periodic * (0.20 + 0.72 * energy) * (0.72 + 0.28 * rhythmic))
            raw.append(value)

        weights = normalize(raw)
        dominant_index = max(range(len(weights)), key=lambda i: weights[i])
        drum_weight = weights[SLOT_IDS.index("broken_beat")]
        texture_weight = weights[SLOT_IDS.index("granular_texture")]
        arp_weight = weights[SLOT_IDS.index("glassy_arps")]
        tension_weight = weights[SLOT_IDS.index("tension_chords")]
        controls = {
            "temperature": round(0.72 + (0.46 * energy) + (0.12 * texture_weight), 4),
            "top_k": int(round(32 + (148 * energy) + (42 * arp_weight))),
            "cfg_musiccoca": round(1.45 + (4.45 * energy) + (0.85 * max(weights)), 4),
            "cfg_notes": round(0.20 + (1.25 * tension_weight) + (0.35 * arp_weight), 4),
            "cfg_drums": round(0.12 + (3.35 * drum_weight) + (1.1 * energy), 4),
            "drum_gate": int(energy > 0.32 and drum_weight > (0.18 - 0.05 * late) and (beat_position % 0.5) < (0.10 + 0.05 * energy)),
        }
        rows.append(
            {
                "frame": frame,
                "time_seconds": round(time_seconds, 4),
                "beat_position": round(beat_position, 4),
                "bar_position": round(bar_position, 4),
                "section": section["id"],
                "energy": round(energy, 6),
                "dominant_slot": SLOT_IDS[dominant_index],
                "weights": {slot_id: round(weights[index], 6) for index, slot_id in enumerate(SLOT_IDS)},
                "weights_array": [round(value, 6) for value in weights],
                "controls": controls,
            }
        )

    segments = build_segments(plan, rows)
    return rows, segments


def build_segments(plan: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segment_seconds = float(plan["segment_seconds"])
    duration = float(plan["duration_seconds"])
    fps = int(plan["fps"])
    count = int(math.ceil(duration / segment_seconds))
    slots_by_id = {slot["id"]: slot for slot in plan["slots"]}
    segments: list[dict[str, Any]] = []
    for index in range(count):
        start = index * segment_seconds
        end = min(duration, start + segment_seconds)
        start_frame = int(round(start * fps))
        end_frame = int(round(end * fps))
        segment_rows = rows[start_frame:end_frame]
        if not segment_rows:
            continue
        averages = {
            slot_id: sum(row["weights"][slot_id] for row in segment_rows) / len(segment_rows)
            for slot_id in SLOT_IDS
        }
        top_slots = sorted(SLOT_IDS, key=lambda slot_id: averages[slot_id], reverse=True)[:2]
        section = section_at(plan, start + ((end - start) * 0.5))
        energy = sum(row["energy"] for row in segment_rows) / len(segment_rows)
        visual_traits = "; ".join(slots_by_id[slot_id]["visual_language"] for slot_id in top_slots)
        prompt = (
            f"{plan['base_runway_rules']} Segment {index + 1:02d} of an 18-part three-minute audiovisual piece. "
            f"Section: {section['label']}. Scene: {section['runway_scene']}. "
            f"Dominant visual systems: {visual_traits}. "
            f"Energy level {energy:.2f}; camera movement should feel continuous from the previous segment but with a clear new phrase. "
            "No humans, no instruments, no UI, no subtitles."
        )
        segments.append(
            {
                "index": index + 1,
                "start": round(start, 4),
                "end": round(end, 4),
                "section": section["id"],
                "top_slots": top_slots,
                "avg_weights": {slot_id: round(averages[slot_id], 6) for slot_id in SLOT_IDS},
                "promptText": prompt,
            }
        )
    return segments


def write_csv(path: Path, frames: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "frame",
                "time_seconds",
                "beat_position",
                "bar_position",
                "section",
                "energy",
                "dominant_slot",
                *SLOT_IDS,
                "temperature",
                "top_k",
                "cfg_musiccoca",
                "cfg_notes",
                "cfg_drums",
                "drum_gate",
            ]
        )
        for row in frames:
            writer.writerow(
                [
                    row["frame"],
                    row["time_seconds"],
                    row["beat_position"],
                    row["bar_position"],
                    row["section"],
                    row["energy"],
                    row["dominant_slot"],
                    *[row["weights"][slot_id] for slot_id in SLOT_IDS],
                    row["controls"]["temperature"],
                    row["controls"]["top_k"],
                    row["controls"]["cfg_musiccoca"],
                    row["controls"]["cfg_notes"],
                    row["controls"]["cfg_drums"],
                    row["controls"]["drum_gate"],
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--out-json", type=Path, default=DATA_DIR / "longform_modulation_180s_104bpm.json")
    parser.add_argument("--out-csv", type=Path, default=DATA_DIR / "longform_modulation_180s_104bpm.csv")
    parser.add_argument("--out-segments", type=Path, default=DATA_DIR / "longform_runway_segments.json")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    frames, segments = build_frames(plan)
    modulation = {
        "schema": "mrt-longform-polyrhythm-modulation-v1",
        "metadata": {
            "title": plan["title"],
            "bpm": plan["bpm"],
            "fps": plan["fps"],
            "duration_seconds": plan["duration_seconds"],
            "frame_count": len(frames),
            "slot_ids": SLOT_IDS,
        },
        "slots": plan["slots"],
        "sections": plan["sections"],
        "segments": segments,
        "frames": frames,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(modulation, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    write_csv(args.out_csv, frames)
    args.out_segments.write_text(json.dumps({"segments": segments}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.out_json)
    print(args.out_csv)
    print(args.out_segments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
