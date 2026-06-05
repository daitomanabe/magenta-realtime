#!/usr/bin/env python3
"""Generate four-prompt polyrhythm modulation data.

The weights are normalized at every frame so they can drive both Magenta prompt
blending and video opacity blending. Frequencies are expressed as cycles per two
bars, producing a 3:5:7:11 relationship with independent phase offsets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT / "data"
PROMPT_PLAN = EXPERIMENT / "prompt_plan.json"


def raised_cosine(time_seconds: float, frequency_hz: float, phase_degrees: float) -> float:
    phase = math.radians(phase_degrees)
    return 0.5 + 0.5 * math.sin((2.0 * math.pi * frequency_hz * time_seconds) + phase)


def beat_pulse(beat_position: float, subdivision: float, phase_degrees: float) -> float:
    phase = math.radians(phase_degrees)
    return math.exp(2.6 * (math.cos((2.0 * math.pi * subdivision * beat_position) + phase) - 1.0))


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def build_modulation(plan: dict[str, Any]) -> dict[str, Any]:
    bpm = float(plan["bpm"])
    fps = int(plan["fps"])
    duration = float(plan["duration_seconds"])
    beat_hz = bpm / 60.0
    two_bar_seconds = (8.0 * 60.0) / bpm
    frames = int(round(duration * fps))

    slots: list[dict[str, Any]] = []
    for slot in plan["slots"]:
        frequency_hz = float(slot["cycles_per_two_bars"]) / two_bar_seconds
        enriched = {
            **slot,
            "frequency_hz": round(frequency_hz, 6),
            "period_seconds": round(1.0 / frequency_hz, 6),
        }
        slots.append(enriched)

    frame_rows: list[dict[str, Any]] = []
    for frame in range(frames):
        time_seconds = frame / fps
        beat_position = time_seconds * beat_hz
        raw: list[float] = []
        for index, slot in enumerate(slots):
            lfo = raised_cosine(time_seconds, slot["frequency_hz"], slot["phase_degrees"])
            pulse = beat_pulse(beat_position, 0.5 + (index * 0.25), slot["phase_degrees"] * 0.5)
            value = 0.055 + (lfo ** float(slot["gamma"])) * (0.72 + 0.28 * pulse)
            raw.append(value)

        weights = normalize(raw)
        frame_rows.append(
            {
                "frame": frame,
                "time_seconds": round(time_seconds, 4),
                "beat_position": round(beat_position, 4),
                "bar_position": round(beat_position / 4.0, 4),
                "dominant_slot": slots[max(range(len(weights)), key=lambda i: weights[i])]["id"],
                "weights": {
                    slot["id"]: round(weights[index], 6)
                    for index, slot in enumerate(slots)
                },
                "weights_array": [round(value, 6) for value in weights],
            }
        )

    return {
        "schema": "mrt-polyrhythm-prompt-modulation-v1",
        "metadata": {
            "bpm": bpm,
            "fps": fps,
            "duration_seconds": duration,
            "frame_count": frames,
            "beat_hz": round(beat_hz, 6),
            "two_bar_seconds": round(two_bar_seconds, 6),
            "normalization": "sum-to-one",
        },
        "base_runway_prompt": plan["base_runway_prompt"],
        "slots": slots,
        "frames": frame_rows,
    }


def write_csv(path: Path, modulation: dict[str, Any]) -> None:
    slot_ids = [slot["id"] for slot in modulation["slots"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "time_seconds", "beat_position", "bar_position", "dominant_slot", *slot_ids])
        for frame in modulation["frames"]:
            writer.writerow(
                [
                    frame["frame"],
                    frame["time_seconds"],
                    frame["beat_position"],
                    frame["bar_position"],
                    frame["dominant_slot"],
                    *[frame["weights"][slot_id] for slot_id in slot_ids],
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROMPT_PLAN)
    parser.add_argument("--out-json", type=Path, default=DATA_DIR / "modulation_10s_104bpm.json")
    parser.add_argument("--out-csv", type=Path, default=DATA_DIR / "modulation_10s_104bpm.csv")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    modulation = build_modulation(plan)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(modulation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.out_csv, modulation)
    print(args.out_json)
    print(args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
