#!/usr/bin/env python3
"""Generate an obvious four-style prompt-weight switch test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPERIMENT = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT / "data"
SLOT_IDS = ["tension_chords", "glassy_arps", "broken_beat", "granular_texture"]


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def style_pair(time_seconds: float) -> tuple[int, int, float]:
    """Return current style, next style, and transition amount."""
    segment = int(time_seconds // 2.0)
    local = time_seconds - (segment * 2.0)
    current = segment % len(SLOT_IDS)
    nxt = (current + 1) % len(SLOT_IDS)
    if local < 1.45:
        return current, current, 0.0
    amount = min(1.0, (local - 1.45) / 0.55)
    return current, nxt, amount


def build_modulation(duration: float = 8.0, fps: int = 25) -> dict[str, Any]:
    frames = int(round(duration * fps))
    rows: list[dict[str, Any]] = []
    for frame in range(frames):
        time_seconds = frame / fps
        current, nxt, amount = style_pair(time_seconds)
        weights = [0.0] * len(SLOT_IDS)
        weights[current] = 1.0 - amount
        weights[nxt] += amount
        weights = normalize(weights)
        rows.append(
            {
                "frame": frame,
                "time_seconds": round(time_seconds, 4),
                "beat_position": round(time_seconds * 104.0 / 60.0, 4),
                "bar_position": round(time_seconds * 104.0 / 60.0 / 4.0, 4),
                "dominant_slot": SLOT_IDS[max(range(len(weights)), key=lambda index: weights[index])],
                "weights": {
                    slot_id: round(weights[index], 6)
                    for index, slot_id in enumerate(SLOT_IDS)
                },
                "weights_array": [round(value, 6) for value in weights],
            }
        )
    return {
        "schema": "mrt-diagnostic-prompt-weight-switch-v1",
        "metadata": {
            "bpm": 104,
            "fps": fps,
            "duration_seconds": duration,
            "frame_count": frames,
            "normalization": "sum-to-one",
            "test_design": "four 2-second style regions with 0.55-second crossfades",
        },
        "slots": [{"id": slot_id, "label": slot_id.replace("_", " ").title()} for slot_id in SLOT_IDS],
        "frames": rows,
    }


def write_csv(path: Path, modulation: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "time_seconds", "dominant_slot", *SLOT_IDS])
        for row in modulation["frames"]:
            writer.writerow(
                [
                    row["frame"],
                    row["time_seconds"],
                    row["dominant_slot"],
                    *[row["weights"][slot_id] for slot_id in SLOT_IDS],
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=DATA_DIR / "diagnostic_weight_switch_8s.json")
    parser.add_argument("--out-csv", type=Path, default=DATA_DIR / "diagnostic_weight_switch_8s.csv")
    args = parser.parse_args()

    modulation = build_modulation()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(modulation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.out_csv, modulation)
    print(args.out_json)
    print(args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
