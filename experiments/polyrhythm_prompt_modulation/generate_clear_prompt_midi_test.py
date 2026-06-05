#!/usr/bin/env python3
"""Generate a clear prompt-weight MIDI-conditioning test."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT / "data"
DEFAULT_MIDI = ROOT / "assets" / "Am-Minor Prog 01 (i-VI-v-iv).mid"
DEFAULT_JSON = DATA_DIR / "clear_prompt_midi_test_8s.json"
DEFAULT_CSV = DATA_DIR / "clear_prompt_midi_test_8s.csv"
DEFAULT_BPM = 120.0
FPS = 25

SLOTS = [
    {
        "id": "solo_piano_chords",
        "label": "Solo Piano Chords",
        "magenta_prompt": (
            "solo acoustic grand piano only, dry close microphone, sustained "
            "minor jazz chord voicings, no drums, no bass, no synths"
        ),
        "controls": {
            "temperature": 0.82,
            "top_k": 36,
            "cfg_musiccoca": 6.2,
            "cfg_notes": 5.8,
            "cfg_drums": 0.0,
        },
    },
    {
        "id": "chiptune_square_arps",
        "label": "Chiptune Square Arps",
        "magenta_prompt": (
            "8-bit chiptune square wave lead, bright retro video game synth, "
            "rapid staccato arpeggios, sharp digital beeps, no drums"
        ),
        "controls": {
            "temperature": 1.32,
            "top_k": 176,
            "cfg_musiccoca": 7.0,
            "cfg_notes": 3.6,
            "cfg_drums": 0.0,
        },
    },
    {
        "id": "distorted_808_drums",
        "label": "Distorted 808 Drums",
        "magenta_prompt": (
            "heavy distorted 808 sub bass with punchy trap drums, loud kick "
            "and snare, aggressive electronic club rhythm"
        ),
        "controls": {
            "temperature": 1.18,
            "top_k": 144,
            "cfg_musiccoca": 6.6,
            "cfg_notes": 4.8,
            "cfg_drums": 6.2,
        },
    },
    {
        "id": "choir_string_drone",
        "label": "Choir String Drone",
        "magenta_prompt": (
            "wide cinematic string orchestra and choir pad, slow ambient "
            "drone, huge reverb, soft tape noise, no percussion"
        ),
        "controls": {
            "temperature": 0.74,
            "top_k": 58,
            "cfg_musiccoca": 6.0,
            "cfg_notes": 5.6,
            "cfg_drums": 0.0,
        },
    },
]

CHORD_LABELS = ["Am11", "F6/9#11", "Em11", "Dm9"]


@dataclass(frozen=True)
class MidiNote:
    start_tick: int
    end_tick: int
    pitch: int
    velocity: int
    channel: int
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class MidiSummary:
    ppq: int
    format_type: int
    track_count: int
    end_tick: int
    duration_seconds: float
    bpm: float
    time_signature: str
    track_name: str
    notes: list[MidiNote]


def read_u16(data: bytes, offset: int) -> tuple[int, int]:
    return int.from_bytes(data[offset : offset + 2], "big"), offset + 2


def read_u32(data: bytes, offset: int) -> tuple[int, int]:
    return int.from_bytes(data[offset : offset + 4], "big"), offset + 4


def read_varlen(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset


def note_name(pitch: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[pitch % 12]}{(pitch // 12) - 1}"


def tick_to_seconds(tick: int, tempo_events: list[tuple[int, int]], ppq: int) -> float:
    events = sorted(tempo_events or [(0, int(60_000_000 / DEFAULT_BPM))])
    if events[0][0] != 0:
        events.insert(0, (0, int(60_000_000 / DEFAULT_BPM)))
    seconds = 0.0
    previous_tick = 0
    current_tempo = events[0][1]
    for tempo_tick, tempo_us in events[1:]:
        if tick <= tempo_tick:
            break
        seconds += (tempo_tick - previous_tick) * current_tempo / 1_000_000.0 / ppq
        previous_tick = tempo_tick
        current_tempo = tempo_us
    seconds += (tick - previous_tick) * current_tempo / 1_000_000.0 / ppq
    return seconds


def parse_midi(path: Path) -> MidiSummary:
    data = path.read_bytes()
    offset = 0
    if data[offset : offset + 4] != b"MThd":
        raise ValueError(f"{path} is not a Standard MIDI file")
    offset += 4
    header_length, offset = read_u32(data, offset)
    format_type, offset = read_u16(data, offset)
    track_count, offset = read_u16(data, offset)
    ppq, offset = read_u16(data, offset)
    offset = 8 + header_length

    tempo_events: list[tuple[int, int]] = []
    time_signature = "4/4"
    track_name = path.stem
    raw_notes: list[tuple[int, int, int, int, int]] = []
    end_tick = 0

    for _ in range(track_count):
        if data[offset : offset + 4] != b"MTrk":
            raise ValueError("Expected MIDI track chunk")
        offset += 4
        track_length, offset = read_u32(data, offset)
        track = data[offset : offset + track_length]
        offset += track_length

        track_offset = 0
        absolute_tick = 0
        running_status: int | None = None
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}

        while track_offset < len(track):
            delta, track_offset = read_varlen(track, track_offset)
            absolute_tick += delta
            status = track[track_offset]
            if status < 0x80:
                if running_status is None:
                    raise ValueError("Running status appeared before status byte")
                status = running_status
            else:
                track_offset += 1
                if status < 0xF0:
                    running_status = status

            if status == 0xFF:
                meta_type = track[track_offset]
                track_offset += 1
                length, track_offset = read_varlen(track, track_offset)
                payload = track[track_offset : track_offset + length]
                track_offset += length
                if meta_type == 0x03:
                    track_name = payload.decode("utf-8", errors="replace")
                elif meta_type == 0x51 and length == 3:
                    tempo_events.append((absolute_tick, int.from_bytes(payload, "big")))
                elif meta_type == 0x58 and length >= 2:
                    time_signature = f"{payload[0]}/{2 ** payload[1]}"
                elif meta_type == 0x2F:
                    end_tick = max(end_tick, absolute_tick)
                    break
                continue

            if status in (0xF0, 0xF7):
                length, track_offset = read_varlen(track, track_offset)
                track_offset += length
                continue

            event_type = status & 0xF0
            channel = status & 0x0F
            if event_type in (0x80, 0x90):
                pitch = track[track_offset]
                velocity = track[track_offset + 1]
                track_offset += 2
                key = (channel, pitch)
                if event_type == 0x90 and velocity > 0:
                    active.setdefault(key, []).append((absolute_tick, velocity))
                else:
                    starts = active.get(key)
                    if starts:
                        start_tick, start_velocity = starts.pop(0)
                        raw_notes.append((start_tick, absolute_tick, pitch, start_velocity, channel))
                end_tick = max(end_tick, absolute_tick)
            elif event_type in (0xA0, 0xB0, 0xE0):
                track_offset += 2
            elif event_type in (0xC0, 0xD0):
                track_offset += 1
            else:
                raise ValueError(f"Unsupported MIDI status: {status:#x}")

    notes = [
        MidiNote(
            start_tick=start,
            end_tick=end,
            pitch=pitch,
            velocity=velocity,
            channel=channel,
            start_seconds=tick_to_seconds(start, tempo_events, ppq),
            end_seconds=tick_to_seconds(end, tempo_events, ppq),
        )
        for start, end, pitch, velocity, channel in sorted(raw_notes)
    ]
    duration_seconds = tick_to_seconds(end_tick, tempo_events, ppq)
    first_tempo = (tempo_events or [(0, int(60_000_000 / DEFAULT_BPM))])[0][1]
    bpm = 60_000_000 / first_tempo
    return MidiSummary(
        ppq=ppq,
        format_type=format_type,
        track_count=track_count,
        end_tick=end_tick,
        duration_seconds=duration_seconds,
        bpm=bpm,
        time_signature=time_signature,
        track_name=track_name,
        notes=notes,
    )


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def style_weights(time_seconds: float, segment_seconds: float = 2.0) -> list[float]:
    transition_seconds = 0.20
    segment = int(time_seconds // segment_seconds)
    local = time_seconds - (segment * segment_seconds)
    current = segment % len(SLOTS)
    nxt = (current + 1) % len(SLOTS)
    weights = [0.0] * len(SLOTS)
    if local < segment_seconds - transition_seconds:
        weights[current] = 1.0
    else:
        amount = min(1.0, (local - (segment_seconds - transition_seconds)) / transition_seconds)
        weights[current] = 1.0 - amount
        weights[nxt] = amount
    return normalize(weights)


def active_notes_for_frame(notes: list[MidiNote], time_seconds: float, frame_seconds: float) -> tuple[list[int], list[int]]:
    active: list[int] = []
    onset: list[int] = []
    for note in notes:
        if note.start_seconds <= time_seconds < note.end_seconds:
            active.append(note.pitch)
            if note.start_seconds <= time_seconds < note.start_seconds + frame_seconds:
                onset.append(note.pitch)
    return sorted(active), sorted(onset)


def build_test_data(midi_path: Path, midi: MidiSummary) -> dict[str, Any]:
    midi_path = midi_path.resolve()
    frame_count = int(round(midi.duration_seconds * FPS))
    frame_seconds = 1.0 / FPS
    rows: list[dict[str, Any]] = []
    for frame in range(frame_count):
        time_seconds = frame * frame_seconds
        weights = style_weights(time_seconds)
        active_notes, onset_notes = active_notes_for_frame(midi.notes, time_seconds, frame_seconds)
        dominant_index = max(range(len(weights)), key=lambda index: weights[index])
        chord_index = min(len(CHORD_LABELS) - 1, int(time_seconds // 2.0))
        rows.append(
            {
                "frame": frame,
                "time_seconds": round(time_seconds, 4),
                "beat_position": round(time_seconds * midi.bpm / 60.0, 4),
                "bar_position": round(time_seconds * midi.bpm / 60.0 / 4.0, 4),
                "chord_label": CHORD_LABELS[chord_index],
                "dominant_slot": SLOTS[dominant_index]["id"],
                "weights": {
                    slot["id"]: round(weights[index], 6)
                    for index, slot in enumerate(SLOTS)
                },
                "weights_array": [round(value, 6) for value in weights],
                "active_midi_notes": active_notes,
                "active_note_names": [note_name(pitch) for pitch in active_notes],
                "onset_midi_notes": onset_notes,
                "onset_note_names": [note_name(pitch) for pitch in onset_notes],
            }
        )
    return {
        "schema": "mrt-clear-prompt-midi-test-v1",
        "metadata": {
            "title": "Clear Prompt MIDI Diagnostic",
            "fps": FPS,
            "duration_seconds": round(midi.duration_seconds, 6),
            "frame_count": frame_count,
            "bpm": round(midi.bpm, 6),
            "time_signature": midi.time_signature,
            "midi_source": str(midi_path.relative_to(ROOT)),
            "midi_track_name": midi.track_name,
            "midi_format": midi.format_type,
            "midi_tracks": midi.track_count,
            "midi_ppq": midi.ppq,
            "midi_end_tick": midi.end_tick,
            "test_design": (
                "four 2-second one-hot prompt regions aligned to the MIDI "
                "i-VI-v-iv chord changes, with 0.20-second transitions"
            ),
        },
        "slots": SLOTS,
        "midi_notes": [
            {
                "start_tick": note.start_tick,
                "end_tick": note.end_tick,
                "start_seconds": round(note.start_seconds, 6),
                "end_seconds": round(note.end_seconds, 6),
                "pitch": note.pitch,
                "note_name": note_name(note.pitch),
                "velocity": note.velocity,
                "channel": note.channel,
            }
            for note in midi.notes
        ],
        "frames": rows,
    }


def write_csv(path: Path, data: dict[str, Any]) -> None:
    slot_ids = [slot["id"] for slot in data["slots"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "frame",
                "time_seconds",
                "chord_label",
                "dominant_slot",
                *slot_ids,
                "active_note_names",
                "onset_note_names",
            ]
        )
        for frame in data["frames"]:
            writer.writerow(
                [
                    frame["frame"],
                    frame["time_seconds"],
                    frame["chord_label"],
                    frame["dominant_slot"],
                    *[frame["weights"][slot_id] for slot_id in slot_ids],
                    " ".join(frame["active_note_names"]),
                    " ".join(frame["onset_note_names"]),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi", type=Path, default=DEFAULT_MIDI)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    midi = parse_midi(args.midi)
    data = build_test_data(args.midi, midi)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.out_csv, data)
    print(args.out_json)
    print(args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
