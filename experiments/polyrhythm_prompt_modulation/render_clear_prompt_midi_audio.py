#!/usr/bin/env python3
"""Render a clear prompt-weight test with MIDI note conditioning."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from magenta_rt import MagentaRT2Mlxfn
from magenta_rt.audio import Waveform


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DEFAULT_DATA = EXPERIMENT / "data" / "clear_prompt_midi_test_8s.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "polyrhythm_prompt_modulation"


def load_test_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        from generate_clear_prompt_midi_test import DEFAULT_MIDI, build_test_data, parse_midi

        midi = parse_midi(DEFAULT_MIDI)
        data = build_test_data(DEFAULT_MIDI, midi)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def blend_controls(slots: list[dict[str, Any]], weights: np.ndarray) -> dict[str, float]:
    controls: dict[str, float] = {}
    for key in ["temperature", "top_k", "cfg_musiccoca", "cfg_notes", "cfg_drums"]:
        values = np.array([slot["controls"][key] for slot in slots], dtype=np.float32)
        controls[key] = float(np.sum(values * weights))
    controls["top_k"] = int(round(controls["top_k"]))
    return controls


def note_tokens(frame: dict[str, Any]) -> list[int]:
    tokens = [0] * 128
    for pitch in frame["active_midi_notes"]:
        tokens[pitch] = 1
    for pitch in frame["onset_midi_notes"]:
        tokens[pitch] = 2
    return tokens


def drum_token(frame: dict[str, Any], slots: list[dict[str, Any]]) -> list[int]:
    slot_ids = [slot["id"] for slot in slots]
    drum_weight = frame["weights_array"][slot_ids.index("distorted_808_drums")]
    beat = float(frame["beat_position"])
    eighth_pulse = (beat % 0.5) < 0.08
    return [1 if drum_weight > 0.55 and eighth_pulse else 0]


def segment_metrics(samples: np.ndarray, sample_rate: int, start_seconds: float, end_seconds: float) -> dict[str, float]:
    start = int(round(start_seconds * sample_rate))
    end = int(round(end_seconds * sample_rate))
    segment = samples[start:end]
    mono = np.mean(segment, axis=1)
    if mono.size == 0:
        return {}
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono))))
    window = np.hanning(mono.size)
    spectrum = np.fft.rfft(mono * window)
    power = np.square(np.abs(spectrum))
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sample_rate)
    total_power = float(np.sum(power) + 1e-12)
    centroid = float(np.sum(freqs * power) / total_power)

    def band_ratio(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(np.sum(power[mask]) / total_power)

    zero_crossings = np.mean(np.abs(np.diff(np.signbit(mono))).astype(np.float32))
    flatness = math.exp(float(np.mean(np.log(power + 1e-12)))) / (float(np.mean(power)) + 1e-12)
    return {
        "start_seconds": round(start_seconds, 4),
        "end_seconds": round(end_seconds, 4),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "spectral_centroid_hz": round(centroid, 3),
        "sub_20_120_ratio": round(band_ratio(20, 120), 6),
        "low_mid_120_800_ratio": round(band_ratio(120, 800), 6),
        "mid_high_800_4000_ratio": round(band_ratio(800, 4000), 6),
        "air_4000_12000_ratio": round(band_ratio(4000, 12000), 6),
        "zero_crossing_rate": round(float(zero_crossings), 6),
        "spectral_flatness": round(flatness, 6),
    }


def match_segment_rms(
    samples: np.ndarray,
    sample_rate: int,
    segment_seconds: float,
    target_rms: float,
    max_gain: float = 8.0,
) -> tuple[np.ndarray, list[float]]:
    matched = np.array(samples, copy=True)
    gains: list[float] = []
    segment_count = int(math.ceil(len(samples) / (segment_seconds * sample_rate)))
    for segment_index in range(segment_count):
        start = int(round(segment_index * segment_seconds * sample_rate))
        end = min(len(samples), int(round((segment_index + 1) * segment_seconds * sample_rate)))
        segment = matched[start:end]
        rms = float(np.sqrt(np.mean(np.square(segment)))) if len(segment) else 0.0
        gain = min(max_gain, target_rms / rms) if rms > 0.000001 else 1.0
        matched[start:end] = segment * gain
        gains.append(gain)
    return matched, gains


def build_frame_args(
    mrt: MagentaRT2Mlxfn,
    prompt_embeddings_np: np.ndarray,
    slots: list[dict[str, Any]],
    frame: dict[str, Any],
) -> list[mx.array]:
    weights = np.array(frame["weights_array"], dtype=np.float32)
    blended_style = np.sum(prompt_embeddings_np * weights[:, None], axis=0).astype(np.float32)
    style_tokens = mrt.tokenize_style(blended_style).tolist()
    controls = blend_controls(slots, weights)
    return mrt._build_mlxfn_args(  # pylint: disable=protected-access
        style_tokens=style_tokens,
        notes=note_tokens(frame),
        drums=drum_token(frame, slots),
        cfg_musiccoca=float(controls["cfg_musiccoca"]),
        cfg_notes=float(controls["cfg_notes"]),
        cfg_drums=float(controls["cfg_drums"]),
        temperature=float(controls["temperature"]),
        top_k=int(controls["top_k"]),
    )


def render(
    data: dict[str, Any],
    data_path: Path,
    model: str,
    output_wav: Path,
    report_path: Path,
    preroll_seconds: float,
    match_rms: bool,
    target_rms: float,
) -> None:
    metadata = data["metadata"]
    slots = data["slots"]
    frames = data["frames"]

    mrt = MagentaRT2Mlxfn(
        size=model,
        temperature=1.0,
        top_k=96,
        cfg_musiccoca=6.0,
        cfg_notes=5.5,
        cfg_drums=0.0,
    )

    prompt_embeddings = [
        mrt.embed_style(slot["magenta_prompt"], use_mapper=True, seed=480 + index)
        for index, slot in enumerate(slots)
    ]
    prompt_embeddings_np = np.stack(prompt_embeddings, axis=0).astype(np.float32)

    state = None
    audio_frames: list[np.ndarray] = []
    started_at = time.time()
    dominant_counts: dict[str, int] = {}

    preroll_frames = int(round(preroll_seconds * metadata["fps"]))
    for _ in range(preroll_frames):
        args = build_frame_args(mrt, prompt_embeddings_np, slots, frames[0])
        if state is None:
            state = list(mrt._initial_state)  # pylint: disable=protected-access
        outputs = mrt._fn(args + state)  # pylint: disable=protected-access
        mx.eval(outputs)
        state = list(outputs[1:])

    for index, frame in enumerate(frames):
        args = build_frame_args(mrt, prompt_embeddings_np, slots, frame)
        if state is None:
            state = list(mrt._initial_state)  # pylint: disable=protected-access
        outputs = mrt._fn(args + state)  # pylint: disable=protected-access
        mx.eval(outputs)
        audio_frames.append(np.array(outputs[0]))
        state = list(outputs[1:])
        dominant_counts[frame["dominant_slot"]] = dominant_counts.get(frame["dominant_slot"], 0) + 1
        if (index + 1) % 50 == 0:
            print(f"rendered {index + 1}/{len(frames)} frames")

    all_audio = np.concatenate(audio_frames, axis=-1)
    samples = all_audio[0].T.astype(np.float32) / 32768.0
    raw_samples = np.array(samples, copy=True)
    segment_seconds = 2.0
    segment_gains = [1.0] * len(slots)
    if match_rms:
        samples, segment_gains = match_segment_rms(samples, 48_000, segment_seconds, target_rms)
    raw_peak = float(np.max(np.abs(raw_samples)))
    processed_peak = float(np.max(np.abs(samples)))
    target_peak = 10 ** (-1.0 / 20.0)
    normalization_gain = min(1.0, target_peak / processed_peak) if processed_peak > 0 else 1.0
    samples = samples * normalization_gain
    wav = Waveform(samples, sample_rate=48_000)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write(str(output_wav))

    segment_reports = []
    for segment_index, slot in enumerate(slots):
        start = segment_index * segment_seconds
        end = min(start + segment_seconds, wav.seconds)
        metrics = segment_metrics(wav.samples, wav.sample_rate, start, end)
        metrics.update(
            {
                "segment": segment_index + 1,
                "slot_id": slot["id"],
                "slot_label": slot["label"],
                "magenta_prompt": slot["magenta_prompt"],
                "chord_label": frames[int(start * metadata["fps"])]["chord_label"],
                "active_note_names": frames[int(start * metadata["fps"])]["active_note_names"],
                "segment_gain": round(segment_gains[segment_index] * normalization_gain, 6),
                "segment_gain_db": round(20.0 * math.log10(max(0.000001, segment_gains[segment_index] * normalization_gain)), 3),
            }
        )
        segment_reports.append(metrics)

    peak = float(np.max(np.abs(wav.samples)))
    rms = float(np.sqrt(np.mean(np.square(wav.samples))))
    report = {
        "schema": "mrt-clear-prompt-midi-render-report-v1",
        "output_wav": str(output_wav),
        "data_path": str(data_path),
        "model": model,
        "duration_seconds": wav.seconds,
        "expected_duration_seconds": metadata["duration_seconds"],
        "sample_rate": wav.sample_rate,
        "channels": wav.num_channels,
        "samples": wav.num_samples,
        "peak": peak,
        "raw_peak": raw_peak,
        "processed_peak_before_normalization": processed_peak,
        "normalization_gain": normalization_gain,
        "segment_rms_matched": match_rms,
        "target_segment_rms": target_rms if match_rms else None,
        "rms": rms,
        "non_silent": bool(peak > 0.0001 and rms > 0.00001),
        "frames": len(frames),
        "dominant_counts": dominant_counts,
        "preroll_seconds": preroll_seconds,
        "midi_source": metadata["midi_source"],
        "midi_track_name": metadata["midi_track_name"],
        "segments": segment_reports,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_wav)
    print(report_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="mrt2_small")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "magenta_clear_prompt_midi_8s.wav")
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT_DIR / "magenta_clear_prompt_midi_8s.report.json")
    parser.add_argument("--preroll-seconds", type=float, default=2.0)
    parser.add_argument("--match-segment-rms", action="store_true")
    parser.add_argument("--target-rms", type=float, default=0.06)
    args = parser.parse_args()

    data = load_test_data(args.data)
    render(
        data,
        args.data,
        args.model,
        args.output,
        args.report,
        args.preroll_seconds,
        args.match_segment_rms,
        args.target_rms,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
