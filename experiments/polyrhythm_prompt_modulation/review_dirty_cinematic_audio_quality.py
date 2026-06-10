#!/usr/bin/env python3
"""Review dirty cinematic Cm9 batch WAVs and optionally move suspect takes."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"numpy is required for audio quality review: {exc}") from exc


def find_float_wav_data(path: Path) -> tuple[dict[str, int], int, int]:
    with path.open("rb") as f:
        header = f.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise RuntimeError(f"{path} is not a RIFF/WAVE file")
        fmt: dict[str, int] | None = None
        data_offset = -1
        data_size = -1
        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[:4]
            chunk_size = int.from_bytes(chunk_header[4:8], "little")
            chunk_start = f.tell()
            if chunk_id == b"fmt ":
                chunk = f.read(chunk_size)
                audio_format, channels, sample_rate, _, block_align, bits = struct.unpack_from(
                    "<HHIIHH", chunk, 0
                )
                fmt = {
                    "audio_format": audio_format,
                    "channels": channels,
                    "sample_rate": sample_rate,
                    "block_align": block_align,
                    "bits": bits,
                }
                if chunk_size & 1:
                    f.seek(1, 1)
            elif chunk_id == b"data":
                data_offset = chunk_start
                data_size = chunk_size
                f.seek(chunk_size + (chunk_size & 1), 1)
            else:
                f.seek(chunk_size + (chunk_size & 1), 1)
        if fmt is None or data_offset < 0:
            raise RuntimeError(f"{path} is missing fmt or data chunk")
        if fmt["audio_format"] != 3 or fmt["bits"] != 32:
            raise RuntimeError(f"{path} must be 32-bit float WAV, got {fmt}")
        return fmt, data_offset, data_size


def window_rms(mono: np.ndarray, sample_rate: int, seconds: float) -> list[tuple[float, float, float, float]]:
    frames_per_window = max(1, int(round(sample_rate * seconds)))
    rows = []
    for start in range(0, len(mono), frames_per_window):
        chunk = np.asarray(mono[start:start + frames_per_window], dtype=np.float64)
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        peak = float(np.max(np.abs(chunk)))
        rows.append((start / sample_rate, (start + chunk.size) / sample_rate, rms, peak))
    return rows


def band_metrics(mono: np.ndarray, sample_rate: int, seconds: float = 2.0) -> list[dict[str, float]]:
    frames_per_window = max(1, int(round(sample_rate * seconds)))
    if len(mono) < frames_per_window:
        return []
    window = np.hanning(frames_per_window)
    freqs = np.fft.rfftfreq(frames_per_window, 1.0 / sample_rate)
    high = freqs >= 8000
    eps = 1e-20
    rows = []
    for start in range(0, len(mono) - frames_per_window + 1, frames_per_window):
        chunk = np.asarray(mono[start:start + frames_per_window], dtype=np.float64)
        chunk = chunk - np.mean(chunk)
        spectrum = np.abs(np.fft.rfft(chunk * window)) ** 2 + eps
        total = float(np.sum(spectrum))
        high_ratio = float(np.sum(spectrum[high]) / total)
        flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
        rows.append({
            "start_seconds": start / sample_rate,
            "highband_ratio": high_ratio,
            "spectral_flatness": flatness,
        })
    return rows


def analyze_wav(path: Path, expected_duration: float) -> dict:
    fmt, data_offset, data_size = find_float_wav_data(path)
    channels = fmt["channels"]
    sample_rate = fmt["sample_rate"]
    frames = data_size // (channels * 4)
    samples = np.memmap(path, dtype="<f4", mode="r", offset=data_offset, shape=(frames, channels))
    mono = np.asarray(samples, dtype=np.float64).mean(axis=1)
    del samples

    duration = frames / sample_rate
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
    dc = float(np.mean(mono)) if mono.size else 0.0
    one_second = window_rms(mono, sample_rate, 1.0)
    tenth_second = window_rms(mono, sample_rate, 0.1)
    band_rows = band_metrics(mono, sample_rate, 2.0)

    adjacent_ratios = []
    for previous, current in zip(one_second[:-1], one_second[1:]):
        low = max(min(previous[2], current[2]), 1e-9)
        high = max(previous[2], current[2])
        adjacent_ratios.append(high / low)
    max_adjacent_ratio = max(adjacent_ratios, default=1.0)

    quiet_1s = [
        row for row in one_second
        if row[0] >= 3.0 and (row[2] < 0.0001 or row[3] < 0.001)
    ]
    low_1s = [
        row for row in one_second
        if row[0] >= 3.0 and row[2] < 0.003
    ]
    flat_100ms = sum(1 for row in tenth_second if row[0] >= 3.0 and row[2] < 0.0001)

    longest_noise_like_run = 0
    current_run = 0
    for row in band_rows:
        if row["highband_ratio"] > 0.72 and row["spectral_flatness"] > 0.20:
            current_run += 1
        else:
            longest_noise_like_run = max(longest_noise_like_run, current_run)
            current_run = 0
    longest_noise_like_run = max(longest_noise_like_run, current_run)

    reasons = []
    if abs(duration - expected_duration) > 0.05:
        reasons.append(f"duration={duration:.3f}")
    if not bool(np.isfinite(mono).all()):
        reasons.append("non_finite_samples")
    if peak > 0.98:
        reasons.append(f"clipping_peak={peak:.3f}")
    if abs(dc) > 0.01:
        reasons.append(f"dc_offset={dc:.5f}")
    if quiet_1s:
        reasons.append(f"quiet_1s={len(quiet_1s)}")
    if low_1s:
        reasons.append(f"low_rms_1s={len(low_1s)}")
    if flat_100ms >= 5:
        reasons.append(f"flat_100ms={flat_100ms}")
    if max_adjacent_ratio > 20:
        reasons.append(f"abrupt_1s_rms_ratio={max_adjacent_ratio:.1f}")
    if longest_noise_like_run >= 8:
        reasons.append(f"sustained_highband_noise_run={longest_noise_like_run * 2:.0f}s")

    highband_values = [row["highband_ratio"] for row in band_rows]
    flatness_values = [row["spectral_flatness"] for row in band_rows]
    post_3s_rms = [row[2] for row in one_second if row[0] >= 3.0]
    return {
        "wav": str(path),
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "rms_mono": rms,
        "peak_mono": peak,
        "dc_offset_mono": dc,
        "min_1s_rms_after_3s": min(post_3s_rms, default=0.0),
        "max_adjacent_1s_rms_ratio": max_adjacent_ratio,
        "median_highband_ratio_2s": float(np.median(highband_values)) if highband_values else 0.0,
        "max_highband_ratio_2s": max(highband_values, default=0.0),
        "median_spectral_flatness_2s": float(np.median(flatness_values)) if flatness_values else 0.0,
        "max_spectral_flatness_2s": max(flatness_values, default=0.0),
        "longest_noise_like_run_seconds": int(longest_noise_like_run * 2),
        "status": "suspect" if reasons else "ok",
        "reasons": reasons,
    }


def take_index_from_dir(path: Path) -> int:
    return int(path.name.split("_", 1)[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-duration", type=float, required=True)
    parser.add_argument("--move-suspects", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    take_dirs = sorted(
        [path for path in output_dir.glob("take_*") if path.is_dir()],
        key=take_index_from_dir,
    )
    if not take_dirs:
        raise SystemExit(f"No take directories found in {output_dir}")

    results = []
    quarantine_dir = output_dir / "_moved_suspect_audio"
    if args.move_suspects:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    for take_dir in take_dirs:
        wavs = sorted(take_dir.glob("*.wav"))
        if len(wavs) != 1:
            result = {
                "take_index": take_index_from_dir(take_dir),
                "take_dir": str(take_dir),
                "status": "suspect",
                "reasons": [f"wav_count={len(wavs)}"],
            }
        else:
            result = analyze_wav(wavs[0], args.expected_duration)
            result["take_index"] = take_index_from_dir(take_dir)
            result["take_dir"] = str(take_dir)

        if args.move_suspects and result["status"] == "suspect":
            destination = quarantine_dir / take_dir.name
            if destination.exists():
                suffix = 1
                while (quarantine_dir / f"{take_dir.name}_old{suffix}").exists():
                    suffix += 1
                destination = quarantine_dir / f"{take_dir.name}_old{suffix}"
            shutil.move(str(take_dir), str(destination))
            result["status"] = "moved_suspect"
            result["moved_to"] = str(destination)
        results.append(result)

    accepted = [result for result in results if result["status"] == "ok"]
    moved = [result for result in results if result["status"] == "moved_suspect"]
    suspects = [result for result in results if result["status"] == "suspect"]
    review = {
        "schema": "dirty-cinematic-cm9-audio-quality-review-v1",
        "output_dir": str(output_dir),
        "review_policy": {
            "duration_seconds": f"{args.expected_duration} +/- 0.05",
            "clip_peak_threshold": 0.98,
            "dc_offset_abs_threshold": 0.01,
            "quiet_1s_after_3s": "rms < 0.0001 or peak < 0.001",
            "low_rms_1s_after_3s": "rms < 0.003",
            "flat_100ms_threshold": ">=5 windows after 3s",
            "abrupt_1s_rms_ratio_threshold": 20,
            "sustained_highband_noise": "2s windows high_ratio > 0.72 and flatness > 0.20 for >=16s",
        },
        "counts": {
            "total": len(results),
            "ok": len(accepted),
            "suspect": len(suspects),
            "moved_suspect": len(moved),
        },
        "results": results,
    }
    review_path = output_dir / "audio_quality_review.json"
    review_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")

    accepted_manifest = {
        "schema": "dirty-cinematic-cm9-accepted-audio-v1",
        "output_dir": str(output_dir),
        "quality_review": str(review_path),
        "accepted_count": len(accepted),
        "moved_suspect_count": len(moved),
        "suspect_count": len(suspects),
        "accepted_takes": accepted,
        "moved_suspect_takes": moved,
        "suspect_takes": suspects,
    }
    accepted_manifest_path = output_dir / "accepted_manifest.json"
    accepted_manifest_path.write_text(json.dumps(accepted_manifest, indent=2) + "\n", encoding="utf-8")

    print(f"review={review_path}")
    print(f"accepted_manifest={accepted_manifest_path}")
    print(
        f"total={len(results)} ok={len(accepted)} "
        f"suspect={len(suspects)} moved_suspect={len(moved)}"
    )
    for result in results:
        reasons = ",".join(result.get("reasons", []))
        print(f"take_{result['take_index']:03d} {result['status']} {reasons}")
    return 1 if suspects else 0


if __name__ == "__main__":
    raise SystemExit(main())
