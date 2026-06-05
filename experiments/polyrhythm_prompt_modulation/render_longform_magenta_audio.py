#!/usr/bin/env python3
"""Render one continuous 180 second Magenta RealTime longform WAV."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from magenta_rt import MagentaRT2Mlxfn
from magenta_rt.audio import Waveform


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DEFAULT_MODULATION = EXPERIMENT / "data" / "longform_modulation_180s_104bpm.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "polyrhythm_prompt_modulation" / "longform"


def load_modulation(path: Path) -> dict[str, Any]:
    if not path.exists():
        from generate_longform_modulation import build_frames

        plan = json.loads((EXPERIMENT / "longform_plan.json").read_text(encoding="utf-8"))
        frames, segments = build_frames(plan)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": "mrt-longform-polyrhythm-modulation-v1",
                    "metadata": {
                        "title": plan["title"],
                        "bpm": plan["bpm"],
                        "fps": plan["fps"],
                        "duration_seconds": plan["duration_seconds"],
                        "frame_count": len(frames),
                    },
                    "slots": plan["slots"],
                    "sections": plan["sections"],
                    "segments": segments,
                    "frames": frames,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def render(modulation: dict[str, Any], modulation_path: Path, model: str, output_wav: Path, report_path: Path) -> None:
    slots = modulation["slots"]
    frames = modulation["frames"]
    metadata = modulation["metadata"]
    slot_ids = metadata.get("slot_ids") or [slot["id"] for slot in slots]

    mrt = MagentaRT2Mlxfn(
        size=model,
        temperature=1.0,
        top_k=96,
        cfg_musiccoca=3.0,
        cfg_notes=0.55,
        cfg_drums=0.65,
    )

    prompt_embeddings = []
    for index, slot in enumerate(slots):
        prompt_embeddings.append(mrt.embed_style(slot["magenta_prompt"], use_mapper=True, seed=240 + index))
    prompt_embeddings_np = np.stack(prompt_embeddings, axis=0).astype(np.float32)

    state = None
    audio_frames: list[np.ndarray] = []
    section_counts: dict[str, int] = {}
    started_at = time.time()

    for index, frame in enumerate(frames):
        weights = np.array(frame["weights_array"], dtype=np.float32)
        blended_style = np.sum(prompt_embeddings_np * weights[:, None], axis=0).astype(np.float32)
        style_tokens = mrt.tokenize_style(blended_style).tolist()
        controls = frame["controls"]

        drums = [1 if int(controls["drum_gate"]) else 0]
        args = mrt._build_mlxfn_args(  # pylint: disable=protected-access
            style_tokens=style_tokens,
            drums=drums,
            cfg_musiccoca=float(controls["cfg_musiccoca"]),
            cfg_notes=float(controls["cfg_notes"]),
            cfg_drums=float(controls["cfg_drums"]),
            temperature=float(controls["temperature"]),
            top_k=int(controls["top_k"]),
        )
        if state is None:
            state = list(mrt._initial_state)  # pylint: disable=protected-access
        outputs = mrt._fn(args + state)  # pylint: disable=protected-access
        mx.eval(outputs)
        audio_frames.append(np.array(outputs[0]))
        state = list(outputs[1:])
        section_counts[frame["section"]] = section_counts.get(frame["section"], 0) + 1
        if (index + 1) % 250 == 0:
            elapsed = time.time() - started_at
            print(f"rendered {index + 1}/{len(frames)} frames ({(index + 1) / metadata['fps']:.1f}s), elapsed {elapsed:.1f}s")

    all_audio = np.concatenate(audio_frames, axis=-1)
    samples = all_audio[0].T.astype(np.float32) / 32768.0
    wav = Waveform(samples, sample_rate=48_000)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write(str(output_wav))

    peak = float(np.max(np.abs(wav.samples)))
    rms = float(np.sqrt(np.mean(np.square(wav.samples))))
    report = {
        "schema": "mrt-longform-render-report-v1",
        "output_wav": str(output_wav),
        "modulation_path": str(modulation_path),
        "model": model,
        "duration_seconds": wav.seconds,
        "expected_duration_seconds": metadata["duration_seconds"],
        "sample_rate": wav.sample_rate,
        "channels": wav.num_channels,
        "samples": wav.num_samples,
        "peak": peak,
        "rms": rms,
        "non_silent": bool(peak > 0.0001 and rms > 0.00001),
        "frames": len(frames),
        "section_counts": section_counts,
        "slot_ids": slot_ids,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_wav)
    print(report_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulation", type=Path, default=DEFAULT_MODULATION)
    parser.add_argument("--model", default="mrt2_small")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "magenta_longform_polyrhythm_180s.wav")
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT_DIR / "magenta_longform_polyrhythm_180s.report.json")
    args = parser.parse_args()

    modulation = load_modulation(args.modulation)
    render(modulation, args.modulation, args.model, args.output, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
