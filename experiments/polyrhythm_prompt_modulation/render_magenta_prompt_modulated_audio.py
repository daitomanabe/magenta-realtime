#!/usr/bin/env python3
"""Render a short Magenta RealTime clip from modulation data."""

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
DEFAULT_MODULATION = EXPERIMENT / "data" / "modulation_10s_104bpm.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "polyrhythm_prompt_modulation"


def load_modulation(path: Path) -> dict[str, Any]:
    if not path.exists():
        from generate_modulation import build_modulation

        plan = json.loads((EXPERIMENT / "prompt_plan.json").read_text(encoding="utf-8"))
        modulation = build_modulation(plan)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(modulation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def render(modulation: dict[str, Any], modulation_path: Path, model: str, output_wav: Path, report_path: Path) -> None:
    metadata = modulation["metadata"]
    slots = modulation["slots"]
    frames = modulation["frames"]

    mrt = MagentaRT2Mlxfn(
        size=model,
        temperature=1.05,
        top_k=96,
        cfg_musiccoca=3.2,
        cfg_notes=0.45,
        cfg_drums=0.8,
    )

    prompt_embeddings = []
    for index, slot in enumerate(slots):
        prompt_embeddings.append(
            mrt.embed_style(slot["magenta_prompt"], use_mapper=True, seed=100 + index)
        )
    prompt_embeddings_np = np.stack(prompt_embeddings, axis=0).astype(np.float32)

    state = None
    audio_frames: list[np.ndarray] = []
    started_at = time.time()

    for frame in frames:
        weights = np.array(frame["weights_array"], dtype=np.float32)
        blended_style = np.sum(prompt_embeddings_np * weights[:, None], axis=0).astype(np.float32)
        style_tokens = mrt.tokenize_style(blended_style).tolist()

        drums = [1 if frame["weights"]["broken_beat"] > 0.29 and (frame["beat_position"] % 0.5) < 0.11 else 0]
        cfg_musiccoca = 1.6 + (3.4 * float(np.max(weights)))
        cfg_drums = 0.2 + (2.5 * frame["weights"]["broken_beat"])
        temperature = 0.82 + (0.38 * frame["weights"]["glassy_arps"]) + (0.22 * frame["weights"]["granular_texture"])
        top_k = int(32 + round(128 * (0.55 * frame["weights"]["glassy_arps"] + 0.45 * frame["weights"]["granular_texture"])))

        args = mrt._build_mlxfn_args(  # pylint: disable=protected-access
            style_tokens=style_tokens,
            drums=drums,
            cfg_musiccoca=cfg_musiccoca,
            cfg_notes=0.45,
            cfg_drums=cfg_drums,
            temperature=temperature,
            top_k=top_k,
        )
        if state is None:
            state = list(mrt._initial_state)  # pylint: disable=protected-access
        outputs = mrt._fn(args + state)  # pylint: disable=protected-access
        mx.eval(outputs)
        audio_frames.append(np.array(outputs[0]))
        state = list(outputs[1:])

    all_audio = np.concatenate(audio_frames, axis=-1)
    samples = all_audio[0].T.astype(np.float32) / 32768.0
    wav = Waveform(samples, sample_rate=48_000)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write(str(output_wav))

    rms = float(np.sqrt(np.mean(np.square(wav.samples))))
    peak = float(np.max(np.abs(wav.samples)))
    report = {
        "schema": "mrt-polyrhythm-render-report-v1",
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
        "elapsed_seconds": round(time.time() - started_at, 3),
        "prompts": [
            {"id": slot["id"], "magenta_prompt": slot["magenta_prompt"]}
            for slot in slots
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_wav)
    print(report_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulation", type=Path, default=DEFAULT_MODULATION)
    parser.add_argument("--model", default="mrt2_small")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "magenta_polyrhythm_prompt_mod_10s.wav")
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT_DIR / "magenta_polyrhythm_prompt_mod_10s.report.json")
    args = parser.parse_args()

    modulation = load_modulation(args.modulation)
    render(modulation, args.modulation, args.model, args.output, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
