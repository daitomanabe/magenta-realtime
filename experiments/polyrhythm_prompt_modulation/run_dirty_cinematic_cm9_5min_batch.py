#!/usr/bin/env python3
"""Render dirty cinematic Cm9 meter+macro prompt-mix takes with prompt variants."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
from pathlib import Path

import run_sustained_synth_sources as sustained

try:
    import numpy as np
except ImportError:  # pragma: no cover - slow fallback for minimal Python envs.
    np = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT / "build/examples/midi_prompt_diagnostic/mrt2_midi_prompt_diagnostic"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/polyrhythm_prompt_modulation/dirty_cinematic_cm9_5min_prompt_variants_10"
BPM = 120
BEATS_PER_BAR = 4
DURATION_SECONDS = 300.0
MACRO_REFERENCE_SECONDS = 128.0
MIDI_REFRESH_SECONDS = 2.75
SAMPLE_RATE = 48000
CHANNELS = 2
BYTES_PER_SAMPLE = 4
WEIGHT_FRAME_RATE = 60
CHORD = [36, 43, 48, 51, 55, 58, 62, 67]
PROMPT_LIBRARY_HEADER = [
    "id",
    "label",
    "category",
    "prompt",
    "guide_kind",
    "temperature",
    "top_k",
    "cfg_musiccoca",
    "cfg_notes",
    "cfg_drums",
]

TAKE_VARIANTS = [
    {
        "scene": "Obsidian Fog Vault",
        "slug": "obsidian_fog",
        "drone": "low detuned VCO fog with unstable transformer hum",
        "noise": "black room tone, soot-like broadband noise, and sub pressure",
        "glitch": "frozen cassette buffer dust with broken resampling edges",
        "pad": "smoked glass wavetable haze with dim metallic overtones",
        "filter": "slow ladder low-pass shadowing and shallow notch drift",
        "space": "long concrete chamber reverb with soft tape compression",
        "motion": "barely moving non-rhythmic phase drift",
    },
    {
        "scene": "Mercury Ice Corridor",
        "slug": "mercury_ice",
        "drone": "cold sine-stack drone with brushed modular detune",
        "noise": "silver filtered air pressure and distant mechanical room hiss",
        "glitch": "icy spectral freeze grains suspended without stutters",
        "pad": "pale wavetable choir pad with frost-like harmonic smear",
        "filter": "slow high-pass air movement and muted comb-color shifts",
        "space": "wide frozen hall reverb with restrained shimmer",
        "motion": "slow non-periodic spectral thawing",
    },
    {
        "scene": "Carbon Cinema Tunnel",
        "slug": "carbon_tunnel",
        "drone": "heavy Cm9 oscillator floor with dirty subharmonic bloom",
        "noise": "charcoal noise bed, low shelf pressure, and smoky midrange",
        "glitch": "dark buffer-freeze sheet with tiny digital ash",
        "pad": "cinematic glass pad buried under carbon tape haze",
        "filter": "broad band-pass sweeps that never form a pulse",
        "space": "deep tunnel convolution tail and saturated low reflections",
        "motion": "slow pressure changes across the stereo field",
    },
    {
        "scene": "Rust Neon Observatory",
        "slug": "rust_neon",
        "drone": "rusted modular chord drone with dim neon harmonic bleed",
        "noise": "oxide noise floor, distant transformer buzz, and dark air",
        "glitch": "corroded granular hold with non-rhythmic bit erosion",
        "pad": "neon spectral pad softened by magnetic tape wear",
        "filter": "slow resonant low-pass opening with off-grid drift",
        "space": "large observatory reverb with blurred reverse reflections",
        "motion": "uneven analog drift without tremolo",
    },
    {
        "scene": "Granite Storm Interior",
        "slug": "granite_storm",
        "drone": "granite-heavy VCO sustain with pressure-wave detune",
        "noise": "storm-static broadband bed and low cinematic rumble",
        "glitch": "frozen storm buffer texture with soft cracked edges",
        "pad": "gray additive pad with mineral shimmer and dust",
        "filter": "slow low-mid filtering and wide non-rhythmic phasing",
        "space": "huge stone room tail with compressed dark reflections",
        "motion": "very slow density drift like suspended weather",
    },
    {
        "scene": "Oil Glass Atrium",
        "slug": "oil_glass",
        "drone": "oil-thick analog drone with smeared pitch drift",
        "noise": "viscous dark noise bed and wet sub pressure",
        "glitch": "liquid buffer smear with tiny non-metric digital bubbles",
        "pad": "black glass pad with oily upper-harmonic bands",
        "filter": "slow formant-like color drift and muted notch movement",
        "space": "glossy atrium reverb with soft overdrive",
        "motion": "fluid non-rhythmic stereo bending",
    },
    {
        "scene": "Ashen Data Cathedral",
        "slug": "ashen_data",
        "drone": "cathedral-size Cm9 drone with burnt digital undertones",
        "noise": "ashen data hiss, dark fan noise, and sub resonance",
        "glitch": "static frozen buffer choir with decomposed sample edges",
        "pad": "wide spectral organ pad with dusty harmonic bloom",
        "filter": "slow resonant notch migration and gentle spectral blur",
        "space": "cathedral tail, tape haze, and distant digital smear",
        "motion": "long-form non-periodic harmonic breathing",
    },
    {
        "scene": "Chrome Dust Horizon",
        "slug": "chrome_dust",
        "drone": "chrome-coated modular drone with unstable sine beating",
        "noise": "bright dust noise filtered into a dark cinematic bed",
        "glitch": "metallic freeze dust with tiny broken codec artifacts",
        "pad": "cold glass pad with chrome shimmer and dirty chorus",
        "filter": "slow high-mid damping and low-pass shade movement",
        "space": "wide horizon reverb with blurred metallic reflections",
        "motion": "slow stereo widening without rhythmic movement",
    },
    {
        "scene": "Basalt Memory Room",
        "slug": "basalt_memory",
        "drone": "basalt-dark polysynth drone with worn tape oscillation",
        "noise": "old memory-room hiss, sub floor, and filtered dust",
        "glitch": "aged frozen buffer sheet with soft memory corruption",
        "pad": "dim spectral pad with cracked glass harmonics",
        "filter": "slow low-pass settling and non-metric phase smear",
        "space": "small-to-vast room morph with saturated tail",
        "motion": "gradual texture aging across the full take",
    },
    {
        "scene": "Violet Machine Weather",
        "slug": "violet_machine",
        "drone": "violet machine drone with dirty modular chord pressure",
        "noise": "electrical weather noise and cinematic low cloud",
        "glitch": "suspended machine-glitch sheet with no rhythmic cuts",
        "pad": "violet wavetable pad with diffuse glassy overtones",
        "filter": "slow asymmetric filter color and post-effect diffusion",
        "space": "dark plate reverb, tape saturation, and soft chorus",
        "motion": "evolving non-rhythmic modulation from start to finish",
    },
]

TAKE_DETAIL_VARIANTS = [
    {
        "emphasis": "burnt transformer grain and a matte black low end",
        "motion": "uneven half-speed phase sag with no repeating cycle",
        "post": "soft diode clipping, dull tape print-through, and a dry-wet reverb swell",
        "drone_edge": "a scorched transformer edge",
        "noise_color": "matte-black dust and filtered electrical residue",
        "glitch_behavior": "slowly oxidized codec flakes suspended inside the sustain",
        "pad_color": "smudged charcoal harmonics",
    },
    {
        "emphasis": "cold silver air, pressure drift, and distant machine skin",
        "motion": "very slow spectral thawing that drifts off-grid",
        "post": "frozen hall blur, restrained shimmer, and gentle high-cut damping",
        "drone_edge": "brushed silver detune and soft high-frequency frost",
        "noise_color": "silver pressure hiss with no tick transients",
        "glitch_behavior": "ice-like spectral grains stretched into a continuous sheet",
        "pad_color": "pale glass harmonics with a cold chorus halo",
    },
    {
        "emphasis": "charcoal sub pressure and heavy tunnel resonance",
        "motion": "slow stereo pressure shifts that never lock to tempo",
        "post": "dark convolution reflections, tape crush, and low-mid saturation",
        "drone_edge": "charred subharmonic bloom",
        "noise_color": "smoky charcoal air and compressed tunnel rumble",
        "glitch_behavior": "digital ash suspended without cuts or stutters",
        "pad_color": "buried carbon-glass upper partials",
    },
    {
        "emphasis": "rusted neon harmonic bleed and corroded voltage",
        "motion": "unstable analog drift with no tremolo or clocked pulse",
        "post": "blurred reverse reflections, oxide saturation, and dark plate reverb",
        "drone_edge": "rust-neon sidebands and worn oscillator smear",
        "noise_color": "oxide floor noise and transformer buzz softened into air",
        "glitch_behavior": "bit erosion that behaves like continuous corrosion",
        "pad_color": "neon spectral haze dulled by magnetic wear",
    },
    {
        "emphasis": "mineral density, storm-static, and granite weight",
        "motion": "weather-like density drift over the whole take",
        "post": "stone-room compression, muted phasing, and distant sub bloom",
        "drone_edge": "granite pressure-wave detune",
        "noise_color": "storm-static bed with low cinematic rumble",
        "glitch_behavior": "soft cracked storm buffer edges with no rhythmic flicker",
        "pad_color": "gray additive dust and mineral shimmer",
    },
    {
        "emphasis": "viscous oil-glass movement and wet dark pressure",
        "motion": "fluid stereo bending that avoids periodic wobble",
        "post": "glossy atrium reflections, muted formant drift, and soft overdrive",
        "drone_edge": "oil-thick pitch smear",
        "noise_color": "viscous dark noise and wet sub pressure",
        "glitch_behavior": "liquid buffer smear without bubbles becoming pulses",
        "pad_color": "black glass harmonic bands with oily diffusion",
    },
    {
        "emphasis": "burnt data ash, organ-like width, and cathedral depth",
        "motion": "long harmonic breathing with uneven phrase lengths",
        "post": "cathedral tail, spectral blur, and dusty digital smear",
        "drone_edge": "burnt digital undertones below the Cm9 body",
        "noise_color": "ashen fan hiss and dark server-room air",
        "glitch_behavior": "static buffer choir decomposed into one continuous tone",
        "pad_color": "dusty organ bloom and wide spectral haze",
    },
    {
        "emphasis": "chrome dust, metallic horizon, and cold stereo width",
        "motion": "slow widening and damping with no rhythmic LFO",
        "post": "blurred metallic reflections, dirty chorus, and high-mid shade",
        "drone_edge": "chrome-coated sine beating flattened into sustain",
        "noise_color": "bright dust filtered down into a dark bed",
        "glitch_behavior": "broken codec artifacts stretched until they stop clicking",
        "pad_color": "cold chrome shimmer with glassy dirt",
    },
    {
        "emphasis": "old memory-room hiss and worn tape oscillation",
        "motion": "gradual texture aging across the full render",
        "post": "small-to-vast room morph, saturated tail, and low-pass settling",
        "drone_edge": "basalt-dark tape wobble kept below pulse speed",
        "noise_color": "filtered dust, old room hiss, and a steady sub floor",
        "glitch_behavior": "soft memory corruption blurred into a frozen layer",
        "pad_color": "cracked glass harmonics under dim spectral dust",
    },
    {
        "emphasis": "violet electrical weather and diffuse machine cloud",
        "motion": "asymmetric filter color changes that do not repeat",
        "post": "dark plate reverb, soft chorus, and post-effect diffusion",
        "drone_edge": "dirty modular chord pressure with violet overtones",
        "noise_color": "electrical weather noise softened into low cloud",
        "glitch_behavior": "machine-glitch residue suspended without rhythmic cuts",
        "pad_color": "diffuse violet glass overtones",
    },
    {
        "emphasis": "magnetic brownout, dim filament hum, and slow voltage sinking",
        "motion": "one-way tonal darkening with tiny non-periodic returns",
        "post": "brown tape saturation, low shelf compression, and dull spring diffusion",
        "drone_edge": "filament hum folded into the low Cm9 sustain",
        "noise_color": "brownout static and old amplifier air",
        "glitch_behavior": "melted sample-hold dust with no clock feel",
        "pad_color": "dim amber partials buried under tape haze",
    },
    {
        "emphasis": "blue-black vacuum pressure and distant frozen radio air",
        "motion": "subtle spectral pressure drift with irregular timing",
        "post": "vacuum-like reverb, high-cut spectral smear, and soft limiter glue",
        "drone_edge": "blue-black oscillator pressure",
        "noise_color": "distant radio air flattened into a stable bed",
        "glitch_behavior": "frozen radio fragments stretched into texture only",
        "pad_color": "blue glass harmonics with a muted upper edge",
    },
    {
        "emphasis": "molten ceramic resonance and low metallic heat",
        "motion": "slow heat-bend detune that never becomes vibrato",
        "post": "ceramic room reflections, wavefolder warmth, and dark notch drift",
        "drone_edge": "molten low-register resonance",
        "noise_color": "heated ceramic hiss and low metallic air",
        "glitch_behavior": "soft thermal bit smear with no stutter pattern",
        "pad_color": "warm mineral harmonics with a glass tail",
    },
    {
        "emphasis": "deep sea cable pressure and submerged digital sediment",
        "motion": "slow underwater filter buoyancy with no pulse",
        "post": "submerged convolution tail, damped chorus, and low-pass sediment",
        "drone_edge": "pressure-bent cable hum",
        "noise_color": "submerged broadband sediment and distant cable hiss",
        "glitch_behavior": "waterlogged buffer dust without rhythmic droplets",
        "pad_color": "dark aquatic harmonics and soft detune haze",
    },
    {
        "emphasis": "white-hot CRT bloom and blackened phosphor smear",
        "motion": "uneven phosphor afterglow drifting slower than breath",
        "post": "CRT saturation, soft scanline blur without rhythm, and smoky hall reverb",
        "drone_edge": "phosphor-bloom sidebands around the Cm9 root",
        "noise_color": "blackened CRT hiss flattened into static air",
        "glitch_behavior": "visual-sounding phosphor flakes held as a continuous sheet",
        "pad_color": "white-hot glass harmonics dimmed by smoke",
    },
    {
        "emphasis": "iron fog, buried machinery, and distant low resonance",
        "motion": "heavy non-metric resonance shifts across a wide stereo field",
        "post": "iron chamber reflections, low-mid tape glue, and muted comb shadows",
        "drone_edge": "buried machine resonance under the Cm9 chord",
        "noise_color": "iron fog and low mechanical room tone",
        "glitch_behavior": "submerged machine dust without audible cuts",
        "pad_color": "dull steel harmonics behind a fogged glass layer",
    },
    {
        "emphasis": "polar night shimmer and granular snow suspended in the chord",
        "motion": "non-repeating aurora-like detune drift",
        "post": "wide polar reverb, muted shimmer, and gentle spectral freeze",
        "drone_edge": "cold aurora detune under a stable low register",
        "noise_color": "fine snow noise with no grains popping forward",
        "glitch_behavior": "granular snow stretched until it becomes a smooth cloud",
        "pad_color": "polar glass shimmer with softened edges",
    },
    {
        "emphasis": "black resin, slow pressure bloom, and sticky harmonic smear",
        "motion": "very slow resin-like stretching without cyclic movement",
        "post": "sticky saturation, dark reverb bloom, and low-pass resin damping",
        "drone_edge": "black resin harmonic drag",
        "noise_color": "sticky dark air and soft pressure bloom",
        "glitch_behavior": "resin-trapped digital dust held inside the sustain",
        "pad_color": "smeared resin-glass harmonics",
    },
    {
        "emphasis": "distant orbital debris, cold telemetry haze, and sub vacuum",
        "motion": "slow orbital drift with irregular phase offsets",
        "post": "large vacuum tail, telemetry blur, and restrained stereo expansion",
        "drone_edge": "cold orbital sine pressure",
        "noise_color": "telemetry haze filtered into a dark continuous floor",
        "glitch_behavior": "debris-like digital flecks stretched away from rhythm",
        "pad_color": "cold satellite glass with diffuse high air",
    },
    {
        "emphasis": "smoke-filled glasshouse resonance and late-night voltage dust",
        "motion": "wandering filter color that never returns on the barline",
        "post": "smoke-room reverb, soft tape compression, and blurred chorus width",
        "drone_edge": "late-night voltage dust around the oscillator body",
        "noise_color": "smoky glasshouse air and distant electrical haze",
        "glitch_behavior": "tiny suspended voltage crumbs without click rhythm",
        "pad_color": "smoke-stained glass harmonics",
    },
]

MODULATION_VARIANTS = [
    {"label": "slow wide macro, shallow meter", "macro_reference_seconds": 176.0, "meter_speed_scale": 0.750, "meter_depth": 0.82, "macro_depth": 1.22, "macro_mix": 1.12, "phase_offset": 0.000},
    {"label": "medium macro, deep meter contrast", "macro_reference_seconds": 144.0, "meter_speed_scale": 1.000, "meter_depth": 1.28, "macro_depth": 0.92, "macro_mix": 0.74, "phase_offset": 0.071},
    {"label": "slow triplet-feel meter drift", "macro_reference_seconds": 160.0, "meter_speed_scale": 0.667, "meter_depth": 1.05, "macro_depth": 1.08, "macro_mix": 1.00, "phase_offset": 0.143},
    {"label": "fast macro pressure, soft meter", "macro_reference_seconds": 104.0, "meter_speed_scale": 0.875, "meter_depth": 0.76, "macro_depth": 1.34, "macro_mix": 1.24, "phase_offset": 0.211},
    {"label": "one-and-quarter meter speed, balanced depth", "macro_reference_seconds": 128.0, "meter_speed_scale": 1.250, "meter_depth": 1.00, "macro_depth": 1.00, "macro_mix": 0.92, "phase_offset": 0.287},
    {"label": "half-speed meter, deep macro wash", "macro_reference_seconds": 192.0, "meter_speed_scale": 0.500, "meter_depth": 0.90, "macro_depth": 1.42, "macro_mix": 1.28, "phase_offset": 0.337},
    {"label": "four-thirds meter polyrhythm, light macro", "macro_reference_seconds": 152.0, "meter_speed_scale": 1.333, "meter_depth": 1.18, "macro_depth": 0.84, "macro_mix": 0.62, "phase_offset": 0.409},
    {"label": "three-halves meter speed, strong macro blend", "macro_reference_seconds": 120.0, "meter_speed_scale": 1.500, "meter_depth": 0.96, "macro_depth": 1.26, "macro_mix": 1.18, "phase_offset": 0.463},
    {"label": "long macro arc, deep meter valley", "macro_reference_seconds": 208.0, "meter_speed_scale": 1.125, "meter_depth": 1.36, "macro_depth": 0.90, "macro_mix": 0.82, "phase_offset": 0.527},
    {"label": "short macro rotation, narrow meter", "macro_reference_seconds": 96.0, "meter_speed_scale": 0.800, "meter_depth": 0.72, "macro_depth": 1.10, "macro_mix": 1.04, "phase_offset": 0.601},
    {"label": "eleven-eighth meter feel, deep dual modulation", "macro_reference_seconds": 136.0, "meter_speed_scale": 1.375, "meter_depth": 1.24, "macro_depth": 1.24, "macro_mix": 1.10, "phase_offset": 0.019},
    {"label": "five-sixths meter drag, shallow macro", "macro_reference_seconds": 184.0, "meter_speed_scale": 0.833, "meter_depth": 1.10, "macro_depth": 0.78, "macro_mix": 0.58, "phase_offset": 0.109},
    {"label": "double-time prompt shimmer, restrained depth", "macro_reference_seconds": 148.0, "meter_speed_scale": 2.000, "meter_depth": 0.70, "macro_depth": 0.86, "macro_mix": 0.72, "phase_offset": 0.173},
    {"label": "seven-four style meter skew, heavy macro", "macro_reference_seconds": 112.0, "meter_speed_scale": 1.750, "meter_depth": 1.06, "macro_depth": 1.38, "macro_mix": 1.30, "phase_offset": 0.259},
    {"label": "slow 5-over-4 feeling, broad macro breath", "macro_reference_seconds": 168.0, "meter_speed_scale": 0.625, "meter_depth": 0.88, "macro_depth": 1.18, "macro_mix": 0.96, "phase_offset": 0.319},
    {"label": "near-grid meter with high macro depth", "macro_reference_seconds": 116.0, "meter_speed_scale": 1.062, "meter_depth": 0.94, "macro_depth": 1.46, "macro_mix": 1.34, "phase_offset": 0.389},
    {"label": "slow meter, low macro mix", "macro_reference_seconds": 200.0, "meter_speed_scale": 0.700, "meter_depth": 1.30, "macro_depth": 0.82, "macro_mix": 0.54, "phase_offset": 0.449},
    {"label": "fast meter, medium macro turn", "macro_reference_seconds": 132.0, "meter_speed_scale": 1.625, "meter_depth": 1.12, "macro_depth": 1.06, "macro_mix": 0.88, "phase_offset": 0.557},
    {"label": "wide shallow breathing, off-grid phase", "macro_reference_seconds": 156.0, "meter_speed_scale": 0.917, "meter_depth": 0.78, "macro_depth": 0.94, "macro_mix": 0.66, "phase_offset": 0.643},
    {"label": "deep final macro sweep, dense meter", "macro_reference_seconds": 108.0, "meter_speed_scale": 1.200, "meter_depth": 1.40, "macro_depth": 1.30, "macro_mix": 1.22, "phase_offset": 0.707},
]


def vlq(value: int) -> bytes:
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= ((value & 0x7F) | 0x80)
        value >>= 7
    out = bytearray()
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def write_cm9_hold_midi(path: Path, duration_seconds: float) -> None:
    ppq = 480
    tempo_us = round(60_000_000 / BPM)
    beats = int(round(duration_seconds * BPM / 60.0))
    end_ticks = beats * ppq
    track = bytearray()

    def add(delta: int, event: bytes) -> None:
        track.extend(vlq(delta))
        track.extend(event)

    name = f"Dirty Cinematic Cm9 {duration_seconds:.0f}s BPM120 hold".encode("ascii")
    add(0, b"\xff\x03" + vlq(len(name)) + name)
    add(0, b"\xff\x51\x03" + tempo_us.to_bytes(3, "big"))
    add(0, b"\xff\x58\x04" + bytes([4, 2, 24, 8]))
    for note in CHORD:
        add(0, bytes([0x90, note, 84]))
    for index, note in enumerate(CHORD):
        add(end_ticks if index == 0 else 0, bytes([0x80, note, 0]))
    add(0, b"\xff\x2f\x00")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"MThd")
        f.write((6).to_bytes(4, "big"))
        f.write((0).to_bytes(2, "big"))
        f.write((1).to_bytes(2, "big"))
        f.write(ppq.to_bytes(2, "big"))
        f.write(b"MTrk")
        f.write(len(track).to_bytes(4, "big"))
        f.write(track)


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(f"Render failed with exit code {proc.returncode}: {log_path}\n{tail}")


def clean_tsv(value: str) -> str:
    return " ".join(value.replace("\t", " ").replace("\n", " ").split())


def prompt_slots_for_take(take_index: int) -> list[dict[str, str | float | int]]:
    variant = TAKE_VARIANTS[(take_index - 1) % len(TAKE_VARIANTS)]
    detail = TAKE_DETAIL_VARIANTS[(take_index - 1) % len(TAKE_DETAIL_VARIANTS)]
    scene = variant["scene"]
    slug = variant["slug"]
    common = (
        "Keep one continuous Cm9 sustained environment for the full requested duration at 120 BPM. "
        "The material is dirty, cinematic, ambient, textural, and beatless. "
        "Let post effects and filters evolve slowly, but do not create a rhythm. "
        f"For this take, emphasize {detail['emphasis']}, {detail['motion']}, and {detail['post']}. "
        "Avoid drums, percussion, kick, snare, hats, impacts, risers, arpeggios, "
        "sequencer patterns, rhythmic pulses, tremolo, gated motion, delay taps, "
        "vocals, lead melody, drops, fade-outs, sudden cuts, and silence."
    )
    return [
        {
            "id": f"take{take_index:03d}_{slug}_voltage_drone",
            "label": f"{scene} Voltage Drone",
            "category": "analog_drone",
            "prompt": (
                f"Create an original sustained synthesizer texture called {scene} Voltage Drone. "
                f"A continuous Cm9 low-register modular synth sustain made from {variant['drone']} "
                f"with {detail['drone_edge']}. "
                f"Use {variant['filter']}, {variant['motion']}, and a stable low end. "
                f"Add {variant['space']}. {common}"
            ),
            "guide_kind": "analog_drone",
            "temperature": 0.58,
            "top_k": 42,
            "cfg_musiccoca": 5.8,
            "cfg_notes": 5.7,
            "cfg_drums": 0.0,
        },
        {
            "id": f"take{take_index:03d}_{slug}_cinema_floor",
            "label": f"{scene} Cinema Floor",
            "category": "cinema_noise",
            "prompt": (
                f"Create an original sustained synthesizer texture called {scene} Cinema Floor. "
                f"A continuous Cm9 cinematic noise bed built from {variant['noise']} "
                f"with {detail['noise_color']}. "
                f"Shape it with {variant['filter']} and {variant['motion']}. "
                f"Add {variant['space']} while keeping the sound heavy and wide. {common}"
            ),
            "guide_kind": "cinema_noise",
            "temperature": 0.64,
            "top_k": 58,
            "cfg_musiccoca": 6.1,
            "cfg_notes": 5.8,
            "cfg_drums": 0.0,
        },
        {
            "id": f"take{take_index:03d}_{slug}_glitch_sheet",
            "label": f"{scene} Glitch Sheet",
            "category": "granular_glitch",
            "prompt": (
                f"Create an original sustained synthesizer texture called {scene} Glitch Sheet. "
                f"A continuous Cm9 frozen-buffer synth sheet made from {variant['glitch']}. "
                f"The glitches are texture only: {detail['glitch_behavior']}; "
                "no stutters, no clicks as rhythm, and no beat grid. "
                f"Use {variant['filter']}, {variant['motion']}, and {variant['space']}. {common}"
            ),
            "guide_kind": "granular_glitch",
            "temperature": 0.72,
            "top_k": 76,
            "cfg_musiccoca": 6.4,
            "cfg_notes": 5.9,
            "cfg_drums": 0.0,
        },
        {
            "id": f"take{take_index:03d}_{slug}_glass_texture",
            "label": f"{scene} Glass Texture",
            "category": "spectral_pad",
            "prompt": (
                f"Create an original sustained synthesizer texture called {scene} Glass Texture. "
                f"A continuous Cm9 spectral pad made from {variant['pad']} "
                f"with {detail['pad_color']}. "
                f"Use {variant['filter']}, {variant['motion']}, gentle chorus, and slow post-effect diffusion. "
                f"Add {variant['space']} while keeping the tone sustained and non-melodic. {common}"
            ),
            "guide_kind": "spectral_pad",
            "temperature": 0.60,
            "top_k": 52,
            "cfg_musiccoca": 5.9,
            "cfg_notes": 5.8,
            "cfg_drums": 0.0,
        },
    ]


def write_prompt_library(path: Path, slots: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(PROMPT_LIBRARY_HEADER)]
    for slot in slots:
        lines.append(
            "\t".join([
                clean_tsv(str(slot["id"])),
                clean_tsv(str(slot["label"])),
                clean_tsv(str(slot["category"])),
                clean_tsv(str(slot["prompt"])),
                clean_tsv(str(slot["guide_kind"])),
                f"{float(slot['temperature']):.3f}",
                str(int(slot["top_k"])),
                f"{float(slot['cfg_musiccoca']):.3f}",
                f"{float(slot['cfg_notes']):.3f}",
                f"{float(slot['cfg_drums']):.3f}",
            ])
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prompt_manifest(slots: list[dict[str, str | float | int]]) -> list[dict[str, str]]:
    return [
        {
            "id": str(slot["id"]),
            "label": str(slot["label"]),
            "category": str(slot["category"]),
            "guide_kind": str(slot["guide_kind"]),
            "prompt": str(slot["prompt"]),
        }
        for slot in slots
    ]


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


def analyze_float_wav_stream(path: Path, window_seconds: float = 1.0) -> dict:
    fmt, data_offset, data_size = find_float_wav_data(path)
    channels = fmt["channels"]
    sample_rate = fmt["sample_rate"]
    bytes_per_frame = channels * 4
    total_frames = data_size // bytes_per_frame
    window_frames = max(1, int(round(sample_rate * window_seconds)))
    windows = []

    if np is not None:
        samples = np.memmap(
            path,
            dtype="<f4",
            mode="r",
            offset=data_offset,
            shape=(total_frames, channels),
        )
        frame_start = 0
        while frame_start < total_frames:
            frames_to_read = min(window_frames, total_frames - frame_start)
            mono = samples[frame_start:frame_start + frames_to_read].mean(axis=1)
            rms = float(np.sqrt(np.mean(np.square(mono)))) if frames_to_read else 0.0
            peak = float(np.max(np.abs(mono))) if frames_to_read else 0.0
            windows.append({
                "start_seconds": frame_start / sample_rate,
                "end_seconds": (frame_start + frames_to_read) / sample_rate,
                "rms": rms,
                "peak": peak,
            })
            frame_start += frames_to_read
        del samples
    else:
        with path.open("rb") as f:
            f.seek(data_offset)
            frame_start = 0
            while frame_start < total_frames:
                frames_to_read = min(window_frames, total_frames - frame_start)
                chunk = f.read(frames_to_read * bytes_per_frame)
                values = struct.unpack("<" + "f" * (len(chunk) // 4), chunk)
                sum_sq = 0.0
                peak = 0.0
                count = 0
                for frame in range(frames_to_read):
                    base = frame * channels
                    mono = sum(values[base + channel] for channel in range(channels)) / channels
                    sum_sq += mono * mono
                    peak = max(peak, abs(mono))
                    count += 1
                rms = math.sqrt(sum_sq / count) if count else 0.0
                windows.append({
                    "start_seconds": frame_start / sample_rate,
                    "end_seconds": (frame_start + frames_to_read) / sample_rate,
                    "rms": rms,
                    "peak": peak,
                })
                frame_start += frames_to_read

    post_start_windows = [window for window in windows if window["start_seconds"] >= 3.0]
    min_post_start_rms = min((window["rms"] for window in post_start_windows), default=0.0)
    first_3s_windows = [window for window in windows if window["start_seconds"] < 3.0]
    last_8s_start = max(0.0, (total_frames / sample_rate) - 8.0)
    last_8s_windows = [window for window in windows if window["start_seconds"] >= last_8s_start]
    first_3s_rms = (
        sum(window["rms"] for window in first_3s_windows) / len(first_3s_windows)
        if first_3s_windows else 0.0
    )
    last_8s_median_rms = (
        sorted(window["rms"] for window in last_8s_windows)[len(last_8s_windows) // 2]
        if last_8s_windows else 0.0
    )
    quiet_windows = [
        window for window in post_start_windows
        if window["rms"] < 0.0001 or window["peak"] < 0.001
    ]
    return {
        "schema": "mrt-audio-activity-check-v1",
        "path": str(path.relative_to(ROOT)),
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": total_frames / sample_rate,
        "window_seconds": window_seconds,
        "post_start_seconds": 3.0,
        "rms_threshold": 0.0001,
        "peak_threshold": 0.001,
        "first_3s_rms": first_3s_rms,
        "last_8s_median_rms": last_8s_median_rms,
        "min_to_first_3s_rms_ratio": (
            min_post_start_rms / first_3s_rms if first_3s_rms > 0.0 else 0.0
        ),
        "last_8s_to_first_3s_rms_ratio": (
            last_8s_median_rms / first_3s_rms if first_3s_rms > 0.0 else 0.0
        ),
        "min_post_start_rms": min_post_start_rms,
        "quiet_window_count": len(quiet_windows),
        "quiet_windows": quiet_windows[:20],
        "windows": windows,
        "continuous_after_post_start": len(quiet_windows) == 0,
    }


def control_summary(frames: list[dict]) -> dict:
    keys = ["cfg_musiccoca", "cfg_notes", "temperature", "top_k"]
    summary = {}
    for key in keys:
        values = [frame[key] for frame in frames]
        summary[key] = {
            "min": min(values),
            "max": max(values),
            "first": values[0],
            "last": values[-1],
        }
    slot_values = [[frame["weights"][slot] for frame in frames] for slot in range(4)]
    summary["prompt_weights"] = [
        {
            "slot": slot,
            "min": min(values),
            "max": max(values),
            "span": max(values) - min(values),
        }
        for slot, values in enumerate(slot_values)
    ]
    summary["weight_sum_max_error"] = max(
        abs(sum(frame["weights"]) - 1.0) for frame in frames
    )
    return summary


def macro_phase_offset_for_take(take_index: int) -> float:
    return ((take_index - 1) * 0.13750352374993502) % 1.0


def modulation_for_take(take_index: int) -> dict[str, float | str]:
    return MODULATION_VARIANTS[(take_index - 1) % len(MODULATION_VARIANTS)]


def duration_stem_label(duration_seconds: float) -> str:
    minutes = duration_seconds / 60.0
    if abs(minutes - round(minutes)) < 1e-6:
        return f"{int(round(minutes))}min"
    return f"{int(round(duration_seconds))}s"


def paths_for_take(output_dir: Path, take_index: int, duration_seconds: float) -> dict[str, Path]:
    take_dir = output_dir / f"take_{take_index:03d}"
    stem = (
        f"dirty_cinematic_cm9_take_{take_index:03d}_meter_macro_sine_"
        f"{duration_stem_label(duration_seconds)}"
    )
    return {
        "dir": take_dir,
        "prompts": take_dir / f"{stem}.prompts.tsv",
        "wav": take_dir / f"{stem}.wav",
        "report": take_dir / f"{stem}.report.json",
        "weights": take_dir / f"{stem}.weights.json",
        "audio_check": take_dir / f"{stem}.audio_check.json",
        "log": take_dir / f"{stem}.render.log",
    }


def validate_take(paths: dict[str, Path], duration_seconds: float) -> dict:
    wav = paths["wav"]
    report = paths["report"]
    weights = paths["weights"]
    audio_check_path = paths["audio_check"]
    probe = sustained.ffprobe(wav)
    report_data = json.loads(report.read_text(encoding="utf-8"))
    weights_data = json.loads(weights.read_text(encoding="utf-8"))
    audio_check = analyze_float_wav_stream(wav.resolve())
    audio_check_path.write_text(json.dumps(audio_check, indent=2) + "\n", encoding="utf-8")

    duration = float(probe["format"]["duration"])
    if abs(duration - duration_seconds) > 0.05:
        raise RuntimeError(f"Duration mismatch: {duration}")
    if report_data.get("weight_mode") != "meter_macro_sine":
        raise RuntimeError(f"Unexpected weight_mode: {report_data.get('weight_mode')}")
    if weights_data.get("frame_rate") != WEIGHT_FRAME_RATE:
        raise RuntimeError(f"Unexpected weight frame rate: {weights_data.get('frame_rate')}")
    expected_weight_frames = int(round(duration_seconds * WEIGHT_FRAME_RATE))
    frames = weights_data.get("frames", [])
    if len(frames) != expected_weight_frames:
        raise RuntimeError(f"Weight frame count mismatch: {len(frames)}")
    if "meter_sine" not in weights_data or "macro_sine" not in weights_data:
        raise RuntimeError("Weight JSON is missing meter_sine or macro_sine metadata")
    if not report_data.get("non_silent", False):
        raise RuntimeError("Rendered WAV is silent")
    if not audio_check["continuous_after_post_start"]:
        raise RuntimeError(f"Quiet windows found: {audio_check['quiet_windows'][:5]}")
    low_rms_windows = [
        window for window in audio_check["windows"]
        if window["start_seconds"] >= 3.0 and window["rms"] < 0.003
    ]
    if low_rms_windows:
        raise RuntimeError(f"Collapsed low-RMS windows found: {low_rms_windows[:5]}")

    controls = control_summary(frames)
    for slot in controls["prompt_weights"]:
        if slot["span"] < 0.10:
            raise RuntimeError(f"Prompt slot {slot['slot']} modulation span is too small: {slot}")

    return {
        "duration_seconds": duration,
        "prompt_library": str(paths["prompts"].relative_to(ROOT)),
        "wav": str(wav.relative_to(ROOT)),
        "report": str(report.relative_to(ROOT)),
        "weights": str(weights.relative_to(ROOT)),
        "audio_check": str(audio_check_path.relative_to(ROOT)),
        "log": str(paths["log"].relative_to(ROOT)),
        "ffprobe": probe,
        "peak": report_data["peak"],
        "rms": report_data["rms"],
        "non_silent": report_data["non_silent"],
        "audio_activity": {
            "continuous_after_3s": audio_check["continuous_after_post_start"],
            "quiet_window_count": audio_check["quiet_window_count"],
            "min_post_3s_rms": audio_check["min_post_start_rms"],
            "last_8s_to_first_3s_rms_ratio": audio_check["last_8s_to_first_3s_rms_ratio"],
        },
        "meter_sine": weights_data["meter_sine"],
        "macro_sine": weights_data["macro_sine"],
        "controls": controls,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--duration", type=float, default=DURATION_SECONDS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=1)
    args = parser.parse_args()

    if not args.binary.exists():
        raise SystemExit(f"Missing binary: {args.binary}")
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.max_attempts <= 0:
        raise SystemExit("--max-attempts must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    midi = output_dir / f"dirty_cinematic_cm9_{duration_stem_label(args.duration)}_bpm120_hold.mid"
    write_cm9_hold_midi(midi, args.duration)

    free_bytes = shutil.disk_usage(output_dir).free
    estimated_wav_bytes = args.count * args.duration * SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE
    estimated_weight_bytes = args.count * args.duration * WEIGHT_FRAME_RATE * 700
    print(
        "Storage preflight: "
        f"free={free_bytes / (1024 ** 3):.2f} GiB, "
        f"estimated_batch={((estimated_wav_bytes + estimated_weight_bytes) / (1024 ** 3)):.2f} GiB",
        flush=True,
    )

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema": "mrt-dirty-cinematic-cm9-5min-prompt-variant-batch-v1",
        "profile": "dirty_cinematic_ambient_cm9",
        "weight_mode": "meter_macro_sine",
        "control_mode": "ambient_crescendo",
        "bpm": BPM,
        "bars": args.duration * BPM / 60.0 / BEATS_PER_BAR,
        "duration_seconds": args.duration,
        "count_requested": args.count,
        "start_index": args.start_index,
        "macro_reference_seconds": MACRO_REFERENCE_SECONDS,
        "midi": str(midi.relative_to(ROOT)),
        "prompt_variant_mode": "per_take_four_prompt_tsv",
        "prompt_variant_count": len(TAKE_VARIANTS),
        "control_targets": {
            "cfg_musiccoca": [5.8, 7.8],
            "cfg_notes": [5.6, 6.4],
            "temperature": [0.6075, 0.7205],
            "top_k": [51, 103],
            "buffer_chunk": "fixed 25 Hz frames / 1920 sample chunks",
            "midi_refresh_seconds": MIDI_REFRESH_SECONDS,
            "meter_speed_scale": "per_take",
            "meter_depth": "per_take",
            "macro_reference_seconds": "per_take",
            "macro_depth": "per_take",
            "macro_mix": "per_take",
            "window_rms_stabilization": {"enabled": False},
        },
        "mix_modulation": {
            "micro": "BPM120 meter_sine for 4/4, 3/4, 6/8, 4/4 phase offset 0.25, with per-take speed/depth scaling",
            "macro": "per-take macro sine reference length, depth, mix amount, and phase offset",
        },
        "takes": [],
    }

    for take_index in range(args.start_index, args.start_index + args.count):
        paths = paths_for_take(output_dir, take_index, args.duration)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        prompt_slots = prompt_slots_for_take(take_index)
        modulation = modulation_for_take(take_index)
        write_prompt_library(paths["prompts"], prompt_slots)
        print(
            "take "
            f"{take_index:03d}: prompts="
            + ", ".join(str(slot["label"]) for slot in prompt_slots),
            flush=True,
        )
        print(
            "take "
            f"{take_index:03d}: modulation={modulation['label']} "
            f"macro_ref={float(modulation['macro_reference_seconds']):.3f}s "
            f"meter_speed={float(modulation['meter_speed_scale']):.3f} "
            f"meter_depth={float(modulation['meter_depth']):.3f} "
            f"macro_depth={float(modulation['macro_depth']):.3f} "
            f"macro_mix={float(modulation['macro_mix']):.3f}",
            flush=True,
        )

        summary = None
        phase_offset = 0.0
        batch_variant = take_index
        last_error: Exception | None = None
        for attempt in range(1, args.max_attempts + 1):
            batch_variant = take_index + (attempt - 1) * 1000
            phase_offset = (
                macro_phase_offset_for_take(batch_variant) + float(modulation["phase_offset"])
            ) % 1.0
            print(
                f"take {take_index:03d}: attempt={attempt} "
                f"batch_variant={batch_variant} macro_phase_offset={phase_offset:.6f}",
                flush=True,
            )

            existing = all(paths[key].exists() for key in ["prompts", "wav", "report", "weights"])
            if args.force or attempt > 1:
                existing = False
            if not existing:
                if args.skip_render:
                    raise RuntimeError(f"Missing rendered files for take {take_index:03d}")
                run_logged(
                    [
                        str(args.binary),
                        "--midi",
                        str(midi),
                        "--profile",
                        "dirty_cinematic_ambient_cm9",
                        "--duration",
                        f"{args.duration:.3f}",
                        "--weight-mode",
                        "meter_macro_sine",
                        "--control-mode",
                        "ambient_crescendo",
                        "--macro-reference-seconds",
                        f"{float(modulation['macro_reference_seconds']):.3f}",
                        "--macro-phase-offset",
                        f"{phase_offset:.9f}",
                        "--meter-speed-scale",
                        f"{float(modulation['meter_speed_scale']):.6f}",
                        "--meter-depth",
                        f"{float(modulation['meter_depth']):.6f}",
                        "--macro-depth",
                        f"{float(modulation['macro_depth']):.6f}",
                        "--macro-mix",
                        f"{float(modulation['macro_mix']):.6f}",
                        "--batch-variant",
                        str(batch_variant),
                        "--midi-refresh-seconds",
                        f"{MIDI_REFRESH_SECONDS:.3f}",
                        "--prompt-library",
                        str(paths["prompts"]),
                        "--prompt-page-seconds",
                        f"{args.duration + 1.0:.3f}",
                        "--transition",
                        "0.000",
                        "--output",
                        str(paths["wav"]),
                        "--report",
                        str(paths["report"]),
                        "--weights-output",
                        str(paths["weights"]),
                        "--text-prompts",
                    ],
                    paths["log"],
                )
            else:
                print(f"take {take_index:03d}: existing render found, validating", flush=True)

            try:
                summary = validate_take(paths, args.duration)
                break
            except RuntimeError as exc:
                last_error = exc
                print(
                    f"take {take_index:03d}: validation failed on attempt {attempt}: {exc}",
                    flush=True,
                )
                if attempt == args.max_attempts:
                    raise

        if summary is None:
            raise RuntimeError(f"take {take_index:03d}: all attempts failed: {last_error}")
        summary.update({
            "take_index": take_index,
            "macro_phase_offset": phase_offset,
            "batch_variant": batch_variant,
            "attempt": attempt,
            "modulation": modulation,
            "prompt_set": prompt_manifest(prompt_slots),
        })
        manifest["takes"].append(summary)
        manifest["completed_count"] = len(manifest["takes"])
        write_manifest(manifest_path, manifest)
        print(
            f"take {take_index:03d}: ok "
            f"rms={summary['rms']:.6f} "
            f"quiet={summary['audio_activity']['quiet_window_count']}",
            flush=True,
        )

    write_manifest(manifest_path, manifest)
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
