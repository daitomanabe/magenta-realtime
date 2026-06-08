#!/usr/bin/env python3
"""Generate a 100-prompt beatless ambient synth library for MRT modulation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments/polyrhythm_prompt_modulation/data"
JSON_PATH = OUT_DIR / "ambient_100_prompt_library.json"
TSV_PATH = OUT_DIR / "ambient_100_prompt_library.tsv"
MD_PATH = OUT_DIR / "ambient_100_prompt_library.md"
PLAN_PATH = OUT_DIR / "ambient_100_prompt_modulation_plan.json"
MIDI_PATH = ROOT / "outputs/polyrhythm_prompt_modulation/ambient_100_prompt_modulation/ambient_100_cm9_hold_100s.mid"
PAGE_SECONDS = 4.0
BPM = 120
DURATION_SECONDS = 100.0


CATEGORIES = [
    {
        "category": "analog_drone",
        "role": "low analog drone",
        "source": "detuned modular VCO stack",
        "guide_kind": "analog_drone",
        "temperature": 0.58,
        "top_k": 36,
        "cfg_musiccoca": 7.4,
        "cfg_notes": 6.5,
        "cfg_drums": 0.0,
        "motions": [
            "slow phase beating",
            "tiny non-rhythmic pitch drift",
            "a dark ladder-filter opening over minutes",
            "subtle oscillator sync instability",
            "barely moving pulse-width drift",
        ],
        "effects": [
            "warm saturation, tape delay smear, and a stable plate reverb tail",
            "soft overdrive, low-pass resonance, and a deep spring reverb",
            "gentle transformer saturation, dark chorus, and wide room reflections",
            "tape compression, slow phaser shadow, and long filtered delay",
            "analog console color, BBD delay blur, and a large reverb floor",
        ],
        "moods": ["heavy fog", "subterranean pressure", "slow voltage weather"],
    },
    {
        "category": "spectral_pad",
        "role": "luminous spectral pad",
        "source": "high-mid wavetable and additive sine cloud",
        "guide_kind": "spectral_pad",
        "temperature": 0.64,
        "top_k": 56,
        "cfg_musiccoca": 7.1,
        "cfg_notes": 6.2,
        "cfg_drums": 0.0,
        "motions": [
            "spectral freeze thawing slowly",
            "phase smear across upper harmonics",
            "formant color drifting without rhythm",
            "small wavetable-index wander",
            "a shimmer layer breathing very slowly",
        ],
        "effects": [
            "shimmer reverb, transparent chorus, and resonant high-pass air",
            "spectral blur, glassy reverb, and a wide stereo halo",
            "soft ensemble, convolution tail, and a smooth tilt filter",
            "subtle granular freeze, long reverb, and slow stereo widening",
            "clean saturation, micro-pitch spread, and a floating late tail",
        ],
        "moods": ["floating glass", "cold aurora", "weightless cinema"],
    },
    {
        "category": "cinema_noise",
        "role": "dark cinematic noise bed",
        "source": "filtered broadband noise and subharmonic pressure",
        "guide_kind": "cinema_noise",
        "temperature": 0.82,
        "top_k": 88,
        "cfg_musiccoca": 7.3,
        "cfg_notes": 3.8,
        "cfg_drums": 0.0,
        "motions": [
            "slow band-pass sweep",
            "distant resonant bloom",
            "low shelf pressure rising without hits",
            "irregular filter turbulence",
            "subtle granular air movement",
        ],
        "effects": [
            "convolution reverb, tape saturation, and a huge distant room",
            "dark notch filtering, soft clipping, and a long metallic tail",
            "low-pass movement, reverse smear, and deep stereo reflections",
            "wide diffusion, gentle compression, and smoky high-frequency dust",
            "resonant EQ, slow phaser depth, and cavernous late reverb",
        ],
        "moods": ["black smoke", "large empty room", "cinematic tension"],
    },
    {
        "category": "granular_glitch",
        "role": "continuous granular glitch sustain",
        "source": "frozen buffer and broken resampling layer",
        "guide_kind": "granular_glitch",
        "temperature": 0.96,
        "top_k": 124,
        "cfg_musiccoca": 6.7,
        "cfg_notes": 3.2,
        "cfg_drums": 0.0,
        "motions": [
            "slow grain-density swelling",
            "bit-depth erosion moving at random",
            "spectral smear opening and closing",
            "tiny buffer slips without a beat grid",
            "unstable stereo fragments orbiting slowly",
        ],
        "effects": [
            "bitcrush edges, reverse delay haze, and moving resonant notches",
            "short freeze delay, spectral blur, and wide unstable reflections",
            "granular hold, soft clipping, and filtered reverb dust",
            "sample-rate erosion, diffused room tone, and subtle pitch scatter",
            "broken resampling, tiny crackles, and a smooth reverb wash",
        ],
        "moods": ["electric dust", "fragile malfunction", "frozen circuitry"],
    },
    {
        "category": "metallic_cloud",
        "role": "metallic harmonic cloud",
        "source": "FM partials, ring-modulated sines, and tuned resonators",
        "guide_kind": "metallic",
        "temperature": 0.74,
        "top_k": 82,
        "cfg_musiccoca": 7.0,
        "cfg_notes": 5.6,
        "cfg_drums": 0.0,
        "motions": [
            "partials sliding by a few cents",
            "ring-modulated sidebands blooming slowly",
            "resonator tuning wandering non-rhythmically",
            "soft inharmonic beating",
            "long metallic overtones folding inward",
        ],
        "effects": [
            "plate reverb, gentle wavefolding, and a smooth comb-filter shadow",
            "tuned resonator feedback, dark chorus, and a long hall tail",
            "soft clipping, ring shimmer, and wide reflective space",
            "spectral tilt, plate diffusion, and a slow stereo rotation",
            "FM haze, filtered delay, and a luminous reverb decay",
        ],
        "moods": ["bronze mist", "chromatic pressure", "quiet industrial light"],
    },
    {
        "category": "modular_feedback",
        "role": "self-balancing modular feedback patch",
        "source": "resonant filter feedback network and wavefolder",
        "guide_kind": "analog_drone",
        "temperature": 0.78,
        "top_k": 70,
        "cfg_musiccoca": 7.2,
        "cfg_notes": 4.8,
        "cfg_drums": 0.0,
        "motions": [
            "feedback hovering below instability",
            "random voltage slowly steering resonance",
            "wavefolder amount drifting in long arcs",
            "filter self-oscillation breathing softly",
            "CV lag creating non-periodic swells",
        ],
        "effects": [
            "BBD delay, spring reverb, and restrained saturation",
            "soft limiter glue, resonant EQ, and a dark room tail",
            "folded harmonics, low-pass damping, and wide feedback delay",
            "analog drive, notch movement, and smeared tape echo",
            "state-variable filtering, spring reflections, and subharmonic weight",
        ],
        "moods": ["unstable voltage", "alive machinery", "slow feedback weather"],
    },
    {
        "category": "tape_loop",
        "role": "warped tape-loop pad",
        "source": "old string machine, soft organ, and saturated tape loop",
        "guide_kind": "spectral_pad",
        "temperature": 0.54,
        "top_k": 32,
        "cfg_musiccoca": 7.5,
        "cfg_notes": 6.8,
        "cfg_drums": 0.0,
        "motions": [
            "wow and flutter without a rhythmic pulse",
            "slow tape-head color drift",
            "magnetic saturation breathing gently",
            "aged chorus movement",
            "minute pitch sag and recovery",
        ],
        "effects": [
            "tape hiss, soft chorus, and a warm room reverb",
            "worn cassette compression, filtered delay, and mellow plate reverb",
            "gentle saturation, broad low-pass color, and stable stereo width",
            "old preamp noise, chorus ensemble, and long velvet reverb",
            "tape wobble, spring echo blur, and a wide late tail",
        ],
        "moods": ["faded memory", "warm nocturne", "soft magnetic haze"],
    },
    {
        "category": "air_dust",
        "role": "high air and dust layer",
        "source": "filtered hiss, bowed noise, and soft harmonic air",
        "guide_kind": "cinema_noise",
        "temperature": 0.86,
        "top_k": 104,
        "cfg_musiccoca": 6.8,
        "cfg_notes": 2.8,
        "cfg_drums": 0.0,
        "motions": [
            "slow stereo drift",
            "high-pass shimmer moving like wind",
            "noise grains thinning and thickening",
            "barely audible resonant whistles",
            "soft random panning",
        ],
        "effects": [
            "high-pass filtering, shimmer tail, and diffuse stereo air",
            "soft saturation, airy reverb, and subtle spectral freeze",
            "fine tape hiss, reverse reverb smear, and wide reflections",
            "bowed-noise filtering, transparent compression, and a long room tail",
            "dusty convolution space, resonant high shelf, and slow pan diffusion",
        ],
        "moods": ["silver dust", "open air", "distant atmosphere"],
    },
    {
        "category": "organ_sustain",
        "role": "steady electronic organ sustain",
        "source": "stacked sine oscillators and soft drawbar-like partials",
        "guide_kind": "spectral_pad",
        "temperature": 0.50,
        "top_k": 28,
        "cfg_musiccoca": 7.6,
        "cfg_notes": 6.9,
        "cfg_drums": 0.0,
        "motions": [
            "tiny non-rhythmic detune",
            "slow partial balance drift",
            "very smooth filter color movement",
            "subtle ensemble thickness",
            "continuous breath-like harmonic leveling",
        ],
        "effects": [
            "gentle chorus, warm saturation, and a large stable reverb",
            "transparent compression, soft high-cut, and a wide chapel-like tail",
            "low-noise ensemble, smooth plate reverb, and centered low end",
            "mellow overdrive, long hall reverb, and slow stereo spread",
            "subtle tape color, anti-click smoothing, and clear sustained space",
        ],
        "moods": ["still harmony", "pure sustained light", "held electronic breath"],
    },
    {
        "category": "electroacoustic_resonance",
        "role": "electroacoustic resonant field",
        "source": "processed bowed metal, contact-mic hum, and resonator bank",
        "guide_kind": "metallic",
        "temperature": 0.88,
        "top_k": 112,
        "cfg_musiccoca": 6.9,
        "cfg_notes": 4.4,
        "cfg_drums": 0.0,
        "motions": [
            "resonance nodes shifting slowly",
            "bowed noise turning into harmonic fog",
            "contact-mic hum widening over time",
            "low feedback tone folding into the room",
            "small spectral resonances appearing and disappearing",
        ],
        "effects": [
            "convolution space, resonator feedback, and gentle limiting",
            "plate reverb, spectral denoise smear, and wide room tone",
            "comb filtering, soft saturation, and a distant concrete chamber",
            "granular stretch, tuned resonators, and dark stereo reflections",
            "bowed-metal diffusion, slow EQ tilt, and long reflective tails",
        ],
        "moods": ["museum machinery", "concrete resonance", "tactile atmosphere"],
    },
]


NAME_PAIRS = [
    ("Low", "Voltage Fog"),
    ("Spectral", "Ice Halo"),
    ("Carbon", "Room Tone"),
    ("Buffer", "Dust Field"),
    ("Bronze", "Harmonic Mist"),
    ("Feedback", "Corridor"),
    ("Tape", "Memory Drift"),
    ("Silver", "Air Veil"),
    ("Sine", "Cathedral Hold"),
    ("Resonant", "Concrete Bloom"),
    ("Obsidian", "Filter Lake"),
    ("Glass", "Phase Weather"),
    ("Charcoal", "Sub Horizon"),
    ("Frozen", "Circuit Rain"),
    ("Chrome", "Bell Cloud"),
    ("Random", "Voltage Garden"),
    ("Cassette", "Organ Glow"),
    ("Dust", "Aurora"),
    ("Drawbar", "Night Sheet"),
    ("Bowed", "Metal Atmosphere"),
    ("Violet", "Oscillator Tide"),
    ("Prism", "Wavetable Snow"),
    ("Asphalt", "Noise Tide"),
    ("Granular", "Static Bloom"),
    ("Nickel", "Resonator Sky"),
    ("Folded", "Feedback Fog"),
    ("Oxide", "Tape Horizon"),
    ("Hiss", "Halo"),
    ("Soft", "Sine Monolith"),
    ("Contact", "Mic Nebula"),
    ("Dusk", "VCO Plain"),
    ("White", "Spectral Meadow"),
    ("Black", "Convolution Sea"),
    ("Cracked", "Freeze Tail"),
    ("Iron", "Partial Drift"),
    ("Self", "Oscillation Room"),
    ("Magnetic", "Chorus Field"),
    ("Bowed", "Airline"),
    ("Pure", "Organ Fog"),
    ("Tuned", "Stone Resonance"),
    ("Midnight", "Ladder Drift"),
    ("Halo", "Formant Cloud"),
    ("Smoke", "Bandpass Floor"),
    ("Broken", "Sampler Sheet"),
    ("Copper", "Ring Mist"),
    ("CV", "Weather Shelf"),
    ("Worn", "Loop Chapel"),
    ("Highpass", "Dust Garden"),
    ("Stacked", "Sine Chapel"),
    ("Concrete", "String Hum"),
    ("Basalt", "Phase Engine"),
    ("Additive", "Glass Plain"),
    ("Sub", "Pressure Weather"),
    ("Freeze", "Index Field"),
    ("Platinum", "FM Halo"),
    ("Spring", "Feedback Bloom"),
    ("Velvet", "Tape Organ"),
    ("Wind", "Hiss Sheet"),
    ("Static", "Drawbar Sea"),
    ("Resonator", "Bank Fog"),
    ("Moss", "Voltage Cathedral"),
    ("Blue", "Spectral Lens"),
    ("Cavern", "Noise Bloom"),
    ("Broken", "Bit Halo"),
    ("Tungsten", "Overtone Garden"),
    ("Foldback", "Resonance Room"),
    ("Flutter", "Memory Bed"),
    ("Needle", "Air Fog"),
    ("Soft", "Partial Organ"),
    ("Bowed", "Concrete Tail"),
    ("Slate", "Oscillator Lake"),
    ("Crystal", "Freeze Bloom"),
    ("Distant", "Smoke Floor"),
    ("Pixel", "Buffer Chapel"),
    ("Mercury", "Comb Cloud"),
    ("Patch", "Cable Weather"),
    ("Old", "Preamp Horizon"),
    ("Pale", "Noise Veil"),
    ("Sustained", "Sine Weather"),
    ("Metal", "Contact Fog"),
    ("Ash", "VCO Monolith"),
    ("Polar", "Wavetable Halo"),
    ("Lowpass", "Carbon Sea"),
    ("Glitch", "Dust Monolith"),
    ("Quiet", "Ring Cathedral"),
    ("Fold", "Voltage Undertow"),
    ("Tape", "Oxide Sheet"),
    ("High", "Air Cathedral"),
    ("Still", "Organ Plain"),
    ("Resonant", "Wire Bloom"),
    ("Shadow", "Oscillator Fog"),
    ("Frost", "Glass Undertow"),
    ("Cinema", "Noise Chapel"),
    ("Frozen", "Grain Lake"),
    ("Alloy", "FM Weather"),
    ("Feedback", "Spring Field"),
    ("Loop", "Velvet Hall"),
    ("Dust", "Phase Halo"),
    ("Pure", "Sine Horizon"),
    ("Bowed", "Resonator Night"),
]


def slug(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_").replace("__", "_")


def build_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for index, (left, right) in enumerate(NAME_PAIRS, start=1):
        category = CATEGORIES[(index - 1) % len(CATEGORIES)]
        name = f"{left} {right}"
        motion = category["motions"][(index - 1) % len(category["motions"])]
        effect = category["effects"][((index - 1) // 2) % len(category["effects"])]
        mood = category["moods"][(index - 1) % len(category["moods"])]
        register = [
            "continuous low-register",
            "continuous mid-register",
            "wide full-register",
            "soft high-mid",
            "deep low-mid",
        ][(index - 1) % 5]
        space = [
            "a wide stable stereo field",
            "slow pan diffusion",
            "a centered low end and wide upper reflections",
            "large distant-room depth",
            "close dry center with a long rear tail",
        ][(index - 1) % 5]
        prompt = (
            f"Create an original beatless ambient synthesizer texture called {name}. "
            f"A {register} {category['source']} {category['role']} with {motion} "
            "and continuous harmonic or noise movement. "
            f"Add {effect} with {space}. "
            f"The sound should feel like {mood} and evolve continuously as a sustained environment, "
            "not a beat, song section, or lead melody. "
            "Avoid drums, percussion, vocals, arpeggios, sequencer patterns, rhythmic pulses, "
            "EDM drops, sudden cuts, fade-outs, and silence."
        )
        entries.append(
            {
                "index": index,
                "id": f"{index:03d}_{slug(name)}",
                "name": name,
                "category": category["category"],
                "role": category["role"],
                "source": category["source"],
                "guide_kind": category["guide_kind"],
                "temperature": category["temperature"],
                "top_k": category["top_k"],
                "cfg_musiccoca": category["cfg_musiccoca"],
                "cfg_notes": category["cfg_notes"],
                "cfg_drums": category["cfg_drums"],
                "prompt": prompt,
            }
        )
    if len(entries) != 100:
        raise RuntimeError(f"Expected 100 prompts, got {len(entries)}")
    return entries


def write_outputs(entries: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mrt-ambient-100-prompt-library-v1",
        "count": len(entries),
        "prompt_page_size": 4,
        "recommended_weight_mode": "page_sine",
        "recommended_prompt_page_seconds": 4.0,
        "negative_constraints": [
            "drums",
            "percussion",
            "vocals",
            "arpeggios",
            "sequencer patterns",
            "rhythmic pulses",
            "EDM drops",
            "fade-outs",
            "silence",
        ],
        "prompts": entries,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    header = [
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
    rows = ["\t".join(header)]
    for entry in entries:
        rows.append(
            "\t".join(
                [
                    str(entry["id"]),
                    str(entry["name"]),
                    str(entry["category"]),
                    str(entry["prompt"]),
                    str(entry["guide_kind"]),
                    f"{float(entry['temperature']):.3f}",
                    str(entry["top_k"]),
                    f"{float(entry['cfg_musiccoca']):.3f}",
                    f"{float(entry['cfg_notes']):.3f}",
                    f"{float(entry['cfg_drums']):.3f}",
                ]
            )
        )
    TSV_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")

    lines = [
        "# Ambient 100 Prompt Library",
        "",
        "Use this library as 25 pages of 4 prompt slots. Each prompt is written for beatless sustained ambient modulation.",
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                f"## {entry['index']:03d}. {entry['name']}",
                "",
                f"- id: `{entry['id']}`",
                f"- category: `{entry['category']}`",
                f"- controls: cfg_musiccoca={entry['cfg_musiccoca']}, cfg_notes={entry['cfg_notes']}, temperature={entry['temperature']}, top_k={entry['top_k']}",
                "",
                str(entry["prompt"]),
                "",
            ]
        )
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")

    pages = []
    for page in range(25):
        page_prompts = entries[page * 4:(page + 1) * 4]
        pages.append(
            {
                "page": page,
                "start_seconds": page * PAGE_SECONDS,
                "end_seconds": (page + 1) * PAGE_SECONDS,
                "prompt_ids": [entry["id"] for entry in page_prompts],
                "prompt_names": [entry["name"] for entry in page_prompts],
                "weight_mode": "page_sine",
                "lane_phases": [0.0, 0.25, 0.5, 0.75],
            }
        )
    PLAN_PATH.write_text(
        json.dumps(
            {
                "schema": "mrt-ambient-100-prompt-modulation-plan-v1",
                "prompt_library": str(JSON_PATH.relative_to(ROOT)),
                "prompt_library_tsv": str(TSV_PATH.relative_to(ROOT)),
                "midi": str(MIDI_PATH.relative_to(ROOT)),
                "bpm": BPM,
                "duration_seconds": DURATION_SECONDS,
                "prompt_page_size": 4,
                "prompt_page_seconds": PAGE_SECONDS,
                "weight_mode": "page_sine",
                "frame_rate": 60,
                "pages": pages,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


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


def write_cm9_hold_midi(path: Path) -> None:
    ppq = 480
    tempo_us = round(60_000_000 / BPM)
    end_ticks = int(round(DURATION_SECONDS * BPM / 60 * ppq))
    chord = [36, 43, 48, 51, 55, 58, 62, 67]
    track = bytearray()

    def add(delta: int, event: bytes) -> None:
        track.extend(vlq(delta))
        track.extend(event)

    name = b"Ambient 100 Prompt Cm9 Hold"
    add(0, b"\xff\x03" + vlq(len(name)) + name)
    add(0, b"\xff\x51\x03" + tempo_us.to_bytes(3, "big"))
    add(0, b"\xff\x58\x04" + bytes([4, 2, 24, 8]))
    for note in chord:
        add(0, bytes([0x90, note, 84]))
    for index, note in enumerate(chord):
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


def main() -> int:
    entries = build_entries()
    write_outputs(entries)
    write_cm9_hold_midi(MIDI_PATH)
    print(JSON_PATH)
    print(TSV_PATH)
    print(MD_PATH)
    print(PLAN_PATH)
    print(MIDI_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
