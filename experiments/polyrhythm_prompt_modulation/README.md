# Polyrhythm Prompt Modulation Test

This experiment renders Magenta RealTime test material from four prompt slots
whose weights move in rhythmic polyrhythm. The same modulation data can be used
to blend video sources or to drive segmented Runway video prompts.

## Prompt Slots

1. Complex jazzy tension chords, warm electric piano.
2. Fast glassy arpeggio figures.
3. Dry broken beat drums, tight room.
4. Granular tape texture and vinyl air.

## Workflow

```bash
.venv/bin/python experiments/polyrhythm_prompt_modulation/generate_modulation.py
.venv/bin/python experiments/polyrhythm_prompt_modulation/render_magenta_prompt_modulated_audio.py
source "${HOME}/.config/secrets.zsh"
.venv/bin/python experiments/polyrhythm_prompt_modulation/runway_generate_videos.py --execute
.venv/bin/python experiments/polyrhythm_prompt_modulation/composite_videos_from_modulation.py
```

## Three Minute Version

```bash
.venv/bin/python experiments/polyrhythm_prompt_modulation/generate_longform_modulation.py
.venv/bin/python experiments/polyrhythm_prompt_modulation/render_longform_magenta_audio.py
source "${HOME}/.config/secrets.zsh"
.venv/bin/python experiments/polyrhythm_prompt_modulation/runway_generate_longform_segments.py --execute
.venv/bin/python experiments/polyrhythm_prompt_modulation/assemble_longform_video.py
```

## Clear MIDI Prompt Diagnostic

This diagnostic uses `assets/Am-Minor Prog 01 (i-VI-v-iv).mid`, whose four
2-second chord regions line up with four exaggerated prompt slots:

1. Solo acoustic piano chords.
2. 8-bit chiptune square-wave arps.
3. Distorted 808 sub bass and trap drums.
4. Choir and string drone.

The control matrix is explicit:

- Primary control: prompt embedding mix.
- Secondary control: CFG weights.
- Expression control: temperature.
- Exploration control: top-k.
- Stability control: fixed 25 Hz frames / 1920 sample chunks.
- Advanced control: optional audio prompt embeddings synthesized from the MIDI
  progression.

```bash
.venv/bin/python experiments/polyrhythm_prompt_modulation/generate_clear_prompt_midi_test.py \
  --midi 'assets/Am-Minor Prog 01 (i-VI-v-iv).mid'
.venv/bin/python experiments/polyrhythm_prompt_modulation/render_clear_prompt_midi_audio.py \
  --embedding-source audio \
  --match-segment-rms \
  --output outputs/polyrhythm_prompt_modulation/magenta_audio_prompt_control_matrix_8s_matched.wav \
  --report outputs/polyrhythm_prompt_modulation/magenta_audio_prompt_control_matrix_8s_matched.report.json
```

Generated audio/video files are written under `outputs/polyrhythm_prompt_modulation/`.
The modulation values and prompt plan are written under
`experiments/polyrhythm_prompt_modulation/data/`.

### C++ MIDI Engine Diagnostic

The C++ diagnostic uses the official `MLXEngine::set_note_on/off` path instead
of Python-side note-token arrays. It parses the same MIDI file, schedules note
events into `MidiNoteTracker`, modulates cached prompt embeddings through
`reblend_musiccoca_tokens`, and changes CFG, temperature, and top-k every 25 Hz
frame.

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -downloadComponent MetalToolchain
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 \
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cmake -S . -B build
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 \
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cmake --build build --target mrt2_midi_prompt_diagnostic -j 8
./build/examples/midi_prompt_diagnostic/mrt2_midi_prompt_diagnostic \
  --midi 'assets/Am-Minor Prog 01 (i-VI-v-iv).mid' \
  --output outputs/polyrhythm_prompt_modulation/magenta_cpp_midi_control_matrix_8s.wav \
  --report outputs/polyrhythm_prompt_modulation/magenta_cpp_midi_control_matrix_8s.report.json
```

Use `--text-prompts` to compare text prompt encoding against the default
MIDI-derived audio prompt embeddings.

### Clear C++ Prompt Experiment Suite

For clearer A/B-style listening tests, the C++ diagnostic can run named
profiles with fixed MIDI input, repeated to a requested duration. Each profile
uses four cached prompt slots, four MIDI-derived audio prompt guide signals,
and frame-level modulation of prompt embedding mix, CFG, temperature, and
top-k.

The profile prompts follow the same concrete genre vocabulary used for Lyria
loop prompt design: original instrumental loop material, explicit BPM/rhythmic
identity, sound sources, mood/space, and exclusions.

Available profiles:

- `clear_extreme`: piano chords, chiptune arps, distorted 808 drums, choir
  string drone.
- `microcinematic_footwork`: footwork grid, sub trap stabs, granular metal
  cuts, noir string pressure.
- `glass_trap_pressure`: glass mallets, 808 trap weight, microcut texture,
  cinematic afterglow.
- `negative_space_club`: dry minimal pulse, dub chord stabs, sub pressure,
  air field.
- `metallic_ambient_bounce`: metallic bounce, ambient pad floor, sub minimal
  knock, digital insect grid.

Run the short tests first, then the 30 second experiments:

```bash
python3 experiments/polyrhythm_prompt_modulation/run_cpp_prompt_experiment_suite.py \
  --stage clear_10s
python3 experiments/polyrhythm_prompt_modulation/run_cpp_prompt_experiment_suite.py \
  --stage long_30s
```

Generated WAVs, per-run reports, MIDI-derived audio prompt guides, and the
combined manifest are written to:

```text
outputs/polyrhythm_prompt_modulation/cpp_prompt_experiments/
```
