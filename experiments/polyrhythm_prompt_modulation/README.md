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

### Sustained Synth Prompt Sources

This test uses the `sustained_synth_textures` C++ profile for long-tone prompt
material instead of rhythm-forward genre prompts. The profile contains four
high-contrast text prompt slots:

1. `Low Voltage Fog`: low modular VCO drone, phase beating, low-pass filter,
   tape delay, reverb.
2. `Spectral Glass Bloom`: bright wavetable pad, spectral freeze, shimmer,
   chorus, stereo widening.
3. `Carbon Cinema Noise`: dark cinematic noise bed, sub pressure, band-pass
   sweep, convolution-like space.
4. `Buffer Freeze Dust`: granular frozen buffer sustain, bit-depth erosion,
   reverse delay, resonant notch movement.

Run four 10 second solo prompt sources and one 30 second prompt-weight
modulation:

```bash
python3 experiments/polyrhythm_prompt_modulation/run_sustained_synth_sources.py
```

Run only the 16 second BPM 120 loop-sine prompt-weight modulation:

```bash
python3 experiments/polyrhythm_prompt_modulation/run_sustained_synth_sources.py \
  --stage loop_16s
```

Run only the 32-bar BPM 130 Cm9 sustained-chord modulation:

```bash
python3 experiments/polyrhythm_prompt_modulation/run_sustained_synth_sources.py \
  --stage cm9_32bars_130
```

Run four 32-bar no-decay prompt trials against a single long-held Cm9 MIDI
chord, without per-bar note retriggers:

```bash
python3 experiments/polyrhythm_prompt_modulation/run_sustained_synth_sources.py \
  --stage no_decay_trials
```

The `loop_sine` weight mode uses one full sine cycle over 16 seconds, with the
four prompt slots phase-shifted by 90 degrees. At 120 BPM this is an exact
8-bar modulation cycle, and the prompt-weight endpoint returns to the starting
weights for seamless looping. The sine values are curved to make the modulation
visibly and audibly obvious: each prompt can dominate at roughly 0.77 while the
quietest prompts fall to roughly 0.01.

The `meter_sine` weight mode is designed for longer MIDI-conditioned exports.
For the Cm9 test, the runner writes a 32-bar BPM 130 MIDI file and drives four
phase-shifted sine modulators at 4/4 phase 0.0, 3/4 phase 0.0, 6/8 phase 0.0,
and 4/4 phase 0.25. The frame-level weight JSON is written at 60 fps for video
modulation. The runner retriggers the Cm9 chord once per bar so the generated
audio stays active across the full 32-bar export, then writes a per-second WAV
activity check JSON to catch accidental dropouts after generation.

The `no_decay_trials` stage uses a separate `sustained_no_decay_trials` profile
to test whether prompt wording alone can keep the sound active for the full
32 bars. It records `first_3s_rms`, `last_8s_median_rms`, and their ratio in
each `.audio_check.json`. The revised beatless synth-pad prompts avoid granular
texture, BBD delay, delay taps, feedback rhythm, tremolo, pulses, and sequenced
motion; the stacked-sine / organ-pad phrasing has been the most reliable
non-decaying sustained synth result.

For non-solo runs, the runner also renders a silent graph MP4 from the
frame-level weight JSON. The graph shows the complete prompt-weight curves, a
moving playhead, and current weight bars for visual QA.

This runner uses text prompt embeddings by default so prompt wording changes
are audible in the generated material. Use `--audio-prompts` to switch the same
profile to MIDI-derived synthetic audio prompt embeddings.

Generated WAVs, reports, frame-level prompt weight JSON files, graph videos,
and the manifest are written to:

```text
outputs/polyrhythm_prompt_modulation/sustained_synth_prompt_sources/
```

### Ambient 100 Prompt Paging

This workflow creates 100 beatless ambient synth prompts, stores them as JSON,
Markdown, and TSV, then renders a 100 second Cm9 ambient pass that pages through
all prompts four at a time. Each four-prompt page lasts 4 seconds, and the
`page_sine` weight mode applies four phase-shifted sine modulators inside the
page while CFG, temperature, and top-k follow the active weighted prompts.

```bash
python3 experiments/polyrhythm_prompt_modulation/generate_ambient_100_prompt_library.py
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 \
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cmake --build build --target mrt2_midi_prompt_diagnostic -j 8
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 \
  build/examples/midi_prompt_diagnostic/mrt2_midi_prompt_diagnostic \
    --midi outputs/polyrhythm_prompt_modulation/ambient_100_prompt_modulation/ambient_100_cm9_hold_100s.mid \
    --profile sustained_synth_textures \
    --duration 100 \
    --weight-mode page_sine \
    --prompt-page-seconds 4 \
    --prompt-library experiments/polyrhythm_prompt_modulation/data/ambient_100_prompt_library.tsv \
    --output outputs/polyrhythm_prompt_modulation/ambient_100_prompt_modulation/ambient_100_prompt_page_sine_modulation_100s.wav \
    --report outputs/polyrhythm_prompt_modulation/ambient_100_prompt_modulation/ambient_100_prompt_page_sine_modulation_100s.report.json \
    --weights-output outputs/polyrhythm_prompt_modulation/ambient_100_prompt_modulation/ambient_100_prompt_page_sine_modulation_100s.weights.json \
    --text-prompts
```

The full 60 fps modulation data remains in `.weights.json`. For preview video,
downsample the weights to 10 fps and render the graph, then mux the WAV:

```bash
python3 experiments/polyrhythm_prompt_modulation/downsample_prompt_weight_frames.py \
  --input outputs/polyrhythm_prompt_modulation/ambient_100_prompt_modulation/ambient_100_prompt_page_sine_modulation_100s.weights.json \
  --output outputs/polyrhythm_prompt_modulation/ambient_100_prompt_modulation/ambient_100_prompt_page_sine_modulation_100s.weights.preview_10fps.json \
  --fps 10
python3 experiments/polyrhythm_prompt_modulation/render_prompt_weight_graph_video.py \
  --weights outputs/polyrhythm_prompt_modulation/ambient_100_prompt_modulation/ambient_100_prompt_page_sine_modulation_100s.weights.preview_10fps.json \
  --output outputs/polyrhythm_prompt_modulation/ambient_100_prompt_modulation/ambient_100_prompt_page_sine_modulation_100s.graph_10fps.mp4
ffmpeg -y \
  -i outputs/polyrhythm_prompt_modulation/ambient_100_prompt_modulation/ambient_100_prompt_page_sine_modulation_100s.graph_10fps.mp4 \
  -i outputs/polyrhythm_prompt_modulation/ambient_100_prompt_modulation/ambient_100_prompt_page_sine_modulation_100s.wav \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart \
  outputs/polyrhythm_prompt_modulation/ambient_100_prompt_modulation/ambient_100_prompt_page_sine_modulation_100s.graph_audio_10fps.mp4
```

The expected verification is: 100.0 second WAV, `non_silent=true`, no quiet
windows after 3 seconds, 6000 weight frames at 60 fps, 25 prompt pages, and
100 unique prompt IDs used by active frames.

### Dirty Cinematic Cm9 64-Bar Test

This test renders one 64-bar, 120 BPM Cm9 held-chord pass with four prompts in
one set: dirty analog drone, cinematic noise floor, frozen glitch texture, and
dirty spectral glass pad. The prompts explicitly avoid drums, percussion,
arpeggios, sequencer patterns, rhythmic pulses, tremolo, delay taps, vocals,
drops, fade-outs, and silence.

The control matrix is:

- Primary control: prompt embedding mix through `meter_macro_sine` prompt
  weights. The micro layer is BPM-synced `meter_sine` for 4/4, 3/4, 6/8,
  and 4/4 with phase offset 0.25. The macro layer uses different sine cycles
  over a 128 second reference and is multiplied into the prompt mix.
- Secondary control: `ambient_crescendo` CFG curve, rising toward the second
  half (`cfg_musiccoca` 5.8 to 7.8, `cfg_notes` 5.6 to 6.4).
- Expression control: temperature moves within 0.6075 to 0.7205.
- Exploration control: top-k moves within 51 to 103.
- Stability control: fixed 25 Hz frames / 1920 sample chunks.
- Long-form output guard: explicit 1 second window RMS floor stabilization
  (`min_window_rms=0.012`, `max_window_gain=1024`) keeps sustained material
  from dropping into near-silence.

```bash
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 \
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cmake --build build --target mrt2_midi_prompt_diagnostic -j 8
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 \
  python3 experiments/polyrhythm_prompt_modulation/run_dirty_cinematic_cm9_64bar_test.py
```

Generated WAV, 60 fps weight JSON, audio activity check, and graph/audio
preview MP4 are written to:

```text
outputs/polyrhythm_prompt_modulation/dirty_cinematic_cm9_64bar/
```

For the prompt-variant validation batch, render 10 five-minute Cm9 takes. Each
take writes a fresh four-prompt TSV, so both the prompt content and the macro
phase offsets vary between takes:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 \
  python3 experiments/polyrhythm_prompt_modulation/run_dirty_cinematic_cm9_5min_batch.py
```

Each take writes a WAV, render log, 60 fps prompt-weight/control JSON, report,
prompt TSV, and audio continuity check under:

```text
outputs/polyrhythm_prompt_modulation/dirty_cinematic_cm9_5min_prompt_variants_10/
```

Use `--count 50` only after the 10-take prompt-variant batch has been checked.
