# Magenta RealTime Sequencer TODO

This document tracks development of a full-featured Magenta RealTime sequencer
for real-time performance, parameter modulation, and offline loop export.

## Development Rules

- Work in small milestones and push each milestone to GitHub.
- Keep the app usable after every push.
- Prefer existing repo surfaces over new abstractions:
  - `magentart::core::RealtimeRunner` for live audio.
  - `MLXEngine` for deterministic offline render paths.
  - Shared React UI components under `examples/common/react_ui`.
  - Existing app patterns from `examples/jam`, `examples/collider`, and
    `examples/mrt2/standalone`.
- Do not re-encode text prompts at step rate. Cache prompt slots and modulate
  prompt weights, parameters, and MIDI notes.
- Keep model load, resource download, prefill, and state I/O off the audio
  callback.
- Every audio export feature must verify sample rate, channel count, duration,
  and non-silence.

## Push Milestones

### 0. Planning And Scaffold

- [x] Fork `magenta/magenta-realtime` to the user GitHub account.
- [x] Clone the fork locally and add `upstream`.
- [x] Initialize submodules.
- [x] Add this TODO.
- [x] Push this TODO branch to GitHub.
- [x] Create `examples/sequencer/` app scaffold.
- [x] Add CMake target and local deploy target.
- [ ] Reuse Jam/Standalone model manager, settings, audio engine, and MIDI setup.
- [x] Add minimal React shell that boots as the sequencer first screen.

### 1. Transport And Timing Core

- [ ] Add native sequencer clock with BPM, bars, steps-per-bar, swing, and loop
  length.
- [ ] Add play, stop, reset, panic, and loop position.
- [x] Sync React pattern state to native with a bulk `sequencerPattern` message.
- [ ] Apply due events on the native/control thread, not from React timers.
- [ ] Clear stuck notes on stop, reset, panic, and pattern replacement.
- [ ] Surface transformer timing, buffer fill, and dropped frames.

### 2. Note And Chord Sequencing

- [ ] Add a step grid with per-step active state, note, velocity-style strength,
  gate, probability, ratchet, octave, and humanize.
- [ ] Add chord lanes with named chord symbols and explicit MIDI note output.
- [x] Add jazz chord generator presets:
  - [x] ii-V-I with substitutions.
  - [x] Modal interchange loops.
  - [x] Diminished passing chords.
  - [x] Tritone substitution movement.
  - [x] Upper-structure triads.
  - [x] Quartal voicings.
  - [x] Dense tension voicings with 9, b9, #9, 11, #11, 13, b13.
- [x] Add 8-bar loop templates for complex jazzy chord materials.
- [x] Add arp extraction from chord lanes with up, down, random, outside-tone,
  skip, Euclidean, and polymetric modes.
- [ ] Add scale/avoid-note helpers for jazz tensions.

### 3. Magenta Parameter Modulation

- [ ] Add automation lanes for every core Magenta control:
  - [ ] `temperature`.
  - [ ] `topk`.
  - [ ] `cfgmusiccoca`.
  - [ ] `cfgnotes`.
  - [ ] `cfgdrums`.
  - [ ] `unmaskwidth`.
  - [ ] `seedrotation`.
  - [ ] `drumless`.
  - [ ] `midigate`.
  - [ ] `onsetmode`.
  - [ ] `volume`.
  - [ ] `mute`.
  - [ ] `bypass`.
  - [ ] prompt slot weights 0-5.
  - [ ] PCA coefficients 0-5 when a corpus is loaded.
- [ ] Add per-lane shapes: step, ramp, sample-and-hold, sine, triangle,
  exponential, random walk, chaos, and probability gates.
- [ ] Add per-lane phase, rate division, smoothing, min/max, and latch modes.
- [ ] Add macro knobs that can modulate multiple lanes at once.
- [ ] Add scene snapshots that morph multiple parameters over bars.

### 4. Prompt And Style Sequencing

- [ ] Add six prompt slots with text prompts, audio prompts, color tags, and
  per-slot weight automation.
- [ ] Add prompt scenes and per-step scene locks.
- [ ] Add prompt morph recording from a 2D surface.
- [ ] Add prompt randomizer constrained by tags such as jazz, arp, drums,
  texture, bass, chords, noisy, clean, sparse, dense.
- [ ] Add audio prompt import through decoded PCM samples and
  `set_audio_prompt_samples`.
- [ ] Add prompt encode status and prevent destructive step-rate re-encoding.

### 5. State, Prefill, And Variation

- [ ] Add factory reset, silent prefill, audio prefill, and state bank actions.
- [ ] Add per-pattern start state: factory, silence, saved bank, or audio prefill.
- [ ] Add variation controls using seed rotation, temperature, top-k, and CFG
  ranges.
- [ ] Add capture of generated RVQ tokens for lossless branch/prefill
  experiments when practical.

### 6. Real-Time Performance UI

- [ ] Build the first screen as the sequencer, not a landing page.
- [ ] Add dense but readable grid, piano roll, automation lanes, prompt lanes,
  transport, and compact metrics.
- [ ] Use icon buttons for transport/tools and tooltips for unfamiliar controls.
- [ ] Add keyboard shortcuts for play, stop, panic, step edit, duplicate, clear,
  randomize, render, and save.
- [ ] Add MIDI source selection and computer-keyboard MIDI input.
- [ ] Add pattern save/load in `NSUserDefaults` or a local project file.

### 7. Offline Export

- [ ] Add offline render engine for 48 kHz stereo WAV.
- [ ] Render exact loop lengths such as 1, 2, 4, 8, and 16 bars.
- [ ] Add tail/crossfade handling for seamless loops.
- [ ] Export stems or variants when possible:
  - [ ] Full loop.
  - [ ] Chord-focused loop.
  - [ ] Arp-focused loop.
  - [ ] Texture loop.
  - [ ] Multiple random seeds.
- [ ] Add verification report with duration, sample rate, channels, peak, RMS,
  and non-silence.
- [ ] Add export presets:
  - [ ] 8-bar complex jazzy tension chord loop.
  - [ ] 8-bar arpeggio material.
  - [ ] 8-bar chord plus texture loop.
  - [ ] 4-bar rhythmic prompt-morph loop.
  - [ ] 16-bar evolving ambient/jazz loop.

### 8. Material Generators

- [x] Add chord progression generator for 8-bar jazz loops.
- [ ] Add voicing generator with close, drop-2, spread, quartal, and cluster
  options.
- [x] Add arp generator from generated voicings.
- [ ] Add prompt/parameter recipe generator for interesting audio materials.
- [ ] Add batch export queue for many variations from one recipe.
- [ ] Add metadata sidecars for exported loops.

### 9. AUv3 And DAW Workflow

- [ ] Decide whether to port the sequencer into AUv3 after standalone stabilizes.
- [ ] Map sequencer macros to AU parameters where practical.
- [ ] Preserve DAW-host MIDI and transport behavior.
- [ ] Add pattern export/import for standalone-to-plugin reuse.

### 10. Quality Gates

- [ ] Build `hello_mrt2` after C++ core changes.
- [x] Build `deploy_mrt2_sequencer` after app changes.
- [x] Run relevant React build/type checks.
- [ ] Run focused Python tests when Python export/generation code changes.
- [x] Generate at least one short WAV after render changes.
- [x] Verify rendered audio with metadata and non-silence checks.
- [ ] Push after each completed milestone or coherent feature slice.

### 11. Polyrhythm Prompt/Video Test

- [x] Generate four prompt weight lanes with 3:5:7:11 polyrhythm frequencies
  and independent phase offsets.
- [x] Save prompt modulation values as JSON and CSV.
- [x] Render a 10 second `mrt2_small` WAV using frame-level prompt blending.
- [x] Generate one dynamic Runway landscape base video.
- [x] Generate four Aleph video variants related to the four Magenta prompts.
- [x] Composite the four Aleph videos with the saved modulation data.
- [x] Attach the 10 second Magenta WAV to the video composite.

### 12. Three Minute Longform Test

- [x] Add intro-to-outro arrangement sections for an approximately 3 minute
  version.
- [x] Generate evolving prompt modulation that intensifies toward the second
  half.
- [x] Render one continuous 180 second Magenta WAV from start to finish.
- [x] Save longform modulation values and 18 Runway segment prompts.
- [x] Generate 18 segmented Runway landscape videos.
- [x] Assemble the segmented videos with the continuous Magenta WAV.
- [x] Verify final duration, video stream, audio stream, and non-silence.

### 13. Diagnostic Prompt Weight Verification

- [x] Clarify that the 10 second test uses base Runway video, four Aleph style
  variants, and prompt-weight compositing.
- [x] Clarify that the 3 minute test uses direct segmented Runway videos, not
  four Aleph style variants.
- [x] Generate an 8 second diagnostic modulation file with obvious one-hot
  prompt weight regions and short crossfades.
- [x] Composite the four Aleph videos with the diagnostic modulation file.
- [x] Verify solo regions by comparing composite frames against the expected
  source Aleph style video.

### 14. Clear Prompt MIDI Audio Test

- [x] Parse the provided `Am-Minor Prog 01 (i-VI-v-iv).mid` progression.
- [x] Generate an 8 second prompt-weight test aligned to the four MIDI chord
  regions.
- [x] Save a frame-level control matrix:
  - [x] Primary control: prompt embedding mix.
  - [x] Secondary control: CFG weights.
  - [x] Expression control: temperature.
  - [x] Exploration control: top-k.
  - [x] Stability control: fixed 25 Hz frames / 1920 sample chunks.
- [x] Replace subtle jazz/texture prompts with four exaggerated diagnostic
  prompts: solo piano, chiptune square arps, distorted 808 drums, and choir
  string drone.
- [x] Generate four MIDI-derived audio prompt guide files and embed them with
  MusicCoCa audio prompt embedding.
- [x] Render `mrt2_small` audio with frame-level prompt blending and 128-note
  MIDI conditioning.
- [x] Add an audition render option that loudness-matches 2 second sections so
  prompt changes are easier to hear.
- [x] Verify duration, sample rate, channels, non-silence, and per-section
  spectral metrics.
- [x] Mux the clearer MIDI-conditioned audition audio into the diagnostic
  prompt-weight video.

### 15. C++ MIDI Prompt Diagnostic

- [x] Add a C++ diagnostic target that drives the official
  `MLXEngine::set_note_on/off` MIDI path.
- [x] Parse the provided MIDI file in C++ and schedule note on/off events at
  25 Hz frame boundaries.
- [x] Modulate primary prompt embedding mix through
  `reblend_musiccoca_tokens`.
- [x] Modulate secondary CFG, expression temperature, and exploration top-k
  from the same four-slot control matrix.
- [x] Generate four MIDI-derived audio prompt guide signals in C++ and encode
  them with `set_audio_prompt_samples`.
- [x] Verify the C++ target after the local Xcode Metal Toolchain is available.

### 16. Clear C++ Genre Prompt Experiments

- [x] Add C++ profile selection for clearer prompt tests inspired by Lyria-style
  loop prompt design.
- [x] Add duration and transition controls so the same MIDI progression can be
  repeated into 10 second and 30 second tests.
- [x] Add profiles for clear extremes, microcinematic footwork, glass trap
  pressure, negative space club, and metallic ambient bounce.
- [x] Run three 10 second tests with exaggerated prompt/audio-guide contrast.
- [x] Run three 30 second experiments for longer evolution checks.
- [x] Save a combined manifest with WAV path, report path, duration, peak, RMS,
  and non-silence verification for all six renders.
- [x] Keep stability controls fixed at 25 Hz frames / 1920 sample chunks while
  modulating prompt embedding mix, CFG, temperature, and top-k.

### 17. Sustained Synth Prompt Sources

- [x] Create a reusable sustained-synth prompt design skill for texture,
  cinematic, glitch, noise, synthesizer, modular synth, and post-effect
  long-tone material.
- [x] Add a `sustained_synth_textures` C++ profile with four high-contrast
  prompt slots: low analog drone, spectral wavetable pad, cinematic noise bed,
  and granular glitch sustain.
- [x] Add `solo` and `modulated` prompt weight modes for rendering individual
  prompt sources and continuous prompt-weight blends.
- [x] Write frame-level prompt weight JSON including weights, CFG, temperature,
  and top-k.
- [x] Render four 10 second solo text-prompt sources.
- [x] Render one 30 second prompt-weight modulation across the four sustained
  synth prompts.
- [x] Verify duration, non-silence, and normalized prompt weights for all five
  renders.

## First Target User Flow

1. Launch the sequencer app.
2. Load or download `mrt2_small`.
3. Pick an 8-bar jazz tension chord recipe.
4. Press Play and hear realtime output.
5. Turn macro knobs that modulate prompt weights, CFG, temperature, top-k, and
   note strength.
6. Randomize variations until the loop is interesting.
7. Export an exact 8-bar WAV loop.
8. Inspect the render report and reuse the loop as chord, arp, or texture
   material.
