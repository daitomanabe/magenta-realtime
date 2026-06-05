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
- [ ] Generate at least one short WAV after render changes.
- [ ] Verify rendered audio with metadata and non-silence checks.
- [ ] Push after each completed milestone or coherent feature slice.

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
