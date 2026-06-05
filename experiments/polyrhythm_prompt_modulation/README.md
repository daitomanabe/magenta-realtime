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

Generated audio/video files are written under `outputs/polyrhythm_prompt_modulation/`.
The modulation values and prompt plan are written under
`experiments/polyrhythm_prompt_modulation/data/`.
