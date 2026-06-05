# Polyrhythm Prompt Modulation Test

This experiment renders a 10 second Magenta RealTime test loop from four prompt
slots whose weights move in rhythmic polyrhythm. The same modulation data can be
used to blend four Runway/Aleph video variants.

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

Generated audio/video files are written under `outputs/polyrhythm_prompt_modulation/`.
The modulation values and prompt plan are written under
`experiments/polyrhythm_prompt_modulation/data/`.
