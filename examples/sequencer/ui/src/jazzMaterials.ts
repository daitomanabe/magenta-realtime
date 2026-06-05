import type {ArpMode, RecipeId, SequencerPattern, SequencerStep} from './sequencerTypes';

type ChordPlan = {
  symbol: string;
  root: number;
  intervals: number[];
  color: number;
};

type Recipe = {
  id: RecipeId;
  label: string;
  bpm: number;
  chords: ChordPlan[];
  prompts: string[];
};

const NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];

const arpModes: ArpMode[] = ['up', 'down', 'random', 'outside', 'skip', 'euclidean', 'polymetric'];

export const RECIPES: Recipe[] = [
  {
    id: 'ii-v-substitutions',
    label: 'ii-V-I substitutions',
    bpm: 104,
    chords: [
      chord('Dm9', 50, [0, 3, 7, 10, 14], 0.20),
      chord('G13b9', 55, [0, 4, 10, 13, 21], 0.58),
      chord('Cmaj9#11', 48, [0, 4, 7, 11, 14, 18], 0.36),
      chord('F#7alt', 54, [0, 4, 10, 13, 15, 20], 0.72),
      chord('Fm11', 53, [0, 3, 7, 10, 14, 17], 0.32),
      chord('Bb13#11', 46, [0, 4, 10, 14, 18, 21], 0.62),
      chord('Em7b5', 52, [0, 3, 6, 10, 14], 0.45),
      chord('A7#9b13', 45, [0, 4, 10, 15, 20], 0.82),
    ],
    prompts: [
      'complex jazzy tension chords, warm electric piano',
      'fast glassy arpeggio figures',
      'dry broken beat drums, tight room',
      'sub bass following altered dominants',
      'granular tape texture and vinyl air',
      'muted guitar harmonics',
    ],
  },
  {
    id: 'modal-interchange',
    label: 'modal interchange drift',
    bpm: 92,
    chords: [
      chord('Cmaj9', 48, [0, 4, 7, 11, 14], 0.22),
      chord('Abmaj7#11', 44, [0, 4, 7, 11, 18], 0.48),
      chord('Db13sus', 49, [0, 5, 10, 14, 21], 0.52),
      chord('Cmaj7/E', 52, [0, 3, 8, 11, 19], 0.38),
      chord('F-9', 53, [0, 3, 7, 10, 14], 0.44),
      chord('Bb13', 46, [0, 4, 10, 14, 21], 0.56),
      chord('Ebmaj9', 51, [0, 4, 7, 11, 14], 0.30),
      chord('G7alt', 55, [0, 4, 10, 13, 15, 20], 0.76),
    ],
    prompts: [
      'smoky modal jazz chords',
      'soft mallet arpeggios',
      'brushed broken beat',
      'round acoustic bass movement',
      'room tone and tape hiss',
      'low passed synth doubling harmony',
    ],
  },
  {
    id: 'diminished-passing',
    label: 'diminished passing chords',
    bpm: 118,
    chords: [
      chord('Cmaj9', 48, [0, 4, 7, 11, 14], 0.24),
      chord('C#dim7', 49, [0, 3, 6, 9, 14], 0.68),
      chord('Dm9', 50, [0, 3, 7, 10, 14], 0.34),
      chord('D#dim7', 51, [0, 3, 6, 9, 15], 0.70),
      chord('Em11', 52, [0, 3, 7, 10, 14, 17], 0.40),
      chord('A7b9', 45, [0, 4, 10, 13], 0.64),
      chord('Dm9/F', 53, [0, 4, 9, 14, 21], 0.36),
      chord('G13', 55, [0, 4, 10, 14, 21], 0.58),
    ],
    prompts: [
      'nimble diminished passing harmony',
      'syncopated piano arpeggio fragments',
      'tight cymbal heavy jazz drums',
      'walking bass with chromatic approach notes',
      'short room reflections',
      'plucked synth accents',
    ],
  },
  {
    id: 'tritone-cycle',
    label: 'tritone dominant cycle',
    bpm: 110,
    chords: [
      chord('D7#11', 50, [0, 4, 10, 18], 0.56),
      chord('Ab13b9', 44, [0, 4, 10, 13, 21], 0.70),
      chord('Gm9', 55, [0, 3, 7, 10, 14], 0.32),
      chord('Db7alt', 49, [0, 4, 10, 13, 15], 0.76),
      chord('Cmaj9#11', 48, [0, 4, 7, 11, 14, 18], 0.38),
      chord('F7#9', 53, [0, 4, 10, 15], 0.66),
      chord('Bmaj7#11', 47, [0, 4, 7, 11, 18], 0.44),
      chord('E7b13', 52, [0, 4, 10, 20], 0.62),
    ],
    prompts: [
      'altered dominant tritone cycle',
      'angular electric piano arpeggios',
      'fragmented swing drums',
      'sine bass locking to roots',
      'metallic ambience',
      'short brass stab texture',
    ],
  },
  {
    id: 'upper-structure',
    label: 'upper-structure triads',
    bpm: 96,
    chords: [
      chord('Cmaj9/D', 50, [0, 2, 6, 9, 13], 0.40),
      chord('E/G13', 55, [0, 4, 8, 14, 21], 0.68),
      chord('Bb/C7sus', 48, [0, 5, 10, 14, 22], 0.54),
      chord('F#alt/G', 55, [0, 1, 5, 10, 15], 0.74),
      chord('A/Bbmaj7', 46, [0, 4, 11, 16, 21], 0.42),
      chord('D/E7sus', 52, [0, 5, 10, 14, 21], 0.60),
      chord('Ab/B13', 47, [0, 4, 8, 13, 21], 0.70),
      chord('G/Cmaj', 48, [0, 7, 11, 14, 19], 0.34),
    ],
    prompts: [
      'upper structure triads over jazz bass',
      'wide right hand arpeggios',
      'minimal kit and rim clicks',
      'sub bass pedal tones',
      'sparkling harmonics',
      'filtered chord swells',
    ],
  },
  {
    id: 'quartal-clusters',
    label: 'quartal clusters',
    bpm: 84,
    chords: [
      chord('Dq9', 50, [0, 5, 10, 15, 21], 0.36),
      chord('Gq13', 55, [0, 5, 10, 16, 21], 0.54),
      chord('CmajCluster', 48, [0, 2, 4, 7, 11, 18], 0.48),
      chord('B7altCluster', 47, [0, 3, 4, 10, 13, 20], 0.78),
      chord('Eq11', 52, [0, 5, 10, 15, 22], 0.42),
      chord('A13sus', 45, [0, 5, 10, 14, 21], 0.58),
      chord('Fmaj9#11', 53, [0, 4, 7, 11, 14, 18], 0.38),
      chord('Bb7#11', 46, [0, 4, 10, 18], 0.64),
    ],
    prompts: [
      'dense quartal jazz clusters',
      'slow cascading arpeggios',
      'half time acoustic drums',
      'upright bass glissandi',
      'dark granular room',
      'breathy synth pad',
    ],
  },
];

function chord(symbol: string, root: number, intervals: number[], color: number): ChordPlan {
  return {symbol, root, intervals, color};
}

function noteName(midi: number): string {
  const name = NOTE_NAMES[((midi % 12) + 12) % 12];
  const octave = Math.floor(midi / 12) - 1;
  return `${name}${octave}`;
}

function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}

function pickArpNote(chordPlan: ChordPlan, mode: ArpMode, stepInChord: number, rand: () => number): number {
  const notes = chordPlan.intervals.map(interval => chordPlan.root + interval);
  const extended = [...notes, ...notes.map(note => note + 12)];
  if (mode === 'down') return extended.slice().reverse()[stepInChord % extended.length];
  if (mode === 'random') return extended[Math.floor(rand() * extended.length)];
  if (mode === 'outside') {
    const base = extended[stepInChord % extended.length];
    return base + (stepInChord % 2 === 0 ? 1 : -1);
  }
  if (mode === 'skip') return extended[(stepInChord * 2) % extended.length];
  if (mode === 'euclidean') return stepInChord % 3 === 0 ? extended[(stepInChord + 4) % extended.length] : extended[stepInChord % extended.length];
  if (mode === 'polymetric') return extended[(stepInChord * 5 + 1) % extended.length];
  return extended[stepInChord % extended.length];
}

export function recipeById(recipeId: RecipeId): Recipe {
  return RECIPES.find(recipe => recipe.id === recipeId) ?? RECIPES[0];
}

export function buildJazzTensionPattern(recipeId: RecipeId, variation = 0): SequencerPattern {
  const recipe = recipeById(recipeId);
  const rand = seeded(hashRecipe(recipeId) + variation * 9973);
  const steps: SequencerStep[] = [];
  const stepsPerChord = 4;

  recipe.chords.forEach((chordPlan, chordIndex) => {
    const arpMode = arpModes[(chordIndex + variation) % arpModes.length];
    for (let localStep = 0; localStep < stepsPerChord; localStep += 1) {
      const index = chordIndex * stepsPerChord + localStep;
      const midiNote = pickArpNote(chordPlan, arpMode, localStep + variation, rand);
      const activityRoll = rand();
      steps.push({
        active: activityRoll > 0.16 || localStep === 0,
        index,
        bar: chordIndex + 1,
        beat: localStep + 1,
        chord: chordPlan.symbol,
        note: noteName(midiNote),
        midiNote,
        chordMidi: chordPlan.intervals.map(interval => chordPlan.root + interval),
        gate: Number((0.38 + rand() * 0.52).toFixed(2)),
        probability: Number((0.72 + rand() * 0.28).toFixed(2)),
        weight: Number((0.18 + chordPlan.color * 0.65 + rand() * 0.16).toFixed(2)),
        cfg: Number((0.9 + chordPlan.color * 4.0 + rand() * 0.42).toFixed(2)),
        temp: Number((0.72 + rand() * 0.68).toFixed(2)),
        arpMode,
      });
    }
  });

  return {
    name: recipe.label,
    recipeId,
    bpm: recipe.bpm,
    bars: 8,
    stepsPerBar: 4,
    prompts: recipe.prompts.map((text, index) => ({
      text,
      weight: index === 0 ? 1 : Number((0.15 + (index % 3) * 0.16).toFixed(2)),
    })),
    steps,
  };
}

function hashRecipe(recipeId: RecipeId): number {
  return recipeId.split('').reduce((hash, char) => hash + char.charCodeAt(0), 0);
}
