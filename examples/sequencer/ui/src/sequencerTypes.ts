export type ArpMode = 'up' | 'down' | 'random' | 'outside' | 'skip' | 'euclidean' | 'polymetric';

export type RecipeId =
  | 'ii-v-substitutions'
  | 'modal-interchange'
  | 'diminished-passing'
  | 'tritone-cycle'
  | 'upper-structure'
  | 'quartal-clusters';

export type PromptSlot = {
  text: string;
  weight: number;
};

export type SequencerStep = {
  active: boolean;
  index: number;
  bar: number;
  beat: number;
  chord: string;
  note: string;
  midiNote: number;
  chordMidi: number[];
  gate: number;
  probability: number;
  weight: number;
  cfg: number;
  temp: number;
  arpMode: ArpMode;
};

export type SequencerPattern = {
  name: string;
  recipeId: RecipeId;
  bpm: number;
  bars: number;
  stepsPerBar: number;
  prompts: PromptSlot[];
  steps: SequencerStep[];
};
