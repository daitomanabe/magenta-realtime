import {useEffect, useMemo, useState} from 'react';
import {
  AlertTriangle,
  Boxes,
  CircleStop,
  Download,
  Gauge,
  Play,
  RefreshCw,
  Shuffle,
  SlidersHorizontal,
  Square,
  Wand2,
} from 'lucide-react';
import {buildJazzTensionPattern, recipeById, RECIPES} from './jazzMaterials';
import type {RecipeId, SequencerStep} from './sequencerTypes';

declare global {
  interface Window {
    updateState?: (state: HostState) => void;
    webkit?: {
      messageHandlers?: {
        sequencerHost?: {
          postMessage: (msg: unknown) => void;
        };
      };
    };
  }
}

type HostState = {
  isPlaying?: boolean;
  modelName?: string;
  patternName?: string;
  nativeReady?: boolean;
  patternSynced?: boolean;
  renderStatus?: string;
  panicCount?: number;
};

const automationLanes = [
  ['CFG Notes', 'bar ramp', '0.8 - 4.6'],
  ['CFG MusicCoCa', 'scene morph', '1.0 - 5.0'],
  ['Temperature', 'random walk', '0.7 - 1.4'],
  ['Top-K', 'sample hold', '24 - 160'],
  ['Seed Rotation', 'step lock', '0 - 11'],
  ['Prompt Weights', 'IDW record', '6 slots'],
];

function post(msg: unknown) {
  window.webkit?.messageHandlers?.sequencerHost?.postMessage(msg);
}

export default function App() {
  const [recipeId, setRecipeId] = useState<RecipeId>('ii-v-substitutions');
  const [variation, setVariation] = useState(0);
  const pattern = useMemo(() => buildJazzTensionPattern(recipeId, variation), [recipeId, variation]);
  const [host, setHost] = useState<HostState>({
    isPlaying: false,
    modelName: 'No model loaded',
    patternName: recipeById(recipeId).label,
  });
  const [steps, setSteps] = useState<SequencerStep[]>(pattern.steps);

  const activeCount = useMemo(() => steps.filter(step => step.active).length, [steps]);
  const avgCfg = useMemo(
    () => steps.reduce((sum, step) => sum + step.cfg, 0) / steps.length,
    [steps],
  );

  useEffect(() => {
    window.updateState = (state: HostState) => {
      setHost(current => ({...current, ...state}));
    };
    post({type: 'uiReady'});
  }, []);

  useEffect(() => {
    post({
      type: 'sequencerPattern',
      pattern: {
        name: pattern.name,
        recipeId: pattern.recipeId,
        bpm: pattern.bpm,
        bars: pattern.bars,
        stepsPerBar: pattern.stepsPerBar,
        prompts: pattern.prompts,
        steps,
      },
    });
  }, [pattern, steps]);

  const toggleTransport = () => {
    const playing = !host.isPlaying;
    setHost(current => ({...current, isPlaying: playing}));
    post({type: 'sequencerTransport', playing});
  };

  const applyPattern = (nextRecipeId: RecipeId, nextVariation: number) => {
    const nextPattern = buildJazzTensionPattern(nextRecipeId, nextVariation);
    setRecipeId(nextRecipeId);
    setVariation(nextVariation);
    setSteps(nextPattern.steps);
    setHost(current => ({...current, patternName: nextPattern.name}));
  };

  const randomize = () => {
    applyPattern(recipeId, variation + 1);
  };

  const toggleStep = (index: number) => {
    setSteps(current =>
      current.map((step, stepIndex) =>
        stepIndex === index ? {...step, active: !step.active} : step,
      ),
    );
  };

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand">
          <Boxes size={22} />
          <div>
            <h1>MRT2 Sequencer</h1>
            <span>{host.patternName}</span>
          </div>
        </div>
        <div className="transport">
          <button className="iconButton primary" onClick={toggleTransport} title="Play or stop">
            {host.isPlaying ? <Square size={18} /> : <Play size={18} />}
          </button>
          <button className="iconButton" onClick={() => post({type: 'sequencerTransport', playing: false})} title="Stop">
            <CircleStop size={18} />
          </button>
          <button className="iconButton danger" onClick={() => post({type: 'sequencerPanic'})} title="Panic">
            <AlertTriangle size={18} />
          </button>
          <button className="iconButton" onClick={randomize} title="Randomize variation">
            <Shuffle size={18} />
          </button>
          <button className="iconButton" onClick={() => applyPattern(recipeId, 0)} title="Reset pattern">
            <RefreshCw size={18} />
          </button>
          <button className="commandButton" onClick={() => post({type: 'sequencerRender'})}>
            <Download size={17} />
            Render 8 bars
          </button>
        </div>
        <div className="statusStrip">
          <span>{host.modelName}</span>
          <span>{host.isPlaying ? 'Playing' : 'Stopped'}</span>
          <span>{host.renderStatus ?? 'Render idle'}</span>
          <span>rev {__COMMIT_HASH__}</span>
        </div>
      </header>

      <section className="recipeBand">
        <label>
          <span>Recipe</span>
          <select
            value={recipeId}
            onChange={event => applyPattern(event.target.value as RecipeId, variation)}
          >
            {RECIPES.map(recipe => (
              <option key={recipe.id} value={recipe.id}>
                {recipe.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Variation</span>
          <input
            value={variation}
            type="number"
            min={0}
            onChange={event => applyPattern(recipeId, Math.max(0, Number(event.target.value) || 0))}
          />
        </label>
        <div>
          <strong>{pattern.name}</strong>
          <span>{pattern.steps.length} chord/arp events ready for native scheduling</span>
        </div>
      </section>

      <section className="overviewBand">
        <div className="metric">
          <Gauge size={18} />
          <strong>{pattern.bpm}</strong>
          <span>BPM</span>
        </div>
        <div className="metric">
          <strong>8</strong>
          <span>Bars</span>
        </div>
        <div className="metric">
          <strong>{activeCount}</strong>
          <span>Active steps</span>
        </div>
        <div className="metric">
          <strong>{avgCfg.toFixed(2)}</strong>
          <span>Avg CFG</span>
        </div>
      </section>

      <section className="workspace">
        <div className="sequencerPane">
          <div className="sectionHead">
            <h2>Chord And Arp Grid</h2>
            <span>32 visible steps / 8-bar pattern</span>
          </div>
          <div className="stepGrid">
            {steps.map((step, index) => (
              <button
                key={index}
                className={`step ${step.active ? 'active' : ''}`}
                onClick={() => toggleStep(index)}
              >
                <span className="stepIndex">{String(index + 1).padStart(2, '0')}</span>
                <strong>{step.chord}</strong>
                <span>{step.note}</span>
                <small>{step.arpMode}</small>
                <i style={{height: `${Math.max(12, step.weight * 58)}px`}} />
              </button>
            ))}
          </div>
        </div>

        <aside className="sidePane">
          <div className="sectionHead">
            <h2>Prompt Slots</h2>
            <Wand2 size={18} />
          </div>
          <div className="promptList">
            {pattern.prompts.map((prompt, index) => (
              <div className="promptRow" key={prompt.text}>
                <span>{index + 1}</span>
                <p>{prompt.text}</p>
                <meter min={0} max={1} value={prompt.weight} />
              </div>
            ))}
          </div>
        </aside>
      </section>

      <section className="automationBand">
        <div className="sectionHead">
          <h2>Modulation Lanes</h2>
          <SlidersHorizontal size={18} />
        </div>
        <div className="laneGrid">
          {automationLanes.map(([name, mode, range]) => (
            <div className="lane" key={name}>
              <strong>{name}</strong>
              <span>{mode}</span>
              <em>{range}</em>
              <div className="laneBars">
                {Array.from({length: 16}, (_, index) => (
                  <i key={index} style={{height: `${18 + ((index * 7 + name.length) % 42)}px`}} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
