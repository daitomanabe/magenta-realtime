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

type Step = {
  active: boolean;
  note: string;
  chord: string;
  weight: number;
  cfg: number;
  temp: number;
};

const chordCycle = [
  'Dm9',
  'G13b9',
  'Cmaj9#11',
  'F#7alt',
  'Fm11',
  'Bb13',
  'Em7b5',
  'A7#9b13',
];

const notes = ['D3', 'F3', 'A3', 'C4', 'E4', 'G4', 'B4', 'Db5'];

function buildSteps(): Step[] {
  return Array.from({length: 32}, (_, index) => ({
    active: index % 2 === 0 || index % 7 === 0,
    note: notes[index % notes.length],
    chord: chordCycle[Math.floor(index / 4) % chordCycle.length],
    weight: index % 8 === 0 ? 0.95 : 0.35 + ((index % 5) * 0.11),
    cfg: 1.2 + ((index % 6) * 0.42),
    temp: 0.8 + ((index % 4) * 0.18),
  }));
}

const promptSlots = [
  'complex jazzy tension chords, warm electric piano',
  'fast glassy arpeggio figures',
  'dry broken beat drums, tight room',
  'sub bass following altered dominants',
  'granular tape texture and vinyl air',
  'muted guitar harmonics',
];

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
  const [host, setHost] = useState<HostState>({
    isPlaying: false,
    modelName: 'No model loaded',
    patternName: '8 Bar Tension Chords',
  });
  const [steps, setSteps] = useState(buildSteps);

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
        name: host.patternName ?? '8 Bar Tension Chords',
        bpm: 104,
        bars: 8,
        stepsPerBar: 16,
        prompts: promptSlots.map((text, index) => ({text, weight: index === 0 ? 1 : 0})),
        steps,
      },
    });
  }, [host.patternName, steps]);

  const toggleTransport = () => {
    const playing = !host.isPlaying;
    setHost(current => ({...current, isPlaying: playing}));
    post({type: 'sequencerTransport', playing});
  };

  const randomize = () => {
    setSteps(current =>
      current.map((step, index) => ({
        ...step,
        active: Math.random() > 0.34,
        weight: Number((0.15 + Math.random() * 0.85).toFixed(2)),
        cfg: Number((0.7 + Math.random() * 4.1).toFixed(2)),
        temp: Number((0.65 + Math.random() * 0.85).toFixed(2)),
        note: notes[(index + Math.floor(Math.random() * notes.length)) % notes.length],
      })),
    );
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
          <button className="iconButton" onClick={() => setSteps(buildSteps())} title="Reset pattern">
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

      <section className="overviewBand">
        <div className="metric">
          <Gauge size={18} />
          <strong>104</strong>
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
            {promptSlots.map((prompt, index) => (
              <div className="promptRow" key={prompt}>
                <span>{index + 1}</span>
                <p>{prompt}</p>
                <meter min={0} max={1} value={index === 0 ? 1 : steps[index * 3]?.weight ?? 0.2} />
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
