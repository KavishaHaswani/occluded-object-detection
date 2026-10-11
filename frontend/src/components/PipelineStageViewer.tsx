import React, { useState, useEffect, useRef } from 'react';
import {
  Film,
  Zap,
  Activity,
  Scan,
  Sliders,
  Play,
  Pause,
  RotateCcw,
  Layers,
  Sparkles,
  Info,
  LucideIcon,
} from 'lucide-react';
import { usePlayback } from '../hooks/usePlayback';
import { API_BASE_URL } from '../api';

export interface PipelineStageViewerProps {
  outputs?: {
    raw: string;
    baseline: string;
    amplified: string;
    amplified_frames?: string;
    flow?: string;
  };
}

interface StageInfo {
  id: number;
  label: string;
  subtitle: string;
  streamKey: 'raw' | 'amplified_frames' | 'flow' | 'amplified';
  icon: LucideIcon;
  captionTitle: string;
  captionDescription: string;
}

const STAGES: StageInfo[] = [
  {
    id: 1,
    label: '1. Raw video',
    subtitle: 'Original capture',
    streamKey: 'raw',
    icon: Film,
    captionTitle: 'Original Camera Capture',
    captionDescription:
      'The raw input footage before processing. The concealed target blends with ambient textures, shadows, and background colors, evading conventional single-frame vision detectors.',
  },
  {
    id: 2,
    label: '2. Motion-amplified',
    subtitle: 'Eulerian micro-motion',
    streamKey: 'amplified_frames',
    icon: Zap,
    captionTitle: 'Eulerian Temporal Magnification',
    captionDescription:
      'Tiny movements are magnified so the camouflaged object starts to shimmer. Multi-scale Laplacian decomposition and temporal Butterworth filtering exaggerate micro-motion like breathing, muscle tremor, and subtle crawling.',
  },
  {
    id: 3,
    label: '3. Motion map (optical flow)',
    subtitle: 'Dense velocity vectors',
    streamKey: 'flow',
    icon: Activity,
    captionTitle: 'Dense Optical Flow Heatmap',
    captionDescription:
      'The magnified micro-motion is transformed into dense velocity vectors. Direction and velocity become vibrant color gradients, exposing unmistakable silhouettes that camouflage cannot hide.',
  },
  {
    id: 4,
    label: '4. Detection mask',
    subtitle: 'Two-stream gated fusion',
    streamKey: 'amplified',
    icon: Scan,
    captionTitle: 'Neural Segmentation Mask',
    captionDescription:
      'Our two-stream neural network fuses spatial appearance with magnified motion vectors. Gated cross-modal attention breaks camouflage to predict the final delineated segmentation mask.',
  },
];

export const PipelineStageViewer: React.FC<PipelineStageViewerProps> = ({ outputs }) => {
  const playback = usePlayback();

  // Helper to ensure full URL path
  const resolveUrl = (path?: string) => {
    if (!path) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) return path;
    return `${API_BASE_URL}${path}`;
  };

  const rawUrl = resolveUrl(outputs?.raw);
  const ampFramesUrl = resolveUrl(outputs?.amplified_frames || outputs?.amplified);
  const flowUrl = resolveUrl(outputs?.flow);
  const maskUrl = resolveUrl(outputs?.amplified);

  // Active stage tab (1, 2, 3, or 4)
  const [activeStage, setActiveStage] = useState<number>(1);

  // Optical flow overlay opacity percentage on raw video (0 to 100)
  const [flowOpacity, setFlowOpacity] = useState<number>(75);

  // Video refs
  const stageVideoRef = useRef<HTMLVideoElement>(null);
  const underlayRawVideoRef = useRef<HTMLVideoElement>(null);

  // Determine current active source
  const getActiveSource = () => {
    switch (activeStage) {
      case 1:
        return rawUrl;
      case 2:
        return ampFramesUrl;
      case 3:
        return flowUrl;
      case 4:
        return maskUrl;
      default:
        return rawUrl;
    }
  };

  const currentSource = getActiveSource();

  // Register stage video with usePlayback hook
  useEffect(() => {
    const video = stageVideoRef.current;
    if (video) {
      playback.registerVideo(`stage-viewer-${activeStage}`, video);
      video.currentTime = playback.currentTime;
      video.playbackRate = playback.playbackSpeed;
      if (playback.isPlaying) video.play().catch(() => {});
    }
    return () => {
      playback.unregisterVideo(`stage-viewer-${activeStage}`);
    };
  }, [activeStage, currentSource, playback]);

  // When stage 3 (flow) is active, register underlay raw video as well
  useEffect(() => {
    const underlay = underlayRawVideoRef.current;
    if (activeStage === 3 && underlay) {
      playback.registerVideo('stage-viewer-underlay-raw', underlay);
      underlay.currentTime = playback.currentTime;
      underlay.playbackRate = playback.playbackSpeed;
      if (playback.isPlaying) underlay.play().catch(() => {});
    }
    return () => {
      playback.unregisterVideo('stage-viewer-underlay-raw');
    };
  }, [activeStage, rawUrl, playback]);

  const currentStageInfo = STAGES.find((s) => s.id === activeStage) || STAGES[0];

  return (
    <div className="glass-card rounded-2xl p-6 sm:p-8 space-y-7 border border-slate-800">
      {/* Header & Tabs */}
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-white tracking-wide flex items-center gap-2">
              <Layers size={18} className="text-cyan-400" />
              Pipeline Stage Breakdown
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Inspect how the vision pipeline extracts motion and isolates camouflaged boundaries
            </p>
          </div>

          <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/60 px-3 py-1 rounded-full border border-cyan-800/40 self-start sm:self-auto">
            Synchronized with Master Player
          </span>
        </div>

        {/* 4 Stage Tabs */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 p-1.5 rounded-xl bg-dark-900/90 border border-slate-800/90">
          {STAGES.map((stage) => {
            const Icon = stage.icon;
            const isActive = activeStage === stage.id;
            return (
              <button
                key={stage.id}
                type="button"
                onClick={() => setActiveStage(stage.id)}
                className={`p-3 rounded-lg flex flex-col items-start gap-1 text-left transition-all cursor-pointer ${
                  isActive
                    ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20 font-semibold'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <div className="flex items-center gap-1.5 text-xs">
                  <Icon size={14} className={isActive ? 'text-slate-950' : 'text-cyan-400'} />
                  <span>{stage.label}</span>
                </div>
                <span
                  className={`text-[10px] font-mono leading-tight ${
                    isActive ? 'text-slate-900/80 font-medium' : 'text-slate-500'
                  }`}
                >
                  {stage.subtitle}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Video Screen Area */}
      <div className="space-y-3">
        <div className="relative w-full aspect-video bg-black rounded-2xl overflow-hidden border border-slate-800 shadow-xl">
          {/* If Stage 3: Render Raw Video as Underlay */}
          {activeStage === 3 && (
            <video
              ref={underlayRawVideoRef}
              src={rawUrl}
              muted
              autoPlay
              loop
              playsInline
              preload="auto"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            />
          )}

          {/* Active Stage Video */}
          <video
            ref={stageVideoRef}
            src={currentSource}
            muted
            autoPlay
            loop
            playsInline
            preload="auto"
            style={{
              opacity: activeStage === 3 ? flowOpacity / 100 : 1.0,
            }}
            className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-150"
          />

          {/* Top-Right Badge */}
          <div className="absolute top-3.5 right-3.5 z-10 pointer-events-none">
            <span className="px-3 py-1 rounded-full text-[11px] font-mono font-medium bg-slate-950/80 text-cyan-300 border border-slate-700/60 backdrop-blur-md shadow-md flex items-center gap-1.5">
              <Sparkles size={11} className="text-cyan-400" />
              Stage {activeStage} of 4
            </span>
          </div>

          {/* If Stage 3: Floating Indicator showing flow blend state */}
          {activeStage === 3 && (
            <div className="absolute bottom-3.5 left-3.5 z-10 pointer-events-none">
              <span className="px-2.5 py-1 rounded-md text-[10px] font-mono bg-black/75 text-slate-300 border border-slate-700 backdrop-blur-md">
                Flow Overlay: {flowOpacity}%
              </span>
            </div>
          )}
        </div>

        {/* Heatmap Overlay Opacity Slider (visible on Tab 3) */}
        {activeStage === 3 && (
          <div className="p-3.5 rounded-xl bg-dark-900/80 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Sliders size={15} className="text-cyan-400 flex-shrink-0" />
              <div>
                <span className="text-xs font-semibold text-slate-200">
                  Heatmap Overlay Opacity:
                </span>
                <span className="text-xs font-mono font-bold text-cyan-400 ml-1.5">
                  {flowOpacity}%
                </span>
                <p className="text-[11px] text-slate-400">
                  Blends optical motion vectors over the raw video footage
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3 w-full sm:w-64">
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={flowOpacity}
                onChange={(e) => setFlowOpacity(parseInt(e.target.value))}
                className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                aria-label="Heatmap overlay opacity"
              />
              <div className="flex items-center gap-1">
                {[0, 50, 100].map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => setFlowOpacity(preset)}
                    className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors cursor-pointer ${
                      flowOpacity === preset
                        ? 'bg-cyan-500 text-slate-950 font-bold'
                        : 'bg-slate-800 text-slate-400 hover:text-white'
                    }`}
                  >
                    {preset}%
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Shared Mini Timeline Bar */}
        <div className="flex items-center justify-between gap-4 p-3 rounded-xl bg-dark-900/70 border border-slate-800/80 text-xs font-mono">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={playback.togglePlay}
              className="w-7 h-7 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 flex items-center justify-center transition-colors cursor-pointer"
              aria-label={playback.isPlaying ? 'Pause' : 'Play'}
            >
              {playback.isPlaying ? (
                <Pause size={13} className="fill-current" />
              ) : (
                <Play size={13} className="fill-current ml-0.5" />
              )}
            </button>
            <button
              type="button"
              onClick={playback.restart}
              className="w-7 h-7 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center justify-center transition-colors cursor-pointer"
              aria-label="Restart"
            >
              <RotateCcw size={13} />
            </button>
            <span className="text-slate-400 pl-1">
              {playback.formatTime(playback.currentTime)} / {playback.formatTime(playback.duration)}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-slate-500 text-[11px] hidden sm:inline">Speed:</span>
            {[0.25, 0.5, 1.0].map((speed) => (
              <button
                key={speed}
                type="button"
                onClick={() => playback.setPlaybackSpeed(speed)}
                className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold transition-all cursor-pointer ${
                  playback.playbackSpeed === speed
                    ? 'bg-cyan-500 text-slate-950'
                    : 'text-slate-400 hover:text-white bg-slate-800/60'
                }`}
              >
                {speed}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Non-Expert Caption Card */}
      <div className="p-4 rounded-xl bg-dark-900/90 border border-slate-800 space-y-1.5">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-200">
          <Info size={14} className="text-cyan-400 flex-shrink-0" />
          <span>Stage {activeStage}: {currentStageInfo.captionTitle}</span>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed pl-5">
          {currentStageInfo.captionDescription}
        </p>
      </div>

      {/* SVG Pipeline Architecture Diagram Section */}
      <div className="space-y-3 pt-2 border-t border-slate-800/80">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-slate-300 uppercase tracking-wider text-[11px]">
            Pipeline Architecture Flow
          </span>
          <span className="text-slate-500 text-[11px] font-mono">
            Click any block to switch preview
          </span>
        </div>

        <div className="p-4 rounded-xl bg-dark-900/60 border border-slate-800 overflow-x-auto">
          <svg
            viewBox="0 0 820 130"
            className="w-full min-w-[700px] h-auto select-none"
            style={{ maxHeight: '150px' }}
          >
            <defs>
              <marker
                id="arrow"
                viewBox="0 0 10 10"
                refX="7"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 9 5 L 0 9 z" fill="#06b6d4" />
              </marker>

              <marker
                id="arrow-muted"
                viewBox="0 0 10 10"
                refX="7"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 9 5 L 0 9 z" fill="#475569" />
              </marker>

              <filter id="glow-cyan" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor="#06b6d4" floodOpacity="0.6" />
              </filter>
            </defs>

            {/* Connecting Lines / Arrows */}
            {/* Raw -> Amplification */}
            <line
              x1="125"
              y1="40"
              x2="175"
              y2="40"
              stroke={activeStage === 2 ? '#22d3ee' : '#334155'}
              strokeWidth="2"
              markerEnd={activeStage === 2 ? 'url(#arrow)' : 'url(#arrow-muted)'}
            />

            {/* Amplification -> Optical Flow */}
            <line
              x1="315"
              y1="40"
              x2="365"
              y2="40"
              stroke={activeStage === 3 ? '#22d3ee' : '#334155'}
              strokeWidth="2"
              markerEnd={activeStage === 3 ? 'url(#arrow)' : 'url(#arrow-muted)'}
            />

            {/* Optical Flow -> Two-Stream Network (Motion branch) */}
            <line
              x1="495"
              y1="40"
              x2="535"
              y2="48"
              stroke={activeStage === 4 ? '#22d3ee' : '#334155'}
              strokeWidth="2"
              markerEnd={activeStage === 4 ? 'url(#arrow)' : 'url(#arrow-muted)'}
            />

            {/* Raw Video -> Two-Stream Network (Appearance branch - Curved path) */}
            <path
              d="M 65 65 C 65 110, 500 115, 535 78"
              fill="none"
              stroke={activeStage === 4 ? '#22d3ee' : '#334155'}
              strokeWidth="1.7"
              strokeDasharray={activeStage === 4 ? 'none' : '4 3'}
              markerEnd={activeStage === 4 ? 'url(#arrow)' : 'url(#arrow-muted)'}
            />
            <text x="270" y="108" fill="#64748b" fontSize="10" fontFamily="monospace">
              Appearance stream (spatial context)
            </text>

            {/* Two-Stream Network -> Output Mask */}
            <line
              x1="675"
              y1="55"
              x2="715"
              y2="55"
              stroke={activeStage === 4 ? '#22d3ee' : '#334155'}
              strokeWidth="2"
              markerEnd={activeStage === 4 ? 'url(#arrow)' : 'url(#arrow-muted)'}
            />

            {/* Node 1: Raw Video */}
            <g
              onClick={() => setActiveStage(1)}
              className="cursor-pointer group"
              filter={activeStage === 1 ? 'url(#glow-cyan)' : undefined}
            >
              <rect
                x="15"
                y="15"
                width="110"
                height="50"
                rx="8"
                fill={activeStage === 1 ? '#083344' : '#0f172a'}
                stroke={activeStage === 1 ? '#22d3ee' : '#334155'}
                strokeWidth={activeStage === 1 ? '2' : '1'}
              />
              <text
                x="70"
                y="37"
                textAnchor="middle"
                fill={activeStage === 1 ? '#ffffff' : '#e2e8f0'}
                fontSize="12"
                fontWeight="600"
              >
                Raw Video
              </text>
              <text
                x="70"
                y="52"
                textAnchor="middle"
                fill={activeStage === 1 ? '#38bdf8' : '#64748b'}
                fontSize="10"
                fontFamily="monospace"
              >
                1. Input footage
              </text>
            </g>

            {/* Node 2: Eulerian Amplification */}
            <g
              onClick={() => setActiveStage(2)}
              className="cursor-pointer group"
              filter={activeStage === 2 ? 'url(#glow-cyan)' : undefined}
            >
              <rect
                x="175"
                y="15"
                width="140"
                height="50"
                rx="8"
                fill={activeStage === 2 ? '#083344' : '#0f172a'}
                stroke={activeStage === 2 ? '#22d3ee' : '#334155'}
                strokeWidth={activeStage === 2 ? '2' : '1'}
              />
              <text
                x="245"
                y="37"
                textAnchor="middle"
                fill={activeStage === 2 ? '#ffffff' : '#e2e8f0'}
                fontSize="12"
                fontWeight="600"
              >
                Amplification
              </text>
              <text
                x="245"
                y="52"
                textAnchor="middle"
                fill={activeStage === 2 ? '#38bdf8' : '#64748b'}
                fontSize="10"
                fontFamily="monospace"
              >
                2. Eulerian factor &alpha;
              </text>
            </g>

            {/* Node 3: Optical Flow */}
            <g
              onClick={() => setActiveStage(3)}
              className="cursor-pointer group"
              filter={activeStage === 3 ? 'url(#glow-cyan)' : undefined}
            >
              <rect
                x="365"
                y="15"
                width="130"
                height="50"
                rx="8"
                fill={activeStage === 3 ? '#083344' : '#0f172a'}
                stroke={activeStage === 3 ? '#22d3ee' : '#334155'}
                strokeWidth={activeStage === 3 ? '2' : '1'}
              />
              <text
                x="430"
                y="37"
                textAnchor="middle"
                fill={activeStage === 3 ? '#ffffff' : '#e2e8f0'}
                fontSize="12"
                fontWeight="600"
              >
                Optical Flow
              </text>
              <text
                x="430"
                y="52"
                textAnchor="middle"
                fill={activeStage === 3 ? '#38bdf8' : '#64748b'}
                fontSize="10"
                fontFamily="monospace"
              >
                3. Motion vectors
              </text>
            </g>

            {/* Node 4: Two-Stream Network */}
            <g
              onClick={() => setActiveStage(4)}
              className="cursor-pointer group"
              filter={activeStage === 4 ? 'url(#glow-cyan)' : undefined}
            >
              <rect
                x="535"
                y="20"
                width="140"
                height="65"
                rx="8"
                fill={activeStage === 4 ? '#083344' : '#0f172a'}
                stroke={activeStage === 4 ? '#22d3ee' : '#334155'}
                strokeWidth={activeStage === 4 ? '2' : '1'}
              />
              <text
                x="605"
                y="45"
                textAnchor="middle"
                fill={activeStage === 4 ? '#ffffff' : '#e2e8f0'}
                fontSize="12"
                fontWeight="600"
              >
                Two-Stream NN
              </text>
              <text
                x="605"
                y="60"
                textAnchor="middle"
                fill={activeStage === 4 ? '#38bdf8' : '#64748b'}
                fontSize="10"
                fontFamily="monospace"
              >
                Cross-modal fusion
              </text>
            </g>

            {/* Node 5: Detection Mask */}
            <g
              onClick={() => setActiveStage(4)}
              className="cursor-pointer group"
              filter={activeStage === 4 ? 'url(#glow-cyan)' : undefined}
            >
              <rect
                x="715"
                y="25"
                width="95"
                height="55"
                rx="8"
                fill={activeStage === 4 ? '#064e3b' : '#0f172a'}
                stroke={activeStage === 4 ? '#34d399' : '#334155'}
                strokeWidth={activeStage === 4 ? '2' : '1'}
              />
              <text
                x="762"
                y="48"
                textAnchor="middle"
                fill={activeStage === 4 ? '#34d399' : '#e2e8f0'}
                fontSize="12"
                fontWeight="bold"
              >
                Mask
              </text>
              <text
                x="762"
                y="63"
                textAnchor="middle"
                fill={activeStage === 4 ? '#6ee7b7' : '#64748b'}
                fontSize="10"
                fontFamily="monospace"
              >
                4. Segmented
              </text>
            </g>
          </svg>
        </div>
      </div>
    </div>
  );
};
