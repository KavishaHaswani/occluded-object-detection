import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  Layers,
  Loader2,
  Maximize2,
  Minimize2,
  SlidersHorizontal,
} from 'lucide-react';
import { API_BASE_URL } from '../api';
import { usePlayback } from '../hooks/usePlayback';

export interface BeforeAfterSliderProps {
  outputs?: {
    raw: string;
    baseline: string;
    amplified: string;
    amplified_frames?: string;
    flow?: string;
  };
  baselineUrl?: string;
  amplifiedUrl?: string;
  rawUrl?: string;
}

export const BeforeAfterSlider: React.FC<BeforeAfterSliderProps> = ({
  outputs,
  baselineUrl,
  amplifiedUrl,
  rawUrl,
}) => {
  const playback = usePlayback();

  // Helper to ensure full URL path
  const resolveUrl = (path?: string) => {
    if (!path) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) return path;
    return `${API_BASE_URL}${path}`;
  };

  const resolvedBaseline = resolveUrl(outputs?.baseline || baselineUrl);
  const resolvedAmplified = resolveUrl(outputs?.amplified || amplifiedUrl);
  const resolvedRaw = resolveUrl(outputs?.raw || rawUrl);

  // View mode: 'baseline' (Baseline vs Amplified) or 'raw' (Raw vs Amplified)
  const [viewMode, setViewMode] = useState<'baseline' | 'raw'>('baseline');

  // Divider position percentage (0 to 100)
  const [dividerPos, setDividerPos] = useState<number>(50);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  // Loading state (show until both can play)
  const [isMasterReady, setIsMasterReady] = useState<boolean>(false);
  const [isSlaveReady, setIsSlaveReady] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  // Video refs: Master is bottom (baseline or raw), Slave is top (amplified)
  const masterVideoRef = useRef<HTMLVideoElement>(null);
  const slaveVideoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const bottomSource = viewMode === 'baseline' ? resolvedBaseline : resolvedRaw;
  const topSource = resolvedAmplified;

  const isBothReady = isMasterReady && isSlaveReady;

  // Register master video (bottom) with usePlayback
  useEffect(() => {
    const el = masterVideoRef.current;
    if (el) {
      playback.registerVideo('slider-master', el);
      el.currentTime = playback.currentTime;
      el.playbackRate = playback.playbackSpeed;
      if (playback.isPlaying) el.play().catch(() => {});
    }
    return () => {
      playback.unregisterVideo('slider-master');
    };
  }, [bottomSource, playback]);

  // Register slave video (top) with usePlayback
  useEffect(() => {
    const el = slaveVideoRef.current;
    if (el) {
      playback.registerVideo('slider-slave', el);
      el.currentTime = playback.currentTime;
      el.playbackRate = playback.playbackSpeed;
      if (playback.isPlaying) el.play().catch(() => {});
    }
    return () => {
      playback.unregisterVideo('slider-slave');
    };
  }, [topSource, playback]);

  // Shared Scrubber Seek
  const handleScrubberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = parseFloat(e.target.value);
    playback.seek(newTime);
  };

  const handleScrubberStart = () => {
    playback.isScrubbingRef.current = true;
  };

  const handleScrubberEnd = () => {
    playback.isScrubbingRef.current = false;
    if (playback.isPlaying) {
      playback.play();
    }
  };

  // Toggle fullscreen
  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    const onFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener('fullscreenchange', onFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', onFullscreenChange);
  }, []);

  // Update divider position from mouse or touch event
  const updateDividerFromClientX = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    if (rect.width <= 0) return;
    const offsetX = clientX - rect.left;
    const percent = Math.min(Math.max((offsetX / rect.width) * 100, 0), 100);
    setDividerPos(percent);
  }, []);

  // Mouse drag handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    updateDividerFromClientX(e.clientX);
  };

  useEffect(() => {
    const handleWindowMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      updateDividerFromClientX(e.clientX);
    };

    const handleWindowMouseUp = () => {
      if (isDragging) setIsDragging(false);
    };

    if (isDragging) {
      window.addEventListener('mousemove', handleWindowMouseMove);
      window.addEventListener('mouseup', handleWindowMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleWindowMouseMove);
      window.removeEventListener('mouseup', handleWindowMouseUp);
    };
  }, [isDragging, updateDividerFromClientX]);

  // Touch drag handlers
  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length > 0) {
      setIsDragging(true);
      updateDividerFromClientX(e.touches[0].clientX);
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (isDragging && e.touches.length > 0) {
      updateDividerFromClientX(e.touches[0].clientX);
    }
  };

  const handleTouchEnd = () => {
    setIsDragging(false);
  };

  // Keyboard navigation for divider (Arrow keys, Home, End)
  const handleDividerKeyDown = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 5 : 2;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
      e.preventDefault();
      setDividerPos((prev) => Math.max(0, prev - step));
    } else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
      e.preventDefault();
      setDividerPos((prev) => Math.min(100, prev + step));
    } else if (e.key === 'Home') {
      e.preventDefault();
      setDividerPos(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      setDividerPos(100);
    }
  };

  return (
    <div className="space-y-4">
      {/* Top Bar: Title & Toggle View Switch */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <SlidersHorizontal size={18} className="text-cyan-400" />
          <h3 className="text-base font-semibold text-white tracking-wide">
            Interactive Before / After Comparison
          </h3>
        </div>

        {/* "Toggle view" Switch: Baseline vs Raw */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-dark-900/90 border border-slate-800 self-start sm:self-auto">
          <span className="text-[11px] font-mono text-slate-400 px-2 flex items-center gap-1">
            <Layers size={12} className="text-cyan-400" />
            Layer:
          </span>
          <button
            type="button"
            onClick={() => setViewMode('baseline')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              viewMode === 'baseline'
                ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Baseline vs Amplified
          </button>
          <button
            type="button"
            onClick={() => setViewMode('raw')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              viewMode === 'raw'
                ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Raw vs Amplified
          </button>
        </div>
      </div>

      {/* Main Video Box with Stacked Videos & Clip-Path Divider */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        className={`relative w-full aspect-video bg-black rounded-2xl overflow-hidden select-none border border-slate-800 shadow-2xl group ${
          isDragging ? 'cursor-ew-resize' : 'cursor-default'
        }`}
      >
        {/* Loading Overlay */}
        {!isBothReady && (
          <div className="absolute inset-0 z-30 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3 text-slate-300">
            <Loader2 size={36} className="animate-spin text-cyan-400" />
            <div className="text-xs font-mono tracking-wide text-slate-400">
              Synchronizing video streams...
            </div>
          </div>
        )}

        {/* 1. Underneath Video: Baseline (or Raw) */}
        <video
          ref={masterVideoRef}
          src={bottomSource}
          muted
          autoPlay
          loop
          playsInline
          preload="auto"
          onCanPlay={() => setIsMasterReady(true)}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />

        {/* 2. Top Video: Amplified (Clipped from divider position to right edge) */}
        <video
          ref={slaveVideoRef}
          src={topSource}
          muted
          autoPlay
          loop
          playsInline
          preload="auto"
          onCanPlay={() => setIsSlaveReady(true)}
          style={{
            clipPath: `inset(0 0 0 ${dividerPos}%)`,
            WebkitClipPath: `inset(0 0 0 ${dividerPos}%)`,
          }}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />

        {/* Labels: Left ("Appearance only" or "Raw input") & Right ("Motion-amplified") */}
        <div className="absolute top-3.5 left-3.5 z-20 pointer-events-none">
          <span className="px-3 py-1 rounded-full text-[11px] font-mono font-semibold bg-slate-950/80 text-slate-200 border border-slate-700/60 backdrop-blur-md shadow-md">
            {viewMode === 'baseline' ? 'Appearance only' : 'Raw input'}
          </span>
        </div>

        <div className="absolute top-3.5 right-3.5 z-20 pointer-events-none">
          <span className="px-3 py-1 rounded-full text-[11px] font-mono font-semibold bg-cyan-950/80 text-cyan-300 border border-cyan-700/60 backdrop-blur-md shadow-md flex items-center gap-1.5">
            <Sparkles size={11} className="text-cyan-400" />
            Motion-amplified
          </span>
        </div>

        {/* Draggable Vertical Divider Line */}
        <div
          style={{ left: `${dividerPos}%` }}
          className="absolute top-0 bottom-0 z-20 pointer-events-none -translate-x-1/2 flex items-center justify-center"
        >
          {/* Vertical Glowing Line */}
          <div className="w-0.5 h-full bg-white shadow-[0_0_10px_rgba(255,255,255,0.7)]" />

          {/* Interactive Center Handle */}
          <div
            tabIndex={0}
            role="slider"
            aria-label="Before and after comparison divider"
            aria-valuenow={Math.round(dividerPos)}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuetext={`${Math.round(dividerPos)}% divider position`}
            onKeyDown={handleDividerKeyDown}
            className="absolute pointer-events-auto w-8 h-8 rounded-full bg-white/95 border-2 border-slate-900 shadow-xl flex items-center justify-center cursor-ew-resize hover:scale-110 active:scale-95 transition-transform focus:outline-none focus:ring-4 focus:ring-cyan-400/50"
          >
            <div className="flex items-center gap-0.5 text-slate-800 text-[10px] font-bold select-none">
              <span>&#9664;</span>
              <span>&#9654;</span>
            </div>
          </div>
        </div>
      </div>

      {/* Shared Control Bar */}
      <div className="bg-dark-900/90 rounded-2xl p-4 sm:p-5 border border-slate-800 space-y-3.5">
        {/* Scrubber Slider */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>{playback.formatTime(playback.currentTime)}</span>
            <span className="text-slate-500">{playback.formatTime(playback.duration)}</span>
          </div>

          <input
            type="range"
            min={0}
            max={playback.duration || 1}
            step={0.01}
            value={playback.currentTime}
            onChange={handleScrubberChange}
            onMouseDown={handleScrubberStart}
            onMouseUp={handleScrubberEnd}
            onTouchStart={handleScrubberStart}
            onTouchEnd={handleScrubberEnd}
            disabled={!isBothReady}
            aria-label="Video timeline scrubber"
            className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer transition-all"
          />
        </div>

        {/* Controls Row: Play/Pause, Restart, Speed Selector, Fullscreen */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-1 border-t border-slate-800/80">
          {/* Left Controls: Play/Pause & Restart */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={playback.togglePlay}
              disabled={!isBothReady}
              aria-label={playback.isPlaying ? 'Pause video' : 'Play video'}
              className="w-9 h-9 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 flex items-center justify-center shadow-md shadow-cyan-500/20 disabled:opacity-40 disabled:cursor-not-allowed transition-all cursor-pointer"
            >
              {playback.isPlaying ? (
                <Pause size={16} className="fill-current" />
              ) : (
                <Play size={16} className="fill-current ml-0.5" />
              )}
            </button>

            <button
              type="button"
              onClick={playback.restart}
              disabled={!isBothReady}
              aria-label="Restart video"
              className="w-9 h-9 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center justify-center transition-colors cursor-pointer"
            >
              <RotateCcw size={15} />
            </button>

            <div className="text-xs font-mono text-slate-400 pl-2 hidden sm:block">
              {playback.isPlaying ? 'Playing in lockstep' : 'Paused'}
            </div>
          </div>

          {/* Right Controls: Shared Playback Speed (0.25x, 0.5x, 1x) & Fullscreen */}
          <div className="flex items-center gap-3">
            {/* Speed Pills */}
            <div className="flex items-center gap-1 bg-slate-800/70 p-1 rounded-xl border border-slate-700/60">
              {[0.25, 0.5, 1.0].map((speed) => (
                <button
                  key={speed}
                  type="button"
                  onClick={() => playback.setPlaybackSpeed(speed)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-mono font-semibold transition-all cursor-pointer ${
                    playback.playbackSpeed === speed
                      ? 'bg-cyan-500 text-slate-950 shadow-sm'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {speed}x
                </button>
              ))}
            </div>

            {/* Fullscreen Button */}
            <button
              type="button"
              onClick={toggleFullscreen}
              aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
              className="w-9 h-9 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center justify-center transition-colors cursor-pointer"
            >
              {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
