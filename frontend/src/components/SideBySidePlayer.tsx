import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  Loader2,
  Maximize2,
  Minimize2,
  Eye,
  EyeOff,
} from 'lucide-react';
import { API_BASE_URL } from '../api';
import { usePlayback } from '../hooks/usePlayback';

export interface SideBySidePlayerProps {
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

export const SideBySidePlayer: React.FC<SideBySidePlayerProps> = ({
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

  // Mask display toggle state (default: ON)
  const [showMask, setShowMask] = useState<boolean>(true);

  // Readiness state for both video elements
  const [isLeftReady, setIsLeftReady] = useState<boolean>(false);
  const [isRightReady, setIsRightReady] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  // References to video elements and main container
  const leftVideoRef = useRef<HTMLVideoElement>(null);
  const rightVideoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Determine actual sources based on showMask toggle
  // When showMask is OFF, swap both panels to raw.mp4
  const leftSource = showMask ? resolvedBaseline : resolvedRaw;
  const rightSource = showMask ? resolvedAmplified : resolvedRaw;

  const isBothReady = isLeftReady && isRightReady;

  // Register master video (Left panel: baseline or raw) with usePlayback
  useEffect(() => {
    const el = leftVideoRef.current;
    if (el) {
      playback.registerVideo('side-master', el);
      el.currentTime = playback.currentTime;
      el.playbackRate = playback.playbackSpeed;
      if (playback.isPlaying) {
        el.play().catch(() => {});
      }
    }
    return () => {
      playback.unregisterVideo('side-master');
    };
  }, [leftSource, playback]);

  // Register slave video (Right panel: amplified or raw) with usePlayback
  useEffect(() => {
    const el = rightVideoRef.current;
    if (el) {
      playback.registerVideo('side-slave', el);
      el.currentTime = playback.currentTime;
      el.playbackRate = playback.playbackSpeed;
      if (playback.isPlaying) {
        el.play().catch(() => {});
      }
    }
    return () => {
      playback.unregisterVideo('side-slave');
    };
  }, [rightSource, playback]);

  // Toggle Show Mask while strictly preserving playback timeline position
  const handleToggleShowMask = useCallback(() => {
    const targetTime = playback.currentTime;
    setShowMask((prev) => !prev);
    setIsLeftReady(false);
    setIsRightReady(false);

    // Apply preserved time to video elements as soon as they mount new src
    requestAnimationFrame(() => {
      if (leftVideoRef.current) {
        leftVideoRef.current.currentTime = targetTime;
      }
      if (rightVideoRef.current) {
        rightVideoRef.current.currentTime = targetTime;
      }
    });
  }, [playback.currentTime]);

  // Handle timeline scrubber interaction
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

  // Toggle Fullscreen mode
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

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Two equal-width panels in a responsive grid (stacks vertically on <700px) */}
      <div className="relative">
        {/* Loading overlay until both videos can play */}
        {!isBothReady && (
          <div className="absolute inset-0 z-30 bg-dark-950/80 backdrop-blur-sm rounded-2xl flex flex-col items-center justify-center gap-3 transition-opacity">
            <Loader2 size={32} className="text-cyan-400 animate-spin" />
            <span className="text-xs font-mono text-slate-300">
              Synchronizing video streams...
            </span>
          </div>
        )}

        <div className="side-by-side-grid">
          {/* Left Panel: Baseline (or Raw when mask toggled off) */}
          <div className="glass-card rounded-2xl p-3 sm:p-4 border border-slate-800 flex flex-col space-y-2.5">
            <div className="flex items-center justify-between px-1">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-500 shadow-sm shadow-amber-500/50" />
                <h4 className="text-xs sm:text-sm font-semibold text-slate-200">
                  Baseline + mask <span className="text-[11px] font-normal text-amber-400 font-mono">(orange mask)</span>
                </h4>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                {showMask ? 'baseline.mp4' : 'raw.mp4'}
              </span>
            </div>

            {/* Video container preserving aspect ratio */}
            <div className="relative w-full aspect-video rounded-xl overflow-hidden bg-black/90 flex items-center justify-center border border-slate-800/80">
              <video
                ref={leftVideoRef}
                src={leftSource}
                muted
                autoPlay
                loop
                playsInline
                preload="auto"
                onLoadedMetadata={(e) => {
                  const vid = e.currentTarget;
                  vid.currentTime = playback.currentTime;
                  vid.playbackRate = playback.playbackSpeed;
                  if (playback.isPlaying) {
                    vid.play().catch(() => {});
                  }
                }}
                onCanPlay={(e) => {
                  const vid = e.currentTarget;
                  if (Math.abs(vid.currentTime - playback.currentTime) > 0.05) {
                    vid.currentTime = playback.currentTime;
                  }
                  setIsLeftReady(true);
                  if (playback.isPlaying && vid.paused) {
                    vid.play().catch(() => {});
                  }
                }}
                className="w-full h-full object-contain pointer-events-none"
              />

              <div className="absolute top-2.5 left-2.5 z-10 pointer-events-none">
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-slate-950/80 text-amber-300 border border-amber-600/50 backdrop-blur-md shadow-md">
                  {showMask ? 'Baseline + mask' : 'Raw footage'}
                </span>
              </div>
            </div>
          </div>

          {/* Right Panel: Motion-amplified + mask (or Raw when mask toggled off) */}
          <div className="glass-card rounded-2xl p-3 sm:p-4 border border-slate-800 flex flex-col space-y-2.5">
            <div className="flex items-center justify-between px-1">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-sm shadow-red-500/50" />
                <h4 className="text-xs sm:text-sm font-semibold text-slate-200 flex items-center gap-1.5">
                  <Sparkles size={13} className="text-cyan-400" />
                  <span>Motion-amplified + mask</span>
                  <span className="text-[11px] font-normal text-rose-400 font-mono">(red mask)</span>
                </h4>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950/60 text-cyan-400 border border-cyan-800/60">
                {showMask ? 'amplified.mp4' : 'raw.mp4'}
              </span>
            </div>

            {/* Video container preserving aspect ratio */}
            <div className="relative w-full aspect-video rounded-xl overflow-hidden bg-black/90 flex items-center justify-center border border-slate-800/80">
              <video
                ref={rightVideoRef}
                src={rightSource}
                muted
                autoPlay
                loop
                playsInline
                preload="auto"
                onLoadedMetadata={(e) => {
                  const vid = e.currentTarget;
                  vid.currentTime = playback.currentTime;
                  vid.playbackRate = playback.playbackSpeed;
                  if (playback.isPlaying) {
                    vid.play().catch(() => {});
                  }
                }}
                onCanPlay={(e) => {
                  const vid = e.currentTarget;
                  if (Math.abs(vid.currentTime - playback.currentTime) > 0.05) {
                    vid.currentTime = playback.currentTime;
                  }
                  setIsRightReady(true);
                  if (playback.isPlaying && vid.paused) {
                    vid.play().catch(() => {});
                  }
                }}
                className="w-full h-full object-contain pointer-events-none"
              />

              <div className="absolute top-2.5 left-2.5 z-10 pointer-events-none">
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-slate-950/80 text-rose-300 border border-rose-600/50 backdrop-blur-md shadow-md">
                  {showMask ? 'Amplified overlay' : 'Raw footage'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Legend & Mask Toggle Row */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-2 py-1 text-xs">
        {/* Swatch Legend */}
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-amber-500 border border-amber-300 shadow-sm shadow-amber-500/50 flex-shrink-0" />
            <span className="text-slate-300 font-medium">baseline mask</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-red-500 border border-red-300 shadow-sm shadow-red-500/50 flex-shrink-0" />
            <span className="text-slate-300 font-medium">motion-amplified mask</span>
          </div>
        </div>

        {/* Show Mask Toggle */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleToggleShowMask}
            className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all cursor-pointer ${
              showMask
                ? 'bg-cyan-950/50 border-cyan-500/40 text-cyan-300 shadow-sm shadow-cyan-500/10'
                : 'bg-slate-800/80 border-slate-700 text-slate-400 hover:text-slate-200'
            }`}
          >
            {showMask ? <Eye size={14} className="text-cyan-400" /> : <EyeOff size={14} />}
            <span>Show mask:</span>
            <span className="font-mono font-bold">{showMask ? 'ON' : 'OFF'}</span>
          </button>
        </div>
      </div>

      {/* Shared Control Bar */}
      <div className="bg-dark-900/90 rounded-2xl p-4 sm:p-5 border border-slate-800 space-y-3.5">
        {/* Timeline Scrubber */}
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

        {/* Controls Row: Play/Pause, Time, Speed Selector, Fullscreen */}
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
              {playback.isPlaying ? 'Playing synchronized' : 'Paused'}
            </div>
          </div>

          {/* Right Controls: Shared Playback Speed (0.25x, 0.5x, 1x) & Fullscreen */}
          <div className="flex items-center gap-3">
            {/* Speed Selector Pills */}
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
