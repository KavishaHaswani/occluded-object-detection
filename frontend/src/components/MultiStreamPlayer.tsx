import React, { useRef, useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Columns, Grid2X2, Maximize2, Split } from 'lucide-react';
import { PipelineVideos, VideoMetadata, ViewMode } from '../types';

interface MultiStreamPlayerProps {
  videos: PipelineVideos;
  metadata: VideoMetadata;
}

export const MultiStreamPlayer: React.FC<MultiStreamPlayerProps> = ({ videos, metadata }) => {
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [sliderPos, setSliderPos] = useState<number>(50); // percentage (0-100)
  const [isDraggingSlider, setIsDraggingSlider] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<ViewMode>('split');
  const [singleStreamKey, setSingleStreamKey] = useState<keyof PipelineVideos>('amplified');

  // Video refs for synchronization
  const baseVideoRef = useRef<HTMLVideoElement>(null);
  const ampVideoRef = useRef<HTMLVideoElement>(null);
  const rawVideoRef = useRef<HTMLVideoElement>(null);
  const ampFramesVideoRef = useRef<HTMLVideoElement>(null);
  const flowVideoRef = useRef<HTMLVideoElement>(null);
  const singleVideoRef = useRef<HTMLVideoElement>(null);

  const containerRef = useRef<HTMLDivElement>(null);

  const getAllVideos = (): (HTMLVideoElement | null)[] => [
    baseVideoRef.current,
    ampVideoRef.current,
    rawVideoRef.current,
    ampFramesVideoRef.current,
    flowVideoRef.current,
    singleVideoRef.current,
  ];

  // Duration detection from master video
  const handleLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    setDuration(e.currentTarget.duration);
  };

  // Time update sync
  const handleTimeUpdate = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    setCurrentTime(e.currentTarget.currentTime);
  };

  const togglePlay = () => {
    const nextPlay = !isPlaying;
    setIsPlaying(nextPlay);
    getAllVideos().forEach((v) => {
      if (v) {
        if (nextPlay) {
          v.play().catch(() => {});
        } else {
          v.pause();
        }
      }
    });
  };

  const handleSeek = (time: number) => {
    setCurrentTime(time);
    getAllVideos().forEach((v) => {
      if (v) v.currentTime = time;
    });
  };

  const handleRestart = () => {
    handleSeek(0);
    if (!isPlaying) togglePlay();
  };

  // Split-slider dragging
  const handleMouseDown = () => setIsDraggingSlider(true);
  const handleMouseUp = () => setIsDraggingSlider(false);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingSlider || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  };

  useEffect(() => {
    const onWindowMouseUp = () => setIsDraggingSlider(false);
    window.addEventListener('mouseup', onWindowMouseUp);
    return () => window.removeEventListener('mouseup', onWindowMouseUp);
  }, []);

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  return (
    <div className="display-area">
      {/* Mode Bar */}
      <div className="glass-panel view-controls">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>
            Visualization Mode:
          </span>
          <div className="mode-pills">
            <button
              className={`mode-pill ${viewMode === 'split' ? 'active' : ''}`}
              onClick={() => setViewMode('split')}
            >
              <Split size={14} style={{ marginRight: '0.35rem', verticalAlign: 'middle' }} />
              Split Slider
            </button>
            <button
              className={`mode-pill ${viewMode === 'grid' ? 'active' : ''}`}
              onClick={() => setViewMode('grid')}
            >
              <Grid2X2 size={14} style={{ marginRight: '0.35rem', verticalAlign: 'middle' }} />
              4-Way Grid
            </button>
            <button
              className={`mode-pill ${viewMode === 'single' ? 'active' : ''}`}
              onClick={() => setViewMode('single')}
            >
              <Maximize2 size={14} style={{ marginRight: '0.35rem', verticalAlign: 'middle' }} />
              Single Stream
            </button>
          </div>
        </div>

        {viewMode === 'single' && (
          <select
            className="select-input"
            style={{ width: 'auto' }}
            value={singleStreamKey}
            onChange={(e) => setSingleStreamKey(e.target.value as keyof PipelineVideos)}
          >
            <option value="amplified">Motion-Amplified Overlay (Ours)</option>
            <option value="baseline">Baseline Appearance Overlay</option>
            <option value="flow">Optical Flow Heatmap</option>
            <option value="amplified_frames">Eulerian Magnified Frames</option>
            <option value="raw">Raw Input</option>
            <option value="demo_output">Side-by-Side Comparison</option>
          </select>
        )}
      </div>

      {/* Main Video View */}
      {viewMode === 'split' && (
        <div
          ref={containerRef}
          className="split-slider-container glass-panel"
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          {/* Bottom Layer: Motion Amplified Overlay (Ours - Red) */}
          <video
            ref={ampVideoRef}
            src={videos.amplified}
            className="split-video"
            playsInline
            muted
            onLoadedMetadata={handleLoadedMetadata}
            onTimeUpdate={handleTimeUpdate}
          />
          <span className="split-tag tag-amplified">Motion-Amplified (Ours)</span>

          {/* Top Layer: Baseline Appearance-Only Overlay (Orange) */}
          <div
            className="split-layer-top"
            style={{ clipPath: `polygon(0 0, ${sliderPos}% 0, ${sliderPos}% 100%, 0 100%)` }}
          >
            <video
              ref={baseVideoRef}
              src={videos.baseline}
              className="split-video"
              playsInline
              muted
            />
            <span className="split-tag tag-baseline">Appearance-Only (Baseline)</span>
          </div>

          {/* Draggable Divider */}
          <div
            className="split-divider"
            style={{ left: `${sliderPos}%` }}
            onMouseDown={handleMouseDown}
          >
            <div className="split-handle">
              <Columns size={16} />
            </div>
          </div>
        </div>
      )}

      {viewMode === 'grid' && (
        <div className="grid-view-container">
          <div className="glass-panel stream-card">
            <div className="stream-card-header">
              <span className="stream-title">Raw Input Stream</span>
              <span className="badge badge-cyan">Stream A</span>
            </div>
            <div className="stream-video-box">
              <video ref={rawVideoRef} src={videos.raw} playsInline muted />
            </div>
          </div>

          <div className="glass-panel stream-card">
            <div className="stream-card-header">
              <span className="stream-title">Eulerian Magnification</span>
              <span className="badge badge-cyan">Laplacian &times; {metadata.fps} fps</span>
            </div>
            <div className="stream-video-box">
              <video ref={ampFramesVideoRef} src={videos.amplified_frames} playsInline muted />
            </div>
          </div>

          <div className="glass-panel stream-card">
            <div className="stream-card-header">
              <span className="stream-title">Optical Flow Heatmap</span>
              <span className="badge badge-cyan">Farneback Dense Flow</span>
            </div>
            <div className="stream-video-box">
              <video ref={flowVideoRef} src={videos.flow} playsInline muted />
            </div>
          </div>

          <div className="glass-panel stream-card">
            <div className="stream-card-header">
              <span className="stream-title" style={{ color: 'var(--accent-red)' }}>
                Detection Mask Overlay
              </span>
              <span className="badge badge-green">Two-Stream Output</span>
            </div>
            <div className="stream-video-box">
              <video
                ref={ampVideoRef}
                src={videos.amplified}
                playsInline
                muted
                onLoadedMetadata={handleLoadedMetadata}
                onTimeUpdate={handleTimeUpdate}
              />
            </div>
          </div>
        </div>
      )}

      {viewMode === 'single' && (
        <div className="glass-panel" style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
          <div style={{ aspectRatio: '16/9', maxHeight: '600px', background: '#000' }}>
            <video
              ref={singleVideoRef}
              src={videos[singleStreamKey]}
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              playsInline
              muted
              onLoadedMetadata={handleLoadedMetadata}
              onTimeUpdate={handleTimeUpdate}
            />
          </div>
        </div>
      )}

      {/* Unified Playback Bar */}
      <div className="glass-panel video-scrubber-panel">
        <div className="playback-controls">
          <button className="btn-icon" onClick={togglePlay} title={isPlaying ? 'Pause' : 'Play'}>
            {isPlaying ? <Pause size={18} /> : <Play size={18} fill="currentColor" />}
          </button>
          <button className="btn-icon" onClick={handleRestart} title="Restart video">
            <RotateCcw size={16} />
          </button>

          <input
            type="range"
            min={0}
            max={duration || 1}
            step={0.01}
            value={currentTime}
            onChange={(e) => handleSeek(parseFloat(e.target.value))}
            className="range-slider timeline-slider"
          />

          <div className="time-display">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>
        </div>
      </div>
    </div>
  );
};
