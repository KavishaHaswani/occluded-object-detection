import React from 'react';
import { VideoMetadata, PipelineParameters } from '../types';

interface MetricBadgeProps {
  metadata: VideoMetadata;
  parameters: PipelineParameters;
}

export const MetricBadge: React.FC<MetricBadgeProps> = ({ metadata, parameters }) => {
  return (
    <div className="meta-grid">
      <div className="glass-panel meta-card">
        <div className="meta-label">Resolution</div>
        <div className="meta-value">{metadata.width} &times; {metadata.height}</div>
      </div>
      <div className="glass-panel meta-card">
        <div className="meta-label">Frame Rate</div>
        <div className="meta-value">{metadata.fps.toFixed(1)} fps</div>
      </div>
      <div className="glass-panel meta-card">
        <div className="meta-label">Evaluated Frames</div>
        <div className="meta-value">{metadata.frame_count}</div>
      </div>
      <div className="glass-panel meta-card">
        <div className="meta-label">Amp Factor (&alpha;)</div>
        <div className="meta-value">{parameters.alpha.toFixed(1)}&times;</div>
      </div>
    </div>
  );
};
