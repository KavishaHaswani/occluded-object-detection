import React from 'react';
import { Loader2 } from 'lucide-react';
import { JobStatusResponse } from '../types';

interface JobProgressBarProps {
  jobStatus: JobStatusResponse | null;
}

const STAGE_DESCRIPTIONS: Record<string, string> = {
  queued: 'Preparing GPU/CPU compute buffers...',
  reading: 'Extracting video frames and validating timestamps...',
  amplifying: 'Eulerian Laplacian pyramid magnification & Butterworth temporal bandpassing...',
  inference: 'Two-stream gated fusion neural network inference...',
};

export const JobProgressBar: React.FC<JobProgressBarProps> = ({ jobStatus }) => {
  if (!jobStatus || jobStatus.status === 'completed') return null;

  const stageText = STAGE_DESCRIPTIONS[jobStatus.stage] || 'Processing video streams...';

  return (
    <div className="glass-panel progress-container">
      <div className="progress-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Loader2 size={16} className="animate-spin" color="var(--accent-cyan)" />
          <span className="stage-badge">Stage: {jobStatus.stage}</span>
        </div>
        <span className="control-value">{jobStatus.percent.toFixed(1)}%</span>
      </div>

      <div className="progress-bar-bg">
        <div
          className="progress-bar-fill"
          style={{ width: `${Math.min(Math.max(jobStatus.percent, 3), 100)}%` }}
        />
      </div>

      <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
        {stageText}
      </div>

      {jobStatus.error && (
        <div style={{ fontSize: '0.8rem', color: 'var(--accent-red)', marginTop: '0.5rem' }}>
          Error: {jobStatus.error}
        </div>
      )}
    </div>
  );
};
