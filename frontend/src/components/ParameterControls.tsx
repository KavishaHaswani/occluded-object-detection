import React from 'react';
import { Sliders, Play, RotateCcw } from 'lucide-react';
import { PipelineParameters } from '../types';

interface ParameterControlsProps {
  parameters: PipelineParameters;
  onChange: (params: PipelineParameters) => void;
  onSubmit: () => void;
  isLoading: boolean;
  canSubmit: boolean;
}

const DEFAULT_PARAMS: PipelineParameters = {
  alpha: 15.0,
  low_hz: 0.2,
  high_hz: 3.0,
  window_size: 60,
  overlap: 30,
};

export const ParameterControls: React.FC<ParameterControlsProps> = ({
  parameters,
  onChange,
  onSubmit,
  isLoading,
  canSubmit,
}) => {
  const updateParam = (key: keyof PipelineParameters, value: number) => {
    const updated = { ...parameters, [key]: value };
    if (key === 'window_size') {
      updated.overlap = Math.floor(value / 2);
    }
    onChange(updated);
  };

  const handleReset = () => {
    onChange(DEFAULT_PARAMS);
  };

  return (
    <div className="glass-panel controls-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div className="panel-title" style={{ margin: 0 }}>
          <Sliders size={16} />
          Magnification Studio
        </div>
        <button
          onClick={handleReset}
          className="mode-pill"
          style={{ fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
          title="Reset to recommended defaults"
        >
          <RotateCcw size={12} />
          Reset
        </button>
      </div>

      <div className="control-group">
        <div className="control-header">
          <span className="control-label">Amplification Factor (&alpha;)</span>
          <span className="control-value">{parameters.alpha.toFixed(1)}&times;</span>
        </div>
        <input
          type="range"
          min="5.0"
          max="35.0"
          step="1.0"
          className="range-slider"
          value={parameters.alpha}
          disabled={isLoading}
          onChange={(e) => updateParam('alpha', parseFloat(e.target.value))}
        />
      </div>

      <div className="control-group">
        <div className="control-header">
          <span className="control-label">Bandpass Low Cutoff (f_low)</span>
          <span className="control-value">{parameters.low_hz.toFixed(2)} Hz</span>
        </div>
        <input
          type="range"
          min="0.05"
          max="1.5"
          step="0.05"
          className="range-slider"
          value={parameters.low_hz}
          disabled={isLoading}
          onChange={(e) => updateParam('low_hz', parseFloat(e.target.value))}
        />
      </div>

      <div className="control-group">
        <div className="control-header">
          <span className="control-label">Bandpass High Cutoff (f_high)</span>
          <span className="control-value">{parameters.high_hz.toFixed(2)} Hz</span>
        </div>
        <input
          type="range"
          min="1.5"
          max="5.0"
          step="0.1"
          className="range-slider"
          value={parameters.high_hz}
          disabled={isLoading}
          onChange={(e) => updateParam('high_hz', parseFloat(e.target.value))}
        />
      </div>

      <div className="control-group">
        <div className="control-header">
          <span className="control-label">Sliding Window Size</span>
          <span className="control-value">{parameters.window_size} frames</span>
        </div>
        <input
          type="range"
          min="24"
          max="120"
          step="6"
          className="range-slider"
          value={parameters.window_size}
          disabled={isLoading}
          onChange={(e) => updateParam('window_size', parseInt(e.target.value))}
        />
      </div>

      <button
        className="btn-primary"
        onClick={onSubmit}
        disabled={!canSubmit || isLoading}
      >
        <Play size={16} fill="white" />
        {isLoading ? 'Processing Pipeline...' : 'Break Camouflage'}
      </button>
    </div>
  );
};
