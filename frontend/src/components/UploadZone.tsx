import React, { useRef } from 'react';
import { UploadCloud, Film } from 'lucide-react';
import { SampleClip } from '../types';

interface UploadZoneProps {
  selectedFile: File | null;
  onSelectFile: (file: File | null) => void;
  samples: SampleClip[];
  selectedSampleId: string | null;
  onSelectSampleId: (id: string | null) => void;
  disabled?: boolean;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  selectedFile,
  onSelectFile,
  samples,
  selectedSampleId,
  onSelectSampleId,
  disabled,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onSelectFile(e.dataTransfer.files[0]);
      onSelectSampleId(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onSelectFile(e.target.files[0]);
      onSelectSampleId(null);
    }
  };

  return (
    <div className="glass-panel upload-container">
      <div className="panel-title">
        <Film size={16} />
        Video Input Source
      </div>

      <div
        className={`dropzone ${selectedFile ? 'active' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => !disabled && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          style={{ display: 'none' }}
          onChange={handleFileChange}
          disabled={disabled}
        />
        <div className="dropzone-icon">
          <UploadCloud size={32} />
        </div>
        <div className="dropzone-label">
          {selectedFile ? selectedFile.name : 'Click or drop video clip here'}
        </div>
        <div className="dropzone-sub">Supports MP4, MOV, AVI up to 1080p</div>
      </div>

      {samples.length > 0 && (
        <div className="sample-selector">
          <div className="control-label" style={{ marginBottom: '0.4rem' }}>
            Or choose a pre-loaded sample clip:
          </div>
          <select
            className="select-input"
            value={selectedSampleId || ''}
            disabled={disabled}
            onChange={(e) => {
              const val = e.target.value;
              if (val) {
                onSelectSampleId(val);
                onSelectFile(null);
              } else {
                onSelectSampleId(null);
              }
            }}
          >
            <option value="">-- Choose Sample Clip --</option>
            {samples.map((s) => (
              <option key={s.id} value={s.id}>
                {s.title}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
};
