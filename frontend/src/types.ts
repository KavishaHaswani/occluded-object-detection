export interface PipelineParameters {
  alpha: number;
  low_hz: number;
  high_hz: number;
  window_size: number;
  overlap: number;
}

export interface VideoMetadata {
  frame_count: number;
  fps: number;
  width: number;
  height: number;
}

export interface PipelineVideos {
  raw: string;
  baseline: string;
  amplified: string;
  amplified_frames: string;
  flow: string;
  demo_output: string;
}

export interface JobResults {
  job_id: string;
  videos: PipelineVideos;
  metadata: VideoMetadata;
  parameters: PipelineParameters;
}

export interface JobStatusResponse {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  stage: 'queued' | 'reading' | 'amplifying' | 'inference';
  percent: number;
  error?: string | null;
}

export interface SampleClip {
  id: string;
  title: string;
  path: string;
  is_precomputed: boolean;
}

export type ViewMode = 'split' | 'grid' | 'single';
