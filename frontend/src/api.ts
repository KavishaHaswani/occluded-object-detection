export const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface CreateJobParams {
  alpha?: number;
  low_hz?: number;
  high_hz?: number;
}

export interface JobOutputs {
  raw: string;
  baseline: string;
  amplified: string;
  amplified_frames: string;
  flow: string;
}

export interface JobResponse {
  status: 'queued' | 'running' | 'done' | 'error';
  stage: string;
  percent: number;
  error?: string | null;
  outputs?: JobOutputs | null;
}

export interface SampleItem {
  id: string;
  title: string;
  thumbnail?: string | null;
  outputs: JobOutputs;
}

export interface SamplesResponse {
  samples: SampleItem[];
}

/**
 * Creates and submits a new video processing job.
 */
export async function createJob(
  file: File,
  params?: CreateJobParams
): Promise<{ job_id: string }> {
  const formData = new FormData();
  formData.append('file', file);

  if (params?.alpha !== undefined) {
    formData.append('alpha', params.alpha.toString());
  }
  if (params?.low_hz !== undefined) {
    formData.append('low_hz', params.low_hz.toString());
  }
  if (params?.high_hz !== undefined) {
    formData.append('high_hz', params.high_hz.toString());
  }

  const response = await fetch(`${API_BASE_URL}/api/jobs`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Upload failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Fetches the status and outputs of an existing job by ID.
 */
export async function getJob(id: string): Promise<JobResponse> {
  const response = await fetch(`${API_BASE_URL}/api/jobs/${id}`);

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch job with status ${response.status}`);
  }

  return response.json();
}

/**
 * Fetches precomputed demo sample folders.
 */
export async function getSamples(): Promise<SamplesResponse> {
  const response = await fetch(`${API_BASE_URL}/api/samples`);

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch samples with status ${response.status}`);
  }

  return response.json();
}
