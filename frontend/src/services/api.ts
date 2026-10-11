import { JobResults, JobStatusResponse, PipelineParameters, SampleClip } from '../types';

const API_BASE = '/api';

export async function fetchSamples(): Promise<SampleClip[]> {
  try {
    const res = await fetch(`${API_BASE}/samples`);
    if (!res.ok) throw new Error('Failed to load samples');
    const data = await res.json();
    return data.samples || [];
  } catch (err) {
    console.error('Error fetching samples:', err);
    return [];
  }
}

export async function submitProcessingJob(
  file: File | null,
  sampleId: string | null,
  params: PipelineParameters
): Promise<{ job_id: string; status: string }> {
  const formData = new FormData();

  if (file) {
    formData.append('file', file);
  } else if (sampleId) {
    formData.append('sample_id', sampleId);
  } else {
    throw new Error('Please select a sample or upload a video file.');
  }

  formData.append('alpha', params.alpha.toString());
  formData.append('low_hz', params.low_hz.toString());
  formData.append('high_hz', params.high_hz.toString());
  formData.append('window_size', params.window_size.toString());
  formData.append('overlap', params.overlap.toString());

  const res = await fetch(`${API_BASE}/process`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || 'Failed to submit pipeline job');
  }

  return await res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/status`);
  if (!res.ok) throw new Error('Failed to fetch job status');
  return await res.json();
}

export async function getJobResults(jobId: string): Promise<JobResults> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/results`);
  if (!res.ok) throw new Error('Failed to fetch job results');
  return await res.json();
}
