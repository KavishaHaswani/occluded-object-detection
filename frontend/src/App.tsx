import React, { useState, useEffect, useRef } from 'react';
import { Header } from './components/Header';
import { UploadZone } from './components/UploadZone';
import { ParameterControls } from './components/ParameterControls';
import { JobProgressBar } from './components/JobProgressBar';
import { MultiStreamPlayer } from './components/MultiStreamPlayer';
import { MetricBadge } from './components/MetricBadge';
import {
  fetchSamples,
  submitProcessingJob,
  getJobStatus,
  getJobResults,
} from './services/api';
import {
  JobResults,
  JobStatusResponse,
  PipelineParameters,
  SampleClip,
} from './types';
import { Sparkles, PlayCircle } from 'lucide-react';

const DEFAULT_PARAMETERS: PipelineParameters = {
  alpha: 15.0,
  low_hz: 0.2,
  high_hz: 3.0,
  window_size: 60,
  overlap: 30,
};

export const App: React.FC = () => {
  const [samples, setSamples] = useState<SampleClip[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [parameters, setParameters] = useState<PipelineParameters>(DEFAULT_PARAMETERS);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null);
  const [jobResults, setJobResults] = useState<JobResults | null>(null);

  const pollingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetchSamples().then((data) => {
      setSamples(data);
      if (data.length > 0) {
        setSelectedSampleId(data[0].id);
      }
    });
  }, []);

  const handleRunPipeline = async () => {
    if (!selectedFile && !selectedSampleId) return;

    setIsLoading(true);
    setJobResults(null);
    setJobStatus({
      job_id: '',
      status: 'pending',
      stage: 'queued',
      percent: 0,
    });

    try {
      const resp = await submitProcessingJob(selectedFile, selectedSampleId, parameters);
      const jobId = resp.job_id;

      // Poll job status every 600ms
      pollingTimerRef.current = setInterval(async () => {
        try {
          const status = await getJobStatus(jobId);
          setJobStatus(status);

          if (status.status === 'completed') {
            if (pollingTimerRef.current) clearInterval(pollingTimerRef.current);
            const results = await getJobResults(jobId);
            setJobResults(results);
            setIsLoading(false);
          } else if (status.status === 'failed') {
            if (pollingTimerRef.current) clearInterval(pollingTimerRef.current);
            setIsLoading(false);
          }
        } catch (err) {
          console.error('Polling error:', err);
        }
      }, 600);
    } catch (err: any) {
      alert(err.message || 'Pipeline submission error');
      setIsLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      if (pollingTimerRef.current) clearInterval(pollingTimerRef.current);
    };
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header />

      <main className="app-container">
        <div className="app-grid">
          {/* Controls & Configuration Sidebar */}
          <aside className="sidebar">
            <UploadZone
              selectedFile={selectedFile}
              onSelectFile={setSelectedFile}
              samples={samples}
              selectedSampleId={selectedSampleId}
              onSelectSampleId={setSelectedSampleId}
              disabled={isLoading}
            />

            <ParameterControls
              parameters={parameters}
              onChange={setParameters}
              onSubmit={handleRunPipeline}
              isLoading={isLoading}
              canSubmit={Boolean(selectedFile || selectedSampleId)}
            />

            {isLoading && <JobProgressBar jobStatus={jobStatus} />}
          </aside>

          {/* Visualization Studio */}
          <section style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {jobResults ? (
              <>
                <MultiStreamPlayer
                  videos={jobResults.videos}
                  metadata={jobResults.metadata}
                />
                <MetricBadge
                  metadata={jobResults.metadata}
                  parameters={jobResults.parameters}
                />
              </>
            ) : (
              <div
                className="glass-panel"
                style={{
                  padding: '3.5rem 2rem',
                  textAlign: 'center',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: '440px',
                }}
              >
                <div
                  style={{
                    width: '64px',
                    height: '64px',
                    borderRadius: '50%',
                    background: 'rgba(6, 182, 212, 0.1)',
                    border: '1px solid rgba(6, 182, 212, 0.3)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginBottom: '1.25rem',
                    color: 'var(--accent-cyan)',
                  }}
                >
                  <Sparkles size={32} />
                </div>
                <h2 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '0.6rem' }}>
                  Eulerian Camouflage Breaking Studio
                </h2>
                <p style={{ color: 'var(--text-muted)', maxWidth: '520px', lineHeight: 1.6, fontSize: '0.92rem' }}>
                  Select a camouflaged sample clip or upload custom footage, tune the magnification bandpass sliders, and click <strong>Break Camouflage</strong> to view the real-time split-slider detection.
                </p>
                <button
                  className="btn-primary"
                  style={{ width: 'auto', marginTop: '1.5rem', padding: '0.75rem 1.75rem' }}
                  onClick={handleRunPipeline}
                  disabled={!selectedFile && !selectedSampleId}
                >
                  <PlayCircle size={18} />
                  Run Demo on Chameleon Clip
                </button>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
};
