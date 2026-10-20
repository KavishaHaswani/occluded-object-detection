import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Video,
  FileCheck,
  RotateCcw,
  AlertTriangle,
  Play,
  Download,
} from 'lucide-react';
import { getJob, JobResponse, API_BASE_URL } from '../api';
import { SideBySidePlayer } from '../components/SideBySidePlayer';
import { BeforeAfterSlider } from '../components/BeforeAfterSlider';
import { PipelineStageViewer } from '../components/PipelineStageViewer';
import { PlaybackProvider } from '../hooks/usePlayback';

interface StepDefinition {
  id: string;
  number: number;
  label: string;
  description: string;
}

const PIPELINE_STEPS: StepDefinition[] = [
  {
    id: 'reading',
    number: 1,
    label: 'Reading',
    description: 'Frame extraction & temporal windowing',
  },
  {
    id: 'amplifying',
    number: 2,
    label: 'Amplifying',
    description: 'Laplacian pyramid & Butterworth bandpass filter',
  },
  {
    id: 'inference',
    number: 3,
    label: 'Inference',
    description: 'Two-stream gated fusion neural network',
  },
];

const STAGE_LABELS: Record<string, string> = {
  queued: 'Waiting in processing queue...',
  reading: 'Reading and validating video frames...',
  amplifying: 'Eulerian motion amplification & temporal bandpass filtering...',
  inference: 'Running two-stream gated fusion neural network...',
  done: 'Processing finished successfully',
};

export const Result: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [job, setJob] = useState<JobResponse | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [isRetrying, setIsRetrying] = useState<boolean>(false);
  const [layoutMode, setLayoutMode] = useState<'side-by-side' | 'slider'>('side-by-side');

  const pollingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchStatus = useCallback(async () => {
    if (!id) return;
    try {
      setFetchError(null);
      const data = await getJob(id);
      setJob(data);

      if (data.status === 'done' || data.status === 'error') {
        if (pollingTimerRef.current) {
          clearInterval(pollingTimerRef.current);
          pollingTimerRef.current = null;
        }
      }
    } catch (err: any) {
      setFetchError(err.message || 'Unable to connect to backend service.');
    }
  }, [id]);

  useEffect(() => {
    if (!id) return;

    fetchStatus();
    // Poll every 1 second
    pollingTimerRef.current = setInterval(fetchStatus, 1000);

    return () => {
      if (pollingTimerRef.current) {
        clearInterval(pollingTimerRef.current);
        pollingTimerRef.current = null;
      }
    };
  }, [id, fetchStatus]);

  const handleRetryConnection = async () => {
    setIsRetrying(true);
    await fetchStatus();
    setIsRetrying(false);
    if (!pollingTimerRef.current && (!job || (job.status !== 'done' && job.status !== 'error'))) {
      pollingTimerRef.current = setInterval(fetchStatus, 1000);
    }
  };

  // Determine each step's status
  const getStepStatus = (stepId: string) => {
    if (!job) return 'pending';

    if (job.status === 'done') {
      return 'completed';
    }

    if (job.status === 'error') {
      if (job.stage === stepId) return 'error';
      // If error happened in a later step, previous steps completed
      if (stepId === 'reading' && (job.stage === 'amplifying' || job.stage === 'inference')) {
        return 'completed';
      }
      if (stepId === 'amplifying' && job.stage === 'inference') {
        return 'completed';
      }
      return 'pending';
    }

    if (job.status === 'queued') {
      return 'pending';
    }

    // job.status === 'running'
    if (job.stage === stepId) {
      return 'active';
    }

    if (stepId === 'reading') {
      return job.stage === 'amplifying' || job.stage === 'inference' ? 'completed' : 'pending';
    }

    if (stepId === 'amplifying') {
      return job.stage === 'inference' ? 'completed' : 'pending';
    }

    return 'pending';
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Top Navigation */}
      <div className="flex items-center justify-between">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-xs font-mono font-medium text-slate-400 hover:text-cyan-400 transition-colors"
        >
          <ArrowLeft size={16} />
          Back to Upload & Gallery
        </Link>
        <span className="text-xs font-mono text-slate-400 bg-slate-900/80 px-3 py-1 rounded-full border border-slate-800">
          Job ID: <span className="text-cyan-400 font-semibold">{id}</span>
        </span>
      </div>

      {/* Fetch / Network Error State */}
      {fetchError && !job && (
        <div className="glass-card rounded-2xl p-8 border-rose-500/30 bg-rose-950/20 text-center space-y-4">
          <div className="w-14 h-14 mx-auto rounded-full bg-rose-500/10 text-rose-400 flex items-center justify-center">
            <XCircle size={32} />
          </div>
          <h2 className="text-lg font-semibold text-rose-300">Connection Error</h2>
          <p className="text-sm text-slate-300 max-w-md mx-auto">{fetchError}</p>
          <div className="flex items-center justify-center gap-3 pt-2">
            <button
              onClick={handleRetryConnection}
              disabled={isRetrying}
              className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-xs font-semibold text-white transition-colors flex items-center gap-1.5 cursor-pointer"
            >
              <RotateCcw size={14} className={isRetrying ? 'animate-spin' : ''} />
              {isRetrying ? 'Retrying...' : 'Retry Connection'}
            </button>
            <button
              onClick={() => navigate('/')}
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 transition-colors cursor-pointer"
            >
              Upload Another Video
            </button>
          </div>
        </div>
      )}

      {/* Initial Loading / Connecting State */}
      {!job && !fetchError && (
        <div className="glass-card rounded-2xl p-16 text-center text-slate-400 space-y-3">
          <Loader2 size={36} className="animate-spin mx-auto text-cyan-400" />
          <p className="text-sm font-medium text-slate-300">Connecting to job runner...</p>
          <p className="text-xs text-slate-500">Checking job status and queue state</p>
        </div>
      )}

      {/* Main Job Card */}
      {job && (
        <div className="glass-card rounded-2xl p-6 sm:p-8 space-y-8">
          {/* Status Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800/80">
            <div>
              <div className="flex items-center gap-2.5">
                {job.status === 'queued' && (
                  <span className="flex h-3 w-3 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-amber-500" />
                  </span>
                )}
                {job.status === 'running' && (
                  <span className="flex h-3 w-3 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500" />
                  </span>
                )}
                {job.status === 'done' && (
                  <span className="inline-flex rounded-full h-3 w-3 bg-emerald-500 shadow-sm shadow-emerald-500/50" />
                )}
                {job.status === 'error' && (
                  <span className="inline-flex rounded-full h-3 w-3 bg-rose-500 shadow-sm shadow-rose-500/50" />
                )}

                <h2 className="text-base font-semibold text-white tracking-wide uppercase">
                  Status:{' '}
                  <span
                    className={`font-mono ${
                      job.status === 'done'
                        ? 'text-emerald-400'
                        : job.status === 'error'
                        ? 'text-rose-400'
                        : job.status === 'queued'
                        ? 'text-amber-400'
                        : 'text-cyan-400'
                    }`}
                  >
                    {job.status}
                  </span>
                </h2>
              </div>
              <p className="text-xs text-slate-400 mt-1.5">
                {STAGE_LABELS[job.stage] || job.stage}
              </p>
            </div>

            <div className="text-right">
              <span className="text-3xl font-bold font-mono text-cyan-400 tracking-tight">
                {job.percent.toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Stepper for "Reading, Amplifying, Inference" */}
          <div className="space-y-4">
            <div className="flex items-center justify-between text-xs text-slate-400 uppercase tracking-wider font-semibold">
              <span>Pipeline Stage Stepper</span>
              <span className="font-mono text-cyan-400">
                {job.status === 'done'
                  ? 'All stages completed'
                  : `Current: ${job.stage || 'queued'}`}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 relative">
              {PIPELINE_STEPS.map((step) => {
                const status = getStepStatus(step.id);
                return (
                  <div
                    key={step.id}
                    className={`rounded-xl p-4 border transition-all duration-300 relative flex flex-col justify-between ${
                      status === 'completed'
                        ? 'bg-emerald-950/20 border-emerald-500/40 shadow-sm shadow-emerald-500/10'
                        : status === 'active'
                        ? 'bg-cyan-950/30 border-cyan-400 ring-2 ring-cyan-500/20 shadow-lg shadow-cyan-500/10'
                        : status === 'error'
                        ? 'bg-rose-950/30 border-rose-500/60 shadow-sm shadow-rose-500/10'
                        : 'bg-dark-900/60 border-slate-800 text-slate-500'
                    }`}
                  >
                    <div className="flex items-center gap-3 mb-2">
                      {/* Step Circle Indicator */}
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center font-mono text-xs font-bold transition-all ${
                          status === 'completed'
                            ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/30'
                            : status === 'active'
                            ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/40 animate-pulse'
                            : status === 'error'
                            ? 'bg-rose-500 text-white shadow-md shadow-rose-500/30'
                            : 'bg-slate-800 text-slate-400 border border-slate-700'
                        }`}
                      >
                        {status === 'completed' ? (
                          <CheckCircle2 size={16} />
                        ) : status === 'active' ? (
                          <Loader2 size={16} className="animate-spin" />
                        ) : status === 'error' ? (
                          <XCircle size={16} />
                        ) : (
                          step.number
                        )}
                      </div>

                      <div>
                        <h4
                          className={`text-sm font-semibold ${
                            status === 'completed'
                              ? 'text-emerald-300'
                              : status === 'active'
                              ? 'text-cyan-300'
                              : status === 'error'
                              ? 'text-rose-300'
                              : 'text-slate-400'
                          }`}
                        >
                          {step.label}
                        </h4>
                        <span className="text-[11px] font-mono text-slate-400 block">
                          Stage {step.number} of 3
                        </span>
                      </div>
                    </div>

                    <p className="text-xs text-slate-400 leading-snug">
                      {step.description}
                    </p>

                    {/* Step Status Badge */}
                    <div className="mt-3 pt-2.5 border-t border-slate-800/60 flex items-center justify-between text-[11px] font-mono">
                      <span className="text-slate-500">State:</span>
                      <span
                        className={`font-semibold capitalize ${
                          status === 'completed'
                            ? 'text-emerald-400'
                            : status === 'active'
                            ? 'text-cyan-400 animate-pulse'
                            : status === 'error'
                            ? 'text-rose-400'
                            : 'text-slate-500'
                        }`}
                      >
                        {status}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Progress Bar */}
          <div className="space-y-2.5">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-400 font-medium">
                {STAGE_LABELS[job.stage] || 'Pipeline processing...'}
              </span>
              <span className="font-mono text-cyan-400 font-semibold">
                {job.percent.toFixed(1)}%
              </span>
            </div>

            <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden p-0.5 border border-slate-800">
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  job.status === 'error'
                    ? 'bg-rose-500'
                    : job.status === 'done'
                    ? 'bg-emerald-500'
                    : 'bg-gradient-to-r from-cyan-500 via-indigo-500 to-emerald-400'
                }`}
                style={{ width: `${Math.min(Math.max(job.percent, 3), 100)}%` }}
              />
            </div>
          </div>

          {/* Readable Error State with Retry Button */}
          {job.status === 'error' && (
            <div className="p-6 rounded-xl bg-rose-950/30 border border-rose-500/40 text-rose-200 space-y-4">
              <div className="flex items-start gap-3">
                <AlertTriangle size={24} className="text-rose-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-sm font-semibold text-rose-300">
                    Pipeline Execution Failed
                  </h3>
                  <p className="text-xs text-rose-200/90 mt-1 leading-relaxed">
                    {job.error ||
                      'An unexpected error occurred during frame extraction, motion magnification, or neural inference.'}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => navigate('/')}
                  className="px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs shadow-md shadow-rose-600/20 transition-colors flex items-center gap-1.5 cursor-pointer"
                >
                  <RotateCcw size={14} />
                  Retry / Upload Again
                </button>
                <button
                  type="button"
                  onClick={handleRetryConnection}
                  className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-colors cursor-pointer"
                >
                  Refresh Status
                </button>
              </div>
            </div>
          )}

          {/* Queued State Card */}
          {job.status === 'queued' && (
            <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-800/40 text-xs text-amber-300 flex items-center gap-3">
              <Clock size={18} className="flex-shrink-0" />
              <span>
                Job queued behind current computation. Sequential worker will automatically begin processing shortly.
              </span>
            </div>
          )}

          {/* Done State: Outputs Manifest, Before/After Slider & Stage Breakdown */}
          {job.status === 'done' && job.outputs && (
            <PlaybackProvider>
              <div className="pt-2 space-y-8">
                {/* 1. Synchronized Comparison Viewer with Layout Switcher */}
                <div className="space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-1 border-b border-slate-800/80">
                    <div>
                      <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
                        <Video size={18} className="text-cyan-400" />
                        <span>Camouflage Breaking Comparison</span>
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Synchronized comparison between appearance baseline and motion-amplified detection
                      </p>
                    </div>

                    {/* Layout switcher: Side by side (default) vs Slider */}
                    <div className="flex items-center gap-2 self-start sm:self-auto">
                      <span className="text-xs text-slate-400 font-medium">Layout:</span>
                      <div className="inline-flex bg-slate-900/90 p-1 rounded-xl border border-slate-800 text-xs">
                        <button
                          type="button"
                          onClick={() => setLayoutMode('side-by-side')}
                          className={`px-3 py-1 rounded-lg font-medium transition-all cursor-pointer ${
                            layoutMode === 'side-by-side'
                              ? 'bg-cyan-500 text-slate-950 font-semibold shadow-sm'
                              : 'text-slate-400 hover:text-white'
                          }`}
                        >
                          Side by side
                        </button>
                        <button
                          type="button"
                          onClick={() => setLayoutMode('slider')}
                          className={`px-3 py-1 rounded-lg font-medium transition-all cursor-pointer ${
                            layoutMode === 'slider'
                              ? 'bg-cyan-500 text-slate-950 font-semibold shadow-sm'
                              : 'text-slate-400 hover:text-white'
                          }`}
                        >
                          Slider
                        </button>
                      </div>
                    </div>
                  </div>

                  {layoutMode === 'side-by-side' ? (
                    <SideBySidePlayer outputs={job.outputs} />
                  ) : (
                    <BeforeAfterSlider outputs={job.outputs} />
                  )}
                </div>

                {/* 2. Synchronized Pipeline Stage Viewer (Raw, Amp, Flow, Mask) with Overlay & SVG Architecture */}
                <PipelineStageViewer outputs={job.outputs} />

                <div className="space-y-4 pt-4 border-t border-slate-800/80">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2 text-emerald-400 text-sm font-semibold">
                        <CheckCircle2 size={18} />
                        <span>Output Streams Generated Successfully</span>
                      </div>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Download or preview synchronized video deliverables
                      </p>
                    </div>

                    {/* Quick Download Buttons */}
                    <div className="flex flex-wrap items-center gap-2">
                      {job.outputs.amplified && (
                        <a
                          href={`${API_BASE_URL}${job.outputs.amplified}`}
                          download="amplified_result.mp4"
                          className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-semibold text-xs transition-all flex items-center gap-1.5 shadow-md shadow-cyan-500/20"
                        >
                          <Download size={13} />
                          Download Amplified Result
                        </a>
                      )}
                      {job.outputs.raw && (
                        <a
                          href={`${API_BASE_URL}${job.outputs.raw}`}
                          download="raw_input.mp4"
                          className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition-colors flex items-center gap-1.5 border border-slate-700/60"
                        >
                          <Download size={13} />
                          Raw Video
                        </a>
                      )}
                    </div>
                  </div>

                  {/* Stream Cards Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {Object.entries(job.outputs).map(([streamKey, streamPath]) => (
                      <div
                        key={streamKey}
                        className="p-3.5 rounded-xl bg-dark-900/70 border border-slate-800 flex items-center justify-between group hover:border-cyan-500/40 transition-colors"
                      >
                        <div className="flex items-center gap-3 min-w-0">
                          <div className="w-8 h-8 rounded-lg bg-cyan-950/60 text-cyan-400 flex items-center justify-center flex-shrink-0">
                            <Video size={16} />
                          </div>
                          <div className="min-w-0">
                            <div className="text-xs font-semibold uppercase text-slate-200 tracking-wide">
                              {streamKey.replace('_', ' ')}
                            </div>
                            <div className="text-[11px] font-mono text-slate-500 truncate max-w-[150px]">
                              {streamPath}
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-1.5 flex-shrink-0">
                          <a
                            href={`${API_BASE_URL}${streamPath}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-cyan-500/20 hover:text-cyan-300 text-[11px] font-mono text-cyan-400 border border-slate-700/60 transition-colors flex items-center gap-1"
                          >
                            <Play size={10} className="fill-current" />
                            View
                          </a>
                          <a
                            href={`${API_BASE_URL}${streamPath}`}
                            download={`${streamKey}.mp4`}
                            className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-[11px] font-mono text-slate-300 border border-slate-700/60 transition-colors flex items-center gap-1"
                            title={`Download ${streamKey}.mp4`}
                          >
                            <Download size={11} />
                            Save
                          </a>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="p-4 rounded-xl bg-cyan-950/20 border border-cyan-800/30 text-xs text-cyan-300 flex items-start gap-2.5">
                    <FileCheck size={16} className="flex-shrink-0 mt-0.5" />
                    <span>
                      All streams have equal frame count and framerate with HTTP 206 Partial Content range support.
                    </span>
                  </div>
                </div>
              </div>
            </PlaybackProvider>
          )}
        </div>
      )}
    </div>
  );
};
