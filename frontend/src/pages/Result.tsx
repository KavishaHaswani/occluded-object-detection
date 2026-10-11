import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Video,
  FileCheck,
} from 'lucide-react';
import { getJob, JobResponse, API_BASE_URL } from '../api';

const STAGE_LABELS: Record<string, string> = {
  queued: 'Waiting in processing queue...',
  reading: 'Reading and validating video frames...',
  amplifying: 'Eulerian motion amplification & temporal filtering...',
  inference: 'Two-stream gated fusion neural network inference...',
  done: 'Processing finished successfully',
};

export const Result: React.FC = () => {
  const { id } = useParams<{ id: string }>();

  const [job, setJob] = useState<JobResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pollingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!id) return;

    const fetchStatus = async () => {
      try {
        const data = await getJob(id);
        setJob(data);

        if (data.status === 'done' || data.status === 'error') {
          if (pollingTimerRef.current) clearInterval(pollingTimerRef.current);
        }
      } catch (err: any) {
        setError(err.message || 'Failed to fetch job status');
        if (pollingTimerRef.current) clearInterval(pollingTimerRef.current);
      }
    };

    fetchStatus();
    pollingTimerRef.current = setInterval(fetchStatus, 800);

    return () => {
      if (pollingTimerRef.current) clearInterval(pollingTimerRef.current);
    };
  }, [id]);

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Back button & Title */}
      <div className="flex items-center justify-between">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-xs font-mono font-medium text-slate-400 hover:text-cyan-400 transition-colors"
        >
          <ArrowLeft size={16} />
          Back to Upload & Gallery
        </Link>
        <span className="text-xs font-mono text-slate-500">
          Job ID: <span className="text-cyan-400">{id}</span>
        </span>
      </div>

      {error ? (
        <div className="glass-card rounded-2xl p-8 border-rose-500/30 bg-rose-950/20 text-center space-y-4">
          <div className="w-12 h-12 mx-auto rounded-full bg-rose-500/10 text-rose-400 flex items-center justify-center">
            <XCircle size={28} />
          </div>
          <h2 className="text-lg font-semibold text-rose-300">Job Error</h2>
          <p className="text-sm text-slate-300">{error}</p>
          <Link
            to="/"
            className="inline-block mt-4 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-white transition-colors"
          >
            Upload Another Video
          </Link>
        </div>
      ) : !job ? (
        <div className="glass-card rounded-2xl p-16 text-center text-slate-400 space-y-3">
          <Loader2 size={32} className="animate-spin mx-auto text-cyan-400" />
          <p className="text-sm">Connecting to job runner...</p>
        </div>
      ) : (
        <div className="glass-card rounded-2xl p-6 sm:p-8 space-y-6">
          {/* Header Status Row */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800/80">
            <div>
              <div className="flex items-center gap-2">
                {job.status === 'queued' && (
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse" />
                )}
                {job.status === 'running' && (
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
                )}
                {job.status === 'done' && (
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
                )}
                {job.status === 'error' && (
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-400" />
                )}
                <h2 className="text-lg font-semibold text-white uppercase tracking-wider text-xs">
                  Status: <span className="text-cyan-400 font-mono">{job.status}</span>
                </h2>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                {STAGE_LABELS[job.stage] || job.stage}
              </p>
            </div>

            <div className="text-right">
              <span className="text-2xl font-bold font-mono text-cyan-400">
                {job.percent.toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden p-0.5 border border-slate-800">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 via-indigo-500 to-purple-500 rounded-full transition-all duration-300"
                style={{ width: `${Math.min(Math.max(job.percent, 3), 100)}%` }}
              />
            </div>
          </div>

          {/* Completed State: Outputs Manifest */}
          {job.status === 'done' && job.outputs && (
            <div className="pt-4 space-y-5">
              <div className="flex items-center gap-2 text-emerald-400 text-sm font-semibold">
                <CheckCircle2 size={18} />
                <span>Video Streams Generated & Re-encoded (H.264 / AVC)</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {Object.entries(job.outputs).map(([streamKey, streamPath]) => (
                  <div
                    key={streamKey}
                    className="p-3.5 rounded-xl bg-dark-900/60 border border-slate-800/80 flex items-center justify-between"
                  >
                    <div className="flex items-center gap-2.5">
                      <div className="w-8 h-8 rounded-lg bg-cyan-950/60 text-cyan-400 flex items-center justify-center">
                        <Video size={16} />
                      </div>
                      <div>
                        <div className="text-xs font-semibold uppercase text-slate-300">
                          {streamKey.replace('_', ' ')}
                        </div>
                        <div className="text-[11px] font-mono text-slate-500 truncate max-w-[200px]">
                          {streamPath}
                        </div>
                      </div>
                    </div>

                    <a
                      href={`${API_BASE_URL}${streamPath}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] font-mono text-cyan-400 transition-colors"
                    >
                      Open
                    </a>
                  </div>
                ))}
              </div>

              <div className="p-4 rounded-xl bg-cyan-950/20 border border-cyan-800/30 text-xs text-cyan-300 flex items-start gap-2.5">
                <FileCheck size={16} className="flex-shrink-0 mt-0.5" />
                <span>
                  All outputs are synchronized in frame count, FPS, and resolution with HTTP 206 Range support. Ready for video slider and multi-stream player integration.
                </span>
              </div>
            </div>
          )}

          {job.status === 'queued' && (
            <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-800/30 text-xs text-amber-300 flex items-center gap-2.5">
              <Clock size={16} />
              <span>
                Job is in queue. Processing will start immediately once the current model forward pass finishes.
              </span>
            </div>
          )}

          {job.status === 'error' && (
            <div className="p-4 rounded-xl bg-rose-950/20 border border-rose-800/30 text-xs text-rose-300 flex items-center gap-2.5">
              <XCircle size={16} />
              <span>{job.error || 'An error occurred during pipeline execution.'}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
