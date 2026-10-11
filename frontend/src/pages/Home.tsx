import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, Film, Sliders, Play, AlertCircle, Sparkles, CheckCircle2 } from 'lucide-react';
import { createJob, getSamples, SampleItem, API_BASE_URL } from '../api';

export const Home: React.FC = () => {
  const navigate = useNavigate();

  // Upload & Form State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [alpha, setAlpha] = useState<number>(15.0);
  const [lowHz, setLowHz] = useState<number>(0.2);
  const [highHz, setHighHz] = useState<number>(3.0);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Samples State
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [isLoadingSamples, setIsLoadingSamples] = useState<boolean>(true);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getSamples()
      .then((res) => {
        setSamples(res.samples || []);
      })
      .catch((err) => {
        console.error('Failed to load samples:', err);
      })
      .finally(() => {
        setIsLoadingSamples(false);
      });
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setErrorMessage(null);
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.size > 50 * 1024 * 1024) {
        setErrorMessage('File size exceeds the 50 MB limit.');
        return;
      }
      setSelectedFile(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (!file.type.startsWith('video/') && !file.name.match(/\.(mp4|mov|avi|mkv|webm)$/i)) {
        setErrorMessage('Please drop a valid video file.');
        return;
      }
      if (file.size > 50 * 1024 * 1024) {
        setErrorMessage('File size exceeds the 50 MB limit.');
        return;
      }
      setSelectedFile(file);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setErrorMessage('Please select a video file first.');
      return;
    }

    setIsUploading(true);
    setErrorMessage(null);

    try {
      const res = await createJob(selectedFile, {
        alpha,
        low_hz: lowHz,
        high_hz: highHz,
      });
      // Navigate to the result page with the newly created job ID
      navigate(`/job/${res.job_id}`);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to submit video job.');
      setIsUploading(false);
    }
  };

  return (
    <div className="space-y-12">
      {/* Hero & Upload Section */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Upload Form */}
        <div className="lg:col-span-7 glass-card rounded-2xl p-6 sm:p-8 space-y-6">
          <div className="flex items-center gap-2.5 text-cyan-400">
            <Film size={20} />
            <h2 className="text-lg font-semibold tracking-wide uppercase text-slate-200">
              Upload Video Footage
            </h2>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Drag & Drop Zone */}
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                selectedFile
                  ? 'border-cyan-500/80 bg-cyan-950/20 shadow-lg shadow-cyan-500/10'
                  : 'border-slate-700/80 hover:border-cyan-500/50 hover:bg-dark-900/50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*,.mp4,.mov,.avi,.mkv,.webm"
                className="hidden"
                onChange={handleFileChange}
                disabled={isUploading}
              />
              <div className="w-12 h-12 mx-auto rounded-full bg-cyan-500/10 text-cyan-400 flex items-center justify-center mb-3">
                {selectedFile ? <CheckCircle2 size={24} /> : <UploadCloud size={24} />}
              </div>
              <p className="text-sm font-medium text-slate-200">
                {selectedFile ? selectedFile.name : 'Click to select or drag & drop video clip'}
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Supported formats: MP4, MOV, AVI &middot; Max 50 MB &middot; Max 15 seconds
              </p>
            </div>

            {errorMessage && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs">
                <AlertCircle size={16} className="flex-shrink-0" />
                <span>{errorMessage}</span>
              </div>
            )}

            {/* Hyperparameter Controls */}
            <div className="bg-dark-900/60 rounded-xl p-4 sm:p-5 border border-slate-800/80 space-y-4">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                <Sliders size={14} className="text-cyan-400" />
                Amplification Parameters
              </div>

              {/* Alpha slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Amplification Factor (&alpha;)</span>
                  <span className="font-mono text-cyan-400 font-semibold">{alpha.toFixed(1)}&times;</span>
                </div>
                <input
                  type="range"
                  min="5.0"
                  max="35.0"
                  step="1.0"
                  value={alpha}
                  onChange={(e) => setAlpha(parseFloat(e.target.value))}
                  disabled={isUploading}
                  className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>

              {/* Bandpass frequencies */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400">Low Cutoff</span>
                    <span className="font-mono text-cyan-400 font-semibold">{lowHz.toFixed(2)} Hz</span>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="1.5"
                    step="0.05"
                    value={lowHz}
                    onChange={(e) => setLowHz(parseFloat(e.target.value))}
                    disabled={isUploading}
                    className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                  />
                </div>
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400">High Cutoff</span>
                    <span className="font-mono text-cyan-400 font-semibold">{highHz.toFixed(2)} Hz</span>
                  </div>
                  <input
                    type="range"
                    min="1.5"
                    max="5.0"
                    step="0.1"
                    value={highHz}
                    onChange={(e) => setHighHz(parseFloat(e.target.value))}
                    disabled={isUploading}
                    className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                  />
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={!selectedFile || isUploading}
              className="w-full py-3.5 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold text-sm shadow-lg shadow-cyan-500/20 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 group cursor-pointer"
            >
              <Play size={16} className="fill-current group-hover:scale-110 transition-transform" />
              {isUploading ? 'Uploading & Enqueuing...' : 'Process Video & Break Camouflage'}
            </button>
          </form>
        </div>

        {/* Right Column: Explanatory Banner */}
        <div className="lg:col-span-5 glass-card rounded-2xl p-6 sm:p-8 space-y-5">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono font-medium bg-indigo-950/60 text-indigo-400 border border-indigo-800/50">
            <Sparkles size={13} />
            Two-Stream Architecture
          </div>
          <h3 className="text-xl font-bold text-white tracking-tight">
            How Eulerian Amplification Reveals Concealed Objects
          </h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            Camouflaged objects evade traditional single-frame computer vision because their texture, color, and reflectance blend into the background.
          </p>
          <ul className="space-y-3 text-xs text-slate-300">
            <li className="flex items-start gap-2.5">
              <span className="w-5 h-5 rounded-md bg-cyan-950 text-cyan-400 flex items-center justify-center flex-shrink-0 font-mono text-[10px] font-bold mt-0.5">
                1
              </span>
              <span>
                <strong>Laplacian Pyramid Decomposition:</strong> Isolates subtle spatial frequency bands across consecutive sliding temporal windows.
              </span>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-5 h-5 rounded-md bg-cyan-950 text-cyan-400 flex items-center justify-center flex-shrink-0 font-mono text-[10px] font-bold mt-0.5">
                2
              </span>
              <span>
                <strong>Temporal Butterworth Bandpass:</strong> Amplifies subtle biological micro-movements (breathing, twitching, crawling) by factor &alpha;.
              </span>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-5 h-5 rounded-md bg-cyan-950 text-cyan-400 flex items-center justify-center flex-shrink-0 font-mono text-[10px] font-bold mt-0.5">
                3
              </span>
              <span>
                <strong>Two-Stream Gated Fusion:</strong> Fuses spatial appearance with optical flow computed directly on amplified frames to delineate masks.
              </span>
            </li>
          </ul>
        </div>
      </section>

      {/* Precomputed Sample Gallery */}
      <section className="space-y-5">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold tracking-wide uppercase text-slate-200">
              Sample Gallery
            </h2>
            <p className="text-xs text-slate-400">
              Explore precomputed sample clips with verified Eulerian motion amplification
            </p>
          </div>
          <span className="text-xs font-mono text-cyan-400 bg-cyan-950/40 px-2.5 py-1 rounded-md border border-cyan-800/40">
            {samples.length} Available
          </span>
        </div>

        {isLoadingSamples ? (
          <div className="glass-card rounded-2xl p-10 text-center text-slate-500 text-sm">
            Loading sample gallery...
          </div>
        ) : samples.length === 0 ? (
          <div className="glass-card rounded-2xl p-10 text-center text-slate-500 text-sm">
            No precomputed samples found in samples/ directory.
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {samples.map((sample) => (
              <div
                key={sample.id}
                className="glass-card glass-card-hover rounded-xl overflow-hidden flex flex-col group"
              >
                {/* Thumbnail */}
                <div className="relative aspect-video bg-black/60 overflow-hidden">
                  {sample.thumbnail ? (
                    <img
                      src={`${API_BASE_URL}${sample.thumbnail}`}
                      alt={sample.title}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-600">
                      <Film size={32} />
                    </div>
                  )}
                  <span className="absolute top-2.5 right-2.5 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-emerald-950/80 text-emerald-400 border border-emerald-800/60 backdrop-blur-md">
                    Precomputed
                  </span>
                </div>

                {/* Body */}
                <div className="p-4 flex-1 flex flex-col justify-between space-y-4">
                  <div>
                    <h3 className="font-semibold text-white text-base group-hover:text-cyan-400 transition-colors">
                      {sample.title}
                    </h3>
                    <p className="text-xs text-slate-400 mt-1">
                      Multi-stream breakdown: raw, baseline, amplified, and motion flow.
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-400 pt-2 border-t border-slate-800/80">
                    <div>
                      Streams: <span className="text-cyan-400">5 synchronized</span>
                    </div>
                    <div>
                      Codec: <span className="text-slate-300">H.264 (AVC)</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};
