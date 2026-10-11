import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  UploadCloud,
  Film,
  Sliders,
  Play,
  AlertCircle,
  Sparkles,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Info,
  RotateCcw,
  X,
  Clock,
  HardDrive,
  FileVideo,
  Loader2,
} from 'lucide-react';
import { createJob, getSamples, SampleItem, API_BASE_URL } from '../api';
import { MetricsPanel } from '../components/MetricsPanel';
import { AboutSection } from '../components/AboutSection';

interface FileValidationState {
  isValidating: boolean;
  sizeBytes: number;
  formattedSize: string;
  isSizeValid: boolean;
  durationSeconds: number | null;
  formattedDuration: string | null;
  isDurationValid: boolean;
  format: string;
  isFormatValid: boolean;
  errorMessage: string | null;
}

const Tooltip: React.FC<{ text: string }> = ({ text }) => (
  <div className="relative inline-flex items-center group ml-1.5">
    <Info size={13} className="text-slate-400 hover:text-cyan-400 cursor-help transition-colors" />
    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-40 w-64 p-2.5 rounded-lg bg-slate-900 border border-slate-700 text-[11px] text-slate-200 shadow-2xl leading-snug pointer-events-none text-left">
      {text}
      <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-700" />
    </div>
  </div>
);

export const Home: React.FC = () => {
  const navigate = useNavigate();

  // File & validation state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileValidation, setFileValidation] = useState<FileValidationState | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  // Advanced settings state
  const [isAdvancedOpen, setIsAdvancedOpen] = useState<boolean>(false);
  const [alpha, setAlpha] = useState<number>(15.0);
  const [lowHz, setLowHz] = useState<number>(0.2);
  const [highHz, setHighHz] = useState<number>(3.0);

  // Submission & samples state
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [isLoadingSamples, setIsLoadingSamples] = useState<boolean>(true);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch precomputed samples on mount
  useEffect(() => {
    getSamples()
      .then((res) => {
        setSamples(res.samples || []);
      })
      .catch((err) => {
        console.error('Failed to load precomputed samples:', err);
      })
      .finally(() => {
        setIsLoadingSamples(false);
      });
  }, []);

  // Validate video file: format, size (<= 50MB), duration (<= 15s)
  const validateAndSetFile = (file: File) => {
    setSubmitError(null);

    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    const validExtensions = ['.mp4', '.mov', '.webm'];
    const validMimes = ['video/mp4', 'video/quicktime', 'video/webm'];

    const isFormatValid =
      validExtensions.includes(ext) || validMimes.includes(file.type.toLowerCase());
    const sizeBytes = file.size;
    const formattedSize = `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
    const isSizeValid = sizeBytes <= 50 * 1024 * 1024; // 50 MB limit

    if (!isFormatValid) {
      setSelectedFile(file);
      setFileValidation({
        isValidating: false,
        sizeBytes,
        formattedSize,
        isSizeValid,
        durationSeconds: null,
        formattedDuration: null,
        isDurationValid: false,
        format: ext.toUpperCase().replace('.', '') || 'UNKNOWN',
        isFormatValid: false,
        errorMessage: 'Invalid format. Only MP4, MOV, and WebM video files are supported.',
      });
      return;
    }

    if (!isSizeValid) {
      setSelectedFile(file);
      setFileValidation({
        isValidating: false,
        sizeBytes,
        formattedSize,
        isSizeValid: false,
        durationSeconds: null,
        formattedDuration: null,
        isDurationValid: false,
        format: ext.toUpperCase().replace('.', '') || 'VIDEO',
        isFormatValid: true,
        errorMessage: `File size (${formattedSize}) exceeds the 50 MB maximum limit.`,
      });
      return;
    }

    // Set preliminary state while reading duration
    setSelectedFile(file);
    setFileValidation({
      isValidating: true,
      sizeBytes,
      formattedSize,
      isSizeValid: true,
      durationSeconds: null,
      formattedDuration: null,
      isDurationValid: false,
      format: ext.toUpperCase().replace('.', '') || 'VIDEO',
      isFormatValid: true,
      errorMessage: null,
    });

    // Inspect video duration via offscreen <video> element
    const objectUrl = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.preload = 'metadata';

    let resolved = false;
    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        URL.revokeObjectURL(objectUrl);
        setFileValidation({
          isValidating: false,
          sizeBytes,
          formattedSize,
          isSizeValid: true,
          durationSeconds: null,
          formattedDuration: null,
          isDurationValid: false,
          format: ext.toUpperCase().replace('.', '') || 'VIDEO',
          isFormatValid: true,
          errorMessage: 'Timeout reading video metadata. Please verify the file is playable.',
        });
      }
    }, 4000);

    video.onloadedmetadata = () => {
      if (resolved) return;
      resolved = true;
      clearTimeout(timeout);
      URL.revokeObjectURL(objectUrl);

      const duration = video.duration;
      // Allow up to 15.5s to accommodate subtle timestamp padding
      const isDurationValid = duration > 0 && duration <= 15.5;
      const formattedDuration = `${duration.toFixed(1)}s`;

      setFileValidation({
        isValidating: false,
        sizeBytes,
        formattedSize,
        isSizeValid: true,
        durationSeconds: duration,
        formattedDuration,
        isDurationValid,
        format: ext.toUpperCase().replace('.', '') || 'VIDEO',
        isFormatValid: true,
        errorMessage: isDurationValid
          ? null
          : `Video length (${formattedDuration}) exceeds the 15-second maximum limit.`,
      });
    };

    video.onerror = () => {
      if (resolved) return;
      resolved = true;
      clearTimeout(timeout);
      URL.revokeObjectURL(objectUrl);

      setFileValidation({
        isValidating: false,
        sizeBytes,
        formattedSize,
        isSizeValid: true,
        durationSeconds: null,
        formattedDuration: null,
        isDurationValid: false,
        format: ext.toUpperCase().replace('.', '') || 'VIDEO',
        isFormatValid: true,
        errorMessage: 'Unable to decode video metadata. The file may be corrupt or encoded with an unsupported codec.',
      });
    };

    video.src = objectUrl;
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleClearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedFile(null);
    setFileValidation(null);
    setSubmitError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleResetDefaults = () => {
    setAlpha(15.0);
    setLowHz(0.2);
    setHighHz(3.0);
  };

  const isFormValid =
    selectedFile &&
    fileValidation &&
    !fileValidation.isValidating &&
    fileValidation.isFormatValid &&
    fileValidation.isSizeValid &&
    fileValidation.isDurationValid;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isFormValid || !selectedFile) return;

    setIsSubmitting(true);
    setSubmitError(null);

    try {
      const res = await createJob(selectedFile, {
        alpha,
        low_hz: lowHz,
        high_hz: highHz,
      });
      // Navigate to /job/:id on successful submission
      navigate(`/job/${res.job_id}`);
    } catch (err: any) {
      setSubmitError(err.message || 'Failed to submit video job.');
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-12">
      {/* Upload & Form Section */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Drag & Drop Zone + Form */}
        <div className="lg:col-span-7 glass-card rounded-2xl p-6 sm:p-8 space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5 text-cyan-400">
              <Film size={20} />
              <h2 className="text-lg font-semibold tracking-wide uppercase text-slate-200">
                Upload Video Footage
              </h2>
            </div>
            <span className="text-[11px] font-mono text-slate-400 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/60">
              MP4, MOV, WEBM &middot; &le; 50 MB &middot; &le; 15s
            </span>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Drag & Drop Zone */}
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
                isDragging
                  ? 'border-cyan-400 bg-cyan-950/40 ring-4 ring-cyan-500/20 scale-[1.01]'
                  : selectedFile && isFormValid
                  ? 'border-emerald-500/70 bg-emerald-950/20 shadow-lg shadow-emerald-500/10'
                  : selectedFile && fileValidation?.errorMessage
                  ? 'border-rose-500/70 bg-rose-950/20 shadow-lg shadow-rose-500/10'
                  : 'border-slate-700/80 hover:border-cyan-500/50 hover:bg-dark-900/50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp4,.mov,.webm,video/mp4,video/quicktime,video/webm"
                className="hidden"
                onChange={handleFileChange}
                disabled={isSubmitting}
              />

              <div
                className={`w-14 h-14 mx-auto rounded-full flex items-center justify-center mb-3 transition-colors ${
                  selectedFile && isFormValid
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : selectedFile && fileValidation?.errorMessage
                    ? 'bg-rose-500/20 text-rose-400'
                    : 'bg-cyan-500/10 text-cyan-400'
                }`}
              >
                {selectedFile && isFormValid ? (
                  <CheckCircle2 size={28} />
                ) : selectedFile && fileValidation?.errorMessage ? (
                  <AlertCircle size={28} />
                ) : (
                  <UploadCloud size={28} />
                )}
              </div>

              <p className="text-sm font-medium text-slate-200">
                {selectedFile
                  ? selectedFile.name
                  : isDragging
                  ? 'Drop video file now'
                  : 'Click to select or drag & drop video clip'}
              </p>
              <p className="text-xs text-slate-400 mt-1.5">
                Compatible with MP4, MOV, and WebM &middot; Max file size 50 MB &middot; Max length 15 seconds
              </p>
            </div>

            {/* Validation Details Card (shown before upload) */}
            {selectedFile && fileValidation && (
              <div className="bg-dark-900/80 rounded-xl p-4 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between text-xs pb-2 border-b border-slate-800/80">
                  <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                    <FileVideo size={14} className="text-cyan-400" />
                    File Validation Status
                  </span>
                  <button
                    type="button"
                    onClick={handleClearFile}
                    className="text-slate-400 hover:text-rose-400 transition-colors flex items-center gap-1 text-[11px]"
                  >
                    <X size={12} />
                    Remove file
                  </button>
                </div>

                {/* Validation Metrics Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs">
                  {/* Format pill */}
                  <div
                    className={`p-2.5 rounded-lg border flex items-center justify-between ${
                      fileValidation.isFormatValid
                        ? 'bg-slate-800/60 border-slate-700/60 text-slate-200'
                        : 'bg-rose-950/30 border-rose-800/40 text-rose-300'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <Film size={13} className="text-slate-400" />
                      <span>Format</span>
                    </div>
                    <span className="font-mono font-semibold flex items-center gap-1">
                      {fileValidation.format}
                      {fileValidation.isFormatValid ? (
                        <CheckCircle2 size={13} className="text-emerald-400" />
                      ) : (
                        <AlertCircle size={13} className="text-rose-400" />
                      )}
                    </span>
                  </div>

                  {/* Size pill */}
                  <div
                    className={`p-2.5 rounded-lg border flex items-center justify-between ${
                      fileValidation.isSizeValid
                        ? 'bg-slate-800/60 border-slate-700/60 text-slate-200'
                        : 'bg-rose-950/30 border-rose-800/40 text-rose-300'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <HardDrive size={13} className="text-slate-400" />
                      <span>Size (Max 50MB)</span>
                    </div>
                    <span className="font-mono font-semibold flex items-center gap-1">
                      {fileValidation.formattedSize}
                      {fileValidation.isSizeValid ? (
                        <CheckCircle2 size={13} className="text-emerald-400" />
                      ) : (
                        <AlertCircle size={13} className="text-rose-400" />
                      )}
                    </span>
                  </div>

                  {/* Duration pill */}
                  <div
                    className={`p-2.5 rounded-lg border flex items-center justify-between ${
                      fileValidation.isValidating
                        ? 'bg-slate-800/60 border-slate-700/60 text-slate-300'
                        : fileValidation.isDurationValid
                        ? 'bg-slate-800/60 border-slate-700/60 text-slate-200'
                        : 'bg-rose-950/30 border-rose-800/40 text-rose-300'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <Clock size={13} className="text-slate-400" />
                      <span>Length (Max 15s)</span>
                    </div>
                    <span className="font-mono font-semibold flex items-center gap-1">
                      {fileValidation.isValidating ? (
                        <Loader2 size={13} className="animate-spin text-cyan-400" />
                      ) : (
                        <>
                          {fileValidation.formattedDuration || 'N/A'}
                          {fileValidation.isDurationValid ? (
                            <CheckCircle2 size={13} className="text-emerald-400" />
                          ) : (
                            <AlertCircle size={13} className="text-rose-400" />
                          )}
                        </>
                      )}
                    </span>
                  </div>
                </div>

                {/* Validation Error Message if any */}
                {fileValidation.errorMessage && (
                  <div className="flex items-center gap-2 p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs">
                    <AlertCircle size={15} className="flex-shrink-0" />
                    <span>{fileValidation.errorMessage}</span>
                  </div>
                )}
              </div>
            )}

            {/* Submission Error Banner */}
            {submitError && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs">
                <AlertCircle size={16} className="flex-shrink-0" />
                <span>{submitError}</span>
              </div>
            )}

            {/* Collapsible "Advanced settings" */}
            <div className="bg-dark-900/60 rounded-xl border border-slate-800/80 overflow-hidden">
              <button
                type="button"
                onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
                className="w-full p-4 flex items-center justify-between text-left hover:bg-slate-800/40 transition-colors cursor-pointer"
              >
                <div className="flex items-center gap-2">
                  <Sliders size={16} className="text-cyan-400" />
                  <span className="text-sm font-semibold text-slate-200">
                    Advanced settings
                  </span>
                  <span className="text-[11px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded border border-slate-700/60">
                    &alpha;={alpha.toFixed(0)} &middot; {lowHz.toFixed(1)}–{highHz.toFixed(1)} Hz
                  </span>
                </div>
                <div className="text-slate-400">
                  {isAdvancedOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                </div>
              </button>

              {isAdvancedOpen && (
                <div className="p-5 pt-1 border-t border-slate-800/80 space-y-5">
                  <div className="flex items-center justify-between text-xs text-slate-400 pt-2">
                    <span>Tune Eulerian temporal magnification filter parameters</span>
                    <button
                      type="button"
                      onClick={handleResetDefaults}
                      className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 text-[11px] transition-colors"
                    >
                      <RotateCcw size={12} />
                      Reset defaults
                    </button>
                  </div>

                  {/* Amplification Factor Alpha Slider */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center">
                        <span className="text-slate-300 font-medium">Amplification Factor (&alpha;)</span>
                        <Tooltip text="Controls the motion magnification multiplier (1–50); higher values reveal fainter movement but can amplify noise." />
                      </div>
                      <span className="font-mono text-cyan-400 font-semibold text-sm">
                        {alpha.toFixed(0)}&times;
                      </span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="50"
                      step="1"
                      value={alpha}
                      onChange={(e) => setAlpha(parseFloat(e.target.value))}
                      disabled={isSubmitting}
                      className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] font-mono text-slate-500">
                      <span>1&times; (Subtle)</span>
                      <span>15&times; (Default)</span>
                      <span>50&times; (Extreme)</span>
                    </div>
                  </div>

                  {/* Frequency Band Sliders (Low Hz and High Hz) */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 pt-1">
                    {/* Low Hz */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center">
                          <span className="text-slate-300 font-medium">Low Cutoff (f_low)</span>
                          <Tooltip text="Lower frequency bound in Hz (0.1–2 Hz); filters out ultra-slow drifts and ambient illumination changes." />
                        </div>
                        <span className="font-mono text-cyan-400 font-semibold">
                          {lowHz.toFixed(2)} Hz
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0.1"
                        max="2.0"
                        step="0.05"
                        value={lowHz}
                        onChange={(e) => setLowHz(parseFloat(e.target.value))}
                        disabled={isSubmitting}
                        className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] font-mono text-slate-500">
                        <span>0.1 Hz</span>
                        <span>0.2 Hz (Def)</span>
                        <span>2.0 Hz</span>
                      </div>
                    </div>

                    {/* High Hz */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center">
                          <span className="text-slate-300 font-medium">High Cutoff (f_high)</span>
                          <Tooltip text="Upper frequency bound in Hz (1–10 Hz); filters out high-frequency sensor noise and rapid camera vibrations." />
                        </div>
                        <span className="font-mono text-cyan-400 font-semibold">
                          {highHz.toFixed(1)} Hz
                        </span>
                      </div>
                      <input
                        type="range"
                        min="1.0"
                        max="10.0"
                        step="0.1"
                        value={highHz}
                        onChange={(e) => setHighHz(parseFloat(e.target.value))}
                        disabled={isSubmitting}
                        className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] font-mono text-slate-500">
                        <span>1.0 Hz</span>
                        <span>3.0 Hz (Def)</span>
                        <span>10.0 Hz</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={!isFormValid || isSubmitting}
              className="w-full py-3.5 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold text-sm shadow-lg shadow-cyan-500/20 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 group cursor-pointer"
            >
              {isSubmitting ? (
                <>
                  <Loader2 size={16} className="animate-spin text-white" />
                  <span>Enqueuing Video Job...</span>
                </>
              ) : (
                <>
                  <Play size={16} className="fill-current group-hover:scale-110 transition-transform" />
                  <span>Break Camouflage with Motion Amplification</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Right Column: Architectural Overview */}
        <div className="lg:col-span-5 glass-card rounded-2xl p-6 sm:p-8 space-y-5">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono font-medium bg-cyan-950/60 text-cyan-400 border border-cyan-800/50">
            <Sparkles size={13} />
            Two-Stream Gated Fusion Architecture
          </div>
          <h3 className="text-xl font-bold text-white tracking-tight">
            Revealing Camouflage via Eulerian Motion Magnification
          </h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            Organisms and concealed assets evade classical detectors by matching ambient color, texture, and boundary gradients. Our pipeline isolates physiological micro-motion to break concealment.
          </p>
          <ul className="space-y-3.5 text-xs text-slate-300">
            <li className="flex items-start gap-2.5">
              <span className="w-5 h-5 rounded-md bg-cyan-950 text-cyan-400 flex items-center justify-center flex-shrink-0 font-mono text-[10px] font-bold mt-0.5 border border-cyan-800/40">
                1
              </span>
              <span>
                <strong>Spatial Decomposition:</strong> Frames are filtered into multi-scale Laplacian pyramid sub-bands to preserve structural details.
              </span>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-5 h-5 rounded-md bg-cyan-950 text-cyan-400 flex items-center justify-center flex-shrink-0 font-mono text-[10px] font-bold mt-0.5 border border-cyan-800/40">
                2
              </span>
              <span>
                <strong>Temporal Bandpass (&alpha;):</strong> Butterworth filtering selectively amplifies biological tremor and breathing within target frequency bands.
              </span>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-5 h-5 rounded-md bg-cyan-950 text-cyan-400 flex items-center justify-center flex-shrink-0 font-mono text-[10px] font-bold mt-0.5 border border-cyan-800/40">
                3
              </span>
              <span>
                <strong>Dense Optical Flow & Fusion:</strong> Dense motion vectors derived from amplified sequences are fused with appearance features to segment the target.
              </span>
            </li>
          </ul>
        </div>
      </section>

      {/* "Try a sample" Gallery Section (from getSamples()) */}
      <section className="space-y-5">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold tracking-wide uppercase text-slate-200">
              Try a sample
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Select precomputed footage to inspect motion amplification and camouflage breaking directly without waiting
            </p>
          </div>
          <span className="text-xs font-mono text-cyan-400 bg-cyan-950/40 px-2.5 py-1 rounded-md border border-cyan-800/40">
            {samples.length} Precomputed {samples.length === 1 ? 'Demo' : 'Demos'}
          </span>
        </div>

        {isLoadingSamples ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3].map((sk) => (
              <div
                key={sk}
                className="glass-card rounded-xl overflow-hidden border border-slate-800 animate-pulse"
              >
                <div className="aspect-video bg-slate-800/60" />
                <div className="p-4 space-y-3">
                  <div className="h-4 bg-slate-800/80 rounded w-2/3" />
                  <div className="h-3 bg-slate-800/60 rounded w-full" />
                  <div className="h-3 bg-slate-800/40 rounded w-4/5" />
                  <div className="pt-2 border-t border-slate-800/60 flex justify-between">
                    <div className="h-3 bg-slate-800/60 rounded w-16" />
                    <div className="h-3 bg-slate-800/60 rounded w-24" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : samples.length === 0 ? (
          <div className="glass-card rounded-2xl p-12 text-center text-slate-400 space-y-3 border border-slate-800">
            <Film size={36} className="mx-auto text-slate-600" />
            <h3 className="text-sm font-semibold text-slate-300">No Demo Samples Available</h3>
            <p className="text-xs text-slate-500 max-w-sm mx-auto">
              Precomputed demo clips can be placed in the samples/ directory on the server to appear here automatically.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {samples.map((sample) => (
              <div
                key={sample.id}
                onClick={() => navigate(`/job/${sample.id}`)}
                className="glass-card glass-card-hover rounded-xl overflow-hidden flex flex-col group cursor-pointer border border-slate-800/80 hover:border-cyan-500/50 transition-all duration-300"
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
                      <Film size={36} />
                    </div>
                  )}

                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-60 group-hover:opacity-40 transition-opacity" />

                  <span className="absolute top-2.5 right-2.5 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-emerald-950/80 text-emerald-400 border border-emerald-800/60 backdrop-blur-md">
                    Precomputed
                  </span>

                  <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                    <div className="w-11 h-11 rounded-full bg-cyan-500 text-white flex items-center justify-center shadow-lg shadow-cyan-500/30 scale-90 group-hover:scale-100 transition-transform">
                      <Play size={18} className="fill-current ml-0.5" />
                    </div>
                  </div>
                </div>

                {/* Body */}
                <div className="p-4 flex-1 flex flex-col justify-between space-y-3">
                  <div>
                    <h3 className="font-semibold text-white text-base group-hover:text-cyan-400 transition-colors">
                      {sample.title}
                    </h3>
                    <p className="text-xs text-slate-400 mt-1 line-clamp-2">
                      Multi-stream breakdown: raw footage, baseline model prediction, Eulerian amplified motion, and optical flow vectors.
                    </p>
                  </div>

                  <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
                    <span className="text-cyan-400 font-semibold">Instant view &rarr;</span>
                    <span className="text-slate-500">5 synchronized streams</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Quantitative Metrics & Benchmark Section */}
      <section>
        <MetricsPanel />
      </section>

      {/* "How It Works" & "Limitations" Technical Documentation */}
      <section>
        <AboutSection />
      </section>
    </div>
  );
};
