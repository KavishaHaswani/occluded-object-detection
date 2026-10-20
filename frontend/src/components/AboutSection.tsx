import { BookOpen, Cpu, Activity, GitMerge, ShieldAlert } from 'lucide-react';

export const AboutSection: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* "How it works" Section (3 paragraphs) */}
      <div className="glass-card rounded-2xl p-6 sm:p-8 space-y-6 border border-slate-800">
        <div className="flex items-center gap-2 text-cyan-400 pb-3 border-b border-slate-800/80">
          <BookOpen size={18} />
          <h3 className="text-base font-semibold text-slate-100 tracking-wide uppercase">
            How It Works
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Paragraph 1: Eulerian Motion Amplification */}
          <div className="p-5 rounded-xl bg-dark-900/70 border border-slate-800/80 space-y-3">
            <div className="flex items-center gap-2.5 text-cyan-400 font-semibold text-sm">
              <Cpu size={16} />
              <h4>1. Eulerian Motion Magnification</h4>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Instead of tracking explicit points across space, our Eulerian approach inspects pixel intensity variation across consecutive sliding temporal windows. By decomposing frames into spatial Laplacian pyramid bands and applying a temporal Butterworth filter, subtle physiological micro-movements—such as breathing tremors, muscle twitches, and creeping motion—are magnified by a configurable factor &alpha;, revealing motion that human eyes and standard single-frame vision systems cannot perceive.
            </p>
          </div>

          {/* Paragraph 2: Optical Flow on Amplified Frames */}
          <div className="p-5 rounded-xl bg-dark-900/70 border border-slate-800/80 space-y-3">
            <div className="flex items-center gap-2.5 text-cyan-400 font-semibold text-sm">
              <Activity size={16} />
              <h4>2. Optical Flow on Amplified Frames</h4>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Dense optical flow (via Farneback quadratic polynomial expansion) is computed directly from the magnified video stream rather than raw video frames. Because micro-motion has been systematically exaggerated across the temporal window, the resulting motion vectors cleanly transcend background camera noise, yielding sharp boundary gradients and directional velocity heatmaps that outline the camouflaged asset.
            </p>
          </div>

          {/* Paragraph 3: Gated Two-Stream Fusion */}
          <div className="p-5 rounded-xl bg-dark-900/70 border border-slate-800/80 space-y-3">
            <div className="flex items-center gap-2.5 text-cyan-400 font-semibold text-sm">
              <GitMerge size={16} />
              <h4>3. Gated Two-Stream Fusion</h4>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              The neural network operates two parallel pathways: a spatial appearance stream processing RGB video textures, and a temporal motion stream ingesting the magnified optical flow vectors. A cross-modal gated attention module dynamically weights and fuses features from both streams, allowing the network to rely on motion cues precisely where visual texture and background contrast fail to separate the camouflaged subject.
            </p>
          </div>
        </div>
      </div>

      {/* "Limitations" Note */}
      <div className="p-6 rounded-2xl bg-amber-950/20 border border-amber-800/40 text-amber-200/90 space-y-4">
        <div className="flex items-center gap-2.5 text-amber-300 font-semibold text-sm">
          <ShieldAlert size={18} className="text-amber-400 flex-shrink-0" />
          <h4 className="uppercase tracking-wider text-xs">Architectural & Experimental Limitations (v1 model)</h4>
        </div>

        {/* Mandatory v1 model limitation disclosure */}
        <div className="p-4 rounded-xl bg-amber-900/30 border border-amber-700/50 text-amber-100 font-medium text-xs leading-relaxed">
          This model was trained before a mask-labeling bug was fixed, so its detections are limited. A retrained version is planned.
        </div>

        <ul className="space-y-3 text-xs text-slate-300/90 leading-relaxed pl-1">
          <li className="flex items-start gap-2.5">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 flex-shrink-0" />
            <span>
              <strong className="text-amber-200">Linear (Not Phase-Based) Amplification:</strong> Our pipeline employs linear Eulerian magnification (first-order Taylor series approximation). While computationally efficient, linear amplification is susceptible to ringing artifacts, clipping, and sensor noise magnification at extreme amplification values (&alpha; &gt; 30), unlike complex steerable pyramid phase-based methods.
            </span>
          </li>
          <li className="flex items-start gap-2.5">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 flex-shrink-0" />
            <span>
              <strong className="text-amber-200">Farneback Dense Flow Approximation:</strong> Optical flow is computed using classical Farneback polynomial expansion to maintain CPU/GPU portability without heavy deep learning flow dependencies like RAFT. Severe camera shake or sudden parallax shifts may generate spurious global motion vectors.
            </span>
          </li>
          <li className="flex items-start gap-2.5">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 flex-shrink-0" />
            <span>
              <strong className="text-amber-200">Baseline Model Formulation:</strong> To ensure an exact apples-to-apples ablation, the appearance baseline utilizes the identical two-stream neural network architecture with motion flow tensors zeroed out (<strong>F</strong> = <strong>0</strong>), isolating the exact performance delta contributed by Eulerian motion magnification.
            </span>
          </li>
        </ul>
      </div>
    </div>
  );
};
