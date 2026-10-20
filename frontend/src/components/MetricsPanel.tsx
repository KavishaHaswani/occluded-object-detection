import React, { useState } from 'react';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Sparkles,
} from 'lucide-react';
import testReport from '../data/test_report.json';
import baselineReport from '../data/baseline_test_report.json';

export const MetricsPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overall' | 'cad'>('overall');

  // Extract metrics with IoU and Dice first
  const overallMetrics = [
    {
      id: 'iou',
      name: 'Intersection over Union',
      symbol: 'mIoU',
      description: 'Area of overlap between predicted camouflage mask and ground truth mask divided by area of union.',
      direction: 'higher_better',
      baseline: baselineReport.overall.IoU,
      amplified: testReport.overall.IoU,
    },
    {
      id: 'dice',
      name: 'Dice Coefficient',
      symbol: 'Dice',
      description: 'Harmonic mean of boundary precision and recall quantifying mask contour accuracy.',
      direction: 'higher_better',
      baseline: baselineReport.overall.Dice,
      amplified: testReport.overall.Dice,
    },
    {
      id: 'mae',
      name: 'Mean Absolute Error',
      symbol: 'MAE',
      description: 'Average pixel-wise error across video frames (lower is better; reflects false-positive clutter).',
      direction: 'lower_better',
      baseline: baselineReport.overall.MAE,
      amplified: testReport.overall.MAE,
    },
    {
      id: 's_measure',
      name: 'Structure-measure',
      symbol: 'S_α',
      description: 'Standard camouflage evaluation metric measuring region-aware and object-aware structural similarity.',
      direction: 'higher_better',
      baseline: baselineReport.overall.S_measure,
      amplified: testReport.overall.S_measure,
    },
    {
      id: 'weighted_f',
      name: 'Weighted F-measure',
      symbol: 'F_β^w',
      description: 'Precision-recall balance adjusted with spatial distance weighting to penalize boundary bleed.',
      direction: 'higher_better',
      baseline: baselineReport.overall.weighted_F,
      amplified: testReport.overall.weighted_F,
    },
  ];

  const cadMetrics = [
    {
      id: 'iou',
      name: 'Intersection over Union',
      symbol: 'mIoU',
      description: 'Area of overlap between predicted camouflage mask and ground truth mask divided by area of union.',
      direction: 'higher_better',
      baseline: baselineReport.by_source.cad.IoU,
      amplified: testReport.by_source.cad.IoU,
    },
    {
      id: 'dice',
      name: 'Dice Coefficient',
      symbol: 'Dice',
      description: 'Harmonic mean of boundary precision and recall quantifying mask contour accuracy.',
      direction: 'higher_better',
      baseline: baselineReport.by_source.cad.Dice,
      amplified: testReport.by_source.cad.Dice,
    },
    {
      id: 'mae',
      name: 'Mean Absolute Error',
      symbol: 'MAE',
      description: 'Average pixel-wise error across video frames (lower is better; reflects false-positive clutter).',
      direction: 'lower_better',
      baseline: baselineReport.by_source.cad.MAE,
      amplified: testReport.by_source.cad.MAE,
    },
    {
      id: 's_measure',
      name: 'Structure-measure',
      symbol: 'S_α',
      description: 'Standard camouflage evaluation metric measuring region-aware and object-aware structural similarity.',
      direction: 'higher_better',
      baseline: baselineReport.by_source.cad.S_measure,
      amplified: testReport.by_source.cad.S_measure,
    },
    {
      id: 'weighted_f',
      name: 'Weighted F-measure',
      symbol: 'F_β^w',
      description: 'Precision-recall balance adjusted with spatial distance weighting to penalize boundary bleed.',
      direction: 'higher_better',
      baseline: baselineReport.by_source.cad.weighted_F,
      amplified: testReport.by_source.cad.weighted_F,
    },
  ];

  const currentMetrics = activeTab === 'overall' ? overallMetrics : cadMetrics;

  return (
    <div className="glass-card rounded-2xl p-6 sm:p-8 space-y-6 border border-slate-800">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
        <div>
          <div className="flex items-center gap-2 text-cyan-400">
            <BarChart3 size={18} />
            <h3 className="text-base font-semibold text-slate-100 tracking-wide uppercase">
              Evaluation Metrics & Benchmark
            </h3>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-800/60 font-semibold">
              v1 model
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Test split performance from <code className="text-slate-300 font-mono">test_report.json</code> ({testReport.overall.n_frames} frames)
          </p>
        </div>

        {/* Tab switcher */}
        <div className="flex items-center gap-2">
          <div className="flex bg-slate-900/90 rounded-lg p-1 border border-slate-800 text-xs">
            <button
              type="button"
              onClick={() => setActiveTab('overall')}
              className={`px-3 py-1 rounded-md font-medium transition-colors ${
                activeTab === 'overall'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Overall Split (911f)
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('cad')}
              className={`px-3 py-1 rounded-md font-medium transition-colors ${
                activeTab === 'cad'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              CAD Subset (71f)
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-mono text-[11px] uppercase tracking-wider">
              <th className="pb-3 font-semibold">Evaluation Metric (IoU &amp; Dice First)</th>
              <th className="pb-3 font-semibold text-center">Appearance Baseline (runs_raw)</th>
              <th className="pb-3 font-semibold text-center text-cyan-400">v1 model (runs_amp)</th>
              <th className="pb-3 font-semibold text-right">Delta</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-sans">
            {currentMetrics.map((m) => {
              const deltaVal = m.amplified - m.baseline;
              const pctDelta = m.baseline !== 0 ? (deltaVal / m.baseline) * 100 : 0;
              const isPositive = m.direction === 'higher_better' ? deltaVal > 0 : deltaVal < 0;
              const formattedDelta = m.baseline !== 0
                ? `${pctDelta > 0 ? '+' : ''}${pctDelta.toFixed(1)}%`
                : `${deltaVal > 0 ? '+' : ''}${deltaVal.toFixed(3)}`;

              return (
                <tr key={m.id} className="hover:bg-slate-800/30 transition-colors group">
                  {/* Metric Name & Description */}
                  <td className="py-3.5 pr-4">
                    <div className="font-semibold text-slate-200 flex items-center gap-1.5">
                      <span>{m.name}</span>
                      <span className="font-mono text-[11px] text-slate-500 font-normal">
                        ({m.symbol})
                      </span>
                      {['iou', 'dice'].includes(m.id) && (
                        <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-cyan-950/80 text-cyan-300 border border-cyan-800/50">
                          Primary
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-slate-400 mt-0.5 line-clamp-1 group-hover:line-clamp-none transition-all">
                      {m.description}
                    </p>
                  </td>

                  {/* Baseline Score */}
                  <td className="py-3.5 px-4 text-center font-mono text-slate-400 font-medium">
                    {m.baseline.toFixed(3)}
                  </td>

                  {/* Amplified Score */}
                  <td className="py-3.5 px-4 text-center font-mono font-bold text-cyan-300 bg-cyan-950/20 rounded-md">
                    {m.amplified.toFixed(3)}
                  </td>

                  {/* Relative Delta Badge */}
                  <td className="py-3.5 pl-4 text-right font-mono">
                    <span
                      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold ${
                        isPositive
                          ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/50'
                          : 'bg-slate-800/60 text-slate-300 border border-slate-700/50'
                      }`}
                    >
                      {isPositive ? <TrendingUp size={11} /> : <TrendingDown size={11} />}
                      {formattedDelta}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Summary Banner */}
      <div className="p-4 rounded-xl bg-dark-900/80 border border-slate-800 flex items-start gap-3">
        <Sparkles size={16} className="text-cyan-400 flex-shrink-0 mt-0.5" />
        <div className="text-xs space-y-1">
          <span className="font-semibold text-slate-200">v1 model Test Report Summary:</span>
          <p className="text-slate-400 leading-relaxed">
            Metrics loaded directly from <code className="text-slate-300 font-mono">test_report.json</code>. On the CAD moving-animal subset (71 frames), motion amplification achieves <strong className="text-cyan-300">{testReport.by_source.cad.IoU.toFixed(3)} IoU</strong> vs <strong className="text-slate-300">{baselineReport.by_source.cad.IoU.toFixed(3)} baseline</strong> (+171% relative gain). Overall test split reflects training before the dataset mask-labeling bugfix.
          </p>
        </div>
      </div>
    </div>
  );
};
