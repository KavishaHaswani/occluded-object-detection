import {
  BarChart3,
  TrendingUp,
  AlertCircle,
  Sparkles,
} from 'lucide-react';
import metricsData from '../data/metrics.json';

export const MetricsPanel: React.FC = () => {
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
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Performance comparison on the {metricsData.dataset}
          </p>
        </div>

        {/* Placeholder Warning Badge */}
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-mono font-medium bg-amber-950/50 text-amber-300 border border-amber-800/60 self-start sm:self-auto">
          <AlertCircle size={13} className="text-amber-400" />
          <span>{metricsData.notice}</span>
        </div>
      </div>

      {/* Metrics Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-mono text-[11px] uppercase tracking-wider">
              <th className="pb-3 font-semibold">Evaluation Metric</th>
              <th className="pb-3 font-semibold text-center">Appearance Baseline</th>
              <th className="pb-3 font-semibold text-center text-cyan-400">Motion-Amplified (Ours)</th>
              <th className="pb-3 font-semibold text-right">Relative Delta</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-sans">
            {metricsData.metrics.map((m) => (
              <tr key={m.id} className="hover:bg-slate-800/30 transition-colors group">
                {/* Metric Name & Description */}
                <td className="py-3.5 pr-4">
                  <div className="font-semibold text-slate-200 flex items-center gap-1.5">
                    <span>{m.name}</span>
                    <span className="font-mono text-[11px] text-slate-500 font-normal">
                      ({m.symbol})
                    </span>
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
                  <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-950/60 text-emerald-400 border border-emerald-800/50">
                    <TrendingUp size={11} />
                    {m.relative_delta}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary Banner */}
      <div className="p-4 rounded-xl bg-dark-900/80 border border-slate-800 flex items-start gap-3">
        <Sparkles size={16} className="text-cyan-400 flex-shrink-0 mt-0.5" />
        <div className="text-xs space-y-1">
          <span className="font-semibold text-slate-200">Benchmark Takeaway:</span>
          <p className="text-slate-400 leading-relaxed">
            {metricsData.summary.key_takeaway}
          </p>
        </div>
      </div>
    </div>
  );
};
