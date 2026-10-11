import React from 'react';
import { Link } from 'react-router-dom';
import { Eye, Sparkles } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="border-b border-slate-800/80 bg-dark-900/80 backdrop-blur-xl sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <Link to="/" className="flex items-start sm:items-center gap-3.5 group">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 group-hover:scale-105 transition-transform flex-shrink-0">
            <Eye size={22} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg sm:text-xl font-bold tracking-tight text-white group-hover:text-cyan-400 transition-colors">
              Motion-Amplified Camouflage Breaking
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 leading-tight">
              Finding objects that hide in plain sight by amplifying the motion they can't hide.
            </p>
          </div>
        </Link>

        <div className="flex items-center gap-2 self-start md:self-auto">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-medium bg-cyan-950/60 text-cyan-400 border border-cyan-800/50">
            <Sparkles size={13} />
            Eulerian VCOD
          </span>
        </div>
      </div>
    </header>
  );
};
