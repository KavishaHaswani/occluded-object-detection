import React from 'react';
import { Eye, Activity, Cpu } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="app-header">
      <div className="brand">
        <div className="brand-icon">
          <Eye size={22} color="#ffffff" />
        </div>
        <div>
          <h1 className="brand-title">CAMOUBREAK</h1>
          <div className="brand-subtitle">Motion-Amplified Camouflage Breaking</div>
        </div>
      </div>

      <div className="header-badges">
        <span className="badge badge-cyan">
          <Activity size={13} />
          Eulerian Magnification
        </span>
        <span className="badge badge-green">
          <Cpu size={13} />
          Two-Stream Gated Fusion
        </span>
      </div>
    </header>
  );
};
