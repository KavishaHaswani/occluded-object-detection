import React from 'react';
import { Header } from './Header';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen flex flex-col bg-dark-950 text-slate-100 relative selection:bg-cyan-500/20 selection:text-cyan-300">
      {/* Background ambient glow */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-cyan-600/10 rounded-full blur-3xl" />
        <div className="absolute top-1/3 -right-40 w-96 h-96 bg-purple-600/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 left-1/3 w-96 h-96 bg-teal-600/10 rounded-full blur-3xl" />
      </div>

      <Header />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {children}
      </main>

      <footer className="border-t border-slate-900 bg-dark-900/40 py-6 text-center text-xs text-slate-500 relative z-10">
        Motion-Amplified Camouflage Breaking &middot; Computer Vision Final Project
      </footer>
    </div>
  );
};
