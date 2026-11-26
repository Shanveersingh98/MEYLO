import React from 'react';
import { Globe, Loader2, Wifi, WifiOff } from 'lucide-react';

interface WebSearchToggleProps {
  enabled: boolean;
  onToggle: (enabled: boolean) => Promise<void>;
  isLoading?: boolean;
}

export const WebSearchToggle: React.FC<WebSearchToggleProps> = ({
  enabled,
  onToggle,
  isLoading = false,
}) => {
  return (
    <button
      type="button"
      disabled={isLoading}
      onClick={() => onToggle(!enabled)}
      className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs sm:text-sm font-medium transition-all shadow-sm focus:outline-none disabled:opacity-60 ${
        enabled
          ? 'bg-cyan-50 dark:bg-accent-cyan/15 border-cyan-300 dark:border-accent-cyan/40 text-cyan-700 dark:text-accent-cyan hover:bg-cyan-100 dark:hover:bg-accent-cyan/25'
          : 'bg-white dark:bg-dark-850 border-slate-200 dark:border-dark-750 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-dark-800'
      }`}
      title={
        enabled
          ? 'Web Search is ON: Online Web Retrieval active with source citations'
          : 'Web Search is OFF: Offline Mode using only local neural model weights'
      }
    >
      {isLoading ? (
        <Loader2 className="w-3.5 h-3.5 animate-spin text-accent-cyan" />
      ) : enabled ? (
        <Wifi className="w-3.5 h-3.5 text-accent-cyan" />
      ) : (
        <WifiOff className="w-3.5 h-3.5 text-slate-400" />
      )}

      <span className="hidden md:inline font-medium">Search:</span>
      <span className={`font-mono text-xs font-semibold ${enabled ? 'text-accent-cyan' : 'text-slate-500 dark:text-slate-400'}`}>
        {enabled ? 'ONLINE' : 'OFFLINE'}
      </span>
      <span
        className={`w-2 h-2 rounded-full ${
          enabled ? 'bg-accent-cyan animate-pulse' : 'bg-slate-300 dark:bg-slate-600'
        }`}
      />
    </button>
  );
};
