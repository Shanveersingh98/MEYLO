import React, { useState } from 'react';
import {
  Menu,
  Settings,
  Sun,
  Moon,
  Laptop,
  Activity,
  ShieldCheck,
  Globe,
  ChevronDown,
  BarChart3
} from 'lucide-react';
import { ModelInfo, SystemStatus, ThemeMode } from '../../types';
import { ModelSelector } from './ModelSelector';
import { WebSearchToggle } from './WebSearchToggle';

interface HeaderProps {
  onToggleSidebar: () => void;
  onOpenSettings: () => void;
  onOpenCapabilities?: () => void;
  activeModel: 'custom' | 'external';
  models: ModelInfo[];
  onSelectModel: (model: 'custom' | 'external') => Promise<void>;
  isSwitchingModel: boolean;
  webSearchEnabled: boolean;
  onToggleWebSearch: (enabled: boolean) => Promise<void>;
  themeMode: ThemeMode;
  onSetThemeMode: (mode: ThemeMode) => void;
  status?: SystemStatus | null;
}

export const Header: React.FC<HeaderProps> = ({
  onToggleSidebar,
  onOpenSettings,
  onOpenCapabilities,
  activeModel,
  models,
  onSelectModel,
  isSwitchingModel,
  webSearchEnabled,
  onToggleWebSearch,
  themeMode,
  onSetThemeMode,
  status,
}) => {
  const [themeDropdownOpen, setThemeDropdownOpen] = useState(false);

  return (
    <header className="h-16 px-4 border-b border-slate-200 dark:border-dark-800 bg-white/80 dark:bg-dark-900/80 backdrop-blur-md flex items-center justify-between sticky top-0 z-30 transition-colors">
      {/* Left section: Hamburger for Mobile + Model Selector + Web Search Toggle */}
      <div className="flex items-center gap-2.5">
        <button
          type="button"
          onClick={onToggleSidebar}
          className="p-2 rounded-xl text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-dark-800 lg:hidden transition-colors"
          title="Toggle conversations sidebar"
        >
          <Menu className="w-5 h-5" />
        </button>

        <ModelSelector
          activeModel={activeModel}
          models={models}
          onSelectModel={onSelectModel}
          isSwitching={isSwitchingModel}
          onOpenCapabilities={onOpenCapabilities}
        />

        <WebSearchToggle
          enabled={webSearchEnabled}
          onToggle={onToggleWebSearch}
        />

        {/* Mode Indicator Pill (Hidden on smaller screens) */}
        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono border border-slate-200 dark:border-dark-750 bg-slate-50 dark:bg-dark-850 text-slate-600 dark:text-slate-300">
          {webSearchEnabled ? (
            <>
              <Globe className="w-3 h-3 text-accent-cyan" />
              <span>Mode: Online (Web Context)</span>
            </>
          ) : (
            <>
              <ShieldCheck className="w-3 h-3 text-emerald-500" />
              <span>Mode: Offline (Local Only)</span>
            </>
          )}
        </div>

        {/* Capabilities Button */}
        {onOpenCapabilities && (
          <button
            type="button"
            onClick={onOpenCapabilities}
            className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border border-slate-200 dark:border-dark-750 bg-white dark:bg-dark-850 hover:bg-slate-100 dark:hover:bg-dark-800 text-slate-700 dark:text-slate-300 transition-colors shadow-sm"
            title="Open Verified Model Capability Index"
          >
            <BarChart3 className="w-3.5 h-3.5 text-violet-500 dark:text-violet-400" />
            <span>Capabilities</span>
          </button>
        )}
      </div>

      {/* Right section: Hardware status badge, 3-Way Theme switcher, Settings */}
      <div className="flex items-center gap-2">
        {/* Hardware Status pill (hidden on small screens) */}
        {status?.hardware && (
          <button
            type="button"
            onClick={onOpenSettings}
            className="hidden xl:flex items-center gap-2 px-3 py-1 rounded-xl bg-slate-100 dark:bg-dark-850 hover:bg-slate-200 dark:hover:bg-dark-800 border border-slate-200 dark:border-dark-750 text-xs font-mono text-slate-700 dark:text-slate-300 transition-colors shadow-sm"
            title="Click to view hardware telemetry in Settings"
          >
            <Activity className="w-3.5 h-3.5 text-emerald-500" />
            <span>RAM: {status.hardware.ram_percent}%</span>
            {status.hardware.cuda_available && (
              <>
                <span className="text-slate-300 dark:text-slate-700">|</span>
                <span className="text-brand-600 dark:text-brand-300">
                  GPU VRAM: {status.hardware.vram_used_mb}MB
                </span>
              </>
            )}
          </button>
        )}

        {/* 3-Way Theme Dropdown */}
        <div className="relative">
          <button
            type="button"
            onClick={() => setThemeDropdownOpen(!themeDropdownOpen)}
            className="flex items-center gap-1.5 p-2 rounded-xl text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-dark-800 border border-transparent hover:border-slate-200 dark:hover:border-dark-750 transition-colors"
            title={`Current theme: ${themeMode.toUpperCase()}`}
          >
            {themeMode === 'light' ? (
              <Sun className="w-4 h-4 text-amber-500" />
            ) : themeMode === 'dark' ? (
              <Moon className="w-4 h-4 text-brand-400" />
            ) : (
              <Laptop className="w-4 h-4 text-slate-500 dark:text-slate-400" />
            )}
            <ChevronDown className="w-3 h-3 text-slate-400" />
          </button>

          {themeDropdownOpen && (
            <>
              <div
                className="fixed inset-0 z-40"
                onClick={() => setThemeDropdownOpen(false)}
              />
              <div className="absolute right-0 mt-2 w-40 rounded-xl border border-slate-200 dark:border-dark-750 bg-white dark:bg-dark-900 shadow-xl p-1.5 z-50 animate-slide-up text-xs font-medium">
                <button
                  type="button"
                  onClick={() => {
                    onSetThemeMode('light');
                    setThemeDropdownOpen(false);
                  }}
                  className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left transition-colors ${
                    themeMode === 'light'
                      ? 'bg-brand-50 dark:bg-brand-500/15 text-brand-700 dark:text-brand-300 font-semibold'
                      : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-dark-800'
                  }`}
                >
                  <Sun className="w-4 h-4 text-amber-500" />
                  <span>Light Mode</span>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    onSetThemeMode('dark');
                    setThemeDropdownOpen(false);
                  }}
                  className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left transition-colors ${
                    themeMode === 'dark'
                      ? 'bg-brand-50 dark:bg-brand-500/15 text-brand-700 dark:text-brand-300 font-semibold'
                      : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-dark-800'
                  }`}
                >
                  <Moon className="w-4 h-4 text-brand-400" />
                  <span>Dark Mode</span>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    onSetThemeMode('system');
                    setThemeDropdownOpen(false);
                  }}
                  className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left transition-colors ${
                    themeMode === 'system'
                      ? 'bg-brand-50 dark:bg-brand-500/15 text-brand-700 dark:text-brand-300 font-semibold'
                      : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-dark-800'
                  }`}
                >
                  <Laptop className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                  <span>System Default</span>
                </button>
              </div>
            </>
          )}
        </div>

        {/* Settings Button */}
        <button
          type="button"
          onClick={onOpenSettings}
          className="p-2 rounded-xl text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-dark-800 border border-transparent hover:border-slate-200 dark:hover:border-dark-750 transition-colors"
          title="Open Settings & Diagnostics"
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
};
