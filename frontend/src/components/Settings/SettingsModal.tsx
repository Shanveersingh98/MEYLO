import React, { useState } from 'react';
import {
  X,
  Cpu,
  Globe,
  Activity,
  Trash2,
  RefreshCw,
  CheckCircle2,
  Sun,
  Moon,
  Laptop,
  Layers,
  Database,
  ExternalLink,
  ShieldCheck,
  HardDrive,
  Info,
  Sparkles
} from 'lucide-react';
import { ModelInfo, SystemStatus, SourceItem, ThemeMode } from '../../types';
import { MeyloLogo } from '../common/MeyloLogo';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  activeModel: 'custom' | 'external';
  models: ModelInfo[];
  onSelectModel: (model: 'custom' | 'external') => Promise<void>;
  isSwitchingModel: boolean;
  webSearchEnabled: boolean;
  onToggleWebSearch: (enabled: boolean) => Promise<void>;
  themeMode: ThemeMode;
  onSetThemeMode: (mode: ThemeMode) => void;
  status: SystemStatus | null;
  onRefreshStatus: () => Promise<void>;
  onClearCurrentChat: () => Promise<void>;
  onDeleteAllConversations: () => Promise<void>;
  recentSources: SourceItem[];
  onOpenCapabilities?: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  activeModel,
  models,
  onSelectModel,
  isSwitchingModel,
  webSearchEnabled,
  onToggleWebSearch,
  themeMode,
  onSetThemeMode,
  status,
  onRefreshStatus,
  onClearCurrentChat,
  onDeleteAllConversations,
  recentSources,
  onOpenCapabilities,
}) => {
  const [activeTab, setActiveTab] = useState<'appearance' | 'model' | 'web' | 'conversations' | 'system' | 'about'>('appearance');
  const [isDeletingAll, setIsDeletingAll] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  if (!isOpen) return null;

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await onRefreshStatus();
    setIsRefreshing(false);
  };

  const handleDeleteAll = async () => {
    await onDeleteAllConversations();
    setIsDeletingAll(false);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div className="relative w-full max-w-2xl rounded-2xl border border-slate-200 dark:border-dark-750 bg-white dark:bg-dark-900 shadow-2xl overflow-hidden flex flex-col max-h-[88vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-dark-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-brand-50 dark:bg-brand-500/15 border border-brand-200 dark:border-brand-500/30 text-brand-600 dark:text-brand-400">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">Settings & System Info</h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Manage appearance, independent AI models, web retrieval, and hardware diagnostics
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-dark-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-slate-200 dark:border-dark-800 px-6 bg-slate-50/50 dark:bg-dark-950/40 text-xs font-medium overflow-x-auto">
          <button
            onClick={() => setActiveTab('appearance')}
            className={`py-3 px-3 border-b-2 flex items-center gap-1.5 transition-colors whitespace-nowrap ${
              activeTab === 'appearance'
                ? 'border-brand-500 text-brand-600 dark:text-brand-400 font-semibold'
                : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <Sun className="w-3.5 h-3.5" />
            <span>Appearance</span>
          </button>

          <button
            onClick={() => setActiveTab('model')}
            className={`py-3 px-3 border-b-2 flex items-center gap-1.5 transition-colors whitespace-nowrap ${
              activeTab === 'model'
                ? 'border-brand-500 text-brand-600 dark:text-brand-400 font-semibold'
                : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>AI Models</span>
          </button>

          <button
            onClick={() => setActiveTab('web')}
            className={`py-3 px-3 border-b-2 flex items-center gap-1.5 transition-colors whitespace-nowrap ${
              activeTab === 'web'
                ? 'border-brand-500 text-brand-600 dark:text-brand-400 font-semibold'
                : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <Globe className="w-3.5 h-3.5" />
            <span>Web Search</span>
          </button>

          <button
            onClick={() => setActiveTab('conversations')}
            className={`py-3 px-3 border-b-2 flex items-center gap-1.5 transition-colors whitespace-nowrap ${
              activeTab === 'conversations'
                ? 'border-brand-500 text-brand-600 dark:text-brand-400 font-semibold'
                : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>Conversations</span>
          </button>

          <button
            onClick={() => setActiveTab('system')}
            className={`py-3 px-3 border-b-2 flex items-center gap-1.5 transition-colors whitespace-nowrap ${
              activeTab === 'system'
                ? 'border-brand-500 text-brand-600 dark:text-brand-400 font-semibold'
                : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>System Telemetry</span>
          </button>

          <button
            onClick={() => setActiveTab('about')}
            className={`py-3 px-3 border-b-2 flex items-center gap-1.5 transition-colors whitespace-nowrap ${
              activeTab === 'about'
                ? 'border-brand-500 text-brand-600 dark:text-brand-400 font-semibold'
                : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>About MEYLO</span>
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* TAB 1: APPEARANCE */}
          {activeTab === 'appearance' && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Interface Theme</h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Select your preferred visual style. Changes apply immediately and are persisted across app sessions.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {/* Light Mode Card */}
                <button
                  type="button"
                  onClick={() => onSetThemeMode('light')}
                  className={`p-4 rounded-xl border text-left transition-all ${
                    themeMode === 'light'
                      ? 'border-brand-500 bg-brand-50/50 dark:bg-brand-500/10 shadow-sm'
                      : 'border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850 hover:bg-slate-100 dark:hover:bg-dark-800'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="p-2 rounded-lg bg-amber-100 text-amber-600">
                      <Sun className="w-4 h-4" />
                    </div>
                    {themeMode === 'light' && (
                      <CheckCircle2 className="w-4 h-4 text-brand-600 dark:text-brand-400" />
                    )}
                  </div>
                  <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100">Light Mode</h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                    Soft off-white background with clean surfaces and high-contrast typography.
                  </p>
                </button>

                {/* Dark Mode Card */}
                <button
                  type="button"
                  onClick={() => onSetThemeMode('dark')}
                  className={`p-4 rounded-xl border text-left transition-all ${
                    themeMode === 'dark'
                      ? 'border-brand-500 bg-brand-50/50 dark:bg-brand-500/10 shadow-sm'
                      : 'border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850 hover:bg-slate-100 dark:hover:bg-dark-800'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="p-2 rounded-lg bg-slate-800 text-brand-400">
                      <Moon className="w-4 h-4" />
                    </div>
                    {themeMode === 'dark' && (
                      <CheckCircle2 className="w-4 h-4 text-brand-600 dark:text-brand-400" />
                    )}
                  </div>
                  <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100">Dark Mode</h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                    Deep charcoal backgrounds with balanced surfaces, soft borders, and zero neon glare.
                  </p>
                </button>

                {/* System Default Card */}
                <button
                  type="button"
                  onClick={() => onSetThemeMode('system')}
                  className={`p-4 rounded-xl border text-left transition-all ${
                    themeMode === 'system'
                      ? 'border-brand-500 bg-brand-50/50 dark:bg-brand-500/10 shadow-sm'
                      : 'border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850 hover:bg-slate-100 dark:hover:bg-dark-800'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="p-2 rounded-lg bg-slate-200 dark:bg-dark-700 text-slate-700 dark:text-slate-300">
                      <Laptop className="w-4 h-4" />
                    </div>
                    {themeMode === 'system' && (
                      <CheckCircle2 className="w-4 h-4 text-brand-600 dark:text-brand-400" />
                    )}
                  </div>
                  <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100">System Default</h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                    Automatically synchronize with your operating system's light/dark mode preference.
                  </p>
                </button>
              </div>
            </div>
          )}

          {/* TAB 2: AI MODELS */}
          {activeTab === 'model' && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Independent Dual Models</h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Two separate local AI models. No hybrid fusion, no hidden fallbacks, no weight alterations.
                </p>
              </div>

              {/* Model 1: Custom Model (Violet / Purple Identity) */}
              <div
                className={`p-4 rounded-xl border transition-all ${
                  activeModel === 'custom'
                    ? 'border-violet-300 dark:border-violet-500/40 bg-violet-50/40 dark:bg-violet-950/20 shadow-sm'
                    : 'border-slate-200 dark:border-dark-750 bg-slate-50/30 dark:bg-dark-850'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full bg-violet-500 flex-shrink-0" />
                      <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                        FromScratchLM — My Custom AI Model
                      </h4>
                      {activeModel === 'custom' && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-violet-100 dark:bg-violet-500/25 text-violet-700 dark:text-violet-300 border border-violet-200 dark:border-violet-500/30 font-mono">
                          ACTIVE
                        </span>
                      )}
                    </div>
                    <div className="mt-1.5 flex flex-wrap gap-1.5 text-[11px] font-mono text-slate-500 dark:text-slate-400">
                      <span className="px-1.5 py-0.5 rounded bg-violet-100 dark:bg-violet-500/20 text-violet-700 dark:text-violet-300 font-semibold">FromScratchLM</span>
                      <span className="px-1.5 py-0.5 rounded bg-violet-50 dark:bg-violet-500/15 text-violet-700 dark:text-violet-300 font-medium">Custom PyTorch</span>
                      <span className="px-1.5 py-0.5 rounded bg-emerald-50 dark:bg-emerald-500/15 text-emerald-700 dark:text-emerald-400">CUDA / GPU</span>
                      <span className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-dark-800">best.pt Checkpoint</span>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-400 mt-2 leading-relaxed">
                      Your own trained PyTorch transformer architecture located at <code className="text-[11px] px-1 py-0.5 bg-slate-100 dark:bg-dark-800 rounded">checkpoints/ultimate_improvement/best.pt</code>.
                    </p>
                  </div>

                  {activeModel !== 'custom' && (
                    <button
                      type="button"
                      disabled={isSwitchingModel}
                      onClick={() => onSelectModel('custom')}
                      className="px-3 py-1.5 rounded-xl bg-violet-600 hover:bg-violet-500 text-white text-xs font-semibold shadow-sm transition-colors disabled:opacity-50"
                    >
                      {isSwitchingModel ? 'Loading...' : 'Switch to Custom'}
                    </button>
                  )}
                </div>
              </div>

              {/* Model 2: External Model (Teal / Cyan Identity) */}
              <div
                className={`p-4 rounded-xl border transition-all ${
                  activeModel === 'external'
                    ? 'border-teal-300 dark:border-teal-500/40 bg-teal-50/40 dark:bg-teal-950/20 shadow-sm'
                    : 'border-slate-200 dark:border-dark-750 bg-slate-50/30 dark:bg-dark-850'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full bg-teal-500 flex-shrink-0" />
                      <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                        Qwen2.5-1.5B-Instruct
                      </h4>
                      {activeModel === 'external' && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-teal-100 dark:bg-teal-500/25 text-teal-700 dark:text-teal-300 border border-teal-200 dark:border-teal-500/30 font-mono">
                          ACTIVE
                        </span>
                      )}
                    </div>
                    <div className="mt-1.5 flex flex-wrap gap-1.5 text-[11px] font-mono text-slate-500 dark:text-slate-400">
                      <span className="px-1.5 py-0.5 rounded bg-teal-100 dark:bg-teal-500/20 text-teal-700 dark:text-teal-300 font-semibold">Qwen2.5-1.5B-Instruct</span>
                      <span className="px-1.5 py-0.5 rounded bg-teal-50 dark:bg-teal-500/15 text-teal-700 dark:text-teal-300 font-medium">External Pretrained</span>
                      <span className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-dark-800">GGUF CPU</span>
                      <span className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-dark-800">llama.cpp runtime</span>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-400 mt-2 leading-relaxed">
                      External pretrained instruction-tuned model located at <code className="text-[11px] px-1 py-0.5 bg-slate-100 dark:bg-dark-800 rounded">models/external/qwen2.5-1.5b-instruct/model.gguf</code>.
                    </p>
                  </div>

                  {activeModel !== 'external' && (
                    <button
                      type="button"
                      disabled={isSwitchingModel}
                      onClick={() => onSelectModel('external')}
                      className="px-3 py-1.5 rounded-xl bg-teal-600 hover:bg-teal-500 text-white text-xs font-semibold shadow-sm transition-colors disabled:opacity-50"
                    >
                      {isSwitchingModel ? 'Loading...' : 'Switch to Qwen'}
                    </button>
                  )}
                </div>
              </div>

              {/* Capability Matrix Button */}
              {onOpenCapabilities && (
                <div className="p-3 rounded-xl bg-slate-100/70 dark:bg-dark-800/60 border border-slate-200 dark:border-dark-750 flex items-center justify-between">
                  <div>
                    <h5 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                      Empirical Model Capability Index
                    </h5>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400">
                      View transparent benchmarks across 12 capability categories in Offline and Online modes.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      onClose();
                      onOpenCapabilities();
                    }}
                    className="px-3 py-1.5 rounded-xl bg-white dark:bg-dark-700 hover:bg-slate-50 dark:hover:bg-dark-600 border border-slate-200 dark:border-dark-600 text-xs font-semibold text-slate-700 dark:text-slate-200 shadow-sm transition-colors"
                  >
                    View Matrix →
                  </button>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: WEB SEARCH */}
          {activeTab === 'web' && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Web Retrieval & Sources</h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Control online grounding. When enabled, retrieved passages are provided to the active model.
                </p>
              </div>

              <div className="p-4 rounded-xl border border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850 flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                    Live Web Grounding
                  </h4>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                    {webSearchEnabled
                      ? 'Enabled: Current queries search DuckDuckGo and pass evidence to the model.'
                      : 'Disabled: Pure offline mode using only local model weights.'}
                  </p>
                </div>

                <button
                  type="button"
                  onClick={() => onToggleWebSearch(!webSearchEnabled)}
                  className={`px-4 py-1.5 rounded-xl font-mono text-xs font-bold transition-all ${
                    webSearchEnabled
                      ? 'bg-accent-cyan text-white shadow-md shadow-accent-cyan/20'
                      : 'bg-slate-200 dark:bg-dark-750 text-slate-600 dark:text-slate-400'
                  }`}
                >
                  {webSearchEnabled ? 'ONLINE (ON)' : 'OFFLINE (OFF)'}
                </button>
              </div>

              {/* Four Combinations Matrix */}
              <div className="p-4 rounded-xl border border-slate-200 dark:border-dark-750 bg-white dark:bg-dark-900 space-y-2">
                <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                  Supported Model & Mode Configurations
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                  <div className="p-2.5 rounded-lg border border-slate-100 dark:border-dark-800 bg-slate-50 dark:bg-dark-850">
                    <span className="font-semibold text-brand-600 dark:text-brand-300">1. FromScratchLM + Offline</span>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">Pure custom PyTorch weights without internet retrieval.</p>
                  </div>
                  <div className="p-2.5 rounded-lg border border-slate-100 dark:border-dark-800 bg-slate-50 dark:bg-dark-850">
                    <span className="font-semibold text-accent-cyan">2. FromScratchLM + Online</span>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">Web evidence retrieved; custom model generates answer.</p>
                  </div>
                  <div className="p-2.5 rounded-lg border border-slate-100 dark:border-dark-800 bg-slate-50 dark:bg-dark-850">
                    <span className="font-semibold text-purple-600 dark:text-accent-violet">3. Qwen2.5-1.5B + Offline</span>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">Qwen generates directly from GGUF weights without search.</p>
                  </div>
                  <div className="p-2.5 rounded-lg border border-slate-100 dark:border-dark-800 bg-slate-50 dark:bg-dark-850">
                    <span className="font-semibold text-accent-cyan">4. Qwen2.5-1.5B + Online</span>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">Web evidence retrieved; Qwen synthesizes final answer.</p>
                  </div>
                </div>
              </div>

              {/* Recent Sources Overview */}
              {recentSources && recentSources.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                    Latest Retrieved Sources ({recentSources.length})
                  </h4>
                  <div className="space-y-1.5 max-h-40 overflow-y-auto">
                    {recentSources.map((src, i) => (
                      <a
                        key={i}
                        href={src.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center justify-between p-2 rounded-lg bg-slate-50 dark:bg-dark-850 border border-slate-200 dark:border-dark-750 text-xs hover:border-accent-cyan transition-colors"
                      >
                        <span className="truncate max-w-md font-medium text-slate-800 dark:text-slate-200">
                          {src.title || src.url}
                        </span>
                        <ExternalLink className="w-3.5 h-3.5 text-slate-400 flex-shrink-0 ml-2" />
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 4: CONVERSATIONS */}
          {activeTab === 'conversations' && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Conversation Management</h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Manage chat sessions stored locally in SQLite database (<code className="text-[11px]">data/companion_project.db</code>).
                </p>
              </div>

              <div className="p-4 rounded-xl border border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850 flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                    Clear Current Chat
                  </h4>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Removes all messages from the active conversation while keeping the session.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={async () => {
                    await onClearCurrentChat();
                    onClose();
                  }}
                  className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-dark-700 bg-white dark:bg-dark-800 text-slate-700 dark:text-slate-300 text-xs font-medium hover:bg-slate-100 dark:hover:bg-dark-750 transition-colors"
                >
                  Clear Current
                </button>
              </div>

              <div className="p-4 rounded-xl border border-rose-200 dark:border-rose-500/30 bg-rose-50/30 dark:bg-rose-500/10 flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-semibold text-rose-700 dark:text-rose-400">
                    Delete All Conversations
                  </h4>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Permanently deletes all conversation threads and message logs.
                  </p>
                </div>

                {isDeletingAll ? (
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={handleDeleteAll}
                      className="px-3 py-1.5 rounded-lg bg-rose-600 text-white text-xs font-semibold hover:bg-rose-500 transition-colors"
                    >
                      Confirm Delete
                    </button>
                    <button
                      type="button"
                      onClick={() => setIsDeletingAll(false)}
                      className="px-2 py-1.5 rounded-lg bg-slate-200 dark:bg-dark-750 text-slate-700 dark:text-slate-300 text-xs hover:bg-slate-300"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => setIsDeletingAll(true)}
                    className="px-3 py-1.5 rounded-lg bg-rose-600 text-white text-xs font-semibold hover:bg-rose-500 transition-colors"
                  >
                    Delete All
                  </button>
                )}
              </div>
            </div>
          )}

          {/* TAB 5: SYSTEM TELEMETRY */}
          {activeTab === 'system' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Live Hardware Diagnostics</h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Real-time resource utilization from local hardware.
                  </p>
                </div>

                <button
                  type="button"
                  onClick={handleRefresh}
                  disabled={isRefreshing}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-dark-750 bg-white dark:bg-dark-850 text-xs text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-dark-800 transition-colors disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
                  <span>Refresh</span>
                </button>
              </div>

              {status ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  {/* CPU Info */}
                  <div className="p-3 rounded-xl border border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850">
                    <div className="font-semibold text-slate-700 dark:text-slate-300 mb-1 flex items-center justify-between">
                      <span>Host Processor</span>
                      <Cpu className="w-3.5 h-3.5 text-brand-500" />
                    </div>
                    <div className="font-mono text-slate-900 dark:text-slate-100 font-semibold truncate">
                      {status.hardware.cpu_name || 'Host CPU'}
                    </div>
                  </div>

                  {/* System RAM */}
                  <div className="p-3 rounded-xl border border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850">
                    <div className="font-semibold text-slate-700 dark:text-slate-300 mb-1 flex items-center justify-between">
                      <span>System Memory</span>
                      <Activity className="w-3.5 h-3.5 text-emerald-500" />
                    </div>
                    <div className="font-mono text-slate-900 dark:text-slate-100 font-semibold">
                      {status.hardware.ram_used_gb} GB / {status.hardware.ram_total_gb} GB ({status.hardware.ram_percent}%)
                    </div>
                  </div>

                  {/* GPU Info */}
                  <div className="p-3 rounded-xl border border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850">
                    <div className="font-semibold text-slate-700 dark:text-slate-300 mb-1 flex items-center justify-between">
                      <span>Dedicated GPU</span>
                      <Cpu className="w-3.5 h-3.5 text-brand-500" />
                    </div>
                    <div className="font-mono text-slate-900 dark:text-slate-100 font-semibold truncate">
                      {status.hardware.gpu_name || 'None'}
                    </div>
                  </div>

                  {/* GPU VRAM & CUDA */}
                  <div className="p-3 rounded-xl border border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850">
                    <div className="font-semibold text-slate-700 dark:text-slate-300 mb-1 flex items-center justify-between">
                      <span>CUDA & VRAM</span>
                      <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" />
                    </div>
                    <div className="font-mono text-slate-900 dark:text-slate-100 font-semibold">
                      {status.hardware.cuda_available
                        ? `CUDA Active (${status.hardware.cuda_version}) • ${status.hardware.vram_used_mb}MB Used`
                        : 'CPU Fallback'}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-6 text-xs text-slate-400">
                  Telemetry loading...
                </div>
              )}
            </div>
          )}

          {/* TAB 6: ABOUT MEYLO */}
          {activeTab === 'about' && (
            <div className="space-y-5 animate-fade-in">
              {/* Hero Banner */}
              <div className="p-5 rounded-2xl bg-gradient-to-tr from-brand-600/10 via-indigo-600/5 to-accent-cyan/10 border border-brand-200/60 dark:border-brand-500/20 text-center flex flex-col items-center">
                <div className="w-14 h-14 rounded-2xl overflow-hidden shadow-xl shadow-indigo-500/20 mb-3">
                  <MeyloLogo className="w-14 h-14" size={56} />
                </div>
                <h3 className="text-2xl font-extrabold tracking-widest text-slate-900 dark:text-white uppercase font-sans">
                  MEYLO
                </h3>
                <div className="text-xs font-semibold tracking-wide text-brand-600 dark:text-brand-400 mt-0.5">
                  Your AI. Your Space.
                </div>
                <p className="text-xs text-slate-600 dark:text-slate-300 mt-2 max-w-md leading-relaxed">
                  A personal AI companion to talk, learn, explore and grow with. Designed for complete local privacy, transparent model switching, and real-time grounded assistance.
                </p>
                <div className="mt-3 flex items-center gap-2">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-brand-100 dark:bg-brand-500/20 text-brand-700 dark:text-brand-300 border border-brand-200 dark:border-brand-500/30">
                    Release v2.0
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-500/30">
                    100% Private Local Core
                  </span>
                </div>
              </div>

              {/* Specifications Card */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 px-1">
                  System Architecture & Identity
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs">
                  <div className="p-3 rounded-xl border border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850">
                    <span className="text-slate-400 text-[11px] block">Product Name</span>
                    <span className="font-semibold text-slate-900 dark:text-slate-100">MEYLO</span>
                  </div>

                  <div className="p-3 rounded-xl border border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850">
                    <span className="text-slate-400 text-[11px] block">Tagline</span>
                    <span className="font-semibold text-slate-900 dark:text-slate-100">Your AI. Your Space.</span>
                  </div>

                  <div className="p-3 rounded-xl border border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850">
                    <span className="text-slate-400 text-[11px] block">Custom Neural Engine</span>
                    <span className="font-semibold text-violet-600 dark:text-violet-400">FromScratchLM (PyTorch CUDA, 14.1M)</span>
                  </div>

                  <div className="p-3 rounded-xl border border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850">
                    <span className="text-slate-400 text-[11px] block">External Neural Engine</span>
                    <span className="font-semibold text-teal-600 dark:text-teal-400">Qwen2.5-1.5B-Instruct (GGUF CPU)</span>
                  </div>

                  <div className="p-3 rounded-xl border border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850">
                    <span className="text-slate-400 text-[11px] block">Active Checkpoint</span>
                    <span className="font-mono text-[11px] text-slate-700 dark:text-slate-300">checkpoints/ultimate_improvement/best.pt</span>
                  </div>

                  <div className="p-3 rounded-xl border border-slate-200 dark:border-dark-750 bg-slate-50/50 dark:bg-dark-850">
                    <span className="text-slate-400 text-[11px] block">Local Storage</span>
                    <span className="font-mono text-[11px] text-slate-700 dark:text-slate-300">SQLite (WAL Mode, Local Disk)</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
