import React, { useState } from 'react';
import { Cpu, ChevronDown, Check, Loader2, Sparkles, ShieldCheck } from 'lucide-react';
import { ModelInfo } from '../../types';

interface ModelSelectorProps {
  activeModel: 'custom' | 'external';
  models: ModelInfo[];
  onSelectModel: (model: 'custom' | 'external') => Promise<void>;
  isSwitching: boolean;
  onOpenCapabilities?: () => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  activeModel,
  models,
  onSelectModel,
  isSwitching,
  onOpenCapabilities,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const customInfo = models.find(m => m.id === 'custom') || {
    name: 'FromScratchLM — My Custom AI Model',
    device: 'CUDA / GPU',
    checkpoint: 'checkpoints/ultimate_improvement/best.pt',
  };

  const externalInfo = models.find(m => m.id === 'external') || {
    name: 'Qwen2.5-1.5B-Instruct',
    device: 'CPU Multi-thread',
    checkpoint: 'models/external/qwen2.5-1.5b-instruct/model.gguf',
  };

  const handleSelect = async (modelId: 'custom' | 'external') => {
    if (modelId === activeModel || isSwitching) return;
    setIsOpen(false);
    await onSelectModel(modelId);
  };

  return (
    <div className="relative inline-block text-left">
      <button
        type="button"
        disabled={isSwitching}
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs sm:text-sm font-medium transition-all shadow-sm focus:outline-none disabled:opacity-60 ${
          activeModel === 'custom'
            ? 'border-violet-200 dark:border-violet-500/30 bg-violet-50/60 dark:bg-violet-950/20 hover:bg-violet-100/60 dark:hover:bg-violet-900/30 text-violet-950 dark:text-violet-100'
            : 'border-teal-200 dark:border-teal-500/30 bg-teal-50/60 dark:bg-teal-950/20 hover:bg-teal-100/60 dark:hover:bg-teal-900/30 text-teal-950 dark:text-teal-100'
        }`}
        title="Switch between Custom PyTorch (Violet) and External Pretrained (Teal) models"
      >
        <div className="flex items-center gap-2">
          {isSwitching ? (
            <Loader2 className={`w-4 h-4 animate-spin ${activeModel === 'custom' ? 'text-violet-500' : 'text-teal-500'}`} />
          ) : (
            <Cpu className={`w-4 h-4 ${activeModel === 'custom' ? 'text-violet-600 dark:text-violet-400' : 'text-teal-600 dark:text-teal-400'}`} />
          )}

          <div className="flex flex-col text-left">
            <span className="font-semibold leading-none">
              {activeModel === 'custom' ? 'FromScratchLM' : 'Qwen2.5-1.5B'}
            </span>
            <span className={`text-[10px] font-mono mt-0.5 font-medium ${
              activeModel === 'custom' ? 'text-violet-700 dark:text-violet-300' : 'text-teal-700 dark:text-teal-300'
            }`}>
              {activeModel === 'custom' ? 'Custom • PyTorch CUDA' : 'External • GGUF CPU'}
            </span>
          </div>
        </div>

        <ChevronDown
          className={`w-3.5 h-3.5 transition-transform ml-1 ${
            activeModel === 'custom' ? 'text-violet-400 dark:text-violet-400' : 'text-teal-400 dark:text-teal-400'
          } ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute left-0 mt-2 w-84 sm:w-96 rounded-2xl border border-slate-200 dark:border-dark-750 bg-white dark:bg-dark-900 shadow-2xl p-2 z-50 animate-slide-up">
            <div className="px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 border-b border-slate-100 dark:border-dark-800 flex items-center justify-between">
              <span>Select Independent Model</span>
              <span className="text-[10px] font-mono lowercase">no hybrid switching</span>
            </div>

            <div className="mt-1 space-y-1.5">
              {/* Custom Model Option (Violet / Purple Accent) */}
              <button
                type="button"
                onClick={() => handleSelect('custom')}
                disabled={isSwitching}
                className={`w-full flex items-start justify-between p-3 rounded-xl text-left transition-all ${
                  activeModel === 'custom'
                    ? 'bg-violet-50 dark:bg-violet-500/15 border border-violet-300 dark:border-violet-500/40 shadow-sm'
                    : 'hover:bg-slate-50 dark:hover:bg-dark-800 border border-transparent'
                }`}
              >
                <div className="flex-1 pr-2">
                  <div className="flex items-center gap-1.5 font-semibold text-xs sm:text-sm text-slate-900 dark:text-slate-100">
                    <span className="w-2.5 h-2.5 rounded-full bg-violet-500 flex-shrink-0" />
                    <span>FromScratchLM — My Custom AI Model</span>
                  </div>
                  <div className="mt-1 flex items-center gap-1.5">
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-violet-100 dark:bg-violet-500/25 text-violet-700 dark:text-violet-300 border border-violet-200 dark:border-violet-500/30 font-semibold">
                      FromScratchLM
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-violet-50 dark:bg-violet-500/15 text-violet-700 dark:text-violet-300 border border-violet-200 dark:border-violet-500/20 font-medium">
                      Custom PyTorch
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-emerald-50 dark:bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/30">
                      CUDA / GPU
                    </span>
                  </div>
                  <p className="mt-1.5 text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
                    Trained autoregressive transformer. Uses your custom checkpoint (<code className="text-[10px]">checkpoints/ultimate_improvement/best.pt</code>).
                  </p>
                </div>
                {activeModel === 'custom' && (
                  <Check className="w-4 h-4 text-violet-600 dark:text-violet-400 flex-shrink-0 mt-1" />
                )}
              </button>

              {/* External Model Option (Teal / Cyan Accent) */}
              <button
                type="button"
                onClick={() => handleSelect('external')}
                disabled={isSwitching}
                className={`w-full flex items-start justify-between p-3 rounded-xl text-left transition-all ${
                  activeModel === 'external'
                    ? 'bg-teal-50 dark:bg-teal-500/15 border border-teal-300 dark:border-teal-500/40 shadow-sm'
                    : 'hover:bg-slate-50 dark:hover:bg-dark-800 border border-transparent'
                }`}
              >
                <div className="flex-1 pr-2">
                  <div className="flex items-center gap-1.5 font-semibold text-xs sm:text-sm text-slate-900 dark:text-slate-100">
                    <span className="w-2.5 h-2.5 rounded-full bg-teal-500 flex-shrink-0" />
                    <span>Qwen2.5-1.5B-Instruct</span>
                  </div>
                  <div className="mt-1 flex items-center gap-1.5">
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-teal-100 dark:bg-teal-500/25 text-teal-700 dark:text-teal-300 border border-teal-200 dark:border-teal-500/30 font-semibold">
                      Qwen2.5-1.5B-Instruct
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-teal-50 dark:bg-teal-500/15 text-teal-700 dark:text-teal-300 border border-teal-200 dark:border-teal-500/20 font-medium">
                      External Pretrained
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-100 dark:bg-dark-750 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-dark-700">
                      GGUF CPU
                    </span>
                  </div>
                  <p className="mt-1.5 text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
                    GGUF Q4_K_M runtime via portable llama.cpp engine. High general-knowledge accuracy on CPU.
                  </p>
                </div>
                {activeModel === 'external' && (
                  <Check className="w-4 h-4 text-teal-600 dark:text-teal-400 flex-shrink-0 mt-1" />
                )}
              </button>
            </div>

            <div className="mt-2 pt-2 border-t border-slate-100 dark:border-dark-800 px-2 py-1 text-[10.5px] text-slate-400 dark:text-slate-500 flex items-center justify-between">
              <span>Switching is hardware-verified</span>
              {onOpenCapabilities ? (
                <button
                  type="button"
                  onClick={() => {
                    setIsOpen(false);
                    onOpenCapabilities();
                  }}
                  className="text-violet-600 dark:text-violet-400 hover:underline font-medium"
                >
                  View Capability Matrix →
                </button>
              ) : (
                <span>Inactive model unloaded</span>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};
