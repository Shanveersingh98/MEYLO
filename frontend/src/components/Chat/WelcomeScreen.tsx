import React from 'react';
import {
  Brain,
  Cpu,
  Globe,
  MessageSquare,
  BookOpen,
  Calculator,
  History,
  HeartHandshake,
  ShieldCheck,
  CheckCircle2,
  Sparkles
} from 'lucide-react';
import { MeyloLogo } from '../common/MeyloLogo';

interface WelcomeScreenProps {
  activeModel: 'custom' | 'external';
  webSearchEnabled: boolean;
  onOpenCapabilities?: () => void;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  activeModel,
  webSearchEnabled,
  onOpenCapabilities,
}) => {
  const capabilities = [
    {
      title: 'Natural AI Conversations',
      description: 'Engage in open-ended dialogues with contextual understanding and balanced responses.',
      icon: <MessageSquare className="w-4 h-4 text-brand-500 dark:text-brand-400" />,
    },
    {
      title: 'Educational & Technical Assistance',
      description: 'Explore programming, algorithms, system concepts, and scientific principles.',
      icon: <BookOpen className="w-4 h-4 text-accent-cyan" />,
    },
    {
      title: 'Mathematical Reasoning',
      description: 'Step-by-step problem solving executed directly through the selected neural engine.',
      icon: <Calculator className="w-4 h-4 text-accent-violet" />,
    },
    {
      title: 'Context-Aware Conversations',
      description: 'Maintains conversational memory across turns for coherent follow-up questions.',
      icon: <History className="w-4 h-4 text-accent-emerald" />,
    },
    {
      title: 'Emotionally Supportive Conversations',
      description: 'Respectful, empathetic, and encouraging interactions for everyday dialogue.',
      icon: <HeartHandshake className="w-4 h-4 text-accent-rose" />,
    },
    {
      title: 'Offline AI Mode',
      description: 'Local neural execution without network dependencies, preserving total privacy.',
      icon: <ShieldCheck className="w-4 h-4 text-accent-emerald" />,
    },
    {
      title: 'Online Web Research',
      description: 'Live information retrieval with verified source attribution when enabled.',
      icon: <Globe className="w-4 h-4 text-accent-cyan" />,
    },
    {
      title: 'Independent Model Selection',
      description: 'Toggle between custom PyTorch FromScratchLM and external Qwen2.5-1.5B.',
      icon: <Cpu className="w-4 h-4 text-brand-500 dark:text-brand-400" />,
    },
  ];

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 max-w-4xl mx-auto text-center animate-fade-in select-none">
      {/* MEYLO Brand Icon Header */}
      <div className="relative mb-4">
        <div className="w-16 h-16 rounded-2xl overflow-hidden shadow-2xl shadow-indigo-500/25 flex items-center justify-center p-1 bg-gradient-to-tr from-brand-600 to-accent-cyan">
          <MeyloLogo className="w-14 h-14" size={56} />
        </div>
        <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-emerald-500 border-2 border-white dark:border-dark-950 flex items-center justify-center">
          <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
        </div>
      </div>

      <h1 className="text-3xl sm:text-4xl font-extrabold tracking-widest text-slate-900 dark:text-white uppercase font-sans">
        MEYLO
      </h1>
      <p className="mt-1 text-sm font-semibold tracking-wide text-brand-600 dark:text-brand-400">
        Your AI. Your Space.
      </p>
      <p className="mt-2 text-xs sm:text-sm text-slate-600 dark:text-slate-400 max-w-xl leading-relaxed">
        A personal AI companion to talk, learn, explore and grow with. Seamlessly toggle between your trained PyTorch neural model (<span className="text-violet-600 dark:text-violet-400 font-semibold">FromScratchLM</span>) and the external pretrained model (<span className="text-teal-600 dark:text-teal-400 font-semibold">Qwen2.5-1.5B</span>).
      </p>

      {/* Model & Web Status Badges */}
      <div className="mt-4 flex flex-wrap items-center justify-center gap-2.5 text-xs">
        <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border font-mono shadow-sm ${
          activeModel === 'custom'
            ? 'bg-violet-50 dark:bg-violet-500/15 border-violet-200 dark:border-violet-500/30 text-violet-700 dark:text-violet-300'
            : 'bg-teal-50 dark:bg-teal-500/15 border-teal-200 dark:border-teal-500/30 text-teal-700 dark:text-teal-300'
        }`}>
          <Cpu className="w-3.5 h-3.5" />
          <span>Active:</span>
          <strong className="font-semibold">
            {activeModel === 'custom' ? 'FromScratchLM (Custom PyTorch, CUDA)' : 'Qwen2.5-1.5B-Instruct (GGUF CPU)'}
          </strong>
        </span>

        <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-100 dark:bg-dark-850 border border-slate-200 dark:border-dark-750 text-slate-700 dark:text-slate-300 font-mono shadow-sm">
          <Globe className="w-3.5 h-3.5 text-accent-cyan" />
          <span>Mode:</span>
          <strong className={webSearchEnabled ? 'text-accent-cyan font-semibold' : 'text-slate-500 dark:text-slate-400'}>
            {webSearchEnabled ? 'Online (Web Grounded)' : 'Offline (Local Weights)'}
          </strong>
        </span>

        {onOpenCapabilities && (
          <button
            type="button"
            onClick={onOpenCapabilities}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-100 hover:bg-slate-200 dark:bg-dark-850 dark:hover:bg-dark-800 border border-slate-200 dark:border-dark-750 text-slate-700 dark:text-slate-300 text-xs font-medium transition-colors"
          >
            <span>📊 Verified Capabilities Matrix</span>
          </button>
        )}
      </div>

      {/* Capabilities Presentation (Informative, NOT clickable question prompts) */}
      <div className="mt-8 w-full text-left">
        <div className="flex items-center justify-between mb-3 px-1">
          <span className="text-xs font-semibold tracking-wider uppercase text-slate-500 dark:text-slate-400">
            System Capabilities & Core Features
          </span>
          <span className="text-[11px] text-slate-400 dark:text-slate-500 font-mono">
            Dual Inference Engine
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
          {capabilities.map((item) => (
            <div
              key={item.title}
              className="p-3 rounded-xl border border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-900/60 shadow-sm hover:border-slate-300 dark:hover:border-dark-750 transition-colors"
            >
              <div className="flex items-center gap-2 mb-1.5">
                <div className="p-1 rounded-md bg-slate-100 dark:bg-dark-800 flex-shrink-0">
                  {item.icon}
                </div>
                <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100 leading-tight">
                  {item.title}
                </h3>
              </div>
              <p className="text-[11.5px] text-slate-500 dark:text-slate-400 leading-snug">
                {item.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
