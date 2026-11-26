import React from 'react';
import { Terminal, Cpu, Globe, Trash2, HelpCircle, Power } from 'lucide-react';
import { SlashCommand } from '../../types';

interface SlashCommandMenuProps {
  filter: string;
  onSelect: (command: string) => void;
  onClose: () => void;
}

const COMMANDS: SlashCommand[] = [
  { command: '/model custom', description: 'Switch to Custom Trained Model (FromScratchLM, CUDA)', usage: '/model custom', category: 'model' },
  { command: '/model external', description: 'Switch to External Model (Qwen2.5-1.5B-Instruct, CPU)', usage: '/model external', category: 'model' },
  { command: '/model status', description: 'Show active model, device, RAM and VRAM telemetry', usage: '/model status', category: 'model' },
  { command: '/model list', description: 'List available models and their readiness status', usage: '/model list', category: 'model' },
  { command: '/web on', description: 'Enable online web search retrieval with source citations', usage: '/web on', category: 'web' },
  { command: '/web off', description: 'Disable web search (pure offline local neural mode)', usage: '/web off', category: 'web' },
  { command: '/web status', description: 'Check whether web search is ON or OFF', usage: '/web status', category: 'web' },
  { command: '/sources', description: 'Display clickable sources from last online response', usage: '/sources', category: 'web' },
  { command: '/clear', description: 'Clear messages in current conversation thread', usage: '/clear', category: 'chat' },
  { command: '/help', description: 'Display full command reference guide and usage notes', usage: '/help', category: 'system' },
  { command: '/exit', description: 'Gracefully shutdown the MEYLO background server', usage: '/exit', category: 'system' },
];

export const SlashCommandMenu: React.FC<SlashCommandMenuProps> = ({
  filter,
  onSelect,
}) => {
  const query = filter.toLowerCase().replace('/', '');
  const filtered = COMMANDS.filter(c =>
    c.command.toLowerCase().includes(query) || c.description.toLowerCase().includes(query)
  );

  if (filtered.length === 0) return null;

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'model': return <Cpu className="w-4 h-4 text-brand-500 dark:text-brand-400" />;
      case 'web': return <Globe className="w-4 h-4 text-accent-cyan" />;
      case 'chat': return <Trash2 className="w-4 h-4 text-amber-500" />;
      case 'system': return <Terminal className="w-4 h-4 text-accent-violet" />;
      default: return <HelpCircle className="w-4 h-4 text-slate-400" />;
    }
  };

  return (
    <div className="absolute bottom-full mb-2 left-0 right-0 max-h-64 overflow-y-auto rounded-xl border border-slate-200 dark:border-dark-700 bg-white dark:bg-dark-900/95 backdrop-blur-md shadow-2xl p-1.5 z-50">
      <div className="px-3 py-1.5 text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider border-b border-slate-100 dark:border-dark-800">
        Slash Commands ({filtered.length})
      </div>
      <div className="mt-1 space-y-0.5">
        {filtered.map((cmd) => (
          <button
            key={cmd.command}
            type="button"
            onClick={() => onSelect(cmd.command)}
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-left hover:bg-slate-100 dark:hover:bg-dark-800 transition-colors group"
          >
            <div className="flex items-center gap-2.5">
              <div className="p-1 rounded bg-slate-100 dark:bg-dark-850 border border-slate-200 dark:border-dark-750">
                {getCategoryIcon(cmd.category)}
              </div>
              <span className="font-mono text-xs sm:text-sm text-brand-600 dark:text-brand-300 font-medium group-hover:text-brand-700 dark:group-hover:text-brand-200">
                {cmd.command}
              </span>
            </div>
            <span className="text-xs text-slate-500 dark:text-slate-400 truncate max-w-[280px]">
              {cmd.description}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
};
