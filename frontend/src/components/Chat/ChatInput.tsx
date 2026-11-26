import React, { useState, useRef, useEffect } from 'react';
import { Send, Square, Terminal, Globe, Cpu, ShieldCheck } from 'lucide-react';
import { SlashCommandMenu } from './SlashCommandMenu';

interface ChatInputProps {
  onSendMessage: (content: string) => void;
  onStopGeneration?: () => void;
  isGenerating: boolean;
  activeModel: 'custom' | 'external';
  webSearchEnabled: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  onStopGeneration,
  isGenerating,
  activeModel,
  webSearchEnabled,
}) => {
  const [input, setInput] = useState('');
  const [showCommands, setShowCommands] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const newHeight = Math.min(textareaRef.current.scrollHeight, 180);
      textareaRef.current.style.height = `${Math.max(newHeight, 48)}px`;
    }
  }, [input]);

  // Focus textarea on load
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    } else if (e.key === 'Escape') {
      setShowCommands(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setInput(val);
    if (val.startsWith('/')) {
      setShowCommands(true);
    } else {
      setShowCommands(false);
    }
  };

  const handleSelectCommand = (cmd: string) => {
    setInput(cmd);
    setShowCommands(false);
    textareaRef.current?.focus();
  };

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed || isGenerating) return;
    onSendMessage(trimmed);
    setInput('');
    setShowCommands(false);
    if (textareaRef.current) {
      textareaRef.current.style.height = '48px';
    }
  };

  return (
    <div className="relative w-full max-w-4xl mx-auto px-4 pb-4 pt-2">
      {showCommands && (
        <SlashCommandMenu
          filter={input}
          onSelect={handleSelectCommand}
          onClose={() => setShowCommands(false)}
        />
      )}

      <div className={`relative flex flex-col rounded-2xl border bg-white dark:bg-dark-900 shadow-xl transition-all ${
        activeModel === 'custom'
          ? 'border-violet-200 dark:border-violet-900/40 focus-within:border-violet-500 focus-within:ring-2 focus-within:ring-violet-500/20'
          : 'border-teal-200 dark:border-teal-900/40 focus-within:border-teal-500 focus-within:ring-2 focus-within:ring-teal-500/20'
      }`}>
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          disabled={isGenerating}
          placeholder={
            isGenerating
              ? 'Generating response...'
              : 'Message MEYLO... (type / for commands, Shift+Enter for new line)'
          }
          rows={1}
          className="w-full resize-none bg-transparent px-4 pt-3.5 pb-2 text-[14.5px] leading-relaxed text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none disabled:opacity-50 min-h-[48px] max-h-[180px]"
        />

        {/* Input Footer Toolbar */}
        <div className="flex items-center justify-between px-3 pb-2.5 pt-1 border-t border-slate-100 dark:border-dark-800/60 text-xs text-slate-500 dark:text-slate-400">
          <div className="flex items-center gap-2">
            <span className={`flex items-center gap-1 font-mono text-[11px] font-medium ${
              activeModel === 'custom'
                ? 'text-violet-700 dark:text-violet-300'
                : 'text-teal-700 dark:text-teal-300'
            }`}>
              <Cpu className="w-3 h-3" />
              <span>{activeModel === 'custom' ? 'FromScratchLM (Custom PyTorch)' : 'Qwen2.5-1.5B (External GGUF)'}</span>
            </span>

            <span className="text-slate-300 dark:text-slate-700">•</span>

            <span className="flex items-center gap-1 font-mono text-[11px]">
              {webSearchEnabled ? (
                <>
                  <Globe className="w-3 h-3 text-accent-cyan" />
                  <span className="text-accent-cyan font-medium">Online Search</span>
                </>
              ) : (
                <>
                  <ShieldCheck className="w-3 h-3 text-emerald-500" />
                  <span>Offline Mode</span>
                </>
              )}
            </span>
          </div>

          <div className="flex items-center gap-2">
            {isGenerating ? (
              <button
                type="button"
                onClick={onStopGeneration}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-rose-50 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400 border border-rose-200 dark:border-rose-500/30 hover:bg-rose-100 dark:hover:bg-rose-500/30 text-xs font-semibold transition-colors shadow-sm"
                title="Stop response generation"
              >
                <Square className="w-3 h-3 fill-current" />
                <span>Stop</span>
              </button>
            ) : (
              <button
                type="button"
                disabled={!input.trim()}
                onClick={handleSubmit}
                className={`flex items-center justify-center p-2 rounded-xl text-white disabled:opacity-40 transition-all shadow-md ${
                  activeModel === 'custom'
                    ? 'bg-violet-600 hover:bg-violet-500 shadow-violet-500/20'
                    : 'bg-teal-600 hover:bg-teal-500 shadow-teal-500/20'
                }`}
                title="Send message (Enter)"
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
