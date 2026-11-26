import React, { useRef, useEffect } from 'react';
import { ArrowDown, Bot, Loader2, AlertCircle } from 'lucide-react';
import { Message, SourceItem } from '../../types';
import { MessageItem } from './MessageItem';
import { WelcomeScreen } from './WelcomeScreen';

interface ChatAreaProps {
  messages: Message[];
  isGenerating: boolean;
  activeModel: 'custom' | 'external';
  webSearchEnabled: boolean;
  onRegenerate?: () => void;
  error?: string | null;
  onDismissError?: () => void;
  onOpenCapabilities?: () => void;
}

export const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  isGenerating,
  activeModel,
  webSearchEnabled,
  onRegenerate,
  error,
  onDismissError,
  onOpenCapabilities,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  // Find index of last assistant message for regenerate action
  let lastAssistantIndex = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'assistant') {
      lastAssistantIndex = i;
      break;
    }
  }

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto flex flex-col relative w-full bg-slate-50 dark:bg-dark-950 transition-colors"
    >
      {/* Error Alert Banner */}
      {error && (
        <div className="sticky top-2 z-20 mx-4 max-w-3xl sm:mx-auto mt-2 p-3 rounded-xl bg-rose-50 dark:bg-rose-500/15 border border-rose-200 dark:border-rose-500/30 text-rose-700 dark:text-rose-400 text-xs sm:text-sm flex items-center justify-between shadow-md animate-slide-up backdrop-blur-md">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0 text-rose-500" />
            <span className="font-medium">{error}</span>
          </div>
          {onDismissError && (
            <button
              onClick={onDismissError}
              className="text-xs font-semibold underline hover:opacity-80 px-2 py-0.5 ml-2"
            >
              Dismiss
            </button>
          )}
        </div>
      )}

      {/* Welcome Screen or Message List */}
      {messages.length === 0 ? (
        <WelcomeScreen
          activeModel={activeModel}
          webSearchEnabled={webSearchEnabled}
          onOpenCapabilities={onOpenCapabilities}
        />
      ) : (
        <div className="flex-1 flex flex-col py-4 w-full">
          {messages.map((msg, index) => (
            <MessageItem
              key={msg.id || index}
              message={msg}
              isLastAssistant={index === lastAssistantIndex}
              onRegenerate={onRegenerate}
              isGenerating={isGenerating}
            />
          ))}

          {/* Typing/Generating Indicator */}
          {isGenerating && (
            <div className="flex gap-4 p-4 md:px-6 bg-white dark:bg-dark-900/60 border-y border-slate-200/60 dark:border-dark-800/50 animate-fade-in">
              <div className="flex-shrink-0">
                {activeModel === 'custom' ? (
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-violet-600 to-purple-600 flex items-center justify-center text-white shadow-md shadow-violet-500/20">
                    <Bot className="w-4 h-4 animate-spin" />
                  </div>
                ) : (
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-teal-600 to-cyan-600 flex items-center justify-center text-white shadow-md shadow-teal-500/20">
                    <Bot className="w-4 h-4 animate-spin" />
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2.5 text-sm text-slate-500 dark:text-slate-400">
                <div className="flex gap-1 items-center">
                  <span
                    className={`w-2 h-2 rounded-full ${activeModel === 'custom' ? 'bg-violet-500' : 'bg-teal-500'} animate-bounce`}
                    style={{ animationDelay: '0ms' }}
                  />
                  <span
                    className={`w-2 h-2 rounded-full ${activeModel === 'custom' ? 'bg-violet-500' : 'bg-teal-500'} animate-bounce`}
                    style={{ animationDelay: '150ms' }}
                  />
                  <span
                    className={`w-2 h-2 rounded-full ${activeModel === 'custom' ? 'bg-violet-500' : 'bg-teal-500'} animate-bounce`}
                    style={{ animationDelay: '300ms' }}
                  />
                </div>
                <span className={`font-mono text-xs ${activeModel === 'custom' ? 'text-violet-700 dark:text-violet-300' : 'text-teal-700 dark:text-teal-300'}`}>
                  {activeModel === 'custom'
                    ? 'FromScratchLM is generating response (PyTorch CUDA)...'
                    : 'Qwen2.5-1.5B is generating response (GGUF CPU)...'}
                </span>
              </div>
            </div>
          )}

          <div ref={bottomRef} className="h-4" />
        </div>
      )}
    </div>
  );
};
