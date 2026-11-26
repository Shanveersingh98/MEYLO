import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  User,
  Bot,
  Copy,
  Check,
  Cpu,
  Globe,
  ExternalLink,
  Clock,
  Terminal,
  RotateCcw,
  Sparkles
} from 'lucide-react';
import { Message, SourceItem } from '../../types';
import { CodeBlock } from './CodeBlock';

interface MessageItemProps {
  message: Message;
  isLastAssistant?: boolean;
  onRegenerate?: () => void;
  isGenerating?: boolean;
}

export const MessageItem: React.FC<MessageItemProps> = ({
  message,
  isLastAssistant = false,
  onRegenerate,
  isGenerating = false,
}) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';
  const isCommand = message.is_command || message.route?.startsWith('slash_command_');

  const handleCopyResponse = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy message:', err);
    }
  };

  const formatTimestamp = (ts?: string) => {
    if (!ts) return '';
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  };

  return (
    <div
      className={`group relative flex gap-3.5 sm:gap-4 p-4 md:px-6 transition-colors ${
        isUser
          ? 'bg-transparent'
          : isCommand
          ? 'bg-purple-50/50 dark:bg-dark-900/40 border-y border-purple-100 dark:border-dark-800/40'
          : 'bg-white dark:bg-dark-900/60 border-y border-slate-200/70 dark:border-dark-800/50 shadow-sm dark:shadow-none'
      }`}
    >
      {/* Avatar */}
      <div className="flex-shrink-0">
        {isUser ? (
          <div className="w-8 h-8 rounded-xl bg-slate-100 dark:bg-dark-800 border border-slate-200 dark:border-dark-700 flex items-center justify-center text-slate-700 dark:text-slate-300 shadow-sm">
            <User className="w-4 h-4" />
          </div>
        ) : isCommand ? (
          <div className="w-8 h-8 rounded-xl bg-amber-50 dark:bg-amber-500/15 border border-amber-200 dark:border-amber-500/30 flex items-center justify-center text-amber-700 dark:text-amber-400 shadow-sm">
            <Terminal className="w-4 h-4" />
          </div>
        ) : message.model_used === 'custom' ? (
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-violet-600 to-purple-600 flex items-center justify-center text-white shadow-md shadow-violet-500/20">
            <Bot className="w-4 h-4" />
          </div>
        ) : (
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-teal-600 to-cyan-600 flex items-center justify-center text-white shadow-md shadow-teal-500/20">
            <Bot className="w-4 h-4" />
          </div>
        )}
      </div>

      {/* Message Content Container */}
      <div className="flex-1 min-w-0 max-w-3xl">
        {/* Header Metadata */}
        <div className="flex items-center justify-between mb-1 text-xs">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-900 dark:text-slate-200">
              {isUser ? 'You' : isCommand ? 'System Command' : 'MEYLO'}
            </span>

            {/* Model Badge for Assistant */}
            {!isUser && !isCommand && (
              <span
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full font-mono text-[10px] font-medium ${
                  message.model_used === 'custom'
                    ? 'bg-violet-50 dark:bg-violet-500/15 text-violet-700 dark:text-violet-300 border border-violet-200 dark:border-violet-500/30'
                    : 'bg-teal-50 dark:bg-teal-500/15 text-teal-700 dark:text-teal-300 border border-teal-200 dark:border-teal-500/30'
                }`}
              >
                <Cpu className="w-2.5 h-2.5" />
                {message.model_used === 'custom' ? 'FromScratchLM (Custom PyTorch)' : 'Qwen2.5-1.5B (External GGUF)'}
              </span>
            )}

            {/* Route Tag if available */}
            {message.route && !isCommand && (
              <span className="hidden sm:inline-block font-mono text-[10px] text-slate-400 dark:text-slate-500">
                ({message.route})
              </span>
            )}
          </div>

          <div className="flex items-center gap-3 text-slate-400 dark:text-slate-500 text-[11px]">
            {message.latency_ms !== undefined && message.latency_ms > 0 && (
              <span className="font-mono flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {message.latency_ms > 1000
                  ? `${(message.latency_ms / 1000).toFixed(2)}s`
                  : `${Math.round(message.latency_ms)}ms`}
              </span>
            )}
            {message.timestamp && <span>{formatTimestamp(message.timestamp)}</span>}
          </div>
        </div>

        {/* Message Body */}
        <div className="prose-companion text-slate-800 dark:text-slate-100 text-[14.5px] leading-relaxed break-words">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              code({ node, inline, className, children, ...props }: any) {
                const match = /language-(\w+)/.exec(className || '');
                const codeString = String(children).replace(/\n$/, '');
                if (!inline && (match || codeString.includes('\n'))) {
                  return (
                    <CodeBlock
                      language={match ? match[1] : ''}
                      code={codeString}
                    />
                  );
                }
                return (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Sources Section if Web Retrieval was Used */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-200 dark:border-dark-800">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-accent-cyan mb-2">
              <Globe className="w-3.5 h-3.5" />
              <span>Verified Sources ({message.sources.length}):</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {message.sources.map((src, i) => {
                let domain = '';
                try {
                  domain = new URL(src.url).hostname.replace('www.', '');
                } catch {
                  domain = 'web';
                }

                return (
                  <a
                    key={i}
                    href={src.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-100 dark:bg-dark-850 hover:bg-cyan-50 dark:hover:bg-dark-800 border border-slate-200 dark:border-dark-750 text-xs text-slate-700 dark:text-slate-300 hover:text-cyan-700 dark:hover:text-cyan-300 transition-colors shadow-sm max-w-xs truncate"
                    title={`${src.title}\n${src.url}`}
                  >
                    <span className="font-semibold text-slate-500 dark:text-slate-400 font-mono text-[10px]">
                      [{domain}]
                    </span>
                    <span className="truncate">{src.title || src.url}</span>
                    <ExternalLink className="w-3 h-3 flex-shrink-0 text-slate-400" />
                  </a>
                );
              })}
            </div>
          </div>
        )}

        {/* Actions Footer */}
        {!isUser && (
          <div className="mt-2.5 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={handleCopyResponse}
              type="button"
              className="flex items-center gap-1 px-2 py-1 rounded-md text-xs text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-dark-800 transition-colors"
              title="Copy response text"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-500" />
                  <span className="text-emerald-500 font-medium">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>

            {/* Regenerate Button on Assistant Turn */}
            {isLastAssistant && onRegenerate && (
              <button
                onClick={onRegenerate}
                disabled={isGenerating}
                type="button"
                className="flex items-center gap-1 px-2 py-1 rounded-md text-xs text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-dark-800 transition-colors disabled:opacity-50"
                title="Regenerate response with current active model"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Regenerate</span>
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
