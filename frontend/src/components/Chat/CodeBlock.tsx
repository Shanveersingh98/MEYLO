import React, { useState } from 'react';
import { Check, Copy } from 'lucide-react';

interface CodeBlockProps {
  language?: string;
  code: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({ language = 'text', code }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  return (
    <div className="relative my-3 rounded-xl border border-slate-700/60 dark:border-dark-700 bg-slate-900 dark:bg-dark-900/90 overflow-hidden shadow-md group">
      <div className="flex items-center justify-between px-4 py-2 bg-slate-950/70 dark:bg-dark-850/80 border-b border-slate-800 dark:border-dark-750 text-xs font-mono text-slate-400">
        <span className="uppercase tracking-wider text-[11px] text-brand-400 font-semibold">
          {language || 'code'}
        </span>
        <button
          onClick={handleCopy}
          type="button"
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-slate-400 hover:text-slate-100 hover:bg-slate-800 dark:hover:bg-dark-700/60 transition-colors focus:outline-none"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-400 font-medium">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <div className="p-4 overflow-x-auto text-[13.5px] leading-relaxed font-mono text-slate-100">
        <pre className="m-0 p-0">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  );
};
