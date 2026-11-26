import React from 'react';
import { CheckCircle2, AlertCircle, Info, AlertTriangle, X } from 'lucide-react';
import { ToastMessage } from '../../types';

interface ToastProps {
  toasts: ToastMessage[];
  onDismiss: (id: string) => void;
}

export const ToastContainer: React.FC<ToastProps> = ({ toasts, onDismiss }) => {
  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-20 right-6 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none">
      {toasts.map(toast => {
        let icon = <Info className="w-4 h-4 text-brand-400 flex-shrink-0" />;
        let borderClass = 'border-slate-200 dark:border-dark-750';
        let bgClass = 'bg-white dark:bg-dark-900';

        if (toast.type === 'success') {
          icon = <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0" />;
          borderClass = 'border-emerald-500/30 dark:border-emerald-500/30';
        } else if (toast.type === 'error') {
          icon = <AlertCircle className="w-4 h-4 text-rose-500 flex-shrink-0" />;
          borderClass = 'border-rose-500/30 dark:border-rose-500/30';
        } else if (toast.type === 'warning') {
          icon = <AlertTriangle className="w-4 h-4 text-amber-500 flex-shrink-0" />;
          borderClass = 'border-amber-500/30 dark:border-amber-500/30';
        }

        return (
          <div
            key={toast.id}
            className={`pointer-events-auto flex items-center justify-between gap-3 px-4 py-3 rounded-xl border ${borderClass} ${bgClass} text-slate-800 dark:text-slate-100 shadow-xl backdrop-blur-md animate-slide-up text-sm transition-all`}
          >
            <div className="flex items-center gap-2.5 min-w-0">
              {icon}
              <span className="truncate text-xs font-medium">{toast.text}</span>
            </div>
            <button
              type="button"
              onClick={() => onDismiss(toast.id)}
              className="p-1 rounded-md text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
};
