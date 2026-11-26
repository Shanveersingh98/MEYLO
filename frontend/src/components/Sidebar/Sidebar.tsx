import React, { useState } from 'react';
import {
  Brain,
  Plus,
  Search,
  Settings,
  Trash2,
  Cpu,
  Globe,
  X,
  CheckCircle2,
  AlertCircle,
  Sun,
  Moon,
  Laptop
} from 'lucide-react';
import { Conversation, ModelInfo, SystemStatus, ThemeMode } from '../../types';
import { ConversationItem } from './ConversationItem';
import { MeyloLogo } from '../common/MeyloLogo';

interface SidebarProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelectConversation: (convId: string) => void;
  onNewChat: () => void;
  onRenameConversation: (convId: string, newTitle: string) => Promise<void>;
  onDeleteConversation: (convId: string) => Promise<void>;
  onClearCurrentConversation: () => Promise<void>;
  onOpenSettings: () => void;
  isOpen: boolean;
  onClose: () => void;
  activeModel: 'custom' | 'external';
  models: ModelInfo[];
  systemStatus: SystemStatus | null;
  isBackendReady: boolean;
  themeMode: ThemeMode;
  onSetThemeMode: (mode: ThemeMode) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onRenameConversation,
  onDeleteConversation,
  onClearCurrentConversation,
  onOpenSettings,
  isOpen,
  onClose,
  activeModel,
  models,
  systemStatus,
  isBackendReady,
  themeMode,
  onSetThemeMode,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isConfirmingClear, setIsConfirmingClear] = useState(false);

  const filteredConversations = conversations.filter(c =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleClear = async () => {
    await onClearCurrentConversation();
    setIsConfirmingClear(false);
  };

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed top-0 bottom-0 left-0 w-72 bg-white dark:bg-dark-900 border-r border-slate-200 dark:border-dark-800 flex flex-col z-50 transition-transform duration-300 ease-in-out lg:static lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Top App Header */}
        <div className="h-16 px-4 border-b border-slate-200 dark:border-dark-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl overflow-hidden shadow-md shadow-indigo-500/20 flex-shrink-0">
              <MeyloLogo className="w-8 h-8" size={32} />
            </div>
            <div>
              <div className="font-bold text-sm text-slate-900 dark:text-slate-100 tracking-wider flex items-center gap-1.5">
                <span className="tracking-widest font-extrabold text-slate-900 dark:text-white">MEYLO</span>
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-brand-100 dark:bg-brand-500/20 text-brand-700 dark:text-brand-300 border border-brand-200 dark:border-brand-500/30">
                  v2.0
                </span>
              </div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400 font-medium">
                Your AI. Your Space.
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-dark-800 lg:hidden"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-3">
          <button
            type="button"
            onClick={() => {
              onNewChat();
              if (window.innerWidth < 1024) onClose();
            }}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-medium text-sm transition-all shadow-md shadow-brand-500/20 active:scale-[0.99]"
          >
            <Plus className="w-4 h-4" />
            <span>New Conversation</span>
          </button>
        </div>

        {/* Search Conversations */}
        <div className="px-3 pb-2">
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search conversations..."
              className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-slate-100 dark:bg-dark-850 border border-slate-200 dark:border-dark-750 text-xs text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:border-brand-500 transition-colors"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 text-xs"
              >
                ✕
              </button>
            )}
          </div>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto px-3 space-y-1 py-1">
          <div className="text-[10.5px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider px-2 py-1">
            History ({filteredConversations.length})
          </div>

          {filteredConversations.length === 0 ? (
            <div className="text-center py-8 text-xs text-slate-400">
              {searchQuery ? 'No matching chats' : 'No previous conversations'}
            </div>
          ) : (
            filteredConversations.map((conv) => (
              <ConversationItem
                key={conv.conversation_id}
                conversation={conv}
                isActive={conv.conversation_id === activeConversationId}
                onSelect={(id) => {
                  onSelectConversation(id);
                  if (window.innerWidth < 1024) onClose();
                }}
                onRename={onRenameConversation}
                onDelete={onDeleteConversation}
              />
            ))
          )}
        </div>

        {/* Bottom Panel: Theme Selector, Clear Chat, Settings, Status */}
        <div className="p-3 border-t border-slate-200 dark:border-dark-800 space-y-2 bg-slate-50/50 dark:bg-dark-950/40">
          {/* Quick Theme Switcher */}
          <div className="flex items-center justify-between px-2 py-1 bg-white dark:bg-dark-850 rounded-xl border border-slate-200 dark:border-dark-750 text-xs">
            <span className="text-slate-500 dark:text-slate-400 font-medium text-[11px]">Theme:</span>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => onSetThemeMode('light')}
                className={`p-1 rounded-md transition-colors ${
                  themeMode === 'light'
                    ? 'bg-brand-100 text-brand-700 shadow-sm'
                    : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                }`}
                title="Light Mode"
              >
                <Sun className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => onSetThemeMode('dark')}
                className={`p-1 rounded-md transition-colors ${
                  themeMode === 'dark'
                    ? 'bg-dark-700 text-brand-300 shadow-sm'
                    : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                }`}
                title="Dark Mode"
              >
                <Moon className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => onSetThemeMode('system')}
                className={`p-1 rounded-md transition-colors ${
                  themeMode === 'system'
                    ? 'bg-slate-200 dark:bg-dark-700 text-brand-600 dark:text-brand-300 shadow-sm'
                    : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                }`}
                title="System Default"
              >
                <Laptop className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Clear Current Chat Button */}
          {activeConversationId && (
            <div>
              {isConfirmingClear ? (
                <div className="flex items-center justify-between p-2 rounded-xl bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 text-xs">
                  <span className="text-rose-600 dark:text-rose-400 font-medium">Clear messages?</span>
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={handleClear}
                      className="px-2 py-0.5 rounded bg-rose-600 text-white font-semibold hover:bg-rose-500"
                    >
                      Clear
                    </button>
                    <button
                      onClick={() => setIsConfirmingClear(false)}
                      className="px-2 py-0.5 rounded bg-slate-200 dark:bg-dark-750 text-slate-700 dark:text-slate-300 hover:bg-slate-300"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => setIsConfirmingClear(true)}
                  className="w-full flex items-center justify-between px-3 py-1.5 rounded-xl text-slate-500 dark:text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-500/10 text-xs transition-colors"
                >
                  <span className="flex items-center gap-2">
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Clear Current Chat</span>
                  </span>
                </button>
              )}
            </div>
          )}

          {/* Settings Launcher */}
          <button
            type="button"
            onClick={onOpenSettings}
            className="w-full flex items-center justify-between px-3 py-2 rounded-xl text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-dark-850 border border-slate-200 dark:border-dark-750 text-xs font-medium transition-colors"
          >
            <div className="flex items-center gap-2">
              <Settings className="w-4 h-4 text-slate-500" />
              <span>Settings & System</span>
            </div>
            <div className="flex items-center gap-1 text-[10.5px] font-mono text-emerald-600 dark:text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span>Ready</span>
            </div>
          </button>
        </div>
      </aside>
    </>
  );
};
