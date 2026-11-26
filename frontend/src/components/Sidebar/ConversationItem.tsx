import React, { useState } from 'react';
import { MessageSquare, Edit2, Trash2, Check, X } from 'lucide-react';
import { Conversation } from '../../types';

interface ConversationItemProps {
  conversation: Conversation;
  isActive: boolean;
  onSelect: (convId: string) => void;
  onRename: (convId: string, newTitle: string) => Promise<void>;
  onDelete: (convId: string) => Promise<void>;
}

export const ConversationItem: React.FC<ConversationItemProps> = ({
  conversation,
  isActive,
  onSelect,
  onRename,
  onDelete,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [titleInput, setTitleInput] = useState(conversation.title);
  const [isConfirmingDelete, setIsConfirmingDelete] = useState(false);

  const handleSaveRename = async (e: React.MouseEvent | React.KeyboardEvent) => {
    e.stopPropagation();
    if (titleInput.trim() && titleInput !== conversation.title) {
      await onRename(conversation.conversation_id, titleInput.trim());
    }
    setIsEditing(false);
  };

  const handleCancelRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    setTitleInput(conversation.title);
    setIsEditing(false);
  };

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation();
    await onDelete(conversation.conversation_id);
    setIsConfirmingDelete(false);
  };

  return (
    <div
      onClick={() => !isEditing && onSelect(conversation.conversation_id)}
      className={`group relative flex items-center justify-between px-3 py-2.5 rounded-xl cursor-pointer text-sm transition-all select-none ${
        isActive
          ? 'bg-slate-200/70 dark:bg-dark-800 text-slate-900 dark:text-slate-100 font-medium border border-slate-300 dark:border-dark-700 shadow-sm'
          : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-dark-850/80 border border-transparent'
      }`}
    >
      <div className="flex items-center gap-2.5 min-w-0 flex-1">
        <MessageSquare
          className={`w-4 h-4 flex-shrink-0 ${
            isActive ? 'text-brand-600 dark:text-brand-400' : 'text-slate-400 group-hover:text-slate-600 dark:group-hover:text-slate-300'
          }`}
        />

        {isEditing ? (
          <input
            type="text"
            value={titleInput}
            onChange={(e) => setTitleInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSaveRename(e);
              if (e.key === 'Escape') setIsEditing(false);
            }}
            onClick={(e) => e.stopPropagation()}
            autoFocus
            className="w-full bg-white dark:bg-dark-900 border border-brand-500 rounded px-2 py-0.5 text-xs text-slate-900 dark:text-slate-100 focus:outline-none"
          />
        ) : (
          <span className="truncate text-xs sm:text-[13px] leading-snug">
            {conversation.title}
          </span>
        )}
      </div>

      {/* Action buttons */}
      <div
        className="flex items-center gap-1 flex-shrink-0 ml-2"
        onClick={(e) => e.stopPropagation()}
      >
        {isEditing ? (
          <>
            <button
              onClick={handleSaveRename}
              className="p-1 rounded hover:bg-slate-200 dark:hover:bg-dark-700 text-emerald-600 dark:text-emerald-400"
              title="Save"
            >
              <Check className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleCancelRename}
              className="p-1 rounded hover:bg-slate-200 dark:hover:bg-dark-700 text-slate-400"
              title="Cancel"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </>
        ) : isConfirmingDelete ? (
          <div className="flex items-center gap-1 bg-white dark:bg-dark-900 px-1.5 py-0.5 rounded-lg border border-rose-300 dark:border-rose-500/40 shadow-sm">
            <span className="text-[10px] text-rose-600 dark:text-rose-400 font-medium">Delete?</span>
            <button
              onClick={handleDelete}
              className="p-0.5 hover:bg-rose-50 dark:hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 rounded"
              title="Confirm Delete"
            >
              <Check className="w-3 h-3" />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setIsConfirmingDelete(false);
              }}
              className="p-0.5 hover:bg-slate-100 dark:hover:bg-dark-700 text-slate-400 rounded"
              title="Cancel"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setIsEditing(true);
              }}
              className="p-1 rounded text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200 dark:hover:bg-dark-700 transition-colors"
              title="Rename conversation"
            >
              <Edit2 className="w-3 h-3" />
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setIsConfirmingDelete(true);
              }}
              className="p-1 rounded text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-500/20 transition-colors"
              title="Delete conversation"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
