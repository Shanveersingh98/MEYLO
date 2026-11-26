import React, { useState, useCallback } from 'react';
import { useTheme } from './hooks/useTheme';
import { useChat } from './hooks/useChat';
import { Sidebar } from './components/Sidebar/Sidebar';
import { Header } from './components/Header/Header';
import { ChatArea } from './components/Chat/ChatArea';
import { ChatInput } from './components/Chat/ChatInput';
import { SettingsModal } from './components/Settings/SettingsModal';
import { CapabilityModal } from './components/Capabilities/CapabilityModal';
import { ToastContainer } from './components/Toast/Toast';
import { ToastMessage } from './types';

export const App: React.FC = () => {
  const { themeMode, resolvedTheme, setThemeMode } = useTheme();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [capabilitiesOpen, setCapabilitiesOpen] = useState(false);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const addToast = useCallback((type: 'info' | 'success' | 'warning' | 'error', text: string) => {
    const id = `toast_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
    const newToast: ToastMessage = { id, type, text };
    setToasts(prev => [...prev, newToast]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 3500);
  }, []);

  const dismissToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const {
    conversations,
    activeConversationId,
    messages,
    activeModel,
    models,
    webSearchEnabled,
    recentSources,
    systemStatus,
    isGenerating,
    isSwitchingModel,
    isBackendReady,
    error,
    selectConversation,
    newChat,
    renameConversation,
    deleteConversation,
    clearCurrentConversation,
    deleteAllConversations,
    selectModel,
    toggleWebSearch,
    sendMessage,
    regenerateResponse,
    stopGeneration,
    refreshStatus,
    dismissError,
  } = useChat();

  const handleSelectModel = async (model: 'custom' | 'external') => {
    await selectModel(model);
    addToast(
      'success',
      model === 'custom'
        ? 'Active Model: FromScratchLM (Custom PyTorch, CUDA)'
        : 'Active Model: Qwen2.5-1.5B-Instruct (GGUF CPU)'
    );
  };

  const handleToggleWebSearch = async (enabled: boolean) => {
    await toggleWebSearch(enabled);
    addToast(
      'info',
      enabled
        ? 'Online Mode: Web retrieval enabled with sources'
        : 'Offline Mode: Local model weights only'
    );
  };

  const handleSetTheme = (mode: 'light' | 'dark' | 'system') => {
    setThemeMode(mode);
    addToast('info', `Theme preference updated to ${mode.toUpperCase()}`);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50 dark:bg-dark-950 text-slate-800 dark:text-slate-100 font-sans transition-colors">
      {/* Left Collapsible / Drawer Sidebar */}
      <Sidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={selectConversation}
        onNewChat={newChat}
        onRenameConversation={renameConversation}
        onDeleteConversation={deleteConversation}
        onClearCurrentConversation={clearCurrentConversation}
        onOpenSettings={() => setSettingsOpen(true)}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        activeModel={activeModel}
        models={models}
        systemStatus={systemStatus}
        isBackendReady={isBackendReady}
        themeMode={themeMode}
        onSetThemeMode={handleSetTheme}
      />

      {/* Main Container */}
      <main className="flex-1 flex flex-col min-w-0 h-full relative bg-slate-50 dark:bg-dark-950 transition-colors">
        {/* Top Header */}
        <Header
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          onOpenSettings={() => setSettingsOpen(true)}
          onOpenCapabilities={() => setCapabilitiesOpen(true)}
          activeModel={activeModel}
          models={models}
          onSelectModel={handleSelectModel}
          isSwitchingModel={isSwitchingModel}
          webSearchEnabled={webSearchEnabled}
          onToggleWebSearch={handleToggleWebSearch}
          themeMode={themeMode}
          onSetThemeMode={handleSetTheme}
          status={systemStatus}
        />

        {/* Chat Messages Area */}
        <ChatArea
          messages={messages}
          isGenerating={isGenerating}
          activeModel={activeModel}
          webSearchEnabled={webSearchEnabled}
          onRegenerate={regenerateResponse}
          error={error}
          onDismissError={dismissError}
          onOpenCapabilities={() => setCapabilitiesOpen(true)}
        />

        {/* Message Input Composer */}
        <ChatInput
          onSendMessage={sendMessage}
          onStopGeneration={stopGeneration}
          isGenerating={isGenerating}
          activeModel={activeModel}
          webSearchEnabled={webSearchEnabled}
        />
      </main>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        activeModel={activeModel}
        models={models}
        onSelectModel={handleSelectModel}
        isSwitchingModel={isSwitchingModel}
        webSearchEnabled={webSearchEnabled}
        onToggleWebSearch={handleToggleWebSearch}
        themeMode={themeMode}
        onSetThemeMode={handleSetTheme}
        status={systemStatus}
        onRefreshStatus={refreshStatus}
        onClearCurrentChat={clearCurrentConversation}
        onDeleteAllConversations={deleteAllConversations}
        recentSources={recentSources}
        onOpenCapabilities={() => setCapabilitiesOpen(true)}
      />

      {/* Verified Model Capability Matrix Modal */}
      <CapabilityModal
        isOpen={capabilitiesOpen}
        onClose={() => setCapabilitiesOpen(false)}
        activeModel={activeModel}
      />

      {/* Toast Notifications */}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
};

export default App;

