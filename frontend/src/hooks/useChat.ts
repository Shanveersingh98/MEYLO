import { useState, useEffect, useCallback, useRef } from 'react';
import { Conversation, Message, ModelInfo, SystemStatus, SourceItem } from '../types';
import { api } from '../services/api';

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeModel, setActiveModel] = useState<'custom' | 'external'>('custom');
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [webSearchEnabled, setWebSearchEnabled] = useState<boolean>(false);
  const [recentSources, setRecentSources] = useState<SourceItem[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isSwitchingModel, setIsSwitchingModel] = useState<boolean>(false);
  const [isBackendReady, setIsBackendReady] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const abortStreamRef = useRef<(() => void) | null>(null);

  // 1. Initial Load: check health, models, web status, conversations, hardware status
  const initialize = useCallback(async () => {
    try {
      const health = await api.getHealth();
      setIsBackendReady(health.ready);
      setActiveModel(health.active_model === 'external' ? 'external' : 'custom');
      setWebSearchEnabled(health.web_search_enabled);

      const modelsData = await api.getModels();
      setModels(modelsData.models);
      setActiveModel(modelsData.active_model);

      const convs = await api.listConversations();
      setConversations(convs);

      if (convs.length > 0) {
        // Load first conversation
        const first = convs[0];
        setActiveConversationId(first.conversation_id);
        const details = await api.getConversation(first.conversation_id);
        setMessages(details.messages);
      } else {
        // Create initial conversation
        const created = await api.createConversation('New Chat', modelsData.active_model);
        setActiveConversationId(created.conversation_id);
        setConversations([
          {
            id: 1,
            conversation_id: created.conversation_id,
            title: created.title,
            model: created.model,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            message_count: 0,
          },
        ]);
        setMessages([]);
      }

      const statusData = await api.getStatus();
      setSystemStatus(statusData);

      const sourcesData = await api.getSources();
      setRecentSources(sourcesData.sources);
    } catch (err: any) {
      console.error('Initialization error:', err);
      setError(err.message || 'Failed to connect to backend.');
    }
  }, []);

  useEffect(() => {
    initialize();
  }, [initialize]);

  // Resource-efficient hardware telemetry refresh (only when tab is visible, throttled to 30s)
  useEffect(() => {
    let isSubscribed = true;

    const fetchStatusIfVisible = async () => {
      if (document.hidden || !isSubscribed) return;
      try {
        const s = await api.getStatus();
        if (isSubscribed) setSystemStatus(s);
      } catch {}
    };

    const interval = setInterval(fetchStatusIfVisible, 30000);

    const handleVisibilityChange = () => {
      if (!document.hidden) {
        fetchStatusIfVisible();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      isSubscribed = false;
      clearInterval(interval);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // 2. Select Conversation
  const selectConversation = useCallback(async (convId: string) => {
    if (convId === activeConversationId) return;
    try {
      setError(null);
      setActiveConversationId(convId);
      const details = await api.getConversation(convId);
      setMessages(details.messages);
    } catch (err: any) {
      console.error('Failed to load conversation:', err);
      setError(err.message || 'Failed to load conversation messages.');
    }
  }, [activeConversationId]);

  // 3. New Chat
  const newChat = useCallback(async () => {
    try {
      setError(null);
      const created = await api.createConversation('New Chat', activeModel);
      const newConv: Conversation = {
        id: Date.now(),
        conversation_id: created.conversation_id,
        title: created.title,
        model: created.model,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        message_count: 0,
      };
      setConversations(prev => [newConv, ...prev]);
      setActiveConversationId(created.conversation_id);
      setMessages([]);
    } catch (err: any) {
      setError(err.message || 'Failed to create new chat.');
    }
  }, [activeModel]);

  // 4. Rename Conversation
  const renameConversation = useCallback(async (convId: string, newTitle: string) => {
    try {
      await api.renameConversation(convId, newTitle);
      setConversations(prev =>
        prev.map(c => (c.conversation_id === convId ? { ...c, title: newTitle } : c))
      );
    } catch (err: any) {
      setError(err.message || 'Failed to rename conversation.');
    }
  }, []);

  // 5. Delete Conversation
  const deleteConversation = useCallback(async (convId: string) => {
    try {
      await api.deleteConversation(convId);
      const remaining = conversations.filter(c => c.conversation_id !== convId);
      setConversations(remaining);

      if (activeConversationId === convId) {
        if (remaining.length > 0) {
          const next = remaining[0];
          setActiveConversationId(next.conversation_id);
          const details = await api.getConversation(next.conversation_id);
          setMessages(details.messages);
        } else {
          // Create new chat
          const created = await api.createConversation('New Chat', activeModel);
          setActiveConversationId(created.conversation_id);
          setConversations([
            {
              id: Date.now(),
              conversation_id: created.conversation_id,
              title: created.title,
              model: created.model,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
              message_count: 0,
            },
          ]);
          setMessages([]);
        }
      }
    } catch (err: any) {
      setError(err.message || 'Failed to delete conversation.');
    }
  }, [conversations, activeConversationId, activeModel]);

  // 6. Clear Current Conversation Messages
  const clearCurrentConversation = useCallback(async () => {
    if (!activeConversationId) return;
    try {
      await api.clearConversation(activeConversationId);
      setMessages([]);
      setRecentSources([]);
    } catch (err: any) {
      setError(err.message || 'Failed to clear conversation.');
    }
  }, [activeConversationId]);

  // 7. Delete All Conversations
  const deleteAllConversations = useCallback(async () => {
    try {
      await api.deleteAllConversations();
      setConversations([]);
      setMessages([]);
      setRecentSources([]);
      // Create fresh conversation
      const created = await api.createConversation('New Chat', activeModel);
      setActiveConversationId(created.conversation_id);
      setConversations([
        {
          id: Date.now(),
          conversation_id: created.conversation_id,
          title: created.title,
          model: created.model,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          message_count: 0,
        },
      ]);
    } catch (err: any) {
      setError(err.message || 'Failed to delete all conversations.');
    }
  }, [activeModel]);

  // 8. Model Switching
  const selectModel = useCallback(async (targetModel: 'custom' | 'external') => {
    if (targetModel === activeModel) return;
    setIsSwitchingModel(true);
    setError(null);
    try {
      const res = await api.switchModel(targetModel);
      if (res.success) {
        setActiveModel(res.active_model);
        setModels(prev =>
          prev.map(m => ({ ...m, active: m.id === res.active_model }))
        );
        const s = await api.getStatus();
        setSystemStatus(s);
      }
    } catch (err: any) {
      console.error('Model switch error:', err);
      setError(`Model switch failed: ${err.message || 'Could not load weights.'}`);
    } finally {
      setIsSwitchingModel(false);
    }
  }, [activeModel]);

  // 9. Web Search Toggle
  const toggleWebSearch = useCallback(async (enabled: boolean) => {
    try {
      setError(null);
      const res = await api.setWebSearch(enabled);
      setWebSearchEnabled(res.enabled);
    } catch (err: any) {
      setError(`Failed to toggle web search: ${err.message}`);
    }
  }, []);

  // 10. Send Message (with SSE Streaming)
  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isGenerating) return;

    let convId = activeConversationId;
    if (!convId) {
      const created = await api.createConversation('New Chat', activeModel);
      convId = created.conversation_id;
      setActiveConversationId(convId);
      setConversations(prev => [
        {
          id: Date.now(),
          conversation_id: convId!,
          title: created.title,
          model: created.model,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          message_count: 0,
        },
        ...prev,
      ]);
    }

    const userMessage: Message = {
      id: `temp_u_${Date.now()}`,
      conversation_id: convId,
      role: 'user',
      content: content.trim(),
      timestamp: new Date().toISOString(),
      model_used: activeModel,
    };

    // Placeholder assistant message for streaming
    const assistantMessageId = `temp_a_${Date.now()}`;
    const placeholderAssistantMessage: Message = {
      id: assistantMessageId,
      conversation_id: convId,
      role: 'assistant',
      content: '',
      model_used: activeModel,
      timestamp: new Date().toISOString(),
      is_streaming: true,
    };

    setMessages(prev => [...prev, userMessage, placeholderAssistantMessage]);
    setIsGenerating(true);
    setError(null);

    // Call streaming SSE
    const cancel = api.streamMessage(
      convId,
      content.trim(),
      activeModel,
      // onToken
      (token: string) => {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMessageId
              ? { ...m, content: m.content + token }
              : m
          )
        );
      },
      // onDone
      (finalData: Message & { title_updated?: string }) => {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMessageId
              ? {
                  ...finalData,
                  is_streaming: false,
                }
              : m
          )
        );
        setIsGenerating(false);
        abortStreamRef.current = null;

        if (finalData.sources && finalData.sources.length > 0) {
          setRecentSources(finalData.sources);
        }

        // If title updated, update sidebar
        if (finalData.title_updated) {
          setConversations(prev =>
            prev.map(c =>
              c.conversation_id === convId
                ? { ...c, title: finalData.title_updated! }
                : c
            )
          );
        }

        // If this was a slash command that switched model or web search, sync state!
        if (finalData.route?.includes('model_custom')) {
          setActiveModel('custom');
        } else if (finalData.route?.includes('model_external')) {
          setActiveModel('external');
        } else if (finalData.route?.includes('web_on')) {
          setWebSearchEnabled(true);
        } else if (finalData.route?.includes('web_off')) {
          setWebSearchEnabled(false);
        } else if (finalData.route?.includes('clear')) {
          setMessages([]);
        }
      },
      // onError
      (err: any) => {
        console.error('Streaming error:', err);
        setError(err.message || 'An error occurred during response generation.');
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMessageId
              ? {
                  ...m,
                  content:
                    m.content ||
                    `I encountered an error generating the response: ${err.message || 'Connection lost'}. Please try again.`,
                  is_streaming: false,
                }
              : m
          )
        );
        setIsGenerating(false);
        abortStreamRef.current = null;
      }
    );

    abortStreamRef.current = cancel;
  }, [activeConversationId, activeModel, isGenerating]);

  // 11. Stop Generation
  const stopGeneration = useCallback(() => {
    if (abortStreamRef.current) {
      abortStreamRef.current();
      abortStreamRef.current = null;
      setIsGenerating(false);
    }
  }, []);

  // 12. Refresh Status
  const refreshStatus = useCallback(async () => {
    try {
      const s = await api.getStatus();
      setSystemStatus(s);
      const src = await api.getSources();
      setRecentSources(src.sources);
    } catch (err: any) {
      setError(err.message);
    }
  }, []);

  // 13. Regenerate Last Response
  const regenerateResponse = useCallback(async () => {
    if (isGenerating || messages.length === 0) return;
    let lastUserContent = '';
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUserContent = messages[i].content;
        break;
      }
    }
    if (!lastUserContent) return;

    // Prune the last assistant response from messages list
    setMessages(prev => {
      if (prev.length > 0 && prev[prev.length - 1].role === 'assistant') {
        return prev.slice(0, -1);
      }
      return prev;
    });

    await sendMessage(lastUserContent);
  }, [isGenerating, messages, sendMessage]);

  return {
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
    dismissError: () => setError(null),
  };
}
