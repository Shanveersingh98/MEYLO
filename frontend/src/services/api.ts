import {
  Conversation,
  Message,
  ModelInfo,
  SystemStatus,
  SourceItem
} from '../types';

const API_BASE = '/api';

export const api = {
  // Health & System
  async getHealth(): Promise<{
    status: string;
    ready: boolean;
    active_model: string;
    web_search_enabled: boolean;
    device: string;
    version: string;
  }> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  },

  async getStatus(): Promise<SystemStatus> {
    const res = await fetch(`${API_BASE}/status`);
    if (!res.ok) throw new Error(`Failed to fetch status: ${res.statusText}`);
    return res.json();
  },

  // Models
  async getModels(): Promise<{ models: ModelInfo[]; active_model: 'custom' | 'external' }> {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
    return res.json();
  },

  async switchModel(model: 'custom' | 'external'): Promise<{
    success: boolean;
    active_model: 'custom' | 'external';
    message: string;
  }> {
    const res = await fetch(`${API_BASE}/models/switch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Model switch failed');
    }
    return res.json();
  },

  async getCapabilities(): Promise<any> {
    const res = await fetch(`${API_BASE}/capabilities`);
    if (!res.ok) throw new Error('Failed to fetch verified capabilities');
    return res.json();
  },

  // Web Search
  async getWebSearch(): Promise<{ enabled: boolean }> {
    const res = await fetch(`${API_BASE}/web-search`);
    if (!res.ok) throw new Error('Failed to fetch web search status');
    return res.json();
  },

  async setWebSearch(enabled: boolean): Promise<{ enabled: boolean; message: string }> {
    const res = await fetch(`${API_BASE}/web-search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    if (!res.ok) throw new Error('Failed to update web search status');
    return res.json();
  },

  async getSources(): Promise<{ sources: SourceItem[]; count: number; text: string }> {
    const res = await fetch(`${API_BASE}/sources`);
    if (!res.ok) throw new Error('Failed to fetch sources');
    return res.json();
  },

  // Conversations
  async listConversations(search?: string): Promise<Conversation[]> {
    const url = search ? `${API_BASE}/conversations?search=${encodeURIComponent(search)}` : `${API_BASE}/conversations`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to load conversations');
    const data = await res.json();
    return data.conversations || [];
  },

  async createConversation(title?: string, model?: string): Promise<{
    conversation_id: string;
    title: string;
    model: string;
  }> {
    const res = await fetch(`${API_BASE}/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, model }),
    });
    if (!res.ok) throw new Error('Failed to create conversation');
    return res.json();
  },

  async getConversation(convId: string): Promise<{
    conversation: Conversation;
    messages: Message[];
  }> {
    const res = await fetch(`${API_BASE}/conversations/${convId}`);
    if (!res.ok) throw new Error('Failed to fetch conversation details');
    return res.json();
  },

  async renameConversation(convId: string, title: string): Promise<{ success: boolean; title: string }> {
    const res = await fetch(`${API_BASE}/conversations/${convId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    if (!res.ok) throw new Error('Failed to rename conversation');
    return res.json();
  },

  async deleteConversation(convId: string): Promise<{ success: boolean }> {
    const res = await fetch(`${API_BASE}/conversations/${convId}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete conversation');
    return res.json();
  },

  async clearConversation(convId: string): Promise<{ success: boolean }> {
    const res = await fetch(`${API_BASE}/conversations/${convId}/clear`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to clear conversation');
    return res.json();
  },

  async deleteAllConversations(): Promise<{ success: boolean }> {
    const res = await fetch(`${API_BASE}/conversations`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete all conversations');
    return res.json();
  },

  // Chat
  async sendMessage(
    conversationId: string,
    message: string,
    model?: string
  ): Promise<Message & { title_updated?: string }> {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conversation_id: conversationId,
        message,
        model,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Message sending failed');
    }
    return res.json();
  },

  // Streaming Chat with SSE
  streamMessage(
    conversationId: string,
    message: string,
    model: string | undefined,
    onToken: (token: string) => void,
    onDone: (finalData: Message & { title_updated?: string }) => void,
    onError: (err: any) => void
  ): () => void {
    const controller = new AbortController();

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            conversation_id: conversationId,
            message,
            model,
          }),
          signal: controller.signal,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail || 'Streaming failed');
        }

        const reader = res.body?.getReader();
        if (!reader) throw new Error('Stream reader unavailable');

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data:')) continue;
            const jsonStr = trimmed.slice(5).trim();
            if (!jsonStr) continue;

            try {
              const event = JSON.parse(jsonStr);
              if (event.type === 'token') {
                onToken(event.text);
              } else if (event.type === 'done') {
                onDone(event.data);
              }
            } catch (parseErr) {
              console.error('Error parsing SSE event:', parseErr);
            }
          }
        }
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          onError(err);
        }
      }
    })();

    // Return cancel function
    return () => controller.abort();
  },

  // Shutdown
  async shutdown(): Promise<void> {
    await fetch(`${API_BASE}/shutdown`, { method: 'POST' }).catch(() => {});
  },
};
