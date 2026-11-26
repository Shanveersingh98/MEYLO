export interface SourceItem {
  title: string;
  url: string;
  snippet?: string;
  publisher?: string;
  date?: string;
}

export interface Message {
  id: number | string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  model_used?: string;
  route?: string;
  sources?: SourceItem[];
  latency_ms?: number;
  timestamp?: string;
  is_command?: boolean;
  is_streaming?: boolean;
}

export interface Conversation {
  id: number;
  conversation_id: string;
  title: string;
  model: string;
  created_at: string;
  updated_at: string;
  message_count?: number;
}

export interface ModelInfo {
  id: 'custom' | 'external';
  name: string;
  description: string;
  checkpoint: string;
  runtime: string;
  device: string;
  active: boolean;
  badge: string;
}

export interface HardwareInfo {
  cpu_name: string;
  ram_total_gb: number;
  ram_used_gb: number;
  ram_avail_gb: number;
  ram_percent: number;
  gpu_name: string;
  vram_total_gb: number;
  vram_used_mb: number;
  cuda_available: boolean;
  cuda_version: string;
}

export interface SystemStatus {
  hardware: HardwareInfo;
  active_model: {
    id: string;
    name: string;
    type: string;
    file: string;
    execution_device: string;
    inference_engine: string;
  };
  web_search_enabled: boolean;
  debug_mode: boolean;
  status_text: string;
}

export interface SlashCommand {
  command: string;
  description: string;
  usage: string;
  category: 'model' | 'web' | 'system' | 'chat';
}

export type ThemeMode = 'light' | 'dark' | 'system';
export type ResolvedTheme = 'light' | 'dark';

export interface ToastMessage {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  text: string;
  duration?: number;
}
