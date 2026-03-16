export type ChatRole = "user" | "assistant";

export interface SourceChunk {
  id?: string;
  text?: string;
  score?: number;
  metadata?: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  sources?: SourceChunk[];
  images?: string[];
}

export interface QueryResponse {
  answer: string;
  sources?: SourceChunk[];
  images?: string[];
}
