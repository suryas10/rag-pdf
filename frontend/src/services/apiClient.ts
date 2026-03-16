import type { QueryResponse } from "../types/chat";
import type { UploadResponse } from "../types/document";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

interface QueryPayload {
  query: string;
  chat_id?: string;
}

interface ChatHistoryResponse {
  chat_id: string;
  messages: Array<{
    role: "user" | "assistant";
    content: string;
    timestamp: string;
    sources?: Array<Record<string, unknown>>;
  }>;
}

interface CreateChatResponse {
  chat_id: string;
  created_at?: string;
}

interface SessionInfo {
  chat_id: string;
  document_id?: string;
  document_name?: string;
  created_at: string;
}

interface IngestionStatusResponse {
  job_id: string;
  status: string;
  progress: number;
  message: string;
  file_id?: string;
}

interface StreamHandler {
  onToken: (token: string) => void;
  onFinal: (result: QueryResponse) => void;
  onError: (message: string) => void;
}

function safeJsonParse(value: string) {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

export async function query(payload: QueryPayload): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/chat/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(`Query failed (${response.status})`);
  }

  const data = await response.json();
  return {
    answer: data.answer ?? data.response ?? "",
    sources: data.sources ?? [],
    images: data.images ?? []
  };
}

export async function getChatHistory(chatId: string): Promise<ChatHistoryResponse> {
  const response = await fetch(`${API_BASE_URL}/chat/history?chat_id=${encodeURIComponent(chatId)}`);
  if (!response.ok) {
    throw new Error(`History fetch failed (${response.status})`);
  }
  return response.json();
}

export async function createChat(): Promise<CreateChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat/create`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`Chat create failed (${response.status})`);
  }
  return response.json();
}

export async function deleteChat(chatId: string): Promise<{ status: string; chat_id: string }> {
  const response = await fetch(`${API_BASE_URL}/chat/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chatId })
  });
  if (!response.ok) {
    throw new Error(`Chat delete failed (${response.status})`);
  }
  return response.json();
}

export async function clearDatabase(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/db/clear`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`Database clear failed (${response.status})`);
  }
  return response.json();
}

export async function listChatSessions(): Promise<SessionInfo[]> {
  const response = await fetch(`${API_BASE_URL}/chat/sessions`);
  if (!response.ok) {
    throw new Error(`Sessions fetch failed (${response.status})`);
  }
  return response.json();
}

export async function getIngestionStatus(jobId: string): Promise<IngestionStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/ingestion/status/${encodeURIComponent(jobId)}`);
  if (!response.ok) {
    throw new Error(`Status check failed (${response.status})`);
  }
  return response.json();
}

export async function queryStream(payload: QueryPayload, handler: StreamHandler) {
  const response = await fetch(`${API_BASE_URL}/chat/query/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/x-ndjson, text/event-stream"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    handler.onError(`Query failed (${response.status})`);
    return;
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (!response.body || (!contentType.includes("application/x-ndjson") && !contentType.includes("text/event-stream"))) {
    const data = await response.json();
    handler.onFinal({
      answer: data.answer ?? data.response ?? "",
      sources: data.sources ?? [],
      images: data.images ?? []
    });
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let accumulated = "";
  let finalSources: QueryResponse["sources"] = [];
  let finalImages: QueryResponse["images"] = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      const trailing = buffer.trim();
      if (trailing) {
        const trailingData = trailing.startsWith("data:") ? trailing.replace(/^data:\s*/, "") : trailing;
        const trailingParsed = safeJsonParse(trailingData);
        if (trailingParsed && typeof trailingParsed === "object" && trailingParsed.type === "final") {
          const trailingFinal =
            trailingParsed.data && typeof trailingParsed.data === "object"
              ? (trailingParsed.data as Record<string, unknown>)
              : undefined;
          finalSources =
            (trailingParsed.sources as QueryResponse["sources"]) ??
            (trailingFinal?.sources as QueryResponse["sources"]) ??
            finalSources;
          finalImages =
            (trailingParsed.images as QueryResponse["images"]) ??
            (trailingFinal?.images as QueryResponse["images"]) ??
            finalImages;
        }
      }
      handler.onFinal({ answer: accumulated, sources: finalSources, images: finalImages });
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const data = chunk.trim().startsWith("data:")
        ? chunk.trim().replace(/^data:\s*/, "")
        : chunk.trim();
      if (!data) {
        continue;
      }

      const parsed = safeJsonParse(data);
      if (parsed && typeof parsed === "object") {
        if (parsed.type === "token" && typeof parsed.data === "string") {
          accumulated += parsed.data;
          handler.onToken(parsed.data);
        } else if (parsed.type === "final") {
          const finalData =
            parsed.data && typeof parsed.data === "object"
              ? (parsed.data as Record<string, unknown>)
              : undefined;
          finalSources =
            (parsed.sources as QueryResponse["sources"]) ??
            (finalData?.sources as QueryResponse["sources"]) ??
            finalSources;
          finalImages =
            (parsed.images as QueryResponse["images"]) ??
            (finalData?.images as QueryResponse["images"]) ??
            finalImages;
          handler.onFinal({
            answer: accumulated,
            sources: finalSources,
            images: finalImages
          });
          return;
        } else if (parsed.type === "error") {
          handler.onError(typeof parsed.message === "string" ? parsed.message : "Streaming error");
          return;
        }
      } else {
        accumulated += data;
        handler.onToken(data);
      }
    }
  }
}

export async function uploadDocument(
  file: File,
  chatId?: string,
  onProgress?: (progress: number) => void
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append("file", file);
    if (chatId) {
      formData.append("chat_id", chatId);
    }

    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE_URL}/document/upload`);

    request.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        const percent = Math.round((event.loaded / event.total) * 100);
        onProgress(percent);
      }
    };

    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        try {
          const data = JSON.parse(request.responseText) as UploadResponse;
          resolve(data);
        } catch {
          resolve({ status: "ok" });
        }
      } else {
        reject(new Error(`Upload failed (${request.status})`));
      }
    };

    request.onerror = () => {
      reject(new Error("Upload failed"));
    };

    request.send(formData);
  });
}
