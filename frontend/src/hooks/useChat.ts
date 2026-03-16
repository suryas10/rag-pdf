import { useCallback, useEffect, useMemo, useState } from "react";
import { nanoid } from "nanoid";
import type { ChatMessage, QueryResponse } from "../types/chat";
import { getChatHistory, queryStream } from "../services/apiClient";

const LOCAL_CHAT_KEY = "wim_rag_chat_id";

function getOrCreateLocalChatId() {
  if (typeof window === "undefined") {
    return nanoid();
  }
  const existing = window.localStorage.getItem(LOCAL_CHAT_KEY);
  if (existing) {
    return existing;
  }
  const created = nanoid();
  window.localStorage.setItem(LOCAL_CHAT_KEY, created);
  return created;
}

export function useChat(chatId?: string) {
  const [activeChatId, setActiveChatId] = useState(() => chatId ?? getOrCreateLocalChatId());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  useEffect(() => {
    if (chatId && chatId !== activeChatId) {
      setActiveChatId(chatId);
    }
  }, [chatId, activeChatId]);

  useEffect(() => {
    let isMounted = true;
    async function loadHistory() {
      try {
        setIsLoadingHistory(true);
        const history = await getChatHistory(activeChatId);
        if (!isMounted) {
          return;
        }
        const mapped = history.messages.map((message) => ({
          id: nanoid(),
          role: message.role,
          content: message.content,
          createdAt: message.timestamp,
          sources: message.sources
        }));
        setMessages(mapped);
      } catch {
        if (!isMounted) {
          return;
        }
      } finally {
        if (isMounted) {
          setIsLoadingHistory(false);
        }
      }
    }

    loadHistory();
    return () => {
      isMounted = false;
    };
  }, [activeChatId]);

  const setChatId = useCallback((newChatId: string) => {
    setActiveChatId(newChatId);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(LOCAL_CHAT_KEY, newChatId);
    }
  }, []);

  const resetMessages = useCallback(() => {
    setMessages([]);
  }, []);

  const updateAssistantMessage = useCallback((id: string, updater: (prev: ChatMessage) => ChatMessage) => {
    setMessages((prev) => prev.map((message) => (message.id === id ? updater(message) : message)));
  }, []);

  const handleFinal = useCallback((assistantId: string, result: QueryResponse) => {
    updateAssistantMessage(assistantId, (prev) => ({
      ...prev,
      content: result.answer || prev.content,
      sources: result.sources ?? prev.sources,
      images: result.images ?? prev.images
    }));
  }, [updateAssistantMessage]);

  const sendMessage = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) {
      return;
    }

    setError(null);
    const userMessage: ChatMessage = {
      id: nanoid(),
      role: "user",
      content: trimmed,
      createdAt: new Date().toISOString()
    };

    const assistantId = nanoid();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      createdAt: new Date().toISOString()
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsStreaming(true);

    try {
      await queryStream(
        { query: trimmed, chat_id: activeChatId },
        {
          onToken: (token) => {
            updateAssistantMessage(assistantId, (prev) => ({
              ...prev,
              content: prev.content + token
            }));
          },
          onFinal: (result) => {
            handleFinal(assistantId, result);
            setIsStreaming(false);
          },
          onError: (message) => {
            setError(message);
            setIsStreaming(false);
          }
        }
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
      setIsStreaming(false);
    }
  }, [activeChatId, handleFinal, updateAssistantMessage]);

  return {
    messages,
    isStreaming,
    sendMessage,
    error,
    chatId: activeChatId,
    isLoadingHistory,
    setChatId,
    resetMessages
  };
}
