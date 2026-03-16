import { useCallback, useEffect, useState } from "react";
import { listChatSessions } from "../services/apiClient";
import type { SessionSummary } from "../components/Sidebar/Sidebar";

export function useSessions() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await listChatSessions();
      const mapped = data.map((session) => ({
        chat_id: session.chat_id,
        document_id: session.document_id,
        document_name: session.document_name,
        created_at: session.created_at
      }));
      setSessions(mapped);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    sessions,
    isLoading,
    refresh
  };
}
