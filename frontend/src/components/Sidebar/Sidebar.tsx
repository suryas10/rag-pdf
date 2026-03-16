import { ReactNode, useMemo } from "react";
import { Plus, Trash2, Database, MessageSquare } from "lucide-react";

export interface SessionSummary {
  chat_id: string;
  document_id?: string;
  document_name?: string;
  created_at: string;
}

interface SidebarProps {
  sessions: SessionSummary[];
  activeChatId?: string;
  uploadArea?: ReactNode;
  isCreatingChat?: boolean;
  isClearingDatabase?: boolean;
  onSelect: (chatId: string) => void;
  onCreate: () => void;
  onDelete: (chatId: string) => void;
  onClearDatabase: () => void;
}

export function Sidebar({
  sessions,
  activeChatId,
  uploadArea,
  isCreatingChat,
  isClearingDatabase,
  onSelect,
  onCreate,
  onDelete,
  onClearDatabase
}: SidebarProps) {
  const sorted = useMemo(() => {
    return [...sessions].sort((a, b) => b.created_at.localeCompare(a.created_at));
  }, [sessions]);

  return (
    <aside className="flex h-full min-h-0 flex-col gap-4 p-4">
      <button
        className="inline-flex items-center justify-center gap-2 rounded-xl border border-border px-3 py-2 text-sm text-text transition hover:border-accent2 disabled:cursor-not-allowed disabled:opacity-60"
        onClick={onCreate}
        disabled={Boolean(isCreatingChat) || Boolean(isClearingDatabase)}
        type="button"
      >
        <Plus size={16} />
        {isCreatingChat ? "Creating..." : "New Chat"}
      </button>

      {uploadArea ? uploadArea : null}

      <div className="text-xs uppercase tracking-wide text-muted">Chat Sessions</div>
      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        <div className="flex flex-col gap-2">
        {sorted.map((session) => {
          const active = session.chat_id === activeChatId;
          return (
            <div
              key={session.chat_id}
              className={
                active
                  ? "flex items-center justify-between rounded-xl border border-accent2 bg-[color:var(--bg-soft)]/70 px-3 py-2 text-left text-sm text-text"
                  : "flex items-center justify-between rounded-xl border border-border bg-[color:var(--bg-soft)]/40 px-3 py-2 text-left text-sm text-muted hover:border-accent2"
              }
            >
              <button
                className="flex min-w-0 flex-1 items-center gap-2 text-left"
                onClick={() => onSelect(session.chat_id)}
                type="button"
              >
                <MessageSquare size={14} />
                <span className="max-w-[140px] truncate">
                  {session.document_name || "Untitled document"}
                </span>
              </button>
              <button
                className="rounded-full p-1 text-muted hover:text-red-300"
                onClick={() => onDelete(session.chat_id)}
                type="button"
                aria-label="Delete chat"
              >
                <Trash2 size={14} />
              </button>
            </div>
          );
        })}
        </div>
      </div>

      <button
        className="inline-flex items-center justify-center gap-2 rounded-xl border border-border px-3 py-2 text-sm text-muted transition hover:border-red-300 hover:text-red-300"
        onClick={onClearDatabase}
        disabled={Boolean(isClearingDatabase)}
        type="button"
      >
        <Database size={16} />
        {isClearingDatabase ? "Clearing..." : "Clear Database"}
      </button>
    </aside>
  );
}
