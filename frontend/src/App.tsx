import { useMemo, useState } from "react";
import { ChatWindow } from "./components/ChatWindow/ChatWindow";
import { ChatInput } from "./components/ChatInput/ChatInput";
import { useChat } from "./hooks/useChat";
import { DocumentUploader } from "./components/DocumentUploader/DocumentUploader";
import { useUpload } from "./hooks/useUpload";
import { SourceViewer } from "./components/SourceViewer/SourceViewer";
import { ImagePreview } from "./components/ImagePreview/ImagePreview";
import { clearDatabase, createChat, deleteChat } from "./services/apiClient";
import { Sidebar } from "./components/Sidebar/Sidebar";
import { useSessions } from "./hooks/useSessions";

const DEMO_SYSTEM_LABEL = "Hybrid WiM-RAG Assistant";

export default function App() {
  const {
    messages,
    isStreaming,
    sendMessage,
    error,
    isLoadingHistory,
    setChatId,
    resetMessages,
    chatId
  } = useChat();
  const { state: uploadState, canChat, uploadFile, reset } = useUpload(chatId);
  const [isCreatingChat, setIsCreatingChat] = useState(false);
  const { sessions, refresh } = useSessions();
  const [isClearingDb, setIsClearingDb] = useState(false);

  const activeSessionHasDocument = useMemo(() => {
    const activeSession = sessions.find((session) => session.chat_id === chatId);
    return Boolean(activeSession?.document_id);
  }, [sessions, chatId]);

  const canSend = useMemo(
    () => !isStreaming && (canChat || activeSessionHasDocument),
    [isStreaming, canChat, activeSessionHasDocument]
  );
  const activeSources = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((msg) => msg.role === "assistant" && msg.sources?.length);
    return lastAssistant?.sources ?? [];
  }, [messages]);
  const activeImages = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((msg) => msg.role === "assistant");
    const messageImages = lastAssistant?.images ?? [];
    if (messageImages.length) {
      return messageImages;
    }
    const sourceImages = (lastAssistant?.sources ?? [])
      .map((source) => (source.metadata as Record<string, unknown> | undefined)?.image_path as string | undefined)
      .filter((value): value is string => Boolean(value));
    return sourceImages;
  }, [messages]);

  const handleNewChat = async () => {
    setIsCreatingChat(true);
    try {
      const result = await createChat();
      setChatId(result.chat_id);
      reset();
      resetMessages();
      refresh();
    } catch (err) {
      // Keep existing chat if creation fails.
    } finally {
      setIsCreatingChat(false);
    }
  };

  const handleSelectChat = (selectedChatId: string) => {
    setChatId(selectedChatId);
    reset();
    resetMessages();
  };

  const handleDeleteChat = async (selectedChatId: string) => {
    try {
      await deleteChat(selectedChatId);
      if (selectedChatId === chatId) {
        const result = await createChat();
        setChatId(result.chat_id);
        reset();
        resetMessages();
      }
      refresh();
    } catch (err) {
      // Ignore delete errors for now.
    }
  };

  const handleClearDatabase = async () => {
    if (!window.confirm("Clear all chats, documents, and embeddings?")) {
      return;
    }
    setIsClearingDb(true);
    try {
      await clearDatabase();
      const result = await createChat();
      setChatId(result.chat_id);
      reset();
      resetMessages();
      refresh();
    } catch (err) {
      // Ignore errors for now.
    } finally {
      setIsClearingDb(false);
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden bg-[radial-gradient(ellipse_at_top,_#111827_0%,_#0F172A_55%,_#0B1224_100%)] text-text">
      <div className="grid h-full w-full grid-cols-[22%_56%_22%] divide-x divide-border">
        <Sidebar
          sessions={sessions}
          activeChatId={chatId}
          uploadArea={(
            <DocumentUploader
              stage={uploadState.stage}
              progress={uploadState.progress}
              message={uploadState.message}
              documentName={uploadState.documentName}
              onUpload={uploadFile}
              onReplace={reset}
            />
          )}
          isCreatingChat={isCreatingChat}
          isClearingDatabase={isClearingDb}
          onSelect={handleSelectChat}
          onCreate={handleNewChat}
          onDelete={handleDeleteChat}
          onClearDatabase={handleClearDatabase}
        />

        <main className="flex h-full min-h-0 flex-col p-4">
          <header className="mb-3 border-b border-border pb-3">
            <h1 className="text-xl font-semibold tracking-tight text-text">{DEMO_SYSTEM_LABEL}</h1>
          </header>

          <div className="flex min-h-0 flex-1 flex-col rounded-2xl border border-border bg-panel/70 p-3 shadow-[0_0_24px_rgba(15,23,42,0.5)] backdrop-blur-md">
            <div className="min-h-0 flex-1">
              <ChatWindow messages={messages} isStreaming={isStreaming} isLoading={isLoadingHistory} />
            </div>

            <div className="mt-3 border-t border-border pt-3">
              <ChatInput
                disabled={!canSend}
                onSend={sendMessage}
                placeholder="Ask about your document..."
              />
              {!canChat && !activeSessionHasDocument ? (
                <p className="mt-2 text-xs text-muted">
                  Upload a PDF to begin chatting.
                </p>
              ) : null}
              {error ? (
                <p className="mt-2 text-xs text-red-300">{error}</p>
              ) : null}
            </div>
          </div>
        </main>

        <aside className="flex h-full min-h-0 flex-col gap-3 p-4">
          <div className="text-xs uppercase tracking-wide text-muted">Retrieved Sources</div>
          <div className="min-h-0 flex-1 overflow-y-auto rounded-2xl border border-border bg-panel/70 p-3 backdrop-blur-md">
            <SourceViewer sources={activeSources} isLoading={isStreaming} />
          </div>
          <div className="max-h-[34%] overflow-y-auto rounded-2xl border border-border bg-panel/70 p-3 backdrop-blur-md">
            <ImagePreview images={activeImages} />
          </div>
        </aside>
        </div>
    </div>
  );
}
