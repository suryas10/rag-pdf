import { useEffect, useRef } from "react";
import type { ChatMessage } from "../../types/chat";
import { MessageBubble } from "../MessageBubble/MessageBubble";
import { ChatSkeleton } from "../Loading/Skeletons";
import { TypingIndicator } from "../Loading/TypingIndicator";

interface ChatWindowProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  isLoading?: boolean;
}

export function ChatWindow({ messages, isStreaming, isLoading }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto pr-2">
      {messages.length === 0 && isLoading ? (
        <ChatSkeleton />
      ) : null}

      {messages.length === 0 && !isLoading ? (
        <div className="rounded-2xl border border-dashed border-border bg-[color:var(--bg-soft)]/60 p-6 text-sm text-muted">
          Upload a document to begin, then ask detailed questions.
        </div>
      ) : null}

      {messages.map((message, index) => {
        const isLast = index === messages.length - 1;
        const showCaret = isStreaming && isLast && message.role === "assistant";
        return (
          <MessageBubble key={message.id} message={message} showCaret={showCaret} />
        );
      })}

      {isStreaming ? <TypingIndicator /> : null}
      <div ref={bottomRef} />
    </div>
  );
}
