import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { Copy, Check } from "lucide-react";
import type { ChatMessage } from "../../types/chat";

interface MessageBubbleProps {
  message: ChatMessage;
  showCaret?: boolean;
}

export function MessageBubble({ message, showCaret }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);
  const isAssistant = message.role === "assistant";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div
      className={
        isAssistant
          ? "rounded-2xl border border-border bg-[color:var(--bg-soft)]/70 p-4 shadow-sm"
          : "ml-auto max-w-[85%] rounded-2xl bg-accent/90 p-4 text-white"
      }
    >
      <div className="flex items-start justify-between gap-4">
        <div className="prose prose-invert max-w-none text-sm">
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
            {message.content || (isAssistant ? "" : message.content)}
          </ReactMarkdown>
          {showCaret ? <span className="typing-caret" /> : null}
        </div>
        {isAssistant ? (
          <button
            className="mt-1 inline-flex items-center gap-1 rounded-full border border-border px-2 py-1 text-xs text-muted transition hover:border-accent2 hover:text-text"
            onClick={handleCopy}
            type="button"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
            {copied ? "Copied" : "Copy"}
          </button>
        ) : null}
      </div>
    </div>
  );
}
