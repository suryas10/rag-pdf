import { useState } from "react";
import { Send } from "lucide-react";

interface ChatInputProps {
  disabled?: boolean;
  placeholder?: string;
  onSend: (text: string) => void;
}

export function ChatInput({ disabled, placeholder, onSend }: ChatInputProps) {
  const [value, setValue] = useState("");

  const handleSubmit = () => {
    if (!value.trim()) {
      return;
    }
    onSend(value);
    setValue("");
  };

  return (
    <div className="flex items-end gap-3">
    <textarea
        className="h-20 flex-1 resize-none rounded-xl border border-border bg-[color:var(--bg-soft)]/70 px-4 py-3 text-sm text-black focus:outline-none focus:ring-2 focus:ring-accent"
        placeholder={placeholder}
        value={value}
        disabled={disabled}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            handleSubmit();
            }
        }}
        />
      <button
        type="button"
        disabled={disabled}
        className="inline-flex h-11 items-center gap-2 rounded-xl bg-accent px-4 text-sm font-medium text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
        onClick={handleSubmit}
      >
        <Send size={16} />
        Send
      </button>
    </div>
  );
}
