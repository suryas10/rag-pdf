export function TypingIndicator() {
  return (
    <div className="flex items-center gap-2 text-xs text-muted">
      <span className="h-2 w-2 animate-pulseDots rounded-full bg-accent" />
      <span className="h-2 w-2 animate-pulseDots rounded-full bg-accent2 [animation-delay:0.2s]" />
      <span className="h-2 w-2 animate-pulseDots rounded-full bg-accent [animation-delay:0.4s]" />
      <span>Thinking...</span>
    </div>
  );
}
