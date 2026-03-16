export function ChatSkeleton() {
  return (
    <div className="flex flex-col gap-3">
      <div className="h-12 w-3/4 rounded-2xl bg-[color:var(--bg-soft)]/80" />
      <div className="h-12 w-5/6 rounded-2xl bg-[color:var(--bg-soft)]/70" />
      <div className="h-10 w-2/3 rounded-2xl bg-[color:var(--bg-soft)]/60" />
    </div>
  );
}

export function PanelSkeleton() {
  return (
    <div className="flex flex-col gap-3">
      <div className="h-4 w-2/3 rounded-lg bg-[color:var(--bg-soft)]/80" />
      <div className="h-20 w-full rounded-xl bg-[color:var(--bg-soft)]/70" />
      <div className="h-20 w-full rounded-xl bg-[color:var(--bg-soft)]/60" />
      <div className="h-20 w-full rounded-xl bg-[color:var(--bg-soft)]/50" />
    </div>
  );
}
