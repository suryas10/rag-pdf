import { useMemo, useState } from "react";
import { Image as ImageIcon, X } from "lucide-react";

interface ImagePreviewProps {
  images?: string[];
}

export function ImagePreview({ images = [] }: ImagePreviewProps) {
  const [active, setActive] = useState<string | null>(null);
  const normalized = useMemo(() => images.filter(Boolean), [images]);

  if (normalized.length === 0) {
    return (
      <div className="rounded-2xl border border-border bg-panel/60 p-5 text-sm text-muted backdrop-blur-md">
        Image previews will appear here when visual sources are retrieved.
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-border bg-panel/70 p-4 backdrop-blur-md">
      <div className="mb-3 flex items-center gap-2 text-sm font-medium text-text">
        <ImageIcon size={16} className="text-accent" />
        Visual Retrieval
      </div>
      <div className="grid grid-cols-2 gap-3">
        {normalized.map((src) => (
          <button
            key={src}
            className="overflow-hidden rounded-xl border border-border bg-[color:var(--bg-soft)]/70"
            onClick={() => setActive(src)}
            type="button"
          >
            <img src={src} alt="Retrieved" className="h-28 w-full object-cover" />
          </button>
        ))}
      </div>

      {active ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6">
          <div className="relative w-full max-w-3xl rounded-2xl border border-border bg-panel/90 p-4 backdrop-blur-md">
            <button
              className="absolute right-4 top-4 inline-flex items-center gap-1 rounded-full border border-border px-2 py-1 text-xs text-text"
              onClick={() => setActive(null)}
              type="button"
            >
              <X size={14} /> Close
            </button>
            <img src={active} alt="Preview" className="max-h-[70vh] w-full rounded-xl object-contain" />
          </div>
        </div>
      ) : null}
    </div>
  );
}
