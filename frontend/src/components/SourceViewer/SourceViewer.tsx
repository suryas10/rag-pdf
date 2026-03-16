import { useMemo, useState } from "react";
import { FileText, Image as ImageIcon, Sparkles } from "lucide-react";
import type { SourceChunk } from "../../types/chat";
import { PanelSkeleton } from "../Loading/Skeletons";

interface SourceViewerProps {
  sources?: SourceChunk[];
  isLoading?: boolean;
}

function getMetadataValue<T>(metadata: Record<string, unknown> | undefined, key: string, fallback?: T) {
  if (!metadata) {
    return fallback;
  }
  return (metadata[key] as T) ?? fallback;
}

function isRenderableUrl(path?: string) {
  if (!path) {
    return false;
  }
  return path.startsWith("http") || path.startsWith("/");
}

export function SourceViewer({ sources = [], isLoading }: SourceViewerProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const normalizedSources = useMemo(() => sources.filter(Boolean), [sources]);
  const selected = normalizedSources[selectedIndex];

  if (isLoading && normalizedSources.length === 0) {
    return (
      <div className="rounded-2xl border border-border bg-panel/60 p-5 text-sm text-muted backdrop-blur-md">
        <PanelSkeleton />
      </div>
    );
  }

  if (normalizedSources.length === 0) {
    return (
      <div className="rounded-2xl border border-border bg-panel/60 p-5 text-sm text-muted backdrop-blur-md">
        Sources will appear here after a response is generated.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="rounded-2xl border border-border bg-panel/70 p-4 backdrop-blur-md">
        <div className="mb-3 flex items-center gap-2 text-sm font-medium text-text">
          <Sparkles size={16} className="text-accent2" />
          Retrieved Sources
        </div>
        <div className="flex flex-col gap-3">
          {normalizedSources.map((source, index) => {
            const metadata = source.metadata as Record<string, unknown> | undefined;
            const type = getMetadataValue<string>(metadata, "type", "text");
            const page = getMetadataValue<number | string>(metadata, "page_no", "?");
            const score = typeof source.score === "number" ? source.score.toFixed(2) : "--";
            const preview = source.text ? `${source.text.slice(0, 180)}${source.text.length > 180 ? "..." : ""}` : "Image source";
            const active = index === selectedIndex;

            return (
              <button
                key={`${type}-${index}`}
                className={
                  active
                    ? "rounded-xl border border-accent2 bg-[color:var(--bg-soft)]/70 p-3 text-left text-sm text-text"
                    : "rounded-xl border border-border bg-[color:var(--bg-soft)]/40 p-3 text-left text-sm text-muted hover:border-accent2"
                }
                onClick={() => setSelectedIndex(index)}
                type="button"
              >
                <div className="flex items-center justify-between">
                  <span className="inline-flex items-center gap-2 text-xs uppercase tracking-wide text-muted">
                    {type === "image" ? <ImageIcon size={14} /> : <FileText size={14} />}
                    {type}
                  </span>
                  <span className="text-xs text-muted">Page {page} · {score}</span>
                </div>
                <div className="mt-2 text-sm text-text">
                  {preview}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex-1 rounded-2xl border border-border bg-panel/70 p-4 backdrop-blur-md">
        {selected ? (
          <div className="flex h-full flex-col">
            <div className="mb-3 flex items-center justify-between text-sm text-muted">
              <span>Source Preview</span>
              <span>Page {getMetadataValue<number | string>(selected.metadata as Record<string, unknown> | undefined, "page_no", "?")}</span>
            </div>
            {(() => {
              const metadata = selected.metadata as Record<string, unknown> | undefined;
              const pageImage = getMetadataValue<string>(metadata, "page_image", "");
              const imagePath = getMetadataValue<string>(metadata, "image_path", "");
              const previewPath = pageImage || imagePath;

              if (isRenderableUrl(previewPath)) {
                return (
                  <img
                    src={previewPath}
                    alt="Source preview"
                    className="h-full w-full rounded-xl object-contain"
                  />
                );
              }

              return (
                <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-border bg-[color:var(--bg-soft)]/60 text-sm text-muted">
                  Preview unavailable. Source page: {getMetadataValue<number | string>(metadata, "page_no", "?")}
                </div>
              );
            })()}
          </div>
        ) : null}
      </div>
    </div>
  );
}
