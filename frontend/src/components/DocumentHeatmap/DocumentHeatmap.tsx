import { useMemo } from "react";
import { Flame } from "lucide-react";
import type { SourceChunk } from "../../types/chat";

interface DocumentHeatmapProps {
  sources?: SourceChunk[];
}

export function DocumentHeatmap({ sources = [] }: DocumentHeatmapProps) {
  const { maxPage, counts } = useMemo(() => {
    const counter: Record<number, number> = {};
    let max = 0;
    sources.forEach((source) => {
      const page = Number((source.metadata as Record<string, unknown> | undefined)?.page_no);
      if (!Number.isFinite(page)) {
        return;
      }
      max = Math.max(max, page);
      counter[page] = (counter[page] || 0) + 1;
    });
    return { maxPage: Math.max(max, 12), counts: counter };
  }, [sources]);

  const cells = Array.from({ length: maxPage }, (_, index) => {
    const page = index + 1;
    const hits = counts[page] || 0;
    const intensity = Math.min(1, hits / 3);
    return { page, intensity, hits };
  });

  return (
    <div className="rounded-2xl border border-border bg-panel/70 p-4 backdrop-blur-md">
      <div className="mb-3 flex items-center gap-2 text-sm font-medium text-text">
        <Flame size={16} className="text-accent2" />
        Document Heatmap
      </div>
      <div className="grid grid-cols-6 gap-2">
        {cells.map((cell) => (
          <div
            key={cell.page}
            className="flex h-10 flex-col items-center justify-center rounded-lg border border-border text-[10px] text-muted"
            style={{
              background: `rgba(139,92,246,${0.1 + cell.intensity * 0.6})`
            }}
            title={`Page ${cell.page} • ${cell.hits} hits`}
          >
            {cell.page}
          </div>
        ))}
      </div>
    </div>
  );
}
