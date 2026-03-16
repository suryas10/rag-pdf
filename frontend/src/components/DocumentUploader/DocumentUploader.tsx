import { useCallback, useRef, useState } from "react";
import { UploadCloud, FileText, RefreshCcw } from "lucide-react";
import type { UploadStage } from "../../hooks/useUpload";

interface DocumentUploaderProps {
  stage: UploadStage;
  progress: number;
  message: string;
  documentName?: string;
  onUpload: (file: File) => void;
  onReplace?: () => void;
}

export function DocumentUploader({
  stage,
  progress,
  message,
  documentName,
  onUpload,
  onReplace
}: DocumentUploaderProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = useCallback((files: FileList | null) => {
    if (!files || files.length === 0) {
      return;
    }
    onUpload(files[0]);
  }, [onUpload]);

  const canReplace = stage === "ready";

  return (
    <div className="rounded-2xl border border-border bg-panel/70 p-5 backdrop-blur-md">
      <div
        className={
          isDragging
            ? "rounded-2xl border border-dashed border-accent2 bg-[color:var(--bg-soft)]/60 p-6 text-center"
            : "rounded-2xl border border-dashed border-border bg-[color:var(--bg-soft)]/60 p-6 text-center"
        }
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          handleFiles(event.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(event) => handleFiles(event.target.files)}
        />

        <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-2xl border border-border bg-panel/80">
          <UploadCloud size={22} className="text-accent2" />
        </div>
        <h3 className="text-base font-semibold text-text">Upload a PDF</h3>
        <p className="mt-1 text-sm text-muted">
          Drag and drop your document here, or browse your files.
        </p>
        <button
          type="button"
          className="mt-4 inline-flex items-center gap-2 rounded-full border border-border px-4 py-2 text-sm text-text transition hover:border-accent2"
          onClick={() => inputRef.current?.click()}
        >
          <FileText size={16} />
          Choose file
        </button>
      </div>

      <div className="mt-4 flex items-center justify-between gap-3 rounded-xl border border-border bg-[color:var(--bg-soft)]/60 p-4">
        <div>
          <p className="text-sm font-medium text-text">
            {documentName ? documentName : "No document selected"}
          </p>
          <p className="text-xs text-muted">{message || "Waiting for upload"}</p>
        </div>
        {canReplace && onReplace ? (
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-full border border-border px-3 py-1 text-xs text-text transition hover:border-accent2"
            onClick={onReplace}
          >
            <RefreshCcw size={14} />
            Replace
          </button>
        ) : null}
      </div>

      <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-[color:var(--bg-soft)]/70">
        <div
          className="h-full rounded-full bg-gradient-to-r from-accent to-accent2 transition-all"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}
