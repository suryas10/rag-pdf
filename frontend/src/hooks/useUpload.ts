import { useCallback, useMemo, useRef, useState } from "react";
import { getIngestionStatus, uploadDocument } from "../services/apiClient";

export type UploadStage = "idle" | "validating" | "uploading" | "processing" | "ready" | "error";

interface UploadState {
  stage: UploadStage;
  progress: number;
  message: string;
  documentName?: string;
}

const MAX_SIZE_MB = 50;

const POLL_INTERVAL_MS = 1200;

export function useUpload(chatId?: string) {
  const [state, setState] = useState<UploadState>({
    stage: "idle",
    progress: 0,
    message: "",
    documentName: undefined
  });
  const pollTimerRef = useRef<number | null>(null);

  const canChat = useMemo(() => state.stage === "ready", [state.stage]);

  const reset = useCallback(() => {
    if (pollTimerRef.current) {
      window.clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    setState({ stage: "idle", progress: 0, message: "", documentName: undefined });
  }, []);

  const uploadFile = useCallback(async (file: File) => {
    if (!file) {
      return;
    }

    if (!chatId) {
      setState({ stage: "error", progress: 0, message: "Create a chat before uploading.", documentName: file.name });
      return;
    }

    setState({ stage: "validating", progress: 0, message: "Validating PDF...", documentName: file.name });

    if (file.type !== "application/pdf") {
      setState({ stage: "error", progress: 0, message: "Only PDF files are supported.", documentName: file.name });
      return;
    }

    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setState({ stage: "error", progress: 0, message: `File exceeds ${MAX_SIZE_MB}MB.`, documentName: file.name });
      return;
    }

    setState({ stage: "uploading", progress: 0, message: "Uploading document...", documentName: file.name });

    try {
      const result = await uploadDocument(file, chatId, (progress) => {
        setState((prev) => ({ ...prev, stage: "uploading", progress }));
      });

      if (!result || !result.job_id) {
        setState({ stage: "error", progress: 0, message: "Upload failed.", documentName: file.name });
        return;
      }

      const pollStatus = async () => {
        try {
          const status = await getIngestionStatus(result.job_id as string);
          if (status.status === "completed") {
            setState({
              stage: "ready",
              progress: 100,
              message: "Document ready",
              documentName: file.name
            });
            return;
          }
          if (status.status === "error") {
            setState({
              stage: "error",
              progress: 0,
              message: status.message || "Ingestion failed.",
              documentName: file.name
            });
            return;
          }
          setState({
            stage: "processing",
            progress: Math.round((status.progress ?? 0) * 100),
            message: status.message || "Parsing and indexing...",
            documentName: file.name
          });
          pollTimerRef.current = window.setTimeout(pollStatus, POLL_INTERVAL_MS);
        } catch (error) {
          setState({
            stage: "error",
            progress: 0,
            message: error instanceof Error ? error.message : "Ingestion status failed.",
            documentName: file.name
          });
        }
      };

      pollStatus();
    } catch (error) {
      setState({
        stage: "error",
        progress: 0,
        message: error instanceof Error ? error.message : "Upload failed.",
        documentName: file.name
      });
    }
  }, [chatId]);

  return {
    state,
    canChat,
    uploadFile,
    reset
  };
}
