/**
 * Progress modal for the async CamtrapDP export job.
 *
 * Subscribes to the job via useTaskProgress. When the job completes,
 * calls onComplete(jobId) so the parent can fetch the finished ZIP.
 * When it errors, calls onError(message).
 */

import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Progress } from "../ui/progress";
import { useTaskProgress } from "../../hooks/useTaskProgress";

interface CamtrapDPProgressModalProps {
  jobId: string | null;
  /** When true, the backend is generating thumbnails; shows file-level
   * progress. When false, the server packages metadata only and
   * finishes in a moment; the modal still flashes briefly. */
  includesThumbnails: boolean;
  onComplete: (jobId: string) => void;
  onError: (message: string | null) => void;
}

export function CamtrapDPProgressModal({
  jobId,
  includesThumbnails,
  onComplete,
  onError,
}: CamtrapDPProgressModalProps) {
  const [handled, setHandled] = useState(false);

  useEffect(() => {
    setHandled(false);
  }, [jobId]);

  const { message, metrics, phaseProgress } = useTaskProgress({
    taskId: jobId,
    onComplete: () => {
      if (!jobId || handled) return;
      setHandled(true);
      onComplete(jobId);
    },
    onError: (msg) => {
      if (handled) return;
      setHandled(true);
      onError(msg);
    },
  });

  const open = jobId !== null;
  const current = metrics?.current ?? 0;
  const total = metrics?.total ?? 0;
  const fraction =
    total > 0 ? current / total : phaseProgress ?? 0;
  const pct = Math.round(fraction * 100);

  return (
    <Dialog open={open}>
      <DialogContent className="sm:max-w-md [&>button.absolute]:hidden">
        <DialogHeader>
          <DialogTitle>Preparing Camtrap DP package</DialogTitle>
          <DialogDescription>
            {includesThumbnails
              ? "Generating thumbnails and bundling them in the ZIP."
              : "Packaging metadata into the ZIP."}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 py-2">
          <div className="flex items-center gap-3">
            <Loader2
              className="h-5 w-5 animate-spin shrink-0"
              style={{ color: "#0f6064" }}
            />
            <span className="text-sm font-medium text-gray-900">
              {total > 0
                ? `Thumbnail ${current.toLocaleString()} of ${total.toLocaleString()}`
                : message || "Starting..."}
            </span>
            {total > 0 && (
              <span className="ml-auto text-xs font-mono text-muted-foreground">
                {pct}%
              </span>
            )}
          </div>
          {total > 0 && <Progress value={pct} className="h-2" />}
        </div>
      </DialogContent>
    </Dialog>
  );
}
