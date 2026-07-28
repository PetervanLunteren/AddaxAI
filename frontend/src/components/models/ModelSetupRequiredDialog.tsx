/**
 * Project-scoped dialog that appears when one or more of the models
 * configured for the current project are not ready for inference.
 *
 * Two entry points:
 *  - Auto-opened by AppLayout when /model-readiness reports `ready:false`.
 *  - Force-opened by the QueueCard pre-analysis check.
 *
 * Each missing model gets its own status row with an inline progress
 * bar driven by the same WebSocket the settings page uses for
 * /api/ml/models/{id}/prepare. The dialog has a single bottom "Set up"
 * button that runs every not-done row sequentially in pipeline order
 * (det → cls → emb). With one missing model this is just "set up
 * that one"; with several it walks through them serially, which both
 * keeps the UI tidy and avoids the rare race where two parallel calls
 * try to (re)build the same env directory. The dialog auto-closes
 * once every model is ready.
 */

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Check, Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { projectsApi } from "@/api/projects";
import { mlModelsApi } from "@/api/ml-models";
import { invalidateModelMetadata } from "@/api/models";
import { useTaskProgress } from "@/hooks/useTaskProgress";
import { useModelSetupGate } from "@/lib/model-setup-gate";
import type { MissingModel } from "@/api/types";

interface ModelSetupRequiredDialogProps {
  projectId: string;
}

export function ModelSetupRequiredDialog({ projectId }: ModelSetupRequiredDialogProps) {
  const [dismissed, setDismissed] = useState(false);
  const { forceOpen, reset: resetForceOpen } = useModelSetupGate();

  // runQueue: FIFO of model_ids the user asked us to set up. Head of
  // the queue auto-starts; when that row reports finished, we shift the
  // queue and the next row's shouldStart flips true. Empty queue means
  // nothing is running.
  const [runQueue, setRunQueue] = useState<string[]>([]);
  // Rows reporting "I'm in flight right now". Drives the bottom
  // button's disabled state.
  const [busyIds, setBusyIds] = useState<Set<string>>(new Set());
  // Tracks rows that have reported "Done" so the queue-builder can
  // skip them on a retry click without round-tripping the readiness
  // query first.
  const [doneIds, setDoneIds] = useState<Set<string>>(new Set());

  const { data } = useQuery({
    queryKey: ["project-model-readiness", projectId],
    queryFn: () => projectsApi.getModelReadiness(projectId),
    enabled: !!projectId,
    refetchInterval: 30_000,
  });

  // Reset state whenever the project changes.
  useEffect(() => {
    setDismissed(false);
    setRunQueue([]);
    setBusyIds(new Set());
    setDoneIds(new Set());
    resetForceOpen();
  }, [projectId, resetForceOpen]);

  // Auto-close once everything is ready.
  useEffect(() => {
    if (data?.ready) {
      setDismissed(false);
      setRunQueue([]);
      setBusyIds(new Set());
      setDoneIds(new Set());
      resetForceOpen();
    }
  }, [data?.ready, resetForceOpen]);

  if (!data) return null;
  const hasMissing = data.missing.length > 0;
  const open = hasMissing && (!dismissed || forceOpen);
  const currentRunId = runQueue[0] ?? null;
  const running = runQueue.length > 0 || busyIds.size > 0;
  const allMissingDone = data.missing.every((m) => doneIds.has(m.model_id));

  const handleOpenChange = (next: boolean) => {
    if (!next) {
      setDismissed(true);
      resetForceOpen();
    }
  };

  const handleSetup = () => {
    // Queue every not-yet-done row in the order the backend returned
    // them (det → cls → emb). Errored rows are included so a click
    // doubles as a retry.
    setRunQueue(
      data.missing
        .map((m) => m.model_id)
        .filter((id) => !doneIds.has(id) && !busyIds.has(id)),
    );
  };

  const handleRowStarted = (id: string) => {
    setBusyIds((s) => {
      const n = new Set(s);
      n.add(id);
      return n;
    });
  };

  const handleRowFinished = (id: string, ok: boolean) => {
    setBusyIds((s) => {
      const n = new Set(s);
      n.delete(id);
      return n;
    });
    setRunQueue((q) => q.filter((qid) => qid !== id));
    if (ok) {
      setDoneIds((s) => {
        const n = new Set(s);
        n.add(id);
        return n;
      });
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle
              className="h-5 w-5 shrink-0"
              style={{ color: "#882000" }}
            />
            Some models for this project need setup
          </DialogTitle>
          <DialogDescription>
            Before you can run analyses, the following models need to be
            downloaded or rebuilt.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3 py-2">
          {data.missing.map((m) => (
            <MissingModelRow
              key={m.model_id}
              missing={m}
              projectId={projectId}
              shouldStart={m.model_id === currentRunId}
              onStarted={handleRowStarted}
              onFinished={handleRowFinished}
            />
          ))}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => handleOpenChange(false)}>
            Close
          </Button>
          {!allMissingDone && (
            <Button onClick={handleSetup} disabled={running}>
              {running ? (
                <>
                  <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  Setting up
                </>
              ) : (
                "Set up"
              )}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function describeNeeds(missing: MissingModel): string {
  if (missing.needs_weights && missing.needs_env) {
    return "Needs weights and environment";
  }
  if (missing.needs_weights) return "Needs weights";
  if (missing.needs_env) return "Needs environment";
  return "Needs setup";
}

interface MissingModelRowProps {
  missing: MissingModel;
  projectId: string;
  /** When true, the row should kick off its own setup if it's idle. */
  shouldStart: boolean;
  onStarted: (modelId: string) => void;
  onFinished: (modelId: string, ok: boolean) => void;
}

function MissingModelRow({
  missing,
  projectId,
  shouldStart,
  onStarted,
  onFinished,
}: MissingModelRowProps) {
  const queryClient = useQueryClient();
  const [taskId, setTaskId] = useState<string | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  // Used to detect the false→true transition on shouldStart so a retry
  // (parent re-queues this row after a prior error) actually fires
  // even though the prop value hasn't changed between renders.
  const prevShouldStart = useRef(false);

  const { message, progress } = useTaskProgress({
    taskId,
    onComplete: () => {
      setTaskId(null);
      setDone(true);
      void queryClient.invalidateQueries({
        queryKey: ["project-model-readiness", projectId],
      });
      invalidateModelMetadata(queryClient, missing.model_id);
      onFinished(missing.model_id, true);
    },
    onError: (msg) => {
      setTaskId(null);
      setErrorText(msg || "Setup failed");
      onFinished(missing.model_id, false);
    },
  });

  const prepare = useMutation({
    mutationFn: () => mlModelsApi.prepare(missing.model_id),
    onSuccess: (resp) => {
      setErrorText(null);
      setTaskId(resp.task_id);
      onStarted(missing.model_id);
    },
    onError: (err: Error) => {
      setErrorText(err.message || "Failed to start setup");
      onFinished(missing.model_id, false);
    },
  });

  const inProgress = taskId !== null || prepare.isPending;
  const pct = Math.min(100, Math.max(0, Math.round(progress * 100)));

  useEffect(() => {
    const justFlippedTrue = shouldStart && !prevShouldStart.current;
    prevShouldStart.current = shouldStart;
    if (!justFlippedTrue) return;
    if (inProgress || done) return;
    setErrorText(null);
    prepare.mutate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shouldStart]);

  return (
    <div className="rounded-md border p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium">
            <span aria-hidden="true">{missing.emoji}</span>{" "}
            {missing.friendly_name}
          </div>
          <div className="text-xs text-muted-foreground">
            {describeNeeds(missing)}
          </div>
        </div>
        <div className="shrink-0 text-sm">
          {done && (
            <span
              className="inline-flex items-center gap-1"
              style={{ color: "#0f6064" }}
            >
              <Check className="h-4 w-4" />
              Done
            </span>
          )}
          {!done && inProgress && (
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          )}
        </div>
      </div>
      {inProgress && (
        <div className="mt-3 space-y-1">
          <Progress value={pct} className="h-1.5" />
          <div className="truncate text-xs text-muted-foreground">
            {message || "Starting..."}
          </div>
        </div>
      )}
      {errorText && (
        <div
          className="mt-2 text-xs"
          style={{ color: "#882000" }}
        >
          {errorText}
        </div>
      )}
    </div>
  );
}
