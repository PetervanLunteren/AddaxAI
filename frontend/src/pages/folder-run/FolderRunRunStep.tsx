/**
 * Step 3: Run analysis.
 *
 * Mirrors the way the Timelapse window kicks off analysis:
 *
 * 1. Verify the project's models are ready (weights + env on disk).
 *    If anything is missing, surface the standard setup dialog the
 *    main app uses (ModelSetupRequiredDialog is mounted globally via
 *    AppLayout for project routes; for folder runs we render an
 *    explicit warning + link to the relevant project's Settings page).
 * 2. POST /api/deployment-queue/process — the backend creates a job
 *    and registers a ready-gated worker.
 * 3. Subscribe to the job via useTaskProgress. The hook auto-sends
 *    the "ready" signal once the WebSocket is open, which triggers
 *    the worker.
 * 4. Render AnalysisProgress (same component the Timelapse page
 *    uses) while the run is in flight.
 * 5. On completion, persist step='review' and navigate.
 *
 * Cancel returns the user to the start of this step so they can
 * adjust models and try again.
 */

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { AlertTriangle, ArrowLeft, Play, X } from "lucide-react";

import { Button } from "../../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../../components/ui/card";
import { AnalysisProgress } from "../../components/analyses/AnalysisProgress";
import { useTaskProgress } from "../../hooks/useTaskProgress";

import { deploymentQueueApi } from "../../api/deployment-queue";
import { folderRunsApi } from "../../api/folder-runs";
import { projectsApi } from "../../api/projects";
import { useFolderRun } from "./FolderRunLayout";

type RunStage = "idle" | "running" | "done" | "error" | "cancelled";

export function FolderRunRunStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId, run, isLoading } = useFolderRun();

  const [stage, setStage] = useState<RunStage>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  // Pre-flight readiness check. Same query the QueueCard uses; we
  // fetch on demand inside the start handler rather than mount so
  // we don't pay the network cost on every step revisit.
  const startMutation = useMutation({
    mutationFn: async () => {
      if (!runId) throw new Error("missing run id");
      const readiness = await queryClient.fetchQuery({
        queryKey: ["project-model-readiness", runId],
        queryFn: () => projectsApi.getModelReadiness(runId),
        staleTime: 0,
      });
      if (!readiness.ready) {
        const names = readiness.missing.map((m) => m.friendly_name).join(", ");
        throw new Error(
          `Some models need setup before you can run: ${names}. Open the project settings to install them.`,
        );
      }
      return deploymentQueueApi.process({ project_id: runId });
    },
    onSuccess: (resp) => {
      if (resp.jobs_started === 0 || resp.job_ids.length === 0) {
        // No pending queue entries. The run was already processed —
        // skip ahead to review without spinning up another job.
        setStage("done");
        return;
      }
      setJobId(resp.job_ids[0]);
      setStage("running");
    },
    onError: (err) => {
      setErrorMessage(
        err instanceof Error ? err.message : "unknown error",
      );
      setStage("error");
    },
  });

  const progress = useTaskProgress({
    taskId: jobId,
    onComplete: async () => {
      setStage("done");
      if (runId) {
        const next = await folderRunsApi.updateStep(runId, "review");
        queryClient.setQueryData(["folder-run", runId], next);
      }
    },
    onError: (msg) => {
      setErrorMessage(msg);
      setStage("error");
    },
    onCancelled: (msg) => {
      setErrorMessage(msg || "Analysis cancelled");
      setStage("cancelled");
      setIsCancelling(false);
      setJobId(null);
    },
  });

  // If the run is already completed (re-entry after success), skip
  // ahead automatically — the user does not need to re-trigger.
  useEffect(() => {
    if (!run) return;
    if (run.queue_entry?.status === "completed") {
      setStage("done");
    }
  }, [run]);

  if (!runId) {
    navigate("/folder-runs/new", { replace: true });
    return null;
  }

  if (isLoading || !run) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-sm text-muted-foreground">
          Loading run...
        </CardContent>
      </Card>
    );
  }

  const queueFailed = run.queue_entry?.status === "failed";

  return (
    <Card>
      <CardHeader>
        <CardTitle>Run analysis</CardTitle>
        <CardDescription>
          AddaxAI will scan the folder, run the selected models, and
          write results to the output folder.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {queueFailed && stage === "idle" && (
          <div className="flex items-start gap-3 rounded-md border border-destructive/30 bg-destructive/5 p-4">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
            <div className="text-sm">
              <p className="font-medium">Previous run failed</p>
              <p className="mt-1 text-muted-foreground">
                {run.queue_entry?.error ?? "Unknown error"}
              </p>
            </div>
          </div>
        )}

        {stage === "idle" && (
          <p className="text-sm text-muted-foreground">
            Ready to start. Use the button below to kick off detection
            (and species identification when configured). You can come
            back to this step if you need to change the model.
          </p>
        )}

        {stage === "running" && (
          <AnalysisProgress
            phase={progress.phase}
            phaseProgress={progress.phaseProgress}
            metrics={progress.metrics}
            computeDevice={progress.computeDevice}
            deploymentContext={progress.deploymentContext}
            message={progress.message}
            hideDeploymentHeader
          />
        )}

        {stage === "done" && (
          <div className="rounded-md border border-primary/30 bg-primary/5 p-4 text-sm">
            <p className="font-medium text-primary">Analysis complete</p>
            <p className="mt-1 text-muted-foreground">
              Continue to review the results, then save the outputs.
            </p>
          </div>
        )}

        {stage === "error" && (
          <div className="flex items-start gap-3 rounded-md border border-destructive/30 bg-destructive/5 p-4">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
            <div className="text-sm">
              <p className="font-medium">Analysis failed</p>
              <p className="mt-1 text-muted-foreground">
                {errorMessage ?? "unknown error"}
              </p>
            </div>
          </div>
        )}

        {stage === "cancelled" && (
          <div className="rounded-md border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            <p className="font-medium">Analysis cancelled</p>
            <p className="mt-1 text-amber-900/80">
              {errorMessage ?? "Run was cancelled."}
            </p>
          </div>
        )}
      </CardContent>

      <CardFooter className="justify-between">
        <Button
          variant="outline"
          onClick={() => navigate(`/folder-runs/${runId}/model`)}
          className="gap-2"
          disabled={stage === "running"}
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </Button>

        {stage === "idle" || stage === "error" || stage === "cancelled" ? (
          <Button
            onClick={() => startMutation.mutate()}
            disabled={startMutation.isPending}
            className="gap-2"
            size="lg"
          >
            <Play className="h-4 w-4" />
            {stage === "idle" ? "Start analysis" : "Try again"}
          </Button>
        ) : stage === "running" ? (
          <Button
            variant="outline"
            onClick={() => {
              setIsCancelling(true);
              progress.cancel();
            }}
            disabled={isCancelling}
            className="gap-2"
          >
            <X className="h-4 w-4" />
            {isCancelling ? "Cancelling..." : "Cancel"}
          </Button>
        ) : (
          <Button
            onClick={() => navigate(`/folder-runs/${runId}/review`)}
            className="gap-2"
            size="lg"
          >
            Continue
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}
