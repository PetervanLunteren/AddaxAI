/**
 * Labels step (slug `labels`, optional).
 *
 * Per-detection label cleanup via the crop grid. The review-vs-skip
 * choice now lives on the completion modal (see FolderRunModelStep), so
 * this step is just the grid: you land here only if you chose to review.
 * Back returns to setup; Continue PATCHes `step=save` and navigates on.
 */

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowRight } from "lucide-react";

import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import { AnalysisSettingsButton } from "../../components/folder-run/AnalysisSettingsButton";
import { StepActionBar } from "../../components/folder-run/StepActionBar";
import { StepHeader } from "../../components/folder-run/StepHeader";
import { LabelsView } from "../../components/verify/LabelsView";
import { folderRunsApi } from "../../api/folder-runs";
import { useFolderRun } from "./FolderRunLayout";

export function FolderRunLabelsStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId, run, isLoading } = useFolderRun();
  // Bumped when the analysis panel finishes an apply-and-reprocess, so
  // the grid re-runs its sort onto the new labels (it renders from a
  // mutation, which query invalidation cannot refresh).
  const [reprocessNonce, setReprocessNonce] = useState(0);
  // Track bulk-selection size from the embedded LabelsView. While a
  // selection is live, the sticky Back / Continue bar is hidden so the
  // floating BulkActionBar has the bottom of the viewport to itself.
  const [selectionCount, setSelectionCount] = useState(0);

  const advance = useMutation({
    mutationFn: () => folderRunsApi.updateStep(runId!, "save"),
    onSuccess: (next) => {
      queryClient.setQueryData(["folder-run", runId], next);
      navigate(`/folder-runs/${runId}/save`);
    },
  });

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

  // The grid. Back returns to setup; Continue advances to Save.
  return (
    <div className="space-y-6 pb-24">
      <StepHeader
        title="Check labels"
        caption="This step is optional. Fix any labels the AI got wrong, or go straight to saving."
      />
      <LabelsView
        projectId={runId}
        onSelectionChange={setSelectionCount}
        // No explicit default floor: like projects mode, the grid rests
        // at the project's counting_threshold (the backend applies it as
        // the threshold-or-verified floor), so the grid, counts, and
        // verification pill all measure the same population.
        refreshSignal={reprocessNonce}
        toolbarExtra={
          <AnalysisSettingsButton
            runId={runId}
            project={run.project}
            onApplied={() => setReprocessNonce((n) => n + 1)}
          />
        }
      />
      {selectionCount === 0 && (
        <StepActionBar>
          <Button
            variant="outline"
            onClick={() => navigate(`/folder-runs/${runId}/setup`)}
            className="gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
          <Button
            onClick={() => advance.mutate()}
            disabled={advance.isPending}
            size="lg"
            className="gap-2"
          >
            Continue
            <ArrowRight className="h-4 w-4" />
          </Button>
        </StepActionBar>
      )}
    </div>
  );
}
