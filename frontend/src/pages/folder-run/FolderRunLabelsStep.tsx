/**
 * Labels step (slug `labels`, optional).
 *
 * The first of the two review steps: per-detection label cleanup via the
 * crop grid. The review-vs-skip choice lives on the completion modal (see
 * FolderRunModelStep), so this step is just the grid: you land here only
 * if you chose to review. Back returns to setup; Continue PATCHes
 * `step=counts` and navigates on to the Counts step.
 */

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowRight } from "lucide-react";

import { Button } from "../../components/ui/button";
import { AnalysisSettingsButton } from "../../components/folder-run/AnalysisSettingsButton";
import { RunGate } from "../../components/folder-run/RunGate";
import { StepActionBar } from "../../components/folder-run/StepActionBar";
import { StepHeader } from "../../components/folder-run/StepHeader";
import { LabelsView } from "../../components/verify/LabelsView";
import { folderRunsApi } from "../../api/folder-runs";
import { useFolderRun } from "./FolderRunLayout";

export function FolderRunLabelsStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId } = useFolderRun();
  // Bumped when the analysis panel finishes an apply-and-reprocess, so
  // the grid re-runs its sort onto the new labels (it renders from a
  // mutation, which query invalidation cannot refresh).
  const [reprocessNonce, setReprocessNonce] = useState(0);
  // Track bulk-selection size from the embedded LabelsView. While a
  // selection is live, the sticky Back / Continue bar is hidden so the
  // floating BulkActionBar has the bottom of the viewport to itself.
  const [selectionCount, setSelectionCount] = useState(0);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const advance = useMutation({
    mutationFn: () => folderRunsApi.updateStep(runId!, "counts"),
    onSuccess: (next) => {
      queryClient.setQueryData(["folder-run", runId], next);
      navigate(`/folder-runs/${runId}/counts`);
    },
  });

  // The grid. Back returns to setup; Continue advances to Counts.
  return (
    <RunGate>
      {(run, runId) => (
        <div className="space-y-6 pb-24">
          <StepHeader
            title="Check labels"
            caption="This step is optional. Fix any labels the AI got wrong, or continue to the counts."
          />
          <LabelsView
            projectId={runId}
            onSelectionChange={setSelectionCount}
            // No explicit default floor: like projects mode, the grid rests
            // at the project's counting_threshold (the backend applies it as
            // the threshold-or-verified floor), so the grid, counts, and
            // verification pill all measure the same population.
            refreshSignal={reprocessNonce}
            // The Files tab note names the threshold and offers to open it.
            // Here that is the Refine results slideout, so this step holds
            // its open state and hands the grid a way in.
            onEditThreshold={() => setSettingsOpen(true)}
            toolbarExtra={
              <AnalysisSettingsButton
                runId={runId}
                project={run.project}
                onApplied={() => setReprocessNonce((n) => n + 1)}
                open={settingsOpen}
                onOpenChange={setSettingsOpen}
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
                // The stored step is the furthest one reached, so a run
                // that already got to Save keeps it: moving on from here
                // must not send the next resume back to Counts.
                onClick={() =>
                  run.step === "save"
                    ? navigate(`/folder-runs/${runId}/counts`)
                    : advance.mutate()
                }
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
      )}
    </RunGate>
  );
}
