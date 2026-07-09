/**
 * Labels step (slug `labels`, optional).
 *
 * Per-detection label cleanup via the crop grid. Editing is optional,
 * so the heavy grid is collapsed behind a "Show editor" toggle and the
 * default page is lightweight so the obvious path is to continue
 * (direct response to feedback that a wall of grid made the step feel
 * required).
 *
 * Continue PATCHes `step=save` server-side and navigates onward.
 */

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowRight, ChevronDown, Pencil } from "lucide-react";

import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import { AnalysisSettingsButton } from "../../components/folder-run/AnalysisSettingsButton";
import { StepHeader } from "../../components/folder-run/StepHeader";
import { LabelsView } from "../../components/verify/LabelsView";
import { folderRunsApi } from "../../api/folder-runs";
import { useFolderRun } from "./FolderRunLayout";

export function FolderRunLabelsStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId, run, isLoading } = useFolderRun();
  const [editorOpen, setEditorOpen] = useState(false);
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

  const actionRow = (
    <div className="grid grid-cols-3 items-center gap-3">
      <Button
        variant="outline"
        onClick={() => navigate(`/folder-runs/${runId}/setup`)}
        className="justify-self-start gap-2"
      >
        <ArrowLeft className="h-4 w-4" />
        Back
      </Button>
      <Button
        variant="outline"
        onClick={() => setEditorOpen((open) => !open)}
        className="justify-self-center gap-2"
      >
        {editorOpen ? (
          <>
            <ChevronDown className="h-4 w-4 rotate-180" />
            Hide editor
          </>
        ) : (
          <>
            <Pencil className="h-4 w-4" />
            Show editor
          </>
        )}
      </Button>
      <Button
        onClick={() => advance.mutate()}
        disabled={advance.isPending}
        className="justify-self-end gap-2"
        size="lg"
      >
        Continue
        <ArrowRight className="h-4 w-4" />
      </Button>
    </div>
  );

  return (
    <div className={editorOpen ? "space-y-6 pb-24" : "space-y-6"}>
      <StepHeader
        title="Check labels"
        caption="Fix any labels the AI got wrong, or continue."
      />
      {editorOpen ? (
        <>
          <LabelsView
            projectId={runId}
            onSelectionChange={setSelectionCount}
            // No explicit default floor: like projects mode, the grid
            // rests at the project's detection_threshold (the backend
            // applies it as the threshold-or-verified floor), so the
            // grid, counts, and verification pill all measure the same
            // population.
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
            <div className="sticky bottom-0 z-30 -mx-4 border-t bg-white/80 px-4 py-3 backdrop-blur-sm sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8">
              <div className="mx-auto max-w-7xl">{actionRow}</div>
            </div>
          )}
        </>
      ) : (
        <Card>
          <CardContent className="py-4">{actionRow}</CardContent>
        </Card>
      )}
    </div>
  );
}
