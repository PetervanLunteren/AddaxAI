/**
 * Labels step (slug `labels`, optional).
 *
 * Per-detection label cleanup via the crop grid. Editing is optional, so
 * the step opens on an explicit two-way choice rather than the heavy
 * grid: "Review the labels" (opens the grid) or "Skip to saving"
 * (advances to Save). This replaced a lone "Show editor" toggle that
 * left the page looking empty and hid the skip path in prose.
 *
 * Continue PATCHes `step=save` server-side and navigates onward.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowRight, Tag } from "lucide-react";

import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import { NextStepRow } from "../../components/ui/next-step-row";
import { AnalysisSettingsButton } from "../../components/folder-run/AnalysisSettingsButton";
import { StepHeader } from "../../components/folder-run/StepHeader";
import { LabelsView } from "../../components/verify/LabelsView";
import { folderRunsApi } from "../../api/folder-runs";
import { useFolderRun } from "./FolderRunLayout";

export function FolderRunLabelsStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId, run, isLoading } = useFolderRun();
  // false: the two-way choice (review vs skip). true: the crop grid.
  const [reviewing, setReviewing] = useState(false);
  // Bumped when the analysis panel finishes an apply-and-reprocess, so
  // the grid re-runs its sort onto the new labels (it renders from a
  // mutation, which query invalidation cannot refresh).
  const [reprocessNonce, setReprocessNonce] = useState(0);
  // Track bulk-selection size from the embedded LabelsView. While a
  // selection is live, the sticky Back / Continue bar is hidden so the
  // floating BulkActionBar has the bottom of the viewport to itself.
  const [selectionCount, setSelectionCount] = useState(0);

  // Run summary for the "Review the labels" row: how much there is to
  // review. The lookup endpoint (keyed by source folder) carries the
  // counts; the step's own `run` object does not. Degrades to a generic
  // description when the folder or the counts aren't available.
  const folderPath = run?.queue_entry?.folder_path;
  const { data: summary } = useQuery({
    queryKey: ["folder-run-summary", folderPath],
    queryFn: () => folderRunsApi.lookup(folderPath!),
    enabled: !!folderPath,
    staleTime: 30_000,
  });

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

  const reviewDescription =
    summary && summary.detection_count > 0
      ? `Check and edit ${summary.detection_count.toLocaleString()} ` +
        `observation${summary.detection_count === 1 ? "" : "s"}` +
        (summary.species_count > 0
          ? ` across ${summary.species_count.toLocaleString()} species`
          : "") +
        ", in a grid."
      : "Check and edit the species AddaxAI assigned, in a grid.";

  // The two-way choice, shown until the user opens the grid.
  if (!reviewing) {
    return (
      <div className="space-y-6">
        <StepHeader
          title="Check labels"
          caption="Review the AI's suggested labels, or skip straight to saving."
        />
        {/* Two side-by-side cards for the either/or choice: this fills the
            width (a single full-width row would be mostly empty) and reads
            as a fork. Stacks on narrow screens. */}
        <div className="grid gap-4 sm:grid-cols-2">
          <NextStepRow
            icon={Tag}
            title="Review the labels"
            description={reviewDescription}
            onClick={() => setReviewing(true)}
            className="bg-card shadow-sm"
          />
          <NextStepRow
            icon={ArrowRight}
            title="Skip to saving"
            description="Keep the AI labels as they are and go to the save step."
            onClick={() => advance.mutate()}
            disabled={advance.isPending}
            className="bg-card shadow-sm"
          />
        </div>
        <div>
          <Button
            variant="outline"
            onClick={() => navigate(`/folder-runs/${runId}/setup`)}
            className="gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
        </div>
      </div>
    );
  }

  // The grid. Back returns to the choice above (one step out); Continue
  // advances to Save.
  return (
    <div className="space-y-6 pb-24">
      <StepHeader
        title="Check labels"
        caption="Fix any labels the AI got wrong, then continue."
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
        <div className="sticky bottom-0 z-30 -mx-4 border-t bg-white/80 px-4 py-3 backdrop-blur-sm sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8">
          <div className="mx-auto flex max-w-7xl items-center justify-between gap-3">
            <Button
              variant="outline"
              onClick={() => setReviewing(false)}
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
          </div>
        </div>
      )}
    </div>
  );
}
