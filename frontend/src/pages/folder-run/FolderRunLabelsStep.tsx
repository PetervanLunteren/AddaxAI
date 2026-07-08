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

import { useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowLeft, ArrowRight, ChevronDown, Pencil } from "lucide-react";

import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import { AnalysisSettingsButton } from "../../components/folder-run/AnalysisSettingsButton";
import { StepHeader } from "../../components/folder-run/StepHeader";
import { LabelsView } from "../../components/verify/LabelsView";
import { folderRunsApi } from "../../api/folder-runs";
import { DEFAULT_COUNTING_THRESHOLD } from "../../lib/confidence";
import { useFolderRun } from "./FolderRunLayout";

/** Default floor for the grid's detection-confidence filter. Folder
 * runs store every detection down to the 0.1 inference floor; showing
 * all of it by default buries the reviewer in near-noise boxes, so the
 * grid opens pre-filtered at 0.2. It is an ordinary filter: the user
 * can drop it to the floor (or clear it) in the filter bar. Affects
 * this review grid only — data exports always contain everything. */
const DEFAULT_GRID_MIN_CONFIDENCE = DEFAULT_COUNTING_THRESHOLD;

export function FolderRunLabelsStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId, run, isLoading } = useFolderRun();
  const [editorOpen, setEditorOpen] = useState(false);
  const [searchParams, setSearchParams] = useSearchParams();

  // Seed the grid's confidence filter once per mount, and only when
  // the URL doesn't already carry one, so a user-cleared or user-set
  // filter is never overridden within the session.
  const seededMinConfidenceRef = useRef(false);
  useEffect(() => {
    if (seededMinConfidenceRef.current) return;
    seededMinConfidenceRef.current = true;
    if (!searchParams.has("lbl_min_confidence")) {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set(
            "lbl_min_confidence",
            String(DEFAULT_GRID_MIN_CONFIDENCE),
          );
          return next;
        },
        { replace: true },
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
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
            toolbarExtra={
              <AnalysisSettingsButton
                runId={runId}
                project={run.project}
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
