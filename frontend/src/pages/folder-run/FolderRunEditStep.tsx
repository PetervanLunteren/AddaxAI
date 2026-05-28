/**
 * Edit step (slug `edit`, optional).
 *
 * Editing is optional, so the heavy editor grid is collapsed behind a
 * toggle. The default page is lightweight — title, caption, an "Open
 * editor" button, and the Back / Continue bar — so the obvious path is
 * to skip ahead. This was a direct response to user feedback that
 * landing on a wall of grid made the step feel required.
 *
 * Only when the user opens the editor do we mount `VerifyView` (the
 * same grid used by the research-projects Edit page), which also means
 * its data isn't fetched until opted in.
 *
 * Layout above us is provided by `FolderRunLayout`, which switches to
 * `max-w-7xl` for this step so the grid breakpoints match the
 * research-projects verify exactly. The Back / Continue bar at the
 * bottom is `sticky` so the user can advance from anywhere in the
 * (potentially long) scrollable body once the editor is open.
 *
 * Continue PATCHes `step=overview` server-side and navigates onward.
 */

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowRight, ChevronDown, Pencil } from "lucide-react";

import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import { StepHeader } from "../../components/folder-run/StepHeader";
import { VerifyView } from "../../components/verify/VerifyView";
import { folderRunsApi } from "../../api/folder-runs";
import { useFolderRun } from "./FolderRunLayout";

export function FolderRunEditStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId, run, isLoading } = useFolderRun();
  const [editorOpen, setEditorOpen] = useState(false);
  // Track bulk-selection size from the embedded VerifyView. While a
  // selection is live, the sticky Back / Continue bar is hidden so the
  // floating BulkActionBar has the bottom of the viewport to itself
  // (no visual overlap, no accidental Continue mid-action). The bar
  // returns when the user clears the selection.
  const [selectionCount, setSelectionCount] = useState(0);

  const advance = useMutation({
    mutationFn: () => folderRunsApi.updateStep(runId!, "overview"),
    onSuccess: (next) => {
      queryClient.setQueryData(["folder-run", runId], next);
      navigate(`/folder-runs/${runId}/overview`);
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

  // Back / editor toggle / Continue, one row, middle centred.
  const actionRow = (
    <div className="grid grid-cols-3 items-center gap-3">
      <Button
        variant="outline"
        onClick={() => navigate(`/folder-runs/${runId}/model`)}
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
            Close editor
          </>
        ) : (
          <>
            <Pencil className="h-4 w-4" />
            Open editor
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

  // Collapsed: a lean carded decision point ("open the editor, or move
  // on"). No grid mounted, no data fetched. It fills the same width as
  // the open grid below, so opening the editor inserts the grid without
  // the page width changing.
  if (!editorOpen) {
    return (
      <div className="space-y-6">
        <StepHeader
          title="Edit predictions"
          caption="Correct the AI's predictions if you want, or skip ahead."
        />
        <Card>
          <CardContent className="py-4">{actionRow}</CardContent>
        </Card>
      </div>
    );
  }

  // Open: full-width grid with the action row pinned to the bottom so
  // the user can advance from anywhere in the long scrollable body.
  return (
    <div className="space-y-6 pb-24">
      <StepHeader
        title="Edit predictions"
        caption="Correct the AI's predictions if you want, or skip ahead."
      />
      <VerifyView projectId={runId} onSelectionChange={setSelectionCount} />
      {selectionCount === 0 && (
        <div className="sticky bottom-0 z-30 -mx-4 border-t bg-white/80 px-4 py-3 backdrop-blur-sm sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8">
          <div className="mx-auto max-w-7xl">{actionRow}</div>
        </div>
      )}
    </div>
  );
}
