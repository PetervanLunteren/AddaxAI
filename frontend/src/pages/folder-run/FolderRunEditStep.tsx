/**
 * Step 4: Review results.
 *
 * Embeds the full Verify UI inline via the shared `VerifyView`
 * component, so users get the same three tabs (Observations / Media /
 * Events), filters, sort modes, modals, and similarity sort they have
 * in research projects — without the Research-projects sidebar
 * appearing and confusing them.
 *
 * Layout above us is provided by `FolderRunLayout`, which switches to
 * `max-w-7xl` for this step so the grid breakpoints match the
 * research-projects verify exactly. The Back / Continue bar at the
 * bottom is `sticky` so the user can advance the step from anywhere
 * in the (potentially long) scrollable verify body.
 *
 * Continue PATCHes `step=save` server-side and navigates to /save.
 * The "Skip review" affordance from earlier slices is gone — with
 * inline verify the user is already looking at the review canvas;
 * Continue serves both "I'm done reviewing" and "I'm skipping" intents.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowRight } from "lucide-react";

import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import { StepHeader } from "../../components/folder-run/StepHeader";
import { VerifyView } from "../../components/verify/VerifyView";
import { folderRunsApi } from "../../api/folder-runs";
import { useFolderRun } from "./FolderRunLayout";

export function FolderRunReviewStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId, run, isLoading } = useFolderRun();

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

  return (
    <div className="space-y-6 pb-24">
      <StepHeader
        title="Verify predictions"
        caption="Check the AI's predictions and correct anything wrong. Optional, but it improves your data."
      />
      {/* Bottom padding clears the sticky action bar so the last row of
          the verify grid (or pagination strip) is fully visible above
          it on short pages. */}
      <VerifyView projectId={runId} />

      <div className="sticky bottom-0 z-30 -mx-4 border-t bg-white/80 px-4 py-3 backdrop-blur-sm sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-3">
          <Button
            variant="outline"
            onClick={() => navigate(`/folder-runs/${runId}/model`)}
            className="gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
          <Button
            onClick={() => advance.mutate()}
            disabled={advance.isPending}
            className="gap-2"
            size="lg"
          >
            Continue
            <ArrowRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
