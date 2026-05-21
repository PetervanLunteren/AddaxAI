/**
 * Summary step (slug `overview`, optional).
 *
 * Seeing the summary is optional, so the dashboard is collapsed behind
 * a toggle — same pattern as the Edit step. The default page is
 * lightweight (title, caption, a "View summary" button, and the Back /
 * Continue bar) so the obvious path is to move on. This was the same
 * user feedback that drove the Edit step: landing on a wall of content
 * made the step feel required when it is not.
 *
 * Only when the user opens the summary do we mount `DashboardView`
 * (the same dashboard the research-projects flow shows, scoped to this
 * run's project id), which also means its stats aren't fetched until
 * opted in.
 *
 * Continue → persist step="save" and navigate to the Output step.
 * Back → the Edit step.
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, ArrowRight, BarChart3, ChevronDown } from "lucide-react";

import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import { DashboardView } from "../../components/dashboard/DashboardView";
import { StepHeader } from "../../components/folder-run/StepHeader";
import { folderRunsApi } from "../../api/folder-runs";
import { useFolderRun } from "./FolderRunLayout";

export function FolderRunOverviewStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId, run, isLoading } = useFolderRun();
  const [summaryOpen, setSummaryOpen] = useState(false);

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

  const handleContinue = async () => {
    const next = await folderRunsApi.updateStep(runId, "save");
    queryClient.setQueryData(["folder-run", runId], next);
    navigate(`/folder-runs/${runId}/save`);
  };

  // Back / summary toggle / Continue, one row, middle centred.
  const actionRow = (
    <div className="grid grid-cols-3 items-center gap-3">
      <Button
        variant="outline"
        onClick={() => navigate(`/folder-runs/${runId}/edit`)}
        className="justify-self-start gap-2"
      >
        <ArrowLeft className="h-4 w-4" />
        Back
      </Button>
      <Button
        variant="outline"
        onClick={() => setSummaryOpen((open) => !open)}
        className="justify-self-center gap-2"
      >
        {summaryOpen ? (
          <>
            <ChevronDown className="h-4 w-4 rotate-180" />
            Hide summary
          </>
        ) : (
          <>
            <BarChart3 className="h-4 w-4" />
            View summary
          </>
        )}
      </Button>
      <Button
        onClick={handleContinue}
        className="justify-self-end gap-2"
        size="lg"
      >
        Continue
        <ArrowRight className="h-4 w-4" />
      </Button>
    </div>
  );

  // Collapsed: a lean carded decision point ("see the summary, or move
  // on"). No dashboard mounted, no stats fetched. It fills the same
  // width as the open dashboard below, so opening the summary inserts
  // the dashboard without the page width changing.
  if (!summaryOpen) {
    return (
      <div className="space-y-6">
        <StepHeader
          title="Summary"
          caption="See top species, counts, and activity, or skip ahead."
        />
        <Card>
          <CardContent className="py-4">{actionRow}</CardContent>
        </Card>
      </div>
    );
  }

  // Open: full dashboard with the action row pinned to the bottom so
  // the user can advance from anywhere in the long scrollable body.
  return (
    <div className="space-y-6 pb-24">
      {/* Bottom padding clears the sticky action bar so the last chart
          on the dashboard is fully visible above it. */}
      <StepHeader
        title="Summary"
        caption="See top species, counts, and activity, or skip ahead."
      />
      <DashboardView projectId={runId} />
      <div className="sticky bottom-0 z-30 -mx-4 border-t bg-white/80 px-4 py-3 backdrop-blur-sm sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8">
        <div className="mx-auto max-w-7xl">{actionRow}</div>
      </div>
    </div>
  );
}
