/**
 * Step 4 of the folder-run stepper: Overview.
 *
 * Shows the same Dashboard the research-projects flow does (summary
 * counts, top taxa, detection trend, activity, alerts, verification
 * progress) — scoped to this folder run's project id. Embedded
 * inline so the user stays inside the stepper rather than jumping
 * to ``/projects/<id>/dashboard``.
 *
 * Continue → persist step="review" and navigate to the Review step.
 * Back → /run.
 */

import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";

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

  return (
    <div className="space-y-6 pb-24">
      {/* Bottom padding clears the sticky action bar so the last chart
          on the dashboard is fully visible above it. */}
      <StepHeader
        title="Summary"
        caption="Top species, detection counts, and activity over time."
      />
      <DashboardView projectId={runId} />

      <div className="sticky bottom-0 z-30 -mx-4 border-t bg-white/80 px-4 py-3 backdrop-blur-sm sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-3">
          <Button
            variant="outline"
            onClick={() => navigate(`/folder-runs/${runId}/review`)}
            className="gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
          <Button onClick={handleContinue} className="gap-2" size="lg">
            Continue
          </Button>
        </div>
      </div>
    </div>
  );
}
