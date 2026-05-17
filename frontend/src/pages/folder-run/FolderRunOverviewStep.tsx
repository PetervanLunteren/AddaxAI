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
    <div className="space-y-6">
      <DashboardView projectId={runId} />

      <div className="flex items-center justify-between">
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
  );
}
