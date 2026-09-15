/**
 * Counts step (slug `counts`, optional).
 *
 * The second of the two review steps, the same event gallery as the
 * Counts page of a project: one card per event, the AI's count per
 * species, confirm or change it. `VerifyView` owns the gallery, its
 * filters and the event detail modal; this file only adds the step
 * chrome. Back returns to Labels; Continue PATCHes `step=save` and
 * navigates on.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowRight } from "lucide-react";

import { Button } from "../../components/ui/button";
import { RunGate } from "../../components/folder-run/RunGate";
import { StepActionBar } from "../../components/folder-run/StepActionBar";
import { StepHeader } from "../../components/folder-run/StepHeader";
import { VerifyView } from "../../components/verify/VerifyView";
import { folderRunsApi } from "../../api/folder-runs";
import { useFolderRun } from "./FolderRunLayout";

export function FolderRunCountsStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId } = useFolderRun();

  const advance = useMutation({
    mutationFn: () => folderRunsApi.updateStep(runId!, "save"),
    onSuccess: (next) => {
      queryClient.setQueryData(["folder-run", runId], next);
      navigate(`/folder-runs/${runId}/save`);
    },
  });

  return (
    <RunGate>
      {(_run, runId) => (
        <div className="space-y-6 pb-24">
          <StepHeader
            title="Check counts"
            caption="This step is optional. Confirm the AI's counts, adjust any that are wrong, or go straight to saving."
          />
          <VerifyView projectId={runId} />
          <StepActionBar>
            <Button
              variant="outline"
              onClick={() => navigate(`/folder-runs/${runId}/labels`)}
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
        </div>
      )}
    </RunGate>
  );
}
