/**
 * Counts step (slug `counts`, optional).
 *
 * Second of the two verification steps. The ecological record: review
 * each event's media in the gallery and confirm the species and counts
 * (the data model and exports keep the standard term "observation").
 * Optional, so the gallery is collapsed behind a "Show" toggle and the
 * default page is lightweight so the obvious path is to continue.
 *
 * Continue PATCHes `step=overview` server-side and navigates onward.
 */

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowRight, ChevronDown, Layers } from "lucide-react";

import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import { StepHeader } from "../../components/folder-run/StepHeader";
import { VerifyView } from "../../components/verify/VerifyView";
import { folderRunsApi } from "../../api/folder-runs";
import { useFolderRun } from "./FolderRunLayout";

export function FolderRunCountsStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId, run, isLoading } = useFolderRun();
  const [galleryOpen, setGalleryOpen] = useState(false);

  const advance = useMutation({
    mutationFn: () => folderRunsApi.updateStep(runId!, "summary"),
    onSuccess: (next) => {
      queryClient.setQueryData(["folder-run", runId], next);
      navigate(`/folder-runs/${runId}/summary`);
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
        onClick={() => navigate(`/folder-runs/${runId}/labels`)}
        className="justify-self-start gap-2"
      >
        <ArrowLeft className="h-4 w-4" />
        Back
      </Button>
      <Button
        variant="outline"
        onClick={() => setGalleryOpen((open) => !open)}
        className="justify-self-center gap-2"
      >
        {galleryOpen ? (
          <>
            <ChevronDown className="h-4 w-4 rotate-180" />
            Hide gallery
          </>
        ) : (
          <>
            <Layers className="h-4 w-4" />
            Show gallery
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

  if (!galleryOpen) {
    return (
      <div className="space-y-6">
        <StepHeader
          title="Confirm counts"
          caption="Confirm the counts, or continue."
        />
        <Card>
          <CardContent className="py-4">{actionRow}</CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-24">
      <StepHeader
        title="Confirm counts"
        caption="Confirm the counts, or continue."
      />
      <VerifyView projectId={runId} />
      <div className="sticky bottom-0 z-30 -mx-4 border-t bg-white/80 px-4 py-3 backdrop-blur-sm sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8">
        <div className="mx-auto max-w-7xl">{actionRow}</div>
      </div>
    </div>
  );
}
