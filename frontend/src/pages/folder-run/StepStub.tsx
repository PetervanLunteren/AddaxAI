/**
 * Shared stub used by steps that have not been implemented yet.
 *
 * Each stubbed step (model, run, review, save) renders this with its
 * own title and description, plus Back / Continue buttons that move
 * along the URL and persist the new step to the backend via
 * folderRunsApi.updateStep. The real implementations land in later
 * slices and replace these stubs one at a time.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { Button } from "../../components/ui/button";
import { Callout } from "../../components/ui/callout";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../../components/ui/card";
import {
  folderRunsApi,
  type FolderRunStep,
} from "../../api/folder-runs";
import { useFolderRun } from "./FolderRunLayout";

interface StepStubProps {
  title: string;
  description: string;
  /** Short copy explaining what the real step will do. */
  comingNext: string;
  /** The folder-run step this stub represents. Used to persist
   * progression server-side. */
  thisStep: FolderRunStep;
  /** URL slug of the previous step. Null on the first non-folder step
   * means "no back button" (rarely useful; folder step has its own
   * component). */
  backTo: string | null;
  /** URL slug of the next step, or null on the last step. */
  nextTo: string | null;
}

export function StepStub({
  title,
  description,
  comingNext,
  thisStep,
  backTo,
  nextTo,
}: StepStubProps) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId } = useFolderRun();

  const updateStep = useMutation({
    mutationFn: (step: FolderRunStep) =>
      folderRunsApi.updateStep(runId!, step),
    onSuccess: (run) => {
      queryClient.setQueryData(["folder-run", run.project.id], run);
    },
  });

  const handleNavigate = (slug: string, persistedStep: FolderRunStep) => {
    if (!runId) {
      // No run id means the user reached this URL without finishing
      // step 1. Bounce them to the home; folder-run state lives on
      // the project row, which doesn't exist yet.
      navigate("/");
      return;
    }
    // Fire-and-forget the persistence. If it fails the user still
    // moves forward; the URL is the source of truth for the in-flight
    // session, and step state is reconciled on the next mount.
    updateStep.mutate(persistedStep);
    navigate(`/folder-runs/${runId}/${slug}`);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <Callout variant="warning" title="Step under construction">
          {comingNext}
        </Callout>
      </CardContent>

      <CardFooter className="justify-between">
        {backTo ? (
          <Button
            variant="outline"
            onClick={() => handleNavigate(backTo, thisStep)}
            className="gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
        ) : (
          <span />
        )}
        {nextTo ? (
          <Button
            onClick={() => {
              // Persist the NEXT step so resume sends users forward
              // rather than back to this stub.
              const next = nextTo as FolderRunStep;
              handleNavigate(nextTo, next);
            }}
            className="gap-2"
          >
            Continue
            <ArrowRight className="h-4 w-4" />
          </Button>
        ) : (
          <Button onClick={() => navigate("/")} className="gap-2">
            Finish
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}
