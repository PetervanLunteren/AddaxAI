/**
 * Step 2: Choose AI models.
 *
 * Lean version of the model picker that powers the Timelapse window
 * and the project create dialog. Surfaces the two decisions every
 * folder run needs to make:
 *
 * - Detection model (which finds animals / people / vehicles)
 * - Classification model (which identifies species; optional)
 *
 * Saves directly onto the underlying project row via the standard
 * `PATCH /api/projects/{id}` endpoint, which already accepts these
 * fields. The folder-run step state is bumped to "run" so resume
 * lands the user on step 3.
 *
 * Country / state geofence and per-class exclusions are deliberately
 * NOT exposed here. They live behind the Research projects' Settings
 * panel; folder-run users who need them can promote and configure
 * later. Keeps the stepper short for the legacy-AddaxAI use case.
 */

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowRight } from "lucide-react";

import { Button } from "../../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../../components/ui/card";
import { Label } from "../../components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../components/ui/select";

import { modelsApi } from "../../api/models";
import { projectsApi } from "../../api/projects";
import { folderRunsApi } from "../../api/folder-runs";
import { useFolderRun } from "./FolderRunLayout";

const NO_CLASSIFIER = "none";

export function FolderRunModelStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId, run, isLoading } = useFolderRun();

  const { data: detectionModels = [] } = useQuery({
    queryKey: ["models", "detection"],
    queryFn: modelsApi.listDetectionModels,
  });

  const { data: classificationModels = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: modelsApi.listClassificationModels,
  });

  // Seed the form from the project row, so resume on this step shows
  // the user's prior selection instead of bouncing to defaults.
  const [detectionModelId, setDetectionModelId] = useState("MD5A-0-0");
  const [classificationModelId, setClassificationModelId] =
    useState<string>(NO_CLASSIFIER);

  useEffect(() => {
    if (!run) return;
    setDetectionModelId(run.project.detection_model_id);
    setClassificationModelId(
      run.project.classification_model_id ?? NO_CLASSIFIER,
    );
  }, [run]);

  const save = useMutation({
    mutationFn: async () => {
      if (!runId) throw new Error("missing run id");
      // Update project model fields.
      const updated = await projectsApi.update(runId, {
        detection_model_id: detectionModelId,
        classification_model_id:
          classificationModelId === NO_CLASSIFIER
            ? null
            : classificationModelId,
      });
      // Persist step progression separately so resume works.
      const next = await folderRunsApi.updateStep(runId, "run");
      return { updated, next };
    },
    onSuccess: ({ next }) => {
      queryClient.setQueryData(["folder-run", runId], next);
      queryClient.invalidateQueries({ queryKey: ["projects", runId] });
      navigate(`/folder-runs/${runId}/run`);
    },
  });

  if (!runId) {
    // User landed on /folder-runs/new/model directly. Step 1 creates
    // the run id; without it there is nothing to PATCH.
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
    <Card>
      <CardHeader>
        <CardTitle>Choose AI models</CardTitle>
        <CardDescription>
          Select what AddaxAI should detect and identify.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label htmlFor="detection-model">Detection model</Label>
          <Select
            value={detectionModelId}
            onValueChange={setDetectionModelId}
          >
            <SelectTrigger id="detection-model">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {detectionModels.map((m) => (
                <SelectItem key={m.model_id} value={m.model_id}>
                  {m.emoji} {m.friendly_name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Finds animals, people, and vehicles in your media.
          </p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="classification-model">
            Species identification
          </Label>
          <Select
            value={classificationModelId}
            onValueChange={setClassificationModelId}
          >
            <SelectTrigger id="classification-model">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NO_CLASSIFIER}>
                ∅ Detection only (no species)
              </SelectItem>
              {classificationModels.map((m) => (
                <SelectItem key={m.model_id} value={m.model_id}>
                  {m.emoji} {m.friendly_name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Optional. Pick a species classifier when you want labels
            beyond animal / person / vehicle.
          </p>
        </div>

        {save.isError && (
          <p className="text-sm text-destructive">
            Could not save model selection:{" "}
            {save.error instanceof Error
              ? save.error.message
              : "unknown error"}
          </p>
        )}
      </CardContent>

      <CardFooter className="justify-between">
        <Button
          variant="outline"
          onClick={() => navigate(`/folder-runs/${runId}/folder`)}
          className="gap-2"
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </Button>
        <Button
          onClick={() => save.mutate()}
          disabled={save.isPending}
          className="gap-2"
        >
          {save.isPending ? "Saving..." : "Continue"}
          <ArrowRight className="h-4 w-4" />
        </Button>
      </CardFooter>
    </Card>
  );
}
