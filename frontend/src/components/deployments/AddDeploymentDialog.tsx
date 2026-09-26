/**
 * Add Deployment Dialog.
 *
 * Allows users to queue a deployment for ML analysis.
 */

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { useParams } from "react-router-dom";
import { jobsApi } from "../../api/jobs";
import { mlModelsApi } from "../../api/ml-models";
import { modelsApi } from "../../api/models";
import { ClassificationModelGroupedItems } from "../models/ClassificationModelGroupedItems";
import { ModelSelect } from "../models/ModelSelect";
import type {
  JobCreate,
  DetectionModel,
  ClassificationModel,
  DeploymentAnalysisPayload,
  ModelStatusResponse,
} from "../../api/types";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "../ui/form";
import { Input } from "../ui/input";
import {
  SelectItem,
} from "../ui/select";
import { Loader2 } from "lucide-react";
import { Callout } from "../ui/callout";
import { useTaskProgress } from "../../hooks/useTaskProgress";
import { Progress } from "../ui/progress";

const deploymentSchema = z.object({
  folder_path: z.string().min(1, "Folder path is required"),
  detection_model: z.string().min(1),
  classification_model: z.string(),
});

type DeploymentFormValues = z.infer<typeof deploymentSchema>;

interface AddDeploymentDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function AddDeploymentDialog({
  open,
  onOpenChange,
}: AddDeploymentDialogProps) {
  const { projectId } = useParams<{ projectId: string }>();
  const queryClient = useQueryClient();
  const [prepareTaskId, setPrepareTaskId] = useState<string | null>(null);

  const { data: detectionModels = [], isLoading: detectionModelsLoading } = useQuery({
    queryKey: ["models", "detection"],
    queryFn: modelsApi.listDetectionModels,
    enabled: open,
  });
  const { data: classificationModels = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: modelsApi.listClassificationModels,
    enabled: open,
  });

  const form = useForm<DeploymentFormValues>({
    resolver: zodResolver(deploymentSchema),
    defaultValues: {
      folder_path: "",
      detection_model: "MD5A-0-0",
      classification_model: "none",
    },
  });

  // Watch the selected detection model
  const selectedDetectionModel = form.watch("detection_model");
  const selectedClassificationId = form.watch("classification_model");
  const selectedClassificationInfo = classificationModels.find((model) => model.model_id === selectedClassificationId);
  const classifierAliasMismatch = Boolean(
    selectedClassificationInfo?.uses_detection_classes &&
    selectedClassificationInfo.model_id !== selectedDetectionModel,
  );

  // Query model status when dialog opens or model changes
  const { data: modelStatus, isLoading: isLoadingStatus } = useQuery({
    queryKey: ["model-status", selectedDetectionModel],
    queryFn: () => mlModelsApi.getStatus(selectedDetectionModel),
    enabled: open && !!selectedDetectionModel,
  });

  // WebSocket progress tracking for model preparation
  const { message: progressMessage, progress: progressValue } = useTaskProgress({
    taskId: prepareTaskId,
    onComplete: () => {
      // Refetch model status when preparation completes
      queryClient.invalidateQueries({
        queryKey: ["model-status", selectedDetectionModel],
      });
      setPrepareTaskId(null);
    },
    onError: (error) => {
      console.error("Model preparation failed:", error);
      setPrepareTaskId(null);
    },
  });

  const prepareWeightsMutation = useMutation({
    mutationFn: (modelId: string) => mlModelsApi.prepareWeights(modelId),
    onSuccess: (response) => {
      // Start tracking progress via WebSocket
      setPrepareTaskId(response.task_id);
    },
  });

  const prepareEnvMutation = useMutation({
    mutationFn: (modelId: string) => mlModelsApi.prepareEnv(modelId),
    onSuccess: (response) => {
      // Start tracking progress via WebSocket
      setPrepareTaskId(response.task_id);
    },
  });

  const createJobMutation = useMutation({
    mutationFn: (data: JobCreate) => jobsApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      form.reset();
      onOpenChange(false);
    },
  });

  const handlePrepareWeights = () => {
    if (selectedDetectionModel) {
      prepareWeightsMutation.mutate(selectedDetectionModel);
    }
  };

  const handlePrepareEnv = () => {
    if (selectedDetectionModel) {
      prepareEnvMutation.mutate(selectedDetectionModel);
    }
  };

  const onSubmit = (values: DeploymentFormValues) => {
    if (!projectId) {
      console.error("Project ID is missing");
      return;
    }
    if (classifierAliasMismatch) {
      form.setError("classification_model", {
        message: "Select the same model for Detection and Classification to reuse detector classes.",
      });
      return;
    }

    const payload: DeploymentAnalysisPayload = {
      project_id: projectId,
      folder_path: values.folder_path,
      detection_model: values.detection_model as DetectionModel,
      classification_model: values.classification_model as ClassificationModel,
    };

    const jobCreate: JobCreate = {
      type: "deployment_analysis",
      payload: payload as unknown as Record<string, unknown>,
    };

    createJobMutation.mutate(jobCreate);
  };

  // Render model status indicator
  const renderModelStatus = (status: ModelStatusResponse | undefined) => {
    if (!status) {
      return null;
    }

    // Show progress if model is being prepared
    if (prepareTaskId) {
      return (
        <Callout variant="info">
          <div className="space-y-3">
            <div className="font-semibold">Preparing model...</div>
            <Progress value={progressValue * 100} className="h-2" />
            <code className="block overflow-x-auto rounded bg-blue-100 px-2 py-1 font-mono text-xs">
              {progressMessage || "Starting..."}
            </code>
          </div>
        </Callout>
      );
    }

    if (status.status === "ready") {
      return (
        <Callout variant="success" size="compact">
          Model ready: Environment configured
        </Callout>
      );
    }

    // Show separate buttons based on what's needed
    if (status.status === "needs_weights") {
      return (
        <Callout
          variant="warning"
          action={
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handlePrepareWeights}
              disabled={prepareWeightsMutation.isPending}
            >
              {prepareWeightsMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Downloading...
                </>
              ) : (
                "Download weights"
              )}
            </Button>
          }
        >
          Model weights not downloaded
        </Callout>
      );
    }

    if (status.status === "needs_env") {
      return (
        <Callout
          variant="warning"
          action={
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handlePrepareEnv}
              disabled={prepareEnvMutation.isPending}
            >
              {prepareEnvMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Building...
                </>
              ) : (
                "Build environment"
              )}
            </Button>
          }
        >
          Environment not built
        </Callout>
      );
    }

    if (status.status === "needs_both") {
      return (
        <Callout variant="warning">
          <div className="space-y-2">
            <div>
              Model needs weights and environment
            </div>
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handlePrepareWeights}
                disabled={prepareWeightsMutation.isPending || prepareEnvMutation.isPending}
              >
                {prepareWeightsMutation.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Downloading...
                  </>
                ) : (
                  "Download weights"
                )}
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handlePrepareEnv}
                disabled={prepareWeightsMutation.isPending || prepareEnvMutation.isPending}
              >
                {prepareEnvMutation.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Building...
                  </>
                ) : (
                  "Build environment"
                )}
              </Button>
            </div>
          </div>
        </Callout>
      );
    }

    return null;
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Add deployment to queue</DialogTitle>
          <DialogDescription>
            Select a folder and configure ML models for camera trap analysis.
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            {/* Folder Path */}
            <FormField
              control={form.control}
              name="folder_path"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Folder path</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="/Users/you/camera-traps/site-a"
                      {...field}
                    />
                  </FormControl>
                  <FormDescription>
                    Paste the absolute path to your camera trap folder.
                    <br />
                    <em className="text-xs text-muted-foreground">
                      Note: Native folder picker coming with desktop app
                    </em>
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            {/* Detection Model */}
            <FormField
              control={form.control}
              name="detection_model"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Detection model</FormLabel>
                  <ModelSelect
                    modelType="detection"
                    value={field.value}
                    onValueChange={field.onChange}
                    models={detectionModels}
                    placeholder="Select detection model"
                  >
                    {detectionModels.map((model) => (
                      <SelectItem key={model.model_id} value={model.model_id}>
                        {model.emoji} {model.friendly_name}
                        {model.description_short ? <><br /><span className="text-xs text-muted-foreground">{model.description_short}</span></> : null}
                      </SelectItem>
                    ))}
                  </ModelSelect>
                  <FormDescription>
                    Model for detecting animals in images
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            {/* Model Status Indicator */}
            {isLoadingStatus || detectionModelsLoading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Checking model status...</span>
              </div>
            ) : (
              renderModelStatus(modelStatus)
            )}

            {/* Classification Model */}
            <FormField
              control={form.control}
              name="classification_model"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Classification model</FormLabel>
                  <ModelSelect
                    modelType="classification"
                    value={field.value}
                    onValueChange={field.onChange}
                    models={classificationModels}
                    placeholder="Select classification model"
                    noneValue="none"
                    noneLabel="No classification model"
                  >
                    <SelectItem value="none">∅ No classification model</SelectItem>
                    <ClassificationModelGroupedItems
                      models={classificationModels.filter((m) => m.model_id !== "none")}
                      detectionModelId={selectedDetectionModel}
                      selectedModelId={form.watch("classification_model")}
                    />
                  </ModelSelect>
                  <FormDescription>
                    {selectedDetectionModel === selectedClassificationId && selectedClassificationInfo?.uses_detection_classes
                      ? `Reuses ${Object.keys(selectedClassificationInfo.class_names ?? {}).length} detector classes and confidence without another inference.`
                      : "Regional species classifier (optional). The same custom detector can be selected here to reuse its classes without another inference."}
                  </FormDescription>
                  {classifierAliasMismatch ? (
                    <p role="alert" className="text-sm text-destructive">
                      This classification choice belongs to another detection model. Select that same detector or choose another classification model.
                    </p>
                  ) : null}
                  <FormMessage />
                </FormItem>
              )}
            />

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={
                  createJobMutation.isPending ||
                  isLoadingStatus ||
                  modelStatus?.status !== "ready"
                }
              >
                {createJobMutation.isPending ? "Adding..." : "Add to Queue"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}
