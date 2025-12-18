/**
 * Add Deployment Dialog.
 *
 * Allows users to queue a deployment for ML analysis.
 * Site is automatically set to "Unknown Site" for MVP simplicity.
 */

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { useParams } from "react-router-dom";
import { jobsApi } from "../../api/jobs";
import { mlModelsApi } from "../../api/ml-models";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { Alert, AlertDescription } from "../ui/alert";
import { useTaskProgress } from "../../hooks/useTaskProgress";
import { Progress } from "../ui/progress";

const deploymentSchema = z.object({
  folder_path: z.string().min(1, "Folder path is required"),
  detection_model: z.enum(["MD5A-0-0", "MD5B-0-0"]),
  classification_model: z.enum(["EUR-DF-v1-3", "NAM-ADS-v1", "none"]),
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

  const prepareModelMutation = useMutation({
    mutationFn: (modelId: string) => mlModelsApi.prepare(modelId),
    onSuccess: (response) => {
      // Start tracking progress via WebSocket
      setPrepareTaskId(response.task_id);
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
        <Alert className="border-blue-500 bg-blue-50 dark:bg-blue-950">
          <Loader2 className="h-4 w-4 text-blue-600 animate-spin" />
          <AlertDescription className="space-y-3">
            <div className="text-blue-800 dark:text-blue-200 font-semibold">
              Preparing model...
            </div>
            <Progress
              value={progressValue * 100}
              className="h-2 bg-blue-200 dark:bg-blue-900"
            />
            <code className="block text-xs text-blue-700 dark:text-blue-300 bg-blue-100 dark:bg-blue-900 px-2 py-1 rounded font-mono overflow-x-auto">
              {progressMessage || "Starting..."}
            </code>
          </AlertDescription>
        </Alert>
      );
    }

    if (status.status === "ready") {
      return (
        <Alert className="border-green-500 bg-green-50 dark:bg-green-950">
          <CheckCircle2 className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800 dark:text-green-200">
            Model ready: Environment configured
          </AlertDescription>
        </Alert>
      );
    }

    // Show separate buttons based on what's needed
    if (status.status === "needs_weights") {
      return (
        <Alert className="border-yellow-500 bg-yellow-50 dark:bg-yellow-950">
          <AlertCircle className="h-4 w-4 text-yellow-600" />
          <AlertDescription className="flex items-center justify-between text-yellow-800 dark:text-yellow-200">
            <span>Model weights not downloaded</span>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handlePrepareWeights}
              disabled={prepareWeightsMutation.isPending}
              className="ml-2"
            >
              {prepareWeightsMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Downloading...
                </>
              ) : (
                "Download Weights"
              )}
            </Button>
          </AlertDescription>
        </Alert>
      );
    }

    if (status.status === "needs_env") {
      return (
        <Alert className="border-yellow-500 bg-yellow-50 dark:bg-yellow-950">
          <AlertCircle className="h-4 w-4 text-yellow-600" />
          <AlertDescription className="flex items-center justify-between text-yellow-800 dark:text-yellow-200">
            <span>Environment not built</span>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handlePrepareEnv}
              disabled={prepareEnvMutation.isPending}
              className="ml-2"
            >
              {prepareEnvMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Building...
                </>
              ) : (
                "Build Environment"
              )}
            </Button>
          </AlertDescription>
        </Alert>
      );
    }

    if (status.status === "needs_both") {
      return (
        <Alert className="border-yellow-500 bg-yellow-50 dark:bg-yellow-950">
          <AlertCircle className="h-4 w-4 text-yellow-600" />
          <AlertDescription className="space-y-2">
            <div className="text-yellow-800 dark:text-yellow-200">
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
                  "Download Weights"
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
                  "Build Environment"
                )}
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      );
    }

    return null;
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Add Deployment to Queue</DialogTitle>
          <DialogDescription>
            Select a folder and configure ML models for camera trap analysis.
            Site will be automatically set to "Unknown Site".
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
                  <FormLabel>Folder Path</FormLabel>
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
                  <FormLabel>Detection Model</FormLabel>
                  <Select
                    onValueChange={field.onChange}
                    defaultValue={field.value}
                  >
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select detection model" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value="MD5A-0-0">
                        MegaDetector 5a
                      </SelectItem>
                      <SelectItem value="MD5B-0-0">
                        MegaDetector 5b
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <FormDescription>
                    Model for detecting animals in images
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            {/* Model Status Indicator */}
            {isLoadingStatus ? (
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
                  <FormLabel>Classification Model</FormLabel>
                  <Select
                    onValueChange={field.onChange}
                    defaultValue={field.value}
                  >
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select classification model" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value="none">
                        None (Detection only)
                      </SelectItem>
                      <SelectItem value="EUR-DF-v1-3">
                        Europe (Deepfaune v1.3)
                      </SelectItem>
                      <SelectItem value="NAM-ADS-v1">
                        Namibia (Addax DS v1)
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <FormDescription>
                    Regional species classifier (optional)
                  </FormDescription>
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
