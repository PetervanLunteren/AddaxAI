/**
 * Create Project Dialog.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear validation
 * - Explicit error handling
 */

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { Info, InfoIcon, ListTodo } from "lucide-react";
import { projectsApi, type ProjectCreate, type ProjectResponse } from "../../api/projects";
import { ImageDropZone } from "./ImageDropZone";
import { modelsApi } from "../../api/models";
import { useTaskProgress } from "../../hooks/useTaskProgress";
import { ModelStatusBadge } from "./ModelStatusBadge";
import { ModelPreparationView } from "./ModelPreparationView";
import { ModelPreparationErrorView } from "./ModelPreparationErrorView";
import { Button } from "../ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { ClassificationModelGroupedItems } from "../models/ClassificationModelGroupedItems";
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
import { Textarea } from "../ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import { ModelInfoSheet } from "../models/ModelInfoSheet";
import { SpeciesSelectionModal } from "../taxonomy/SpeciesSelectionModal";

const projectSchema = z.object({
  name: z.string().min(1, "Project name is required").max(100, "Name too long"),
  description: z.string().max(500, "Description too long").optional(),
  detection_model_id: z.literal("MD5A-0-0"),
  classification_model_id: z.string().nullable().optional(),
  embedding_model_id: z.string().nullable(),
  excluded_classes: z.array(z.string()),
  country_code: z.string().optional().nullable(),
  state_code: z.string().optional().nullable(),
  detection_threshold: z.number().min(0).max(1),
  event_smoothing: z.boolean(),
  smoothing_strength: z.enum(["mild", "normal", "aggressive"]),
  taxonomic_rollup: z.boolean(),
  taxonomic_rollup_threshold: z.number().min(0.1).max(1.0),
  independence_interval: z.number().min(0),
});

interface CreateProjectDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CreateProjectDialog({
  open,
  onOpenChange,
}: CreateProjectDialogProps) {
  const queryClient = useQueryClient();
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [showModelInfo, setShowModelInfo] = useState(false);

  // Model preparation state
  type DialogStage = "form" | "preparing" | "error";
  const [stage, setStage] = useState<DialogStage>("form");
  const [preparingTaskId, setPreparingTaskId] = useState<string | null>(null);
  const [preparationError, setPreparationError] = useState<string | null>(null);

  // Fetch available classification models (already sorted alphabetically by backend)
  const { data: classificationModels = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: () => modelsApi.listClassificationModels(),
    enabled: open,
  });

  const form = useForm<ProjectCreate>({
    resolver: zodResolver(projectSchema),
    defaultValues: {
      name: "",
      description: "",
      detection_model_id: "MD5A-0-0",
      classification_model_id: null,
      embedding_model_id: "DINOV2-VITB14",
      excluded_classes: [],
      country_code: null,
      state_code: null,
      detection_threshold: 0.5,
      event_smoothing: true,
      smoothing_strength: "normal" as const,
      taxonomic_rollup: true,
      taxonomic_rollup_threshold: 0.65,
      independence_interval: 1800, // Will be converted from minutes in UI
    },
  });

  // Watch classification model changes
  const classificationModelId = form.watch("classification_model_id");
  const hasClassificationModel = !!classificationModelId && classificationModelId !== "none";

  // Label selection state
  const [labelSelectionModalOpen, setLabelSelectionModalOpen] = useState(false);
  const excludedClasses = form.watch("excluded_classes") ?? [];

  // Fetch taxonomy for selected classification model
  const { data: taxonomy } = useQuery({
    queryKey: ["taxonomy", classificationModelId],
    queryFn: () => modelsApi.getTaxonomy(classificationModelId!),
    enabled: hasClassificationModel && open,
  });

  // Fetch model status when model is selected
  const { data: modelStatus } = useQuery({
    queryKey: ["model-status", classificationModelId],
    queryFn: () => modelsApi.getModelStatus(classificationModelId!),
    enabled: !!classificationModelId && classificationModelId !== "none" && open,
  });

  // WebSocket progress tracking for model preparation
  const { progress, message } = useTaskProgress({
    taskId: preparingTaskId,
    onComplete: () => {
      // Refresh model status
      queryClient.invalidateQueries({ queryKey: ["model-status", classificationModelId] });
      setPreparingTaskId(null);
      setStage("form");
    },
    onError: (error) => {
      setPreparationError(error);
      setStage("error");
      setPreparingTaskId(null);
    },
  });

  const createMutation = useMutation({
    mutationFn: (data: ProjectCreate) => projectsApi.create(data),
    onSuccess: async (newProject: ProjectResponse) => {
      if (imageFile) {
        try {
          await projectsApi.uploadThumbnail(newProject.id, imageFile);
        } catch (e) {
          // Project was created, image upload failed. Not critical.
          console.error("Failed to upload project image:", e);
        }
      }
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      form.reset();
      setImageFile(null);
      onOpenChange(false);
    },
    onError: (error: Error) => {
      console.error("Failed to create project:", error);
      form.setError("root", {
        message: error.message || "Failed to create project",
      });
    },
  });

  // Handler for model preparation
  const handlePrepareModel = async () => {
    if (!classificationModelId) return;

    try {
      setStage("preparing");
      const response = await modelsApi.prepareModel(classificationModelId);
      setPreparingTaskId(response.task_id);
    } catch (error: any) {
      setPreparationError(error.message || "Failed to start model preparation");
      setStage("error");
    }
  };

  // Handler for canceling preparation
  const handleCancelPreparation = () => {
    setPreparingTaskId(null);
    setStage("form");
  };

  // Handler for retrying after error
  const handleRetryPreparation = () => {
    setPreparationError(null);
    handlePrepareModel();
  };

  const onSubmit = (data: ProjectCreate) => {
    // Silently fill in the project timezone using the browser's IANA
    // zone. Electron's Chromium engine honors the OS setting. No UI
    // field here — users can change it later in Project settings if
    // the data was recorded in a different timezone.
    const detectedTimezone =
      Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
    createMutation.mutate({ ...data, timezone: detectedTimezone });
  };

  // Get selected model info for preparation view
  const selectedModel = classificationModels.find((m) => m.model_id === classificationModelId);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl max-h-[90vh] overflow-y-auto">
        {/* Form View */}
        {stage === "form" && (
          <>
            <DialogHeader>
              <DialogTitle>Create new project</DialogTitle>
              <DialogDescription>
                Projects organize your camera trap sites, deployments, and analysis settings
              </DialogDescription>
            </DialogHeader>

            <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            <TooltipProvider>
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="flex items-center gap-1.5">
                      Project name
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">
                            A unique name for your project
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </FormLabel>
                    <FormControl>
                      <Input placeholder="e.g., Yellowstone camera trap project" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="description"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="flex items-center gap-1.5">
                      Description
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">
                            Optional notes about the project's purpose, location, team members, etc.
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </FormLabel>
                    <FormControl>
                      <Textarea
                        placeholder="Brief description of the project"
                        className="resize-y"
                        rows={2}
                        maxLength={500}
                        {...field}
                      />
                    </FormControl>
                    <div className="flex items-center justify-between">
                      <FormMessage />
                      <p className={`text-xs ${
                        (field.value?.length || 0) > 450
                          ? "text-orange-600"
                          : "text-muted-foreground"
                      }`}>
                        {field.value?.length || 0} / 500
                      </p>
                    </div>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="classification_model_id"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="flex items-center gap-1.5">
                      Classification model
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">
                            The AI model that will identify species in your camera trap images.
                            Choose a model trained on species from your geographic region.
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </FormLabel>
                    <div className="flex gap-2 items-stretch">
                      <Select
                        onValueChange={(val) => field.onChange(val === "none" ? "none" : val)}
                        defaultValue={field.value ?? "none"}
                        value={field.value ?? "none"}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select classification model">
                            {(() => {
                              if (!field.value || field.value === "none") {
                                return (
                                  <div className="flex flex-col items-start py-1">
                                    <div>∅ No classification model</div>
                                    <div className="text-xs text-muted-foreground">
                                      Run animal detector only, identify species manually
                                    </div>
                                  </div>
                                );
                              }
                              const selectedModel = classificationModels.find(
                                (m) => m.model_id === field.value
                              );
                              if (!selectedModel) return null;
                              return (
                                <div className="flex flex-col items-start py-1">
                                  <div>
                                    {selectedModel.emoji} {selectedModel.friendly_name}
                                  </div>
                                  {selectedModel.description_short && (
                                    <div className="text-xs text-muted-foreground">
                                      {selectedModel.description_short}
                                    </div>
                                  )}
                                </div>
                              );
                            })()}
                          </SelectValue>
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="none">
                          ∅ No classification model
                          <br />
                          <span className="text-xs text-muted-foreground">Run animal detector only, identify species manually</span>
                        </SelectItem>
                        <ClassificationModelGroupedItems
                          models={classificationModels.filter((m) => m.model_id !== "none")}
                        />
                      </SelectContent>
                    </Select>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="self-center">
                          <Button
                            type="button"
                            variant="outline"
                            className="px-3"
                            onClick={() => field.value && field.value !== "none" && setShowModelInfo(true)}
                            disabled={!field.value || field.value === "none"}
                          >
                            <InfoIcon className="h-4 w-4" />
                          </Button>
                        </span>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>
                          {field.value
                            ? "View model information"
                            : "Select a classification model to view details"}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <FormMessage />
                </FormItem>
              )}
            />

              {/* Model Status Badge */}
              {classificationModelId && modelStatus && (
                <ModelStatusBadge
                  status={modelStatus}
                  onPrepare={handlePrepareModel}
                  isPreparing={stage === "preparing"}
                />
              )}

              {/* Label selection */}
              {hasClassificationModel && taxonomy && (
                <FormItem>
                  <FormLabel className="flex items-center gap-1.5">
                    Label selection
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">
                          Limit predictions to labels expected in your project area to reduce false positives. You can change this later in settings.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </FormLabel>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setLabelSelectionModalOpen(true)}
                    className="w-full min-h-14 flex flex-col items-start justify-center gap-1 text-left"
                  >
                    <div className="flex items-center gap-2">
                      <ListTodo className="h-4 w-4" />
                      <span>Select labels</span>
                    </div>
                    <span className="text-xs text-muted-foreground">
                      Currently included {(taxonomy.all_classes?.length || 0) - excludedClasses.length} of {taxonomy.all_classes?.length || 0}
                    </span>
                  </Button>
                </FormItem>
              )}

              <ImageDropZone
                value={imageFile}
                existingUrl={null}
                onChange={setImageFile}
              />

              {form.formState.errors.root && (
                <p className="text-sm font-medium text-destructive">
                  {form.formState.errors.root.message}
                </p>
              )}

              <DialogFooter>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => onOpenChange(false)}
                >
                  Cancel
                </Button>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span>
                      <Button
                        type="submit"
                        disabled={
                          createMutation.isPending ||
                          (!!classificationModelId && classificationModelId !== "none" && modelStatus?.status !== "ready")
                        }
                      >
                        {createMutation.isPending ? "Creating..." : "Create project"}
                      </Button>
                    </span>
                  </TooltipTrigger>
                  {!!classificationModelId && classificationModelId !== "none" && modelStatus?.status !== "ready" && (
                    <TooltipContent>
                      <p>Model needs preparing first</p>
                    </TooltipContent>
                  )}
                </Tooltip>
              </DialogFooter>
            </TooltipProvider>
          </form>
        </Form>
          </>
        )}

        {/* Preparing View */}
        {stage === "preparing" && selectedModel && (
          <ModelPreparationView
            modelName={selectedModel.friendly_name}
            modelEmoji={selectedModel.emoji}
            progress={progress}
            message={message}
            onCancel={handleCancelPreparation}
          />
        )}

        {/* Error View */}
        {stage === "error" && (
          <ModelPreparationErrorView
            errorMessage={preparationError || "Unknown error occurred"}
            onRetry={handleRetryPreparation}
            onCancel={() => setStage("form")}
          />
        )}
      </DialogContent>

      {/* Model Info Sheet */}
      <ModelInfoSheet
        modelId={form.watch("classification_model_id")}
        open={showModelInfo}
        onOpenChange={setShowModelInfo}
      />

      {/* Label Selection Modal */}
      {classificationModelId && taxonomy && (
        <SpeciesSelectionModal
          modelId={classificationModelId}
          excludedClasses={excludedClasses}
          onExclusionChange={(classes) => {
            form.setValue("excluded_classes", classes, { shouldDirty: true });
          }}
          open={labelSelectionModalOpen}
          onOpenChange={setLabelSelectionModalOpen}
          totalSpeciesCount={taxonomy.all_classes?.length || 0}
          countryCode={form.watch("country_code")}
          stateCode={form.watch("state_code")}
          onLocationChange={(country, state) => {
            form.setValue("country_code", country, { shouldDirty: true });
            form.setValue("state_code", state, { shouldDirty: true });
          }}
        />
      )}
    </Dialog>
  );
}
