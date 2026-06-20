/**
 * Create Project Dialog.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear validation
 * - Explicit error handling
 */

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useForm, type Resolver } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { projectsApi, type ProjectCreate, type ProjectResponse } from "../../api/projects";
import { ImageDropZone } from "./ImageDropZone";
import { modelsApi } from "../../api/models";
import { useTaskProgress } from "../../hooks/useTaskProgress";
import { ModelStatusBadge } from "./ModelStatusBadge";
import { ModelPreparationView } from "./ModelPreparationView";
import { ModelPreparationErrorView } from "./ModelPreparationErrorView";
import { Button } from "../ui/button";
import { SelectItem } from "../ui/select";
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
import { LabelSelectionField, useLabelSelectionCaption } from "../taxonomy/LabelSelectionField";
import { ModelSelect } from "../models/ModelSelect";
import {
  loadLastUsedSettings,
  saveLastUsedSettings,
} from "../../lib/folderRunSettings";

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
  const navigate = useNavigate();
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

  // Pre-fill the model + species selection from the last analysis (project or
  // folder run) via the shared last-used-settings store. Read once so
  // re-renders don't churn it. Falls back to the cold defaults on first run.
  // The stored classification model is validated against the catalog once it
  // loads (effect below).
  const [lastSelection] = useState(loadLastUsedSettings);

  const form = useForm<ProjectCreate>({
    resolver: zodResolver(projectSchema) as unknown as Resolver<ProjectCreate>,
    defaultValues: {
      name: "",
      description: "",
      detection_model_id: "MD5A-0-0",
      classification_model_id: lastSelection?.classification_model_id ?? null,
      embedding_model_id: lastSelection?.embedding_model_id ?? "DINOV2-VITB14",
      excluded_classes: lastSelection?.excluded_classes ?? [],
      country_code: lastSelection?.country_code ?? null,
      state_code: lastSelection?.state_code ?? null,
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
  const labelCaption = useLabelSelectionCaption(
    hasClassificationModel ? classificationModelId! : "",
  );

  // Label selection state
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

  // A sticky classification model that is no longer installed would leave the
  // form pointing at a dead id. Once the catalog loads, drop it (and its
  // region/exclusions) back to "no classification model".
  useEffect(() => {
    if (!open || classificationModels.length === 0) return;
    const cur = form.getValues("classification_model_id");
    if (cur && cur !== "none" && !classificationModels.some((m) => m.model_id === cur)) {
      form.setValue("classification_model_id", null);
      form.setValue("country_code", null);
      form.setValue("state_code", null);
      form.setValue("excluded_classes", []);
    }
  }, [open, classificationModels, form]);

  // Keep exclusions consistent with the selected model's taxonomy. On a
  // sticky pre-fill the stored exclusions match the stored model and survive;
  // switching models drops the now-stale ones. Mirrors SettingsPage.
  useEffect(() => {
    if (hasClassificationModel && taxonomy?.all_classes) {
      const current = form.getValues("excluded_classes");
      const valid = current.filter((c) => taxonomy.all_classes.includes(c));
      if (valid.length !== current.length) {
        form.setValue("excluded_classes", valid, { shouldDirty: true });
      }
    }
  }, [classificationModelId, taxonomy, hasClassificationModel, form]);

  // WebSocket progress tracking for model preparation
  const { progress, message, cancel } = useTaskProgress({
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
    onCancelled: () => {
      // Backend confirmed the cancel (subprocess killed, partial cleaned).
      setPreparingTaskId(null);
      setStage("form");
    },
  });

  const createMutation = useMutation({
    mutationFn: (data: ProjectCreate) => projectsApi.create(data),
    onSuccess: async (newProject: ProjectResponse, variables: ProjectCreate) => {
      if (imageFile) {
        try {
          await projectsApi.uploadThumbnail(newProject.id, imageFile);
        } catch (e) {
          // Project was created, image upload failed. Not critical.
          console.error("Failed to upload project image:", e);
        }
      }
      // Remember this model + species selection so the next project / folder
      // run pre-fills it. Only this subset is written; folder-run-only params
      // in the shared store are left untouched. "none" is normalised to null
      // so the store's canonical "no model" value is null.
      const noneToNull = (v: string | null | undefined) =>
        !v || v === "none" ? null : v;
      saveLastUsedSettings({
        classification_model_id: noneToNull(variables.classification_model_id),
        embedding_model_id: noneToNull(variables.embedding_model_id),
        country_code: variables.country_code ?? null,
        state_code: variables.state_code ?? null,
        excluded_classes: variables.excluded_classes,
      });
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      form.reset();
      setImageFile(null);
      onOpenChange(false);
      // Drop the user straight into the new project so the next step is
      // obvious. An empty project redirects to the Analyses page (see
      // ProjectIndexRoute), which is exactly "add a deployment / analyze".
      navigate(`/projects/${newProject.id}`);
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

  // Handler for canceling preparation. Sends a real cancel over the
  // WebSocket; the worker kills its subprocess and replies with a
  // "cancelled" terminal event, which onCancelled handles. We keep the
  // taskId mounted so the socket stays open to deliver the cancel.
  const handleCancelPreparation = () => {
    cancel();
  };

  // Handler for retrying after error
  const handleRetryPreparation = () => {
    setPreparationError(null);
    handlePrepareModel();
  };

  const onSubmit = (data: ProjectCreate) => {
    // No timezone is sent: a new project starts with none and the backend
    // auto-derives the camera timezone from the first site's coordinates
    // (the authoritative source for the sun-based insights). The browser's
    // zone is the wrong guess for remote cameras, so we no longer use it.
    // Users can still set it explicitly in Project settings.
    createMutation.mutate(data);
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
                A project is a persistent workspace for many cameras over
                time, holding sites, deployments, verification history,
                dashboards, and exports.
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
                    <FormLabel>Project name</FormLabel>
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
                    <FormLabel>Description</FormLabel>
                    <FormControl>
                      <Textarea
                        placeholder="Notes about purpose, location, team members, etc."
                        className="resize-y"
                        rows={2}
                        maxLength={500}
                        {...field}
                        value={field.value ?? ""}
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
                    <FormLabel>Classification model</FormLabel>
                    <p className="text-xs text-muted-foreground">
                      The AI model that identifies species in your images.
                      Pick one trained for your region.
                    </p>
                    <ModelSelect
                      value={field.value ?? "none"}
                      onValueChange={(val) => field.onChange(val === "none" ? "none" : val)}
                      models={classificationModels}
                      placeholder="Select classification model"
                      noneValue="none"
                      noneLabel="No classification model"
                      onShowInfo={() => setShowModelInfo(true)}
                    >
                      <SelectItem value="none">
                        ∅ No classification model
                        <br />
                        <span className="text-xs text-muted-foreground">Run animal detector only, identify species manually</span>
                      </SelectItem>
                      <ClassificationModelGroupedItems
                        models={classificationModels.filter((m) => m.model_id !== "none")}
                      />
                    </ModelSelect>
                  <FormMessage />
                </FormItem>
              )}
            />

              {/* Model Status Badge */}
              {classificationModelId && modelStatus && (
                <ModelStatusBadge
                  status={modelStatus}
                  onPrepare={handlePrepareModel}
                  isPreparing={false}
                />
              )}

              {/* Label selection */}
              {hasClassificationModel && taxonomy && (
                <FormItem>
                  <FormLabel>Label selection</FormLabel>
                  <p className="text-xs text-muted-foreground">
                    {labelCaption}
                  </p>
                  <LabelSelectionField
                    modelId={classificationModelId}
                    excludedClasses={excludedClasses}
                    allClasses={taxonomy.all_classes ?? []}
                    countryCode={form.watch("country_code")}
                    stateCode={form.watch("state_code")}
                    onExclusionChange={(classes) => {
                      form.setValue("excluded_classes", classes, { shouldDirty: true });
                    }}
                    onLocationChange={(country, state) => {
                      form.setValue("country_code", country, { shouldDirty: true });
                      form.setValue("state_code", state, { shouldDirty: true });
                    }}
                  />
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
    </Dialog>
  );
}
