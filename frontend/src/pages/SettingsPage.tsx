/**
 * Project Settings Page.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear structure
 * - Explicit error handling
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { Save, RotateCcw, Undo2, Check, ChevronsUpDown, ListTodo, InfoIcon, RefreshCw, X } from "lucide-react";
import { toast } from "sonner";
import { projectsApi, type ProjectUpdate } from "../api/projects";
import { modelsApi } from "../api/models";
import { SpeciesSelectionModal } from "../components/taxonomy/SpeciesSelectionModal";
import { ModelInfoSheet } from "../components/models/ModelInfoSheet";
import { ModelStatusBadge } from "../components/projects/ModelStatusBadge";
import { ModelPreparationView } from "../components/projects/ModelPreparationView";
import { ModelPreparationErrorView } from "../components/projects/ModelPreparationErrorView";
import {
  SaveResultsModal,
  type SaveResults,
  type StatSnapshot,
} from "../components/projects/SaveResultsModal";

import { useTaskProgress } from "../hooks/useTaskProgress";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../components/ui/tooltip";
import {
  Dialog,
  DialogContent,
} from "../components/ui/dialog";
import { Button } from "../components/ui/button";
import { Slider } from "../components/ui/slider";
import { Switch } from "../components/ui/switch";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "../components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "../components/ui/popover";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormLabel,
  FormMessage,
} from "../components/ui/form";
import { cn } from "../lib/utils";

const settingsSchema = z.object({
  detection_model_id: z.string().min(1, "Detection model is required"),
  classification_model_id: z.string().min(1, "Classification model is required"),
  excluded_classes: z.array(z.string()),
  country_code: z.string().optional().nullable(),
  state_code: z.string().optional().nullable(),
  video_fps: z.number().min(0.1).max(10),
  detection_threshold: z.number().min(0).max(1),
  event_smoothing: z.boolean(),
  taxonomic_rollup: z.boolean(),
  taxonomic_rollup_threshold: z.number().min(0.1).max(1.0),
  independence_interval: z.number().min(0),
});

const INDEPENDENCE_INTERVAL_OPTIONS = [
  { value: "0", label: "Disabled" },
  { value: "60", label: "1 minute" },
  { value: "300", label: "5 minutes" },
  { value: "900", label: "15 minutes" },
  { value: "1800", label: "30 minutes" },
  { value: "3600", label: "60 minutes" },
  // Debugging option: large interval to group many videos into one event
  { value: "2592000", label: "1 month (for debugging)" },
];

const VIDEO_FPS_OPTIONS = [
  { value: "0.1", label: "1 frame every 10 seconds" },
  { value: "0.25", label: "1 frame every 4 seconds" },
  { value: "0.5", label: "1 frame every 2 seconds" },
  { value: "1", label: "1 frame per second" },
  { value: "2", label: "2 frames per second" },
  { value: "4", label: "4 frames per second" },
  { value: "10", label: "10 frames per second" },
];

type SettingsFormData = z.infer<typeof settingsSchema>;

/** Settings that trigger classification reprocessing when changed. */
const SMOOTHING_SETTINGS = [
  "event_smoothing",
  "taxonomic_rollup",
  "taxonomic_rollup_threshold",
  "independence_interval",
  "excluded_classes",
];

/** Check if any smoothing-relevant setting differs between old and new form data. */
function hasSmoothingChanges(
  before: SettingsFormData,
  after: SettingsFormData,
): boolean {
  for (const key of SMOOTHING_SETTINGS) {
    const a = before[key as keyof SettingsFormData];
    const b = after[key as keyof SettingsFormData];
    if (Array.isArray(a) && Array.isArray(b)) {
      if (a.length !== b.length || a.some((v, i) => v !== b[i])) return true;
    } else if (a !== b) {
      return true;
    }
  }
  return false;
}

/** Fetch observation and event snapshots for the current project settings. */
async function fetchStats(
  projectId: string,
  threshold: number,
  interval: number,
): Promise<{ observations: StatSnapshot; events: StatSnapshot }> {
  const [detectionCount, speciesStats, eventStats] = await Promise.all([
    projectsApi.getDetectionCount(projectId, threshold),
    projectsApi.getSpeciesStats(projectId, threshold),
    projectsApi.getIndependentEventStats(projectId, interval, threshold),
  ]);
  return {
    observations: { total: detectionCount.count, species: speciesStats },
    events: { total: eventStats.total, species: eventStats.species },
  };
}

export default function SettingsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const queryClient = useQueryClient();
  const [excludedClasses, setExcludedClasses] = useState<string[]>([]);
  const [speciesModalOpen, setSpeciesModalOpen] = useState(false);
  const [countryOpen, setCountryOpen] = useState(false);
  const [stateOpen, setStateOpen] = useState(false);
  const [showModelInfo, setShowModelInfo] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  // Model preparation state
  type PreparationStage = "form" | "preparing" | "error";
  type PreparingModelType = "detection" | "classification" | null;
  const [preparationStage, setPreparationStage] = useState<PreparationStage>("form");
  const [preparingTaskId, setPreparingTaskId] = useState<string | null>(null);
  const [preparationError, setPreparationError] = useState<string | null>(null);
  const [preparingModelType, setPreparingModelType] = useState<PreparingModelType>(null);

  // Unified save flow state
  const [saveJobId, setSaveJobId] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false); // shows modal before job ID is known
  const [saveResults, setSaveResults] = useState<SaveResults | null>(null);
  const [toastResults, setToastResults] = useState<SaveResults | null>(null);
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Stores before-stats + new settings values while reprocessing runs
  const pendingBeforeStats = useRef<{
    before: { observations: StatSnapshot; events: StatSnapshot };
    newThreshold: number;
    newInterval: number;
  } | null>(null);

  /** Show the custom save toast with auto-dismiss. */
  const showSaveToast = useCallback((results: SaveResults) => {
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    setToastResults(results);
    toastTimerRef.current = setTimeout(() => setToastResults(null), 5000);
  }, []);

  const dismissSaveToast = useCallback(() => {
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    setToastResults(null);
  }, []);

  // Fetch current project
  const { data: project, isLoading: projectLoading } = useQuery({
    queryKey: ["projects", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  });

  // Fetch available models
  const { data: detectionModels = [] } = useQuery({
    queryKey: ["models", "detection"],
    queryFn: () => modelsApi.listDetectionModels(),
  });

  const { data: classificationModels = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: () => modelsApi.listClassificationModels(),
  });

  const form = useForm<SettingsFormData>({
    resolver: zodResolver(settingsSchema),
    defaultValues: {
      detection_model_id: "MD5A-0-0",
      classification_model_id: "",
      excluded_classes: [],
      country_code: null,
      state_code: null,
      video_fps: 1.0,
      detection_threshold: 0.5,
      event_smoothing: true,
      taxonomic_rollup: true,
      taxonomic_rollup_threshold: 0.65,
      independence_interval: 1800,
    },
  });

  // Update form values when project loads
  useEffect(() => {
    if (project) {
      const values: SettingsFormData = {
        detection_model_id: project.detection_model_id,
        classification_model_id: project.classification_model_id || "",
        excluded_classes: project.excluded_classes || [],
        country_code: project.country_code || null,
        state_code: project.state_code || null,
        video_fps: project.video_fps,
        detection_threshold: project.detection_threshold,
        event_smoothing: project.event_smoothing,
        taxonomic_rollup: project.taxonomic_rollup,
        taxonomic_rollup_threshold: project.taxonomic_rollup_threshold,
        independence_interval: project.independence_interval,
      };
      form.reset(values);

      // WORKAROUND: Set state_code again after a tick to ensure the field is rendered
      // This handles the race condition where the state field is conditionally rendered
      // based on country_code === "USA"
      if (project.state_code) {
        setTimeout(() => {
          form.setValue("state_code", project.state_code);
        }, 0);
      }
    }
  }, [project, form]);

  // Watch model changes
  const detectionModelId = form.watch("detection_model_id");
  const classificationModelId = form.watch("classification_model_id");
  const countryCode = form.watch("country_code");

  // Check if current model is SpeciesNet
  const isSpeciesNet = classificationModelId?.toLowerCase().includes("speciesnet");

  // Fetch taxonomy for selected classification model (non-SpeciesNet only)
  const { data: taxonomy } = useQuery({
    queryKey: ["taxonomy", classificationModelId],
    queryFn: () => modelsApi.getTaxonomy(classificationModelId!),
    enabled: !!classificationModelId && !isSpeciesNet,
  });

  // Fetch locations for SpeciesNet models
  const { data: locations } = useQuery({
    queryKey: ["speciesnet-locations"],
    queryFn: () => modelsApi.getSpeciesNetLocations(),
    enabled: isSpeciesNet,
  });

  // Fetch detection model status
  const { data: detectionModelStatus } = useQuery({
    queryKey: ["model-status", detectionModelId],
    queryFn: () => modelsApi.getModelStatus(detectionModelId!),
    enabled: !!detectionModelId,
  });

  // Fetch classification model status
  const { data: classificationModelStatus } = useQuery({
    queryKey: ["model-status", classificationModelId],
    queryFn: () => modelsApi.getModelStatus(classificationModelId!),
    enabled: !!classificationModelId && classificationModelId !== "none",
  });

  // WebSocket progress tracking for model preparation
  const { progress, message } = useTaskProgress({
    taskId: preparingTaskId,
    onComplete: () => {
      // Refresh the correct model status based on which model was being prepared
      const modelIdToRefresh = preparingModelType === "detection" ? detectionModelId : classificationModelId;
      queryClient.invalidateQueries({ queryKey: ["model-status", modelIdToRefresh] });
      setPreparingTaskId(null);
      setPreparationStage("form");
      setPreparingModelType(null);
    },
    onError: (error) => {
      setPreparationError(error);
      setPreparationStage("error");
      setPreparingTaskId(null);
    },
  });

  // Initialize excludedClasses state when project loads
  useEffect(() => {
    if (project) {
      const savedExcluded = project.excluded_classes || [];
      setExcludedClasses(savedExcluded);
    }
  }, [project]);

  // Clear state_code when country changes away from USA
  useEffect(() => {
    if (countryCode !== "USA" && form.getValues("state_code")) {
      form.setValue("state_code", null, { shouldDirty: true });
    }
  }, [countryCode, form]);

  // Clear excluded_classes when classification model changes
  useEffect(() => {
    if (classificationModelId && taxonomy?.all_classes) {
      // Filter excluded_classes to only keep species that exist in the new model
      const currentExcluded = form.getValues("excluded_classes");
      const validExcluded = currentExcluded.filter(cls =>
        taxonomy.all_classes.includes(cls)
      );

      // Only update if some species were removed
      if (validExcluded.length !== currentExcluded.length) {
        form.setValue("excluded_classes", validExcluded, { shouldDirty: true });
        setExcludedClasses(validExcluded);
      }
    }
  }, [classificationModelId, taxonomy, form]);

  // Handler for detection model preparation
  const handlePrepareDetectionModel = async () => {
    if (!detectionModelId) return;

    try {
      setPreparationStage("preparing");
      setPreparingModelType("detection");
      const response = await modelsApi.prepareModel(detectionModelId);
      setPreparingTaskId(response.task_id);
    } catch (error: any) {
      setPreparationError(error.message || "Failed to start model preparation");
      setPreparationStage("error");
      setPreparingModelType(null);
    }
  };

  // Handler for classification model preparation
  const handlePrepareClassificationModel = async () => {
    if (!classificationModelId) return;

    try {
      setPreparationStage("preparing");
      setPreparingModelType("classification");
      const response = await modelsApi.prepareModel(classificationModelId);
      setPreparingTaskId(response.task_id);
    } catch (error: any) {
      setPreparationError(error.message || "Failed to start model preparation");
      setPreparationStage("error");
      setPreparingModelType(null);
    }
  };

  // Handler for canceling preparation
  const handleCancelPreparation = () => {
    setPreparingTaskId(null);
    setPreparationStage("form");
    setPreparingModelType(null);
  };

  // Handler for retrying after error
  const handleRetryPreparation = () => {
    setPreparationError(null);
    // Retry the same model type that failed
    if (preparingModelType === "detection") {
      handlePrepareDetectionModel();
    } else {
      handlePrepareClassificationModel();
    }
  };

  // Save-triggered reprocess progress
  const saveProgress = useTaskProgress({
    taskId: saveJobId,
    onComplete: async () => {
      setSaveJobId(null);
      setIsSaving(false);
      // Invalidate all project-related caches so every page (images, dashboard,
      // review, etc.) picks up the reprocessed species/annotations immediately.
      queryClient.invalidateQueries({ queryKey: ["postprocessing-status", projectId] });
      queryClient.invalidateQueries({ queryKey: ["projects", projectId] });
      queryClient.invalidateQueries({ queryKey: ["species-stats", projectId] });
      queryClient.invalidateQueries({ queryKey: ["detection-stats", projectId] });
      queryClient.invalidateQueries({ queryKey: ["observation-type-stats", projectId] });
      queryClient.invalidateQueries({ queryKey: ["files", projectId] });
      // Invalidate individual file detail queries (species annotations)
      queryClient.invalidateQueries({ queryKey: ["file"] });
      // Invalidate events (auto-regenerated after postprocessing)
      queryClient.invalidateQueries({ queryKey: ["events"] });
      queryClient.invalidateQueries({ queryKey: ["event-count"] });

      // Fetch after-stats now that reprocessing is done
      const pending = pendingBeforeStats.current;
      if (!pending || !projectId) {
        pendingBeforeStats.current = null;
        toast.success("Settings saved!");
        return;
      }

      try {
        const afterStats = await fetchStats(
          projectId, pending.newThreshold, pending.newInterval,
        );
        const results: SaveResults = {
          observations: { before: pending.before.observations, after: afterStats.observations },
          events: { before: pending.before.events, after: afterStats.events },
        };
        pendingBeforeStats.current = null;
        showSaveToast(results);
      } catch {
        pendingBeforeStats.current = null;
        toast.success("Settings saved!");
      }
    },
    onError: () => {
      setSaveJobId(null);
      setIsSaving(false);
      pendingBeforeStats.current = null;
      toast.error("Reprocessing failed");
    },
  });

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: (data: ProjectUpdate) => projectsApi.update(projectId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      queryClient.invalidateQueries({ queryKey: ["projects", projectId] });
      queryClient.invalidateQueries({ queryKey: ["postprocessing-status", projectId] });
      // Reset form dirty state
      form.reset(form.getValues());
    },
    onError: (error: Error) => {
      form.setError("root", {
        message: error.message || "Failed to update project settings",
      });
    },
  });

  const onSubmit = async (data: SettingsFormData) => {
    if (!projectId) return;

    // Validate that at least one species remains included
    if (taxonomy && !isSpeciesNet) {
      const allCount = taxonomy.all_classes?.length || 0;
      if (allCount > 0 && data.excluded_classes.length >= allCount) {
        form.setError("excluded_classes", {
          message: "At least one species must remain included",
        });
        return;
      }
    }

    try {
      const currentValues = form.formState.defaultValues as SettingsFormData;
      const willReprocess = hasSmoothingChanges(currentValues, data);

      // Show progress modal immediately if reprocessing will happen
      if (willReprocess) {
        setIsSaving(true);
      }

      // 1. Start fetching before-stats in the background (don't await yet)
      const beforeStatsPromise = fetchStats(
        projectId,
        currentValues.detection_threshold,
        currentValues.independence_interval,
      );

      // 2. Save settings
      await updateMutation.mutateAsync(data);

      // 3. If smoothing settings changed, trigger reprocess
      if (willReprocess) {
        const status = await projectsApi.getPostprocessingStatus(projectId);
        if (status.has_classifications) {
          // Trigger reprocess and connect WebSocket for progress tracking
          const reprocessResult = await projectsApi.reprocess(projectId);
          setSaveJobId(reprocessResult.job_id);

          // Await before-stats (likely already resolved by now)
          const beforeStats = await beforeStatsPromise;
          pendingBeforeStats.current = {
            before: beforeStats,
            newThreshold: data.detection_threshold,
            newInterval: data.independence_interval,
          };
          return; // Progress modal takes over; toast shown in onComplete
        }
        // No classifications to reprocess — close the modal
        setIsSaving(false);
      }

      // 4. No reprocess needed — await before-stats and fetch after-stats
      const beforeStats = await beforeStatsPromise;
      const afterStats = await fetchStats(
        projectId, data.detection_threshold, data.independence_interval,
      );

      const results: SaveResults = {
        observations: { before: beforeStats.observations, after: afterStats.observations },
        events: { before: beforeStats.events, after: afterStats.events },
      };

      showSaveToast(results);
    } catch (error: any) {
      setIsSaving(false);
      toast.error(error.message || "Failed to save settings");
    }
  };

  const handleReset = () => {
    if (project) {
      form.reset({
        detection_model_id: project.detection_model_id,
        classification_model_id: project.classification_model_id || "",
        excluded_classes: project.excluded_classes || [],
        country_code: project.country_code || null,
        state_code: project.state_code || null,
        video_fps: project.video_fps,
        detection_threshold: project.detection_threshold,
        event_smoothing: project.event_smoothing,
        taxonomic_rollup: project.taxonomic_rollup,
        taxonomic_rollup_threshold: project.taxonomic_rollup_threshold,
        independence_interval: project.independence_interval,
      });
      setExcludedClasses(project.excluded_classes || []);
    }
  };

  if (projectLoading) {
    return (
      <div className="p-8">
        <div className="text-muted-foreground">Loading project settings...</div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="p-8">
        <div className="text-destructive">Project not found</div>
      </div>
    );
  }

  const isDirty = form.formState.isDirty;

  return (
    <div className="p-8 bg-gradient-to-br from-slate-50 to-slate-100 min-h-screen">
      <div className="mx-auto max-w-5xl space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Project settings</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Configure AI models, species selection, and analysis parameters
          </p>
        </div>

        {/* Settings form */}
        <TooltipProvider>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6" key={project?.id}>
            {/* Card 1: Models */}
            <Card>
              <CardHeader>
                <CardTitle>Models</CardTitle>
                <CardDescription>
                  Models used to detect objects and classify species. Changes apply to new analyses only and do not reprocess existing results.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-0 divide-y">
                {/* Detection Model */}
                <FormField
                  control={form.control}
                  name="detection_model_id"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Detection model</FormLabel>
                        <FormDescription className="text-sm">
                          Used to find animals, people, and vehicles.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <div className="flex gap-2 items-stretch">
                          <Select key={field.value} onValueChange={field.onChange} value={field.value}>
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue placeholder="Select detection model">
                                  {field.value && (() => {
                                    const selectedModel = detectionModels.find(
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
                              {detectionModels.map((model) => (
                                <SelectItem key={model.model_id} value={model.model_id}>
                                  {model.emoji} {model.friendly_name}
                                  {model.description_short && (
                                    <>
                                      <br />
                                      <span className="text-xs text-muted-foreground">{model.description_short}</span>
                                    </>
                                  )}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className="self-center">
                                <Button
                                  type="button"
                                  variant="outline"
                                  className="px-3"
                                  onClick={() => {
                                    if (field.value) {
                                      setSelectedModelId(field.value);
                                      setShowModelInfo(true);
                                    }
                                  }}
                                  disabled={!field.value}
                                >
                                  <InfoIcon className="h-4 w-4" />
                                </Button>
                              </span>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>
                                {field.value
                                  ? "View model information"
                                  : "Select a detection model to view details"}
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <FormMessage />

                        {/* Model Status Badge */}
                        {field.value && detectionModelStatus && (
                          <ModelStatusBadge
                            status={detectionModelStatus}
                            onPrepare={handlePrepareDetectionModel}
                            isPreparing={preparationStage === "preparing" && preparingModelType === "detection"}
                          />
                        )}
                      </div>
                    </div>
                  )}
                />

                {/* Classification Model */}
                <FormField
                  control={form.control}
                  name="classification_model_id"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Classification model</FormLabel>
                        <FormDescription className="text-sm">
                          Used to identify species for detected animals.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <div className="flex gap-2 items-stretch">
                          <Select
                            key={field.value}
                            onValueChange={field.onChange}
                            value={field.value}
                            defaultValue={field.value}
                          >
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue placeholder="Select classification model">
                                  {field.value && (() => {
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
                              {classificationModels
                                .filter((model) => model.model_id !== "none")
                                .map((model) => (
                                  <SelectItem key={model.model_id} value={model.model_id}>
                                    {model.emoji} {model.friendly_name}
                                    {model.description_short && (
                                      <>
                                        <br />
                                        <span className="text-xs text-muted-foreground">{model.description_short}</span>
                                      </>
                                    )}
                                  </SelectItem>
                                ))}
                            </SelectContent>
                          </Select>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className="self-center">
                                <Button
                                  type="button"
                                  variant="outline"
                                  className="px-3"
                                  onClick={() => {
                                    if (field.value) {
                                      setSelectedModelId(field.value);
                                      setShowModelInfo(true);
                                    }
                                  }}
                                  disabled={!field.value}
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

                        {/* Model Status Badge */}
                        {field.value && classificationModelStatus && (
                          <ModelStatusBadge
                            status={classificationModelStatus}
                            onPrepare={handlePrepareClassificationModel}
                            isPreparing={preparationStage === "preparing" && preparingModelType === "classification"}
                          />
                        )}
                      </div>
                    </div>
                  )}
                />

              </CardContent>
            </Card>

            {/* Card 2: Geographic location (SpeciesNet) OR Species selection (other models) */}
            {classificationModelId && isSpeciesNet && locations && (
              <Card>
                <CardHeader>
                  <CardTitle>Geographic location</CardTitle>
                  <CardDescription>
                    Select the location used for SpeciesNet predictions. Changes apply to new analyses only and do not reprocess existing results.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-0 divide-y">
                  {/* Country Selection */}
                  <FormField
                    control={form.control}
                    name="country_code"
                    render={({ field }) => (
                      <div className="grid grid-cols-2 gap-8 py-6">
                        <div className="space-y-1">
                          <FormLabel>Country</FormLabel>
                          <FormDescription className="text-sm">
                            Select the country where your camera traps are located.
                          </FormDescription>
                        </div>
                        <div className="space-y-2">
                          <Popover open={countryOpen} onOpenChange={setCountryOpen}>
                            <PopoverTrigger asChild>
                              <FormControl>
                                <Button
                                  variant="outline"
                                  role="combobox"
                                  className={cn(
                                    "w-full justify-between",
                                    !field.value && "text-muted-foreground"
                                  )}
                                >
                                  {field.value
                                    ? Object.entries(locations.countries).find(
                                        ([_, code]) => code === field.value
                                      )?.[0]
                                    : "Select country"}
                                  <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                                </Button>
                              </FormControl>
                            </PopoverTrigger>
                            <PopoverContent className="w-[400px] p-0">
                              <Command>
                                <CommandInput placeholder="Search countries..." />
                                <CommandList>
                                  <CommandEmpty>No country found.</CommandEmpty>
                                  <CommandGroup>
                                    {Object.entries(locations.countries).map(([name, code]) => (
                                      <CommandItem
                                        key={code}
                                        value={name}
                                        onSelect={() => {
                                          form.setValue("country_code", code, { shouldDirty: true });
                                          setCountryOpen(false);
                                        }}
                                      >
                                        <Check
                                          className={cn(
                                            "mr-2 h-4 w-4",
                                            field.value === code ? "opacity-100" : "opacity-0"
                                          )}
                                        />
                                        {name}
                                      </CommandItem>
                                    ))}
                                  </CommandGroup>
                                </CommandList>
                              </Command>
                            </PopoverContent>
                          </Popover>
                          <FormMessage />
                        </div>
                      </div>
                    )}
                  />

                  {/* State Selection (USA only) */}
                  {countryCode === "USA" && (
                    <FormField
                      control={form.control}
                      name="state_code"
                      render={({ field }) => (
                        <div className="grid grid-cols-2 gap-8 py-6">
                          <div className="space-y-1">
                            <FormLabel>State</FormLabel>
                            <FormDescription className="text-sm">
                              Select a US state for more specific predictions.
                            </FormDescription>
                          </div>
                          <div className="space-y-2">
                            <Popover open={stateOpen} onOpenChange={setStateOpen}>
                              <PopoverTrigger asChild>
                                <FormControl>
                                  <Button
                                    variant="outline"
                                    role="combobox"
                                    className={cn(
                                      "w-full justify-between",
                                      !field.value && "text-muted-foreground"
                                    )}
                                  >
                                    {field.value
                                      ? Object.entries(locations.us_states).find(
                                          ([_, code]) => code === field.value
                                        )?.[0]
                                      : "Select state"}
                                    <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                                  </Button>
                                </FormControl>
                              </PopoverTrigger>
                              <PopoverContent className="w-[400px] p-0">
                                <Command>
                                  <CommandInput placeholder="Search states..." />
                                  <CommandList>
                                    <CommandEmpty>No state found.</CommandEmpty>
                                    <CommandGroup>
                                      {Object.entries(locations.us_states).map(([name, code]) => (
                                        <CommandItem
                                          key={code}
                                          value={name}
                                          onSelect={() => {
                                            form.setValue("state_code", code, { shouldDirty: true });
                                            setStateOpen(false);
                                          }}
                                        >
                                          <Check
                                            className={cn(
                                              "mr-2 h-4 w-4",
                                              field.value === code ? "opacity-100" : "opacity-0"
                                            )}
                                          />
                                          {name}
                                        </CommandItem>
                                      ))}
                                    </CommandGroup>
                                  </CommandList>
                                </Command>
                              </PopoverContent>
                            </Popover>
                            <FormMessage />
                          </div>
                        </div>
                      )}
                    />
                  )}
                </CardContent>
              </Card>
            )}

            {classificationModelId && !isSpeciesNet && taxonomy && (
              <Card>
                <CardHeader>
                  <CardTitle>Species selection</CardTitle>
                  <CardDescription>
                    Control which species can be predicted
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-8">
                    <div className="space-y-1">
                      <FormLabel>Species selection</FormLabel>
                      <FormDescription className="text-sm">
                        Limit predictions to species expected in your project area to reduce false positives.
                      </FormDescription>
                    </div>
                    <div className="space-y-1">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => setSpeciesModalOpen(true)}
                        className="w-full min-h-14 flex flex-col items-start justify-center gap-1 text-left"
                      >
                        <div className="flex items-center gap-2">
                          <ListTodo className="h-4 w-4" />
                          <span>Select species</span>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          Currently included {(taxonomy.all_classes?.length || 0) - excludedClasses.length} of {taxonomy.all_classes?.length || 0}
                        </span>
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Card 3: Analysis and counting */}
            <Card>
              <CardHeader>
                <CardTitle>Analysis and counting</CardTitle>
                <CardDescription>
                  Control how detections are filtered, grouped, and aggregated. Changes apply to all analyses (past and future) and affect how data is interpreted, not the underlying detections.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-0 divide-y">
                {/* Video Frame Rate */}
                <FormField
                  control={form.control}
                  name="video_fps"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Video frame rate</FormLabel>
                        <FormDescription className="text-sm">
                          How many frames per second to extract from videos for detection. Higher values find more but take longer. Applies to new analyses only.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <Select
                          key={String(field.value)}
                          value={String(field.value)}
                          onValueChange={(val) => field.onChange(parseFloat(val))}
                        >
                          <FormControl>
                            <SelectTrigger className="max-w-xs">
                              <SelectValue />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {VIDEO_FPS_OPTIONS.map((opt) => (
                              <SelectItem key={opt.value} value={opt.value}>
                                {opt.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </div>
                    </div>
                  )}
                />

                {/* Detection Threshold */}
                <FormField
                  control={form.control}
                  name="detection_threshold"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Detection confidence threshold</FormLabel>
                        <FormDescription className="text-sm">
                          Hide and exclude detections below this value. Applies to existing and new analyses.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Slider
                            min={0.1}
                            max={1.0}
                            step={0.01}
                            value={[field.value]}
                            onValueChange={(vals) => field.onChange(vals[0])}
                            className="flex-1 mr-4"
                          />
                          <span className="text-sm font-medium min-w-[3rem] text-right">{field.value.toFixed(2)}</span>
                        </div>
                        <FormMessage />
                      </div>
                    </div>
                  )}
                />

                {/* Independence Interval */}
                <FormField
                  control={form.control}
                  name="independence_interval"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Independence interval</FormLabel>
                        <FormDescription className="text-sm">
                          Group detections into one event when they occur within this time gap.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <Select
                          key={String(field.value)}
                          value={String(field.value)}
                          onValueChange={(val) => field.onChange(parseInt(val))}
                        >
                          <FormControl>
                            <SelectTrigger className="max-w-xs">
                              <SelectValue />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {INDEPENDENCE_INTERVAL_OPTIONS.map((opt) => (
                              <SelectItem key={opt.value} value={opt.value}>
                                {opt.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </div>
                    </div>
                  )}
                />

                {/* Event Smoothing */}
                <FormField
                  control={form.control}
                  name="event_smoothing"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Event smoothing</FormLabel>
                        <FormDescription className="text-sm">
                          Reduce noisy labels within an event by averaging predictions.
                        </FormDescription>
                      </div>
                      <div className="flex items-center">
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </div>
                    </div>
                  )}
                />

                {/* Taxonomic Rollup */}
                <FormField
                  control={form.control}
                  name="taxonomic_rollup"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Taxonomic rollup</FormLabel>
                        <FormDescription className="text-sm">
                          If the model's confidence at the species level is below 0.65, it rolls up to the next higher taxonomic level at which the summed confidence reaches 0.65.
                        </FormDescription>
                      </div>
                      <div className="flex items-center">
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </div>
                    </div>
                  )}
                />

              </CardContent>
            </Card>

            {form.formState.errors.root && (
              <p className="text-sm font-medium text-destructive">
                {form.formState.errors.root.message}
              </p>
            )}
            {form.formState.errors.excluded_classes && (
              <p className="text-sm font-medium text-destructive">
                {form.formState.errors.excluded_classes.message}
              </p>
            )}

            {/* Action buttons */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => {
                    // Use setValue (not reset) so the saved project values stay as the
                    // dirty-check baseline. This way isDirty becomes true and the user
                    // must press "Save changes" to persist the defaults.
                    form.setValue("detection_model_id", "MD5A-0-0", { shouldDirty: true });
                    form.setValue("video_fps", 1.0, { shouldDirty: true });
                    form.setValue("detection_threshold", 0.5, { shouldDirty: true });
                    form.setValue("event_smoothing", true, { shouldDirty: true });
                    form.setValue("taxonomic_rollup", true, { shouldDirty: true });
                    form.setValue("taxonomic_rollup_threshold", 0.65, { shouldDirty: true });
                    form.setValue("independence_interval", 1800, { shouldDirty: true });
                  }}
                  disabled={updateMutation.isPending}
                >
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Restore defaults
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleReset}
                  disabled={!isDirty || updateMutation.isPending}
                >
                  <Undo2 className="h-4 w-4 mr-2" />
                  Reset changes
                </Button>
              </div>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span>
                    <Button
                      type="submit"
                      disabled={
                        !isDirty ||
                        updateMutation.isPending ||
                        !!saveJobId ||
                        detectionModelStatus?.status !== "ready" ||
                        classificationModelStatus?.status !== "ready"
                      }
                    >
                      <Save className="h-4 w-4 mr-2" />
                      {updateMutation.isPending ? "Saving..." : "Save changes"}
                    </Button>
                  </span>
                </TooltipTrigger>
                {(detectionModelStatus?.status !== "ready" || classificationModelStatus?.status !== "ready") && (
                  <TooltipContent>
                    <p>Model needs preparing first</p>
                  </TooltipContent>
                )}
              </Tooltip>
            </div>
          </form>
        </Form>

        </TooltipProvider>

        {/* Model Info Sheet */}
        <ModelInfoSheet
          modelId={selectedModelId}
          open={showModelInfo}
          onOpenChange={setShowModelInfo}
        />

        {/* Species Selection Modal */}
        {classificationModelId && taxonomy && (
          <SpeciesSelectionModal
            modelId={classificationModelId}
            excludedClasses={excludedClasses}
            onExclusionChange={(classes) => {
              setExcludedClasses(classes);
              form.setValue("excluded_classes", classes, { shouldDirty: true });
            }}
            open={speciesModalOpen}
            onOpenChange={setSpeciesModalOpen}
            totalSpeciesCount={taxonomy.all_classes?.length || 0}
          />
        )}

        {/* Model Preparation Dialog */}
        <Dialog open={preparationStage === "preparing"} onOpenChange={(open) => !open && handleCancelPreparation()}>
          <DialogContent className="max-w-xl">
            {preparingModelType === "detection" && detectionModels.find((m) => m.model_id === detectionModelId) && (
              <ModelPreparationView
                modelName={detectionModels.find((m) => m.model_id === detectionModelId)!.friendly_name}
                modelEmoji={detectionModels.find((m) => m.model_id === detectionModelId)!.emoji}
                progress={progress}
                message={message}
                onCancel={handleCancelPreparation}
              />
            )}
            {preparingModelType === "classification" && classificationModels.find((m) => m.model_id === classificationModelId) && (
              <ModelPreparationView
                modelName={classificationModels.find((m) => m.model_id === classificationModelId)!.friendly_name}
                modelEmoji={classificationModels.find((m) => m.model_id === classificationModelId)!.emoji}
                progress={progress}
                message={message}
                onCancel={handleCancelPreparation}
              />
            )}
          </DialogContent>
        </Dialog>

        {/* Applying Settings Progress Dialog */}
        <Dialog open={isSaving || !!saveJobId}>
          <DialogContent className="max-w-md" onInteractOutside={(e) => e.preventDefault()}>
            <div className="flex flex-col items-center gap-4 py-4">
              <div className="rounded-full bg-primary/10 p-3">
                <RefreshCw className="h-6 w-6 text-primary animate-spin" />
              </div>
              <div className="text-center space-y-2">
                <h3 className="font-semibold text-lg">Applying settings</h3>
                <p className="text-sm text-muted-foreground">
                  {saveProgress.message || (isSaving && !saveJobId ? "Saving settings..." : "Starting...")}
                </p>
              </div>
              {saveProgress.progress > 0 && saveProgress.progress < 1 && (
                <div className="w-full space-y-1">
                  <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary rounded-full transition-all duration-300"
                      style={{ width: `${Math.round(saveProgress.progress * 100)}%` }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground text-center">
                    {Math.round(saveProgress.progress * 100)}%
                  </p>
                </div>
              )}
            </div>
          </DialogContent>
        </Dialog>

        {/* Save toast */}
        {toastResults && (
          <div
            className="fixed bottom-6 right-6 z-50 flex items-center gap-3 rounded-lg border border-gray-200 bg-white px-4 py-3 shadow-lg"
            style={{ animation: "toast-slide-up 0.2s ease-out" }}
          >
            <Check className="h-4 w-4 flex-shrink-0 text-primary" />
            <span className="text-sm">
              Settings saved!{" "}
              <button
                onClick={() => {
                  setSaveResults(toastResults);
                  dismissSaveToast();
                }}
                className="font-medium text-primary hover:underline"
              >
                See effect
              </button>
            </span>
            <button
              onClick={dismissSaveToast}
              className="ml-1 text-gray-400 hover:text-gray-600"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}

        {/* Save Results Modal */}
        {saveResults && (
          <SaveResultsModal
            open={saveResults !== null}
            onOpenChange={(open) => !open && setSaveResults(null)}
            results={saveResults}
          />
        )}

        {/* Model Preparation Error Dialog */}
        <Dialog open={preparationStage === "error"} onOpenChange={(open) => !open && setPreparationStage("form")}>
          <DialogContent className="max-w-xl">
            <ModelPreparationErrorView
              errorMessage={preparationError || "Unknown error occurred"}
              onRetry={handleRetryPreparation}
              onCancel={() => setPreparationStage("form")}
            />
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
