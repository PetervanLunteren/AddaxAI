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
import { Save, RotateCcw, Undo2, Check, ListTodo, InfoIcon, RefreshCw, X } from "lucide-react";
import { toast } from "sonner";
import { projectsApi, type ProjectUpdate } from "../api/projects";
import { modelsApi } from "../api/models";
import { DiagnosticReportButton } from "../components/diagnostics/DiagnosticReportButton";
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
import { ReEmbedModal } from "../components/projects/ReEmbedModal";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../components/ui/alert-dialog";

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
import { ClassificationModelGroupedItems } from "../components/models/ClassificationModelGroupedItems";
import { BatchSizeRow } from "../components/analyses/BatchSizeRow";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormLabel,
  FormMessage,
} from "../components/ui/form";
import { TimezoneSelect } from "../components/ui/timezone-select";
import { invalidateProjectData } from "../lib/invalidate-project";

const settingsSchema = z.object({
  detection_model_id: z.string().min(1, "Detection model is required"),
  classification_model_id: z.string().nullable(),
  embedding_model_id: z.string().optional().nullable(),
  excluded_classes: z.array(z.string()),
  country_code: z.string().optional().nullable(),
  state_code: z.string().optional().nullable(),
  // Empty string means "Auto (derive from site location)".
  timezone: z.string(),
  video_fps: z.number().min(0.1).max(10),
  detection_threshold: z.number().min(0).max(1),
  event_smoothing: z.boolean(),
  smoothing_strength: z.enum(["mild", "normal", "aggressive"]),
  taxonomic_rollup: z.boolean(),
  taxonomic_rollup_threshold: z.number().min(0.1).max(1.0),
  independence_interval: z.number().min(0),
  // null = use the per-pipeline default; integer = user override
  detection_batch_size: z.number().int().min(1).max(256).nullable(),
  classification_batch_size: z.number().int().min(1).max(256).nullable(),
  embedding_batch_size: z.number().int().min(1).max(256).nullable(),
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
  "smoothing_strength",
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

// BatchSizeRow lives in components/analyses/BatchSizeRow.tsx and is
// shared between this page and the folder-run setup step. Imported above.

/** Fetch observation and event snapshots for the current project settings. */
async function fetchStats(
  projectId: string,
  threshold: number,
  interval: number,
): Promise<{
  observations: StatSnapshot;
  independent_observations: StatSnapshot;
  events: StatSnapshot;
}> {
  const [detectionCount, labelStats, indepObsStats, eventStats] = await Promise.all([
    projectsApi.getDetectionCount(projectId, threshold),
    projectsApi.getLabelStats(projectId, threshold),
    projectsApi.getIndependentObservationStats(projectId, interval, threshold),
    projectsApi.getIndependentEventStats(projectId, interval, threshold),
  ]);
  return {
    observations: { total: detectionCount.count, labels: labelStats },
    independent_observations: {
      total: indepObsStats.total,
      labels: indepObsStats.labels,
    },
    events: { total: eventStats.total, labels: eventStats.labels },
  };
}

export default function SettingsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const queryClient = useQueryClient();
  const [excludedClasses, setExcludedClasses] = useState<string[]>([]);
  const [labelSelectionModalOpen, setLabelSelectionModalOpen] = useState(false);
  const [showModelInfo, setShowModelInfo] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  // Model preparation state
  type PreparationStage = "form" | "preparing" | "error";
  type PreparingModelType = "detection" | "classification" | "embedding" | null;
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
    before: {
      observations: StatSnapshot;
      independent_observations: StatSnapshot;
      events: StatSnapshot;
    };
    newThreshold: number;
    newInterval: number;
  } | null>(null);

  // Classification model removal confirmation
  const [removeClsConfirmOpen, setRemoveClsConfirmOpen] = useState(false);

  // Re-embed confirmation + progress state
  const [reEmbedConfirmOpen, setReEmbedConfirmOpen] = useState(false);
  const [reEmbedJobId, setReEmbedJobId] = useState<string | null>(null);
  const [reEmbedDetectionCount, setReEmbedDetectionCount] = useState(0);
  const pendingFormData = useRef<SettingsFormData | null>(null);

  /** Show the save toast. When before/after stats are identical the
   * change didn't touch any counts (e.g., switching to a new model
   * that only affects future analyses), so fall back to a plain
   * "Settings saved!" without the "See effect" link that would open
   * an empty diff modal. Equal totals aren't enough: relabel /
   * rollup changes can shuffle counts between labels and net to zero
   * at the aggregate level, so also compare per-label counts. */
  const showSaveToast = useCallback((results: SaveResults) => {
    const snapshotsDiffer = (
      before: { total: number; labels: { label: string; count: number }[] },
      after: { total: number; labels: { label: string; count: number }[] },
    ): boolean => {
      if (before.total !== after.total) return true;
      const beforeByLabel = new Map(
        before.labels.map((l) => [l.label, l.count]),
      );
      const afterByLabel = new Map(after.labels.map((l) => [l.label, l.count]));
      const allLabels = new Set([
        ...beforeByLabel.keys(),
        ...afterByLabel.keys(),
      ]);
      for (const label of allLabels) {
        if ((beforeByLabel.get(label) ?? 0) !== (afterByLabel.get(label) ?? 0)) {
          return true;
        }
      }
      return false;
    };

    const changed =
      snapshotsDiffer(results.observations.before, results.observations.after) ||
      snapshotsDiffer(
        results.independent_observations.before,
        results.independent_observations.after,
      ) ||
      snapshotsDiffer(results.events.before, results.events.after);
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    if (!changed) {
      setToastResults(null);
      toast.success("Settings saved!");
      return;
    }
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

  const { data: embeddingModels = [] } = useQuery({
    queryKey: ["models", "embedding"],
    queryFn: () => modelsApi.listEmbeddingModels(),
  });

  const form = useForm<SettingsFormData>({
    resolver: zodResolver(settingsSchema),
    defaultValues: {
      detection_model_id: "MD5A-0-0",
      classification_model_id: null,
      embedding_model_id: "DINOV2-VITB14",
      excluded_classes: [],
      country_code: null,
      state_code: null,
      timezone: "",
      video_fps: 1.0,
      detection_threshold: 0.5,
      event_smoothing: true,
      smoothing_strength: "normal" as const,
      taxonomic_rollup: true,
      taxonomic_rollup_threshold: 0.65,
      independence_interval: 1800,
      detection_batch_size: null,
      classification_batch_size: null,
      embedding_batch_size: null,
    },
  });

  // Update form values when project loads
  useEffect(() => {
    if (project) {
      const values: SettingsFormData = {
        detection_model_id: project.detection_model_id,
        classification_model_id: project.classification_model_id ?? null,
        embedding_model_id: project.embedding_model_id || "none",
        excluded_classes: project.excluded_classes || [],
        country_code: project.country_code || null,
        state_code: project.state_code || null,
        timezone: project.timezone ?? "",
        video_fps: project.video_fps,
        detection_threshold: project.detection_threshold,
        event_smoothing: project.event_smoothing,
        smoothing_strength: (project.smoothing_strength || "normal") as "mild" | "normal" | "aggressive",
        taxonomic_rollup: project.taxonomic_rollup,
        taxonomic_rollup_threshold: project.taxonomic_rollup_threshold,
        independence_interval: project.independence_interval,
        detection_batch_size: project.detection_batch_size ?? null,
        classification_batch_size: project.classification_batch_size ?? null,
        embedding_batch_size: project.embedding_batch_size ?? null,
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
  const embeddingModelId = form.watch("embedding_model_id");
  const countryCode = form.watch("country_code");

  // Check if a classification model is selected
  const hasClassificationModel = !!classificationModelId && classificationModelId !== "none";

  // Fetch taxonomy for selected classification model
  const { data: taxonomy } = useQuery({
    queryKey: ["taxonomy", classificationModelId],
    queryFn: () => modelsApi.getTaxonomy(classificationModelId!),
    enabled: !!classificationModelId && classificationModelId !== "none",
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

  // Fetch embedding model status
  const { data: embeddingModelStatus } = useQuery({
    queryKey: ["model-status", embeddingModelId],
    queryFn: () => modelsApi.getModelStatus(embeddingModelId!),
    enabled: !!embeddingModelId && embeddingModelId !== "none",
  });

  // WebSocket progress tracking for model preparation
  const { progress, message } = useTaskProgress({
    taskId: preparingTaskId,
    onComplete: () => {
      // Refresh the correct model status based on which model was being prepared
      const modelIdToRefresh = preparingModelType === "detection" ? detectionModelId
        : preparingModelType === "embedding" ? embeddingModelId
        : classificationModelId;
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
      // Filter excluded_classes to only keep classes that exist in the new model
      const currentExcluded = form.getValues("excluded_classes");
      const validExcluded = currentExcluded.filter(cls =>
        taxonomy.all_classes.includes(cls)
      );

      // Only update if some classes were removed
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

  // Handler for embedding model preparation
  const handlePrepareEmbeddingModel = async () => {
    if (!embeddingModelId) return;

    try {
      setPreparationStage("preparing");
      setPreparingModelType("embedding");
      const response = await modelsApi.prepareModel(embeddingModelId);
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
    } else if (preparingModelType === "embedding") {
      handlePrepareEmbeddingModel();
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
      // Blanket invalidate so every page (images, dashboard, review,
      // insights) picks up the reprocessed labels/annotations
      // immediately.
      if (projectId) {
        invalidateProjectData(queryClient, projectId);
      }
      queryClient.invalidateQueries({ queryKey: ["postprocessing-status", projectId] });

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
          observations: {
            before: pending.before.observations,
            after: afterStats.observations,
          },
          independent_observations: {
            before: pending.before.independent_observations,
            after: afterStats.independent_observations,
          },
          events: {
            before: pending.before.events,
            after: afterStats.events,
          },
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
      // Some deployments may have been touched before the error; make
      // sure every page reflects whatever state the DB is actually in.
      if (projectId) {
        invalidateProjectData(queryClient, projectId);
      }
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

    // Validate that at least one label remains included
    if (taxonomy) {
      const allCount = taxonomy.all_classes?.length || 0;
      if (allCount > 0 && data.excluded_classes.length >= allCount) {
        form.setError("excluded_classes", {
          message: "At least one label must remain included",
        });
        return;
      }
    }

    // Intercept embedding model change — confirm only when replacing an existing model
    // and there are detections to re-embed. Skip for "none" → model (first-time enable).
    const currentValues = form.formState.defaultValues as SettingsFormData;
    const oldEmbModel = currentValues.embedding_model_id || "none";
    const newEmbModel = data.embedding_model_id || "none";
    if (oldEmbModel !== newEmbModel && oldEmbModel !== "none" && newEmbModel !== "none") {
      let count = 0;
      try {
        ({ count } = await projectsApi.getDetectionCount(projectId, 0));
      } catch { /* fall through with 0 */ }

      if (count > 0) {
        pendingFormData.current = data;
        setReEmbedDetectionCount(count);
        setReEmbedConfirmOpen(true);
        return; // Wait for user confirmation
      }
    }

    // No confirmation needed — save and trigger re-embed inline if model changed
    await runSaveFlow(data);
    if (oldEmbModel !== newEmbModel && newEmbModel !== "none") {
      try {
        const result = await projectsApi.reEmbed(projectId);
        if (result.job_id) setReEmbedJobId(result.job_id);
      } catch { /* non-fatal */ }
    }
  };

  /** Core save flow — extracted so confirmation handlers can call it too. */
  const runSaveFlow = async (data: SettingsFormData) => {
    if (!projectId) return;

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

      // 2. Save settings. Empty timezone means "Auto" — send null so the
      // backend leaves it unset and keeps deriving it from site coords.
      await updateMutation.mutateAsync({
        ...data,
        timezone: data.timezone || null,
      });

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
        observations: {
          before: beforeStats.observations,
          after: afterStats.observations,
        },
        independent_observations: {
          before: beforeStats.independent_observations,
          after: afterStats.independent_observations,
        },
        events: {
          before: beforeStats.events,
          after: afterStats.events,
        },
      };

      showSaveToast(results);
    } catch (error: any) {
      setIsSaving(false);
      toast.error(error.message || "Failed to save settings");
    }
  };

  /** User confirmed re-embedding — save settings, then trigger re-embed. */
  const handleConfirmReEmbed = async () => {
    setReEmbedConfirmOpen(false);
    const data = pendingFormData.current;
    pendingFormData.current = null;
    if (!data || !projectId) return;

    // Run the full save flow (saves all settings including new embedding model)
    await runSaveFlow(data);

    // Trigger re-embedding and open progress modal
    try {
      const reEmbedResult = await projectsApi.reEmbed(projectId);
      if (reEmbedResult.job_id) {
        setReEmbedJobId(reEmbedResult.job_id);
      } else {
        toast.success(reEmbedResult.message);
      }
    } catch (error: any) {
      toast.error(error.message || "Failed to start re-embedding");
    }
  };

  /** User declined re-embedding — revert embedding model, save other settings. */
  const handleRevertReEmbed = async () => {
    setReEmbedConfirmOpen(false);
    const data = pendingFormData.current;
    pendingFormData.current = null;
    if (!data) return;

    // Revert embedding model to old value, save everything else
    const currentValues = form.formState.defaultValues as SettingsFormData;
    const revertedData = { ...data, embedding_model_id: currentValues.embedding_model_id };
    form.setValue("embedding_model_id", currentValues.embedding_model_id || "none");
    await runSaveFlow(revertedData);
  };

  const handleReset = () => {
    if (project) {
      form.reset({
        detection_model_id: project.detection_model_id,
        classification_model_id: project.classification_model_id ?? null,
        embedding_model_id: project.embedding_model_id || "none",
        excluded_classes: project.excluded_classes || [],
        country_code: project.country_code || null,
        state_code: project.state_code || null,
        timezone: project.timezone ?? "",
        video_fps: project.video_fps,
        detection_threshold: project.detection_threshold,
        event_smoothing: project.event_smoothing,
        smoothing_strength: (project.smoothing_strength || "normal") as "mild" | "normal" | "aggressive",
        taxonomic_rollup: project.taxonomic_rollup,
        taxonomic_rollup_threshold: project.taxonomic_rollup_threshold,
        independence_interval: project.independence_interval,
        detection_batch_size: project.detection_batch_size ?? null,
        classification_batch_size: project.classification_batch_size ?? null,
        embedding_batch_size: project.embedding_batch_size ?? null,
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
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Project settings</h1>
              <p className="text-sm text-muted-foreground">
                Configure AI models, labels, and analysis parameters
              </p>
            </div>
            <DiagnosticReportButton />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 pb-20 sm:px-6 lg:px-8 space-y-6">
        {/* Settings form */}
        <TooltipProvider>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6" key={project?.id}>
            {/* Card 0: Project info */}
            <Card>
              <CardHeader>
                <CardTitle>Project info</CardTitle>
                <CardDescription>
                  High-level metadata about this project.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-0 divide-y border-t">
                <FormField
                  control={form.control}
                  name="timezone"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Camera timezone</FormLabel>
                        <FormDescription className="text-sm">
                          Whatever your cameras were set to. Used for
                          exports and activity charts. Doesn't shift the
                          capture times on your photos or videos. Leave on
                          "Auto" to derive it from the first site's
                          location. If the cameras
                          follow a regional timezone with daylight saving,
                          pick the city name. If they use a fixed offset all
                          year (no daylight saving), pick a UTC±N entry
                          instead.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <FormControl>
                          <TimezoneSelect
                            value={field.value}
                            onChange={field.onChange}
                            autoLabel="Auto (from site location)"
                          />
                        </FormControl>
                        <FormMessage />
                      </div>
                    </div>
                  )}
                />
              </CardContent>
            </Card>

            {/* Card 1: Models */}
            <Card>
              <CardHeader>
                <CardTitle>Models</CardTitle>
                <CardDescription>
                  Models used to detect objects and classify labels. Changes apply to new analyses only and do not reprocess existing results.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-0 divide-y border-t">
                {/* Detection Model */}
                <FormField
                  control={form.control}
                  name="detection_model_id"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Detection model</FormLabel>
                        <FormDescription className="text-sm">
                          The first step in the pipeline. Scans each image or video frame and draws bounding boxes around animals, people, and vehicles. Everything downstream (classification, embedding, statistics) depends on what the detection model finds.
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
                          {field.value && (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="self-center">
                                  <Button
                                    type="button"
                                    variant="outline"
                                    className="px-3"
                                    onClick={() => {
                                      setSelectedModelId(field.value ?? null);
                                      setShowModelInfo(true);
                                    }}
                                  >
                                    <InfoIcon className="h-4 w-4" />
                                  </Button>
                                </span>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>View model information</p>
                              </TooltipContent>
                            </Tooltip>
                          )}
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
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Classification model</FormLabel>
                        <FormDescription className="text-sm">
                          The second step. After the detection model finds an animal, the classification model identifies the species by analyzing the cropped region. People and vehicles are not classified further. Optional, select "none" for detection-only projects.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <div className="flex gap-2 items-stretch">
                          <Select
                            key={field.value ?? "none"}
                            onValueChange={(val) => {
                              // Show confirmation when removing classification model
                              if (val === "none" && field.value && field.value !== "none") {
                                setRemoveClsConfirmOpen(true);
                              } else {
                                field.onChange(val);
                              }
                            }}
                            value={field.value ?? "none"}
                            defaultValue={field.value ?? "none"}
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
                          {field.value && field.value !== "none" && (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="self-center">
                                  <Button
                                    type="button"
                                    variant="outline"
                                    className="px-3"
                                    onClick={() => {
                                      setSelectedModelId(field.value ?? null);
                                      setShowModelInfo(true);
                                    }}
                                  >
                                    <InfoIcon className="h-4 w-4" />
                                  </Button>
                                </span>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>View model information</p>
                              </TooltipContent>
                            </Tooltip>
                          )}
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

                {/* Embedding Model */}
                <FormField
                  control={form.control}
                  name="embedding_model_id"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Embedding model</FormLabel>
                        <FormDescription className="text-sm">
                          The third step. Computes a visual fingerprint for each detected animal. Used to sort and search detections by visual similarity.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <div className="flex gap-2 items-stretch">
                          <Select
                            key={field.value ?? "none"}
                            onValueChange={field.onChange}
                            value={field.value ?? "none"}
                          >
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue placeholder="Select embedding model">
                                  {(() => {
                                    if (!field.value || field.value === "none") {
                                      return (
                                        <div className="flex flex-col items-start py-1">
                                          <div>∅ No embedding model</div>
                                          <div className="text-xs text-muted-foreground">
                                            Skip if you do not need similarity sort or clustering
                                          </div>
                                        </div>
                                      );
                                    }
                                    const selectedModel = embeddingModels.find(
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
                                ∅ No embedding model
                                <br />
                                <span className="text-xs text-muted-foreground">
                                  Skip if you do not need similarity sort or clustering
                                </span>
                              </SelectItem>
                              {embeddingModels
                                .filter((m) => m.model_id !== "none")
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
                          {field.value && field.value !== "none" && (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="self-center">
                                  <Button
                                    type="button"
                                    variant="outline"
                                    className="px-3"
                                    onClick={() => {
                                      setSelectedModelId(field.value ?? null);
                                      setShowModelInfo(true);
                                    }}
                                  >
                                    <InfoIcon className="h-4 w-4" />
                                  </Button>
                                </span>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>View model information</p>
                              </TooltipContent>
                            </Tooltip>
                          )}
                        </div>
                        <FormMessage />

                        {/* Model Status Badge */}
                        {field.value && field.value !== "none" && embeddingModelStatus && (
                          <ModelStatusBadge
                            status={embeddingModelStatus}
                            onPrepare={handlePrepareEmbeddingModel}
                            isPreparing={preparationStage === "preparing" && preparingModelType === "embedding"}
                          />
                        )}
                      </div>
                    </div>
                  )}
                />

              </CardContent>
            </Card>

            {/* Card 2: Performance */}
            {(() => {
              const detectionModel = detectionModels.find((m) => m.model_id === detectionModelId);
              const classificationModel = classificationModels.find(
                (m) => m.model_id === classificationModelId,
              );
              const embeddingModel = embeddingModels.find((m) => m.model_id === embeddingModelId);
              const showClassificationRow = hasClassificationModel && !!classificationModel;
              const showEmbeddingRow =
                !!embeddingModelId && embeddingModelId !== "none" && !!embeddingModel;
              return (
                <Card>
                  <CardHeader>
                    <CardTitle>Performance</CardTitle>
                    <CardDescription>
                      Override how many images each model processes in parallel. Higher
                      values are faster but use more memory. Leave at default unless
                      you're running into out-of-memory errors or want to push
                      throughput on a powerful machine.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-0 divide-y border-t">
                    {detectionModel && (
                      <BatchSizeRow
                        control={form.control}
                        name="detection_batch_size"
                        label="Detection batch size"
                        description="Images processed per batch by the detection model."
                        defaultGpu={detectionModel.default_batch_size_gpu}
                        defaultCpu={detectionModel.default_batch_size_cpu}
                      />
                    )}
                    {showClassificationRow && (
                      <BatchSizeRow
                        control={form.control}
                        name="classification_batch_size"
                        label="Classification batch size"
                        description="Crops processed per batch by the classification model."
                        defaultGpu={classificationModel.default_batch_size_gpu}
                        defaultCpu={classificationModel.default_batch_size_cpu}
                      />
                    )}
                    {showEmbeddingRow && (
                      <BatchSizeRow
                        control={form.control}
                        name="embedding_batch_size"
                        label="Embedding batch size"
                        description="Crops processed per batch by the embedding model."
                        defaultGpu={embeddingModel.default_batch_size_gpu}
                        defaultCpu={embeddingModel.default_batch_size_cpu}
                      />
                    )}
                  </CardContent>
                </Card>
              );
            })()}

            {/* Card 3: Label selection */}
            {hasClassificationModel && taxonomy && (
              <Card>
                <CardHeader>
                  <CardTitle>Label selection</CardTitle>
                  <CardDescription>
                    Control which labels can be predicted
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-0 divide-y border-t">
                  <div className="grid grid-cols-2 items-center gap-8 py-6">
                    <div className="space-y-1">
                      <FormLabel>Label selection</FormLabel>
                      <FormDescription className="text-sm">
                        Limit predictions to labels expected in your project area to reduce false positives.
                      </FormDescription>
                    </div>
                    <div>
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
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Card 4: Analysis and counting */}
            <Card>
              <CardHeader>
                <CardTitle>Analysis and counting</CardTitle>
                <CardDescription>
                  Control how detections are filtered, grouped, and aggregated. Changes apply to all analyses (past and future) and affect how data is interpreted, not the underlying detections.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-0 divide-y border-t">
                {/* Video Frame Rate */}
                <FormField
                  control={form.control}
                  name="video_fps"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
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
                            <SelectTrigger>
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
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Detection confidence threshold</FormLabel>
                        <FormDescription className="text-sm">
                          Hide detections below this confidence score. Only affects unverified images. Verified observations are always included. Applies retroactively to all statistics and exports.
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
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Independence interval</FormLabel>
                        <FormDescription className="text-sm">
                          Consecutive files at the same camera within this window are merged into one independent event. The count for each event uses MaxN, the peak number of individuals visible in a single image within that event. Affects all statistics retroactively.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <Select
                          key={String(field.value)}
                          value={String(field.value)}
                          onValueChange={(val) => field.onChange(parseInt(val))}
                        >
                          <FormControl>
                            <SelectTrigger>
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

                {/* Event Smoothing (only when classification model is selected) */}
                {hasClassificationModel && <FormField
                  control={form.control}
                  name="event_smoothing"
                  render={() => (
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Smoothing</FormLabel>
                        <FormDescription className="text-sm">
                          Cleans up classification labels in two steps. Image-level smoothing picks the dominant species when multiple detections in the same image disagree. Event-level smoothing then looks across all images in an event and overwrites outlier labels with the dominant species, based on the strength setting below. Taxonomic relationships are considered when resolving conflicts, so labels within the same family are treated more leniently than cross-family disagreements.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <Select
                          value={form.watch("event_smoothing") ? form.watch("smoothing_strength") : "off"}
                          onValueChange={(value) => {
                            if (!value) return;
                            if (value === "off") {
                              form.setValue("event_smoothing", false, { shouldDirty: true });
                            } else {
                              form.setValue("event_smoothing", true, { shouldDirty: true });
                              form.setValue("smoothing_strength", value as "mild" | "normal" | "aggressive", { shouldDirty: true });
                            }
                          }}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="off">Off</SelectItem>
                            <SelectItem value="mild">Mild</SelectItem>
                            <SelectItem value="normal">Normal</SelectItem>
                            <SelectItem value="aggressive">Aggressive</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  )}
                />}

                {/* Taxonomic Rollup (only when classification model is selected) */}
                {hasClassificationModel && <FormField
                  control={form.control}
                  name="taxonomic_rollup"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Taxonomic rollup</FormLabel>
                        <FormDescription className="text-sm">
                          When the model is not confident enough at the species level, it sums probabilities up the taxonomy tree (species, genus, family, order, class) and picks the most specific level where the combined confidence reaches 0.65. For example, a detection uncertain between "lion" and "leopard" may roll up to "felidae" if the family-level confidence is high enough.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </div>
                    </div>
                  )}
                />}

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

            {/* Sticky footer. Always visible so the primary actions
                (Restore defaults, Reset, Save) are one click away no matter
                how far the user has scrolled. The "unsaved changes" dot
                and label appear only when the form is dirty. Stays inside
                the <form> so the Save button's type="submit" still
                triggers form.handleSubmit. */}
            <div className="fixed bottom-0 left-64 right-0 z-40 border-t bg-card/95 backdrop-blur-sm">
              <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  {isDirty && (
                    <>
                      <span
                        aria-hidden="true"
                        className="inline-block h-2 w-2 rounded-full"
                        style={{ backgroundColor: "#71b7ba" }}
                      />
                      You have unsaved changes
                    </>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => {
                      // Use setValue (not reset) so the saved project values stay as the
                      // dirty-check baseline. This way isDirty becomes true and the user
                      // must press "Save changes" to persist the defaults.
                      form.setValue("detection_model_id", "MD5A-0-0", { shouldDirty: true });
                      form.setValue("video_fps", 1.0, { shouldDirty: true });
                      form.setValue("detection_threshold", 0.5, { shouldDirty: true });
                      form.setValue("event_smoothing", true, { shouldDirty: true });
                      form.setValue("smoothing_strength", "normal", { shouldDirty: true });
                      form.setValue("taxonomic_rollup", true, { shouldDirty: true });
                      form.setValue("taxonomic_rollup_threshold", 0.65, { shouldDirty: true });
                      form.setValue("independence_interval", 1800, { shouldDirty: true });
                      form.setValue("detection_batch_size", null, { shouldDirty: true });
                      form.setValue("classification_batch_size", null, { shouldDirty: true });
                      form.setValue("embedding_batch_size", null, { shouldDirty: true });
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
                            (hasClassificationModel && classificationModelStatus?.status !== "ready") ||
                            Boolean(embeddingModelId && embeddingModelId !== "none" && embeddingModelStatus?.status !== "ready")
                          }
                        >
                          <Save className="h-4 w-4 mr-2" />
                          {updateMutation.isPending ? "Saving..." : "Save changes"}
                        </Button>
                      </span>
                    </TooltipTrigger>
                    {(detectionModelStatus?.status !== "ready" || (hasClassificationModel && classificationModelStatus?.status !== "ready") || (embeddingModelId && embeddingModelId !== "none" && embeddingModelStatus?.status !== "ready")) && (
                      <TooltipContent>
                        <p>Model needs preparing first</p>
                      </TooltipContent>
                    )}
                  </Tooltip>
                </div>
              </div>
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

        {/* Label Selection Modal */}
        {classificationModelId && taxonomy && (
          <SpeciesSelectionModal
            modelId={classificationModelId}
            excludedClasses={excludedClasses}
            onExclusionChange={(classes) => {
              setExcludedClasses(classes);
              form.setValue("excluded_classes", classes, { shouldDirty: true });
            }}
            open={labelSelectionModalOpen}
            onOpenChange={setLabelSelectionModalOpen}
            totalSpeciesCount={taxonomy.all_classes?.length || 0}
            countryCode={countryCode}
            stateCode={form.watch("state_code")}
            onLocationChange={(country, state) => {
              form.setValue("country_code", country, { shouldDirty: true });
              form.setValue("state_code", state, { shouldDirty: true });
            }}
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
            {preparingModelType === "embedding" && embeddingModels.find((m) => m.model_id === embeddingModelId) && (
              <ModelPreparationView
                modelName={embeddingModels.find((m) => m.model_id === embeddingModelId)!.friendly_name}
                modelEmoji={embeddingModels.find((m) => m.model_id === embeddingModelId)!.emoji}
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

        {/* Re-embed Confirmation Dialog */}
        <AlertDialog open={reEmbedConfirmOpen} onOpenChange={setReEmbedConfirmOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Re-embed observations?</AlertDialogTitle>
              <AlertDialogDescription>
                Changing the embedding model from{" "}
                <strong>
                  {embeddingModels.find(m => m.model_id === (form.formState.defaultValues as SettingsFormData)?.embedding_model_id)?.friendly_name ?? "None"}
                </strong>{" "}
                to{" "}
                <strong>
                  {embeddingModels.find(m => m.model_id === pendingFormData.current?.embedding_model_id)?.friendly_name ?? "None"}
                </strong>{" "}
                requires re-embedding{" "}
                <strong>{reEmbedDetectionCount.toLocaleString()}</strong> observations.
                This may take a while.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel onClick={handleRevertReEmbed}>
                No, keep current model
              </AlertDialogCancel>
              <AlertDialogAction onClick={handleConfirmReEmbed}>
                Yes, re-embed
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        {/* Classification Model Removal Confirmation */}
        <AlertDialog open={removeClsConfirmOpen} onOpenChange={setRemoveClsConfirmOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Remove classification model?</AlertDialogTitle>
              <AlertDialogDescription>
                Existing classifications will remain, but no new ones will be generated for future deployments. You can re-enable classification at any time.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction onClick={() => {
                form.setValue("classification_model_id", "none", { shouldDirty: true });
                setRemoveClsConfirmOpen(false);
              }}>
                Remove model
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        {/* Re-embed Progress Modal */}
        <ReEmbedModal
          open={!!reEmbedJobId}
          onOpenChange={(open) => { if (!open) setReEmbedJobId(null); }}
          jobId={reEmbedJobId}
          onComplete={() => {
            queryClient.invalidateQueries({ queryKey: ["projects", projectId] });
            toast.success("Re-embedding complete!");
          }}
        />
      </main>
    </div>
  );
}


