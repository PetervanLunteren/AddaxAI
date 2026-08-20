/**
 * Project Settings Page.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear structure
 * - Explicit error handling
 */

import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { Save, RotateCcw, Undo2 } from "lucide-react";
import { toast } from "sonner";
import { projectsApi, type ProjectUpdate } from "../api/projects";
import { invalidateModelMetadata, modelsApi } from "../api/models";
import {
  LabelSelectionField,
  toApiCountryCode,
  useLabelSelectionCaption,
} from "../components/taxonomy/LabelSelectionField";
import { AnalysisSettingsRows } from "../components/settings/AnalysisSettingsRows";
import { ApplySettingsModal } from "../components/settings/ApplySettingsModal";
import {
  fetchRegroupImpact,
  hasReprocessChanges,
  type RegroupImpact,
  startReprocessIfNeeded,
  warnIfDeploymentsSkipped,
} from "../lib/reprocessSettings";
import { RegroupConfirmDialog } from "../components/settings/RegroupConfirmDialog";
import {
  buildSaveResults,
  fetchStats,
  type ProjectStats,
} from "../lib/reprocessStats";
import {
  MEDIA_FILTER_OPTIONS,
  VIDEO_FPS_OPTIONS,
  advancedNonDefaultKeys,
  restoreAdvancedDefaults,
} from "../lib/advancedSettingsDefaults";
import { useSidebarCollapsed } from "../components/layout/sidebar-context";
import { ModelSelect } from "../components/models/ModelSelect";
import { toApiModelId } from "../lib/model-id";
import { NoClassifierNotice } from "../components/models/NoClassifierNotice";
import { ModelInfoSheet } from "../components/models/ModelInfoSheet";
import { ModelStatusBadge } from "../components/projects/ModelStatusBadge";
import { ModelPreparationView } from "../components/projects/ModelPreparationView";
import { ModelPreparationErrorView } from "../components/projects/ModelPreparationErrorView";
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
import { useReprocessSummary } from "../hooks/useReprocessSummary";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "../components/ui/dialog";
import { Button } from "../components/ui/button";
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
import { Switch } from "../components/ui/switch";
import {
  DEFAULT_CLASSIFICATION_GATE,
  DEFAULT_COUNTING_THRESHOLD,
  formatConfidencePct,
} from "../lib/confidence";
import { ConfidenceSlider } from "../components/ui/confidence-slider";
import { SETTING_CAPTIONS } from "../lib/settingCaptions";
import { ClassificationModelGroupedItems } from "../components/models/ClassificationModelGroupedItems";
import { BatchSizeRow } from "../components/analyses/BatchSizeRow";
import { ImageSizeRow } from "../components/analyses/ImageSizeRow";
import { SettingRow } from "../components/analyses/SettingRow";
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
  media_filter: z.enum(["all", "images", "videos"]),
  detection_augment: z.boolean(),
  detection_image_size: z.number().int().nullable(),
  counting_threshold: z.number().min(0).max(1),
  classification_gate: z.number().min(0.01).max(1),
  event_smoothing: z.boolean(),
  smoothing_strength: z.enum(["mild", "normal", "aggressive"]),
  taxonomic_rollup: z.boolean(),
  independence_interval: z.number().min(0),
  // null = use the per-pipeline default; integer = user override
  detection_batch_size: z.number().int().min(1).max(256).nullable(),
  classification_batch_size: z.number().int().min(1).max(256).nullable(),
  embedding_batch_size: z.number().int().min(1).max(256).nullable(),
});

type SettingsFormData = z.infer<typeof settingsSchema>;

// BatchSizeRow lives in components/analyses/BatchSizeRow.tsx and is
// shared between this page and the folder-run setup step. Imported above.

export default function SettingsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const queryClient = useQueryClient();
  // The sticky save bar is `fixed` and spans the content area, so its
  // left edge has to track the sidebar/rail width.
  const sidebarCollapsed = useSidebarCollapsed();
  const [excludedClasses, setExcludedClasses] = useState<string[]>([]);
  const [showModelInfo, setShowModelInfo] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  // Model preparation state
  type PreparationStage = "form" | "preparing" | "error";
  type PreparingModelType = "detection" | "classification" | "embedding" | null;
  const [preparationStage, setPreparationStage] = useState<PreparationStage>("form");
  const [preparingTaskId, setPreparingTaskId] = useState<string | null>(null);
  const [preparationError, setPreparationError] = useState<string | null>(null);
  // Always set alongside preparationError, so a kind cannot outlive the
  // failure it described. Undefined for anything but a named cause.
  const [preparationErrorKind, setPreparationErrorKind] = useState<
    string | undefined
  >(undefined);
  const [preparingModelType, setPreparingModelType] = useState<PreparingModelType>(null);

  // Unified save flow state
  const [saveJobId, setSaveJobId] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false); // shows modal before job ID is known
  const { showSummary, summaryUI } = useReprocessSummary(projectId ?? "");
  // Stores before-stats + the new threshold while reprocessing runs (the
  // after-stats fetch needs the threshold for the Detections card).
  const pendingBeforeStats = useRef<{
    before: ProjectStats;
    newThreshold: number;
  } | null>(null);

  // Classification model removal confirmation
  const [removeClsConfirmOpen, setRemoveClsConfirmOpen] = useState(false);

  // Re-embed confirmation + progress state
  const [reEmbedConfirmOpen, setReEmbedConfirmOpen] = useState(false);
  const [reEmbedJobId, setReEmbedJobId] = useState<string | null>(null);
  const [reEmbedDetectionCount, setReEmbedDetectionCount] = useState(0);
  const pendingFormData = useRef<SettingsFormData | null>(null);

  // Interval-change regroup confirmation. When an interval change would
  // reset confirmed counts, we hold the pending save until the user types
  // the confirm word.
  const [regroupImpact, setRegroupImpact] = useState<RegroupImpact | null>(null);
  const pendingRegroupData = useRef<SettingsFormData | null>(null);

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
      embedding_model_id: "DINOV2-VITS14",
      excluded_classes: [],
      country_code: null,
      state_code: null,
      timezone: "",
      video_fps: 1.0,
      media_filter: "all",
      detection_augment: false,
      detection_image_size: null,
      counting_threshold: DEFAULT_COUNTING_THRESHOLD,
      classification_gate: DEFAULT_CLASSIFICATION_GATE,
      event_smoothing: true,
      smoothing_strength: "normal" as const,
      taxonomic_rollup: true,
      independence_interval: 1800,
      detection_batch_size: null,
      classification_batch_size: null,
      embedding_batch_size: null,
    },
  });

  // Which advanced settings differ from their factory default, so each row
  // can chip itself. Same helper the folder-run setup step uses, so "what
  // counts as default" has one definition across both flows.
  const changedAdvanced = advancedNonDefaultKeys(form.watch());

  // The classification model as loaded from the project. The excluded_classes
  // filter below must fire only on a genuine USER model change, never on the
  // initial load, or it would drop saved exclusions and dirty the form with no
  // user action. Compare against this ref, never against a watch() value: the
  // two are written in different commits (see the note on that effect).
  const loadedModelRef = useRef<string | null | undefined>(undefined);

  // Update form values when project loads
  useEffect(() => {
    if (project) {
      loadedModelRef.current = project.classification_model_id ?? null;
      const values: SettingsFormData = {
        detection_model_id: project.detection_model_id,
        classification_model_id: project.classification_model_id ?? null,
        embedding_model_id: project.embedding_model_id || "none",
        excluded_classes: project.excluded_classes || [],
        country_code: project.country_code || null,
        state_code: project.state_code || null,
        timezone: project.timezone ?? "",
        video_fps: project.video_fps,
        media_filter: project.media_filter,
        detection_augment: project.detection_augment,
        detection_image_size: project.detection_image_size,
        counting_threshold: project.counting_threshold,
        classification_gate: project.classification_gate,
        event_smoothing: project.event_smoothing,
        smoothing_strength: (project.smoothing_strength || "normal") as "mild" | "normal" | "aggressive",
        taxonomic_rollup: project.taxonomic_rollup,
        independence_interval: project.independence_interval,
        detection_batch_size: project.detection_batch_size ?? null,
        classification_batch_size: project.classification_batch_size ?? null,
        embedding_batch_size: project.embedding_batch_size ?? null,
      };
      form.reset(values);
    }
  }, [project, form]);

  // Watch model changes
  const detectionModelId = form.watch("detection_model_id");
  const classificationModelId = form.watch("classification_model_id");
  const labelCaption = useLabelSelectionCaption(
    classificationModelId && classificationModelId !== "none" ? classificationModelId : "",
  );
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
  const { progress, message, cancel: cancelPreparationTask } = useTaskProgress({
    taskId: preparingTaskId,
    onComplete: () => {
      // Refresh the correct model status based on which model was being prepared
      const modelIdToRefresh = preparingModelType === "detection" ? detectionModelId
        : preparingModelType === "embedding" ? embeddingModelId
        : classificationModelId;
      invalidateModelMetadata(queryClient, modelIdToRefresh);
      setPreparingTaskId(null);
      setPreparationStage("form");
      setPreparingModelType(null);
    },
    onError: (error, kind) => {
      setPreparationError(error);
      setPreparationErrorKind(kind);
      setPreparationStage("error");
      setPreparingTaskId(null);
    },
    onCancelled: () => {
      setPreparingTaskId(null);
      setPreparationStage("form");
      setPreparingModelType(null);
    },
  });

  // Initialize excludedClasses state when project loads
  useEffect(() => {
    if (project) {
      const savedExcluded = project.excluded_classes || [];
      setExcludedClasses(savedExcluded);
    }
  }, [project]);

  // There is deliberately no "clear state_code when the country changes"
  // effect. The location picker sets country AND state together (see
  // onLocationChange below), so state is already null before any such effect
  // could run — it had no reachable job. What it DID do was fire on mount:
  // with the project already cached, it ran in the same commit as the reset
  // above and compared a stale watched country against a freshly-written ref,
  // then nulled a freshly-reset state_code with shouldDirty. That was the
  // phantom "You have unsaved changes". FolderRunModelStep uses the same
  // picker with no such effect and never had the bug.

  // Drop excluded classes the newly-picked model doesn't have.
  useEffect(() => {
    // Read the LIVE value, not the watched one. When the project is already
    // in the react-query cache, this effect and the project-load effect above
    // run in the SAME commit: the watched value is still the pre-reset
    // default while the ref and the form are already updated. Comparing
    // stale-to-fresh is exactly what produced the phantom-dirty bug in the
    // sibling effect this replaced. `classificationModelId` stays in the deps
    // as the trigger; only the read moves.
    const modelId = form.getValues("classification_model_id");
    if (modelId === loadedModelRef.current) return;
    if (modelId && taxonomy?.all_classes) {
      const currentExcluded = form.getValues("excluded_classes");
      const validExcluded = currentExcluded.filter((cls) =>
        taxonomy.all_classes.includes(cls),
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
      setPreparationErrorKind(undefined);
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
      setPreparationErrorKind(undefined);
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
      setPreparationErrorKind(undefined);
      setPreparationStage("error");
      setPreparingModelType(null);
    }
  };

  // Handler for canceling preparation. Sends a real cancel over the
  // WebSocket; the worker kills its subprocess and replies "cancelled",
  // handled by onCancelled above. If the socket isn't open (rare), fall
  // back to detaching the UI so the modal can't get stuck.
  const handleCancelPreparation = () => {
    if (preparingTaskId) {
      cancelPreparationTask();
    } else {
      setPreparationStage("form");
      setPreparingModelType(null);
    }
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
    onComplete: async (data) => {
      setSaveJobId(null);
      setIsSaving(false);
      const nothingApplied = warnIfDeploymentsSkipped(data);
      // Blanket invalidate so every page (images, dashboard, review,
      // insights) picks up the reprocessed labels/annotations
      // immediately.
      if (projectId) {
        invalidateProjectData(queryClient, projectId);
      }
      queryClient.invalidateQueries({ queryKey: ["postprocessing-status", projectId] });

      // Not one folder was reprocessed: there is no "what changed" to show
      // and no save to celebrate. The warning above is the whole message,
      // and a "Settings saved!" beside it would only contradict it.
      if (nothingApplied) {
        pendingBeforeStats.current = null;
        return;
      }

      // Fetch after-stats now that reprocessing is done
      const pending = pendingBeforeStats.current;
      if (!pending || !projectId) {
        pendingBeforeStats.current = null;
        toast.success("Settings saved!");
        return;
      }

      try {
        const afterStats = await fetchStats(projectId, pending.newThreshold);
        pendingBeforeStats.current = null;
        showSummary(buildSaveResults(pending.before, afterStats));
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

  const saveSettings = async (
    data: SettingsFormData,
    regroupConfirmed = false,
  ) => {
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

    const currentValues = form.formState.defaultValues as SettingsFormData;

    // Gate (outermost): an interval change that would reset confirmed counts
    // must be confirmed first, before any save or re-embed. Preview failures
    // don't block the save.
    if (!regroupConfirmed) {
      const impact = await fetchRegroupImpact(
        projectId,
        currentValues.independence_interval,
        data.independence_interval,
      ).catch(() => null);
      if (impact) {
        pendingRegroupData.current = data;
        setRegroupImpact(impact);
        return;
      }
    }

    // Intercept embedding model change — confirm only when replacing an existing model
    // and there are detections to re-embed. Skip for "none" → model (first-time enable).
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
      const willReprocess = hasReprocessChanges(currentValues, data);

      // Show progress modal immediately if reprocessing will happen
      if (willReprocess) {
        setIsSaving(true);
      }

      // 1. Start fetching before-stats in the background (don't await yet)
      const beforeStatsPromise = fetchStats(
        projectId,
        currentValues.counting_threshold,
      );

      // 2. Save settings. Empty timezone means "Auto" — send null so the
      // backend leaves it unset and keeps deriving it from site coords.
      await updateMutation.mutateAsync({
        ...data,
        timezone: data.timezone || null,
        // ALL is a form-only sentinel; the API knows ISO codes or null.
        country_code: toApiCountryCode(data.country_code),
        // "none" is a form-only sentinel too. Without this the project
        // stores the string "none" as its model and every later analysis
        // is blocked by a "none needs setup" dialog.
        classification_model_id: toApiModelId(data.classification_model_id),
        embedding_model_id: toApiModelId(data.embedding_model_id),
      });

      // 3. If reprocess-triggering settings changed, kick off the job
      if (willReprocess) {
        const reprocessJobId = await startReprocessIfNeeded(projectId);
        if (reprocessJobId) {
          setSaveJobId(reprocessJobId);

          // Await before-stats (likely already resolved by now)
          const beforeStats = await beforeStatsPromise;
          pendingBeforeStats.current = {
            before: beforeStats,
            newThreshold: data.counting_threshold,
          };
          return; // Progress modal takes over; toast shown in onComplete
        }
        // No classifications to reprocess — close the modal
        setIsSaving(false);
      }

      // 4. No reprocess needed — await before-stats and fetch after-stats
      const beforeStats = await beforeStatsPromise;
      const afterStats = await fetchStats(projectId, data.counting_threshold);

      showSummary(buildSaveResults(beforeStats, afterStats));
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
        media_filter: project.media_filter,
        detection_augment: project.detection_augment,
        detection_image_size: project.detection_image_size,
        counting_threshold: project.counting_threshold,
        classification_gate: project.classification_gate,
        event_smoothing: project.event_smoothing,
        smoothing_strength: (project.smoothing_strength || "normal") as "mild" | "normal" | "aggressive",
        taxonomic_rollup: project.taxonomic_rollup,
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
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 pb-20 sm:px-6 lg:px-8 space-y-6">
        {/* Settings form */}
        <TooltipProvider>
          <Form {...form}>
            <form onSubmit={form.handleSubmit((data) => saveSettings(data))} className="space-y-6" key={project?.id}>
            {/* Card: Models */}
            <Card>
              <CardHeader>
                <CardTitle>Models</CardTitle>
                <CardDescription>
                  The models used to analyze your images. Changes apply to new analyses only.
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
                          Finds animals, people, and vehicles in each image or video frame. Everything else builds on what it finds.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <ModelSelect
                          value={field.value}
                          onValueChange={field.onChange}
                          models={detectionModels}
                          placeholder="Select detection model"
                          onShowInfo={() => {
                            setSelectedModelId(field.value ?? null);
                            setShowModelInfo(true);
                          }}
                        >
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
                        </ModelSelect>
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
                          Identifies the species of each animal the detection model finds. Optional: choose "none" for a detection-only project.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <ModelSelect
                          value={field.value ?? "none"}
                          onValueChange={(val) => {
                            // Show confirmation when removing classification model
                            if (val === "none" && field.value && field.value !== "none") {
                              setRemoveClsConfirmOpen(true);
                            } else {
                              field.onChange(val);
                            }
                          }}
                          models={classificationModels}
                          placeholder="Select classification model"
                          noneValue="none"
                          noneLabel="No classification model"
                          onShowInfo={() => {
                            setSelectedModelId(field.value ?? null);
                            setShowModelInfo(true);
                          }}
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

                        {/* Model Status Badge */}
                        {field.value && classificationModelStatus && (
                          <ModelStatusBadge
                            status={classificationModelStatus}
                            onPrepare={handlePrepareClassificationModel}
                            isPreparing={preparationStage === "preparing" && preparingModelType === "classification"}
                          />
                        )}

                        {!hasClassificationModel && <NoClassifierNotice />}
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
                          Creates a visual fingerprint of each animal, used to sort and search by similarity.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <ModelSelect
                          value={field.value ?? "none"}
                          onValueChange={field.onChange}
                          models={embeddingModels}
                          placeholder="Select embedding model"
                          noneValue="none"
                          noneLabel="No embedding model"
                          onShowInfo={() => {
                            setSelectedModelId(field.value ?? null);
                            setShowModelInfo(true);
                          }}
                        >
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
                        </ModelSelect>
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
                      How many images each model processes at once. Higher is faster
                      but uses more memory. Leave as is unless you hit out-of-memory
                      errors.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-0 divide-y border-t">
                    {detectionModel && (
                      <BatchSizeRow
                        control={form.control}
                        name="detection_batch_size"
                        label="Detection batch size"
                        description={SETTING_CAPTIONS.detectionBatchSize}
                        defaultGpu={detectionModel.default_batch_size_gpu}
                        defaultCpu={detectionModel.default_batch_size_cpu}
                      />
                    )}
                    {showClassificationRow && (
                      <BatchSizeRow
                        control={form.control}
                        name="classification_batch_size"
                        label="Classification batch size"
                        description={SETTING_CAPTIONS.classificationBatchSize}
                        defaultGpu={classificationModel.default_batch_size_gpu}
                        defaultCpu={classificationModel.default_batch_size_cpu}
                      />
                    )}
                    {showEmbeddingRow && (
                      <BatchSizeRow
                        control={form.control}
                        name="embedding_batch_size"
                        label="Embedding batch size"
                        description={SETTING_CAPTIONS.embeddingBatchSize}
                        defaultGpu={embeddingModel.default_batch_size_gpu}
                        defaultCpu={embeddingModel.default_batch_size_cpu}
                      />
                    )}
                  </CardContent>
                </Card>
              );
            })()}

            {/* Card 3: Species selection. Two-column row (label + caption
                left, control right) to match the other settings rows. The
                caption comes from the shared useLabelSelectionCaption hook. */}
            {hasClassificationModel && taxonomy && (
              <Card>
                <CardHeader>
                  <CardTitle>Species selection</CardTitle>
                </CardHeader>
                <CardContent className="space-y-0 divide-y border-t">
                  <div className="grid grid-cols-2 items-center gap-8 py-6">
                    <div className="space-y-1">
                      <FormLabel>Species selection</FormLabel>
                      <FormDescription className="text-sm">
                        {labelCaption}
                      </FormDescription>
                    </div>
                    <div>
                      <LabelSelectionField
                        modelId={classificationModelId}
                        excludedClasses={excludedClasses}
                        allClasses={taxonomy.all_classes ?? []}
                        countryCode={countryCode}
                        stateCode={form.watch("state_code")}
                        onExclusionChange={(classes) => {
                          setExcludedClasses(classes);
                          form.setValue("excluded_classes", classes, { shouldDirty: true });
                        }}
                        onLocationChange={(country, state) => {
                          form.setValue("country_code", country, { shouldDirty: true });
                          form.setValue("state_code", state, { shouldDirty: true });
                        }}
                      />
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
                  How detections are filtered, grouped into events, and counted.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-0 divide-y border-t">
                {/* Media to analyse — above the frame rate, which only
                    matters once videos are being read at all. */}
                <FormField
                  control={form.control}
                  name="media_filter"
                  render={({ field }) => (
                    <SettingRow
                      label="Media to analyse"
                      isCustom={changedAdvanced.includes("media_filter")}
                      description={
                        <>
                          {SETTING_CAPTIONS.mediaFilter} Applies to new
                          analyses only.
                        </>
                      }
                    >
                      <Select
                        value={field.value}
                        onValueChange={field.onChange}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {MEDIA_FILTER_OPTIONS.map((opt) => (
                            <SelectItem key={opt.value} value={opt.value}>
                              {opt.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </SettingRow>
                  )}
                />

                {/* Video Frame Rate */}
                <FormField
                  control={form.control}
                  name="video_fps"
                  render={({ field }) => (
                    <SettingRow
                      label="Video frame rate"
                      isCustom={changedAdvanced.includes("video_fps")}
                      description={
                        <>
                          {SETTING_CAPTIONS.videoFrameRate} Applies to new
                          analyses only.
                        </>
                      }
                    >
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
                    </SettingRow>
                  )}
                />

                {/* Detection image size (inference-time) */}
                <ImageSizeRow
                  control={form.control}
                  name="detection_image_size"
                  label="Detection image size"
                  description={`${SETTING_CAPTIONS.detectionImageSize} Applies to new analyses only.`}
                />

                {/* Image augmentation (inference-time) */}
                <FormField
                  control={form.control}
                  name="detection_augment"
                  render={({ field }) => (
                    <SettingRow
                      label="Image augmentation"
                      isCustom={changedAdvanced.includes("detection_augment")}
                      description={
                        <>
                          {SETTING_CAPTIONS.imageAugmentation} Applies to new
                          analyses only.
                        </>
                      }
                    >
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                        <FormMessage />
                    </SettingRow>
                  )}
                />

                {/* Classification gate (inference-time) */}
                <FormField
                  control={form.control}
                  name="classification_gate"
                  render={({ field }) => (
                    <SettingRow
                      label="Classify detections above"
                      isCustom={changedAdvanced.includes("classification_gate")}
                      description={
                        <>
                          {SETTING_CAPTIONS.classificationGate} Applies to new
                          analyses only.
                        </>
                      }
                    >
                        <ConfidenceSlider
                          value={field.value}
                          onChange={(vals) => field.onChange(vals[0])}
                          valueLabel={
                            <span className="text-sm font-medium min-w-[3.5rem] shrink-0 text-right">{formatConfidencePct(field.value)}</span>
                          }
                        />
                        <FormMessage />
                    </SettingRow>
                  )}
                />

                {/* Detection Threshold */}
                <FormField
                  control={form.control}
                  name="counting_threshold"
                  render={({ field }) => (
                    <SettingRow
                      label="Count detections above"
                      isCustom={changedAdvanced.includes("counting_threshold")}
                      description={
                        <>
                          {SETTING_CAPTIONS.detectionThreshold} Applies
                          retroactively to all statistics and exports.
                        </>
                      }
                    >
                        <ConfidenceSlider
                          value={field.value}
                          onChange={(vals) => field.onChange(vals[0])}
                          valueLabel={
                            <span className="text-sm font-medium min-w-[3rem] shrink-0 text-right">{formatConfidencePct(field.value)}</span>
                          }
                        />
                        <FormMessage />
                    </SettingRow>
                  )}
                />

                {/* Independence interval, smoothing, taxonomic rollup —
                    shared rows with the folder-run Labels step's
                    analysis panel (one source of truth). */}
                <AnalysisSettingsRows
                  values={{
                    event_smoothing: form.watch("event_smoothing"),
                    smoothing_strength: form.watch("smoothing_strength"),
                    taxonomic_rollup: form.watch("taxonomic_rollup"),
                    independence_interval: form.watch(
                      "independence_interval",
                    ),
                  }}
                  onIntervalChange={(seconds) =>
                    form.setValue("independence_interval", seconds, {
                      shouldDirty: true,
                    })
                  }
                  onSmoothingChange={(level) => {
                    if (level === "off") {
                      form.setValue("event_smoothing", false, {
                        shouldDirty: true,
                      });
                    } else {
                      form.setValue("event_smoothing", true, {
                        shouldDirty: true,
                      });
                      form.setValue("smoothing_strength", level, {
                        shouldDirty: true,
                      });
                    }
                  }}
                  onRollupChange={(enabled) =>
                    form.setValue("taxonomic_rollup", enabled, {
                      shouldDirty: true,
                    })
                  }
                  showClassifierFields={hasClassificationModel}
                  intervalNote="Affects all statistics retroactively."
                />

                {/* Camera timezone: affects how event/activity times and
                    exports read (moved from the old one-setting card). */}
                <FormField
                  control={form.control}
                  name="timezone"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Camera timezone</FormLabel>
                        <FormDescription className="text-sm">
                          Your cameras' timezone, used for exports and
                          activity charts. It doesn't change the capture times
                          on your files. Leave on "Auto" to use the first
                          site's location. Pick a city for regional time with
                          daylight saving, or a UTC±N for a fixed offset.
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
            <div
              className={`fixed bottom-0 right-0 z-40 border-t bg-card/95 backdrop-blur-sm transition-[left] duration-200 ${
                sidebarCollapsed ? "left-16" : "left-64"
              }`}
            >
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
                      restoreAdvancedDefaults(form.setValue);
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

        {/* Model Preparation Dialog */}
        <Dialog open={preparationStage === "preparing"} onOpenChange={(open) => !open && handleCancelPreparation()}>
          <DialogContent className="max-w-xl">
            <DialogTitle className="sr-only">Preparing model</DialogTitle>
            <DialogDescription className="sr-only">
              AddaxAI is downloading and preparing the selected model.
            </DialogDescription>
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

        {/* Applying Settings Progress Dialog (shared with the
            folder-run Labels step's analysis panel) */}
        <ApplySettingsModal
          open={isSaving || !!saveJobId}
          message={saveProgress.message}
          progress={saveProgress.progress}
          fallbackMessage={
            isSaving && !saveJobId ? "Saving settings..." : "Starting..."
          }
        />

        {/* Save toast + effect-on-statistics modal */}
        {summaryUI}

        {/* Interval-change regroup confirmation */}
        {regroupImpact && (
          <RegroupConfirmDialog
            open
            onOpenChange={(o) => !o && setRegroupImpact(null)}
            impact={regroupImpact}
            fromInterval={
              form.formState.defaultValues?.independence_interval ?? 0
            }
            toInterval={pendingRegroupData.current?.independence_interval ?? 0}
            isPending={isSaving || !!saveJobId}
            onConfirm={() => {
              setRegroupImpact(null);
              const data = pendingRegroupData.current;
              pendingRegroupData.current = null;
              if (data) saveSettings(data, true);
            }}
          />
        )}

        {/* Model Preparation Error Dialog */}
        <Dialog open={preparationStage === "error"} onOpenChange={(open) => !open && setPreparationStage("form")}>
          <DialogContent className="max-w-xl">
            <ModelPreparationErrorView
              errorMessage={preparationError || "Unknown error occurred"}
              errorKind={preparationErrorKind}
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


