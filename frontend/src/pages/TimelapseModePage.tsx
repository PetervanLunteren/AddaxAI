/**
 * Timelapse Analyser integration page.
 *
 * Layout matches the main app exactly:
 * - canonical header / main wrapper (FRONTEND_CONVENTIONS.md), but
 *   narrower (max-w-5xl) than the main app pages because Timelapse is
 *   a single-column focused form rather than a dashboard or grid.
 * - Card sections with 2-column rows: bold title + grey caption left,
 *   widget right (same shape used in pages/SettingsPage.tsx and
 *   components/projects/CreateProjectDialog.tsx)
 *
 * Reused from the main app, no parallel implementations:
 * - FolderSelector (folder picker with file count preview)
 * - ClassificationModelGroupedItems (region-grouped cls dropdown)
 * - SpeciesSelectionModal (label tree + country/state geofilter)
 * - ModelInfoSheet (info-button drawer)
 * - ModelStatusBadge / ModelPreparationView / ModelPreparationErrorView
 *   (model download + env build flow, identical to CreateProjectDialog)
 * - useTaskProgress (websocket progress hook used by the main worker)
 * - SetupPage as inline fallback when AddaxAI isn't set up yet
 */

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import {
  CheckCircle2,
  ChevronDown,
  FolderOpen,
  InfoIcon,
  ListTodo,
  Loader2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import { AnalysisProgress } from "@/components/analyses/AnalysisProgress";
import { BatchSizeRow } from "@/components/analyses/BatchSizeRow";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { DiagnosticReportButton } from "@/components/diagnostics/DiagnosticReportButton";
import { FolderSelector } from "@/components/analyses/FolderSelector";
import { ClassificationModelGroupedItems } from "@/components/models/ClassificationModelGroupedItems";
import { ModelInfoSheet } from "@/components/models/ModelInfoSheet";
import { ModelPreparationErrorView } from "@/components/projects/ModelPreparationErrorView";
import { ModelPreparationView } from "@/components/projects/ModelPreparationView";
import { ModelStatusBadge } from "@/components/projects/ModelStatusBadge";
import { SpeciesSelectionModal } from "@/components/taxonomy/SpeciesSelectionModal";

import { useTaskProgress } from "@/hooks/useTaskProgress";

import { modelsApi } from "@/api/models";
import { setupApi } from "@/api/setup";
import { timelapseApi, type SmoothingStrength } from "@/api/timelapse";

import SetupPage from "./SetupPage";

const NO_CLASSIFIER = "none";

const VIDEO_FPS_OPTIONS = [
  { value: "0.1", label: "1 frame every 10 seconds" },
  { value: "0.25", label: "1 frame every 4 seconds" },
  { value: "0.5", label: "1 frame every 2 seconds" },
  { value: "1", label: "1 frame per second" },
  { value: "2", label: "2 frames per second" },
  { value: "4", label: "4 frames per second" },
  { value: "10", label: "10 frames per second" },
];

const timelapseSchema = z.object({
  folder_path: z.string().min(1, "Select a folder"),
  detection_model_id: z.string().min(1),
  classification_model_id: z.string().nullable(),
  excluded_classes: z.array(z.string()),
  country_code: z.string().nullable(),
  state_code: z.string().nullable(),
  // Detection confidence is not user-controlled in Timelapse integration.
  // The runner hardcodes 0.1, matching the main app's worker. Users
  // do their own filtering inside Timelapse Analyser.
  // null = let the subprocess pick its own default. See app/ml/batch_size.py.
  detection_batch_size: z.number().int().min(1).max(256).nullable(),
  classification_batch_size: z.number().int().min(1).max(256).nullable(),
  video_fps: z.number().min(0.1).max(10),
  event_smoothing: z.boolean(),
  smoothing_strength: z.enum(["mild", "normal", "aggressive"]),
  taxonomic_rollup: z.boolean(),
});

type TimelapseFormData = z.infer<typeof timelapseSchema>;

function readQueryFolder(): string {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("path");
  if (fromQuery) return fromQuery;
  const hash = window.location.hash;
  const qIndex = hash.indexOf("?");
  if (qIndex >= 0) {
    const hashParams = new URLSearchParams(hash.slice(qIndex + 1));
    return hashParams.get("path") ?? "";
  }
  return "";
}

export default function TimelapseModePage() {
  const { data: setupStatus, isLoading: setupLoading } = useQuery({
    queryKey: ["setup-status"],
    queryFn: setupApi.getStatus,
    refetchInterval: 5000,
  });

  if (setupLoading || !setupStatus) return null;
  if (!setupStatus.ready) {
    // Run setup inline so users launched via Timelapse never have to
    // hop windows to install models.
    return <SetupPage />;
  }
  return <TimelapseFormPage />;
}

type Stage = "form" | "preparing" | "error" | "running" | "done";

function TimelapseFormPage() {
  const queryClient = useQueryClient();
  const [stage, setStage] = useState<Stage>("form");
  const [labelModalOpen, setLabelModalOpen] = useState(false);
  const [showClsInfo, setShowClsInfo] = useState(false);
  const [showDetInfo, setShowDetInfo] = useState(false);
  // Advanced settings start collapsed: defaults work for the common
  // case; power users open the disclosure when they need a knob.
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Model preparation state (mirrors CreateProjectDialog).
  const [preparingModelId, setPreparingModelId] = useState<string | null>(null);
  const [preparingTaskId, setPreparingTaskId] = useState<string | null>(null);
  const [preparationError, setPreparationError] = useState<string | null>(null);

  // Run state.
  const [jobId, setJobId] = useState<string | null>(null);
  const [outputPath, setOutputPath] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  // Tracks the in-flight cancel request between button click and the
  // backend's `cancelled` websocket reply. Disables the Cancel button so
  // a second click does not fire a second cancel.
  const [isCancelling, setIsCancelling] = useState(false);

  const form = useForm<TimelapseFormData>({
    resolver: zodResolver(timelapseSchema),
    // Validate ONLY on submit, not on field change. The Run button is
    // already disabled with a tooltip ("Pick a folder first") when a
    // required field is missing, so an aggressive red label on the
    // folder field whenever a user picks a classifier first is just
    // noise — users do not always work top-to-bottom.
    mode: "onSubmit",
    reValidateMode: "onSubmit",
    defaultValues: {
      folder_path: readQueryFolder(),
      detection_model_id: "MD5A-0-0",
      classification_model_id: NO_CLASSIFIER,
      excluded_classes: [],
      country_code: null,
      state_code: null,
      // null = use the subprocess default. Same convention as the main
      // app's project settings. The Performance card lets the user
      // override.
      detection_batch_size: null,
      classification_batch_size: null,
      video_fps: 1.0,
      event_smoothing: true,
      smoothing_strength: "normal",
      taxonomic_rollup: true,
    },
  });

  const folderPath = form.watch("folder_path");
  const detectionModelId = form.watch("detection_model_id");
  const classificationModelId = form.watch("classification_model_id");
  const hasClassifier =
    !!classificationModelId && classificationModelId !== NO_CLASSIFIER;
  const excludedClasses = form.watch("excluded_classes") ?? [];

  const { data: detectionModels = [] } = useQuery({
    queryKey: ["models", "detection"],
    queryFn: modelsApi.listDetectionModels,
  });

  const { data: classificationModels = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: modelsApi.listClassificationModels,
  });

  const detectionModel = detectionModels.find(
    (m) => m.model_id === detectionModelId,
  );
  const classificationModel = classificationModels.find(
    (m) => m.model_id === classificationModelId,
  );

  const { data: detectionStatus } = useQuery({
    queryKey: ["model-status", detectionModelId],
    queryFn: () => modelsApi.getModelStatus(detectionModelId),
    enabled: !!detectionModelId,
  });

  const { data: classificationStatus } = useQuery({
    queryKey: ["model-status", classificationModelId],
    queryFn: () => modelsApi.getModelStatus(classificationModelId!),
    enabled: hasClassifier,
  });

  const { data: taxonomy } = useQuery({
    queryKey: ["taxonomy", classificationModelId],
    queryFn: () => modelsApi.getTaxonomy(classificationModelId!),
    enabled: hasClassifier,
  });

  // WebSocket progress hook for model preparation. Same hook the main
  // app uses; the only difference is which job_id we pass it.
  const prepProgress = useTaskProgress({
    taskId: preparingTaskId,
    onComplete: () => {
      queryClient.invalidateQueries({
        queryKey: ["model-status", preparingModelId],
      });
      setPreparingTaskId(null);
      setPreparingModelId(null);
      setStage("form");
    },
    onError: (msg) => {
      setPreparationError(msg);
      setStage("error");
      setPreparingTaskId(null);
    },
  });

  // WebSocket progress hook for the actual Timelapse run.
  const runProgress = useTaskProgress({
    taskId: jobId,
    onComplete: (data) => {
      const path =
        data && typeof data === "object" && "output_path" in data
          ? String((data as { output_path: unknown }).output_path)
          : null;
      setOutputPath(path);
      setStage("done");
    },
    onError: (msg) => {
      setErrorMessage(msg);
      setStage("form");
      setIsCancelling(false);
    },
    onCancelled: (msg) => {
      // Cancellation snaps back to the form so the user can try again
      // with different settings. The backend has already torn down the
      // subprocess and emitted the cancelled message.
      setErrorMessage(msg || "Analysis cancelled");
      setStage("form");
      setIsCancelling(false);
      setJobId(null);
    },
  });

  const startPrepare = async (modelId: string) => {
    try {
      setPreparingModelId(modelId);
      setStage("preparing");
      const response = (await modelsApi.prepareModel(modelId)) as {
        task_id: string;
      };
      setPreparingTaskId(response.task_id);
    } catch (err) {
      setPreparationError(
        err instanceof Error ? err.message : String(err),
      );
      setStage("error");
    }
  };

  const cancelPrepare = () => {
    setPreparingTaskId(null);
    setPreparingModelId(null);
    setStage("form");
  };

  const retryPrepare = () => {
    if (preparingModelId) {
      setPreparationError(null);
      startPrepare(preparingModelId);
    }
  };

  const startRun = useMutation({
    mutationFn: (data: TimelapseFormData) =>
      timelapseApi.run({
        folder_path: data.folder_path,
        classification_model_id:
          data.classification_model_id === NO_CLASSIFIER
            ? null
            : data.classification_model_id,
        detection_model_id: data.detection_model_id,
        excluded_classes: data.excluded_classes,
        detection_batch_size: data.detection_batch_size,
        classification_batch_size: data.classification_batch_size,
        video_fps: data.video_fps,
        smoothing_strength: data.event_smoothing
          ? (data.smoothing_strength as SmoothingStrength)
          : "off",
        taxonomic_rollup: data.taxonomic_rollup,
      }),
    onMutate: () => {
      setErrorMessage(null);
      setOutputPath(null);
      setStage("running");
    },
    onSuccess: (response) => setJobId(response.job_id),
    onError: (err: Error) => {
      setErrorMessage(err.message);
      setStage("form");
    },
  });

  const onSubmit = (data: TimelapseFormData) => startRun.mutate(data);

  // Setup is required before submit when either selected model is not
  // ready. The Run button is disabled in that state and the per-model
  // ModelStatusBadge takes the user through preparation.
  const detReady = detectionStatus?.status === "ready";
  const clsReady = !hasClassifier || classificationStatus?.status === "ready";
  const canRun = !!folderPath && detReady && clsReady;

  // Model preparation runs in an overlay Dialog so the form stays
  // mounted behind it (same pattern CreateProjectDialog uses). The
  // "running" and "done" stages take over the page because the form
  // is no longer relevant once analysis has started.
  const prepModel =
    preparingModelId === detectionModelId
      ? detectionModel
      : classificationModel;

  if (stage === "running") {
    // Re-uses the same per-phase progress UI the main app's
    // RunQueueModal renders (extracted into AnalysisProgress.tsx so
    // both call sites stay in sync). Timelapse runs are always single-
    // deployment, so we hide the "Deployment 1 of 1" badge.
    return (
      <PageShell>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-medium">
              {isCancelling ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Loader2 className="h-4 w-4 animate-spin" />
              )}
              <span>{isCancelling ? "Cancelling..." : "Running analysis"}</span>
            </div>
            <Button
              variant="outline"
              size="sm"
              disabled={isCancelling}
              onClick={() => {
                setIsCancelling(true);
                runProgress.cancel();
              }}
            >
              Cancel
            </Button>
          </div>
          <AnalysisProgress
            phase={runProgress.phase}
            phaseProgress={runProgress.phaseProgress}
            metrics={runProgress.metrics}
            computeDevice={runProgress.computeDevice}
            deploymentContext={runProgress.deploymentContext}
            message={runProgress.message}
            hideDeploymentHeader
          />
        </div>
      </PageShell>
    );
  }

  if (stage === "done" && outputPath) {
    return (
      <PageShell>
        <SuccessCard
          outputPath={outputPath}
          onRunAnother={() => {
            // Form state is preserved across stage transitions because
            // TimelapseFormPage itself does not unmount — useForm keeps
            // the user's classifier, label exclusions, and advanced
            // settings between runs. Only the folder is reset so the
            // user is forced to pick a new one (re-running the same
            // folder by accident would just overwrite the previous
            // timelapse_recognition_file.json with no warning).
            setJobId(null);
            setOutputPath(null);
            setErrorMessage(null);
            form.setValue("folder_path", "");
            form.clearErrors();
            setAdvancedOpen(false);
            setStage("form");
          }}
        />
      </PageShell>
    );
  }

  return (
    <PageShell>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
          <TooltipProvider>
            {/* Required inputs: folder, classifier, labels. Rendered as
                plain widgets (no card chrome) so they get visual weight
                proportional to their importance — they're what every
                user fills in. The page header already frames the
                section, so a redundant card title would just add noise. */}
            <div className="space-y-0 divide-y border-y">
                {/* Folder */}
                <FormField
                  control={form.control}
                  name="folder_path"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Folder</FormLabel>
                        <FormDescription className="text-sm">
                          Analyses all images and videos found recursively
                          in this folder.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <FormControl>
                          <FolderSelector
                            value={field.value || null}
                            onChange={field.onChange}
                            hideLabel
                            hideGps
                            hideDatetimeWarning
                            compactScanResult
                          />
                        </FormControl>
                        <FormMessage />
                      </div>
                    </div>
                  )}
                />

                {/* Classification model */}
                <FormField
                  control={form.control}
                  name="classification_model_id"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Classification model</FormLabel>
                        <FormDescription className="text-sm">
                          Identifies the species behind each animal detection.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <div className="flex gap-2 items-stretch">
                          <Select
                            onValueChange={(val) =>
                              field.onChange(val === NO_CLASSIFIER ? NO_CLASSIFIER : val)
                            }
                            value={field.value ?? NO_CLASSIFIER}
                          >
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectItem value={NO_CLASSIFIER}>
                                ∅ No classification model
                                <br />
                                <span className="text-xs text-muted-foreground">
                                  Run animal detector only, identify species manually
                                </span>
                              </SelectItem>
                              <ClassificationModelGroupedItems
                                models={classificationModels.filter(
                                  (m) => m.model_id !== "none",
                                )}
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
                                  onClick={() =>
                                    hasClassifier && setShowClsInfo(true)
                                  }
                                  disabled={!hasClassifier}
                                >
                                  <InfoIcon className="h-4 w-4" />
                                </Button>
                              </span>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>
                                {hasClassifier
                                  ? "View model information"
                                  : "Select a classification model to view details"}
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        {hasClassifier &&
                          classificationStatus &&
                          classificationStatus.status !== "ready" && (
                            <ModelStatusBadge
                              status={classificationStatus}
                              onPrepare={() =>
                                startPrepare(classificationModelId!)
                              }
                              isPreparing={false}
                            />
                          )}
                        <FormMessage />
                      </div>
                    </div>
                  )}
                />

                {/* Label selection */}
                {hasClassifier && taxonomy && (
                  <div className="grid grid-cols-2 items-center gap-8 py-6">
                    <div className="space-y-1">
                      <FormLabel>Label selection</FormLabel>
                      <FormDescription className="text-sm">
                        Limit predictions to labels expected in your project
                        area to reduce false positives.
                      </FormDescription>
                    </div>
                    <div>
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => setLabelModalOpen(true)}
                        className="w-full min-h-14 flex flex-col items-start justify-center gap-1 text-left"
                      >
                        <div className="flex items-center gap-2">
                          <ListTodo className="h-4 w-4" />
                          <span>Select labels</span>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          Currently included{" "}
                          {(taxonomy.all_classes?.length || 0) -
                            excludedClasses.length}{" "}
                          of {taxonomy.all_classes?.length || 0}
                        </span>
                      </Button>
                    </div>
                  </div>
                )}
            </div>

            {/* Advanced settings: collapsed by default. Defaults are
                tuned to match the main AddaxAI app, so most users never
                open this. The disclosure pattern matches CreateProjectDialog's
                one-shot form mental model better than a heavy second
                card would.
                Two settings are deliberately hardcoded and not exposed:
                  - Detection confidence threshold (0.1, matching the main
                    app worker) — Timelapse handles user-facing filtering.
                  - Independence interval (1800s, main app default) — only
                    affects the sequence-level smoother, has no other
                    user-visible effect on the output JSON. */}
            <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
              <CollapsibleTrigger asChild>
                <button
                  type="button"
                  className="flex items-center gap-2 py-3 text-left text-sm font-semibold hover:text-primary transition-colors"
                >
                  <span>Advanced settings</span>
                  <ChevronDown
                    className={`h-4 w-4 transition-transform ${
                      advancedOpen ? "rotate-180" : ""
                    }`}
                  />
                </button>
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-0 divide-y border-t">
                {/* Detection model */}
                <FormField
                  control={form.control}
                  name="detection_model_id"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Detection model</FormLabel>
                        <FormDescription className="text-sm">
                          The model that finds animals, people, and vehicles.
                          MegaDetector 5a is the default and works well in
                          most regions.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <div className="flex gap-2 items-stretch">
                          <Select
                            onValueChange={field.onChange}
                            value={field.value}
                          >
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              {detectionModels.map((m) => (
                                <SelectItem
                                  key={m.model_id}
                                  value={m.model_id}
                                >
                                  {m.emoji} {m.friendly_name}
                                  {m.description_short && (
                                    <>
                                      <br />
                                      <span className="text-xs text-muted-foreground">
                                        {m.description_short}
                                      </span>
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
                                  onClick={() => setShowDetInfo(true)}
                                  disabled={!field.value}
                                >
                                  <InfoIcon className="h-4 w-4" />
                                </Button>
                              </span>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>View model information</p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        {detectionStatus &&
                          detectionStatus.status !== "ready" && (
                            <ModelStatusBadge
                              status={detectionStatus}
                              onPrepare={() => startPrepare(detectionModelId)}
                              isPreparing={false}
                            />
                          )}
                        <FormMessage />
                      </div>
                    </div>
                  )}
                />

                {detectionModel && (
                  <BatchSizeRow
                    control={form.control}
                    name="detection_batch_size"
                    label="Detection batch size"
                    description="Images processed per batch by the detection model. Higher values are faster but use more memory."
                    defaultGpu={detectionModel.default_batch_size_gpu}
                    defaultCpu={detectionModel.default_batch_size_cpu}
                  />
                )}

                {hasClassifier && classificationModel && (
                  <BatchSizeRow
                    control={form.control}
                    name="classification_batch_size"
                    label="Classification batch size"
                    description="Crops processed per batch by the classification model. Higher values are faster but use more memory."
                    defaultGpu={classificationModel.default_batch_size_gpu}
                    defaultCpu={classificationModel.default_batch_size_cpu}
                  />
                )}

                <FormField
                  control={form.control}
                  name="video_fps"
                  render={({ field }) => (
                    <SettingRow
                      label="Video frame rate"
                      description="How many frames per second to extract from videos for detection. Higher values find more but take longer."
                    >
                      <Select
                        key={String(field.value)}
                        value={String(field.value)}
                        onValueChange={(v) => field.onChange(parseFloat(v))}
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

                {hasClassifier && (
                  <SettingRow
                    label="Smoothing"
                    description="Cleans up classification labels at the image and event level. Off / mild / normal / aggressive controls how much weight outliers get."
                  >
                    <Select
                      value={
                        form.watch("event_smoothing")
                          ? form.watch("smoothing_strength")
                          : "off"
                      }
                      onValueChange={(value) => {
                        if (!value) return;
                        if (value === "off") {
                          form.setValue("event_smoothing", false, {
                            shouldDirty: true,
                          });
                        } else {
                          form.setValue("event_smoothing", true, {
                            shouldDirty: true,
                          });
                          form.setValue(
                            "smoothing_strength",
                            value as "mild" | "normal" | "aggressive",
                            { shouldDirty: true },
                          );
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
                  </SettingRow>
                )}

                {hasClassifier && (
                  <FormField
                    control={form.control}
                    name="taxonomic_rollup"
                    render={({ field }) => (
                      <SettingRow
                        label="Taxonomic rollup"
                        description="When the model is unsure at species level, sums probabilities up the taxonomy tree and picks the most specific level above the confidence threshold."
                      >
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </SettingRow>
                    )}
                  />
                )}
              </CollapsibleContent>
            </Collapsible>

            {errorMessage && (
              <p className="text-sm font-medium text-destructive">
                {errorMessage}
              </p>
            )}

            <div className="flex items-center justify-end gap-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <span>
                    <Button
                      type="submit"
                      disabled={!canRun || startRun.isPending}
                      className="gap-2"
                    >
                      {startRun.isPending && (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      )}
                      Run analysis
                    </Button>
                  </span>
                </TooltipTrigger>
                {!canRun && (
                  <TooltipContent>
                    <p>
                      {!folderPath
                        ? "Pick a folder first"
                        : "Models need preparing first"}
                    </p>
                  </TooltipContent>
                )}
              </Tooltip>
            </div>
          </TooltipProvider>
        </form>
      </Form>

      {/* Slide-out info drawers (same component the main app uses). */}
      <ModelInfoSheet
        modelId={detectionModelId}
        open={showDetInfo}
        onOpenChange={setShowDetInfo}
      />
      <ModelInfoSheet
        modelId={hasClassifier ? classificationModelId : null}
        open={showClsInfo}
        onOpenChange={setShowClsInfo}
      />

      {/* Label selection modal — same one used in CreateProjectDialog
          and SettingsPage. Includes the country/state geofilter. */}
      {hasClassifier && taxonomy && (
        <SpeciesSelectionModal
          modelId={classificationModelId!}
          excludedClasses={excludedClasses}
          onExclusionChange={(classes) =>
            form.setValue("excluded_classes", classes, { shouldDirty: true })
          }
          open={labelModalOpen}
          onOpenChange={setLabelModalOpen}
          totalSpeciesCount={taxonomy.all_classes?.length || 0}
          countryCode={form.watch("country_code")}
          stateCode={form.watch("state_code")}
          onLocationChange={(country, state) => {
            form.setValue("country_code", country, { shouldDirty: true });
            form.setValue("state_code", state, { shouldDirty: true });
          }}
        />
      )}

      {/* Model preparation overlay. ModelPreparationView and
          ModelPreparationErrorView are built around DialogTitle /
          DialogDescription primitives, so they need a parent <Dialog>
          context. Match CreateProjectDialog's pattern: a non-dismissable
          overlay that swaps content based on stage. */}
      <Dialog
        open={stage === "preparing" || stage === "error"}
        onOpenChange={(open) => {
          // Closing via the overlay click / Esc cancels prep, mirroring
          // the explicit Cancel button.
          if (!open) {
            if (stage === "preparing") cancelPrepare();
            if (stage === "error") {
              setPreparationError(null);
              setStage("form");
            }
          }
        }}
      >
        <DialogContent className="max-w-xl">
          {stage === "preparing" && preparingModelId && (
            <ModelPreparationView
              modelName={prepModel?.friendly_name ?? preparingModelId}
              modelEmoji={prepModel?.emoji ?? "📦"}
              progress={prepProgress.progress}
              message={prepProgress.message}
              onCancel={cancelPrepare}
            />
          )}
          {stage === "error" && (
            <ModelPreparationErrorView
              errorMessage={preparationError ?? "Unknown error"}
              onRetry={retryPrepare}
              onCancel={() => {
                setPreparationError(null);
                setStage("form");
              }}
            />
          )}
        </DialogContent>
      </Dialog>
    </PageShell>
  );
}

function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen">
      <Breadcrumbs />
      {/* Tall AddaxAI + Timelapse lockup centered as a hero. The logo
          already contains both wordmarks, so we drop the redundant
          "Timelapse integration" h1 — the artwork carries the title role.
          Subtitle keeps the verb-led action sentence so a first-time
          user knows what the page does. */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="relative mx-auto max-w-5xl px-4 py-10 sm:px-6 lg:px-8 text-center">
          <img
            src="/branding/addaxai-timelapse-logo-tall.png"
            alt="AddaxAI + Timelapse"
            className="h-48 w-auto mx-auto"
          />
          <p className="text-lg text-muted-foreground mt-4">
            Use AddaxAI to identify wildlife in your Timelapse projects
          </p>
          {/* BETA-ONLY: bug-report shortcut. Mirrors the main app's
              in-header bug icon so testers don't have to hop windows to
              email a diagnostic bundle. Pinned to the content column's
              right edge (parent is max-w-5xl) so it sits next to the
              form on wide monitors instead of drifting off into empty
              space. Remove once the Timelapse integration ships out of
              beta. */}
          <div className="absolute top-4 right-4 sm:right-6 lg:right-8">
            <DiagnosticReportButton />
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-8 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  );
}

function SettingRow({
  label,
  description,
  children,
}: {
  label: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-2 items-center gap-8 py-6">
      <div className="space-y-1">
        <FormLabel>{label}</FormLabel>
        <FormDescription className="text-sm">{description}</FormDescription>
      </div>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

function SuccessCard({
  outputPath,
  onRunAnother,
}: {
  outputPath: string;
  onRunAnother: () => void;
}) {
  const reveal = async () => {
    if (window.electronAPI?.showItemInFolder) {
      await window.electronAPI.showItemInFolder(outputPath);
    }
  };
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-primary">
          <CheckCircle2 className="h-5 w-5" /> Analysis complete
        </CardTitle>
        <CardDescription>
          In Timelapse, go to{" "}
          <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
            Recognition
          </code>{" "}
          →{" "}
          <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
            Import recognition data for this image set
          </code>
          , then pick the file below.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 pb-6">
        <div className="font-mono text-xs break-all rounded border bg-card-background px-3 py-2">
          {outputPath}
        </div>
        <div className="flex gap-2">
          {window.electronAPI && (
            <Button variant="outline" onClick={reveal}>
              <FolderOpen className="h-4 w-4 mr-2" />
              Reveal in Explorer
            </Button>
          )}
          <Button onClick={onRunAnother}>Run another folder</Button>
        </div>
      </CardContent>
    </Card>
  );
}

