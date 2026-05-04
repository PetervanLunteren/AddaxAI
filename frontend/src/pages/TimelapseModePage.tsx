/**
 * Timelapse Analyser integration page.
 *
 * Layout matches the main app exactly:
 * - canonical header / max-w-7xl main wrapper (FRONTEND_CONVENTIONS.md)
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
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

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

const INDEPENDENCE_INTERVAL_OPTIONS = [
  { value: "0", label: "Disabled" },
  { value: "60", label: "1 minute" },
  { value: "300", label: "5 minutes" },
  { value: "900", label: "15 minutes" },
  { value: "1800", label: "30 minutes" },
  { value: "3600", label: "60 minutes" },
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

const timelapseSchema = z.object({
  folder_path: z.string().min(1, "Select a folder"),
  detection_model_id: z.string().min(1),
  classification_model_id: z.string().nullable(),
  excluded_classes: z.array(z.string()),
  country_code: z.string().nullable(),
  state_code: z.string().nullable(),
  detection_threshold: z.number().min(0).max(1),
  detection_batch_size: z.number().int().min(1).max(256),
  classification_batch_size: z.number().int().min(1).max(256),
  video_fps: z.number().min(0.1).max(10),
  independence_interval: z.number().min(0),
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

  // Model preparation state (mirrors CreateProjectDialog).
  const [preparingModelId, setPreparingModelId] = useState<string | null>(null);
  const [preparingTaskId, setPreparingTaskId] = useState<string | null>(null);
  const [preparationError, setPreparationError] = useState<string | null>(null);

  // Run state.
  const [jobId, setJobId] = useState<string | null>(null);
  const [outputPath, setOutputPath] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const form = useForm<TimelapseFormData>({
    resolver: zodResolver(timelapseSchema),
    defaultValues: {
      folder_path: readQueryFolder(),
      detection_model_id: "MD5A-0-0",
      classification_model_id: NO_CLASSIFIER,
      excluded_classes: [],
      country_code: null,
      state_code: null,
      detection_threshold: 0.5,
      detection_batch_size: 1,
      classification_batch_size: 16,
      video_fps: 1.0,
      independence_interval: 1800,
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
        detection_threshold: data.detection_threshold,
        detection_batch_size: data.detection_batch_size,
        classification_batch_size: data.classification_batch_size,
        video_fps: data.video_fps,
        independence_interval: data.independence_interval,
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

  // Stages other than "form" replace the form area entirely.
  if (stage === "preparing" && preparingModelId) {
    const model =
      preparingModelId === detectionModelId
        ? detectionModel
        : classificationModel;
    return (
      <PageShell>
        <Card>
          <CardContent className="py-6">
            <ModelPreparationView
              modelName={model?.friendly_name ?? preparingModelId}
              modelEmoji={model?.emoji ?? "📦"}
              progress={prepProgress.progress}
              message={prepProgress.message}
              onCancel={cancelPrepare}
            />
          </CardContent>
        </Card>
      </PageShell>
    );
  }

  if (stage === "error") {
    return (
      <PageShell>
        <Card>
          <CardContent className="py-6">
            <ModelPreparationErrorView
              errorMessage={preparationError ?? "Unknown error"}
              onRetry={retryPrepare}
              onCancel={() => {
                setPreparationError(null);
                setStage("form");
              }}
            />
          </CardContent>
        </Card>
      </PageShell>
    );
  }

  if (stage === "running") {
    return (
      <PageShell>
        <RunningCard
          phase={runProgress.phase}
          phaseProgress={runProgress.phaseProgress}
          message={runProgress.message}
          metricsLine={runProgress.metrics?.raw_line ?? ""}
        />
      </PageShell>
    );
  }

  if (stage === "done" && outputPath) {
    return (
      <PageShell>
        <SuccessCard
          outputPath={outputPath}
          onRunAnother={() => {
            setJobId(null);
            setOutputPath(null);
            setErrorMessage(null);
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
            {/* Card 1: source data + models. */}
            <Card>
              <CardHeader>
                <CardTitle>Analysis input</CardTitle>
                <CardDescription>
                  Pick a folder and the models to run on it.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-0 divide-y border-t">
                {/* Folder */}
                <FormField
                  control={form.control}
                  name="folder_path"
                  render={({ field }) => (
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Folder</FormLabel>
                        <FormDescription className="text-sm">
                          Analyses all images and videos found recursively in
                          this folder. Camera-trap timestamps are read from
                          EXIF; files without a capture date will block the
                          run.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <FormControl>
                          <FolderSelector
                            value={field.value || null}
                            onChange={field.onChange}
                            hideLabel
                          />
                        </FormControl>
                        <FormMessage />
                      </div>
                    </div>
                  )}
                />

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
                          Choose a model trained on species from your
                          geographic region, or skip to keep raw detections.
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
                                  Run detector only, identify species manually
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
              </CardContent>
            </Card>

            {/* Card 2: Performance (batch sizes). */}
            <Card>
              <CardHeader>
                <CardTitle>Performance</CardTitle>
                <CardDescription>
                  How many crops each model processes in parallel. Higher
                  values are faster but use more memory.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-0 divide-y border-t">
                <FormField
                  control={form.control}
                  name="detection_batch_size"
                  render={({ field }) => (
                    <SettingRow
                      label="Detection batch size"
                      description="Crops processed per batch by the detection model."
                    >
                      <Input
                        type="number"
                        min={1}
                        max={256}
                        value={field.value}
                        onChange={(e) =>
                          field.onChange(parseInt(e.target.value, 10) || 1)
                        }
                      />
                      <FormMessage />
                    </SettingRow>
                  )}
                />
                <FormField
                  control={form.control}
                  name="classification_batch_size"
                  render={({ field }) => (
                    <SettingRow
                      label="Classification batch size"
                      description="Crops processed per batch by the classification model."
                    >
                      <Input
                        type="number"
                        min={1}
                        max={256}
                        value={field.value}
                        onChange={(e) =>
                          field.onChange(parseInt(e.target.value, 10) || 1)
                        }
                      />
                      <FormMessage />
                    </SettingRow>
                  )}
                />
              </CardContent>
            </Card>

            {/* Card 3: Analysis and counting. */}
            <Card>
              <CardHeader>
                <CardTitle>Analysis and counting</CardTitle>
                <CardDescription>
                  Detection threshold, video frame rate, event grouping, and
                  classification cleanup.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-0 divide-y border-t">
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

                <FormField
                  control={form.control}
                  name="detection_threshold"
                  render={({ field }) => (
                    <SettingRow
                      label="Detection confidence threshold"
                      description="Hide detections below this confidence score."
                    >
                      <div className="flex items-center justify-between">
                        <Slider
                          min={0.1}
                          max={1.0}
                          step={0.01}
                          value={[field.value]}
                          onValueChange={(vals) => field.onChange(vals[0])}
                          className="flex-1 mr-4"
                        />
                        <span className="text-sm font-medium min-w-[3rem] text-right">
                          {field.value.toFixed(2)}
                        </span>
                      </div>
                      <FormMessage />
                    </SettingRow>
                  )}
                />

                <FormField
                  control={form.control}
                  name="independence_interval"
                  render={({ field }) => (
                    <SettingRow
                      label="Independence interval"
                      description="Consecutive detections within this window are merged into one independent event. The count for each event uses MaxN."
                    >
                      <Select
                        key={String(field.value)}
                        value={String(field.value)}
                        onValueChange={(v) => field.onChange(parseInt(v, 10))}
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
              </CardContent>
            </Card>

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
    </PageShell>
  );
}

function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen">
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              Timelapse mode
            </h1>
            <p className="text-sm text-muted-foreground">
              Run AddaxAI on a folder and write a results.json that Timelapse
              can import.
            </p>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
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

function RunningCard({
  phase,
  phaseProgress,
  message,
  metricsLine,
}: {
  phase: string | null;
  phaseProgress: number;
  message: string;
  metricsLine: string;
}) {
  const pct = Math.round((phaseProgress || 0) * 100);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Loader2 className="h-5 w-5 animate-spin" /> Running analysis
        </CardTitle>
        <CardDescription>
          {phase
            ? phase.replace(/_/g, " ")
            : "Starting"}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 pb-6">
        <Progress value={pct} className="h-2" />
        <div className="text-xs text-right text-muted-foreground">{pct}%</div>
        {(metricsLine || message) && (
          <pre className="text-xs text-muted-foreground whitespace-pre-wrap font-mono">
            {metricsLine || message}
          </pre>
        )}
      </CardContent>
    </Card>
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
          In Timelapse, open <strong>Recognition</strong> &gt;{" "}
          <strong>Import recognition data for this image set</strong> and pick
          this file.
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

