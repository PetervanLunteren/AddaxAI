/**
 * Step 2: Configure analysis.
 *
 * Mirrors the Timelapse integration form (TimelapseModePage) layout
 * widget-for-widget so users moving between the two paths see the
 * same titles, captions, and controls. The folder picker stays on
 * Step 1; this step is the single place where every run-time knob
 * lives AND where the analysis actually kicks off — the dedicated
 * "Analysis" step was merged in (it was a dead page hosting a single
 * Play button before the modal opened).
 *
 * Main fields (always visible):
 * - Classification model (info button + status badge + grouped items)
 * - Label selection (SpeciesSelectionModal with country / state
 *   geofilter), only when a classifier is picked
 *
 * Advanced (collapsed by default; defaults are tuned for the common
 * case):
 * - Detection model + status badge
 * - Detection / classification / embedding batch sizes (BatchSizeRow)
 * - Embedding model + status badge
 * - Video frame rate
 * - Smoothing (Off / Mild / Normal / Aggressive)
 * - Taxonomic rollup
 *
 * Start analysis: PATCH the project row, then POST
 * /api/deployment-queue/process, then open ``RunQueueModal`` (the
 * single source of truth for the run-progress + terminal-state UI
 * shared with research-projects mode). The modal's terminal footer
 * is overridden so the user gets a "Continue" button that bumps the
 * folder-run step to "overview" and navigates forward.
 *
 * Model preparation lives on this step (same as Timelapse): each
 * picker has a status badge that opens an inline prep overlay.
 * ``Start analysis`` is disabled until every selected model reports
 * ready, so download / env-build failures surface here and never
 * reach the worker.
 */

import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import * as z from "zod";
import {
  ArrowLeft,
  ChevronDown,
  InfoIcon,
  ListTodo,
  Loader2,
  Play,
} from "lucide-react";

import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "../../components/ui/collapsible";
import { Dialog, DialogContent } from "../../components/ui/dialog";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormLabel,
  FormMessage,
} from "../../components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../components/ui/select";
import { Slider } from "../../components/ui/slider";
import { Switch } from "../../components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../../components/ui/tooltip";

import { BatchSizeRow } from "../../components/analyses/BatchSizeRow";
import { RunQueueModal } from "../../components/analyses/RunQueueModal";
import { StepHeader } from "../../components/folder-run/StepHeader";
import { ClassificationModelGroupedItems } from "../../components/models/ClassificationModelGroupedItems";
import { ModelInfoSheet } from "../../components/models/ModelInfoSheet";
import { ModelPreparationErrorView } from "../../components/projects/ModelPreparationErrorView";
import { ModelPreparationView } from "../../components/projects/ModelPreparationView";
import { ModelStatusBadge } from "../../components/projects/ModelStatusBadge";
import { SpeciesSelectionModal } from "../../components/taxonomy/SpeciesSelectionModal";

import { useTaskProgress } from "../../hooks/useTaskProgress";

import { deploymentQueueApi } from "../../api/deployment-queue";
import { folderRunsApi } from "../../api/folder-runs";
import { modelsApi } from "../../api/models";
import { projectsApi } from "../../api/projects";

import { useFolderRun } from "./FolderRunLayout";

const NO_CLASSIFIER = "none";
const NO_EMBEDDING = "none";

const VIDEO_FPS_OPTIONS = [
  { value: "0.1", label: "1 frame every 10 seconds" },
  { value: "0.25", label: "1 frame every 4 seconds" },
  { value: "0.5", label: "1 frame every 2 seconds" },
  { value: "1", label: "1 frame per second" },
  { value: "2", label: "2 frames per second" },
  { value: "4", label: "4 frames per second" },
  { value: "10", label: "10 frames per second" },
];

const settingsSchema = z.object({
  detection_model_id: z.string().min(1),
  classification_model_id: z.string().nullable(),
  embedding_model_id: z.string().nullable(),
  excluded_classes: z.array(z.string()),
  country_code: z.string().nullable(),
  state_code: z.string().nullable(),
  detection_batch_size: z.number().int().min(1).max(256).nullable(),
  classification_batch_size: z.number().int().min(1).max(256).nullable(),
  embedding_batch_size: z.number().int().min(1).max(256).nullable(),
  detection_threshold: z.number().min(0.1).max(1),
  video_fps: z.number().min(0.1).max(10),
  event_smoothing: z.boolean(),
  smoothing_strength: z.enum(["mild", "normal", "aggressive"]),
  taxonomic_rollup: z.boolean(),
});

type SettingsFormData = z.infer<typeof settingsSchema>;

type PrepStage = "form" | "preparing" | "error";

export function FolderRunModelStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId, run, isLoading } = useFolderRun();

  // Model preparation state — mirrors TimelapseModePage / CreateProjectDialog.
  const [prepStage, setPrepStage] = useState<PrepStage>("form");
  const [preparingModelId, setPreparingModelId] = useState<string | null>(null);
  const [preparingTaskId, setPreparingTaskId] = useState<string | null>(null);
  const [preparationError, setPreparationError] = useState<string | null>(null);

  const [labelModalOpen, setLabelModalOpen] = useState(false);
  const [showDetInfo, setShowDetInfo] = useState(false);
  const [showClsInfo, setShowClsInfo] = useState(false);
  const [showEmbInfo, setShowEmbInfo] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Run-modal state — populated when Start analysis kicks off the
  // worker. RunQueueModal handles the whole running + terminal-state
  // flow (single source of truth shared with research-projects mode).
  const [runState, setRunState] = useState<
    { jobIds: string[]; queueEntryIds: string[] } | null
  >(null);

  const { data: detectionModels = [] } = useQuery({
    queryKey: ["models", "detection"],
    queryFn: modelsApi.listDetectionModels,
  });
  const { data: classificationModels = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: modelsApi.listClassificationModels,
  });
  const { data: embeddingModels = [] } = useQuery({
    queryKey: ["models", "embedding"],
    queryFn: modelsApi.listEmbeddingModels,
  });

  const form = useForm<SettingsFormData>({
    resolver: zodResolver(settingsSchema),
    mode: "onSubmit",
    reValidateMode: "onSubmit",
    defaultValues: {
      detection_model_id: "MD5A-0-0",
      classification_model_id: NO_CLASSIFIER,
      embedding_model_id: NO_EMBEDDING,
      excluded_classes: [],
      country_code: null,
      state_code: null,
      detection_batch_size: null,
      classification_batch_size: null,
      embedding_batch_size: null,
      detection_threshold: 0.5,
      video_fps: 1.0,
      event_smoothing: true,
      smoothing_strength: "normal",
      taxonomic_rollup: true,
    },
  });

  // Seed from the project row so resume lands on the user's prior
  // selection rather than the bare defaults.
  useEffect(() => {
    if (!run) return;
    form.reset({
      detection_model_id: run.project.detection_model_id,
      classification_model_id:
        run.project.classification_model_id ?? NO_CLASSIFIER,
      embedding_model_id: run.project.embedding_model_id ?? NO_EMBEDDING,
      excluded_classes: run.project.excluded_classes ?? [],
      country_code: run.project.country_code,
      state_code: run.project.state_code,
      detection_batch_size: run.project.detection_batch_size ?? null,
      classification_batch_size:
        run.project.classification_batch_size ?? null,
      embedding_batch_size: run.project.embedding_batch_size ?? null,
      detection_threshold: run.project.detection_threshold,
      video_fps: run.project.video_fps,
      event_smoothing: run.project.event_smoothing,
      smoothing_strength: (run.project.smoothing_strength ??
        "normal") as "mild" | "normal" | "aggressive",
      taxonomic_rollup: run.project.taxonomic_rollup,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.project.id]);

  const detectionModelId = form.watch("detection_model_id");
  const classificationModelId = form.watch("classification_model_id");
  const embeddingModelId = form.watch("embedding_model_id");
  const excludedClasses = form.watch("excluded_classes") ?? [];
  const hasClassifier =
    !!classificationModelId && classificationModelId !== NO_CLASSIFIER;
  const hasEmbedding =
    !!embeddingModelId && embeddingModelId !== NO_EMBEDDING;

  const detectionModel = detectionModels.find(
    (m) => m.model_id === detectionModelId,
  );
  const classificationModel = classificationModels.find(
    (m) => m.model_id === classificationModelId,
  );
  const embeddingModel = embeddingModels.find(
    (m) => m.model_id === embeddingModelId,
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
  const { data: embeddingStatus } = useQuery({
    queryKey: ["model-status", embeddingModelId],
    queryFn: () => modelsApi.getModelStatus(embeddingModelId!),
    enabled: hasEmbedding,
  });

  const { data: taxonomy } = useQuery({
    queryKey: ["taxonomy", classificationModelId],
    queryFn: () => modelsApi.getTaxonomy(classificationModelId!),
    enabled: hasClassifier,
  });

  const prepProgress = useTaskProgress({
    taskId: preparingTaskId,
    onComplete: () => {
      queryClient.invalidateQueries({
        queryKey: ["model-status", preparingModelId],
      });
      setPreparingTaskId(null);
      setPreparingModelId(null);
      setPrepStage("form");
    },
    onError: (msg) => {
      setPreparationError(msg);
      setPrepStage("error");
      setPreparingTaskId(null);
    },
  });

  const startPrepare = async (modelId: string) => {
    try {
      setPreparingModelId(modelId);
      setPrepStage("preparing");
      const response = (await modelsApi.prepareModel(modelId)) as {
        task_id: string;
      };
      setPreparingTaskId(response.task_id);
    } catch (err) {
      setPreparationError(
        err instanceof Error ? err.message : String(err),
      );
      setPrepStage("error");
    }
  };
  const cancelPrepare = () => {
    setPreparingTaskId(null);
    setPreparingModelId(null);
    setPrepStage("form");
  };
  const retryPrepare = () => {
    if (preparingModelId) {
      setPreparationError(null);
      startPrepare(preparingModelId);
    }
  };

  const startAnalysis = useMutation({
    mutationFn: async (data: SettingsFormData) => {
      if (!runId) throw new Error("missing run id");
      await projectsApi.update(runId, {
        detection_model_id: data.detection_model_id,
        classification_model_id:
          data.classification_model_id === NO_CLASSIFIER
            ? null
            : data.classification_model_id,
        embedding_model_id:
          data.embedding_model_id === NO_EMBEDDING
            ? null
            : data.embedding_model_id,
        excluded_classes: data.excluded_classes,
        country_code: data.country_code,
        state_code: data.state_code,
        detection_batch_size: data.detection_batch_size,
        classification_batch_size: data.classification_batch_size,
        embedding_batch_size: data.embedding_batch_size,
        detection_threshold: data.detection_threshold,
        video_fps: data.video_fps,
        event_smoothing: data.event_smoothing,
        smoothing_strength: data.smoothing_strength,
        taxonomic_rollup: data.taxonomic_rollup,
      });
      const resp = await deploymentQueueApi.process({ project_id: runId });
      return resp;
    },
    onSuccess: async (resp) => {
      queryClient.invalidateQueries({ queryKey: ["projects", runId] });
      // No pending queue entries means the deployment already ran on
      // an earlier visit (queue entry is in a terminal state). Skip
      // ahead to verification rather than spawning a redundant job.
      if (resp.jobs_started === 0 || resp.job_ids.length === 0) {
        if (runId) {
          const next = await folderRunsApi.updateStep(runId, "review");
          queryClient.setQueryData(["folder-run", runId], next);
        }
        navigate(`/folder-runs/${runId}/review`);
        return;
      }
      setRunState({
        jobIds: resp.job_ids,
        queueEntryIds: resp.queue_entry_ids,
      });
    },
  });

  const onSubmit = (data: SettingsFormData) => startAnalysis.mutate(data);

  if (!runId) {
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

  const detReady = detectionStatus?.status === "ready";
  const clsReady = !hasClassifier || classificationStatus?.status === "ready";
  const embReady = !hasEmbedding || embeddingStatus?.status === "ready";
  const canStart =
    detReady && clsReady && embReady && !startAnalysis.isPending;

  const prepModel =
    preparingModelId === detectionModelId
      ? detectionModel
      : preparingModelId === classificationModelId
        ? classificationModel
        : embeddingModel;

  return (
    <>
      <StepHeader
        title="Set up the analysis"
        caption="Pick the AI models and tune how AddaxAI will process the folder."
      />
      <Card>
        <CardContent className="space-y-6 p-6">
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(onSubmit)}
              className="space-y-6"
            >
              <TooltipProvider>
                <div className="space-y-0 divide-y border-y">
                  <FormField
                    control={form.control}
                    name="classification_model_id"
                    render={({ field }) => (
                      <div className="grid grid-cols-2 items-center gap-8 py-6">
                        <div className="space-y-1">
                          <FormLabel>Classification model</FormLabel>
                          <FormDescription className="text-sm">
                            Identifies the species behind each animal
                            detection.
                          </FormDescription>
                        </div>
                        <div className="space-y-2">
                          <div className="flex items-stretch gap-2">
                            <Select
                              onValueChange={(val) =>
                                field.onChange(
                                  val === NO_CLASSIFIER
                                    ? NO_CLASSIFIER
                                    : val,
                                )
                              }
                              value={field.value ?? NO_CLASSIFIER}
                            >
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue placeholder="Select classification model">
                                    {(() => {
                                      if (
                                        !field.value ||
                                        field.value === NO_CLASSIFIER
                                      ) {
                                        return (
                                          <div className="flex flex-col items-start py-1">
                                            <div>
                                              ∅ No classification model
                                            </div>
                                            <div className="text-xs text-muted-foreground">
                                              Run animal detector only,
                                              identify species manually
                                            </div>
                                          </div>
                                        );
                                      }
                                      const selected =
                                        classificationModels.find(
                                          (m) =>
                                            m.model_id === field.value,
                                        );
                                      if (!selected) return null;
                                      return (
                                        <div className="flex flex-col items-start py-1">
                                          <div>
                                            {selected.emoji}{" "}
                                            {selected.friendly_name}
                                          </div>
                                          {selected.description_short && (
                                            <div className="text-xs text-muted-foreground">
                                              {selected.description_short}
                                            </div>
                                          )}
                                        </div>
                                      );
                                    })()}
                                  </SelectValue>
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value={NO_CLASSIFIER}>
                                  ∅ No classification model
                                  <br />
                                  <span className="text-xs text-muted-foreground">
                                    Run animal detector only,
                                    identify species manually
                                  </span>
                                </SelectItem>
                                <ClassificationModelGroupedItems
                                  models={classificationModels.filter(
                                    (m) => m.model_id !== "none",
                                  )}
                                />
                              </SelectContent>
                            </Select>
                            {hasClassifier && (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span className="self-center">
                                    <Button
                                      type="button"
                                      variant="outline"
                                      className="px-3"
                                      onClick={() =>
                                        setShowClsInfo(true)
                                      }
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

                  {hasClassifier && taxonomy && (
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Label selection</FormLabel>
                        <FormDescription className="text-sm">
                          Limit predictions to labels expected in your
                          project area to reduce false positives.
                        </FormDescription>
                      </div>
                      <div>
                        <Button
                          type="button"
                          variant="outline"
                          onClick={() => setLabelModalOpen(true)}
                          className="flex min-h-14 w-full flex-col items-start justify-center gap-1 text-left"
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

                <Collapsible
                  open={advancedOpen}
                  onOpenChange={setAdvancedOpen}
                >
                  <CollapsibleTrigger asChild>
                    <button
                      type="button"
                      className="flex items-center gap-2 py-3 text-left text-sm font-semibold transition-colors hover:text-primary"
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
                    <FormField
                      control={form.control}
                      name="detection_model_id"
                      render={({ field }) => (
                        <div className="grid grid-cols-2 items-center gap-8 py-6">
                          <div className="space-y-1">
                            <FormLabel>Detection model</FormLabel>
                            <FormDescription className="text-sm">
                              The model that finds animals, people, and
                              vehicles. MegaDetector 5a is the default
                              and works well in most regions.
                            </FormDescription>
                          </div>
                          <div className="space-y-2">
                            <div className="flex items-stretch gap-2">
                              <Select
                                onValueChange={field.onChange}
                                value={field.value}
                              >
                                <FormControl>
                                  <SelectTrigger>
                                    <SelectValue placeholder="Select detection model">
                                      {field.value &&
                                        (() => {
                                          const selected =
                                            detectionModels.find(
                                              (m) =>
                                                m.model_id ===
                                                field.value,
                                            );
                                          if (!selected) return null;
                                          return (
                                            <div className="flex flex-col items-start py-1">
                                              <div>
                                                {selected.emoji}{" "}
                                                {selected.friendly_name}
                                              </div>
                                              {selected.description_short && (
                                                <div className="text-xs text-muted-foreground">
                                                  {
                                                    selected.description_short
                                                  }
                                                </div>
                                              )}
                                            </div>
                                          );
                                        })()}
                                    </SelectValue>
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
                              {field.value && (
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <span className="self-center">
                                      <Button
                                        type="button"
                                        variant="outline"
                                        className="px-3"
                                        onClick={() =>
                                          setShowDetInfo(true)
                                        }
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
                            {detectionStatus &&
                              detectionStatus.status !== "ready" && (
                                <ModelStatusBadge
                                  status={detectionStatus}
                                  onPrepare={() =>
                                    startPrepare(detectionModelId)
                                  }
                                  isPreparing={false}
                                />
                              )}
                            <FormMessage />
                          </div>
                        </div>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="embedding_model_id"
                      render={({ field }) => (
                        <div className="grid grid-cols-2 items-center gap-8 py-6">
                          <div className="space-y-1">
                            <FormLabel>Embedding model</FormLabel>
                            <FormDescription className="text-sm">
                              Computes a feature vector per detection
                              for similarity sort and clustering.
                              Optional; skip if you do not need those
                              features.
                            </FormDescription>
                          </div>
                          <div className="space-y-2">
                            <div className="flex items-stretch gap-2">
                              <Select
                                onValueChange={(val) =>
                                  field.onChange(
                                    val === NO_EMBEDDING
                                      ? NO_EMBEDDING
                                      : val,
                                  )
                                }
                                value={field.value ?? NO_EMBEDDING}
                              >
                                <FormControl>
                                  <SelectTrigger>
                                    <SelectValue placeholder="Select embedding model">
                                      {(() => {
                                        if (
                                          !field.value ||
                                          field.value === NO_EMBEDDING
                                        ) {
                                          return (
                                            <div className="flex flex-col items-start py-1">
                                              <div>
                                                ∅ No embedding model
                                              </div>
                                              <div className="text-xs text-muted-foreground">
                                                Skip if you do not need
                                                similarity sort or
                                                clustering
                                              </div>
                                            </div>
                                          );
                                        }
                                        const selected =
                                          embeddingModels.find(
                                            (m) =>
                                              m.model_id ===
                                              field.value,
                                          );
                                        if (!selected) return null;
                                        return (
                                          <div className="flex flex-col items-start py-1">
                                            <div>
                                              {selected.emoji}{" "}
                                              {selected.friendly_name}
                                            </div>
                                            {selected.description_short && (
                                              <div className="text-xs text-muted-foreground">
                                                {selected.description_short}
                                              </div>
                                            )}
                                          </div>
                                        );
                                      })()}
                                    </SelectValue>
                                  </SelectTrigger>
                                </FormControl>
                                <SelectContent>
                                  <SelectItem value={NO_EMBEDDING}>
                                    ∅ No embedding model
                                    <br />
                                    <span className="text-xs text-muted-foreground">
                                      Skip if you do not need
                                      similarity sort or clustering
                                    </span>
                                  </SelectItem>
                                  {embeddingModels
                                    .filter(
                                      (m) => m.model_id !== "none",
                                    )
                                    .map((m) => (
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
                              {hasEmbedding && (
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <span className="self-center">
                                      <Button
                                        type="button"
                                        variant="outline"
                                        className="px-3"
                                        onClick={() =>
                                          setShowEmbInfo(true)
                                        }
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
                            {hasEmbedding &&
                              embeddingStatus &&
                              embeddingStatus.status !== "ready" && (
                                <ModelStatusBadge
                                  status={embeddingStatus}
                                  onPrepare={() =>
                                    startPrepare(embeddingModelId!)
                                  }
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
                        defaultGpu={
                          classificationModel.default_batch_size_gpu
                        }
                        defaultCpu={
                          classificationModel.default_batch_size_cpu
                        }
                      />
                    )}

                    {hasEmbedding && embeddingModel && (
                      <BatchSizeRow
                        control={form.control}
                        name="embedding_batch_size"
                        label="Embedding batch size"
                        description="Crops processed per batch by the embedding model. Higher values are faster but use more memory."
                        defaultGpu={embeddingModel.default_batch_size_gpu}
                        defaultCpu={embeddingModel.default_batch_size_cpu}
                      />
                    )}

                    <FormField
                      control={form.control}
                      name="detection_threshold"
                      render={({ field }) => (
                        <SettingRow
                          label="Detection confidence threshold"
                          description="Hide detections below this confidence score. Verified observations are always included."
                        >
                          <div className="flex items-center justify-between">
                            <Slider
                              min={0.1}
                              max={1.0}
                              step={0.01}
                              value={[field.value]}
                              onValueChange={(vals) =>
                                field.onChange(vals[0])
                              }
                              className="mr-4 flex-1"
                            />
                            <span className="min-w-[3rem] text-right text-sm font-medium">
                              {field.value.toFixed(2)}
                            </span>
                          </div>
                          <FormMessage />
                        </SettingRow>
                      )}
                    />

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
                            onValueChange={(v) =>
                              field.onChange(parseFloat(v))
                            }
                          >
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              {VIDEO_FPS_OPTIONS.map((opt) => (
                                <SelectItem
                                  key={opt.value}
                                  value={opt.value}
                                >
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
                                value as
                                  | "mild"
                                  | "normal"
                                  | "aggressive",
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
                            <SelectItem value="aggressive">
                              Aggressive
                            </SelectItem>
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

                {startAnalysis.isError && (
                  <p className="text-sm text-destructive">
                    Could not start analysis:{" "}
                    {startAnalysis.error instanceof Error
                      ? startAnalysis.error.message
                      : "unknown error"}
                  </p>
                )}

                <div className="flex items-center justify-between">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() =>
                      navigate(`/folder-runs/${runId}/folder`)
                    }
                    className="gap-2"
                  >
                    <ArrowLeft className="h-4 w-4" />
                    Back
                  </Button>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span>
                        <Button
                          type="submit"
                          disabled={!canStart}
                          className="gap-2"
                          size="lg"
                        >
                          {startAnalysis.isPending ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Play className="h-4 w-4" />
                          )}
                          {startAnalysis.isPending
                            ? "Starting..."
                            : "Start analysis"}
                        </Button>
                      </span>
                    </TooltipTrigger>
                    {!canStart && !startAnalysis.isPending && (
                      <TooltipContent>
                        <p>Models need preparing first</p>
                      </TooltipContent>
                    )}
                  </Tooltip>
                </div>
              </TooltipProvider>
            </form>
          </Form>
        </CardContent>
      </Card>

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
      <ModelInfoSheet
        modelId={hasEmbedding ? embeddingModelId : null}
        open={showEmbInfo}
        onOpenChange={setShowEmbInfo}
      />

      {hasClassifier && taxonomy && (
        <SpeciesSelectionModal
          modelId={classificationModelId!}
          excludedClasses={excludedClasses}
          onExclusionChange={(classes) =>
            form.setValue("excluded_classes", classes, {
              shouldDirty: true,
            })
          }
          open={labelModalOpen}
          onOpenChange={setLabelModalOpen}
          totalSpeciesCount={taxonomy.all_classes?.length || 0}
          countryCode={form.watch("country_code")}
          stateCode={form.watch("state_code")}
          onLocationChange={(country, state) => {
            form.setValue("country_code", country, {
              shouldDirty: true,
            });
            form.setValue("state_code", state, { shouldDirty: true });
          }}
        />
      )}

      <Dialog
        open={prepStage === "preparing" || prepStage === "error"}
        onOpenChange={(open) => {
          if (!open) {
            if (prepStage === "preparing") cancelPrepare();
            if (prepStage === "error") {
              setPreparationError(null);
              setPrepStage("form");
            }
          }
        }}
      >
        <DialogContent className="max-w-xl">
          {prepStage === "preparing" && preparingModelId && (
            <ModelPreparationView
              modelName={prepModel?.friendly_name ?? preparingModelId}
              modelEmoji={prepModel?.emoji ?? "📦"}
              progress={prepProgress.progress}
              message={prepProgress.message}
              onCancel={cancelPrepare}
            />
          )}
          {prepStage === "error" && (
            <ModelPreparationErrorView
              errorMessage={preparationError ?? "Unknown error"}
              onRetry={retryPrepare}
              onCancel={() => {
                setPreparationError(null);
                setPrepStage("form");
              }}
            />
          )}
        </DialogContent>
      </Dialog>

      {runState && runId && (
        <RunQueueModal
          open={runState !== null}
          onOpenChange={(open) => {
            if (!open) setRunState(null);
          }}
          queueCount={1}
          jobIds={runState.jobIds}
          queueEntryIds={runState.queueEntryIds}
          projectId={runId}
          mode="folder-run"
          deleteQueueEntriesOnClose={false}
          renderTerminalFooter={({ kind, close, isClosing }) => {
            const advance = async () => {
              await close();
              const next = await folderRunsApi.updateStep(
                runId,
                "review",
              );
              queryClient.setQueryData(["folder-run", runId], next);
              navigate(`/folder-runs/${runId}/review`);
            };
            if (kind === "completed") {
              return (
                <Button
                  disabled={isClosing}
                  onClick={advance}
                  className="gap-2"
                >
                  Continue
                </Button>
              );
            }
            return (
              <Button
                variant="outline"
                disabled={isClosing}
                onClick={close}
              >
                Close
              </Button>
            );
          }}
        />
      )}
    </>
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
