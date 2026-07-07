/**
 * Step 2: Configure analysis.
 *
 * The folder picker stays on Step 1; this step is the single place
 * where every run-time knob lives AND where the analysis actually
 * kicks off — the dedicated
 * "Analysis" step was merged in (it was a dead page hosting a single
 * start button before the modal opened).
 *
 * Main fields (always visible):
 * - Classification model (info button + status badge + grouped items)
 * - Label selection (LabelSelectionField with country / state
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
 * folder-run step to "labels" and navigates forward.
 *
 * Model preparation lives on this step: each picker has a status
 * badge that opens an inline prep overlay.
 * ``Start analysis`` is disabled until every selected model reports
 * ready, so download / env-build failures surface here and never
 * reach the worker.
 */

import { useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import * as z from "zod";
import {
  ChevronDown,
  Loader2,
  RotateCcw,
  Sparkles,
} from "lucide-react";

import { Button } from "../../components/ui/button";
import { Callout } from "../../components/ui/callout";
import {
  isAnyAdvancedNonDefault,
  restoreAdvancedDefaults,
} from "../../lib/advancedSettingsDefaults";
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
import { CaptionedSelect } from "../../components/ui/captioned-select";
import { SMOOTHING_LEVELS } from "../../lib/smoothing";
import { SETTING_CAPTIONS } from "../../lib/settingCaptions";
import { Slider } from "../../components/ui/slider";
import { Switch } from "../../components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../../components/ui/tooltip";

import { BatchSizeRow } from "../../components/analyses/BatchSizeRow";
import { IntervalControl } from "../../components/analyses/IntervalControl";
import { FolderSelector } from "../../components/analyses/FolderSelector";
import { RunQueueModal } from "../../components/analyses/RunQueueModal";
import { CompletedRunNotice } from "../../components/folder-run/CompletedRunNotice";
import { RerunConfirmDialog } from "../../components/folder-run/RerunConfirmDialog";
import { StepHeader } from "../../components/folder-run/StepHeader";
import { ClassificationModelGroupedItems } from "../../components/models/ClassificationModelGroupedItems";
import { ModelInfoSheet } from "../../components/models/ModelInfoSheet";
import { ModelPreparationErrorView } from "../../components/projects/ModelPreparationErrorView";
import { ModelPreparationView } from "../../components/projects/ModelPreparationView";
import { ModelStatusBadge } from "../../components/projects/ModelStatusBadge";
import {
  LabelSelectionField,
  toApiCountryCode,
  useLabelSelectionCaption,
} from "../../components/taxonomy/LabelSelectionField";
import { ModelSelect } from "../../components/models/ModelSelect";
import { NoClassifierNotice } from "../../components/models/NoClassifierNotice";

import { useFolderScan } from "../../hooks/useFolderScan";
import { useTaskProgress } from "../../hooks/useTaskProgress";

import {
  loadLastUsedSettings,
  saveLastUsedSettings,
} from "../../lib/folderRunSettings";

import { deploymentQueueApi } from "../../api/deployment-queue";
import {
  folderRunsApi,
  type FolderRunCreate,
} from "../../api/folder-runs";
import { invalidateModelMetadata, modelsApi } from "../../api/models";
import { projectsApi } from "../../api/projects";

import { useFolderRun } from "./FolderRunLayout";

const NO_CLASSIFIER = "none";
const NO_EMBEDDING = "none";
// Default embedding model for brand-new users (no saved settings yet).
// Returning users get their last-used choice from localStorage, and
// resumed runs seed from the project row.
const DEFAULT_EMBEDDING = "DINOV2-VITB14";

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
  folder_path: z.string().min(1, "Pick a folder"),
  detection_model_id: z.string().min(1),
  classification_model_id: z.string().nullable(),
  embedding_model_id: z.string().nullable(),
  excluded_classes: z.array(z.string()),
  country_code: z.string().nullable(),
  state_code: z.string().nullable(),
  detection_batch_size: z.number().int().min(1).max(256).nullable(),
  classification_batch_size: z.number().int().min(1).max(256).nullable(),
  embedding_batch_size: z.number().int().min(1).max(256).nullable(),
  video_fps: z.number().min(0.1).max(10),
  event_smoothing: z.boolean(),
  smoothing_strength: z.enum(["mild", "normal", "aggressive"]),
  taxonomic_rollup: z.boolean(),
  independence_interval: z.number().int().min(0),
});

type SettingsFormData = z.infer<typeof settingsSchema>;

type PrepStage = "form" | "preparing" | "error";

/**
 * Read a `?path=<folder>` query param. The desktop launcher
 * (`AddaxAI.exe --timelapse <folder>`, used by Timelapse Analyser) opens
 * a brand-new folder run with this set so the folder picker starts
 * pre-filled. Returns "" when absent.
 */
function readQueryFolder(): string {
  return new URLSearchParams(window.location.search).get("path") ?? "";
}

export function FolderRunModelStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId, run, isLoading } = useFolderRun();

  // Model preparation state — mirrors CreateProjectDialog.
  const [prepStage, setPrepStage] = useState<PrepStage>("form");
  const [preparingModelId, setPreparingModelId] = useState<string | null>(null);
  const [preparingTaskId, setPreparingTaskId] = useState<string | null>(null);
  const [preparationError, setPreparationError] = useState<string | null>(null);

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

  // KISS lifecycle: modal == running run. When the server reports an
  // active analysis job for this folder run (refresh / navigation back
  // mid-analysis), re-attach the progress modal so the user lands back
  // on it instead of a stale Setup form. The effect is keyed on the
  // job id so closing the modal after the run finishes (which
  // invalidates the folder-run query and flips active_job_id to null)
  // does not immediately reopen it.
  useEffect(() => {
    const jobId = run?.active_job_id;
    const queueEntryId = run?.queue_entry?.id;
    if (!jobId || !queueEntryId) return;
    setRunState((prev) => {
      if (prev && prev.jobIds[0] === jobId) return prev;
      return { jobIds: [jobId], queueEntryIds: [queueEntryId] };
    });
  }, [run?.active_job_id, run?.queue_entry?.id]);

  const { data: detectionModels = [], isLoading: detectionModelsLoading } =
    useQuery({
      queryKey: ["models", "detection"],
      queryFn: modelsApi.listDetectionModels,
    });
  const {
    data: classificationModels = [],
    isLoading: classificationModelsLoading,
  } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: modelsApi.listClassificationModels,
  });
  const { data: embeddingModels = [], isLoading: embeddingModelsLoading } =
    useQuery({
      queryKey: ["models", "embedding"],
      queryFn: modelsApi.listEmbeddingModels,
    });

  const form = useForm<SettingsFormData>({
    resolver: zodResolver(settingsSchema),
    mode: "onSubmit",
    reValidateMode: "onSubmit",
    defaultValues: {
      folder_path: "",
      detection_model_id: "MD5A-0-0",
      classification_model_id: NO_CLASSIFIER,
      embedding_model_id: DEFAULT_EMBEDDING,
      excluded_classes: [],
      country_code: null,
      state_code: null,
      detection_batch_size: null,
      classification_batch_size: null,
      embedding_batch_size: null,
      video_fps: 1.0,
      event_smoothing: true,
      smoothing_strength: "normal",
      taxonomic_rollup: true,
      independence_interval: 1800,
    },
  });

  // Seed from the project row so resume lands on the user's prior
  // selection rather than the bare defaults. The folder path comes
  // from the queue entry; we prefill exactly once so the user can
  // clear it via the FolderSelector's Change button without the
  // effect snapping it back.
  const hasPrefilledFolderRef = useRef(false);
  useEffect(() => {
    if (!run) return;
    const shouldPrefillFolder =
      !hasPrefilledFolderRef.current &&
      !!run.queue_entry?.folder_path;
    if (shouldPrefillFolder) {
      hasPrefilledFolderRef.current = true;
    }
    form.reset({
      folder_path: shouldPrefillFolder
        ? (run.queue_entry?.folder_path ?? "")
        : form.getValues("folder_path"),
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
      video_fps: run.project.video_fps,
      event_smoothing: run.project.event_smoothing,
      smoothing_strength: (run.project.smoothing_strength ??
        "normal") as "mild" | "normal" | "aggressive",
      taxonomic_rollup: run.project.taxonomic_rollup,
      independence_interval: run.project.independence_interval,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.project.id]);

  // Brand-new run (no runId): seed the form from the user's last-used
  // settings instead of bare defaults. Fires once on mount. We keep
  // any saved model ids verbatim even if the model is no longer
  // installed — the model-status badges + the warning banner below
  // surface that rather than silently swapping the choice. Resume
  // (runId set) is handled by the effect above and must not be
  // overridden here.
  const hasRestoredRef = useRef(false);
  useEffect(() => {
    if (runId || hasRestoredRef.current) return;
    hasRestoredRef.current = true;
    const saved = loadLastUsedSettings();
    // A `?path=` query (set by the desktop --timelapse launcher) pre-fills
    // the folder for a brand-new run; it wins over the empty default.
    const queryFolder = readQueryFolder();
    if (saved || queryFolder) {
      form.reset({
        ...form.getValues(),
        ...(saved ?? {}),
        folder_path: queryFolder,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  const folderPath = form.watch("folder_path") || null;
  const detectionModelId = form.watch("detection_model_id");
  const classificationModelId = form.watch("classification_model_id");
  const labelCaption = useLabelSelectionCaption(
    classificationModelId && classificationModelId !== "none" ? classificationModelId : "",
  );
  const embeddingModelId = form.watch("embedding_model_id");
  const excludedClasses = form.watch("excluded_classes") ?? [];
  const hasClassifier =
    !!classificationModelId && classificationModelId !== NO_CLASSIFIER;
  const hasEmbedding =
    !!embeddingModelId && embeddingModelId !== NO_EMBEDDING;

  // Show "Restore defaults" only when an advanced setting differs from its
  // factory default (e.g. carried over from the last run via sticky settings).
  const advancedNonDefault = isAnyAdvancedNonDefault(form.watch());

  // Folder scan + previous-run lookup. The lookup only fires after a
  // valid scan so we don't probe folders the user is mid-typing or
  // ones that turned out to be empty.
  const { data: scanResult, isLoading: isScanning } = useFolderScan(folderPath);
  const lookupReady =
    !!folderPath && !!scanResult && scanResult.total_count > 0;
  const { data: lookupRun, isFetching: isLookingUp } = useQuery({
    queryKey: ["folder-run-lookup", folderPath],
    queryFn: () => folderRunsApi.lookup(folderPath!),
    enabled: lookupReady,
    staleTime: 30_000,
  });
  // Dialog visibility for the destructive Re-run / discard flow.
  const [rerunOpen, setRerunOpen] = useState(false);
  // Set when a start / re-run finds nothing pending to process, so we
  // surface it instead of silently fast-forwarding to Edit.
  const [nothingToRun, setNothingToRun] = useState(false);

  const detectionModel = detectionModels.find(
    (m) => m.model_id === detectionModelId,
  );
  const classificationModel = classificationModels.find(
    (m) => m.model_id === classificationModelId,
  );
  const embeddingModel = embeddingModels.find(
    (m) => m.model_id === embeddingModelId,
  );

  const { data: detectionStatus, isLoading: detStatusLoading } = useQuery({
    queryKey: ["model-status", detectionModelId],
    queryFn: () => modelsApi.getModelStatus(detectionModelId),
    enabled: !!detectionModelId,
  });
  const { data: classificationStatus, isLoading: clsStatusLoading } =
    useQuery({
      queryKey: ["model-status", classificationModelId],
      queryFn: () => modelsApi.getModelStatus(classificationModelId!),
      enabled: hasClassifier,
    });
  const { data: embeddingStatus, isLoading: embStatusLoading } = useQuery({
    queryKey: ["model-status", embeddingModelId],
    queryFn: () => modelsApi.getModelStatus(embeddingModelId!),
    enabled: hasEmbedding,
  });
  // True while any relevant status query is still on its first load
  // for the current model id. Used to hold back the "needs setup"
  // warning so it doesn't flash when the user switches models (the
  // new id's status is briefly unknown).
  const statusLoading =
    detStatusLoading ||
    (hasClassifier && clsStatusLoading) ||
    (hasEmbedding && embStatusLoading);

  const { data: taxonomy } = useQuery({
    queryKey: ["taxonomy", classificationModelId],
    queryFn: () => modelsApi.getTaxonomy(classificationModelId!),
    enabled: hasClassifier,
  });

  // Same query (and cache entry) as LabelSelectionField uses internally.
  // Needed here to know whether the selected classifier is geofenced,
  // which makes the country choice a required field on submit.
  const { data: clsGeofence } = useQuery({
    queryKey: ["model-geofence", classificationModelId],
    queryFn: () => modelsApi.getModelGeofence(classificationModelId!),
    enabled: hasClassifier,
    staleTime: Infinity,
  });
  const requiresCountryChoice =
    hasClassifier && !!clsGeofence?.has_geofence && !!clsGeofence?.countries;

  const prepProgress = useTaskProgress({
    taskId: preparingTaskId,
    onComplete: () => {
      invalidateModelMetadata(queryClient, preparingModelId);
      setPreparingTaskId(null);
      setPreparingModelId(null);
      setPrepStage("form");
    },
    onError: (msg) => {
      setPreparationError(msg);
      setPrepStage("error");
      setPreparingTaskId(null);
    },
    onCancelled: () => {
      setPreparingTaskId(null);
      setPreparingModelId(null);
      setPrepStage("form");
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
    // Real cancel over the WebSocket; the worker kills its subprocess and
    // replies "cancelled" (handled by onCancelled). Falls back to detaching
    // if no task is live so the modal can't get stuck.
    if (preparingTaskId) {
      prepProgress.cancel();
    } else {
      setPreparingModelId(null);
      setPrepStage("form");
    }
  };
  const retryPrepare = () => {
    if (preparingModelId) {
      setPreparationError(null);
      startPrepare(preparingModelId);
    }
  };

  /** Shared "push the form fields onto the project row" helper. */
  const persistSettings = (projectId: string, data: SettingsFormData) =>
    projectsApi.update(projectId, {
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
      // ALL is a form-only sentinel ("user explicitly chose all
      // labels"); the API and DB know only real ISO codes or null.
      country_code: toApiCountryCode(data.country_code),
      state_code: data.state_code,
      detection_batch_size: data.detection_batch_size,
      classification_batch_size: data.classification_batch_size,
      embedding_batch_size: data.embedding_batch_size,
      video_fps: data.video_fps,
      event_smoothing: data.event_smoothing,
      smoothing_strength: data.smoothing_strength,
      taxonomic_rollup: data.taxonomic_rollup,
      independence_interval: data.independence_interval,
    });

  /** Start analysis in an existing run. PATCH settings, kick the
   * deployment queue, and either pop the RunQueueModal or skip
   * straight to verification if the queue had nothing pending. */
  const startAnalysis = useMutation({
    mutationFn: async (data: SettingsFormData) => {
      if (!runId) throw new Error("missing run id");
      await persistSettings(runId, data);
      const resp = await deploymentQueueApi.process({ project_id: runId });
      return resp;
    },
    onSuccess: (resp) => {
      queryClient.invalidateQueries({ queryKey: ["projects", runId] });
      if (resp.jobs_started === 0 || resp.job_ids.length === 0) {
        // Nothing pending to process (e.g. the queue entry was already
        // consumed or is in a non-pending state). Surface it rather
        // than silently jumping to Edit with stale results.
        setNothingToRun(true);
        return;
      }
      setRunState({
        jobIds: resp.job_ids,
        queueEntryIds: resp.queue_entry_ids,
      });
    },
  });

  /** Create a brand-new folder run. Used when there's no `runId` in
   * the URL (the `/folder-runs/new` path) and when the user changed
   * the folder of an existing run to a different one (handled via
   * `force_new` to discard any matching previous run). After create
   * the page reroutes to the new run's `/model` URL and starts
   * analysis there. */
  const createRun = useMutation({
    mutationFn: async ({
      data,
      payload,
    }: {
      data: SettingsFormData;
      payload: FolderRunCreate;
    }) => {
      const created = await folderRunsApi.create(payload);
      // create() returns the project at its default settings, before
      // persistSettings saves the chosen models. Merge the updated
      // project back in so the run we cache in onSuccess hydrates the
      // form with the real selections, not empty defaults. Without this,
      // closing the run modal reveals an empty model step until reload.
      const project = await persistSettings(created.project.id, data);
      const run = { ...created, project };
      const resp = await deploymentQueueApi.process({
        project_id: created.project.id,
      });
      return { run, resp };
    },
    onSuccess: ({ run, resp }) => {
      queryClient.setQueryData(["folder-run", run.project.id], run);
      queryClient.invalidateQueries({
        queryKey: ["folder-run-lookup", run.queue_entry?.folder_path],
      });
      if (resp.jobs_started === 0 || resp.job_ids.length === 0) {
        navigate(`/folder-runs/${run.project.id}/labels`);
        return;
      }
      navigate(`/folder-runs/${run.project.id}/setup`);
      setRunState({
        jobIds: resp.job_ids,
        queueEntryIds: resp.queue_entry_ids,
      });
    },
  });

  /** Wipe the existing analysis output and re-process under the
   * current settings. Destructive: detections + verifications are
   * deleted by `POST /api/folder-runs/{id}/rerun`. The confirm
   * dialog gates this mutation. */
  const rerunAnalysis = useMutation({
    mutationFn: async (data: SettingsFormData) => {
      if (!runId) throw new Error("missing run id");
      await persistSettings(runId, data);
      const reset = await folderRunsApi.rerun(runId);
      queryClient.setQueryData(["folder-run", runId], reset);
      const resp = await deploymentQueueApi.process({ project_id: runId });
      return resp;
    },
    onSuccess: (resp) => {
      queryClient.invalidateQueries({ queryKey: ["projects", runId] });
      if (resp.jobs_started === 0 || resp.job_ids.length === 0) {
        setNothingToRun(true);
        return;
      }
      setRunState({
        jobIds: resp.job_ids,
        queueEntryIds: resp.queue_entry_ids,
      });
    },
  });

  const skipAnalysis = () => {
    if (!lookupRun) return;
    navigate(`/folder-runs/${lookupRun.id}/labels`);
  };

  /** Re-run handler. The destructive path depends on whether the
   * matched lookup is the user's current run or a different one:
   * inside the same run we use the dedicated reset endpoint, for
   * a different run we discard via force_new create. The user
   * sees the same dialog and the same end state either way. */
  /** Remember the committed settings (everything but the folder path)
   * so the next brand-new run seeds from them. Called on every commit
   * path: start, create-new, re-run. */
  const persistLastUsed = (data: SettingsFormData) => {
    saveLastUsedSettings({
      detection_model_id: data.detection_model_id,
      classification_model_id: data.classification_model_id,
      embedding_model_id: data.embedding_model_id,
      excluded_classes: data.excluded_classes,
      country_code: data.country_code,
      state_code: data.state_code,
      detection_batch_size: data.detection_batch_size,
      classification_batch_size: data.classification_batch_size,
      embedding_batch_size: data.embedding_batch_size,
      video_fps: data.video_fps,
      event_smoothing: data.event_smoothing,
      smoothing_strength: data.smoothing_strength,
      taxonomic_rollup: data.taxonomic_rollup,
      independence_interval: data.independence_interval,
    });
  };

  const confirmRerun = () => {
    setRerunOpen(false);
    setNothingToRun(false);
    if (!lookupRun) return;
    const data = form.getValues();
    persistLastUsed(data);
    if (lookupRun.id === runId) {
      rerunAnalysis.mutate(data);
      return;
    }
    if (!folderPath || !scanResult) return;
    createRun.mutate({
      data,
      payload: {
        source_folder: folderPath,
        image_count: scanResult.image_count,
        video_count: scanResult.video_count,
        force_new: true,
      },
    });
  };

  /** Submit dispatcher. The button label / action depend on the run
   * state — see `actionMode` below for the full matrix. */
  const onSubmit = (data: SettingsFormData) => {
    if (!folderPath || !scanResult) return;
    // Geofenced classifiers require an explicit location choice: a
    // country, or knowingly "All labels". Enforced here rather than in
    // the zod schema because the requirement depends on the selected
    // model's geofence, which lives in a query, not in the form data.
    if (requiresCountryChoice && data.country_code == null) {
      form.setError("country_code", {
        type: "required",
        message: "Choose a country, or pick All labels to run unfiltered.",
      });
      return;
    }
    const currentFolder = run?.queue_entry?.folder_path;
    const folderChanged = !!runId && currentFolder !== folderPath;

    setNothingToRun(false);
    persistLastUsed(data);

    if (!runId || folderChanged) {
      createRun.mutate({
        data,
        payload: {
          source_folder: folderPath,
          image_count: scanResult.image_count,
          video_count: scanResult.video_count,
          force_new: folderChanged,
        },
      });
      return;
    }
    if (isTerminal) {
      setRerunOpen(true);
      return;
    }
    startAnalysis.mutate(data);
  };

  // /folder-runs/new mounts this page with no runId. The Setup page
  // handles that case end-to-end (no project yet, no queue entry, no
  // run cache). Only show the loading card when we have a runId but
  // the run hasn't loaded yet.
  // Hold the form until the model lists are in. The pickers are
  // controlled Selects seeded from the run's project row (existing run)
  // or last-used settings (brand-new run). If a model id is assigned
  // before its <SelectItem> is mounted, Radix can't match the value and
  // falls back to the placeholder, leaving the classifier (and the
  // detection / embedding pickers) blank on a cold load / hard refresh.
  const modelListsLoading =
    detectionModelsLoading ||
    classificationModelsLoading ||
    embeddingModelsLoading;
  if ((runId && (isLoading || !run)) || modelListsLoading) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-sm text-muted-foreground">
          Loading...
        </CardContent>
      </Card>
    );
  }

  const detReady = detectionStatus?.status === "ready";
  const clsReady = !hasClassifier || classificationStatus?.status === "ready";
  const embReady = !hasEmbedding || embeddingStatus?.status === "ready";
  // Every selected model installed + ready. Gates both Start and
  // Re-run, and drives the "needs setup" warning. Important for the
  // restored-settings case: a remembered model may no longer be
  // installed, and we never silently swap it.
  const modelsReady = detReady && clsReady && embReady;
  // Missing capture dates no longer block a run: the backend ingests
  // date-less files (they just drop out of time-based stats), and the
  // scan surfaces a non-blocking note. So `folderReady` only needs
  // files present.
  const folderReady =
    !!folderPath &&
    !isScanning &&
    !!scanResult &&
    scanResult.total_count > 0;
  const queueStatus = run?.queue_entry?.status;
  const isTerminal =
    queueStatus === "completed" || queueStatus === "failed";
  const folderChanged =
    !!runId && run?.queue_entry?.folder_path !== folderPath;
  const isMutating =
    startAnalysis.isPending ||
    createRun.isPending ||
    rerunAnalysis.isPending;
  // The notice row collapses both legacy flows into one: same vocab
  // (Skip / Re-run) for an in-progress terminal run AND for a brand
  // new path that landed on an already-analysed folder. The Re-run
  // handler routes to the right destructive mutation based on which
  // case applies, but the user sees one consistent action area.
  // Inside-the-current-run case additionally requires terminal state,
  // otherwise mid-analysis chip back-nav would hide the Start button
  // users need to kick the analysis off.
  const lookupIsCurrent = !!lookupRun && lookupRun.id === runId;
  const showCompletedNotice =
    !!lookupRun &&
    !folderChanged &&
    (lookupIsCurrent ? isTerminal : true);

  const canStart =
    modelsReady &&
    folderReady &&
    !isLookingUp &&
    !showCompletedNotice &&
    !isMutating;

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
        caption="Pick the folder and AI models. Adjust settings if needed."
      />
      <Card>
        <CardContent className="space-y-6 p-6">
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)}>
              <TooltipProvider>
                {/* Folder row + model rows share one form-slot so the
                    form's space-y-6 doesn't insert an extra gap on top
                    of the divider between them. */}
                <div>
                <div className="space-y-0">
                  <FormField
                    control={form.control}
                    name="folder_path"
                    render={({ field }) => (
                      <div className="grid grid-cols-2 items-center gap-8 pb-6 border-b">
                        <div className="space-y-1">
                          <FormLabel>Folder</FormLabel>
                          <FormDescription className="text-sm">
                            The folder with the images or videos you
                            want to analyse. Subfolders are included.
                          </FormDescription>
                        </div>
                        <div className="space-y-2">
                          <FormControl>
                            <FolderSelector
                              value={field.value || null}
                              onChange={(v) => field.onChange(v ?? "")}
                              hideLabel
                              compactScanResult
                            />
                          </FormControl>
                          <FormMessage />
                        </div>
                      </div>
                    )}
                  />
                </div>

                {/* Hide the model + advanced config until a folder is
                    picked, so the only task on first view is "select a
                    folder". It reappears once a folder is chosen. */}
                <div className={folderPath ? undefined : "hidden"}>
                <div className="space-y-0 divide-y border-b">
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
                          <ModelSelect
                            value={field.value ?? NO_CLASSIFIER}
                            onValueChange={(val) =>
                              field.onChange(
                                val === NO_CLASSIFIER ? NO_CLASSIFIER : val,
                              )
                            }
                            models={classificationModels}
                            placeholder="Select classification model"
                            noneValue={NO_CLASSIFIER}
                            noneLabel="No classification model"
                            onShowInfo={() => setShowClsInfo(true)}
                          >
                            <SelectItem value={NO_CLASSIFIER}>
                              ∅ No classification model
                              <br />
                              <span className="text-xs text-muted-foreground">
                                Run animal detector only, identify species
                                manually
                              </span>
                            </SelectItem>
                            <ClassificationModelGroupedItems
                              models={classificationModels.filter(
                                (m) => m.model_id !== "none",
                              )}
                            />
                          </ModelSelect>
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
                          {!hasClassifier && <NoClassifierNotice />}
                          <FormMessage />
                        </div>
                      </div>
                    )}
                  />

                  {hasClassifier && taxonomy && (
                    <div className="grid grid-cols-2 items-center gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Species selection</FormLabel>
                        <FormDescription className="text-sm">
                          {labelCaption}
                        </FormDescription>
                      </div>
                      <div>
                        <LabelSelectionField
                          modelId={classificationModelId!}
                          excludedClasses={excludedClasses}
                          allClasses={taxonomy.all_classes ?? []}
                          countryCode={form.watch("country_code")}
                          stateCode={form.watch("state_code")}
                          onExclusionChange={(classes) =>
                            form.setValue("excluded_classes", classes, {
                              shouldDirty: true,
                            })
                          }
                          onLocationChange={(country, state) => {
                            form.clearErrors("country_code");
                            form.setValue("country_code", country, {
                              shouldDirty: true,
                            });
                            form.setValue("state_code", state, { shouldDirty: true });
                          }}
                          error={
                            form.formState.errors.country_code?.message
                          }
                        />
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
                      className="flex items-center gap-2 py-4 text-left text-sm font-semibold leading-none transition-colors hover:text-primary"
                    >
                      <span>Advanced settings</span>
                      <ChevronDown
                        className={`h-4 w-4 transition-transform ${
                          advancedOpen ? "rotate-180" : ""
                        }`}
                      />
                    </button>
                  </CollapsibleTrigger>
                  {/* Bracket the advanced fields with border-y to match
                      the basic models section above. The toggle sits
                      centered between that section's bottom rule and this
                      top rule (its py-4 gives equal space to each). */}
                  <CollapsibleContent className="space-y-0 divide-y border-y mb-4">
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
                            <ModelSelect
                              value={field.value}
                              onValueChange={field.onChange}
                              models={detectionModels}
                              placeholder="Select detection model"
                              onShowInfo={() => setShowDetInfo(true)}
                            >
                              {detectionModels.map((m) => (
                                <SelectItem key={m.model_id} value={m.model_id}>
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
                            </ModelSelect>
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
                              Lets you sort and group visually similar
                              detections in the Edit step. Optional,
                              skip if you don't need it.
                            </FormDescription>
                          </div>
                          <div className="space-y-2">
                            <ModelSelect
                              value={field.value ?? NO_EMBEDDING}
                              onValueChange={(val) =>
                                field.onChange(
                                  val === NO_EMBEDDING ? NO_EMBEDDING : val,
                                )
                              }
                              models={embeddingModels}
                              placeholder="Select embedding model"
                              noneValue={NO_EMBEDDING}
                              noneLabel="No embedding model"
                              onShowInfo={() => setShowEmbInfo(true)}
                            >
                              <SelectItem value={NO_EMBEDDING}>
                                ∅ No embedding model
                                <br />
                                <span className="text-xs text-muted-foreground">
                                  Skip if you don't need similarity sort or
                                  clustering
                                </span>
                              </SelectItem>
                              {embeddingModels
                                .filter((m) => m.model_id !== "none")
                                .map((m) => (
                                  <SelectItem key={m.model_id} value={m.model_id}>
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
                            </ModelSelect>
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
                      name="video_fps"
                      render={({ field }) => (
                        <SettingRow
                          label="Video frame rate"
                          description={SETTING_CAPTIONS.videoFrameRate}
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

                    <FormField
                      control={form.control}
                      name="independence_interval"
                      render={({ field }) => (
                        <SettingRow
                          label="Independence interval"
                          description={SETTING_CAPTIONS.independenceInterval}
                        >
                          <IntervalControl
                            value={field.value}
                            onChange={field.onChange}
                          />
                          <FormMessage />
                        </SettingRow>
                      )}
                    />

                    {hasClassifier && (
                      <SettingRow
                        label="Smoothing"
                        description={SETTING_CAPTIONS.smoothing}
                      >
                        <CaptionedSelect
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
                          options={SMOOTHING_LEVELS}
                        />
                      </SettingRow>
                    )}

                    {hasClassifier && (
                      <FormField
                        control={form.control}
                        name="taxonomic_rollup"
                        render={({ field }) => (
                          <SettingRow
                            label="Taxonomic rollup"
                            description={SETTING_CAPTIONS.taxonomicRollup}
                          >
                            <Switch
                              checked={field.value}
                              onCheckedChange={field.onChange}
                            />
                          </SettingRow>
                        )}
                      />
                    )}
                    {advancedNonDefault && (
                      <div className="flex justify-end py-3">
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => restoreAdvancedDefaults(form.setValue)}
                        >
                          <RotateCcw className="mr-2 h-4 w-4" />
                          Restore defaults
                        </Button>
                      </div>
                    )}
                  </CollapsibleContent>
                </Collapsible>
                </div>
                </div>

                {/* Action area floats below the form rows (no divider —
                    the buttons are self-evidently the action). Light top
                    padding because the content above already carries its
                    own bottom padding (row py-6, or the collapsed
                    trigger py-3). */}
                <div className="space-y-3 pt-2">
                  {folderReady && !modelsReady && !statusLoading && (
                    <Callout variant="warning" size="compact">
                      One or more selected models need to be set up before
                      you can run. Check the model rows above.
                    </Callout>
                  )}

                  {nothingToRun && (
                    <Callout
                      variant="warning"
                      size="compact"
                      action={
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() =>
                            navigate(`/folder-runs/${runId}/labels`)
                          }
                        >
                          View results
                        </Button>
                      }
                    >
                      Nothing to analyse — this folder looks already
                      processed.
                    </Callout>
                  )}

                  {(startAnalysis.isError ||
                    createRun.isError ||
                    rerunAnalysis.isError) && (
                    <p className="text-sm text-destructive">
                      Could not start analysis:{" "}
                      {(() => {
                        const err =
                          startAnalysis.error ??
                          createRun.error ??
                          rerunAnalysis.error;
                        return err instanceof Error
                          ? err.message
                          : "unknown error";
                      })()}
                    </p>
                  )}

                  {showCompletedNotice ? (
                    <CompletedRunNotice
                      failed={lookupIsCurrent && queueStatus === "failed"}
                      isBusy={isMutating}
                      canRerun={modelsReady}
                      onSkipAnalysis={skipAnalysis}
                      onRerun={() => setRerunOpen(true)}
                    />
                  ) : (
                    <div className="flex items-center justify-end">
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span>
                            <Button
                              type="submit"
                              disabled={!canStart}
                              className="gap-2"
                              size="lg"
                            >
                              {isMutating ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <Sparkles className="h-4 w-4" />
                              )}
                              {isMutating
                                ? "Starting..."
                                : "Start analysis"}
                            </Button>
                          </span>
                        </TooltipTrigger>
                        {!canStart && !isMutating && (
                          <TooltipContent>
                            <p>{actionTooltip(
                              folderReady,
                              !!folderPath,
                            )}</p>
                          </TooltipContent>
                        )}
                      </Tooltip>
                    </div>
                  )}
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

      <RerunConfirmDialog
        open={rerunOpen}
        run={lookupRun ?? null}
        isBusy={rerunAnalysis.isPending || createRun.isPending}
        onCancel={() => setRerunOpen(false)}
        onConfirm={confirmRerun}
      />

      {runState && runId && (
        <RunQueueModal
          open={runState !== null}
          onOpenChange={(open) => {
            if (!open) {
              setRunState(null);
              // Refresh the folder run so `active_job_id` flips to null
              // server-side; otherwise the re-attach effect would
              // immediately reopen the modal we just dismissed.
              queryClient.invalidateQueries({
                queryKey: ["folder-run", runId],
              });
            }
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
              // The run regenerated detections, events, observations,
              // files, and every derived stat. Drop the whole cache so
              // the Edit step and dashboard load fresh data instead of
              // stale pre-run ids and counts. (Re-set the folder-run
              // entry below so the layout doesn't flash a refetch.)
              queryClient.invalidateQueries();
              const next = await folderRunsApi.updateStep(
                runId,
                "labels",
              );
              queryClient.setQueryData(["folder-run", runId], next);
              navigate(`/folder-runs/${runId}/labels`);
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

function actionTooltip(
  folderReady: boolean,
  hasFolderPath: boolean,
): string {
  if (!hasFolderPath) return "Pick a folder first";
  if (!folderReady) return "Waiting for folder scan";
  return "Models need preparing first";
}
