/**
 * Timelapse Analyser integration window.
 *
 * One-page form, no sidebar. Reuses the main app's folder picker, label
 * tree selector, model dropdowns, and websocket progress hook so the
 * UI stays in lockstep with whatever the regular Analyses page does.
 *
 * The output is a single results.json next to the user's image folder
 * that Timelapse imports through "Recognition > Import recognition data
 * for this image set".
 *
 * If AddaxAI has not been set up yet, the SetupPage component is
 * rendered inline (per user choice) so users invoking from Timelapse
 * never have to leave the window to install models.
 */

import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, FolderOpen, Loader2, Settings2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Alert, AlertDescription } from "@/components/ui/alert";

import { FolderSelector } from "@/components/analyses/FolderSelector";
import { TreeSelector } from "@/components/taxonomy/TreeSelector";
import { useTaskProgress } from "@/hooks/useTaskProgress";
import { collectLeafIds } from "@/lib/taxonomy-utils";
import { isElectron } from "@/lib/platform";

import { modelsApi } from "@/api/models";
import { setupApi } from "@/api/setup";
import { timelapseApi, type SmoothingStrength } from "@/api/timelapse";
import type { ModelInfo, TaxonomyResponse } from "@/api/types";

import SetupPage from "./SetupPage";

const NO_CLASSIFIER_VALUE = "__none__";

interface AdvancedSettings {
  detectionModelId: string;
  detectionConfidence: number;
  detectionBatchSize: number;
  classificationBatchSize: number;
  videoFps: number;
  independenceIntervalMinutes: number;
  smoothing: SmoothingStrength;
  taxonomicRollup: boolean;
}

const DEFAULT_ADVANCED: AdvancedSettings = {
  detectionModelId: "MegaDetector-5a",
  detectionConfidence: 0.2,
  detectionBatchSize: 1,
  classificationBatchSize: 16,
  videoFps: 1.0,
  independenceIntervalMinutes: 120,
  smoothing: "normal",
  taxonomicRollup: true,
};

function readQueryFolder(): string | null {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("path");
  if (fromQuery) return fromQuery;
  // Fall back to hash params for `#/timelapse?path=...` shape used when
  // launched through Electron's loadURL.
  const hash = window.location.hash;
  const qIndex = hash.indexOf("?");
  if (qIndex >= 0) {
    const hashParams = new URLSearchParams(hash.slice(qIndex + 1));
    return hashParams.get("path");
  }
  return null;
}

export default function TimelapseModePage() {
  const { data: setupStatus, isLoading: setupLoading } = useQuery({
    queryKey: ["setup-status"],
    queryFn: setupApi.getStatus,
    refetchInterval: 5000,
  });

  if (setupLoading || !setupStatus) {
    return null;
  }

  if (!setupStatus.ready) {
    // Run setup inline so users launched via Timelapse never have to
    // hop windows to install models.
    return <SetupPage />;
  }

  return <TimelapseForm />;
}

function TimelapseForm() {
  const [folderPath, setFolderPath] = useState<string | null>(readQueryFolder);
  const [classifierId, setClassifierId] = useState<string>(NO_CLASSIFIER_VALUE);
  const [excludedIds, setExcludedIds] = useState<Set<string>>(new Set());
  const [advanced, setAdvanced] = useState<AdvancedSettings>(DEFAULT_ADVANCED);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [outputPath, setOutputPath] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);

  const detectionModels = useQuery<ModelInfo[]>({
    queryKey: ["models", "detection"],
    queryFn: modelsApi.listDetectionModels,
  });

  const classificationModels = useQuery<ModelInfo[]>({
    queryKey: ["models", "classification"],
    queryFn: modelsApi.listClassificationModels,
  });

  const taxonomyQuery = useQuery<TaxonomyResponse>({
    queryKey: ["models", "taxonomy", classifierId],
    queryFn: () => modelsApi.getTaxonomy(classifierId),
    enabled: classifierId !== NO_CLASSIFIER_VALUE,
  });

  // When the classifier or its taxonomy changes, default to "all included"
  // (no exclusions). The user opts species OUT to mark them absent.
  useEffect(() => {
    if (taxonomyQuery.data) {
      setExcludedIds(new Set());
    }
  }, [taxonomyQuery.data]);

  const allTaxonomyLeafIds = useMemo(() => {
    if (!taxonomyQuery.data) return new Set<string>();
    return collectLeafIds(taxonomyQuery.data.tree);
  }, [taxonomyQuery.data]);

  // TreeSelector emits "selected = NOT excluded" in inclusion mode. We
  // store excluded IDs so the request mirrors the project setting shape.
  const includedIds = useMemo(() => {
    const result = new Set(allTaxonomyLeafIds);
    for (const id of excludedIds) result.delete(id);
    return result;
  }, [allTaxonomyLeafIds, excludedIds]);

  const handleSelectionChange = (newSelection: Set<string>) => {
    const newExcluded = new Set<string>();
    for (const id of allTaxonomyLeafIds) {
      if (!newSelection.has(id)) newExcluded.add(id);
    }
    setExcludedIds(newExcluded);
  };

  const progress = useTaskProgress({
    taskId: jobId,
    onComplete: (data) => {
      const path =
        data && typeof data === "object" && "output_path" in data
          ? String((data as { output_path: unknown }).output_path)
          : null;
      setOutputPath(path);
    },
    onError: (msg) => setErrorMessage(msg),
  });

  const reset = () => {
    setJobId(null);
    setOutputPath(null);
    setErrorMessage(null);
  };

  const handleRun = async () => {
    if (!folderPath) return;
    setErrorMessage(null);
    setOutputPath(null);
    setIsStarting(true);
    try {
      const excludedNames = Array.from(excludedIds);
      const response = await timelapseApi.run({
        folder_path: folderPath,
        classification_model_id:
          classifierId === NO_CLASSIFIER_VALUE ? null : classifierId,
        detection_model_id: advanced.detectionModelId,
        excluded_classes: excludedNames,
        detection_confidence_threshold: advanced.detectionConfidence,
        detection_batch_size: advanced.detectionBatchSize,
        classification_batch_size: advanced.classificationBatchSize,
        video_fps: advanced.videoFps,
        independence_interval_minutes: advanced.independenceIntervalMinutes,
        smoothing_strength: advanced.smoothing,
        taxonomic_rollup: advanced.taxonomicRollup,
      });
      setJobId(response.job_id);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setIsStarting(false);
    }
  };

  const reveal = async () => {
    if (!outputPath) return;
    if (window.electronAPI) {
      await window.electronAPI.showItemInFolder(outputPath);
    }
  };

  const isRunning = jobId !== null && outputPath === null && !errorMessage;
  const showSuccess = outputPath !== null;

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-3xl px-4 py-4 sm:px-6 lg:px-8">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Timelapse mode</h1>
            <p className="text-sm text-muted-foreground">
              Run AddaxAI on a folder and write a results.json that Timelapse
              can import.
            </p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        {showSuccess ? (
          <SuccessCard
            outputPath={outputPath!}
            onReveal={reveal}
            onRunAnother={reset}
          />
        ) : isRunning ? (
          <RunningCard
            phase={progress.phase}
            phaseProgress={progress.phaseProgress}
            message={progress.message}
            metricsLine={progress.metrics?.raw_line ?? ""}
          />
        ) : (
          <>
            <section className="space-y-2">
              <Label>Folder</Label>
              <FolderSelector
                value={folderPath}
                onChange={setFolderPath}
                hideLabel
              />
            </section>

            <section className="space-y-2">
              <Label>Classification model</Label>
              <Select value={classifierId} onValueChange={setClassifierId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a classifier" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NO_CLASSIFIER_VALUE}>
                    No classifier (detection only)
                  </SelectItem>
                  {(classificationModels.data ?? []).map((m) => (
                    <SelectItem key={m.model_id} value={m.model_id}>
                      {m.emoji} {m.friendly_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </section>

            {classifierId !== NO_CLASSIFIER_VALUE && (
              <section className="space-y-2">
                <Label>Label selection</Label>
                <p className="text-xs text-muted-foreground">
                  Uncheck species that do not occur in your project area. Those
                  predictions will be redirected up the taxonomy tree to the
                  closest allowed ancestor.
                </p>
                {taxonomyQuery.isLoading ? (
                  <div className="text-sm text-muted-foreground">
                    Loading taxonomy...
                  </div>
                ) : taxonomyQuery.data ? (
                  <TreeSelector
                    tree={taxonomyQuery.data.tree}
                    selectedIds={includedIds}
                    mode="inclusion"
                    onSelectionChange={handleSelectionChange}
                    height="320px"
                    emptyMessage="No taxonomy available for this classifier."
                  />
                ) : (
                  <div className="text-sm text-muted-foreground">
                    No taxonomy available.
                  </div>
                )}
              </section>
            )}

            <section className="rounded-md border bg-card-background">
              <button
                type="button"
                className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium"
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                <span className="flex items-center gap-2">
                  <Settings2 className="h-4 w-4" />
                  Advanced settings
                </span>
                <span className="text-xs text-muted-foreground">
                  {showAdvanced ? "Hide" : "Show"}
                </span>
              </button>
              {showAdvanced && (
                <AdvancedFields
                  value={advanced}
                  onChange={setAdvanced}
                  detectionModels={detectionModels.data ?? []}
                />
              )}
            </section>

            {errorMessage && (
              <Alert className="border-destructive text-destructive">
                <AlertDescription>{errorMessage}</AlertDescription>
              </Alert>
            )}

            <div className="flex justify-end">
              <Button
                onClick={handleRun}
                disabled={!folderPath || isStarting}
              >
                {isStarting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Starting...
                  </>
                ) : (
                  "Run analysis"
                )}
              </Button>
            </div>

            {!isElectron() && (
              <p className="text-xs text-muted-foreground">
                Tip: this window is intended to run inside the AddaxAI desktop
                app. The dev browser variant works for testing the form, but
                folder pickers and reveal-in-explorer require Electron.
              </p>
            )}
          </>
        )}
      </main>
    </div>
  );
}

function AdvancedFields({
  value,
  onChange,
  detectionModels,
}: {
  value: AdvancedSettings;
  onChange: (v: AdvancedSettings) => void;
  detectionModels: ModelInfo[];
}) {
  const set = <K extends keyof AdvancedSettings>(
    key: K,
    val: AdvancedSettings[K],
  ) => onChange({ ...value, [key]: val });

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 px-4 pb-4">
      <div className="space-y-1.5 sm:col-span-2">
        <Label>Detection model</Label>
        <Select
          value={value.detectionModelId}
          onValueChange={(v) => set("detectionModelId", v)}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {detectionModels.map((m) => (
              <SelectItem key={m.model_id} value={m.model_id}>
                {m.emoji} {m.friendly_name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-1.5">
        <Label>Detection confidence threshold</Label>
        <Input
          type="number"
          min={0}
          max={1}
          step={0.05}
          value={value.detectionConfidence}
          onChange={(e) =>
            set("detectionConfidence", parseFloat(e.target.value))
          }
        />
      </div>

      <div className="space-y-1.5">
        <Label>Detection batch size</Label>
        <Input
          type="number"
          min={1}
          step={1}
          value={value.detectionBatchSize}
          onChange={(e) =>
            set("detectionBatchSize", parseInt(e.target.value, 10) || 1)
          }
        />
      </div>

      <div className="space-y-1.5">
        <Label>Classification batch size</Label>
        <Input
          type="number"
          min={1}
          step={1}
          value={value.classificationBatchSize}
          onChange={(e) =>
            set("classificationBatchSize", parseInt(e.target.value, 10) || 1)
          }
        />
      </div>

      <div className="space-y-1.5">
        <Label>Video frame rate (fps)</Label>
        <Input
          type="number"
          min={0.1}
          max={30}
          step={0.1}
          value={value.videoFps}
          onChange={(e) => set("videoFps", parseFloat(e.target.value))}
        />
      </div>

      <div className="space-y-1.5">
        <Label>Independence interval (minutes)</Label>
        <Input
          type="number"
          min={1}
          step={1}
          value={value.independenceIntervalMinutes}
          onChange={(e) =>
            set(
              "independenceIntervalMinutes",
              parseInt(e.target.value, 10) || 1,
            )
          }
        />
      </div>

      <div className="space-y-1.5">
        <Label>Smoothing</Label>
        <Select
          value={value.smoothing}
          onValueChange={(v) => set("smoothing", v as SmoothingStrength)}
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

      <div className="flex items-center justify-between">
        <Label>Taxonomic rollup</Label>
        <Switch
          checked={value.taxonomicRollup}
          onCheckedChange={(v) => set("taxonomicRollup", v)}
        />
      </div>
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
  return (
    <div className="rounded-md border bg-card-background p-6 space-y-3">
      <div className="flex items-center gap-2">
        <Loader2 className="h-5 w-5 animate-spin" />
        <span className="font-medium">Running analysis</span>
      </div>
      <div className="text-sm text-muted-foreground">
        {phase ? phase.replace(/_/g, " ") : "Starting"}
      </div>
      <div className="h-2 w-full rounded-full bg-secondary overflow-hidden">
        <div
          className="h-full bg-primary transition-all"
          style={{ width: `${Math.round((phaseProgress || 0) * 100)}%` }}
        />
      </div>
      {(metricsLine || message) && (
        <pre className="text-xs text-muted-foreground whitespace-pre-wrap font-mono">
          {metricsLine || message}
        </pre>
      )}
    </div>
  );
}

function SuccessCard({
  outputPath,
  onReveal,
  onRunAnother,
}: {
  outputPath: string;
  onReveal: () => void;
  onRunAnother: () => void;
}) {
  return (
    <div className="rounded-md border bg-card-background p-6 space-y-4">
      <div className="flex items-center gap-2 text-primary">
        <CheckCircle2 className="h-5 w-5" />
        <span className="font-medium">Analysis complete</span>
      </div>

      <div className="text-sm">
        Wrote results to:
        <div className="mt-1 font-mono text-xs break-all rounded border bg-white px-2 py-1">
          {outputPath}
        </div>
      </div>

      <div className="text-sm text-muted-foreground">
        In Timelapse, open <strong>Recognition</strong> &gt;{" "}
        <strong>Import recognition data for this image set</strong> and pick
        this file.
      </div>

      <div className="flex gap-2">
        {isElectron() && (
          <Button variant="outline" onClick={onReveal}>
            <FolderOpen className="h-4 w-4 mr-2" />
            Reveal in Explorer
          </Button>
        )}
        <Button onClick={onRunAnother}>Run another folder</Button>
      </div>
    </div>
  );
}
