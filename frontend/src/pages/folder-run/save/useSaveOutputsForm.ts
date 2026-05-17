/**
 * Form state + submit logic for the Save outputs step.
 *
 * Owns the option state (groups + output dir + filter), spawns the
 * save-outputs background job, and holds the final result payload
 * the JobProgressModal hands back on completion. The page component
 * is responsible for the WebSocket subscription (``useTaskProgress``)
 * and the modal rendering — this hook just exposes the handles.
 */

import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  folderRunsApi,
  type ExifMode,
  type SaveOutputsRequest,
  type SaveOutputsResult,
  type SeparateGroupBy,
  type SeparateMode,
} from "../../../api/folder-runs";
import { isElectron } from "../../../lib/platform";

export interface SeparateState {
  enabled: boolean;
  method: SeparateMode;
  groupBy: SeparateGroupBy;
}

export interface VisualiseState {
  enabled: boolean;
  blur: boolean;
}

export interface ExifState {
  enabled: boolean;
  mode: ExifMode;
}

/** Label-exclusion filter — list of LabelTaxonomy ids and / or raw
 * label strings to remove from every output. */
export type ExcludedLabelIds = readonly string[];

export interface ExportState {
  enabled: boolean;
  csv: boolean;
  xlsx: boolean;
  recognitionJson: boolean;
}

function buildRequest(
  outputDir: string,
  separate: SeparateState,
  visualise: VisualiseState,
  exif: ExifState,
  exportOpts: ExportState,
  excludedLabelIds: ExcludedLabelIds,
): SaveOutputsRequest {
  return {
    output_dir: outputDir,
    separate_folders: separate.enabled,
    separate_method: separate.method,
    separate_group_by: separate.groupBy,
    excluded_label_ids: [...excludedLabelIds],
    visualised_images: visualise.enabled,
    blur_people: visualise.enabled && visualise.blur,
    write_exif: exif.enabled,
    exif_mode: exif.mode,
    csv: exportOpts.enabled && exportOpts.csv,
    xlsx: exportOpts.enabled && exportOpts.xlsx,
    recognition_json: exportOpts.enabled && exportOpts.recognitionJson,
  };
}

/** Default output directory: sibling "AddaxAI results" folder next
 * to the source folder, run name as the leaf. Empty string when
 * the source folder isn't known yet. */
function defaultOutputDir(
  sourceFolder: string | undefined,
  runName: string,
): string {
  if (!sourceFolder) return "";
  const sep = sourceFolder.includes("\\") ? "\\" : "/";
  const trimmed = sourceFolder.replace(/[\\/]+$/, "");
  const parts = trimmed.split(sep);
  parts.pop();
  parts.push("AddaxAI results");
  parts.push(runName);
  return parts.join(sep);
}

export interface UseSaveOutputsFormParams {
  runId: string;
  sourceFolder: string | undefined;
  runName: string;
}

export interface UseSaveOutputsFormResult {
  outputDir: string;
  setOutputDir: (v: string) => void;
  /** The dir that will actually be used: typed value or the default. */
  effectiveOutputDir: string;

  separate: SeparateState;
  setSeparate: (s: SeparateState) => void;
  visualise: VisualiseState;
  setVisualise: (s: VisualiseState) => void;
  exif: ExifState;
  setExif: (s: ExifState) => void;
  exportOpts: ExportState;
  setExportOpts: (s: ExportState) => void;
  excludedLabelIds: ExcludedLabelIds;
  setExcludedLabelIds: (v: ExcludedLabelIds) => void;

  promoteOpen: boolean;
  setPromoteOpen: (v: boolean) => void;

  /** Job currently running (null when idle or after completion). */
  jobId: string | null;
  /** Final result payload from the job's complete event. */
  result: SaveOutputsResult | null;
  /** Set during the brief HTTP roundtrip that spawns the job. */
  isSpawning: boolean;
  saveError: Error | null;

  /** Spawn the save-outputs job. */
  saveAll: () => void;
  /** Called by the modal when the job's complete event arrives. */
  onJobComplete: (data: SaveOutputsResult) => void;
  /** Called by the modal on job error. */
  onJobError: (message: string) => void;
  /** Called by the modal on user-cancellation. */
  onJobCancelled: (message: string) => void;
  /** Reset the completion screen back to the form. */
  clearResult: () => void;

  handleBrowse: () => Promise<void>;
  handleOpenResults: () => Promise<void>;

  exportPicked: boolean;
  canSave: boolean;
}

export function useSaveOutputsForm({
  runId,
  sourceFolder,
  runName,
}: UseSaveOutputsFormParams): UseSaveOutputsFormResult {
  const initialOutputDir = useMemo(
    () => defaultOutputDir(sourceFolder, runName),
    [sourceFolder, runName],
  );
  const [outputDir, setOutputDir] = useState("");
  const effectiveOutputDir = outputDir || initialOutputDir;

  const [separate, setSeparate] = useState<SeparateState>({
    enabled: true,
    method: "copy",
    groupBy: "taxonomic",
  });
  const [visualise, setVisualise] = useState<VisualiseState>({
    enabled: false,
    blur: false,
  });
  const [exif, setExif] = useState<ExifState>({
    enabled: false,
    mode: "copy",
  });
  const [exportOpts, setExportOpts] = useState<ExportState>({
    enabled: false,
    csv: true,
    xlsx: false,
    recognitionJson: false,
  });

  const [excludedLabelIds, setExcludedLabelIdsState] = useState<
    readonly string[]
  >([]);
  const setExcludedLabelIds = (v: ExcludedLabelIds) =>
    setExcludedLabelIdsState([...v]);

  const [promoteOpen, setPromoteOpen] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<SaveOutputsResult | null>(null);
  const [saveError, setSaveError] = useState<Error | null>(null);

  const spawn = useMutation({
    mutationFn: (payload: SaveOutputsRequest) =>
      folderRunsApi.saveOutputs(runId, payload),
    onSuccess: (resp) => setJobId(resp.job_id),
    onError: (e) =>
      setSaveError(e instanceof Error ? e : new Error("unknown")),
  });

  const saveAll = () => {
    setSaveError(null);
    setResult(null);
    spawn.mutate(
      buildRequest(
        effectiveOutputDir,
        separate,
        visualise,
        exif,
        exportOpts,
        excludedLabelIds,
      ),
    );
  };

  const onJobComplete = (data: SaveOutputsResult) => {
    setResult(data);
    setJobId(null);
  };

  const onJobError = (message: string) => {
    setSaveError(new Error(message));
    setJobId(null);
  };

  const onJobCancelled = (_message: string) => {
    setJobId(null);
  };

  const clearResult = () => setResult(null);

  const handleBrowse = async () => {
    if (!isElectron() || !window.electronAPI?.selectFolder) return;
    const folder = await window.electronAPI.selectFolder();
    if (folder) setOutputDir(folder);
  };

  const handleOpenResults = async () => {
    if (!result || !window.electronAPI?.openPath) return;
    await window.electronAPI.openPath(result.output_dir);
  };

  const exportPicked =
    exportOpts.enabled &&
    (exportOpts.csv || exportOpts.xlsx || exportOpts.recognitionJson);
  const canSave =
    !!effectiveOutputDir &&
    !spawn.isPending &&
    jobId === null &&
    (separate.enabled ||
      visualise.enabled ||
      exif.enabled ||
      exportPicked);

  return {
    outputDir,
    setOutputDir,
    effectiveOutputDir,
    separate,
    setSeparate,
    visualise,
    setVisualise,
    exif,
    setExif,
    exportOpts,
    setExportOpts,
    excludedLabelIds,
    setExcludedLabelIds,
    promoteOpen,
    setPromoteOpen,
    jobId,
    result,
    isSpawning: spawn.isPending,
    saveError,
    saveAll,
    onJobComplete,
    onJobError,
    onJobCancelled,
    clearResult,
    handleBrowse,
    handleOpenResults,
    exportPicked,
    canSave,
  };
}
