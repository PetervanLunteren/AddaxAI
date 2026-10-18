/**
 * Form state + submit logic for the Save outputs step.
 *
 * Owns the export options, the output dir and the run's threshold,
 * spawns the save-outputs background job, and holds the final result
 * payload the JobProgressModal hands back on completion. The media
 * options, their preview and their persistence come from
 * `useMediaOptions`, the hook the project image copies dialog shares;
 * this one spreads its handles so the Save step reads one form. The
 * page component is responsible for the WebSocket subscription
 * (``useTaskProgress``) and the modal rendering.
 *
 * The output dir defaults to the source folder itself. That is safe
 * because the backend writes media copies into an ``addaxai-media``
 * subfolder (with a scan-skip marker) and the loose data files carry
 * the ``addaxai-`` prefix, so originals are never overwritten and the
 * recognition JSON lands where its source-relative paths resolve
 * (what Timelapse needs).
 */

import { useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  folderRunsApi,
  type SaveOutputsRequest,
  type SaveOutputsResult,
} from "../../../api/folder-runs";
import { projectsApi } from "../../../api/projects";
import { DEFAULT_COUNTING_THRESHOLD } from "../../../lib/confidence";
import { isElectron } from "../../../lib/platform";
import {
  loadLastUsedSaveOutputs,
  saveLastUsedSaveOutputs,
} from "../../../lib/folderRunSettings";
import { useMediaOptions, type MediaFormHandles } from "./useMediaOptions";

export type { MediaFormHandles } from "./useMediaOptions";

export type SpreadsheetFormat = "csv" | "xlsx";

export interface ExportState {
  enabled: boolean;
  /** The files + detections tables. One output with a format, not two
   * outputs: CSV and XLSX hold the same tables, so two checkboxes read
   * as two datasets. The format and the threshold hang off this row as
   * its children, which is what makes clear they govern the tables and
   * not the recognition JSON beside them. */
  spreadsheet: boolean;
  format: SpreadsheetFormat;
  recognitionJson: boolean;
  /** The addaxai-run-info.txt run manifest (models, settings, results).
   * Default on: it's the provenance record, but now opt-out like the rest. */
  summary: boolean;
}

export interface UseSaveOutputsFormParams {
  runId: string;
  sourceFolder: string | undefined;
  /** The run's current counting threshold, straight off the project row.
   * Undefined until the run query resolves. */
  projectThreshold: number | undefined;
}

export interface UseSaveOutputsFormResult extends MediaFormHandles {
  outputDir: string;
  setOutputDir: (v: string) => void;
  /** The dir the form will submit. Defaults to the source folder
   * itself; equal to ``outputDir``. */
  effectiveOutputDir: string;

  exportOpts: ExportState;
  setExportOpts: (s: ExportState) => void;

  /** The run's counting threshold, shown on the Export results card so
   * the number governing the spreadsheet is visible where it lands. This
   * is the project setting itself, not an export-only override, so the
   * grid, the counts and the tables can never disagree. */
  countingThreshold: number;
  /** Drag feedback: moves the handle and the label, saves nothing. */
  setCountingThreshold: (v: number) => void;
  /** Release: persist it and refresh the preview. */
  commitCountingThreshold: (v: number) => void;
  thresholdSaving: boolean;

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
  projectThreshold,
}: UseSaveOutputsFormParams): UseSaveOutputsFormResult {
  const queryClient = useQueryClient();
  const [outputDir, setOutputDir] = useState("");
  const effectiveOutputDir = outputDir;

  // The counting threshold is the project's own, mirrored here only so
  // the handle can move smoothly while dragging. `draggedThreshold` is
  // that local echo; null means "no drag in flight, trust the server".
  // Seeding from the prop each render would fight the drag, and holding
  // it in state permanently would go stale after a reprocess elsewhere.
  const [draggedThreshold, setDraggedThreshold] = useState<number | null>(
    null,
  );
  const countingThreshold =
    draggedThreshold ?? projectThreshold ?? DEFAULT_COUNTING_THRESHOLD;

  const thresholdMutation = useMutation({
    mutationFn: (value: number) =>
      projectsApi.update(runId, { counting_threshold: value }),
    // Settled, not success: a failed save must also drop the local echo,
    // so the handle springs back to what the server actually holds
    // instead of showing a value nothing was written for.
    onSettled: () => {
      setDraggedThreshold(null);
      queryClient.invalidateQueries({ queryKey: ["folder-run", runId] });
      queryClient.invalidateQueries({
        queryKey: ["media-preview"],
      });
    },
  });

  // Debounced, because a commit is not always one gesture. A mouse drag
  // commits once on release, but Radix has no release for the keyboard,
  // so it commits on every arrow press: five taps meant five PATCHes and
  // five MaxN recomputes, which is seconds of backend work per keystroke
  // on a large run. Stepping with the arrows now settles into one save.
  //
  // A debounced write can be outrun, so the pending value lives in a ref
  // beside the timer and there are two ways to force it out: `flush`
  // before the save job is spawned, and the unmount cleanup below.
  // Without those, the two natural things to do straight after moving the
  // slider both lost the change: hitting Save spawned the job before the
  // PATCH landed, so the worker read the old threshold, and stepping Back
  // dropped the timer on the floor.
  const thresholdTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingThreshold = useRef<number | null>(null);

  /** Take the pending value off the queue, cancelling its timer. */
  const takePending = (): number | null => {
    if (thresholdTimer.current) {
      clearTimeout(thresholdTimer.current);
      thresholdTimer.current = null;
    }
    const value = pendingThreshold.current;
    pendingThreshold.current = null;
    return value;
  };

  /** Write a debounced value now and wait for it. */
  const flushCountingThreshold = async (): Promise<void> => {
    const value = takePending();
    if (value === null) return;
    try {
      await thresholdMutation.mutateAsync(value);
    } catch {
      // Already surfaced by the handle springing back to the server's
      // value; the save carries on with whatever is actually stored.
    }
  };

  useEffect(() => {
    return () => {
      const value = takePending();
      if (value === null) return;
      // The component is going away, so this cannot go through the
      // mutation: its onSettled sets state. Write it directly and refresh
      // the caches the rest of the app reads.
      projectsApi
        .update(runId, { counting_threshold: value })
        .then(() => {
          queryClient.invalidateQueries({ queryKey: ["folder-run", runId] });
          queryClient.invalidateQueries({
            queryKey: ["media-preview"],
          });
        })
        .catch(() => {});
    };
  }, [runId, queryClient]);

  const commitCountingThreshold = (value: number) => {
    takePending();
    // Stepped back to where it started: nothing to save, and the local
    // echo has to go or the handle would sit on a value we never wrote.
    if (Math.abs(value - (projectThreshold ?? -1)) < 1e-9) {
      setDraggedThreshold(null);
      return;
    }
    pendingThreshold.current = value;
    thresholdTimer.current = setTimeout(() => {
      const pending = takePending();
      if (pending !== null) thresholdMutation.mutate(pending);
    }, 400);
  };

  // Seed a sensible default once the source folder is known: the
  // source folder itself. The recognition JSON only resolves its
  // source-relative paths there (Timelapse requirement), the data
  // files carry the addaxai- prefix, and media copies go into an
  // addaxai-media subfolder with a scan-skip marker — so nothing
  // collides with the originals. Fires once, and only while the field
  // is still empty, so it never clobbers a user-picked path.
  const hasSeededOutputRef = useRef(false);
  useEffect(() => {
    if (hasSeededOutputRef.current || !sourceFolder) return;
    hasSeededOutputRef.current = true;
    // One-time seed of a default from an async-loaded prop, guarded by
    // the ref so it can't loop.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setOutputDir((cur) => (cur ? cur : sourceFolder));
  }, [sourceFolder]);

  // The media card, shared with the project image copies dialog. Its
  // state is seeded from the same last-used record the export options
  // below read.
  const media = useMediaOptions({ projectId: runId });
  const [persisted] = useState(loadLastUsedSaveOutputs);

  const [exportOpts, setExportOpts] = useState<ExportState>(() => ({
    enabled: persisted?.exportEnabled ?? true,
    // `spreadsheet` / `format` replaced an older `csv` / `xlsx` pair, so
    // fall back to those when a stored setting predates the change. A
    // user who had only XLSX ticked keeps XLSX; anyone else lands on CSV.
    spreadsheet:
      // `||`, not `??`: a stored `csv: false` is a real value, so `??`
      // returned it and never looked at `xlsx`. Anyone upgrading with
      // XLSX ticked and CSV unticked lost the spreadsheet entirely.
      persisted?.spreadsheet ?? (persisted?.csv || persisted?.xlsx) ?? true,
    format:
      persisted?.format ??
      (persisted?.xlsx && !persisted?.csv ? "xlsx" : "csv"),
    recognitionJson: persisted?.recognitionJson ?? true,
    summary: persisted?.summary ?? true,
  }));

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

  const runSpawn = async () => {
    // Write any debounced threshold first and wait for it. The worker
    // reads the project row when it runs, so spawning the job before the
    // PATCH lands is a race the user always loses silently.
    await flushCountingThreshold();
    // Remember these choices for the next save (not the output folder,
    // which is derived per run from the source). Each half from the
    // hook that owns it; the store merges.
    saveLastUsedSaveOutputs({
      exportEnabled: exportOpts.enabled,
      spreadsheet: exportOpts.spreadsheet,
      format: exportOpts.format,
      recognitionJson: exportOpts.recognitionJson,
      summary: exportOpts.summary,
    });
    media.remember();
    const request: SaveOutputsRequest = {
      output_dir: effectiveOutputDir,
      // The same fields the preview was counted with.
      ...media.requestFields,
      separate_folders: media.separate.enabled,
      // The backend still takes one flag per format, so the picker maps
      // onto the two booleans here and no worker code has to change.
      csv:
        exportOpts.enabled &&
        exportOpts.spreadsheet &&
        exportOpts.format === "csv",
      xlsx:
        exportOpts.enabled &&
        exportOpts.spreadsheet &&
        exportOpts.format === "xlsx",
      recognition_json: exportOpts.enabled && exportOpts.recognitionJson,
      run_readme: exportOpts.enabled && exportOpts.summary,
    };
    spawn.mutate(request);
  };

  const saveAll = () => {
    setSaveError(null);
    setResult(null);
    void runSpawn();
  };

  const onJobComplete = (data: SaveOutputsResult) => {
    setResult(data);
    setJobId(null);
  };

  const onJobError = (message: string) => {
    setSaveError(new Error(message));
    setJobId(null);
  };

  const onJobCancelled = () => {
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
    (exportOpts.spreadsheet ||
      exportOpts.recognitionJson ||
      exportOpts.summary);
  // The media card, when on, must have something to copy: the job wipes
  // the previous addaxai-media before it writes, so a save with nothing
  // in scope would only delete. The rule is the shared hook's; the
  // image copies dialog applies the same one.
  const mediaOk = !media.separate.enabled || media.hasFilesToCopy;
  const canSave =
    !!effectiveOutputDir &&
    !spawn.isPending &&
    jobId === null &&
    mediaOk &&
    (media.separate.enabled || exportPicked);

  return {
    ...media,
    outputDir,
    setOutputDir,
    effectiveOutputDir,
    exportOpts,
    setExportOpts,
    countingThreshold,
    setCountingThreshold: setDraggedThreshold,
    commitCountingThreshold,
    thresholdSaving: thresholdMutation.isPending,
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
