/**
 * "Analysis settings" button + slideout for the folder-run Labels step.
 *
 * Lives in the grid's toolbar (via LabelsTab's ``toolbarExtra`` slot)
 * and opens a sheet hosting the retroactive knobs (independence
 * interval, smoothing, taxonomic rollup). Changing them does not
 * re-run the models: Apply PATCHes the project and starts a reprocess
 * job (the backend re-reads the raw results.json and re-applies the
 * transforms), with the same blocking progress modal the project
 * Settings page uses.
 *
 * A slideout rather than a standing bar: these settings mutate data,
 * so they sit behind an explicit, labeled opening instead of next to
 * the view filters (which only change what you see).
 *
 * Applied values are also persisted to localStorage so the next
 * folder run seeds its analysis with them (same sticky-settings
 * mechanism as the setup step's model choices).
 */

import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { SlidersHorizontal } from "lucide-react";
import { toast } from "sonner";

import { projectsApi } from "../../api/projects";
import type { ProjectResponse } from "../../api/types";
import { useTaskProgress } from "../../hooks/useTaskProgress";
import { useReprocessSummary } from "../../hooks/useReprocessSummary";
import type { SaveMetric } from "../../lib/saveMetrics";
import { invalidateProjectData } from "../../lib/invalidate-project";
import { saveLastUsedSettings } from "../../lib/folderRunSettings";
import {
  fetchRegroupImpact,
  hasReprocessChanges,
  type RegroupImpact,
  startReprocessIfNeeded,
} from "../../lib/reprocessSettings";
import {
  buildSaveResults,
  fetchStats,
  type ProjectStats,
} from "../../lib/reprocessStats";
import { RegroupConfirmDialog } from "../settings/RegroupConfirmDialog";
import {
  AnalysisSettingsRows,
  type AnalysisSettingsValues,
  type SmoothingLevel,
} from "../settings/AnalysisSettingsRows";
import { ApplySettingsModal } from "../settings/ApplySettingsModal";
import { Button } from "../ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "../ui/sheet";

// Refining a folder run is label triage, so the summary shows only the
// Labels diff. Counts are a Counts-step concern and stay in projects mode.
const FOLDER_RUN_METRICS: SaveMetric[] = ["labels"];

function valuesFromProject(project: ProjectResponse): AnalysisSettingsValues {
  return {
    event_smoothing: project.event_smoothing,
    smoothing_strength: (project.smoothing_strength ?? "normal") as
      | "mild"
      | "normal"
      | "aggressive",
    taxonomic_rollup: project.taxonomic_rollup,
    independence_interval: project.independence_interval,
  };
}

export function AnalysisSettingsButton({
  runId,
  project,
  onApplied,
}: {
  runId: string;
  project: ProjectResponse;
  /** Fired after settings are applied (and any reprocess finished), so
   *  the host can re-run the grid's sort onto the new labels. */
  onApplied?: () => void;
}) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [values, setValues] = useState<AnalysisSettingsValues>(() =>
    valuesFromProject(project),
  );
  // isApplying covers the PATCH + job kick-off roundtrip before a job
  // id exists; jobId drives the progress modal afterwards.
  const [isApplying, setIsApplying] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  // Set when an interval change would regroup confirmed events; drives the
  // type-to-confirm gate before the reprocess actually runs.
  const [regroupImpact, setRegroupImpact] = useState<RegroupImpact | null>(null);
  // Before-stats captured at apply time; diffed against after-stats once
  // the reprocess finishes to show the "how the DB changed" summary.
  const pendingBeforeStats = useRef<ProjectStats | null>(null);
  const { showSummary, summaryUI } = useReprocessSummary(
    runId,
    "Changes applied",
    FOLDER_RUN_METRICS,
  );

  const dirty = hasReprocessChanges(
    valuesFromProject(project) as unknown as Record<string, unknown>,
    values as unknown as Record<string, unknown>,
  );

  // Invalidate queries and re-sort the grid onto the reprocessed labels.
  // The caller shows the toast (summary or plain), so this stays quiet.
  const finish = () => {
    invalidateProjectData(queryClient, runId);
    queryClient.invalidateQueries({ queryKey: ["folder-run", runId] });
    // The grid renders from a streaming sort mutation, not a query, so
    // query invalidation alone won't refresh it. Tell the host to
    // re-sort onto the reprocessed labels.
    onApplied?.();
  };

  const progress = useTaskProgress({
    taskId: jobId,
    onComplete: async () => {
      setJobId(null);
      setIsApplying(false);
      finish();
      const before = pendingBeforeStats.current;
      pendingBeforeStats.current = null;
      if (!before) {
        toast.success("Changes applied");
        return;
      }
      // Threshold is unchanged (this slideout has no threshold control).
      // The reprocess has already rewritten the materialized observations
      // that the after-stats read from.
      try {
        const after = await fetchStats(runId, project.counting_threshold);
        showSummary(buildSaveResults(before, after));
      } catch {
        toast.success("Changes applied");
      }
    },
  });

  // Button handler: warn first if the interval change would regroup
  // confirmed events, otherwise apply straight away.
  const apply = async () => {
    try {
      const impact = await fetchRegroupImpact(
        runId,
        project.independence_interval,
        values.independence_interval,
      );
      if (impact) {
        setRegroupImpact(impact);
        setOpen(false); // hand over to the confirm dialog
        return;
      }
    } catch {
      // Preview failed: don't block the user, fall through and apply.
    }
    await runApply();
  };

  const runApply = async () => {
    setIsApplying(true);
    setOpen(false);
    try {
      // Capture before-stats before the PATCH rewrites project settings.
      pendingBeforeStats.current = await fetchStats(
        runId, project.counting_threshold,
      );
      await projectsApi.update(runId, values);
      // Sticky settings: the next run's analysis seeds from these.
      saveLastUsedSettings(values);
      const newJobId = await startReprocessIfNeeded(runId);
      if (newJobId) {
        setJobId(newJobId);
        return; // modal takes over; summary shown in onComplete
      }
      // No classifications to reprocess: the DB is unchanged, so there is
      // nothing to diff.
      pendingBeforeStats.current = null;
      setIsApplying(false);
      finish();
      toast.success("Changes applied");
    } catch (err) {
      pendingBeforeStats.current = null;
      setIsApplying(false);
      toast.error(
        err instanceof Error ? err.message : "Could not apply settings",
      );
    }
  };

  const hasClassifier = !!project.classification_model_id;

  return (
    <>
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="h-8 gap-2 font-normal"
        onClick={() => setOpen(true)}
      >
        <SlidersHorizontal className="h-3.5 w-3.5 text-muted-foreground" />
        Refine results
      </Button>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent side="right" className="w-full overflow-y-auto sm:max-w-4xl">
          <SheetHeader className="space-y-1">
            <SheetTitle>Refine results</SheetTitle>
            <SheetDescription>
              Applies to the results below without re-running the models.
            </SheetDescription>
          </SheetHeader>
          <div className="mt-2 space-y-0 divide-y">
            <AnalysisSettingsRows
              values={values}
              onIntervalChange={(seconds) =>
                setValues((v) => ({
                  ...v,
                  independence_interval: seconds,
                }))
              }
              onSmoothingChange={(level: SmoothingLevel) =>
                setValues((v) =>
                  level === "off"
                    ? { ...v, event_smoothing: false }
                    : {
                        ...v,
                        event_smoothing: true,
                        smoothing_strength: level,
                      },
                )
              }
              onRollupChange={(enabled) =>
                setValues((v) => ({ ...v, taxonomic_rollup: enabled }))
              }
              showClassifierFields={hasClassifier}
            />
          </div>
          <div className="mt-4 flex justify-end">
            <Button
              onClick={apply}
              disabled={!dirty || isApplying || !!jobId}
            >
              {isApplying || jobId ? "Applying..." : "Apply and reprocess"}
            </Button>
          </div>
        </SheetContent>
      </Sheet>

      <ApplySettingsModal
        open={isApplying || !!jobId}
        message={progress.message}
        progress={progress.progress}
        fallbackMessage="Saving settings..."
      />

      {regroupImpact && (
        <RegroupConfirmDialog
          open
          onOpenChange={(o) => !o && setRegroupImpact(null)}
          impact={regroupImpact}
          fromInterval={project.independence_interval}
          toInterval={values.independence_interval}
          isPending={isApplying || !!jobId}
          onConfirm={() => {
            setRegroupImpact(null);
            runApply();
          }}
        />
      )}

      {summaryUI}
    </>
  );
}
