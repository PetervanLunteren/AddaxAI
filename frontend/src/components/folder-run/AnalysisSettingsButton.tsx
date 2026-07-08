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

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { SlidersHorizontal } from "lucide-react";
import { toast } from "sonner";

import { projectsApi } from "../../api/projects";
import type { ProjectResponse } from "../../api/types";
import { useTaskProgress } from "../../hooks/useTaskProgress";
import { invalidateProjectData } from "../../lib/invalidate-project";
import { saveLastUsedSettings } from "../../lib/folderRunSettings";
import {
  hasReprocessChanges,
  startReprocessIfNeeded,
} from "../../lib/reprocessSettings";
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

  const dirty = hasReprocessChanges(
    valuesFromProject(project) as unknown as Record<string, unknown>,
    values as unknown as Record<string, unknown>,
  );

  const finish = () => {
    invalidateProjectData(queryClient, runId);
    queryClient.invalidateQueries({ queryKey: ["folder-run", runId] });
    // The grid renders from a streaming sort mutation, not a query, so
    // query invalidation alone won't refresh it. Tell the host to
    // re-sort onto the reprocessed labels.
    onApplied?.();
    toast.success("Analysis settings applied");
  };

  const progress = useTaskProgress({
    taskId: jobId,
    onComplete: () => {
      setJobId(null);
      setIsApplying(false);
      finish();
    },
  });

  const apply = async () => {
    setIsApplying(true);
    setOpen(false);
    try {
      await projectsApi.update(runId, values);
      // Sticky settings: the next run's analysis seeds from these.
      saveLastUsedSettings(values);
      const newJobId = await startReprocessIfNeeded(runId);
      if (newJobId) {
        setJobId(newJobId);
        return; // modal takes over; finish() runs onComplete
      }
      setIsApplying(false);
      finish();
    } catch (err) {
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
        Analysis settings
      </Button>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent side="right" className="w-full overflow-y-auto sm:max-w-4xl">
          <SheetHeader>
            <SheetTitle>Analysis settings</SheetTitle>
            <SheetDescription>
              Applied to the results below without re-running the models.
              Also used as the starting point for your next run.
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
    </>
  );
}
