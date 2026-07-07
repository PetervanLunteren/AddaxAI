/**
 * Analysis settings panel for the folder-run Labels step.
 *
 * Hosts the retroactive knobs (independence interval, smoothing,
 * taxonomic rollup) next to the label cleanup grid, where their effect
 * is visible. Changing them does not re-run the models: Apply PATCHes
 * the project and starts a reprocess job (the backend re-reads the raw
 * results.json and re-applies the transforms), with the same blocking
 * progress modal the project Settings page uses.
 *
 * Applied values are also persisted to localStorage so the next
 * folder run seeds its analysis with them (same sticky-settings
 * mechanism as the setup step's model choices).
 *
 * Collapsed by default: the step's default view stays light, and most
 * runs never need to retune these.
 */

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ChevronDown, SlidersHorizontal } from "lucide-react";
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
import { Card, CardContent } from "../ui/card";

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

export function AnalysisSettingsPanel({
  runId,
  project,
}: {
  runId: string;
  project: ProjectResponse;
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
      <Card>
        <CardContent className="p-0">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="flex w-full items-center justify-between px-6 py-4 text-left"
          >
            <span className="flex items-center gap-2 text-sm font-semibold">
              <SlidersHorizontal className="h-4 w-4 text-muted-foreground" />
              Analysis settings
              <span className="font-normal text-muted-foreground">
                grouping, smoothing, rollup
              </span>
            </span>
            <ChevronDown
              className={`h-4 w-4 text-muted-foreground transition-transform ${
                open ? "rotate-180" : ""
              }`}
            />
          </button>
          {open && (
            <div className="space-y-0 divide-y border-t px-6 [&>*:last-child]:pb-6">
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
              <div className="flex items-center justify-between gap-3 py-4">
                <p className="text-xs text-muted-foreground">
                  Applies to the results below without re-running the
                  models. Also used as the starting point for your next
                  run.
                </p>
                <Button
                  size="sm"
                  onClick={apply}
                  disabled={!dirty || isApplying || !!jobId}
                >
                  {isApplying || jobId ? "Applying..." : "Apply and reprocess"}
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <ApplySettingsModal
        open={isApplying || !!jobId}
        message={progress.message}
        progress={progress.progress}
        fallbackMessage="Saving settings..."
      />
    </>
  );
}
