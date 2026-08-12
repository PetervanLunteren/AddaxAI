/**
 * Body for the JobProgressModal shown during the Save outputs job.
 *
 * Renders the per-module checklist the worker drives via WebSocket
 * progress events:
 * - ☑ done — modules whose index is below the current
 * - ▶ running — the one currently being executed
 * - ☐ pending — modules waiting their turn
 *
 * The worker's progress message also feeds the small status line
 * at the top, e.g. "Separating files (3 / 8)".
 *
 * The bar under the running module shows that module's own fraction
 * (the worker's phase_progress), so it always matches the "(N / M)"
 * count in the status line. It used to be one overall bar across all
 * modules, which sat near zero for the whole first module — on a big
 * save that is most of the wall time, and it read as stuck. Same
 * per-phase convention as AnalysisProgress. Modules that report no
 * per-file counts (the quick data writers) show the spinner only.
 */

import { Check, Loader2, Square } from "lucide-react";

import { Progress } from "../ui/progress";

const MODULE_LABELS: Record<string, string> = {
  separate_folders: "Separating files",
  annotated_copies: "Writing annotated copies",
  recognition_json: "Writing recognition JSON",
  csv: "Writing CSV",
  xlsx: "Writing XLSX",
  run_readme: "Writing run details",
};

interface SaveOutputsProgressProps {
  modules: string[];
  currentModule: string | null;
  moduleIndex: number;
  totalModules: number;
  message: string;
  /** 0..1 fraction within the running module (worker phase_progress).
   * 0 when the running module reports no per-file counts. */
  phaseProgress: number;
}

function moduleState(
  idx: number,
  currentModule: string | null,
  moduleIndex: number,
  totalModules: number,
): "done" | "running" | "pending" {
  if (currentModule === null) {
    // Pre-start: moduleIndex === 0, nothing done yet.
    // Post-end: moduleIndex === totalModules, everything done.
    return moduleIndex >= totalModules ? "done" : "pending";
  }
  if (idx === moduleIndex) return "running";
  return idx < moduleIndex ? "done" : "pending";
}

export function SaveOutputsProgress({
  modules,
  currentModule,
  moduleIndex,
  totalModules,
  message,
  phaseProgress,
}: SaveOutputsProgressProps) {
  return (
    <div className="space-y-4">
      {message && (
        <p className="text-sm text-muted-foreground">{message}</p>
      )}
      <ul className="space-y-1.5 text-sm">
        {modules.map((module, idx) => {
          const state = moduleState(
            idx,
            currentModule,
            moduleIndex,
            totalModules,
          );
          return (
            <li key={module}>
              <div className="flex items-center gap-2">
                {state === "done" && (
                  <Check className="h-4 w-4 text-primary" />
                )}
                {state === "running" && (
                  <Loader2 className="h-4 w-4 animate-spin text-primary" />
                )}
                {state === "pending" && (
                  <Square className="h-4 w-4 text-muted-foreground" />
                )}
                <span
                  className={
                    state === "pending"
                      ? "text-muted-foreground"
                      : "font-medium"
                  }
                >
                  {MODULE_LABELS[module] ?? module}
                </span>
              </div>
              {state === "running" && phaseProgress > 0 && (
                <Progress
                  value={Math.round(phaseProgress * 100)}
                  className="ml-6 mt-1.5 h-1.5 w-auto"
                />
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
