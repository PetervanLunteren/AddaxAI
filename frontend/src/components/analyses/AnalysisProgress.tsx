/**
 * Per-phase analysis progress bars.
 *
 * Single source of truth for the visual breakdown of a running ML
 * pipeline (video detection, video classification, image detection,
 * image classification, embedding). Driven by the live data the
 * `useTaskProgress` hook returns; agnostic about which run drives it
 * (research-projects deployment queue or a folder run), both via
 * RunQueueModal.
 *
 * Why this exists as a shared file: prior versions had RunQueueModal's
 * detailed PhaseRow / phase-order logic copied or simplified for other
 * running screens. That meant a tweak to the detection-progress row had
 * to be made in two places. Call sites now render the same
 * <AnalysisProgress /> and inherit fixes for free.
 */

import { useState } from "react";
import { ChevronRight, Loader2 } from "lucide-react";
import { Progress } from "../ui/progress";
import { Separator } from "../ui/separator";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "../ui/collapsible";
import { cn } from "../../lib/utils";
import {
  formatRate,
  formatRateShort,
  humanizeTqdmTime,
} from "../../lib/duration";
import type {
  DeploymentContext,
  TqdmMetrics,
} from "../../hooks/useTaskProgress";

/** Phase identifiers in the order the worker emits them. */
const PHASE_ORDER = [
  "init",
  "video_detection",
  "video_frame_selection",
  "video_classification",
  "image_detection",
  "image_classification",
  "saving",
  "postprocessing",
  "embedding",
  "finalize",
] as const;

type PhaseName = (typeof PHASE_ORDER)[number];

/**
 * Compute completion percentage (0-100) for a target phase given the
 * live progress signal.
 *
 * Earlier phases (already passed) report 100. Later phases (not yet
 * reached) report 0. The current phase reports either `phaseProgress`
 * scaled to 100, or — if tqdm metrics are richer than the bare
 * phaseProgress — `metrics.current / metrics.total * 100`.
 */
export function getPhaseProgress(
  targetPhase: string,
  currentPhase: string | null,
  phaseProgress: number | undefined,
  metrics: TqdmMetrics | null,
): number {
  const currentIndex = currentPhase
    ? (PHASE_ORDER as readonly string[]).indexOf(currentPhase)
    : -1;
  const targetIndex = (PHASE_ORDER as readonly string[]).indexOf(targetPhase);
  if (currentIndex < targetIndex) return 0;
  if (currentIndex > targetIndex) return 100;

  if (currentPhase === targetPhase) {
    if (phaseProgress !== undefined && phaseProgress >= 1.0) return 100;
    if (
      metrics?.current !== undefined &&
      metrics?.total !== undefined &&
      metrics.total > 0
    ) {
      return (metrics.current / metrics.total) * 100;
    }
    return 0;
  }
  return 0;
}

/**
 * Backend pushes a mix of clean status strings ("Loading DINOv2 model...",
 * "Saving classifications...") and raw tqdm lines ("Embedding: 53%|██..."
 * etc). Raw tqdm lines are already conveyed by the progress bar and look
 * awful in a caption, so we strip them and fall back to the generic
 * caption for anything that smells like tqdm output.
 */
function _cleanStatusMessage(message: string | undefined): string | null {
  if (!message) return null;
  const trimmed = message.trim();
  if (!trimmed) return null;
  if (trimmed.includes("%|") || trimmed.includes("it/s") || trimmed.includes("/s]")) {
    return null;
  }
  return trimmed;
}

interface PhaseRowProps {
  label: string;
  phaseName: string;
  progress: number;
  currentPhase: string | null;
  phaseProgress: number | undefined;
  metrics: TqdmMetrics | null;
  computeDevice: string | null;
  /**
   * Latest backend status line (e.g. "Loading DINOv2 model...", "Saving
   * classifications..."). Used to replace the generic "Starting up..."
   * and "Finalizing..." captions so the user sees what's actually
   * happening during long stretches with no tqdm progress.
   */
  message?: string;
  /**
   * Whether the metrics block is expanded. Owned by the parent and
   * shared by every row, not held per row: only one phase is ever
   * active, so per-row state would silently reset each time the run
   * moved on and the user would have to reopen it six times.
   */
  detailsOpen: boolean;
  onDetailsOpenChange: (open: boolean) => void;
}

function PhaseRow({
  label,
  phaseName,
  progress,
  currentPhase,
  phaseProgress,
  metrics,
  computeDevice,
  message,
  detailsOpen,
  onDetailsOpenChange,
}: PhaseRowProps) {
  const isActive = phaseName === currentPhase;
  const isFinalizing = isActive && progress >= 100;
  const hasValidMetrics =
    isActive &&
    !isFinalizing &&
    metrics?.current !== undefined &&
    metrics?.total !== undefined &&
    metrics.current < metrics.total;
  const isStartingUp =
    isActive &&
    !isFinalizing &&
    !hasValidMetrics &&
    (phaseProgress === undefined || phaseProgress < 1.0);

  const unit = metrics?.unit || "items";
  const rateDisplay = metrics?.rate ? formatRate(metrics.rate, unit) : null;

  // What the collapsed toggle shows beside it, so closing the details
  // never costs you the answer to "how long is this going to take".
  // Every part is optional: the first update of a phase has a count but
  // no rate or estimate yet.
  const summary = [
    metrics?.current !== undefined && metrics?.total !== undefined
      ? `${metrics.current.toLocaleString()} of ${metrics.total.toLocaleString()}`
      : null,
    metrics?.rate ? formatRateShort(metrics.rate, unit) : null,
    metrics?.remaining
      ? `${humanizeTqdmTime(metrics.remaining, true)} left`
      : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div>
      {/* One line per phase. The label and percentage columns are fixed
          so the bars line up down the list, which is what makes six of
          them scannable at a glance. */}
      <div className="flex items-center gap-3 py-1">
        {/* w-36, not w-32. The longest label ("Video frame selection")
            measures 126px, so a 128px column clears it by 2px here and
            would truncate on a machine whose fonts render a hair wider.
            The 16px comes off the bar, which has it to spare. */}
        <p className="w-36 shrink-0 truncate text-xs font-medium text-gray-700">
          {label}
        </p>
        <Progress value={progress} className="h-1.5 flex-1" />
        <span className="w-9 shrink-0 text-right text-xs text-gray-500 font-mono">
          {progress.toFixed(0)}%
        </span>
      </div>

      {/* The active phase adds exactly one caption line: either what it
          is busy with (no tqdm output yet) or its numbers. Never both,
          so a row can only ever grow by one line. */}
      {hasValidMetrics && metrics && (
        <Collapsible open={detailsOpen} onOpenChange={onDetailsOpenChange}>
          {/* items-center, not items-baseline. The trigger is a flex box
              whose first child is the chevron SVG, and a flex container
              takes its baseline from its first item, so baseline
              alignment lifts "Details" above the summary beside it. The
              two spans are the same size, so centring lines them up. */}
          <div className="flex items-center gap-2 pb-1 pl-1">
            <CollapsibleTrigger className="flex shrink-0 items-center gap-1 text-[11px] font-medium text-primary hover:underline">
              <ChevronRight
                className={cn(
                  "h-3 w-3 transition-transform",
                  detailsOpen && "rotate-90",
                )}
              />
              Details
            </CollapsibleTrigger>
            {!detailsOpen && summary && (
              <span className="truncate text-[11px] font-mono text-gray-500">
                {summary}
              </span>
            )}
          </div>
          <CollapsibleContent>
            <div className="mb-1 text-[11px] space-y-0.5 rounded-md bg-gray-50 p-2.5 font-mono text-gray-600">
          <div className="flex justify-between">
            <span>Processing {unit}:</span>
            <span>
              {metrics.current?.toLocaleString()} of{" "}
              {metrics.total?.toLocaleString()}
            </span>
          </div>
          {metrics.elapsed && (
            <div className="flex justify-between">
              <span>Elapsed time:</span>
              <span>{humanizeTqdmTime(metrics.elapsed)}</span>
            </div>
          )}
          {metrics.remaining && (
            <div className="flex justify-between">
              <span>Remaining time:</span>
              <span>{humanizeTqdmTime(metrics.remaining, true)}</span>
            </div>
          )}
          {rateDisplay && (
            <div className="flex justify-between">
              <span>{rateDisplay.label}:</span>
              <span>{rateDisplay.value}</span>
            </div>
          )}
          {/* Only when this phase told us what it runs on. A phase that
              does its own CPU work between two GPU phases (video frame
              selection) says nothing, because a lone "CPU" row reads as
              a fault rather than as the ordinary fact that decoding
              video is not GPU work. */}
          {computeDevice && (
            <div className="flex justify-between">
              <span>Running on:</span>
              <span>{computeDevice}</span>
            </div>
          )}
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {(isStartingUp || isFinalizing) && (
        <div className="flex items-center gap-2 pb-1 pl-1 text-[11px] font-mono text-gray-500">
          <Loader2 className="h-3 w-3 animate-spin" style={{ color: "#156065" }} />
          <span className="truncate">
            {_cleanStatusMessage(message) ??
              (isFinalizing ? "Finalizing..." : "Starting up...")}
          </span>
        </div>
      )}
    </div>
  );
}

interface AnalysisProgressProps {
  /** Current phase (null while waiting for the first websocket event). */
  phase: PhaseName | string | null;
  /** Progress within the current phase, 0..1. */
  phaseProgress: number | undefined;
  /** Latest tqdm-style metrics block, if any. */
  metrics: TqdmMetrics | null;
  /** Compute device label as reported by the subprocess. */
  computeDevice: string | null;
  /** Deployment context from the Init websocket message. */
  deploymentContext: DeploymentContext | null;
  /**
   * Latest status string from the worker (e.g. "Loading DINOv2 model...",
   * "Saving classifications..."). Surfaces inside the current phase's
   * starting-up / finalizing captions so the user sees what's actually
   * happening when there are no tqdm metrics yet.
   */
  message?: string;
  /**
   * Hide the "Deployment X of N" badge. Set this for one-shot runs
   * where there is no concept of multiple deployments.
   */
  hideDeploymentHeader?: boolean;
}

/**
 * Render the per-phase progress block. Decides which phases to show
 * based on the deployment context (videos? images? classifier?
 * embedding?). Returns null when the deployment context has not yet
 * arrived from the worker — the caller renders a spinner in that case.
 */
export function AnalysisProgress({
  phase,
  phaseProgress,
  metrics,
  computeDevice,
  deploymentContext,
  message,
  hideDeploymentHeader = false,
}: AnalysisProgressProps) {
  // One toggle for the whole block, and it must be declared before the
  // early return below or the hook count changes with the props.
  const [detailsOpen, setDetailsOpen] = useState(false);

  if (!deploymentContext) return null;

  // Only show the "Deployment X of N" badge when there's actually a
  // queue to position the user in. A single deployment (always the
  // case in folder mode, and common for one-off project runs) carries
  // no information as "1 of 1", so we drop it.
  const showDeploymentHeader =
    !hideDeploymentHeader && deploymentContext.totalDeployments > 1;

  const phases = [
    deploymentContext.videoCount > 0 && {
      label: "Video detection",
      phase: "video_detection",
    },
    // Only without a classifier. With one, the classification worker
    // picks and writes each video's frame as it goes, so the work is
    // already inside the row below and a second row would be a lie.
    deploymentContext.videoCount > 0 &&
      !deploymentContext.hasClassifier && {
        label: "Video frame selection",
        phase: "video_frame_selection",
      },
    deploymentContext.videoCount > 0 &&
      deploymentContext.hasClassifier && {
        label: "Video classification",
        phase: "video_classification",
      },
    deploymentContext.imageCount > 0 && {
      label: "Image detection",
      phase: "image_detection",
    },
    deploymentContext.imageCount > 0 &&
      deploymentContext.hasClassifier && {
        label: "Image classification",
        phase: "image_classification",
      },
    // The worker emits phase="saving" between classification and
    // embedding (merging results, loading to database). Without this
    // row, the user sees a long silent gap with image classification
    // stuck on Finalizing... and embedding still at 0%.
    {
      label: "Saving",
      phase: "saving",
    },
    // Its own row rather than the tail of "Saving". This is label
    // exclusion, geofencing, taxonomic rollup and smoothing, which on a
    // large deployment is minutes of a different kind of work. Sharing
    // the saving row meant that row sat at 100% while this ran, which is
    // the lie these rows exist to avoid. Named for the "Refine results"
    // settings that drive it.
    {
      label: "Refining results",
      phase: "postprocessing",
    },
    deploymentContext.hasEmbedding && {
      label: "Embedding",
      phase: "embedding",
    },
  ].filter(Boolean) as { label: string; phase: string }[];

  return (
    <div className="border rounded-lg p-4">
      {showDeploymentHeader && (
        <>
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-gray-600">Deployment</span>
            <span className="inline-flex items-center rounded-md bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-800">
              {deploymentContext.deploymentIndex} of{" "}
              {deploymentContext.totalDeployments}
            </span>
          </div>
          <Separator className="my-3" />
        </>
      )}

      {/* No rule between rows. Six of them each needed a separator plus
          its margins, which cost more vertical space than the bars. The
          fixed label and percentage columns already read as a table. */}
      {phases.map((entry) => (
        <div key={entry.phase}>
          <PhaseRow
            label={entry.label}
            phaseName={entry.phase}
            progress={getPhaseProgress(
              entry.phase,
              phase as string | null,
              phaseProgress,
              metrics,
            )}
            currentPhase={phase as string | null}
            phaseProgress={phaseProgress}
            metrics={metrics}
            computeDevice={computeDevice}
            message={message}
            detailsOpen={detailsOpen}
            onDetailsOpenChange={setDetailsOpen}
          />
        </div>
      ))}
    </div>
  );
}
