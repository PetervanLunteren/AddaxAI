/**
 * Per-phase analysis progress bars.
 *
 * Single source of truth for the visual breakdown of a running ML
 * pipeline (video detection, video classification, image detection,
 * image classification, embedding). Driven by the live data the
 * `useTaskProgress` hook returns; agnostic about whether the run is a
 * main-app deployment queue (RunQueueModal) or a one-shot Timelapse
 * run (TimelapseModePage).
 *
 * Why this exists as a shared file: prior versions had RunQueueModal's
 * detailed PhaseRow / phase-order logic copied or simplified for the
 * Timelapse running screen. That meant a tweak to the detection-progress
 * row had to be made in two places. Both call sites now render the same
 * <AnalysisProgress /> and inherit fixes for free.
 */

import { Loader2 } from "lucide-react";
import { Progress } from "../ui/progress";
import { Separator } from "../ui/separator";
import type {
  DeploymentContext,
  TqdmMetrics,
} from "../../hooks/useTaskProgress";

/** Phase identifiers in the order the worker emits them. */
const PHASE_ORDER = [
  "init",
  "video_detection",
  "video_classification",
  "image_detection",
  "image_classification",
  "saving",
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

interface PhaseRowProps {
  label: string;
  phaseName: string;
  progress: number;
  currentPhase: string | null;
  phaseProgress: number | undefined;
  metrics: TqdmMetrics | null;
  computeDevice: string | null;
}

function PhaseRow({
  label,
  phaseName,
  progress,
  currentPhase,
  phaseProgress,
  metrics,
  computeDevice,
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
  const capitalizedUnit = unit.charAt(0).toUpperCase() + unit.slice(1);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-gray-700">{label}</p>
        <span className="text-xs text-gray-500 font-mono">
          {progress.toFixed(0)}%
        </span>
      </div>
      <Progress value={progress} className="h-2" />

      {hasValidMetrics && metrics && (
        <div className="text-[11px] space-y-0.5 rounded-md bg-gray-50 p-3 font-mono text-gray-600">
          <div className="flex justify-between">
            <span>Processing {unit}:</span>
            <span>
              {metrics.current} of {metrics.total}
            </span>
          </div>
          {metrics.elapsed && (
            <div className="flex justify-between">
              <span>Elapsed time:</span>
              <span>{metrics.elapsed}</span>
            </div>
          )}
          {metrics.remaining && (
            <div className="flex justify-between">
              <span>Remaining time:</span>
              <span>{metrics.remaining}</span>
            </div>
          )}
          {metrics.rate && (
            <div className="flex justify-between">
              <span>{capitalizedUnit} per second:</span>
              <span>{metrics.rate.toFixed(2)}</span>
            </div>
          )}
          <div className="flex justify-between">
            <span>Running on:</span>
            <span className={computeDevice ? "" : "text-gray-400"}>
              {computeDevice ?? "detecting..."}
            </span>
          </div>
        </div>
      )}

      {isStartingUp && (
        <div className="flex items-center gap-2 text-[11px] font-mono text-gray-500 px-1">
          <Loader2 className="h-3 w-3 animate-spin" style={{ color: "#156065" }} />
          <span>Starting up...</span>
        </div>
      )}

      {isFinalizing && (
        <div className="flex items-center gap-2 text-[11px] font-mono text-gray-500 px-1">
          <Loader2 className="h-3 w-3 animate-spin" style={{ color: "#156065" }} />
          <span>Finalizing...</span>
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
   * Hide the "Deployment X of N" badge. Set this for one-shot runs
   * (Timelapse integration) where there is no concept of multiple deployments.
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
  hideDeploymentHeader = false,
}: AnalysisProgressProps) {
  if (!deploymentContext) return null;

  const phases = [
    deploymentContext.videoCount > 0 && {
      label: "Video detection",
      phase: "video_detection",
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
    deploymentContext.hasEmbedding && {
      label: "Embedding",
      phase: "embedding",
    },
  ].filter(Boolean) as { label: string; phase: string }[];

  return (
    <div className="border rounded-lg p-4 space-y-4">
      {!hideDeploymentHeader && (
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-gray-600">Deployment</span>
          <span className="inline-flex items-center rounded-md bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-800">
            {deploymentContext.deploymentIndex} of{" "}
            {deploymentContext.totalDeployments}
          </span>
        </div>
      )}

      {phases.map((entry, idx) => (
        <div key={entry.phase}>
          {(idx > 0 || !hideDeploymentHeader) && <Separator className="mb-4" />}
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
          />
        </div>
      ))}
    </div>
  );
}
