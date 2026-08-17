/**
 * Shared confidence slider — the one scale for every confidence control
 * in the app.
 *
 * Every instance renders the full 0.01–1.00 track (step 0.01), so the
 * same physical position always means the same confidence, whichever
 * page the slider is on. A control whose values cannot go that low
 * passes ``effectiveMin``: the track still shows the full scale, the
 * handle simply stops there.
 *
 * Explanations under the track come in two visual weights:
 * - ``clampReason`` (a resting boundary, often the slider's default
 *   position) renders as a quiet one-line caption — always true, never
 *   alarming, must not shout.
 * - ``adviseBelow`` (reached only by a deliberate drag) renders as a
 *   compact warning callout — the user did something that deserves a
 *   real flag.
 *
 * The component owns the row layout (track + optional value label) so
 * captions and callouts span the full width *below* the row and the
 * value label stays aligned with the track.
 */

import type { ReactNode } from "react";

import { RotateCcw } from "lucide-react";

import { Callout } from "./callout";
import { Slider } from "./slider";

export const CONFIDENCE_SCALE_MIN = 0.01;
export const CONFIDENCE_SCALE_MAX = 1.0;
export const CONFIDENCE_SCALE_STEP = 0.01;

interface ConfidenceSliderProps {
  /** One value (single handle) or [low, high] (range). */
  value: number | [number, number];
  onChange: (value: number[]) => void;
  /** Fires once when the handle is released, not on every step. Use it
   * for work too expensive to run per tick, such as a request. Radix
   * gives us this for free; ``onChange`` still fires while dragging so
   * the handle and the label keep up. */
  onCommit?: (value: number[]) => void;
  /** Lowest value the (low) handle may take. Default: the scale min.
   * The track always renders the full scale regardless. */
  effectiveMin?: number;
  /** Quiet caption shown while the (low) handle rests on a clamped
   * minimum above the scale min. */
  clampReason?: string;
  /** Advisory (not a clamp): a compact info callout shown while the
   * (low) handle sits below ``value``. Info, not warning: the user did
   * something deliberate and supported; the message provides context,
   * the container must not imply a mistake. */
  adviseBelow?: { value: number; message: string };
  /** Rendered right of the track, aligned with it (caller styles it). */
  valueLabel?: ReactNode;
  /** When provided, a small ghost reset icon renders after the value
   * label — always, so the row never shifts while dragging. Pair with
   * ``resetDisabled`` to make it inert while already at the default. */
  onReset?: () => void;
  /** Dims and disables the reset icon (already at the default). */
  resetDisabled?: boolean;
  className?: string;
}

export function ConfidenceSlider({
  value,
  onChange,
  onCommit,
  effectiveMin = CONFIDENCE_SCALE_MIN,
  clampReason,
  adviseBelow,
  valueLabel,
  onReset,
  resetDisabled = false,
  className,
}: ConfidenceSliderProps) {
  const values = Array.isArray(value) ? value : [value];
  // Values persisted below the clamp (older URLs / settings) display at
  // the clamp so the handle and the effective behaviour agree.
  const shown = values.map((v, i) =>
    i === 0 ? Math.max(v, effectiveMin) : v,
  );

  const atClampedMin =
    effectiveMin > CONFIDENCE_SCALE_MIN &&
    shown[0] <= effectiveMin + 1e-9;

  return (
    <div className="min-w-0 flex-1">
      <div className="flex items-center gap-3">
        <Slider
          min={CONFIDENCE_SCALE_MIN}
          max={CONFIDENCE_SCALE_MAX}
          step={CONFIDENCE_SCALE_STEP}
          value={shown}
          onValueChange={(next) => {
            const clamped = next.map((v, i) =>
              i === 0 ? Math.max(v, effectiveMin) : v,
            );
            onChange(clamped);
          }}
          onValueCommit={
            onCommit
              ? (next) =>
                  onCommit(
                    next.map((v, i) =>
                      i === 0 ? Math.max(v, effectiveMin) : v,
                    ),
                  )
              : undefined
          }
          className={className ?? "flex-1"}
        />
        {valueLabel}
        {onReset && (
          <button
            type="button"
            title="Reset to default"
            aria-label="Reset to default"
            onClick={onReset}
            disabled={resetDisabled}
            className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-30"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
      {atClampedMin && clampReason && (
        <p className="mt-1.5 text-xs text-muted-foreground">
          {clampReason}
        </p>
      )}
      {adviseBelow && shown[0] < adviseBelow.value - 1e-9 && (
        <div className="mt-2">
          <Callout variant="info" size="compact">
            {adviseBelow.message}
          </Callout>
        </div>
      )}
    </div>
  );
}
