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
 * - ``warnBelow`` (reached only by a deliberate drag) renders as a
 *   compact warning callout — the user did something that deserves a
 *   real flag.
 *
 * The component owns the row layout (track + optional value label) so
 * captions and callouts span the full width *below* the row and the
 * value label stays aligned with the track.
 */

import type { ReactNode } from "react";

import { Callout } from "./callout";
import { Slider } from "./slider";

export const CONFIDENCE_SCALE_MIN = 0.01;
export const CONFIDENCE_SCALE_MAX = 1.0;
export const CONFIDENCE_SCALE_STEP = 0.01;

interface ConfidenceSliderProps {
  /** One value (single handle) or [low, high] (range). */
  value: number | [number, number];
  onChange: (value: number[]) => void;
  /** Lowest value the (low) handle may take. Default: the scale min.
   * The track always renders the full scale regardless. */
  effectiveMin?: number;
  /** Quiet caption shown while the (low) handle rests on a clamped
   * minimum above the scale min. */
  clampReason?: string;
  /** Advisory (not a clamp): a compact warning callout shown while the
   * (low) handle sits below ``value``. */
  warnBelow?: { value: number; message: string };
  /** Rendered right of the track, aligned with it (caller styles it). */
  valueLabel?: ReactNode;
  className?: string;
}

export function ConfidenceSlider({
  value,
  onChange,
  effectiveMin = CONFIDENCE_SCALE_MIN,
  clampReason,
  warnBelow,
  valueLabel,
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
          className={className ?? "flex-1"}
        />
        {valueLabel}
      </div>
      {atClampedMin && clampReason && (
        <p className="mt-1.5 text-xs text-muted-foreground">
          {clampReason}
        </p>
      )}
      {warnBelow && shown[0] < warnBelow.value - 1e-9 && (
        <div className="mt-2">
          <Callout variant="warning" size="compact">
            {warnBelow.message}
          </Callout>
        </div>
      )}
    </div>
  );
}
