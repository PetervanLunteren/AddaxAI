/**
 * Shared confidence slider — the one scale for every confidence control
 * in the app.
 *
 * Every instance renders the full 0.01–1.00 track (step 0.01), so the
 * same physical position always means the same confidence, whichever
 * page the slider is on. A control whose values cannot go that low
 * passes ``effectiveMin``: the track still shows the full scale, the
 * handle simply stops there. While a handle rests on a clamped minimum
 * (> 0.01), the ``clampReason`` renders as a compact info callout so
 * the stop never reads as a broken slider.
 *
 * Single- and range-mode share the component so the clamp and scale
 * logic exist once.
 */

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
  /** Shown as a compact info callout while the (low) handle rests on a
   * clamped minimum above the scale min. */
  clampReason?: string;
  className?: string;
}

export function ConfidenceSlider({
  value,
  onChange,
  effectiveMin = CONFIDENCE_SCALE_MIN,
  clampReason,
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
        className={className}
      />
      {atClampedMin && clampReason && (
        <div className="mt-2">
          <Callout variant="info" size="compact">
            {clampReason}
          </Callout>
        </div>
      )}
    </div>
  );
}
