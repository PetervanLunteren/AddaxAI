/**
 * Confidence range sliders block.
 *
 * Two flat label-plus-slider stacks matching the look of the
 * surrounding filter selects in the More popover. Both render on the
 * app-wide uniform 0.01–1.00 confidence scale (`ConfidenceSlider`).
 *
 * The detection slider's low handle behaviour depends on the page:
 * - ``floorMode="clamp"`` (Counts): the handle stops at the project's
 *   detection threshold, with a quiet caption while it rests there —
 *   counting is governed by the threshold setting, not this filter.
 * - ``floorMode="open"`` (Labels): the handle goes down to the scale
 *   minimum; digging below the threshold makes the grid show the
 *   low-confidence tail (the grid's banner reports any detections
 *   there that were never embedded and offers the backfill).
 *
 * Classification slider is hidden when the project has no
 * classification model.
 *
 * The component reads/writes through `EventFilterParams`'s confidence
 * fields. URL params are only emitted by the parent serializer when the
 * user has actively narrowed the range (default values pass `undefined`,
 * which the API client skips).
 */

import {
  CONFIDENCE_SCALE_MIN,
  ConfidenceSlider,
} from "../ui/confidence-slider";
import { DEFAULT_COUNTING_THRESHOLD } from "../../lib/confidence";
import type { EventFilterParams } from "../../api/types";

// Below the counting default, most boxes are false positives. Purely
// advisory: the slider still goes there, with a warning.
const DETECTION_NOISE_ADVISORY = DEFAULT_COUNTING_THRESHOLD;

interface ConfidenceRangeFilterProps {
  filters: EventFilterParams;
  onChange: (next: EventFilterParams) => void;
  /** Project's detection_threshold; the det slider's resting position
   * when no filter is set, and its clamp in ``floorMode="clamp"``. */
  detectionFloor: number;
  /** Whether the low handle stops at the floor or the scale minimum. */
  floorMode: "clamp" | "open";
  /** Caption shown while the handle rests on the clamped floor. */
  clampReason?: string;
  /** Whether to render the classification slider at all. */
  showClassification: boolean;
  /** Lowest classification confidence present in the project (from the
   * filter-options endpoint). The cls slider's low handle stops there:
   * dragging further would select nothing. Null / undefined = no
   * classifications yet, no clamp. */
  minLabelConfidence?: number | null;
}

function pct(v: number): string {
  return `${Math.round(v * 100)}%`;
}

export function ConfidenceRangeFilter({
  filters,
  onChange,
  detectionFloor,
  floorMode,
  clampReason,
  showClassification,
  minLabelConfidence,
}: ConfidenceRangeFilterProps) {
  const detEffectiveMin =
    floorMode === "clamp"
      ? Math.max(detectionFloor, CONFIDENCE_SCALE_MIN)
      : CONFIDENCE_SCALE_MIN;

  // Data-driven clamp for the cls slider: floored to the 0.01 grid so
  // the handle rests on a grid position just below the lowest value.
  const clsEffectiveMin =
    minLabelConfidence != null
      ? Math.max(
          CONFIDENCE_SCALE_MIN,
          Math.floor(minLabelConfidence * 100) / 100,
        )
      : CONFIDENCE_SCALE_MIN;

  // Effective slider values: fall back to defaults when filter is unset.
  const detMin = filters.min_confidence ?? detectionFloor;
  const detMax = filters.max_confidence ?? 1;
  const clsMin = filters.min_label_confidence ?? clsEffectiveMin;
  const clsMax = filters.max_label_confidence ?? 1;

  return (
    <>
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-muted-foreground">
          Detection confidence
        </label>
        <ConfidenceSlider
          className="h-9 px-2 flex-1"
          value={[detMin, detMax]}
          effectiveMin={detEffectiveMin}
          clampReason={floorMode === "clamp" ? clampReason : undefined}
          warnBelow={{
            value: DETECTION_NOISE_ADVISORY,
            message:
              "Most detections below 20% are false positives. " +
              "Expect noise when reviewing this range.",
          }}
          valueLabel={
            <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
              {pct(detMin)} – {pct(detMax)}
            </span>
          }
          onChange={([nextMin, nextMax]) => {
            onChange({
              ...filters,
              // At the resting floor = no filter. Below it (open
              // mode) or above it are deliberate choices and persist.
              min_confidence:
                Math.abs(nextMin - detectionFloor) < 1e-6
                  ? undefined
                  : nextMin,
              max_confidence: nextMax < 1 - 1e-6 ? nextMax : undefined,
            });
          }}
        />
      </div>

      {showClassification && (
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Classification confidence
          </label>
          <ConfidenceSlider
            className="h-9 px-2 flex-1"
            value={[clsMin, clsMax]}
            effectiveMin={clsEffectiveMin}
            clampReason={
              `No classifications below ` +
              `${Math.round(clsEffectiveMin * 100)}% in this run.`
            }
            valueLabel={
              <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                {pct(clsMin)} – {pct(clsMax)}
              </span>
            }
            onChange={([nextMin, nextMax]) => {
              onChange({
                ...filters,
                // Resting on the clamp selects everything -> no filter.
                min_label_confidence:
                  nextMin > clsEffectiveMin + 1e-6 ? nextMin : undefined,
                max_label_confidence:
                  nextMax < 1 - 1e-6 ? nextMax : undefined,
              });
            }}
          />
        </div>
      )}
    </>
  );
}
