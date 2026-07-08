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
import { DETECTION_CONFIDENCE_ADVICE } from "../../lib/confidence";
import type { EventFilterParams } from "../../api/types";

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
  /** Resting position of the det slider's low handle when no explicit
   *  min filter is set (a page default, not a filter — moving the
   *  handle away from it creates the filter; moving it back clears
   *  it). Falls back to ``detectionFloor``. */
  defaultMinConfidence?: number;
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
  defaultMinConfidence,
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

  // Where the low handle rests when no explicit filter is set.
  const detRestingMin = defaultMinConfidence ?? detectionFloor;

  // A reset icon shows only while a slider is off its default.
  const detDirty =
    filters.min_confidence !== undefined ||
    filters.max_confidence !== undefined;
  const clsDirty =
    filters.min_label_confidence !== undefined ||
    filters.max_label_confidence !== undefined;

  // Effective slider values: fall back to defaults when filter is unset.
  const detMin = filters.min_confidence ?? detRestingMin;
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
          adviseBelow={DETECTION_CONFIDENCE_ADVICE}
          valueLabel={
            <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
              {pct(detMin)} – {pct(detMax)}
            </span>
          }
          onReset={() =>
            onChange({
              ...filters,
              min_confidence: undefined,
              max_confidence: undefined,
            })
          }
          resetDisabled={!detDirty}
          onChange={([nextMin, nextMax]) => {
            onChange({
              ...filters,
              // At the resting default = no filter. Below it (open
              // mode) or above it are deliberate choices and persist.
              min_confidence:
                Math.abs(nextMin - detRestingMin) < 1e-6
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
            onReset={() =>
              onChange({
                ...filters,
                min_label_confidence: undefined,
                max_label_confidence: undefined,
              })
            }
            resetDisabled={!clsDirty}
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
