/**
 * Confidence range sliders block.
 *
 * Two flat label-plus-slider stacks matching the look of the surrounding
 * filter selects in the More popover. Detection slider is clamped at the
 * project's detection_threshold (low handle never goes below).
 * Classification slider is hidden when the project has no classification
 * model.
 *
 * The component reads/writes through `EventFilterParams`'s confidence
 * fields. URL params are only emitted by the parent serializer when the
 * user has actively narrowed the range (default values pass `undefined`,
 * which the API client skips).
 */

import { Slider } from "../ui/slider";
import type { EventFilterParams } from "../../api/types";

interface ConfidenceRangeFilterProps {
  filters: EventFilterParams;
  onChange: (next: EventFilterParams) => void;
  /** Project's detection_threshold; clamps the low handle of the det slider. */
  detectionFloor: number;
  /** Whether to render the classification slider at all. */
  showClassification: boolean;
}

const STEP = 0.05;

function pct(v: number): string {
  return `${Math.round(v * 100)}%`;
}

export function ConfidenceRangeFilter({
  filters,
  onChange,
  detectionFloor,
  showClassification,
}: ConfidenceRangeFilterProps) {
  // Effective slider values: fall back to defaults when filter is unset.
  const detMin = filters.min_confidence ?? detectionFloor;
  const detMax = filters.max_confidence ?? 1;
  const clsMin = filters.min_label_confidence ?? 0;
  const clsMax = filters.max_label_confidence ?? 1;

  return (
    <>
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-muted-foreground">
          Detection confidence
        </label>
        <div className="flex items-center gap-3">
          <Slider
            className="h-9 px-2 flex-1"
            value={[detMin, detMax]}
            min={detectionFloor}
            max={1}
            step={STEP}
            onValueChange={([nextMin, nextMax]) => {
              onChange({
                ...filters,
                min_confidence:
                  nextMin > detectionFloor + 1e-6 ? nextMin : undefined,
                max_confidence: nextMax < 1 - 1e-6 ? nextMax : undefined,
              });
            }}
          />
          <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
            {pct(detMin)} – {pct(detMax)}
          </span>
        </div>
      </div>

      {showClassification && (
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Classification confidence
          </label>
          <div className="flex items-center gap-3">
            <Slider
              className="h-9 px-2 flex-1"
              value={[clsMin, clsMax]}
              min={0}
              max={1}
              step={STEP}
              onValueChange={([nextMin, nextMax]) => {
                onChange({
                  ...filters,
                  min_label_confidence: nextMin > 1e-6 ? nextMin : undefined,
                  max_label_confidence: nextMax < 1 - 1e-6 ? nextMax : undefined,
                });
              }}
            />
            <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
              {pct(clsMin)} – {pct(clsMax)}
            </span>
          </div>
        </div>
      )}
    </>
  );
}
