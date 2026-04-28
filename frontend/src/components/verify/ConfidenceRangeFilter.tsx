/**
 * ConfidenceRangeFilter — collapsible block with two range sliders.
 *
 * Detection slider clamped at the project's detection_threshold (low
 * handle never goes below). Classification slider hidden when the
 * project has no classification model.
 *
 * The component reads/writes through `EventFilterParams`'s confidence
 * fields. URL params are only emitted by the parent serializer when
 * the user has actively narrowed the range (default values pass
 * `undefined`, which the API client skips).
 */

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

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
  const [open, setOpen] = useState(false);

  // Effective slider values: fall back to defaults when filter is unset.
  const detMin = filters.min_confidence ?? detectionFloor;
  const detMax = filters.max_confidence ?? 1;
  const clsMin = filters.min_label_confidence ?? 0;
  const clsMax = filters.max_label_confidence ?? 1;

  const detActive =
    detMin > detectionFloor + 1e-6 || detMax < 1 - 1e-6;
  const clsActive =
    showClassification && (clsMin > 1e-6 || clsMax < 1 - 1e-6);
  const activeCount = (detActive ? 1 : 0) + (clsActive ? 1 : 0);

  return (
    <div className="rounded-md border bg-muted/30">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground"
      >
        <span className="flex items-center gap-1.5">
          {open ? (
            <ChevronDown className="h-3.5 w-3.5" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5" />
          )}
          Advanced
          {activeCount > 0 && (
            <span className="ml-1 rounded-full bg-primary px-1.5 py-0.5 text-[10px] text-primary-foreground">
              {activeCount} active
            </span>
          )}
        </span>
      </button>

      {open && (
        <div className="px-3 pb-3 pt-1 space-y-4">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium text-muted-foreground">
                Detection confidence
              </span>
              <span className="tabular-nums text-muted-foreground">
                {pct(detMin)} – {pct(detMax)}
              </span>
            </div>
            <Slider
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
            <p className="text-[11px] text-muted-foreground">
              Project floor: {pct(detectionFloor)}. Slider cannot go below.
            </p>
          </div>

          {showClassification && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-muted-foreground">
                  Classification confidence
                </span>
                <span className="tabular-nums text-muted-foreground">
                  {pct(clsMin)} – {pct(clsMax)}
                </span>
              </div>
              <Slider
                value={[clsMin, clsMax]}
                min={0}
                max={1}
                step={STEP}
                onValueChange={([nextMin, nextMax]) => {
                  onChange({
                    ...filters,
                    min_label_confidence:
                      nextMin > 1e-6 ? nextMin : undefined,
                    max_label_confidence:
                      nextMax < 1 - 1e-6 ? nextMax : undefined,
                  });
                }}
              />
              <p className="text-[11px] text-muted-foreground">
                Detections without a classification (NULL) are excluded
                when this range is active.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
