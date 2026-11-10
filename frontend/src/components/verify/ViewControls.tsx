/**
 * ViewControls - the "image" popover for the viewer tool rail: brightness
 * and contrast for seeing a dark IR animal. View-only CSS image filters;
 * they never change stored data. Detection-confidence thresholding is a
 * Labels-page concern and deliberately not here (the boxes shown should
 * be exactly the boxes the count was computed from).
 *
 * The sliders read and write the shared store in viewer-tools.ts, so the
 * rail, the grid "View options" popovers and the Counts toolbar all show
 * one value. `ImageAdjustRows` is the bare rows for hosts that already
 * have a popover of their own.
 */

import { SlidersHorizontal, RotateCcw } from "lucide-react";

import { Button } from "../ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { Slider } from "../ui/slider";

import { useImageAdjust } from "./viewer-tools";

function AdjustRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium">{label}</span>
        <div className="flex items-center gap-1">
          <span className="text-xs text-muted-foreground tabular-nums">
            {value}%
          </span>
          {value !== 50 && (
            <button
              onClick={() => onChange(50)}
              className="text-muted-foreground hover:text-foreground"
              title="Reset to 50%"
            >
              <RotateCcw className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>
      <Slider
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={0}
        max={100}
        step={5}
      />
    </div>
  );
}

/** The brightness and contrast rows, bound to the shared store. */
export function ImageAdjustRows() {
  const { brightness, setBrightness, contrast, setContrast } =
    useImageAdjust();
  return (
    <>
      <AdjustRow
        label="Brightness"
        value={brightness}
        onChange={setBrightness}
      />
      <AdjustRow label="Contrast" value={contrast} onChange={setContrast} />
    </>
  );
}

export function ViewControls() {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          title="Image (brightness, contrast)"
        >
          <SlidersHorizontal className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent side="right" align="start" className="w-56 p-3 space-y-3">
        <ImageAdjustRows />
      </PopoverContent>
    </Popover>
  );
}
