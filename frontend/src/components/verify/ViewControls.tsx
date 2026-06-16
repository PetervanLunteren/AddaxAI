/**
 * ViewControls - the "image" popover for the Counts-modal tool rail:
 * brightness and contrast for seeing a dark IR animal. View-only CSS image
 * filters; they never change stored data. Detection-confidence thresholding
 * is a Labels-page concern and deliberately not here (the boxes shown should
 * be exactly the boxes the count was computed from).
 */

import { SlidersHorizontal, RotateCcw } from "lucide-react";

import { Button } from "../ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { Slider } from "../ui/slider";

interface ViewControlsProps {
  brightness: number;
  onBrightnessChange: (v: number) => void;
  contrast: number;
  onContrastChange: (v: number) => void;
}

export function ViewControls({
  brightness,
  onBrightnessChange,
  contrast,
  onContrastChange,
}: ViewControlsProps) {
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
        {/* Brightness */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium">Brightness</span>
            <div className="flex items-center gap-1">
              <span className="text-xs text-muted-foreground tabular-nums">
                {brightness}%
              </span>
              {brightness !== 50 && (
                <button
                  onClick={() => onBrightnessChange(50)}
                  className="text-muted-foreground hover:text-foreground"
                  title="Reset to 50%"
                >
                  <RotateCcw className="h-3 w-3" />
                </button>
              )}
            </div>
          </div>
          <Slider
            value={[brightness]}
            onValueChange={([v]) => onBrightnessChange(v)}
            min={0}
            max={100}
            step={5}
          />
        </div>

        {/* Contrast */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium">Contrast</span>
            <div className="flex items-center gap-1">
              <span className="text-xs text-muted-foreground tabular-nums">
                {contrast}%
              </span>
              {contrast !== 50 && (
                <button
                  onClick={() => onContrastChange(50)}
                  className="text-muted-foreground hover:text-foreground"
                  title="Reset to 50%"
                >
                  <RotateCcw className="h-3 w-3" />
                </button>
              )}
            </div>
          </div>
          <Slider
            value={[contrast]}
            onValueChange={([v]) => onContrastChange(v)}
            min={0}
            max={100}
            step={5}
          />
        </div>
      </PopoverContent>
    </Popover>
  );
}
