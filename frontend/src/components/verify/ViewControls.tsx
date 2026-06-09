/**
 * ViewControls - the view-adjust popovers shared by the event modal's
 * frame toolbar and the gallery toolbar: detection-confidence threshold,
 * brightness, and contrast. All three are view-only (a local threshold
 * override and CSS image filters); they never change stored data.
 */

import { Scale, Sun, Contrast, RotateCcw } from "lucide-react";

import { Button } from "../ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { Slider } from "../ui/slider";

interface ViewControlsProps {
  detectionThreshold: number;
  projectThreshold: number;
  localThreshold: number | null;
  onThresholdChange: (v: number | null) => void;
  brightness: number;
  onBrightnessChange: (v: number) => void;
  contrast: number;
  onContrastChange: (v: number) => void;
}

export function ViewControls({
  detectionThreshold,
  projectThreshold,
  localThreshold,
  onThresholdChange,
  brightness,
  onBrightnessChange,
  contrast,
  onContrastChange,
}: ViewControlsProps) {
  return (
    <>
      {/* View threshold (local override; does not change the project's
          detection_threshold). */}
      <Popover>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            title={`View threshold: ${(detectionThreshold * 100).toFixed(0)}%`}
          >
            <Scale className="h-4 w-4" />
          </Button>
        </PopoverTrigger>
        <PopoverContent side="bottom" className="w-48 p-3">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium">View threshold</span>
              <div className="flex items-center gap-1">
                <span className="text-xs text-muted-foreground tabular-nums">
                  {(detectionThreshold * 100).toFixed(0)}%
                </span>
                {localThreshold !== null && (
                  <button
                    onClick={() => onThresholdChange(null)}
                    className="text-xs text-muted-foreground hover:text-foreground"
                    title={`Reset to project default (${(projectThreshold * 100).toFixed(0)}%)`}
                  >
                    <RotateCcw className="h-3 w-3" />
                  </button>
                )}
              </div>
            </div>
            <Slider
              value={[detectionThreshold]}
              onValueChange={([v]) => onThresholdChange(v)}
              min={0}
              max={1}
              step={0.05}
            />
          </div>
        </PopoverContent>
      </Popover>

      {/* Brightness */}
      <Popover>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            title={`Brightness: ${brightness}%`}
          >
            <Sun className="h-4 w-4" />
          </Button>
        </PopoverTrigger>
        <PopoverContent side="bottom" className="w-48 p-3">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium">Brightness</span>
              <div className="flex items-center gap-1">
                <span className="text-xs text-muted-foreground tabular-nums">
                  {brightness}%
                </span>
                {brightness !== 50 && (
                  <button
                    onClick={() => onBrightnessChange(50)}
                    className="text-xs text-muted-foreground hover:text-foreground"
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
        </PopoverContent>
      </Popover>

      {/* Contrast */}
      <Popover>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            title={`Contrast: ${contrast}%`}
          >
            <Contrast className="h-4 w-4" />
          </Button>
        </PopoverTrigger>
        <PopoverContent side="bottom" className="w-48 p-3">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium">Contrast</span>
              <div className="flex items-center gap-1">
                <span className="text-xs text-muted-foreground tabular-nums">
                  {contrast}%
                </span>
                {contrast !== 50 && (
                  <button
                    onClick={() => onContrastChange(50)}
                    className="text-xs text-muted-foreground hover:text-foreground"
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
    </>
  );
}
