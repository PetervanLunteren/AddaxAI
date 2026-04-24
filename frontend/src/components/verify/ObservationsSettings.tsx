/**
 * ObservationsSettings — popover with sort/display settings for the
 * Observations verify tab.
 *
 * Contains toggles for noise-first sorting, label dividers,
 * and tile size selection.
 */

import { Settings2 } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { Switch } from "../ui/switch";
import { cn } from "../../lib/utils";
import type { TileSize } from "./CropGrid";

interface ObservationsSettingsProps {
  reverseSort: boolean;
  onReverseSortChange: (v: boolean) => void;
  showLabelDividers: boolean;
  onShowLabelDividersChange: (v: boolean) => void;
  tileSize: TileSize;
  onTileSizeChange: (v: TileSize) => void;
}

const TILE_SIZES: TileSize[] = ["S", "M", "L"];

export function ObservationsSettings({
  reverseSort,
  onReverseSortChange,
  showLabelDividers,
  onShowLabelDividersChange,
  tileSize,
  onTileSizeChange,
}: ObservationsSettingsProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          className="text-muted-foreground hover:text-foreground transition-colors"
          title="Settings"
        >
          <Settings2 className="h-4 w-4" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-64 space-y-4">
        <p className="text-sm font-medium">Settings</p>

        {/* Noise first */}
        <label className="flex items-center justify-between gap-2">
          <div>
            <p className="text-sm">Noise first</p>
            <p className="text-xs text-muted-foreground">Show outliers at top</p>
          </div>
          <Switch checked={reverseSort} onCheckedChange={onReverseSortChange} />
        </label>

        {/* Label dividers */}
        <label className="flex items-center justify-between gap-2">
          <div>
            <p className="text-sm">Label dividers</p>
            <p className="text-xs text-muted-foreground">Show headers between label groups</p>
          </div>
          <Switch checked={showLabelDividers} onCheckedChange={onShowLabelDividersChange} />
        </label>

        {/* Tile size */}
        <div className="space-y-1.5">
          <p className="text-sm">Tile size</p>
          <div className="flex rounded-lg bg-muted p-0.5">
            {TILE_SIZES.map((size) => (
              <button
                key={size}
                className={cn(
                  "flex-1 px-3 py-1 text-xs font-medium rounded-md transition-colors",
                  tileSize === size
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
                onClick={() => onTileSizeChange(size)}
              >
                {size}
              </button>
            ))}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
