/**
 * Observations sort/display settings popover.
 *
 * Renders its own toolbar icon trigger (Settings2) so it sits inline
 * with the other utility icons in the verify toolbar. Hosts the
 * label-divider toggle and tile-size segmented control.
 *
 * Label dividers only group adjacent same-label tiles in similarity
 * mode, so the toggle is auto-disabled for non-similarity sorts. The
 * caller decides which sort is active via the `similaritySort` flag.
 */

import { Settings2 } from "lucide-react";

import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { Switch } from "../ui/switch";
import { cn } from "../../lib/utils";
import type { TileSize } from "./CropGrid";

interface ObservationsSettingsProps {
  showLabelDividers: boolean;
  onShowLabelDividersChange: (v: boolean) => void;
  tileSize: TileSize;
  onTileSizeChange: (v: TileSize) => void;
  /** When false, label dividers don't group anything — toggle is disabled. */
  similaritySort: boolean;
}

const TILE_SIZES: TileSize[] = ["S", "M", "L"];

export function ObservationsSettings({
  showLabelDividers,
  onShowLabelDividersChange,
  tileSize,
  onTileSizeChange,
  similaritySort,
}: ObservationsSettingsProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          title="View options"
          aria-label="View options"
          className="text-muted-foreground hover:text-foreground transition-colors"
        >
          <Settings2 className="h-4 w-4" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-72 space-y-4">
        <p className="text-sm font-medium">View options</p>

        <label
          className={cn(
            "flex items-center justify-between gap-2",
            !similaritySort && "opacity-60",
          )}
        >
          <div>
            <p className="text-sm">Label dividers</p>
            <p className="text-xs text-muted-foreground">
              {similaritySort
                ? "Show headers between groups of the same label"
                : "Only available when sorting by similarity"}
            </p>
          </div>
          <Switch
            checked={showLabelDividers && similaritySort}
            disabled={!similaritySort}
            onCheckedChange={onShowLabelDividersChange}
          />
        </label>

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
                    : "text-muted-foreground hover:text-foreground",
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
