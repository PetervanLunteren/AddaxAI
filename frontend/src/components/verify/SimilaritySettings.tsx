/**
 * SimilaritySettings — popover with sort/display settings for the Similarity tab.
 *
 * Contains toggles for noise-first sorting, auto-hide verified, species dividers,
 * and tile size selection.
 */

import { Settings2 } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { Switch } from "../ui/switch";
import { cn } from "../../lib/utils";
import type { TileSize } from "./CropGrid";

interface SimilaritySettingsProps {
  reverseSort: boolean;
  onReverseSortChange: (v: boolean) => void;
  autoHideVerified: boolean;
  onAutoHideVerifiedChange: (v: boolean) => void;
  showSpeciesDividers: boolean;
  onShowSpeciesDividersChange: (v: boolean) => void;
  tileSize: TileSize;
  onTileSizeChange: (v: TileSize) => void;
}

const TILE_SIZES: TileSize[] = ["S", "M", "L"];

export function SimilaritySettings({
  reverseSort,
  onReverseSortChange,
  autoHideVerified,
  onAutoHideVerifiedChange,
  showSpeciesDividers,
  onShowSpeciesDividersChange,
  tileSize,
  onTileSizeChange,
}: SimilaritySettingsProps) {
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

        {/* Hide as I verify */}
        <label className="flex items-center justify-between gap-2">
          <div>
            <p className="text-sm">Hide as I verify</p>
            <p className="text-xs text-muted-foreground">Fade out items as you verify them</p>
          </div>
          <Switch checked={autoHideVerified} onCheckedChange={onAutoHideVerifiedChange} />
        </label>

        {/* Species dividers */}
        <label className="flex items-center justify-between gap-2">
          <div>
            <p className="text-sm">Species dividers</p>
            <p className="text-xs text-muted-foreground">Show headers between species groups</p>
          </div>
          <Switch checked={showSpeciesDividers} onCheckedChange={onShowSpeciesDividersChange} />
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
