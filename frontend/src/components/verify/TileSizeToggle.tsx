/**
 * TileSizeToggle - S / M / L segmented control for thumbnail density.
 * Shared by the Labels settings popover and the Counts-modal filmstrip.
 */

import { cn } from "../../lib/utils";
import type { TileSize } from "./CropGrid";

const TILE_SIZES: TileSize[] = ["S", "M", "L"];

interface TileSizeToggleProps {
  value: TileSize;
  onChange: (v: TileSize) => void;
  className?: string;
}

export function TileSizeToggle({
  value,
  onChange,
  className,
}: TileSizeToggleProps) {
  return (
    <div className={cn("flex rounded-lg bg-muted p-0.5", className)}>
      {TILE_SIZES.map((size) => (
        <button
          key={size}
          type="button"
          className={cn(
            "flex-1 px-3 py-1 text-xs font-medium rounded-md transition-colors",
            value === size
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground",
          )}
          onClick={() => onChange(size)}
        >
          {size}
        </button>
      ))}
    </div>
  );
}
