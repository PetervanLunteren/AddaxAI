/**
 * Labels view-options popover.
 *
 * Renders its own toolbar icon trigger (LayoutGrid) so it sits inline
 * with the other utility icons in the verify toolbar. Hosts the
 * tile-size segmented control. The value is persisted to localStorage by
 * the parent (see LabelsTab's persistSetting helper).
 */

import { LayoutGrid } from "lucide-react";

import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { VERIFY_TOOLBAR_ICON_CLASS } from "./VerifyToolbar";
import type { TileSize } from "./CropGrid";
import { TileSizeToggle } from "./TileSizeToggle";

interface LabelsSettingsProps {
  tileSize: TileSize;
  onTileSizeChange: (v: TileSize) => void;
}

export function LabelsSettings({
  tileSize,
  onTileSizeChange,
}: LabelsSettingsProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          title="View options"
          aria-label="View options"
          className={VERIFY_TOOLBAR_ICON_CLASS}
        >
          <LayoutGrid className="h-4 w-4" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-72">
        <div className="space-y-1.5">
          <p className="text-sm">Tile size</p>
          <TileSizeToggle value={tileSize} onChange={onTileSizeChange} />
        </div>
      </PopoverContent>
    </Popover>
  );
}
