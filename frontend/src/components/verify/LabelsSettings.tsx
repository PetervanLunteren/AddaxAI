/**
 * Labels sort/display settings popover.
 *
 * Renders its own toolbar icon trigger (Settings2) so it sits inline
 * with the other utility icons in the verify toolbar. Hosts the
 * tile-size segmented control and the per-user max-detections cap for
 * similarity sort.
 *
 * All values are persisted to localStorage by the parent (see
 * LabelsTab's persistSetting helper). The max-detections cap
 * used to live on the project DB row; it moved here because it's a
 * per-user memory budget that benefits from being one click away
 * from the feature it controls.
 */

import { Settings2 } from "lucide-react";

import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { VERIFY_TOOLBAR_ICON_CLASS } from "./VerifyToolbar";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import type { TileSize } from "./CropGrid";
import { TileSizeToggle } from "./TileSizeToggle";
import { LABELS_MAX_DETECTIONS_OPTIONS } from "./labelsViewOptions";

interface LabelsSettingsProps {
  tileSize: TileSize;
  onTileSizeChange: (v: TileSize) => void;
  maxDetections: number;
  onMaxDetectionsChange: (v: number) => void;
}

export function LabelsSettings({
  tileSize,
  onTileSizeChange,
  maxDetections,
  onMaxDetectionsChange,
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
          <Settings2 className="h-4 w-4" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-72 space-y-4">
        <p className="text-sm font-medium">View options</p>

        <div className="space-y-1.5">
          <p className="text-sm">Tile size</p>
          <TileSizeToggle value={tileSize} onChange={onTileSizeChange} />
        </div>

        <div className="space-y-1.5">
          <p className="text-sm">Max labels per sort</p>
          <p className="text-xs text-muted-foreground">
            The most labels to load in one sort. A higher limit uses more
            memory and is slower, so narrowing the filters first is usually
            easier and faster.
          </p>
          <Select
            value={String(maxDetections)}
            onValueChange={(v) => onMaxDetectionsChange(parseInt(v, 10))}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {LABELS_MAX_DETECTIONS_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={String(opt.value)}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </PopoverContent>
    </Popover>
  );
}
