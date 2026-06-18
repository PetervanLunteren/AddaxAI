/**
 * Labels sort/display settings popover.
 *
 * Renders its own toolbar icon trigger (Settings2) so it sits inline
 * with the other utility icons in the verify toolbar. Hosts the
 * label-divider toggle, tile-size segmented control, and the per-user
 * max-detections cap for similarity sort.
 *
 * All values are persisted to localStorage by the parent (see
 * LabelsTab's persistSetting helper). The max-detections cap
 * used to live on the project DB row; it moved here because it's a
 * per-user memory budget that benefits from being one click away
 * from the feature it controls.
 *
 * Label dividers only group adjacent same-label tiles in similarity
 * mode, so the toggle is auto-disabled for non-similarity sorts. The
 * caller decides which sort is active via the `similaritySort` flag.
 */

import { Settings2 } from "lucide-react";

import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Switch } from "../ui/switch";
import { cn } from "../../lib/utils";
import type { TileSize } from "./CropGrid";
import { TileSizeToggle } from "./TileSizeToggle";
import { LABELS_MAX_DETECTIONS_OPTIONS } from "./labelsViewOptions";

interface LabelsSettingsProps {
  showLabelDividers: boolean;
  onShowLabelDividersChange: (v: boolean) => void;
  tileSize: TileSize;
  onTileSizeChange: (v: TileSize) => void;
  maxDetections: number;
  onMaxDetectionsChange: (v: number) => void;
  /** When false, label dividers don't group anything — toggle is disabled. */
  similaritySort: boolean;
}

export function LabelsSettings({
  showLabelDividers,
  onShowLabelDividersChange,
  tileSize,
  onTileSizeChange,
  maxDetections,
  onMaxDetectionsChange,
  similaritySort,
}: LabelsSettingsProps) {
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
