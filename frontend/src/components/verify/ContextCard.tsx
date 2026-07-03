/**
 * ContextCard — one of the two "context" panels in the detection detail
 * modal: Event context (the event's other frames) and Similarity context
 * (the crop's look-alike neighbours). They are the two ways of seeing a
 * crop in context, mirroring the "By event" and "Similarity" sorts, so
 * they share one shape: an optional meter/header, a grid of thumbnails
 * that enlarge on hover, and a caption.
 *
 * The enlarged preview is anchored to the whole grid (not to each tile)
 * and pinned to its left, so it always appears in the same spot over the
 * image area and never flips back over the thumbnail you're hovering.
 */
import { useState } from "react";
import type { ReactNode } from "react";

import { cn } from "../../lib/utils";
import { Popover, PopoverAnchor, PopoverContent } from "../ui/popover";

export interface ContextItem {
  key: string;
  /** Small tile content; fills the aspect-[4/3] box. */
  tile: ReactNode;
  /** Enlarged content shown on hover. */
  preview: ReactNode;
  /** Border class for the tile (e.g. highlight the current / agreeing crop). */
  borderClassName?: string;
}

// Static classes so Tailwind keeps them; a template string would be purged.
const GRID_COLS: Record<number, string> = {
  4: "grid-cols-4",
  5: "grid-cols-5",
};

interface ContextCardProps {
  title: string;
  caption: string;
  items: ContextItem[];
  columns: 4 | 5;
  /** Optional content above the grid (e.g. the agreement meter). Shown
   *  even before the thumbnails load, so an immediate signal like the
   *  meter isn't held back by the neighbour query. */
  header?: ReactNode;
}

export function ContextCard({
  title,
  caption,
  items,
  columns,
  header,
}: ContextCardProps) {
  // The tile currently hovered; drives the single shared preview. Cleared
  // only when the mouse leaves the whole grid, so moving tile-to-tile
  // swaps the preview content without flicker.
  const [hovered, setHovered] = useState<ContextItem | null>(null);

  if (items.length === 0 && !header) return null;

  return (
    <Popover
      open={hovered != null}
      onOpenChange={(o) => {
        if (!o) setHovered(null);
      }}
    >
      <div className="mx-3 mt-3 rounded-lg border bg-muted/40">
        <h3 className="px-3 pt-3 pb-1 text-sm font-semibold">{title}</h3>
        <p className="px-3 pb-2 text-xs text-muted-foreground">{caption}</p>
        <div className="px-3 pb-3 space-y-2">
          {items.length > 0 && (
            <PopoverAnchor asChild>
              <div
                className={cn("grid gap-1.5", GRID_COLS[columns])}
                onMouseLeave={() => setHovered(null)}
              >
                {items.map((item) => (
                  <div
                    key={item.key}
                    onMouseEnter={() => setHovered(item)}
                    className={cn(
                      "relative aspect-[4/3] overflow-hidden rounded border-2",
                      item.borderClassName ?? "border-transparent",
                    )}
                  >
                    {item.tile}
                  </div>
                ))}
              </div>
            </PopoverAnchor>
          )}
          {header}
        </div>
      </div>
      <PopoverContent
        side="left"
        align="center"
        sideOffset={12}
        collisionPadding={16}
        onOpenAutoFocus={(e) => e.preventDefault()}
        className="pointer-events-none w-[32rem] max-w-[45vw] p-1.5"
      >
        {hovered?.preview}
      </PopoverContent>
    </Popover>
  );
}
