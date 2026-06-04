/**
 * CropGrid - virtualized grid of detection crop thumbnails.
 *
 * Uses @tanstack/react-virtual for efficient rendering of large detection sets.
 * Responsive columns: 4 (sm), 6 (md), 8 (lg), 10 (xl).
 * Optional divider rows: `label` groups by current label, `cohort`
 * groups by `(current_label, neighbor_top_label, category)` and
 * surfaces a "Relabel all (N)" button so the suggestions sort mode can
 * promote a whole cohort in one click.
 */

import { memo, useRef, useMemo, useEffect, useLayoutEffect, useState, useSyncExternalStore } from "react";
import { useWindowVirtualizer } from "@tanstack/react-virtual";
import { Button } from "../ui/button";
import { CropCard } from "./CropCard";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import { cn } from "../../lib/utils";
import type { CohortItem, DetectionSummary } from "../../api/types";

/**
 * Lightweight pub/sub store that lets individual GridCells subscribe to
 * their own selection state without re-rendering the entire CropGrid.
 */
/**
 * Lightweight pub/sub store that lets individual GridCells subscribe to
 * their own selection state without re-rendering the entire CropGrid.
 */
class SelectionStore {
  ids: Set<string> = new Set();
  private listeners = new Set<() => void>();

  getSnapshot = () => this.ids;

  subscribe = (cb: () => void) => {
    this.listeners.add(cb);
    return () => { this.listeners.delete(cb); };
  };

  /** Update data and notify subscribers. Safe to call from useLayoutEffect. */
  update(next: Set<string>) {
    if (next === this.ids) return;
    this.ids = next;
    for (const cb of this.listeners) cb();
  }
}

export type TileSize = "S" | "M" | "L";

export type GridDividerMode = "none" | "label" | "cohort";

type CohortRowPos = "first" | "middle" | "last" | "only";

type GridRow =
  | {
      type: "cards";
      detections: DetectionSummary[];
      /** Position within a cohort card when `dividers === "cohort"`.
       * Drives the row's card-border styling (sides only mid-card,
       * adds the bottom-rounded border on the last row). Undefined
       * for non-cohort modes. */
      cohortPos?: CohortRowPos;
    }
  | { type: "divider"; label: string; count: number }
  | { type: "cohort_divider"; cohort: CohortItem }
  | { type: "cohort_gap" };

interface CropGridProps {
  detections: DetectionSummary[];
  selectedIds: Set<string>;
  onSelect: (detectionId: string, e: React.MouseEvent) => void;
  onDoubleClick?: (detection: DetectionSummary) => void;
  onBackgroundClick?: () => void;
  /** Fires when the user clicks "Relabel all (N)" on a cohort divider.
   * Parent owns the destructive confirm flow and the bulk-relabel
   * mutation; the divider is presentational. */
  onRelabelCohort?: (cohort: CohortItem) => void;
  tileSize?: TileSize;
  dividers?: GridDividerMode;
}

const COLUMN_PRESETS: Record<TileSize, [number, number, number, number]> = {
  S: [6, 8, 12, 14],
  M: [3, 5, 7, 9],
  L: [2, 3, 3, 4],
};

const ESTIMATE_SIZE: Record<TileSize, number> = {
  S: 140,
  M: 180,
  L: 380,
};

const DIVIDER_HEIGHT = 32;
// Cohort dividers carry a header line + chips + a Relabel-all button so
// they need more vertical real estate than a plain label divider. The
// header uses `text-base` and symmetric vertical padding (`py-3`) so
// the button isn't crowded against the top border.
const COHORT_DIVIDER_HEIGHT = 84;
// Breathing room between cohort cards.
const COHORT_GAP_HEIGHT = 16;

function useColumns(tileSize: TileSize = "M"): number {
  const [cols, setCols] = useState(COLUMN_PRESETS[tileSize][1]);
  useEffect(() => {
    const preset = COLUMN_PRESETS[tileSize];
    function update() {
      const w = window.innerWidth;
      if (w < 640) setCols(preset[0]);
      else if (w < 1024) setCols(preset[1]);
      else if (w < 1280) setCols(preset[2]);
      else setCols(preset[3]);
    }
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [tileSize]);
  return cols;
}

const GAP_CLASS: Record<TileSize, string> = {
  S: "gap-2 pb-2",
  M: "gap-3 pb-3",
  L: "gap-4 pb-4",
};

interface GridCellProps {
  detection: DetectionSummary;
  selectionStore: SelectionStore;
  tileSize: TileSize;
  onSelect: (detectionId: string, e: React.MouseEvent) => void;
  onDoubleClick?: (detection: DetectionSummary) => void;
}

const GridCell = memo(function GridCell({
  detection,
  selectionStore,
  tileSize,
  onSelect,
  onDoubleClick,
}: GridCellProps) {
  const selected = useSyncExternalStore(
    selectionStore.subscribe,
    () => selectionStore.getSnapshot().has(detection.detection_id),
  );

  return (
    <div data-crop-card>
      <CropCard
        detection={detection}
        selected={selected}
        tileSize={tileSize}
        onSelect={onSelect}
        onDoubleClick={onDoubleClick}
      />
    </div>
  );
});

export function CropGrid({
  detections,
  selectedIds,
  onSelect,
  onDoubleClick,
  onBackgroundClick,
  onRelabelCohort,
  tileSize = "M",
  dividers = "none",
}: CropGridProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const columns = useColumns(tileSize);

  // Selection store — individual GridCells subscribe to their own selection
  // state via useSyncExternalStore, avoiding full grid re-renders.
  const [selectionStore] = useState(() => new SelectionStore());
  // useLayoutEffect fires synchronously after DOM mutation but before paint,
  // so cells see updated selection before the browser paints the frame.
  useLayoutEffect(() => { selectionStore.update(selectedIds); }, [selectedIds, selectionStore]);

  // Build rows: optionally insert divider rows between groups.
  const rows = useMemo((): GridRow[] => {
    if (dividers === "none") {
      const result: GridRow[] = [];
      for (let i = 0; i < detections.length; i += columns) {
        result.push({ type: "cards", detections: detections.slice(i, i + columns) });
      }
      return result;
    }

    const result: GridRow[] = [];
    let i = 0;
    while (i < detections.length) {
      // Group key changes with dividers mode: label groups by current
      // label / category, cohort groups by the (label, suggested
      // descendant, category) triple that defines a cohort.
      const groupKey = dividers === "cohort" ? cohortKey(detections[i]) : labelKey(detections[i]);
      let j = i;
      while (
        j < detections.length &&
        (dividers === "cohort" ? cohortKey(detections[j]) : labelKey(detections[j])) === groupKey
      ) {
        j++;
      }
      const slice = detections.slice(i, j);
      if (dividers === "cohort") {
        result.push({ type: "cohort_divider", cohort: cohortFromSlice(slice) });
        // Tag each card row inside the cohort with its position so the
        // renderer can paint card-side borders on every row and round
        // the bottom of the final row only.
        const cardRowStarts: number[] = [];
        for (let k = 0; k < slice.length; k += columns) cardRowStarts.push(k);
        for (let idx = 0; idx < cardRowStarts.length; idx++) {
          const k = cardRowStarts[idx];
          const isLast = idx === cardRowStarts.length - 1;
          const pos: CohortRowPos =
            cardRowStarts.length === 1 ? "only" : isLast ? "last" : idx === 0 ? "first" : "middle";
          result.push({
            type: "cards",
            detections: slice.slice(k, k + columns),
            cohortPos: pos,
          });
        }
        // Spacer between this cohort and the next so cards don't run
        // together vertically.
        if (j < detections.length) {
          result.push({ type: "cohort_gap" });
        }
      } else {
        result.push({ type: "divider", label: labelKey(slice[0]), count: slice.length });
        for (let k = 0; k < slice.length; k += columns) {
          result.push({ type: "cards", detections: slice.slice(k, k + columns) });
        }
      }
      i = j;
    }
    return result;
  }, [detections, columns, dividers]);

  const cardHeight = ESTIMATE_SIZE[tileSize];

  const virtualizer = useWindowVirtualizer({
    count: rows.length,
    estimateSize: (index) => {
      const row = rows[index];
      if (row.type === "divider") return DIVIDER_HEIGHT;
      if (row.type === "cohort_divider") return COHORT_DIVIDER_HEIGHT;
      if (row.type === "cohort_gap") return COHORT_GAP_HEIGHT;
      return cardHeight;
    },
    overscan: 5,
    scrollMargin: listRef.current?.offsetTop ?? 0,
    measureElement: (el) => el.getBoundingClientRect().height,
  });

  return (
    <div
      ref={listRef}
      style={{
        height: `${virtualizer.getTotalSize()}px`,
        width: "100%",
        position: "relative",
      }}
      onClick={(e) => {
        if (
          onBackgroundClick &&
          !(e.target as HTMLElement).closest("[data-crop-card]")
        ) {
          onBackgroundClick();
        }
      }}
    >
      {virtualizer.getVirtualItems().map((virtualRow) => {
        const row = rows[virtualRow.index];

        if (row.type === "divider") {
          return (
            <div
              key={`divider-${virtualRow.index}`}
              data-index={virtualRow.index}
              ref={virtualizer.measureElement}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                transform: `translateY(${virtualRow.start - virtualizer.options.scrollMargin}px)`,
              }}
            >
              <div className="flex items-center gap-2 px-1 h-full">
                <div className="h-px flex-1 bg-border" />
                <span className="text-xs text-muted-foreground font-medium capitalize whitespace-nowrap">
                  {row.label} ({row.count})
                </span>
                <div className="h-px flex-1 bg-border" />
              </div>
            </div>
          );
        }

        if (row.type === "cohort_divider") {
          const c = row.cohort;
          return (
            <div
              key={`cohort-${virtualRow.index}`}
              data-index={virtualRow.index}
              ref={virtualizer.measureElement}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                transform: `translateY(${virtualRow.start - virtualizer.options.scrollMargin}px)`,
              }}
            >
              {/* Top of the cohort card: borders on three sides + rounded
                  top corners, tinted bg so it stands out from the page,
                  a bottom border under the header to mark the seam
                  between the title and the crop grid. Symmetric `py-3`
                  keeps the button away from the top + bottom borders.
                  The card-side borders continue on every `cards` row
                  below until the cohort's final row, which adds the
                  rounded bottom. */}
              <div className="flex items-center justify-between gap-3 px-4 py-3 h-full rounded-t-lg border-x border-t border-b bg-card">
                <div className="text-base flex flex-wrap items-center gap-2">
                  <span className="font-semibold">{c.count}</span>
                  <span className="text-muted-foreground">
                    observation{c.count === 1 ? "" : "s"} labelled
                  </span>
                  <LabelChip
                    label={c.current_label}
                    displayName={c.current_scientific_name}
                  />
                  <span className="text-muted-foreground">look like</span>
                  <LabelChip
                    label={c.suggested_label}
                    displayName={c.suggested_scientific_name}
                  />
                </div>
                {onRelabelCohort && (
                  <TooltipProvider delayDuration={300}>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          type="button"
                          size="sm"
                          onClick={() => onRelabelCohort(c)}
                        >
                          Accept suggestion ({c.count})
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        Relabels {c.count} observation
                        {c.count === 1 ? "" : "s"} to{" "}
                        {c.suggested_scientific_name || c.suggested_label} and
                        marks {c.count === 1 ? "it" : "them"} verified.
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                )}
              </div>
            </div>
          );
        }

        if (row.type === "cohort_gap") {
          // Empty spacer between cohort cards. No content, just height
          // — the virtualizer reserves the space and the visual break
          // lets each cohort read as a distinct card.
          return (
            <div
              key={`gap-${virtualRow.index}`}
              data-index={virtualRow.index}
              ref={virtualizer.measureElement}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: `${COHORT_GAP_HEIGHT}px`,
                transform: `translateY(${virtualRow.start - virtualizer.options.scrollMargin}px)`,
              }}
            />
          );
        }

        // cohortPos is set only when dividers === "cohort". It paints
        // the card-side borders on every row of a cohort and rounds
        // the bottom of the last (or only) row.
        const inCohort = row.cohortPos !== undefined;
        const isLast = row.cohortPos === "last" || row.cohortPos === "only";
        return (
          <div
            key={virtualRow.index}
            data-index={virtualRow.index}
            ref={virtualizer.measureElement}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              transform: `translateY(${virtualRow.start - virtualizer.options.scrollMargin}px)`,
            }}
          >
              <div
                className={cn(
                  `grid ${GAP_CLASS[tileSize]}`,
                  inCohort
                    ? cn(
                        "px-4 border-x bg-card",
                        // First row of a cohort body: breathing room
                        // under the header's border-b before the crops
                        // start.
                        (row.cohortPos === "first" || row.cohortPos === "only") && "pt-2",
                        isLast && "border-b rounded-b-lg pb-3",
                      )
                    : "px-1",
                )}
                style={{
                  gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                }}
              >
                {row.detections.map((det) => (
                  <GridCell
                    key={det.detection_id}
                    detection={det}
                    selectionStore={selectionStore}
                    tileSize={tileSize}
                    onSelect={onSelect}
                    onDoubleClick={onDoubleClick}
                  />
                ))}
              </div>
            </div>
          );
        })}
    </div>
  );
}


// ── Grouping helpers (used by the dividers union above) ─────────────────


function labelKey(d: DetectionSummary): string {
  return d.label || d.category;
}


function cohortKey(d: DetectionSummary): string {
  // Three parts on a separator unlikely to appear in a label.
  return `${d.label ?? ""}${d.neighbor_top_label ?? ""}${d.category ?? ""}`;
}


/** Inline label token rendered inside the cohort divider header.
 * Deliberately uncoloured: the header is an explanatory sentence
 * ("46 observations labelled X look like Y"), so the labels read as
 * plain values, not as colour-coded data. Colour lives in the crop
 * grid below, where it carries meaning. A muted code-style chip keeps
 * the two from competing and avoids fragile colour-key matching. */
function LabelChip({
  label,
  displayName,
}: {
  label: string | null;
  displayName: string | null;
}) {
  const text = displayName || label || "(no label)";
  return (
    <span className="rounded-sm bg-muted px-1.5 py-0.5 font-mono text-xs capitalize text-foreground">
      {text}
    </span>
  );
}


function cohortFromSlice(slice: DetectionSummary[]): CohortItem {
  // Every detection in the slice already shares the cohort key
  // (current label, suggested label, category) by construction, so the
  // first one is authoritative for the divider's display fields.
  const head = slice[0];
  return {
    current_label: head.label,
    current_label_taxonomy_id: head.label_taxonomy_id ?? null,
    current_common_name: head.common_name ?? null,
    current_scientific_name: head.scientific_name ?? null,
    suggested_label: head.neighbor_top_label ?? "",
    suggested_common_name: head.neighbor_top_common_name ?? null,
    suggested_scientific_name: head.neighbor_top_scientific_name ?? null,
    category: head.category,
    count: slice.length,
    detection_ids: slice.map((d) => d.detection_id),
  };
}
