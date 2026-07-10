/**
 * CropGrid - virtualized grid of detection crop thumbnails.
 *
 * Uses @tanstack/react-virtual for efficient rendering of large detection sets.
 * Responsive columns: 4 (sm), 6 (md), 8 (lg), 10 (xl).
 * Optional divider rows: `cohort` groups by
 * `(current_label, neighbor_top_label, category)` and surfaces a
 * "Relabel all (N)" button for the suggestions sort; `event` groups by
 * event and surfaces a "Select" link to select that event's crops.
 */

import { forwardRef, memo, useImperativeHandle, useRef, useMemo, useEffect, useLayoutEffect, useState, useSyncExternalStore } from "react";
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
import { columnsForWidth, useWideModeValue } from "./wide-mode";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
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

export type GridDividerMode = "none" | "cohort" | "event";

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
  | { type: "divider"; label: string; count: number; detectionIds: string[] }
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
  /** Fires when the user clicks "Dismiss (N)" on a cohort divider.
   * Hides the suggestion without touching the crops; parent owns the
   * mutation and the Undo toast. The divider is presentational. */
  onDismissCohort?: (cohort: CohortItem) => void;
  /** Fires when the user clicks "Select" on an event divider. Selects
   *  that event's in-view crops so the existing bulk actions apply. */
  onSelectEvent?: (detectionIds: string[]) => void;
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

// Event divider height estimate (real height is measured). The header
// below uses asymmetric padding (pt-1 pb-4): the crop row above already
// carries its own bottom padding (GAP_CLASS `pb-*`), so a smaller top +
// larger bottom padding lands the label visually centered in the gap
// between the two events rather than hugging one side.
const DIVIDER_HEIGHT = 44;
// Cohort dividers carry a header line + chips + a Relabel-all button so
// they need more vertical real estate than a plain label divider. The
// header uses `text-base` and symmetric vertical padding (`py-3`) so
// the button isn't crowded against the top border.
const COHORT_DIVIDER_HEIGHT = 84;
// Breathing room between cohort cards.
const COHORT_GAP_HEIGHT = 16;

// Minimum tile width (px) per size in wide mode, tuned so the measured
// container yields the same column counts as the viewport presets at the
// normal capped width (~1216px content), then adds columns as it widens.
const MIN_TILE: Record<TileSize, number> = { S: 80, M: 124, L: 290 };
// Inter-tile gap (px) matching GAP_CLASS, needed for the fit math.
const GAP_PX: Record<TileSize, number> = { S: 8, M: 12, L: 16 };

/**
 * Column count for the crop grid.
 *
 * Normal mode: viewport breakpoints (unchanged, so the default view is
 * byte-identical to before). Wide mode: measured from the grid
 * container's width so removing the page width cap (or widening the
 * window) adds columns instead of just enlarging crops.
 */
function useColumns(
  tileSize: TileSize,
  containerRef: React.RefObject<HTMLDivElement | null>,
  fullWidth: boolean,
): number {
  const preset = COLUMN_PRESETS[tileSize];
  const [cols, setCols] = useState(preset[1]);
  useEffect(() => {
    function fromViewport(): number {
      const w = window.innerWidth;
      if (w < 640) return preset[0];
      if (w < 1024) return preset[1];
      if (w < 1280) return preset[2];
      return preset[3];
    }
    function update() {
      if (!fullWidth) {
        setCols(fromViewport());
        return;
      }
      const el = containerRef.current;
      const w = el ? el.clientWidth : window.innerWidth;
      setCols(
        Math.max(
          preset[0],
          columnsForWidth(w, MIN_TILE[tileSize], GAP_PX[tileSize]),
        ),
      );
    }
    update();
    window.addEventListener("resize", update);
    let ro: ResizeObserver | undefined;
    if (fullWidth && containerRef.current) {
      ro = new ResizeObserver(update);
      ro.observe(containerRef.current);
    }
    return () => {
      window.removeEventListener("resize", update);
      ro?.disconnect();
    };
  }, [tileSize, fullWidth, containerRef, preset]);
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

export interface CropGridHandle {
  /** Scroll the given detection's row into view (jumping to the event's
   *  divider header when it has one). `align` "start" pins it to the top
   *  (the "E" event jump); "auto" scrolls the minimum to make it visible
   *  and is a no-op when it already is (the post-action advance, so the
   *  grid doesn't lurch on every keypress). Defaults to "start". */
  scrollToDetection: (detectionId: string, align?: "start" | "auto") => void;
}

export const CropGrid = forwardRef<CropGridHandle, CropGridProps>(
  function CropGrid({
  detections,
  selectedIds,
  onSelect,
  onDoubleClick,
  onBackgroundClick,
  onRelabelCohort,
  onDismissCohort,
  onSelectEvent,
  tileSize = "M",
  dividers = "none",
}: CropGridProps, ref) {
  const listRef = useRef<HTMLDivElement>(null);
  const fullWidth = useWideModeValue();
  const columns = useColumns(tileSize, listRef, fullWidth);

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

    // Group key by dividers mode: cohort groups by the (label, suggested
    // descendant, category) triple; event groups by the detection's
    // event. ("none" is handled above.)
    const keyOf = (d: DetectionSummary) =>
      dividers === "cohort" ? cohortKey(d) : eventKey(d);

    const result: GridRow[] = [];
    let i = 0;
    while (i < detections.length) {
      const groupKey = keyOf(detections[i]);
      let j = i;
      while (j < detections.length && keyOf(detections[j]) === groupKey) {
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
        // Only event dividers remain in this branch (label dividers were
        // removed; cohort is handled above).
        result.push({
          type: "divider",
          label: eventDividerLabel(slice[0]),
          count: slice.length,
          detectionIds: slice.map((d) => d.detection_id),
        });
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

  useImperativeHandle(
    ref,
    () => ({
      scrollToDetection(detectionId: string, align: "start" | "auto" = "start") {
        const idx = rows.findIndex(
          (r) =>
            r.type === "cards" &&
            r.detections.some((d) => d.detection_id === detectionId),
        );
        if (idx < 0) return;
        // Land on the event's divider header when there is one, so the
        // reader gets the event label plus its first crops in view.
        const target =
          idx > 0 && rows[idx - 1].type === "divider" ? idx - 1 : idx;
        virtualizer.scrollToIndex(target, { align });
      },
    }),
    [rows, virtualizer],
  );

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
              <div className="flex items-center gap-2 px-1 pt-1 pb-4">
                <span className="text-xs text-muted-foreground font-medium capitalize whitespace-nowrap">
                  {row.label} ({row.count})
                </span>
                <div className="h-px flex-1 bg-border" />
                {onSelectEvent && row.detectionIds.length > 0 && (
                  <button
                    type="button"
                    onClick={(e) => {
                      // Don't bubble to the grid's background handler,
                      // which would clear the selection we just set.
                      e.stopPropagation();
                      onSelectEvent(row.detectionIds);
                    }}
                    className="text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline whitespace-nowrap"
                  >
                    Select
                  </button>
                )}
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
                    displayName={resolveSpeciesName({
                      common_name: c.current_common_name,
                      scientific_name: c.current_scientific_name,
                      label: c.current_label,
                    })}
                  />
                  <span className="text-muted-foreground">look like</span>
                  <LabelChip
                    label={c.suggested_label}
                    displayName={resolveSpeciesName({
                      common_name: c.suggested_common_name,
                      scientific_name: c.suggested_scientific_name,
                      label: c.suggested_label,
                    })}
                  />
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  {onDismissCohort && (
                    <TooltipProvider delayDuration={300}>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            onClick={() => onDismissCohort(c)}
                          >
                            Dismiss ({c.count})
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                          Hides this suggestion. The {c.count} observation
                          {c.count === 1 ? "" : "s"} keep their current label
                          and stay unverified, so you can relabel{" "}
                          {c.count === 1 ? "it" : "them"} in the normal sort.
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  )}
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
                          {resolveSpeciesName({
                            common_name: c.suggested_common_name,
                            scientific_name: c.suggested_scientific_name,
                            label: c.suggested_label,
                          })}{" "}
                          and
                          marks {c.count === 1 ? "it" : "them"} verified.
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  )}
                </div>
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
});


// ── Grouping helpers (used by the dividers union above) ─────────────────


function eventKey(d: DetectionSummary): string {
  // Detections without an event (clustering not run) all fall into one
  // trailing group; the "By event" sort places them last.
  return d.event_id ?? "__no_event__";
}


/** Header text for an event divider: the event start (camera-local) and
 *  site, matching how event cards label themselves on the Counts page. */
function eventDividerLabel(d: DetectionSummary): string {
  if (!d.event_start_local) return d.site_name || "No event";
  const date = formatCameraDate(d.event_start_local, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
  const time = formatCameraTime(d.event_start_local, {
    hour: "2-digit",
    minute: "2-digit",
  });
  return d.site_name ? `${date} · ${time} · ${d.site_name}` : `${date} · ${time}`;
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
