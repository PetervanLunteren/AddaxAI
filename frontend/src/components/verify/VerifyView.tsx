/**
 * Verify view — the Counts page body, decoupled from page chrome.
 *
 * Keeps the `Verify*` name because it operates on the verification data
 * model. It renders the sequence gallery: a filter bar, a grid of
 * `EventCard`s, and the `EventDetailModal` (gallery + tools on the
 * left, panel on the right). The per-detection label-cleanup view lives
 * separately in `LabelsTab` / the Labels page. Mounted in two contexts:
 *
 * - Research projects: wrapped by `pages/CountsPage.tsx`, which adds the
 *   page chrome (header "Counts").
 * - Folder runs: wrapped by `pages/folder-run/FolderRunCountsStep`, which
 *   mounts it inside the stepper with a sticky Back / Continue bar.
 *
 * Filter URL state lives in `useSearchParams`, path-agnostic so the same
 * wiring works on `/projects/<id>/counts` and the folder-run step.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  FILTER_DEBOUNCE_MS,
  useDebouncedValue,
} from "../../hooks/useDebouncedValue";
import {
  Image as ImageIcon,
  Layers,
  Loader2,
  Maximize2,
  Minimize2,
  SlidersHorizontal,
  Video as VideoIcon,
} from "lucide-react";
import { sequencesApi } from "../../api/sequences";
import { warmFiles } from "./warm-files";
import { projectQueryOptions } from "../../api/projects";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import { getObservationBadge } from "../../lib/detection-utils";
import { speciesLabelMap } from "../../lib/species-name-mode";
import {
  getSpeciesColor,
  getContrastTextColor,
  useSpeciesColorsVersion,
} from "../../utils/species-colors";
import type {
  SequenceSummary,
  SequenceFilterParams,
  VerificationFilter,
  VerifySort,
} from "../../api/types";

import { SequenceCollage } from "./SequenceCollage";
import { SequenceDetailModal } from "./SequenceDetailModal";
import { VerifyFilterBar } from "./VerifyFilterBar";
import { hasAnyActiveFilter } from "./FilterChips";
import { WelcomePopover } from "./WelcomePopover";
import { SortSelector } from "./SortSelector";
import {
  VerifyGuideLink,
  VerifyProgressPill,
  VERIFY_TOOLBAR_ICON_CLASS,
  VerifyToolbar,
  VerifyToolbarIcon,
} from "./VerifyToolbar";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { ImageAdjustRows } from "./ViewControls";
import { StatusBadgeCluster } from "./StatusBadgeCluster";
import { columnsForWidth, useWideModeControls } from "./wide-mode";
import { cn } from "../../lib/utils";

const EVENTS_SORT_MODES = ["newest", "oldest", "random"] as const;

// 48 = LCM(1,2,3,4), so every page lays out cleanly at every grid breakpoint
// (1/2/3/4 columns). Avoids orphan rows on intermediate pages.
const PAGE_SIZE = 48;

/** Parse filter state from URL search params. */
function filtersFromSearchParams(sp: URLSearchParams): SequenceFilterParams {
  const filters: SequenceFilterParams = {};
  const sites = sp.get("sites");
  if (sites) filters.site_ids = sites.split(",");
  const from = sp.get("from");
  if (from) filters.date_from = from;
  const to = sp.get("to");
  if (to) filters.date_to = to;
  const labels = sp.get("labels");
  if (labels) filters.labels = labels.split(",");
  const verification = sp.get("verification") as VerificationFilter | null;
  if (verification && verification !== "all") filters.verification = verification;
  const flagged = sp.get("flagged") as SequenceFilterParams["flagged"] | null;
  if (flagged && flagged !== "all") filters.flagged = flagged;
  const favorited = sp.get("favorited") as SequenceFilterParams["favorited"] | null;
  if (favorited && favorited !== "all") filters.favorited = favorited;
  // Empty defaults to "hide": most users don't want to scroll blank
  // tiles / all-blank sequences, and it's one click to "All". Absent
  // param ⇒ "hide"; an explicit "all" / "show_only" is persisted.
  const empty = sp.get("empty") as SequenceFilterParams["empty"] | null;
  filters.empty = empty ?? "hide";
  const minC = sp.get("min_confidence");
  if (minC !== null) filters.min_confidence = parseFloat(minC);
  const maxC = sp.get("max_confidence");
  if (maxC !== null) filters.max_confidence = parseFloat(maxC);
  const minLC = sp.get("min_label_confidence");
  if (minLC !== null) filters.min_label_confidence = parseFloat(minLC);
  const maxLC = sp.get("max_label_confidence");
  if (maxLC !== null) filters.max_label_confidence = parseFloat(maxLC);
  const sort = sp.get("sort") as VerifySort | null;
  if (sort && sort !== "newest") filters.sort = sort;
  const seed = sp.get("seed");
  if (seed !== null) filters.seed = parseInt(seed, 10);
  return filters;
}

/** Serialize filter state to URL search params. */
function filtersToSearchParams(filters: SequenceFilterParams): URLSearchParams {
  const sp = new URLSearchParams();
  if (filters.site_ids?.length) sp.set("sites", filters.site_ids.join(","));
  if (filters.date_from) sp.set("from", filters.date_from);
  if (filters.date_to) sp.set("to", filters.date_to);
  if (filters.labels?.length) sp.set("labels", filters.labels.join(","));
  if (filters.verification && filters.verification !== "all")
    sp.set("verification", filters.verification);
  if (filters.flagged && filters.flagged !== "all")
    sp.set("flagged", filters.flagged);
  if (filters.favorited && filters.favorited !== "all")
    sp.set("favorited", filters.favorited);
  // "hide" is the implicit default, so only persist a deviation
  // ("all" to show everything, or "show_only").
  if (filters.empty && filters.empty !== "hide") sp.set("empty", filters.empty);
  if (filters.min_confidence !== undefined)
    sp.set("min_confidence", String(filters.min_confidence));
  if (filters.max_confidence !== undefined)
    sp.set("max_confidence", String(filters.max_confidence));
  if (filters.min_label_confidence !== undefined)
    sp.set("min_label_confidence", String(filters.min_label_confidence));
  if (filters.max_label_confidence !== undefined)
    sp.set("max_label_confidence", String(filters.max_label_confidence));
  if (filters.sort && filters.sort !== "newest") sp.set("sort", filters.sort);
  if (filters.seed !== undefined) sp.set("seed", String(filters.seed));
  return sp;
}

export interface VerifyViewProps {
  projectId: string;
}

export function VerifyView({ projectId }: VerifyViewProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  // Wide mode: measure the gallery container and fit as many sequence cards
  // as its width allows. Normal mode keeps the viewport-breakpoint grid.
  // The grid node is held in state (callback ref) so the observer also
  // attaches when the grid mounts after data loads, not just on toggle.
  const { wide, toggle: toggleWide } = useWideModeControls();
  const [gridNode, setGridNode] = useState<HTMLDivElement | null>(null);
  const [wideCols, setWideCols] = useState(4);
  useEffect(() => {
    if (!wide || !gridNode) return;
    const measure = () =>
      setWideCols(columnsForWidth(gridNode.clientWidth, 290, 16));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(gridNode);
    return () => ro.disconnect();
  }, [wide, gridNode]);
  const [page, setPage] = useState(0);
  const [selectedSequenceId, setSelectedSequenceId] = useState<string | null>(null);
  const [showSequencesWelcome, setShowSequencesWelcome] = useState(
    () => !localStorage.getItem("addaxai:verifyWelcomeDismissed"),
  );
  const handleDismissSequencesWelcome = useCallback(() => {
    setShowSequencesWelcome(false);
    localStorage.setItem("addaxai:verifyWelcomeDismissed", "1");
  }, []);

  // Parse filters from URL
  const filters = useMemo(
    () => filtersFromSearchParams(searchParams),
    [searchParams],
  );

  // Track previous filters to reset page on change
  const prevFiltersRef = useRef(filters);
  useEffect(() => {
    const prev = prevFiltersRef.current;
    if (JSON.stringify(prev) !== JSON.stringify(filters)) {
      setPage(0);
      prevFiltersRef.current = filters;
    }
  }, [filters]);

  // Debounced filters for API queries — prevents rapid-fire requests
  const debouncedFilters = useDebouncedValue(filters, FILTER_DEBOUNCE_MS);

  const setFilters = useCallback(
    (next: SequenceFilterParams) => {
      setSearchParams(filtersToSearchParams(next), { replace: true });
    },
    [setSearchParams],
  );

  // Listen for navigation sequences from the modal
  useEffect(() => {
    const handler = (e: Event) => {
      const targetId = (e as CustomEvent).detail;
      if (targetId) setSelectedSequenceId(targetId);
    };
    window.addEventListener("navigate-sequence", handler);
    return () => window.removeEventListener("navigate-sequence", handler);
  }, []);

  const queryClient = useQueryClient();

  // Get project detection threshold
  const { data: project } = useQuery({
    ...projectQueryOptions(projectId),
  });
  const detectionThreshold = project?.counting_threshold ?? 0;

  // The unfiltered sequence count, and how many of those are confirmed for
  // the progress pill. Deliberately UNFILTERED: the pill measures
  // progress against the whole project, not the current filter view, so
  // the user can't mistake a narrowed view for "everything confirmed".
  // Under the ["sequences"] prefix, so everything that refreshes the sequence
  // list (a confirm, a relabel) refreshes the pill with it.
  const { data: totalCountData } = useQuery({
    queryKey: ["sequences", "count", projectId],
    queryFn: () => sequencesApi.count(projectId),
  });

  // Get filtered sequence count
  const isFiltered = hasAnyActiveFilter(filters);
  const isDebouncedFiltered = hasAnyActiveFilter(debouncedFilters);
  const { data: filteredCountData } = useQuery({
    queryKey: ["sequence-count-filtered", projectId, debouncedFilters],
    queryFn: () => sequencesApi.count(projectId, debouncedFilters),
    enabled: isDebouncedFiltered,
  });

  // Get sequences with debounced filters
  const {
    data: sequences,
    isLoading,
    isFetching,
    isPlaceholderData,
  } = useQuery({
    queryKey: ["sequences", projectId, page, debouncedFilters],
    queryFn: () =>
      sequencesApi.list({
        project_id: projectId,
        skip: page * PAGE_SIZE,
        limit: PAGE_SIZE,
        filters: debouncedFilters,
      }),
    placeholderData: (prev) => prev,
  });

  // Warm the next page while this one is being worked: its sequence rows,
  // then each card's collage files and thumbnails (`warmFiles`), so
  // clicking Next paints instantly instead of starting cold. Skipped
  // while the current page is the held-over placeholder, so the
  // prefetch never competes with the fetch the user is waiting on.
  useEffect(() => {
    if (!sequences || isPlaceholderData || sequences.length < PAGE_SIZE) return;
    const nextPage = page + 1;
    queryClient
      .prefetchQuery({
        queryKey: ["sequences", projectId, nextPage, debouncedFilters],
        queryFn: () =>
          sequencesApi.list({
            project_id: projectId,
            skip: nextPage * PAGE_SIZE,
            limit: PAGE_SIZE,
            filters: debouncedFilters,
          }),
      })
      .then(() => {
        const next = queryClient.getQueryData<SequenceSummary[]>([
          "sequences",
          projectId,
          nextPage,
          debouncedFilters,
        ]);
        warmFiles(
          queryClient,
          (next ?? []).flatMap((e) => e.collage_file_ids ?? []),
        );
      });
  }, [sequences, isPlaceholderData, page, debouncedFilters, projectId, queryClient]);

  // Repaint the sequence cards when the project's colour map lands.
  useSpeciesColorsVersion();

  const totalSequences = totalCountData?.count ?? 0;
  const filteredSequences = isFiltered
    ? (filteredCountData?.count ?? totalSequences)
    : totalSequences;
  const hasMore = sequences && sequences.length === PAGE_SIZE;

  return (
    <div className="space-y-4">
      <VerifyFilterBar
        filters={filters}
        onChange={setFilters}
        projectId={projectId}
        classificationModelId={project?.classification_model_id}
        detectionFloor={detectionThreshold}
        countBy="sequence"
      />
      {totalSequences > 0 && (
        <VerifyToolbar>
          {/* Left: sort. Right: meta icons + progress, matching the
              Labels grid's toolbar grouping. */}
          <SortSelector
            sort={filters.sort ?? "newest"}
            seed={filters.seed ?? null}
            availableSorts={EVENTS_SORT_MODES}
            onChange={(next, seed) => {
              const updated: SequenceFilterParams = { ...filters };
              if (next === "newest") delete updated.sort;
              else updated.sort = next;
              if (seed === null) delete updated.seed;
              else updated.seed = seed;
              setFilters(updated);
            }}
          />
          <div className="ml-auto flex items-center gap-1">
            <VerifyToolbarIcon
              icon={wide ? Minimize2 : Maximize2}
              title={wide ? "Exit full width" : "Full width"}
              onClick={toggleWide}
              active={wide}
            />
            {/* Shared brightness/contrast (viewer-tools store); the
                collage tiles and the sequence modal follow the same value. */}
            <Popover>
              <PopoverTrigger asChild>
                <button
                  type="button"
                  title="Image (brightness, contrast)"
                  aria-label="Image (brightness, contrast)"
                  className={VERIFY_TOOLBAR_ICON_CLASS}
                >
                  <SlidersHorizontal className="h-4 w-4" />
                </button>
              </PopoverTrigger>
              <PopoverContent align="end" className="w-56 p-3 space-y-3">
                <ImageAdjustRows />
              </PopoverContent>
            </Popover>
            <VerifyGuideLink step="counts" />
            <div className="ml-2">
              <VerifyProgressPill
                pct={((totalCountData?.confirmed ?? 0) / totalSequences) * 100}
                label="confirmed"
              />
            </div>
          </div>
        </VerifyToolbar>
      )}
      {/* Sequence cards */}
      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : !sequences || sequences.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16 text-center">
            <Layers className="h-12 w-12 text-muted-foreground/50 mb-4" />
            <p className="text-lg font-medium text-muted-foreground">
              {totalSequences === 0
                ? "No sequences yet"
                : "No sequences match your filters"}
            </p>
            <p className="text-sm text-muted-foreground mt-1 max-w-md">
              {totalSequences === 0
                ? "Sequences are generated automatically when you run a deployment analysis. They group your camera trap images by time."
                : "Try adjusting or clearing your filters to see more sequences."}
            </p>
            {totalSequences > 0 && (
              <Button
                variant="outline"
                size="sm"
                className="mt-4"
                onClick={() => setFilters({})}
              >
                Clear all filters
              </Button>
            )}
          </CardContent>
        </Card>
      ) : (
        <>
          <div
            ref={setGridNode}
            className={cn(
              "grid gap-4 transition-opacity",
              !wide && "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4",
              isPlaceholderData && "opacity-60",
            )}
            style={
              wide
                ? { gridTemplateColumns: `repeat(${wideCols}, minmax(0, 1fr))` }
                : undefined
            }
          >
            {sequences.map((sequence) => (
              <SequenceCard
                key={sequence.id}
                sequence={sequence}
                detectionThreshold={detectionThreshold}
                onClick={() => setSelectedSequenceId(sequence.id)}
              />
            ))}
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-center gap-4">
            <Button
              variant="outline"
              size="sm"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
            >
              Previous
            </Button>
            <span className="text-sm text-muted-foreground">
              Showing {page * PAGE_SIZE + 1}-
              {page * PAGE_SIZE + sequences.length}
              {" of "}
              {isFiltered ? filteredSequences : totalSequences} sequences
              {isFetching && " (loading...)"}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={!hasMore}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </Button>
          </div>
        </>
      )}

      {/* Sequence detail modal */}
      <SequenceDetailModal
        sequenceId={selectedSequenceId}
        projectId={projectId}
        isOpen={!!selectedSequenceId}
        onClose={() => setSelectedSequenceId(null)}
        filters={debouncedFilters}
      />

      <WelcomePopover
        open={showSequencesWelcome && totalSequences > 0}
        onDismiss={handleDismissSequencesWelcome}
      />
    </div>
  );
}

function SequenceCard({
  sequence,
  detectionThreshold,
  onClick,
}: {
  sequence: SequenceSummary;
  detectionThreshold: number;
  onClick: () => void;
}) {
  // Render observational datetimes as the camera's wall-clock time, not
  // the viewer's browser-local time. See lib/datetime.ts.
  const startDate = formatCameraDate(sequence.sequence_start_local, {
    month: "short",
    day: "numeric",
  });
  const startTime = formatCameraTime(sequence.sequence_start_local);
  const endDate = formatCameraDate(sequence.sequence_end_local, {
    month: "short",
    day: "numeric",
  });
  const endTime = formatCameraTime(sequence.sequence_end_local);
  const sameTime = sequence.sequence_start_local === sequence.sequence_end_local;
  const sameDay = startDate === endDate;

  const dateTimeStr = sameTime
    ? `${startDate} · ${startTime}`
    : sameDay
      ? `${startDate} · ${startTime} – ${endTime}`
      : `${startDate} ${startTime} – ${endDate} ${endTime}`;

  // Drop species chips whose display name duplicates an observation
  // badge ("Vehicle" / "Person") that's already on this card.
  const observationBadgeNames = new Set<string>(
    sequence.observation_types.filter(
      (t) => t === "person" || t === "vehicle",
    ),
  );
  const speciesLabels = sequence.labels.filter((sp) => {
    const display = (speciesLabelMap(sequence)[sp] || sp).toLowerCase();
    return !observationBadgeNames.has(display);
  });

  // Effective count per species chip. Keyed the same way as `labels`
  // (label_taxonomy_id when present, else the raw label), so the count
  // joins to each chip. The peak frame per species carries the count.
  const countByKey = new Map<string, number>();
  for (const mf of sequence.max_n_frames) {
    const key = mf.label_taxonomy_id ?? mf.label;
    if (!key) continue;
    countByKey.set(key, Math.max(countByKey.get(key) ?? 0, mf.effective_count));
  }

  return (
    <Card
      className="relative hover:shadow-lg transition-shadow cursor-pointer"
      onClick={onClick}
    >
      <StatusBadgeCluster
        confirmed={sequence.is_confirmed}
        favorited={sequence.any_file_favorited}
        flagged={sequence.any_file_flagged}
      />
      <SequenceCollage
        fileIds={sequence.collage_file_ids}
        detectionThreshold={detectionThreshold}
      />

      <CardContent className="p-3 space-y-1.5">
        {/* Label chips — moved out of the thumbnail so the image is
            unobstructed. Same order as FileCard. [&>*]:rounded-sm
            overrides the Badge default rounded-full. */}
        <div className="flex flex-wrap gap-1 [&>*]:rounded-sm">
          {sequence.observation_types
            .filter((t) => t === "person" || t === "vehicle")
            .map((t) => {
              const badge = getObservationBadge(t);
              return (
                <Badge
                  key={t}
                  variant="outline"
                  className={`text-[10px] px-1.5 py-0.5 ${badge.className}`}
                  style={badge.style}
                >
                  {badge.label}
                </Badge>
              );
            })}
          {speciesLabels.length > 0 ? (
            <>
              {speciesLabels.slice(0, 2).map((sp) => {
                const count = countByKey.get(sp) ?? 0;
                return (
                  <Badge
                    key={sp}
                    variant="default"
                    className="text-[10px] px-1.5 py-0.5 max-w-[100px]"
                    style={{
                      backgroundColor: getSpeciesColor(sp),
                      color: getContrastTextColor(getSpeciesColor(sp)),
                    }}
                  >
                    <span className="truncate">
                      {speciesLabelMap(sequence)[sp] ||
                        sp.charAt(0).toUpperCase() + sp.slice(1)}
                    </span>
                    {count > 0 && (
                      <span className="ml-1 shrink-0 tabular-nums opacity-90">
                        ×{count}
                      </span>
                    )}
                  </Badge>
                );
              })}
              {speciesLabels.length > 2 && (
                <Badge variant="default" className="text-[10px] px-1.5 py-0.5">
                  +{speciesLabels.length - 2}
                </Badge>
              )}
            </>
          ) : (
            observationBadgeNames.size === 0 && (
              <Badge
                variant="outline"
                className="text-[10px] px-1.5 py-0.5 border-muted-foreground/40 text-muted-foreground"
              >
                Empty
              </Badge>
            )
          )}
        </div>

        {/* Same 3-row footer pattern as FileCard:
              row 2 — site name (font-medium, own row, omitted when absent)
              row 3 — date · time on the left, content-summary chip on the right
            The chip shows what's inside the sequence: image/video count.
            Mixed sequences drop the icon and use the compact "N img + M vid"
            form because both icons in one chip looks crowded. */}
        {sequence.site_name && (
          <div className="text-sm font-medium truncate">{sequence.site_name}</div>
        )}
        <div className="flex items-center justify-between gap-1.5 text-xs text-muted-foreground">
          <span>{dateTimeStr}</span>
          <span className="inline-flex shrink-0 items-center gap-1 rounded-sm border border-muted-foreground/40 px-1.5 py-0.5 text-[10px] font-medium">
            {sequence.image_count > 0 && sequence.video_count === 0 && (
              <>
                <ImageIcon className="h-3 w-3" />
                {sequence.image_count}{" "}
                {sequence.image_count === 1 ? "image" : "images"}
              </>
            )}
            {sequence.video_count > 0 && sequence.image_count === 0 && (
              <>
                <VideoIcon className="h-3 w-3" />
                {sequence.video_count}{" "}
                {sequence.video_count === 1 ? "video" : "videos"}
              </>
            )}
            {sequence.image_count > 0 && sequence.video_count > 0 && (
              <>
                {sequence.image_count} img + {sequence.video_count} vid
              </>
            )}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
