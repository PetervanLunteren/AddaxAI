/**
 * Verify view — the Counts page body, decoupled from page chrome.
 *
 * Keeps the `Verify*` name because it operates on the verification data
 * model. It renders the event gallery: a filter bar, a grid of
 * `EventCard`s, and the `EventDetailModal` (gallery + tools on the
 * left, panel on the right). The per-detection label-cleanup view lives
 * separately in `LabelsTab` / the Labels page. Mounted in two contexts:
 *
 * - Research projects: wrapped by `pages/CountsPage.tsx`, which adds the
 *   page chrome (header "Counts").
 *
 * Filter URL state lives in `useSearchParams`, path-agnostic.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  FILTER_DEBOUNCE_MS,
  useDebouncedValue,
} from "../../hooks/useDebouncedValue";
import {
  CircleHelp,
  Image as ImageIcon,
  Layers,
  Loader2,
  Maximize2,
  Minimize2,
  Video as VideoIcon,
} from "lucide-react";
import { eventsApi } from "../../api/events";
import { projectsApi } from "../../api/projects";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import { getObservationBadge } from "../../lib/detection-utils";
import { speciesLabelMap } from "../../lib/species-name-mode";
import {
  setSpeciesContext,
  getSpeciesColor,
  getSpeciesTextColor,
} from "../../utils/species-colors";
import type {
  EventSummary,
  EventFilterParams,
  VerificationFilter,
  VerifySort,
} from "../../api/types";

import { EventCollage } from "./EventCollage";
import { EventDetailModal } from "./EventDetailModal";
import { VerifyFilterBar } from "./VerifyFilterBar";
import { hasAnyActiveFilter } from "./FilterChips";
import { VerifyHelpSheet } from "./VerifyHelpSheet";
import { WelcomePopover } from "./WelcomePopover";
import { SortSelector } from "./SortSelector";
import {
  VerifyProgressPill,
  VerifyToolbar,
  VerifyToolbarIcon,
} from "./VerifyToolbar";
import { StatusBadgeCluster } from "./StatusBadgeCluster";
import { columnsForWidth, useWideModeControls } from "./wide-mode";
import { cn } from "../../lib/utils";

const EVENTS_SORT_MODES = ["newest", "oldest", "random"] as const;

// 48 = LCM(1,2,3,4), so every page lays out cleanly at every grid breakpoint
// (1/2/3/4 columns). Avoids orphan rows on intermediate pages.
const PAGE_SIZE = 48;

/** Parse filter state from URL search params. */
function filtersFromSearchParams(sp: URLSearchParams): EventFilterParams {
  const filters: EventFilterParams = {};
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
  const flagged = sp.get("flagged") as EventFilterParams["flagged"] | null;
  if (flagged && flagged !== "all") filters.flagged = flagged;
  const favorited = sp.get("favorited") as EventFilterParams["favorited"] | null;
  if (favorited && favorited !== "all") filters.favorited = favorited;
  // Empty defaults to "hide": most users don't want to scroll blank
  // tiles / all-blank events, and it's one click to "All". Absent
  // param ⇒ "hide"; an explicit "all" / "show_only" is persisted.
  const empty = sp.get("empty") as EventFilterParams["empty"] | null;
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
function filtersToSearchParams(filters: EventFilterParams): URLSearchParams {
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
  // Wide mode: measure the gallery container and fit as many event cards
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
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [showEventsWelcome, setShowEventsWelcome] = useState(
    () => !localStorage.getItem("addaxai:verifyWelcomeDismissed"),
  );
  const handleDismissEventsWelcome = useCallback(() => {
    setShowEventsWelcome(false);
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
    (next: EventFilterParams) => {
      setSearchParams(filtersToSearchParams(next), { replace: true });
    },
    [setSearchParams],
  );

  // Listen for navigation events from the modal
  useEffect(() => {
    const handler = (e: Event) => {
      const targetId = (e as CustomEvent).detail;
      if (targetId) setSelectedEventId(targetId);
    };
    window.addEventListener("navigate-event", handler);
    return () => window.removeEventListener("navigate-event", handler);
  }, []);

  // Get project detection threshold
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId),
  });
  const detectionThreshold = project?.counting_threshold ?? 0;

  // Get total event count (unfiltered)
  const { data: totalCountData } = useQuery({
    queryKey: ["event-count", projectId],
    queryFn: () => eventsApi.count(projectId),
  });

  // Get filtered event count
  const isFiltered = hasAnyActiveFilter(filters);
  const isDebouncedFiltered = hasAnyActiveFilter(debouncedFilters);
  const { data: filteredCountData } = useQuery({
    queryKey: ["event-count-filtered", projectId, debouncedFilters],
    queryFn: () => eventsApi.count(projectId, debouncedFilters),
    enabled: isDebouncedFiltered,
  });

  // Verification stats for the progress pill. Deliberately UNFILTERED:
  // the bar measures progress against the whole project, not the
  // current filter view, so the user can't mistake a narrowed view for
  // "everything verified".
  const { data: verificationStats } = useQuery({
    queryKey: ["events", "verification-stats", projectId],
    queryFn: () => eventsApi.verificationStats(projectId),
  });

  // Get events with debounced filters
  const {
    data: events,
    isLoading,
    isFetching,
    isPlaceholderData,
  } = useQuery({
    queryKey: ["events", projectId, page, debouncedFilters],
    queryFn: () =>
      eventsApi.list({
        project_id: projectId,
        skip: page * PAGE_SIZE,
        limit: PAGE_SIZE,
        filters: debouncedFilters,
      }),
    placeholderData: (prev) => prev,
  });

  // Set species color context from current events.
  // Labels are taxonomy UUIDs; register the active-mode name map as aliases
  // so name-string lookups (in LabelPicker, overlay, etc.) get
  // the same color.
  useMemo(() => {
    if (events?.length) {
      const allLabels = [...new Set(events.flatMap((e) => e.labels))];
      // Collect UUID -> name aliases from all events (active name mode).
      const aliases: Record<string, string> = {};
      for (const e of events) {
        for (const [uuid, name] of Object.entries(speciesLabelMap(e))) {
          aliases[uuid] = name;
        }
      }
      if (allLabels.length > 0) setSpeciesContext(allLabels, aliases);
    }
  }, [events]);

  const totalEvents = totalCountData?.count ?? 0;
  const filteredEvents = isFiltered
    ? (filteredCountData?.count ?? totalEvents)
    : totalEvents;
  const hasMore = events && events.length === PAGE_SIZE;

  return (
    <div className="space-y-4">
      <VerifyFilterBar
        filters={filters}
        onChange={setFilters}
        projectId={projectId}
        classificationModelId={project?.classification_model_id}
        detectionFloor={detectionThreshold}
        countBy="event"
      />
      {totalEvents > 0 && verificationStats && (
        <VerifyToolbar>
          {/* Left: sort. Right: meta icons + progress, matching the
              Labels grid's toolbar grouping. */}
          <SortSelector
            sort={filters.sort ?? "newest"}
            seed={filters.seed ?? null}
            availableSorts={EVENTS_SORT_MODES}
            onChange={(next, seed) => {
              const updated: EventFilterParams = { ...filters };
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
            <VerifyToolbarIcon
              icon={CircleHelp}
              title="Help"
              onClick={() => setHelpOpen(true)}
            />
            <div className="ml-2">
              <VerifyProgressPill
                pct={
                  verificationStats.events_total > 0
                    ? (verificationStats.events_confirmed /
                        verificationStats.events_total) *
                      100
                    : 0
                }
                label="confirmed"
              />
            </div>
          </div>
        </VerifyToolbar>
      )}
      {/* Event cards */}
      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : !events || events.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16 text-center">
            <Layers className="h-12 w-12 text-muted-foreground/50 mb-4" />
            <p className="text-lg font-medium text-muted-foreground">
              {totalEvents === 0
                ? "No events yet"
                : "No events match your filters"}
            </p>
            <p className="text-sm text-muted-foreground mt-1 max-w-md">
              {totalEvents === 0
                ? "Events are generated automatically when you run a deployment analysis. They group your camera trap images by time based on the project's independence interval."
                : "Try adjusting or clearing your filters to see more events."}
            </p>
            {totalEvents > 0 && (
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
            {events.map((event) => (
              <EventCard
                key={event.id}
                event={event}
                detectionThreshold={detectionThreshold}
                onClick={() => setSelectedEventId(event.id)}
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
              {page * PAGE_SIZE + events.length}
              {" of "}
              {isFiltered ? filteredEvents : totalEvents} events
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

      {/* Event detail modal */}
      <EventDetailModal
        eventId={selectedEventId}
        projectId={projectId}
        isOpen={!!selectedEventId}
        onClose={() => setSelectedEventId(null)}
        filters={debouncedFilters}
      />

      <VerifyHelpSheet open={helpOpen} onOpenChange={setHelpOpen} step="counts" />

      <WelcomePopover
        open={showEventsWelcome && totalEvents > 0}
        onDismiss={handleDismissEventsWelcome}
      />
    </div>
  );
}

function EventCard({
  event,
  detectionThreshold,
  onClick,
}: {
  event: EventSummary;
  detectionThreshold: number;
  onClick: () => void;
}) {
  // Render observational datetimes as the camera's wall-clock time, not
  // the viewer's browser-local time. See lib/datetime.ts.
  const startDate = formatCameraDate(event.event_start_local, {
    month: "short",
    day: "numeric",
  });
  const startTime = formatCameraTime(event.event_start_local);
  const endDate = formatCameraDate(event.event_end_local, {
    month: "short",
    day: "numeric",
  });
  const endTime = formatCameraTime(event.event_end_local);
  const sameTime = event.event_start_local === event.event_end_local;
  const sameDay = startDate === endDate;

  const dateTimeStr = sameTime
    ? `${startDate} · ${startTime}`
    : sameDay
      ? `${startDate} · ${startTime} – ${endTime}`
      : `${startDate} ${startTime} – ${endDate} ${endTime}`;

  // Drop species chips whose display name duplicates an observation
  // badge ("Vehicle" / "Person") that's already on this card.
  const observationBadgeNames = new Set<string>(
    event.observation_types.filter(
      (t) => t === "person" || t === "vehicle",
    ),
  );
  const speciesLabels = event.labels.filter((sp) => {
    const display = (speciesLabelMap(event)[sp] || sp).toLowerCase();
    return !observationBadgeNames.has(display);
  });

  // Effective count per species chip. Keyed the same way as `labels`
  // (label_taxonomy_id when present, else the raw label), so the count
  // joins to each chip. The peak frame per species carries the count.
  const countByKey = new Map<string, number>();
  for (const mf of event.max_n_frames) {
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
        confirmed={event.is_confirmed}
        favorited={event.any_file_favorited}
        flagged={event.any_file_flagged}
      />
      <EventCollage
        fileIds={event.collage_file_ids}
        detectionThreshold={detectionThreshold}
      />

      <CardContent className="p-3 space-y-1.5">
        {/* Label chips — moved out of the thumbnail so the image is
            unobstructed. Same order as FileCard. [&>*]:rounded-sm
            overrides the Badge default rounded-full. */}
        <div className="flex flex-wrap gap-1 [&>*]:rounded-sm">
          {event.observation_types
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
                      color: getSpeciesTextColor(sp),
                    }}
                  >
                    <span className="truncate">
                      {speciesLabelMap(event)[sp] ||
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
            The chip shows what's inside the event: image/video count.
            Mixed events drop the icon and use the compact "N img + M vid"
            form because both icons in one chip looks crowded. */}
        {event.site_name && (
          <div className="text-sm font-medium truncate">{event.site_name}</div>
        )}
        <div className="flex items-center justify-between gap-1.5 text-xs text-muted-foreground">
          <span>{dateTimeStr}</span>
          <span className="inline-flex shrink-0 items-center gap-1 rounded-sm border border-muted-foreground/40 px-1.5 py-0.5 text-[10px] font-medium">
            {event.image_count > 0 && event.video_count === 0 && (
              <>
                <ImageIcon className="h-3 w-3" />
                {event.image_count}{" "}
                {event.image_count === 1 ? "image" : "images"}
              </>
            )}
            {event.video_count > 0 && event.image_count === 0 && (
              <>
                <VideoIcon className="h-3 w-3" />
                {event.video_count}{" "}
                {event.video_count === 1 ? "video" : "videos"}
              </>
            )}
            {event.image_count > 0 && event.video_count > 0 && (
              <>
                {event.image_count} img + {event.video_count} vid
              </>
            )}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
