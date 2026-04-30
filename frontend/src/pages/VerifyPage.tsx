/**
 * Browse and verify page - event-centric browse and annotation workflow.
 *
 * Displays events as cards with thumbnails, label tags, and verification progress.
 * Clicking a card opens the event detail modal (Phase 2).
 * Supports filtering by site, date range, label, verification status, and confidence.
 * Filter state is persisted in URL search params.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleHelp, Layers, Loader2 } from "lucide-react";
import { eventsApi } from "../api/events";
import { sitesApi } from "../api/sites";
import { projectsApi } from "../api/projects";
import { formatCameraDate, formatCameraTime } from "../lib/datetime";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { getObservationBadge } from "../lib/detection-utils";
import { setSpeciesContext, getSpeciesColor, getSpeciesTextColor } from "../utils/species-colors";
import type {
  EventSummary,
  EventFilterParams,
  VerificationFilter,
  VerifySort,
} from "../api/types";

import { EventCollage } from "../components/verify/EventCollage";
import { EventDetailModal } from "../components/verify/EventDetailModal";
import { VerifyFilterBar } from "../components/verify/VerifyFilterBar";
import {
  FilterChips,
  hasAnyActiveFilter,
} from "../components/verify/FilterChips";
import { HelpSheet } from "../components/verify/HelpSheet";
import { WelcomePopover } from "../components/verify/WelcomePopover";
import { FilesTab } from "../components/verify/FilesTab";
import { ObservationsTab } from "../components/verify/ObservationsTab";
import { SortSelector } from "../components/verify/SortSelector";
import {
  VerifyProgressPill,
  VerifyToolbar,
  VerifyToolbarIcon,
} from "../components/verify/VerifyToolbar";
import { StatusBadgeCluster } from "../components/verify/StatusBadgeCluster";

const EVENTS_SORT_MODES = ["newest", "oldest", "random", "cls_low"] as const;

type VerifyTab = "events" | "captures" | "observations";

// 48 = LCM(1,2,3,4), so every page lays out cleanly at every grid breakpoint
// (1/2/3/4 columns). Avoids orphan rows on intermediate pages.
const PAGE_SIZE = 48;
const FILTER_DEBOUNCE_MS = 300;

/** Debounce a value by `delay` ms. Compares by JSON serialization. */
function useDebouncedValue<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  const serialized = JSON.stringify(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(JSON.parse(serialized)), delay);
    return () => clearTimeout(timer);
  }, [serialized, delay]);
  return debounced;
}

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
  const empty = sp.get("empty") as EventFilterParams["empty"] | null;
  if (empty && empty !== "all") filters.empty = empty;
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
  if (filters.empty && filters.empty !== "all")
    sp.set("empty", filters.empty);
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


export default function VerifyPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const [page, setPage] = useState(0);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [showEventsWelcome, setShowEventsWelcome] = useState(
    () => !localStorage.getItem("addaxai:verifyWelcomeDismissed")
  );
  const handleDismissEventsWelcome = useCallback(() => {
    setShowEventsWelcome(false);
    localStorage.setItem("addaxai:verifyWelcomeDismissed", "1");
  }, []);

  // Tab state from URL
  const rawTab = searchParams.get("tab");
  const activeTab: VerifyTab =
    rawTab === "captures" || rawTab === "events" ? rawTab : "observations";
  const setActiveTab = useCallback(
    (tab: VerifyTab) => {
      // Cancel in-flight queries for the tab we are leaving so the browser
      // connection pool isn't tied up loading data that is about to be hidden.
      if (tab !== "events") {
        queryClient.cancelQueries({ queryKey: ["events"] });
        queryClient.cancelQueries({ queryKey: ["event-count-filtered"] });
      }
      if (tab !== "captures") {
        queryClient.cancelQueries({ queryKey: ["files-for-verify"] });
        queryClient.cancelQueries({ queryKey: ["files-count-for-verify"] });
        queryClient.cancelQueries({ queryKey: ["files-verification-stats"] });
      }
      setSearchParams(
        (prev) => {
          // Observations is the default tab, so it has no ?tab= param.
          if (tab === "observations") {
            prev.delete("tab");
          } else {
            prev.set("tab", tab);
          }
          // Observations-only params are stripped when leaving the Observations tab.
          if (tab !== "observations") {
            prev.delete("mode");
            prev.delete("anchor");
            for (const key of [...prev.keys()]) {
              if (key.startsWith("obs_")) prev.delete(key);
            }
          }
          return prev;
        },
        { replace: true }
      );
    },
    [setSearchParams, queryClient]
  );

  // Parse filters from URL
  const filters = useMemo(
    () => filtersFromSearchParams(searchParams),
    [searchParams]
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
      const sp = filtersToSearchParams(next);
      // Preserve tab-related params. Observations is the implicit
      // default and has no tab= param; Events and Captures keep
      // theirs so a filter edit doesn't warp the user back home.
      if (activeTab !== "observations") sp.set("tab", activeTab);
      const mode = searchParams.get("mode");
      if (mode) sp.set("mode", mode);
      const anchor = searchParams.get("anchor");
      if (anchor) sp.set("anchor", anchor);
      // Preserve Observations-owned filter params (obs_*). Events and Files
      // share the unprefixed filter params above; Observations has its own
      // namespace and must survive a shared-filter edit from another tab.
      for (const key of [...searchParams.keys()]) {
        if (key.startsWith("obs_")) {
          sp.set(key, searchParams.get(key)!);
        }
      }
      setSearchParams(sp, { replace: true });
    },
    [setSearchParams, activeTab, searchParams]
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
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  });
  const detectionThreshold = project?.detection_threshold ?? 0;

  // Get total event count (unfiltered)
  const { data: totalCountData } = useQuery({
    queryKey: ["event-count", projectId],
    queryFn: () => eventsApi.count(projectId!),
    enabled: !!projectId,
  });

  // Get filtered event count
  const isFiltered = hasAnyActiveFilter(filters);
  const isDebouncedFiltered = hasAnyActiveFilter(debouncedFilters);
  const { data: filteredCountData } = useQuery({
    queryKey: ["event-count-filtered", projectId, debouncedFilters],
    queryFn: () => eventsApi.count(projectId!, debouncedFilters),
    enabled: !!projectId && isDebouncedFiltered,
  });

  // Get verification stats across filtered events (only for Events tab)
  const { data: verificationStats } = useQuery({
    queryKey: ["events", "verification-stats", projectId, debouncedFilters],
    queryFn: () => eventsApi.verificationStats(projectId!, debouncedFilters),
    enabled: !!projectId && activeTab === "events",
  });

  // Data queries for Events and Files share filters, so Events-only data
  // fetches below gate on activeTab === "events" to avoid busy connections
  // while sitting on the Files or Observations tab.

  // Get events with debounced filters (only when Events tab is active)
  const {
    data: events,
    isLoading,
    isFetching,
    isPlaceholderData,
  } = useQuery({
    queryKey: ["events", projectId, page, debouncedFilters],
    queryFn: () =>
      eventsApi.list({
        project_id: projectId!,
        skip: page * PAGE_SIZE,
        limit: PAGE_SIZE,
        filters: debouncedFilters,
      }),
    enabled: !!projectId && activeTab === "events",
    placeholderData: (prev) => prev,
  });

  // Set species color context from current events.
  // Labels are taxonomy UUIDs; register display_labels as aliases
  // so name-string lookups (in LabelPicker, overlay, etc.) get
  // the same color.
  useMemo(() => {
    if (events?.length) {
      const allLabels = [...new Set(events.flatMap((e) => e.labels))];
      // Collect UUID -> name aliases from all events' display_labels
      const aliases: Record<string, string> = {};
      for (const e of events) {
        if (e.display_labels) {
          for (const [uuid, name] of Object.entries(e.display_labels)) {
            aliases[uuid] = name;
          }
        }
      }
      if (allLabels.length > 0) setSpeciesContext(allLabels, aliases);
    }
  }, [events]);

  // Fetch filter options for display_labels mapping (shared cache with VerifyFilterBar)
  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId!),
    enabled: !!projectId,
  });

  // Fetch sites for name mapping in filter chips
  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId!),
    enabled: !!projectId && (filters.site_ids?.length ?? 0) > 0,
  });
  const siteNames = useMemo(() => {
    const map: Record<string, string> = {};
    for (const s of sites ?? []) map[s.id] = s.name;
    return map;
  }, [sites]);

  const totalEvents = totalCountData?.count ?? 0;
  const filteredEvents = isFiltered
    ? (filteredCountData?.count ?? totalEvents)
    : totalEvents;
  const hasMore = events && events.length === PAGE_SIZE;

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Verify</h1>
              <p className="text-sm text-muted-foreground">
                {totalEvents > 0
                  ? "Confirm or correct AI detections"
                  : "Run a deployment analysis to get started"}
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-4">
        {/* Tab strip */}
        <div className="flex items-center gap-1 border-b">
          {(
            [
              ["observations", "Observations"],
              ["captures", "Captures"],
              ["events", "Events"],
            ] as const
          ).map(([tab, label]) => (
            <button
              key={tab}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
              onClick={() => setActiveTab(tab)}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {activeTab === "captures" ? (
          <FilesTab
            projectId={projectId!}
            filters={filters}
            onFiltersChange={setFilters}
            classificationModelId={project?.classification_model_id ?? null}
          />
        ) : activeTab === "events" ? (
          <>
            <VerifyFilterBar
              filters={filters}
              onChange={setFilters}
              projectId={projectId!}
              classificationModelId={project?.classification_model_id}
              detectionFloor={detectionThreshold}
              countBy="event"
            />
            {isFiltered && (
              <FilterChips
                filters={filters}
                onChange={setFilters}
                filteredCount={filteredEvents}
                totalCount={totalEvents}
                siteNames={siteNames}
                displayLabels={filterOptions?.display_labels}
                detectionFloor={detectionThreshold}
              />
            )}
            {totalEvents > 0 && verificationStats && (
              <VerifyToolbar>
                <VerifyToolbarIcon
                  icon={CircleHelp}
                  title="Help"
                  onClick={() => setHelpOpen(true)}
                />
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
                <VerifyProgressPill
                  pct={
                    verificationStats.events_total > 0
                      ? (verificationStats.events_fully_verified /
                          verificationStats.events_total) *
                        100
                      : 0
                  }
                  label="events verified"
                />
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
                    {isFiltered ? "No events match your filters" : "No events yet"}
                  </p>
                  <p className="text-sm text-muted-foreground mt-1 max-w-md">
                    {isFiltered
                      ? "Try adjusting or clearing your filters to see more events."
                      : "Events are generated automatically when you run a deployment analysis. They group your camera trap images by time based on the project's independence interval."}
                  </p>
                  {isFiltered && (
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
                <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 transition-opacity ${isPlaceholderData ? "opacity-60" : ""}`}>
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
                    {isFiltered
                      ? `${filteredEvents} of ${totalEvents} events`
                      : `${totalEvents} events`}
                    {" · "}Page {page + 1}
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
              projectId={projectId!}
              isOpen={!!selectedEventId}
              onClose={() => setSelectedEventId(null)}
              filters={debouncedFilters}
            />
          </>
        ) : (
          <ObservationsTab
            projectId={projectId!}
            classificationModelId={project?.classification_model_id ?? null}
          />
        )}

        <HelpSheet open={helpOpen} onOpenChange={setHelpOpen} />

        <WelcomePopover
          open={activeTab === "events" && showEventsWelcome && totalEvents > 0}
          onDismiss={handleDismissEventsWelcome}
        />
      </main>
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
  const startDate = formatCameraDate(event.event_start_local, { month: "short", day: "numeric" });
  const startTime = formatCameraTime(event.event_start_local);
  const endDate = formatCameraDate(event.event_end_local, { month: "short", day: "numeric" });
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
  const observationBadgeNames = new Set(
    event.observation_types
      .filter((t) => t === "human" || t === "vehicle")
      .map((t) => (t === "human" ? "person" : t)),
  );
  const speciesLabels = event.labels.filter((sp) => {
    const display = (event.display_labels?.[sp] || sp).toLowerCase();
    return !observationBadgeNames.has(display);
  });

  return (
    <Card
      className="relative hover:shadow-lg transition-shadow cursor-pointer"
      onClick={onClick}
    >
      <StatusBadgeCluster
        verified={event.is_verified}
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
            .filter((t) => t === "human" || t === "vehicle")
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
              {speciesLabels.slice(0, 2).map((sp) => (
                <Badge
                  key={sp}
                  variant="default"
                  className="text-[10px] px-1.5 py-0.5 max-w-[100px]"
                  style={{ backgroundColor: getSpeciesColor(sp), color: getSpeciesTextColor(sp) }}
                >
                  <span className="truncate">{event.display_labels?.[sp] || sp.charAt(0).toUpperCase() + sp.slice(1)}</span>
                </Badge>
              ))}
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

        <div className="flex items-center justify-between text-sm">
          <span className="font-medium">
            {[
              event.image_count > 0 && `${event.image_count} ${event.image_count === 1 ? "image" : "images"}`,
              event.video_count > 0 && `${event.video_count} ${event.video_count === 1 ? "video" : "videos"}`,
            ].filter(Boolean).join(", ")}
          </span>
          {event.site_name && (
            <span className="text-xs text-muted-foreground truncate ml-2 max-w-[120px]">{event.site_name}</span>
          )}
        </div>
        <div className="text-xs text-muted-foreground">{dateTimeStr}</div>
      </CardContent>
    </Card>
  );
}
