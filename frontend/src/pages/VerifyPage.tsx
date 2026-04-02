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
import { Loader2, Layers, Check, Circle } from "lucide-react";
import { eventsApi } from "../api/events";
import { sitesApi } from "../api/sites";
import { filesApi } from "../api/files";
import { projectsApi } from "../api/projects";
import { API_BASE_URL } from "../lib/api-client";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { getDetectionColor, getObservationBadge } from "../lib/detection-utils";
import { setSpeciesContext, getSpeciesColor, getSpeciesTextColor } from "../utils/species-colors";
import type { EventSummary, EventFilterParams, VerificationFilter } from "../api/types";

import { EventDetailModal } from "../components/verify/EventDetailModal";
import { FilterPanel } from "../components/verify/FilterPanel";
import { FilterChips } from "../components/verify/FilterChips";
import { HelpSheet } from "../components/verify/HelpSheet";
import { SimilarityTab } from "../components/verify/SimilarityTab";
import { EventsStatsToolbar } from "../components/verify/EventsStatsToolbar";

const PAGE_SIZE = 50;
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
  return sp;
}

/** Check if any filter is active. */
function hasActiveFilters(filters: EventFilterParams): boolean {
  return (
    (filters.site_ids?.length ?? 0) > 0 ||
    !!filters.date_from ||
    !!filters.date_to ||
    (filters.labels?.length ?? 0) > 0 ||
    (!!filters.verification && filters.verification !== "all")
  );
}

export default function VerifyPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const [page, setPage] = useState(0);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);

  // Tab state from URL
  const activeTab = (searchParams.get("tab") as "events" | "similarity") || "events";
  const setActiveTab = useCallback(
    (tab: "events" | "similarity") => {
      // Cancel in-flight event queries to free browser connections
      if (tab === "similarity") {
        queryClient.cancelQueries({ queryKey: ["events"] });
        queryClient.cancelQueries({ queryKey: ["event-count-filtered"] });
        queryClient.cancelQueries({ queryKey: ["file"] });
      }
      setSearchParams(
        (prev) => {
          if (tab === "events") {
            prev.delete("tab");
            prev.delete("mode");
            prev.delete("anchor");
            // Clean sim_* params when leaving Similarity tab
            for (const key of [...prev.keys()]) {
              if (key.startsWith("sim_")) prev.delete(key);
            }
          } else {
            prev.set("tab", tab);
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
      // Preserve tab-related params
      if (activeTab !== "events") sp.set("tab", activeTab);
      const mode = searchParams.get("mode");
      if (mode) sp.set("mode", mode);
      const anchor = searchParams.get("anchor");
      if (anchor) sp.set("anchor", anchor);
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
  const isFiltered = hasActiveFilters(filters);
  const isDebouncedFiltered = hasActiveFilters(debouncedFilters);
  const { data: filteredCountData } = useQuery({
    queryKey: ["event-count-filtered", projectId, debouncedFilters],
    queryFn: () => eventsApi.count(projectId!, debouncedFilters),
    enabled: !!projectId && isDebouncedFiltered,
  });

  // Get verification stats across filtered events
  const { data: verificationStats } = useQuery({
    queryKey: ["events", "verification-stats", projectId, debouncedFilters],
    queryFn: () => eventsApi.verificationStats(projectId!, debouncedFilters),
    enabled: !!projectId && activeTab === "events",
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
        project_id: projectId!,
        skip: page * PAGE_SIZE,
        limit: PAGE_SIZE,
        filters: debouncedFilters,
      }),
    enabled: !!projectId,
    placeholderData: (prev) => prev,
  });

  // Set species color context from current events
  useMemo(() => {
    if (events?.length) {
      const allLabels = [...new Set(events.flatMap((e) => e.labels))];
      if (allLabels.length > 0) setSpeciesContext(allLabels);
    }
  }, [events]);

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
    <div className="p-8 bg-gradient-to-br from-slate-50 to-slate-100 min-h-screen">
      <div className="mx-auto max-w-7xl space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              Verify
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              {totalEvents > 0
                ? "Review and verify AI detections. Events verifies at the file level, similarity at the detection level."
                : "Run a deployment analysis to get started"}
            </p>
          </div>
          {/* Filters are always visible below */}
        </div>

        {/* Tab strip */}
        <div className="flex items-center gap-1 border-b">
          <button
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "events"
                ? "border-primary text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
            onClick={() => setActiveTab("events")}
          >
            Events
          </button>
          <button
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "similarity"
                ? "border-primary text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
            onClick={() => setActiveTab("similarity")}
          >
            Similarity
          </button>
        </div>

        {/* Tab content */}
        {activeTab === "events" ? (
          <>
            {/* Filter panel */}
            <FilterPanel
              filters={filters}
              onChange={setFilters}
              projectId={projectId!}
              isOpen={true}
              onToggle={() => {}}
              classificationModelId={project?.classification_model_id}
            >
              {isFiltered && (
                <FilterChips
                  filters={filters}
                  onChange={setFilters}
                  filteredCount={filteredEvents}
                  totalCount={totalEvents}
                  siteNames={siteNames}
                />
              )}
            </FilterPanel>
            {totalEvents > 0 && (
              <EventsStatsToolbar
                stats={verificationStats}
                onHelpClick={() => setHelpOpen(true)}
              />
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
          <SimilarityTab
            projectId={projectId!}
            classificationModelId={project?.classification_model_id ?? null}
          />
        )}

        <HelpSheet open={helpOpen} onOpenChange={setHelpOpen} />
      </div>
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
  const startTime = new Date(event.start_time);
  const endTime = new Date(event.end_time);
  const sameTime = event.start_time === event.end_time;
  const sameDay = startTime.toDateString() === endTime.toDateString();

  const fmtDate = (d: Date) => d.toLocaleDateString([], { month: "short", day: "numeric" });
  const fmtTime = (d: Date) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const dateTimeStr = sameTime
    ? `${fmtDate(startTime)} · ${fmtTime(startTime)}`
    : sameDay
      ? `${fmtDate(startTime)} · ${fmtTime(startTime)} – ${fmtTime(endTime)}`
      : `${fmtDate(startTime)} ${fmtTime(startTime)} – ${fmtDate(endTime)} ${fmtTime(endTime)}`;

  const thumbnailUrl = event.thumbnail_file_id
    ? `${API_BASE_URL}/api/files/${event.thumbnail_file_id}/image`
    : undefined;

  // Fetch thumbnail file detections for overlay
  const { data: thumbFile } = useQuery({
    queryKey: ["file", event.thumbnail_file_id],
    queryFn: ({ signal }) => filesApi.get(event.thumbnail_file_id!, { signal }),
    enabled: !!event.thumbnail_file_id,
  });


  return (
    <Card
      className="overflow-hidden hover:shadow-lg transition-shadow cursor-pointer"
      onClick={onClick}
    >
      {/* Thumbnail */}
      <div className="aspect-video bg-muted relative">
        {thumbnailUrl ? (
          <img
            src={thumbnailUrl}
            alt="Event thumbnail"
            className="w-full h-full object-cover"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        ) : (
          <div className="flex items-center justify-center h-full">
            <Layers className="h-8 w-8 text-muted-foreground/30" />
          </div>
        )}
        {/* Detection overlay */}
        {thumbFile && (() => {
          const dets = thumbFile.detections.filter(
            (d) => d.confidence >= detectionThreshold
          );
          if (dets.length === 0) return null;
          const imgW = thumbFile.width_px || 1;
          const imgH = thumbFile.height_px || 1;
          // Use a fixed reference size for 16:9 viewBox
          const VW = 320;
          const VH = 180;
          const scale = Math.max(VW / imgW, VH / imgH);
          const dw = imgW * scale;
          const dh = imgH * scale;
          const ox = (VW - dw) / 2;
          const oy = (VH - dh) / 2;
          const maskId = `m-card-${event.thumbnail_file_id}`;
          const boxes = dets.map((det) => {
            const bx = ox + det.bbox_x * dw;
            const by = oy + det.bbox_y * dh;
            const bw = det.bbox_width * dw;
            const bh = det.bbox_height * dh;
            const color = getDetectionColor(det);
            return { bx, by, bw, bh, color };
          });
          return (
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none"
              viewBox={`0 0 ${VW} ${VH}`}
            >
              <defs>
                <mask id={maskId}>
                  <rect width={VW} height={VH} fill="white" />
                  {boxes.map((b, i) => (
                    <rect key={i} x={b.bx} y={b.by} width={b.bw} height={b.bh} rx={4} fill="black" />
                  ))}
                </mask>
              </defs>
              <rect width={VW} height={VH} fill="rgba(0,0,0,0.55)" mask={`url(#${maskId})`} />
              {boxes.map((b, i) => (
                <rect
                  key={i}
                  x={b.bx}
                  y={b.by}
                  width={b.bw}
                  height={b.bh}
                  rx={4}
                  fill="none"
                  stroke={b.color}
                  strokeWidth={2.5}
                  opacity={1}
                />
              ))}
            </svg>
          );
        })()}
        {/* Label chips */}
        <div className="absolute bottom-2 left-2 flex gap-1">
          {event.observation_types
            .filter((t) => t === "human" || t === "vehicle")
            .map((t) => {
              const badge = getObservationBadge(t);
              return (
                <Badge
                  key={t}
                  variant="outline"
                  className={`text-[10px] px-1.5 py-0.5 shadow-sm ${badge.className}`}
                  style={badge.style}
                >
                  {badge.label}
                </Badge>
              );
            })}
          {event.labels.length > 0 ? (
            <>
              {event.labels.slice(0, 2).map((sp) => (
                <Badge
                  key={sp}
                  variant="default"
                  className="text-[10px] px-1.5 py-0.5 shadow-sm max-w-[100px]"
                  style={{ backgroundColor: getSpeciesColor(sp), color: getSpeciesTextColor(sp) }}
                >
                  <span className="truncate">{event.display_labels?.[sp] || sp.charAt(0).toUpperCase() + sp.slice(1)}</span>
                </Badge>
              ))}
              {event.labels.length > 2 && (
                <Badge
                  variant="default"
                  className="text-[10px] px-1.5 py-0.5 shadow-sm"
                >
                  +{event.labels.length - 2}
                </Badge>
              )}
            </>
          ) : (
            <Badge
              variant="secondary"
              className="text-[10px] px-1.5 py-0.5 shadow-sm"
            >
              Empty
            </Badge>
          )}
        </div>
      </div>

      <CardContent className="p-3 space-y-2">
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
        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-1">
            MaxN {event.verified_maxn_count}/{event.total_maxn_count}
            {event.verified_maxn_count === event.total_maxn_count && event.total_maxn_count > 0 ? (
              <div className="bg-primary rounded-full p-0.5">
                <Check className="h-2.5 w-2.5 text-primary-foreground" />
              </div>
            ) : (
              <Circle className="h-3 w-3" />
            )}
          </span>
          <span className="text-muted-foreground">
            All {event.verified_count}/{event.total_count}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
