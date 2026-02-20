/**
 * Browse and verify page - event-centric browse and annotation workflow.
 *
 * Displays events as cards with thumbnails, species tags, and verification progress.
 * Clicking a card opens the event detail modal (Phase 2).
 * Supports filtering by site, date range, species, verification status, and confidence.
 * Filter state is persisted in URL search params.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Filter, Loader2, Layers } from "lucide-react";
import { eventsApi } from "../api/events";
import { sitesApi } from "../api/sites";
import { filesApi } from "../api/files";
import { projectsApi } from "../api/projects";
import { API_BASE_URL } from "../lib/api-client";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { getCategoryColor, getObservationBadge } from "../lib/detection-utils";
import type { EventSummary, EventFilterParams, VerificationFilter } from "../api/types";

import { EventDetailModal } from "../components/verify/EventDetailModal";
import { FilterPanel } from "../components/verify/FilterPanel";
import { FilterChips } from "../components/verify/FilterChips";

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
  const species = sp.get("species");
  if (species) filters.species = species.split(",");
  const verification = sp.get("verification") as VerificationFilter | null;
  if (verification && verification !== "all") filters.verification = verification;
  const confMin = sp.get("conf_min");
  if (confMin) filters.min_confidence = parseFloat(confMin);
  const confMax = sp.get("conf_max");
  if (confMax) filters.max_confidence = parseFloat(confMax);
  return filters;
}

/** Serialize filter state to URL search params. */
function filtersToSearchParams(filters: EventFilterParams): URLSearchParams {
  const sp = new URLSearchParams();
  if (filters.site_ids?.length) sp.set("sites", filters.site_ids.join(","));
  if (filters.date_from) sp.set("from", filters.date_from);
  if (filters.date_to) sp.set("to", filters.date_to);
  if (filters.species?.length) sp.set("species", filters.species.join(","));
  if (filters.verification && filters.verification !== "all")
    sp.set("verification", filters.verification);
  if (filters.min_confidence !== undefined)
    sp.set("conf_min", filters.min_confidence.toString());
  if (filters.max_confidence !== undefined)
    sp.set("conf_max", filters.max_confidence.toString());
  return sp;
}

/** Check if any filter is active. */
function hasActiveFilters(filters: EventFilterParams): boolean {
  return (
    (filters.site_ids?.length ?? 0) > 0 ||
    !!filters.date_from ||
    !!filters.date_to ||
    (filters.species?.length ?? 0) > 0 ||
    (!!filters.verification && filters.verification !== "all") ||
    filters.min_confidence !== undefined ||
    filters.max_confidence !== undefined
  );
}

export default function VerifyPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const [page, setPage] = useState(0);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [filterPanelOpen, setFilterPanelOpen] = useState(false);

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
      setSearchParams(filtersToSearchParams(next), { replace: true });
    },
    [setSearchParams]
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
              Browse and verify
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              {totalEvents > 0
                ? isFiltered
                  ? `${filteredEvents} of ${totalEvents} events`
                  : `${totalEvents} events`
                : "Run a deployment analysis to get started"}
            </p>
          </div>
          {totalEvents > 0 && (
            <Button
              variant={filterPanelOpen ? "default" : "outline"}
              size="sm"
              onClick={() => setFilterPanelOpen((v) => !v)}
              className="gap-1.5"
            >
              <Filter className="h-4 w-4" />
              Filters
              {isFiltered && (
                <Badge variant="secondary" className="ml-1 h-5 px-1.5 text-xs">
                  {[
                    filters.site_ids?.length,
                    filters.date_from ? 1 : 0,
                    filters.date_to ? 1 : 0,
                    filters.species?.length,
                    filters.verification && filters.verification !== "all" ? 1 : 0,
                    filters.min_confidence !== undefined || filters.max_confidence !== undefined ? 1 : 0,
                  ].reduce((a: number, b) => a + (b || 0), 0)}
                </Badge>
              )}
            </Button>
          )}
        </div>

        {/* Filter panel */}
        <FilterPanel
          filters={filters}
          onChange={setFilters}
          projectId={projectId!}
          isOpen={filterPanelOpen}
          onToggle={() => setFilterPanelOpen((v) => !v)}
          classificationModelId={project?.classification_model_id}
        />

        {/* Filter chips */}
        {isFiltered && (
          <FilterChips
            filters={filters}
            onChange={setFilters}
            filteredCount={filteredEvents}
            totalCount={totalEvents}
            siteNames={siteNames}
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
                Page {page + 1}
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
  const badge = getObservationBadge(event.observation_type);

  const timeRange = sameTime
    ? startTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : `${startTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} – ${endTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;

  const dateStr = startTime.toLocaleDateString([], {
    month: "short",
    day: "numeric",
  });

  const verifiedPct =
    event.total_count > 0
      ? Math.round((event.verified_count / event.total_count) * 100)
      : 0;

  const thumbnailUrl = event.representative_file_id
    ? `${API_BASE_URL}/api/files/${event.representative_file_id}/image`
    : undefined;

  // Fetch representative file detections for overlay
  const { data: repFile } = useQuery({
    queryKey: ["file", event.representative_file_id],
    queryFn: () => filesApi.get(event.representative_file_id!),
    enabled: !!event.representative_file_id,
    staleTime: Infinity,
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
        {repFile && (() => {
          const dets = repFile.detections.filter(
            (d) => d.confidence >= detectionThreshold
          );
          if (dets.length === 0) return null;
          const imgW = repFile.width_px || 1;
          const imgH = repFile.height_px || 1;
          // Use a fixed reference size for 16:9 viewBox
          const VW = 320;
          const VH = 180;
          const scale = Math.max(VW / imgW, VH / imgH);
          const dw = imgW * scale;
          const dh = imgH * scale;
          const ox = (VW - dw) / 2;
          const oy = (VH - dh) / 2;
          let d = `M0,0H${VW}V${VH}H0Z`;
          const boxes = dets.map((det) => {
            const bx = ox + det.bbox_x * dw;
            const by = oy + det.bbox_y * dh;
            const bw = det.bbox_width * dw;
            const bh = det.bbox_height * dh;
            const color = getCategoryColor(det.category);
            d += `M${bx},${by}h${bw}v${bh}h${-bw}Z`;
            return { bx, by, bw, bh, color };
          });
          return (
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none"
              viewBox={`0 0 ${VW} ${VH}`}
            >
              <path fillRule="evenodd" d={d} fill="rgba(0,0,0,0.35)" />
              {boxes.map((b, i) => (
                <rect
                  key={i}
                  x={b.bx}
                  y={b.by}
                  width={b.bw}
                  height={b.bh}
                  rx={2}
                  fill="none"
                  stroke={b.color}
                  strokeWidth={1.5}
                  opacity={0.5}
                />
              ))}
            </svg>
          );
        })()}
        {/* Observation badge */}
        <Badge
          variant="outline"
          className={`absolute top-2 right-2 text-xs ${badge.className}`}
        >
          {badge.label}
        </Badge>
      </div>

      <CardContent className="p-3 space-y-2">
        {/* File count and time */}
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium">
            {event.file_count} {event.file_count === 1 ? "file" : "files"}
          </span>
          <span className="text-muted-foreground text-xs">{dateStr}</span>
        </div>
        <div className="text-xs text-muted-foreground">{timeRange}</div>

        {/* Species tags */}
        {event.species.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {event.species.slice(0, 3).map((sp) => (
              <Badge
                key={sp}
                variant="secondary"
                className="text-xs capitalize"
              >
                {sp}
              </Badge>
            ))}
            {event.species.length > 3 && (
              <Badge variant="secondary" className="text-xs">
                +{event.species.length - 3}
              </Badge>
            )}
          </div>
        )}

        {/* Verification progress bar */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              {event.verified_count}/{event.total_count} verified
            </span>
            <span>{verifiedPct}%</span>
          </div>
          <div className="h-1.5 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 rounded-full transition-all"
              style={{ width: `${verifiedPct}%` }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
