/**
 * Browse and verify page - event-centric browse and annotation workflow.
 *
 * Displays events as cards with thumbnails, species tags, and verification progress.
 * Clicking a card opens the event detail modal (Phase 2).
 */

import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Loader2, Layers } from "lucide-react";
import { eventsApi } from "../api/events";
import { API_BASE_URL } from "../lib/api-client";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { getObservationBadge } from "../lib/detection-utils";
import type { EventSummary } from "../api/types";
import { EventDetailModal } from "../components/verify/EventDetailModal";

const PAGE_SIZE = 50;

export default function VerifyPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [page, setPage] = useState(0);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);

  // Listen for navigation events from the modal
  useEffect(() => {
    const handler = (e: Event) => {
      const targetId = (e as CustomEvent).detail;
      if (targetId) setSelectedEventId(targetId);
    };
    window.addEventListener("navigate-event", handler);
    return () => window.removeEventListener("navigate-event", handler);
  }, []);

  // Get event count
  const { data: countData } = useQuery({
    queryKey: ["event-count", projectId],
    queryFn: () => eventsApi.count(projectId!),
    enabled: !!projectId,
  });

  // Get events
  const {
    data: events,
    isLoading,
    isFetching,
  } = useQuery({
    queryKey: ["events", projectId, page],
    queryFn: () =>
      eventsApi.list({
        project_id: projectId!,
        skip: page * PAGE_SIZE,
        limit: PAGE_SIZE,
      }),
    enabled: !!projectId,
  });

  const totalEvents = countData?.count ?? 0;
  const hasMore = events && events.length === PAGE_SIZE;

  return (
    <div className="p-8 bg-gradient-to-br from-slate-50 to-slate-100 min-h-screen">
      <div className="mx-auto max-w-7xl space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Browse and verify
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {totalEvents > 0
              ? `${totalEvents} events`
              : "Run a deployment analysis to get started"}
          </p>
        </div>

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
                No events yet
              </p>
              <p className="text-sm text-muted-foreground mt-1 max-w-md">
                Events are generated automatically when you run a deployment
                analysis. They group your camera trap images by time based on
                the project's independence interval.
              </p>
            </CardContent>
          </Card>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {events.map((event) => (
                <EventCard
                  key={event.id}
                  event={event}
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
        />
      </div>
    </div>
  );
}

function EventCard({
  event,
  onClick,
}: {
  event: EventSummary;
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
