/**
 * Event detail modal - full-screen event viewer with filmstrip navigation.
 *
 * Shows the selected event's images with interactive annotation canvas,
 * filmstrip for multi-file navigation, verification panel, and
 * event-to-event navigation.
 */

import { useCallback, useEffect, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  ChevronsRight,
  X,
  Scale,
  Sun,
  Contrast,
} from "lucide-react";
import { eventsApi } from "../../api/events";
import { filesApi } from "../../api/files";
import { projectsApi } from "../../api/projects";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog";
import { Slider } from "../ui/slider";
import type { FileWithDetections } from "../../api/types";
import { EventFilmstrip } from "./EventFilmstrip";
import { AnnotationCanvas } from "./AnnotationCanvas";
import { FileVerificationPanel } from "./FileVerificationPanel";

interface EventDetailModalProps {
  eventId: string | null;
  projectId: string;
  isOpen: boolean;
  onClose: () => void;
}

export function EventDetailModal({
  eventId,
  projectId,
  isOpen,
  onClose,
}: EventDetailModalProps) {
  const queryClient = useQueryClient();
  const [selectedFileIndex, setSelectedFileIndex] = useState(0);
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(
    null
  );
  const [drawMode, setDrawMode] = useState(false);
  const [localThreshold, setLocalThreshold] = useState<number | null>(null);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);

  // Fetch event data
  const { data: event } = useQuery({
    queryKey: ["event", eventId],
    queryFn: () => eventsApi.get(eventId!),
    enabled: !!eventId && isOpen,
  });

  // Fetch adjacent events for navigation
  const { data: adjacent } = useQuery({
    queryKey: ["event-adjacent", eventId, projectId],
    queryFn: () => eventsApi.getAdjacent(eventId!, projectId),
    enabled: !!eventId && isOpen,
  });

  // Fetch project for detection threshold
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  });

  // Reset file index when event changes
  useEffect(() => {
    setSelectedFileIndex(0);
    setSelectedDetectionId(null);
  }, [eventId]);

  const files = event?.files ?? [];
  const currentFile = files[selectedFileIndex] as
    | FileWithDetections
    | undefined;
  const projectThreshold = project?.detection_threshold ?? 0;
  const detectionThreshold = localThreshold ?? projectThreshold;
  const imageFilter =
    brightness !== 100 || contrast !== 100
      ? `brightness(${brightness / 100}) contrast(${contrast / 100})`
      : undefined;

  // Verify current file mutation
  const verifyMutation = useMutation({
    mutationFn: () => {
      if (!currentFile) return Promise.resolve(null);
      return filesApi.update(currentFile.id, {
        verified: !currentFile.verified,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
      queryClient.invalidateQueries({ queryKey: ["events"] });
    },
  });

  // Navigation handlers
  const navigateEvent = useCallback(
    (targetEventId: string | null | undefined) => {
      if (!targetEventId) return;
      window.dispatchEvent(
        new CustomEvent("navigate-event", { detail: targetEventId })
      );
    },
    []
  );

  // Keyboard shortcuts
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't fire when typing in inputs
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (e.key) {
        case "ArrowLeft":
          e.preventDefault();
          if (adjacent?.previous_id) navigateEvent(adjacent.previous_id);
          break;
        case "ArrowRight":
          e.preventDefault();
          if (adjacent?.next_id) navigateEvent(adjacent.next_id);
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedFileIndex((i) => Math.max(0, i - 1));
          break;
        case "ArrowDown":
          e.preventDefault();
          setSelectedFileIndex((i) => Math.min(files.length - 1, i + 1));
          break;
        case "n":
        case "N":
          e.preventDefault();
          if (adjacent?.next_unverified_id)
            navigateEvent(adjacent.next_unverified_id);
          break;
        case "v":
        case "V":
          e.preventDefault();
          verifyMutation.mutate();
          break;
        case "Escape":
          e.preventDefault();
          if (selectedDetectionId) {
            setSelectedDetectionId(null);
          } else {
            onClose();
          }
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    isOpen,
    adjacent,
    files.length,
    navigateEvent,
    onClose,
    selectedDetectionId,
    verifyMutation,
  ]);

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={() => onClose()}>
      <DialogContent
        className="max-w-[95vw] max-h-[95vh] w-full h-[95vh] p-0 gap-0 overflow-hidden [&>button.absolute]:hidden"
        aria-describedby={undefined}
      >
        <DialogTitle className="sr-only">Event detail viewer</DialogTitle>
        {/* Navigation bar */}
        <div className="flex items-center justify-between px-4 py-2 border-b bg-white shrink-0">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-muted-foreground">
              {adjacent
                ? `Event ${adjacent.current_index + 1} of ${adjacent.total_count}`
                : "Loading..."}
            </span>

            {/* Session-local sliders */}
            <div className="flex items-center gap-4 ml-4">
              {/* Detection threshold */}
              <div className="flex items-center gap-1.5" title="Detection threshold">
                <Scale className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                <Slider
                  value={[detectionThreshold]}
                  onValueChange={([v]) => setLocalThreshold(v)}
                  min={0}
                  max={1}
                  step={0.05}
                  className="w-24"
                />
                <span className="text-xs text-muted-foreground w-10 tabular-nums">
                  {(detectionThreshold * 100).toFixed(0)}%
                </span>
              </div>

              {/* Brightness */}
              <div className="flex items-center gap-1.5" title="Brightness">
                <Sun className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                <Slider
                  value={[brightness]}
                  onValueChange={([v]) => setBrightness(v)}
                  min={50}
                  max={200}
                  step={5}
                  className="w-24"
                />
                <span className="text-xs text-muted-foreground w-10 tabular-nums">
                  {brightness}%
                </span>
              </div>

              {/* Contrast */}
              <div className="flex items-center gap-1.5" title="Contrast">
                <Contrast className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                <Slider
                  value={[contrast]}
                  onValueChange={([v]) => setContrast(v)}
                  min={50}
                  max={200}
                  step={5}
                  className="w-24"
                />
                <span className="text-xs text-muted-foreground w-10 tabular-nums">
                  {contrast}%
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              disabled={!adjacent?.previous_id}
              onClick={() => navigateEvent(adjacent?.previous_id)}
              title="Previous event (Left arrow)"
            >
              <ChevronLeft className="h-4 w-4" />
              Prev
            </Button>
            <Button
              variant="ghost"
              size="sm"
              disabled={!adjacent?.next_id}
              onClick={() => navigateEvent(adjacent?.next_id)}
              title="Next event (Right arrow)"
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              disabled={!adjacent?.next_unverified_id}
              onClick={() => navigateEvent(adjacent?.next_unverified_id)}
              title="Next unverified event (N)"
            >
              <ChevronsRight className="h-4 w-4" />
              Unverified
            </Button>

            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="ml-2"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Main content */}
        <div className="flex flex-1 min-h-0 overflow-hidden">
          {/* Image area */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Main image with detections */}
            <div className="flex-1 flex items-center justify-center bg-black/95 min-h-0 p-2">
              {currentFile ? (
                <AnnotationCanvas
                  file={currentFile}
                  detectionThreshold={detectionThreshold}
                  eventId={eventId!}
                  selectedDetectionId={selectedDetectionId}
                  onSelectDetection={setSelectedDetectionId}
                  drawMode={drawMode}
                  onDrawModeChange={setDrawMode}
                  imageFilter={imageFilter}
                />
              ) : (
                <div className="text-white/50">Loading...</div>
              )}
            </div>

            {/* Filmstrip */}
            {files.length > 0 && (
              <EventFilmstrip
                files={files}
                selectedIndex={selectedFileIndex}
                onSelectIndex={setSelectedFileIndex}
              />
            )}
          </div>

          {/* Verification panel */}
          {currentFile && (
            <FileVerificationPanel
              file={currentFile}
              projectId={projectId}
              eventId={eventId!}
              drawMode={drawMode}
              onDrawModeChange={setDrawMode}
              detectionThreshold={detectionThreshold}
            />
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
