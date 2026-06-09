/**
 * Event detail modal — the Counts-page event viewer.
 *
 * Center: one focused image (the annotation canvas / video player) where
 * every tool acts, above a resizable wrapping-grid filmstrip of the event's
 * frames; drag the divider to trade focus size for scanning room, and click
 * a thumbnail to focus it. Left: the tool rail (draw, tag, zoom, brightness /
 * contrast / threshold, flag, like, download). Right: the event-level species
 * + count editor (EventCountPanel) with the single "Confirm" sign-off.
 * Per-detection label cleanup at scale lives on the Labels page.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  ChevronsRight,
  X,
  SquareDashed,
  Download,
  Flag,
  Heart,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  FolderOpen,
  Play,
  Tag,
} from "lucide-react";
import { ApiError } from "../../lib/api-client";
import { eventsApi } from "../../api/events";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { projectsApi } from "../../api/projects";
import { cn } from "../../lib/utils";
import { basename } from "../../lib/path-utils";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { useRevealInFolder } from "../../lib/file-reveal";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import type { EventFilterParams, FileWithDetections } from "../../api/types";
import { EventFilmstrip } from "./EventFilmstrip";
import { ViewControls } from "./ViewControls";
import type { TileSize } from "./CropGrid";
import { AnnotationCanvas } from "./AnnotationCanvas";
import { EventCountPanel } from "./EventCountPanel";
import { LabelPicker } from "./LabelPicker";
import { VideoPlayer, isPlayableVideo } from "./VideoPlayer";
import { useLabelOptions, type LabelOption } from "../../hooks/useLabelOptions";

interface EventDetailModalProps {
  eventId: string | null;
  projectId: string;
  isOpen: boolean;
  onClose: () => void;
  filters?: EventFilterParams;
}

export function EventDetailModal({
  eventId,
  projectId,
  isOpen,
  onClose,
  filters,
}: EventDetailModalProps) {
  const queryClient = useQueryClient();
  const revealInFolder = useRevealInFolder();
  const [selectedFileIndex, setSelectedFileIndex] = useState(0);
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(
    null
  );
  const [drawMode, setDrawMode] = useState(false);
  // Active species for both drawing new boxes AND adding event-level
  // observations. Sidebar Tag picker is the manual override; falls
  // back to `defaultActiveLabel` (most common in this event) when null.
  const [activeLabel, setActiveLabel] = useState<{ category: string; label: string | undefined } | null>(null);
  const [viewMode, setViewMode] = useState<"frame" | "video">("frame");
  // One-shot flag: set when Download is clicked on a video while in frame
  // view, so the VideoPlayer runs the annotated-video export once it mounts.
  const [pendingVideoExport, setPendingVideoExport] = useState(false);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [videoPopoverOpen, setVideoPopoverOpen] = useState(false);
  const [boxesHidden, setBoxesHidden] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [shortcutLabels, setShortcutLabels] = useState<Record<number, LabelOption>>({});
  const [relabelDetectionId, setRelabelDetectionId] = useState<string | null>(null);
  const [localThreshold, setLocalThreshold] = useState<number | null>(null);
  const [brightness, setBrightness] = useState(50);
  const [contrast, setContrast] = useState(50);

  // Filmstrip view settings, persisted per user: the resizable filmstrip
  // height (set by dragging the divider) and the S/M/L thumbnail size.
  const countsSettings = useMemo(() => {
    try {
      return JSON.parse(localStorage.getItem("addaxai:countsSettings") || "{}");
    } catch {
      return {};
    }
  }, []);
  const persistCountsSetting = useCallback((key: string, value: unknown) => {
    try {
      const cur = JSON.parse(
        localStorage.getItem("addaxai:countsSettings") || "{}",
      );
      cur[key] = value;
      localStorage.setItem("addaxai:countsSettings", JSON.stringify(cur));
    } catch {
      /* ignore */
    }
  }, []);
  const [filmstripHeight, setFilmstripHeight] = useState<number>(
    countsSettings.filmstripHeight ?? 220,
  );
  const [tileSize, _setTileSize] = useState<TileSize>(
    countsSettings.tileSize ?? "M",
  );
  const setTileSize = useCallback(
    (v: TileSize) => {
      _setTileSize(v);
      persistCountsSetting("tileSize", v);
    },
    [persistCountsSetting],
  );
  const centerColumnRef = useRef<HTMLDivElement>(null);

  // Drag the divider between the focus image and the filmstrip.
  const startDividerDrag = useCallback(
    (e: ReactMouseEvent) => {
      e.preventDefault();
      let latest = filmstripHeight;
      const onMove = (ev: MouseEvent) => {
        const el = centerColumnRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        latest = Math.max(
          120,
          Math.min(rect.height * 0.8, rect.bottom - ev.clientY),
        );
        setFilmstripHeight(latest);
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        persistCountsSetting("filmstripHeight", latest);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [filmstripHeight, persistCountsSetting],
  );

  const exportFnRef = useRef<(() => void) | null>(null);
  const zoomFnRef = useRef<{
    zoomIn: () => void;
    zoomOut: () => void;
    resetZoom: () => void;
    getZoom: () => number;
  } | null>(null);
  const [viewport, setViewport] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  // Fetch event data
  const { data: event, isError: eventError, error: eventErrorObj } = useQuery({
    queryKey: ["event", eventId],
    queryFn: () => eventsApi.get(eventId!),
    enabled: !!eventId && isOpen,
    // A re-run regenerates events with new ids, so a stale id 404s.
    // Don't retry that (it can't succeed); other errors keep one retry.
    retry: (failureCount, err) =>
      !(err instanceof ApiError && err.status === 404) && failureCount < 1,
  });

  // The event no longer exists (its id went stale after a re-run). Close
  // the modal and refresh the lists so the dead row disappears, instead
  // of leaving the user staring at an empty modal.
  useEffect(() => {
    if (
      eventError &&
      eventErrorObj instanceof ApiError &&
      eventErrorObj.status === 404
    ) {
      onClose();
      queryClient.invalidateQueries({ queryKey: ["events"] });
    }
  }, [eventError, eventErrorObj, onClose, queryClient]);

  // Fetch adjacent events for navigation (scoped to filtered set)
  const { data: adjacent } = useQuery({
    queryKey: ["event-adjacent", eventId, projectId, filters],
    queryFn: () => eventsApi.getAdjacent(eventId!, projectId, filters),
    enabled: !!eventId && isOpen,
  });

  // Fetch project for detection threshold
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  });

  // Fetch label options from classification model taxonomy
  const { options: labelOptions, isLoading: labelOptionsLoading } =
    useLabelOptions(project?.classification_model_id ?? null, projectId);

  // Track viewport size for responsive modal sizing
  useEffect(() => {
    const handleResize = () => {
      setViewport({ width: window.innerWidth, height: window.innerHeight });
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Load shortcut label mappings from project data
  useEffect(() => {
    if (project?.shortcut_labels) {
      const parsed: Record<number, LabelOption> = {};
      for (const [k, v] of Object.entries(project.shortcut_labels)) {
        parsed[Number(k)] = v as LabelOption;
      }
      setShortcutLabels(parsed);
    }
  }, [project?.shortcut_labels]);

  // Update shortcut labels in state and persist to backend
  const updateShortcutLabels = useCallback(
    (updater: (prev: Record<number, LabelOption>) => Record<number, LabelOption>) => {
      setShortcutLabels((prev) => {
        const next = updater(prev);
        projectsApi.update(projectId, { shortcut_labels: next });
        return next;
      });
    },
    [projectId]
  );

  // Event summary for the top card: media breakdown + date/time span. This
  // describes the event you're confirming, not the frame you happen to be on.
  const eventSummary = useMemo(() => {
    if (!event) return null;
    const videos = event.files.filter((f) => f.file_type === "video").length;
    const images = event.files.length - videos;
    const parts: string[] = [];
    if (images) parts.push(`${images} image${images > 1 ? "s" : ""}`);
    if (videos) parts.push(`${videos} video${videos > 1 ? "s" : ""}`);
    const media =
      parts.join(" · ") ||
      `${event.file_count} file${event.file_count === 1 ? "" : "s"}`;

    let when: string | null = null;
    if (event.event_start_local) {
      const dateOpts = { day: "numeric", month: "short", year: "numeric" } as const;
      const timeOpts = { hour: "2-digit", minute: "2-digit" } as const;
      const start = event.event_start_local;
      const end = event.event_end_local || start;
      const startDate = formatCameraDate(start, dateOpts, "en-GB");
      const startTime = formatCameraTime(start, timeOpts, "en-GB");
      const endTime = formatCameraTime(end, timeOpts, "en-GB");
      const sameDay = start.slice(0, 10) === end.slice(0, 10);
      if (sameDay) {
        when =
          startTime === endTime
            ? `${startDate}, ${startTime}`
            : `${startDate}, ${startTime} – ${endTime}`;
      } else {
        const endDate = formatCameraDate(end, dateOpts, "en-GB");
        when = `${startDate} ${startTime} – ${endDate} ${endTime}`;
      }
    }
    return { media, when };
  }, [event]);

  // On open or event change, focus the busiest frame: the image (or video
  // best frame) with the most detections, regardless of species. Gives
  // multi-species events one unambiguous landing frame.
  useEffect(() => {
    const fs = event?.files ?? [];
    const peakCount = (f: FileWithDetections) =>
      f.file_type === "video" && f.best_frame_number != null
        ? f.detections.filter((d) => d.frame_number === f.best_frame_number)
            .length
        : f.detections.length;
    let bestIdx = 0;
    let bestCount = -1;
    fs.forEach((f, i) => {
      const c = peakCount(f);
      if (c > bestCount) {
        bestCount = c;
        bestIdx = i;
      }
    });
    setSelectedFileIndex(bestIdx);
    setSelectedDetectionId(null);
    setViewMode("frame");
    setPendingVideoExport(false);
    setSelectedVideoId(null);
    setVideoPopoverOpen(false);
  }, [eventId, event?.id]);

  const files = event?.files ?? [];
  const currentFile = files[selectedFileIndex] as
    | FileWithDetections
    | undefined;
  const projectThreshold = project?.detection_threshold ?? 0;
  const detectionThreshold = localThreshold ?? projectThreshold;
  const imageFilter =
    brightness !== 50 || contrast !== 50
      ? `brightness(${brightness / 50}) contrast(${contrast / 50})`
      : undefined;

  // Derive list of video File rows from the event's files. Post-2026-05
  // refactor, each video File row carries all its detections directly
  // (with `frame_number` set per detection), so `frameCount` is the
  // number of distinct frames that produced ≥1 detection.
  const sourceVideos = useMemo(() => {
    const videos: { id: string; frameCount: number }[] = [];
    for (const f of files) {
      if (f.file_type !== "video") continue;
      const frameSet = new Set<number>();
      for (const d of f.detections) {
        if (d.frame_number != null) frameSet.add(d.frame_number);
      }
      videos.push({ id: f.id, frameCount: frameSet.size });
    }
    return videos;
  }, [files]);

  // For video files: hand the VideoPlayer every detection on that
  // file so boxes can render against the right frame during playback.
  const videoPlaybackProps = useMemo(() => {
    const videoId = selectedVideoId ?? (currentFile?.file_type === "video" ? currentFile.id : null);
    if (!videoId) return undefined;
    const videoFile = files.find((f) => f.id === videoId);
    if (!videoFile) return undefined;
    return {
      sourceVideoId: videoId,
      allDetections: videoFile.detections,
    };
  }, [currentFile, files, selectedVideoId]);

  // Default active label: most common species across all detections
  // in the event. The sidebar Tag picker uses this when the user
  // hasn't set a manual override. Same value drives Draw-box and
  // Add-Obs so both apply the same species.
  const defaultActiveLabel = useMemo(() => {
    if (!event?.files)
      return { category: "animal", label: undefined as string | undefined };

    const labelCounts = new Map<
      string,
      { count: number; category: string; label: string | undefined }
    >();

    for (const f of event.files) {
      for (const d of f.detections) {
        if (d.confidence >= detectionThreshold) {
          const key = d.label || d.category;
          const existing = labelCounts.get(key);
          if (existing) {
            existing.count++;
          } else {
            labelCounts.set(key, {
              count: 1,
              category: d.category,
              label: d.label || undefined,
            });
          }
        }
      }
    }

    let best = { category: "animal", label: undefined as string | undefined };
    let bestCount = 0;
    for (const entry of labelCounts.values()) {
      if (entry.count > bestCount) {
        bestCount = entry.count;
        best = { category: entry.category, label: entry.label };
      }
    }

    return best;
  }, [event?.files, detectionThreshold]);

  // Effective active label: manual override falls through to default.
  const effectiveActiveLabel = activeLabel ?? defaultActiveLabel;

  // Reset manual override when the event changes. Within one event we
  // keep the override across files so a "wolf" pick survives navigating
  // between files of the same encounter.
  useEffect(() => {
    setActiveLabel(null);
  }, [eventId]);

  // A stable, large modal: the focus + filmstrip both want the room, and
  // the focus canvas re-fits its container as the divider moves.
  const modalStyle = useMemo(
    () => ({
      width: Math.round(viewport.width * 0.95),
      height: Math.round(viewport.height * 0.95),
    }),
    [viewport],
  );

  // Event confirmation (the count panel's "Confirm"; also the Enter key).
  const eventConfirmMutation = useMutation({
    mutationFn: (confirmed: boolean) => {
      if (!event) return Promise.resolve(null);
      return eventsApi.setConfirmed(event.id, confirmed);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
      queryClient.invalidateQueries({ queryKey: ["events"] });
    },
  });


  // Like / flag live at the file level (the Event card badge lights up if
  // any file is set). Keyed by fileId.
  const favoriteMutation = useMutation({
    mutationFn: ({ fileId, favorited }: { fileId: string; favorited: boolean }) =>
      filesApi.update(fileId, { favorited }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
      queryClient.invalidateQueries({ queryKey: ["events"] });
      queryClient.invalidateQueries({ queryKey: ["file"] });
    },
  });
  const flagMutation = useMutation({
    mutationFn: ({ fileId, flagged }: { fileId: string; flagged: boolean }) =>
      filesApi.update(fileId, { flagged }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
      queryClient.invalidateQueries({ queryKey: ["events"] });
      queryClient.invalidateQueries({ queryKey: ["file"] });
    },
  });

  // Filtered detections for the current file (for Tab cycling)
  const filteredDetections = useMemo(() => {
    if (!currentFile) return [];
    let dets = currentFile.detections.filter(
      (d) => d.confidence >= detectionThreshold
    );
    // For videos in frame view, only include best-frame detections
    if (
      currentFile.file_type === "video" &&
      currentFile.best_frame_number != null &&
      viewMode === "frame"
    ) {
      dets = dets.filter((d) => d.frame_number === currentFile.best_frame_number);
    }
    return dets;
  }, [currentFile, detectionThreshold, viewMode]);

  // Delete selected detection mutation
  const deleteDetectionMutation = useMutation({
    mutationFn: (id: string) => {
      // Capture the next detection before deleting
      const idx = filteredDetections.findIndex((d) => d.id === id);
      const next =
        filteredDetections[idx + 1] ?? filteredDetections[idx - 1] ?? null;
      return detectionsApi.delete(id).then(() => next);
    },
    onSuccess: (next) => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
      queryClient.invalidateQueries({ queryKey: ["label-tree"] });
      setSelectedDetectionId(next?.id ?? null);
    },
  });

  // Relabel a box in the focus (driven by clicking its label pill / Tab).
  const relabelTargetDetection = relabelDetectionId
    ? currentFile?.detections.find((d) => d.id === relabelDetectionId) ?? null
    : null;
  const relabelMutation = useMutation({
    mutationFn: ({ id, option }: { id: string; option: LabelOption }) =>
      detectionsApi.update(id, {
        category: option.category,
        label: option.label ?? null,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
      queryClient.invalidateQueries({ queryKey: ["events"] });
      queryClient.invalidateQueries({ queryKey: ["file"] });
      queryClient.invalidateQueries({ queryKey: ["label-tree"] });
      setRelabelDetectionId(null);
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

  // Jump to the next event that still needs its species + counts confirmed.
  const handleNextUnconfirmed = useCallback(() => {
    navigateEvent(adjacent?.next_unconfirmed_id);
  }, [adjacent, navigateEvent]);

  // Download button. For a video, always produce the annotated VIDEO:
  // switch to video view (mounting the player) and flag a one-shot export
  // that runs once the clip is playable — even if the click came from the
  // frame view. For an image, export the annotated still PNG.
  const handleDownload = useCallback(() => {
    if (currentFile && isPlayableVideo(currentFile)) {
      setViewMode("video");
      setPendingVideoExport(true);
    } else {
      exportFnRef.current?.();
    }
  }, [currentFile]);

  // Click a thumbnail in the filmstrip: make it the focus. A video focuses
  // straight into playback.
  const handleSelectFile = useCallback(
    (index: number) => {
      setSelectedFileIndex(index);
      setSelectedDetectionId(null);
      const f = files[index];
      setViewMode(f && isPlayableVideo(f) ? "video" : "frame");
    },
    [files],
  );

  const prevDisabled = !adjacent?.previous_id;
  const nextDisabled = !adjacent?.next_id;
  const nextUnconfirmedDisabled = !adjacent?.next_unconfirmed_id;

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

      // Confirm + advance to the next unconfirmed event.
      if (e.key === "Enter") {
        e.preventDefault();
        if (event && !event.confirmed) {
          eventConfirmMutation
            .mutateAsync(true)
            .then(() => handleNextUnconfirmed());
        } else {
          handleNextUnconfirmed();
        }
        return;
      }

      switch (e.key) {
        case "ArrowUp":
          e.preventDefault();
          if (filteredDetections.length === 0) break;
          setSelectedDetectionId((prev) => {
            const currentIdx = prev
              ? filteredDetections.findIndex((d) => d.id === prev)
              : -1;
            const nextIdx =
              currentIdx <= 0
                ? filteredDetections.length - 1
                : currentIdx - 1;
            return filteredDetections[nextIdx].id;
          });
          break;
        case "ArrowDown":
          e.preventDefault();
          if (filteredDetections.length === 0) break;
          setSelectedDetectionId((prev) => {
            const currentIdx = prev
              ? filteredDetections.findIndex((d) => d.id === prev)
              : -1;
            const nextIdx =
              currentIdx < 0 || currentIdx >= filteredDetections.length - 1
                ? 0
                : currentIdx + 1;
            return filteredDetections[nextIdx].id;
          });
          break;
        case "ArrowLeft":
          e.preventDefault();
          if (selectedFileIndex > 0) {
            setSelectedFileIndex((i) => i - 1);
            setSelectedDetectionId(null);
          }
          break;
        case "ArrowRight":
          e.preventDefault();
          if (selectedFileIndex < files.length - 1) {
            setSelectedFileIndex((i) => i + 1);
            setSelectedDetectionId(null);
          }
          break;
        case "Tab":
          if (selectedDetectionId) {
            e.preventDefault();
            setRelabelDetectionId(selectedDetectionId);
          }
          break;
        case "p":
        case "P":
          if (currentFile && isPlayableVideo(currentFile)) {
            e.preventDefault();
            if (viewMode === "video") {
              // Toggling OFF — just switch back to frame mode
              setViewMode("frame");
              setVideoPopoverOpen(false);
            } else if (sourceVideos.length > 1) {
              // Multiple videos — open selector popover
              setVideoPopoverOpen(true);
            } else {
              // Single video — toggle directly
              setViewMode("video");
            }
          }
          break;
        case "f":
        case "F":
          e.preventDefault();
          if (currentFile)
            flagMutation.mutate({
              fileId: currentFile.id,
              flagged: !currentFile.flagged,
            });
          break;
        case "Delete":
        case "Backspace":
          if (selectedDetectionId) {
            e.preventDefault();
            deleteDetectionMutation.mutate(selectedDetectionId);
          }
          break;
        case "Escape":
          e.preventDefault();
          if (relabelDetectionId) {
            setRelabelDetectionId(null);
          } else if (selectedDetectionId) {
            setSelectedDetectionId(null);
          } else if (drawMode) {
            setDrawMode(false);
          } else {
            onClose();
          }
          break;
      }
    };

    // Register in capture phase so our preventDefault on Enter fires
    // before any focused button's implicit Enter-activates-click. Without
    // capture, clicking the > nav button grabs focus, and the next Enter
    // would re-fire that button's onClick (handleNext) instead of going
    // through the case "Enter" branch (verify + handleNextUnverified).
    window.addEventListener("keydown", handleKeyDown, true);
    return () => window.removeEventListener("keydown", handleKeyDown, true);
  }, [
    isOpen,
    event,
    currentFile,
    drawMode,
    filteredDetections,
    handleNextUnconfirmed,
    relabelDetectionId,
    onClose,
    selectedDetectionId,
    selectedFileIndex,
    files.length,
    eventConfirmMutation,
    flagMutation,
    deleteDetectionMutation,
    files,
    viewMode,
    sourceVideos,
  ]);

  // B key hold: momentarily hide boxes in the focus image (and, mirrored,
  // in the filmstrip thumbnails).
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }
      if ((e.key === "b" || e.key === "B") && !e.repeat) {
        setBoxesHidden(true);
      }
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === "b" || e.key === "B") {
        setBoxesHidden(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={() => onClose()}>
      <DialogContent
        className="flex flex-col p-0 pt-3 gap-0 overflow-hidden [&>button.absolute]:hidden"
        onOpenAutoFocus={(e) => e.preventDefault()}
        style={{
          width: modalStyle.width,
          height: modalStyle.height,
          maxWidth: "95vw",
          maxHeight: "95vh",
        }}
        aria-describedby={undefined}
      >
        <DialogTitle className="sr-only">Event detail viewer</DialogTitle>

        {/* Main content */}
        <div className="flex flex-1 min-h-0 overflow-hidden">
          {/* Left toolbar — tools that act on the focused image. */}
          {currentFile && (
            <div className="flex flex-col items-center gap-1 px-1.5 py-2 bg-white shrink-0">
              <Button
                variant={drawMode ? "default" : "ghost"}
                size="icon"
                className="h-8 w-8"
                onClick={() => setDrawMode(!drawMode)}
                title="Draw new box (D)"
              >
                <SquareDashed className="h-4 w-4" />
              </Button>
              {/* Active species picker — always visible. Sets the
                  species applied to a newly drawn box. Auto-defaults to
                  the event's most-common label; click to override. */}
              <div className="[&_button]:h-8 [&_button]:w-8 [&_button]:p-0 [&_button]:justify-center [&_svg]:opacity-100">
                <LabelPicker
                  value={effectiveActiveLabel.label || effectiveActiveLabel.category}
                  onSelect={(option) =>
                    setActiveLabel({ category: option.category, label: option.label ?? undefined })
                  }
                  options={labelOptions}
                  isLoading={labelOptionsLoading}
                  pinnedOptions={Object.entries(shortcutLabels).map(([k, v]) => ({
                    key: Number(k),
                    option: v,
                  }))}
                  hideDot
                  hideLabel
                  projectId={projectId}
                  triggerIcon={Tag}
                  triggerTitle={
                    effectiveActiveLabel.label
                      ? `Set label for new boxes · current: ${
                          effectiveActiveLabel.label.charAt(0).toUpperCase() +
                          effectiveActiveLabel.label.slice(1).replace(/[_-]+/g, " ")
                        }`
                      : "Set label for new boxes · not set (defaults to animal)"
                  }
                />
              </div>
              {/* Video toggle (for video file rows) */}
              {currentFile.file_type === "video" && (
                sourceVideos.length > 1 ? (
                  <Popover open={videoPopoverOpen} onOpenChange={setVideoPopoverOpen}>
                    <PopoverTrigger asChild>
                      <Button
                        variant={viewMode === "video" ? "default" : "ghost"}
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => {
                          if (viewMode === "video") {
                            setViewMode("frame");
                            setVideoPopoverOpen(false);
                          }
                          // Opening is handled by Popover onOpenChange
                        }}
                        disabled={!isPlayableVideo(currentFile)}
                        title={
                          !isPlayableVideo(currentFile)
                            ? "Video format not supported for browser playback"
                            : viewMode === "video"
                              ? "View frame"
                              : "Play video"
                        }
                      >
                        <Play className="h-4 w-4" />
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent side="right" className="w-48 p-2">
                      <div className="space-y-1">
                        <p className="text-xs font-medium text-muted-foreground px-2 pb-1">
                          Select video
                        </p>
                        {sourceVideos.map((sv, i) => (
                          <button
                            key={sv.id}
                            className="w-full text-left text-sm px-2 py-1.5 rounded hover:bg-accent transition-colors"
                            onClick={() => {
                              setSelectedVideoId(sv.id);
                              const videoFileIndex = files.findIndex(
                                (f) => f.id === sv.id
                              );
                              if (videoFileIndex >= 0)
                                setSelectedFileIndex(videoFileIndex);
                              setVideoPopoverOpen(false);
                              setViewMode("video");
                            }}
                          >
                            Video {i + 1}{" "}
                            <span className="text-muted-foreground">
                              ({sv.frameCount} frame{sv.frameCount !== 1 ? "s" : ""})
                            </span>
                          </button>
                        ))}
                      </div>
                    </PopoverContent>
                  </Popover>
                ) : (
                  <Button
                    variant={viewMode === "video" ? "default" : "ghost"}
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setViewMode(viewMode === "video" ? "frame" : "video")}
                    disabled={!isPlayableVideo(currentFile)}
                    title={
                      !isPlayableVideo(currentFile)
                        ? "Video format not supported for browser playback"
                        : viewMode === "video"
                          ? "View frame"
                          : "Play video"
                    }
                  >
                    <Play className="h-4 w-4" />
                  </Button>
                )
              )}
              <div className="w-6 border-t my-0.5" />
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => zoomFnRef.current?.zoomIn()}
                title="Zoom in"
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => zoomFnRef.current?.zoomOut()}
                title="Zoom out"
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => zoomFnRef.current?.resetZoom()}
                title="Reset zoom"
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
              <div className="w-6 border-t my-0.5" />
              <ViewControls
                detectionThreshold={detectionThreshold}
                projectThreshold={projectThreshold}
                localThreshold={localThreshold}
                onThresholdChange={setLocalThreshold}
                brightness={brightness}
                onBrightnessChange={setBrightness}
                contrast={contrast}
                onContrastChange={setContrast}
              />
              <div className="w-6 border-t my-0.5" />
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() =>
                  favoriteMutation.mutate({
                    fileId: currentFile.id,
                    favorited: !currentFile.favorited,
                  })
                }
                disabled={favoriteMutation.isPending}
                title={currentFile.favorited ? "Unlike" : "Like"}
              >
                <Heart
                  className={cn(
                    "h-4 w-4",
                    currentFile.favorited && "fill-[#882000] text-[#882000]"
                  )}
                />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() =>
                  flagMutation.mutate({
                    fileId: currentFile.id,
                    flagged: !currentFile.flagged,
                  })
                }
                disabled={flagMutation.isPending}
                title={currentFile.flagged ? "Remove flag" : "Flag for review (F)"}
              >
                <Flag
                  className={cn(
                    "h-4 w-4",
                    currentFile.flagged && "fill-[#71b7ba] text-[#71b7ba]"
                  )}
                />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={handleDownload}
                title={
                  currentFile && isPlayableVideo(currentFile)
                    ? "Download annotated video"
                    : "Download annotated image"
                }
              >
                <Download className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => revealInFolder(currentFile)}
                title="Open in file explorer"
              >
                <FolderOpen className="h-4 w-4" />
              </Button>
            </div>
          )}

          {/* Center column: focused image + resizable filmstrip below. */}
          <div ref={centerColumnRef} className="flex-1 flex flex-col min-w-0">
            {/* Focused image / video — all tools act here. */}
            <div className="flex-1 flex items-center justify-center bg-black/95 min-h-0 p-2 rounded-lg">
              {currentFile ? (
                viewMode === "video" && isPlayableVideo(currentFile) ? (
                  <VideoPlayer
                    file={currentFile}
                    detectionThreshold={detectionThreshold}
                    sourceVideoId={videoPlaybackProps?.sourceVideoId}
                    allDetections={videoPlaybackProps?.allDetections}
                    exportFnRef={exportFnRef}
                    autoExport={pendingVideoExport}
                    onAutoExportConsumed={() => setPendingVideoExport(false)}
                  />
                ) : (
                  <AnnotationCanvas
                    file={currentFile}
                    detectionThreshold={detectionThreshold}
                    selectedDetectionId={selectedDetectionId}
                    onSelectDetection={setSelectedDetectionId}
                    onRequestRelabel={setRelabelDetectionId}
                    drawMode={drawMode}
                    onDrawModeChange={setDrawMode}
                    onMutated={() => {
                      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
                      queryClient.invalidateQueries({ queryKey: ["events"] });
                      queryClient.invalidateQueries({ queryKey: ["file"] });
                      queryClient.invalidateQueries({ queryKey: ["label-tree"] });
                    }}
                    imageFilter={imageFilter}
                    defaultCategory={effectiveActiveLabel.category}
                    defaultLabel={effectiveActiveLabel.label}
                    boxesHidden={boxesHidden}
                    exportFnRef={exportFnRef}
                    zoomFnRef={zoomFnRef}
                  />
                )
              ) : (
                <div className="text-white/50">Loading...</div>
              )}
            </div>

            {/* Draggable divider between the focus and the filmstrip. */}
            <div
              onMouseDown={startDividerDrag}
              className="h-1.5 shrink-0 cursor-row-resize bg-border transition-colors hover:bg-primary/40"
              title="Drag to resize the filmstrip"
            />

            {/* Resizable filmstrip grid; thumbnails mirror the focus. */}
            <div style={{ height: filmstripHeight }} className="shrink-0 min-h-0">
              <EventFilmstrip
                files={files}
                selectedIndex={selectedFileIndex}
                onSelectFile={handleSelectFile}
                detectionThreshold={detectionThreshold}
                showBoxes={!boxesHidden}
                imageFilter={imageFilter}
                tileSize={tileSize}
                onTileSizeChange={setTileSize}
              />
            </div>
          </div>

          {/* Right sidebar: navigation + verification panel */}
          <div className="w-80 bg-white flex flex-col shrink-0">
            <div className="flex items-center justify-between px-3 py-1.5 shrink-0">
              <div className="flex items-center gap-0.5">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  disabled={prevDisabled}
                  onClick={() => navigateEvent(adjacent?.previous_id)}
                  title="Previous event (←)"
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  disabled={nextDisabled}
                  onClick={() => navigateEvent(adjacent?.next_id)}
                  title="Next event (→)"
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  disabled={nextUnconfirmedDisabled}
                  onClick={handleNextUnconfirmed}
                  title="Next unconfirmed event"
                >
                  <ChevronsRight className="h-4 w-4" />
                </Button>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={onClose}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Event summary — what you're confirming (media, when, where).
                The muted last line names the frame you're currently viewing. */}
            {event && eventSummary && (
              <div className="mx-3 mt-3 rounded-lg border bg-muted/40 px-3 py-2 space-y-0.5 text-xs text-muted-foreground">
                <div className="font-medium text-foreground">
                  {eventSummary.media}
                </div>
                {eventSummary.when && <div>{eventSummary.when}</div>}
                {event.site_name && <div>{event.site_name}</div>}
                {currentFile && (
                  <div className="truncate pt-0.5 text-[11px] text-muted-foreground/70">
                    viewing {basename(currentFile.file_path)}
                  </div>
                )}
              </div>
            )}

            {/* Event-level species + count editor (the ecological record). */}
            {event && (
              <EventCountPanel
                eventId={event.id}
                projectId={projectId}
                observations={event.observations}
                confirmed={event.confirmed}
                labelOptions={labelOptions}
                labelOptionsLoading={labelOptionsLoading}
              />
            )}

            {/* In-image relabel: clicking a box's label opens this dialog
                for that detection (the box stays highlighted). */}
            {relabelDetectionId && (
              <LabelPicker
                headless
                forceOpen
                value={relabelTargetDetection?.label || relabelTargetDetection?.category || null}
                options={labelOptions}
                isLoading={labelOptionsLoading}
                projectId={projectId}
                pinnedOptions={Object.entries(shortcutLabels).map(([k, v]) => ({
                  key: Number(k),
                  option: v,
                }))}
                onSelect={(option) =>
                  relabelMutation.mutate({ id: relabelDetectionId, option })
                }
                onOpenChange={(open) => {
                  if (!open) setRelabelDetectionId(null);
                }}
              />
            )}

            {/* Keyboard shortcuts */}
            <div className="mt-auto shrink-0 px-3 pb-2 relative">
              {showShortcuts && (
                <>
                <div className="fixed inset-0 z-40" onClick={() => setShowShortcuts(false)} />
                <div className="absolute bottom-10 right-6 mb-2 rounded-lg border bg-background shadow-lg px-4 py-3 z-50 whitespace-nowrap">
                  <div className="flex gap-8">
                    {/* Left column: navigation */}
                    <div>
                      <div className="text-[11px] font-semibold text-muted-foreground mb-1">Navigate</div>
                      {[
                        ["Enter", "Confirm + next event"],
                        ["← →", "Prev / next frame"],
                        ["Click", "Focus a thumbnail"],
                        ["B (hold)", "Hide boxes"],
                        ["Esc", "Close"],
                      ].map(([key, action]) => (
                        <div key={key} className="flex items-center text-xs gap-3 h-7">
                          <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">{key.split("+").map((part, i, arr) => <span key={i}>{part}{i < arr.length - 1 && <span className="text-[#bbbbc1]">+</span>}</span>)}</code>
                          <span>{action}</span>
                        </div>
                      ))}
                    </div>

                    {/* Right column: editing the focused image */}
                    <div>
                      <div className="text-[11px] font-semibold text-muted-foreground mb-1">Edit the focus</div>
                      {[
                        ["↑ ↓", "Select box"],
                        ["Click label", "Relabel box"],
                        ["Tab", "Relabel selected box"],
                        ["D", "Draw a box"],
                        ["Del", "Delete box"],
                        ["P", "Play video"],
                        ["F", "Flag / unflag"],
                      ].map(([key, action]) => (
                        <div key={key} className="flex items-center text-xs gap-3 h-7">
                          <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">{key.split("+").map((part, i, arr) => <span key={i}>{part}{i < arr.length - 1 && <span className="text-[#bbbbc1]">+</span>}</span>)}</code>
                          <span>{action}</span>
                        </div>
                      ))}

                      <div className="border-t my-2" />
                      <div className="text-[11px] font-semibold text-muted-foreground mb-1">Quick labels (for the box pickers)</div>
                      {[1, 2, 3, 4, 5].map((n) => (
                        <div key={n} className="flex items-center text-xs gap-3 h-7">
                          <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">{n}</code>
                          <span>Pin label</span>
                          <LabelPicker
                            value={shortcutLabels[n]?.value ?? null}
                            onSelect={(option) =>
                              updateShortcutLabels((prev) => ({ ...prev, [n]: option }))
                            }
                            options={labelOptions}
                            isLoading={labelOptionsLoading}
                            projectId={projectId}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                </>
              )}
              <button
                onClick={() => setShowShortcuts((s) => !s)}
                className="w-full py-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
              >
                {showShortcuts ? "Hide" : "Show"} keyboard shortcuts
              </button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
