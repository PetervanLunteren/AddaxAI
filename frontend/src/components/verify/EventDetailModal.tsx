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
  Download,
  Flag,
  Heart,
  FolderOpen,
  Play,
  Pause,
  Repeat,
  MoreVertical,
  Eye,
  EyeOff,
  RotateCcw,
} from "lucide-react";
import { toast } from "sonner";
import { ApiError } from "../../lib/api-client";
import { eventsApi } from "../../api/events";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { projectsApi } from "../../api/projects";
import { cn } from "../../lib/utils";
import { basename } from "../../lib/path-utils";
import { describeEventMedia } from "../../lib/event-media";
import {
  formatCameraDate,
  formatCameraTime,
  formatTimeOffset,
} from "../../lib/datetime";
import { useRevealInFolder } from "../../lib/file-reveal";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import type {
  EventFilterParams,
  EventWithFiles,
  FileWithDetections,
} from "../../api/types";
import { EventFilmstrip } from "./EventFilmstrip";
import { ViewControls } from "./ViewControls";
import type { TileSize } from "./CropGrid";
import { AnnotationCanvas } from "./AnnotationCanvas";
import { EventCountPanel } from "./EventCountPanel";
import { LabelPicker } from "./LabelPicker";
import { VideoPlayer, isPlayableVideo } from "./VideoPlayer";
import { VideoFilmstrip } from "./VideoFilmstrip";
import { useLabelOptions } from "../../hooks/useLabelOptions";

// Minimum gap between Shift+wheel frame steps, so a trackpad's burst of
// wheel events per swipe scrubs at a steady ~8 frames/second.
const SCRUB_THROTTLE_MS = 120;

// Auto-play cadence: how long each frame is shown before advancing (2 fps).
const AUTOPLAY_INTERVAL_MS = 500;

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
  const [viewMode, setViewMode] = useState<"frame" | "video">("frame");
  // Auto-play: cine-loop the event's frames on a timer. A ref mirrors the
  // value so the busiest-frame open effect can read it without re-running.
  const [autoPlay, setAutoPlay] = useState(false);
  const autoPlayRef = useRef(false);
  useEffect(() => {
    autoPlayRef.current = autoPlay;
  }, [autoPlay]);
  // Brief "looped back to the start" cue, flashed when auto-play wraps to 0.
  // A nonce that bumps on each wrap so the overlay re-mounts and re-animates
  // (and the CSS fade self-hides it). indexRef lets the interval see the
  // latest frame without re-subscribing.
  const [restartFlash, setRestartFlash] = useState(0);
  const indexRef = useRef(0);
  useEffect(() => {
    indexRef.current = selectedFileIndex;
  }, [selectedFileIndex]);
  // One-shot flag: set when Download is clicked on a video while in frame
  // view, so the VideoPlayer runs the annotated-video export once it mounts.
  const [pendingVideoExport, setPendingVideoExport] = useState(false);
  // When set, the relabel LabelPicker is open for this detection (clicked
  // its label pill on the focus image). Counts-page in-place single-box
  // relabel; per-detection cleanup at scale still lives on the Labels page.
  const [relabelDetectionId, setRelabelDetectionId] = useState<string | null>(
    null,
  );
  const [boxesHidden, setBoxesHidden] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  // "More" menu open state, so picking an item closes it (menu behaviour).
  const [moreOpen, setMoreOpen] = useState(false);
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

  // Prefetch the next event and, if it has a video, its filmstrip, so moving
  // forward through observations feels instant (the filmstrip decode is the
  // slow part — see VideoFilmstrip / the backend's build_filmstrip).
  useEffect(() => {
    const nextId = adjacent?.next_id;
    if (!nextId || !isOpen) return;
    let cancelled = false;
    queryClient
      .prefetchQuery({
        queryKey: ["event", nextId],
        queryFn: () => eventsApi.get(nextId),
      })
      .then(() => {
        if (cancelled) return;
        const nextEvent = queryClient.getQueryData<EventWithFiles>([
          "event",
          nextId,
        ]);
        const video = nextEvent?.files.find((f) => f.file_type === "video");
        if (video) {
          queryClient.prefetchQuery({
            queryKey: ["filmstrip", video.id],
            queryFn: () => filesApi.getFilmstrip(video.id),
            staleTime: Infinity,
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [adjacent?.next_id, isOpen, queryClient]);

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

  // Event summary for the top card: media breakdown + date/time span. This
  // describes the event you're confirming, not the frame you happen to be on.
  const eventSummary = useMemo(() => {
    if (!event) return null;
    const media = describeEventMedia(event.files);

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
    // Auto-play restarts each event from the beginning; otherwise land on the
    // busiest frame.
    setSelectedFileIndex(autoPlayRef.current ? 0 : bestIdx);
    setViewMode("frame");
    setPendingVideoExport(false);
    setRelabelDetectionId(null);
  }, [eventId, event?.id]);

  const files = event?.files ?? [];
  const currentFile = files[selectedFileIndex] as
    | FileWithDetections
    | undefined;
  // Time gap from the previous frame to the focused one (same signal as the
  // filmstrip's per-tile labels); null on the first frame or missing times.
  const focusGap = (() => {
    const cur = currentFile?.captured_at_local;
    const prev =
      selectedFileIndex > 0
        ? files[selectedFileIndex - 1]?.captured_at_local
        : null;
    if (!cur || !prev) return null;
    return formatTimeOffset(
      (new Date(cur).getTime() - new Date(prev).getTime()) / 1000,
    );
  })();
  const detectionThreshold = project?.detection_threshold ?? 0;
  const imageFilter =
    brightness !== 50 || contrast !== 50
      ? `brightness(${brightness / 50}) contrast(${contrast / 50})`
      : undefined;

  // The detection whose label pill was clicked (pills only render for the
  // focused file), seeding the relabel picker's current value.
  const relabelDetection =
    relabelDetectionId != null
      ? currentFile?.detections.find((d) => d.id === relabelDetectionId)
      : undefined;

  // For the focused video file: hand the VideoPlayer every detection on
  // that file so boxes can render against the right frame during playback.
  // Each video is its own filmstrip tile, so the focused file is always
  // the one to play — no separate video selection.
  const videoPlaybackProps = useMemo(() => {
    if (currentFile?.file_type !== "video") return undefined;
    return {
      sourceVideoId: currentFile.id,
      allDetections: currentFile.detections,
    };
  }, [currentFile]);

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

  // In-place single-box relabel: clicking a label pill on the focus image
  // relabels that one detection through the same path the Labels page uses
  // (detectionsApi.bulkRelabel with a one-id array). The backend re-derives
  // the event's MaxN and clears Event.confirmed if the species/count set
  // changed, so we just refetch. Mirrors the Labels page's invalidation set
  // (LabelsTab) so the filter tree / cohorts / Labels grid stay consistent.
  const relabelMutation = useMutation({
    mutationFn: ({
      id,
      label,
      category,
    }: {
      id: string;
      label: string | null;
      category: string;
    }) => detectionsApi.bulkRelabel([id], label, category),
    onSuccess: () => {
      setRelabelDetectionId(null);
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
      queryClient.invalidateQueries({ queryKey: ["events"] });
      queryClient.invalidateQueries({ queryKey: ["label-tree"] });
      queryClient.invalidateQueries({ queryKey: ["cohorts", projectId] });
    },
    onError: (err: Error) => toast.error(err.message),
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

  // Confirm the event (if not already) and jump to the next unconfirmed one.
  // Shared by the Enter key and the count panel's Confirm button.
  const handleConfirmAndAdvance = useCallback(() => {
    if (event && !event.confirmed) {
      eventConfirmMutation.mutateAsync(true).then(handleNextUnconfirmed);
    } else {
      handleNextUnconfirmed();
    }
  }, [event, eventConfirmMutation, handleNextUnconfirmed]);

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
      setAutoPlay(false); // manual navigation takes over from auto-play
      setSelectedFileIndex(index);
      const f = files[index];
      setViewMode(f && isPlayableVideo(f) ? "video" : "frame");
    },
    [files],
  );

  // Shift+wheel over the focus scrubs through frames (stills, so scrubbing
  // past a video does not start playback). Throttled to a steady pace: a
  // trackpad fires a burst of wheel events per swipe, so without this one
  // flick would jump dozens of frames. Magnitude is ignored, so the speed
  // feels the same on a mouse wheel and a trackpad.
  const lastScrubRef = useRef(0);
  const handleScrubFrame = useCallback(
    (delta: number) => {
      const now = performance.now();
      if (now - lastScrubRef.current < SCRUB_THROTTLE_MS) return;
      lastScrubRef.current = now;
      setAutoPlay(false); // manual navigation takes over from auto-play
      const dir = delta > 0 ? 1 : -1;
      setSelectedFileIndex((i) =>
        Math.max(0, Math.min(files.length - 1, i + dir)),
      );
      setViewMode("frame");
    },
    [files.length],
  );

  // Auto-play toggle. Turning it on restarts the playthrough from frame 0.
  const toggleAutoPlay = useCallback(() => {
    if (!autoPlayRef.current) {
      setSelectedFileIndex(0);
      setViewMode("frame");
    }
    setAutoPlay((on) => !on);
  }, []);

  // Watch the focused video: swap the best-frame still for the real player.
  // Stops the cine-loop first (you've grabbed control of one clip).
  const playFocusedVideo = useCallback(() => {
    setAutoPlay(false);
    setViewMode("video");
  }, []);

  // The loop: advance one frame each tick, wrapping to the start. Flash the
  // restart cue on the wrap (last frame -> 0), not on the first pass.
  useEffect(() => {
    if (!autoPlay || !isOpen || files.length <= 1) return;
    const id = setInterval(() => {
      const next = (indexRef.current + 1) % files.length;
      if (next === 0) setRestartFlash((n) => n + 1);
      setSelectedFileIndex(next);
    }, AUTOPLAY_INTERVAL_MS);
    return () => clearInterval(id);
  }, [autoPlay, isOpen, files.length]);

  // Warm the next frame's full-res image one step ahead so the loop stays
  // smooth on the first pass (same URL AnnotationCanvas fetches, so the
  // browser cache is shared).
  const preloadFileId =
    autoPlay && files.length > 1
      ? (files[(selectedFileIndex + 1) % files.length]?.id ?? null)
      : null;
  useEffect(() => {
    if (!preloadFileId) return;
    const img = new Image();
    img.src = `/api/files/${preloadFileId}/image`;
  }, [preloadFileId]);

  // Auto-play is session-scoped: it resets off when the modal closes. Also
  // clear the restart-cue nonce, otherwise it stays > 0 and the overlay
  // replays its fade the next time the modal mounts (the subtree unmounts on
  // close, so a fresh mount re-runs the CSS animation).
  useEffect(() => {
    if (!isOpen) {
      setAutoPlay(false);
      setRestartFlash(0);
      setRelabelDetectionId(null);
    }
  }, [isOpen]);

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

      // The relabel picker is open: let it own the keyboard (Escape closes
      // it, not the modal; arrows/Enter drive its list) instead of the
      // modal's shortcuts firing underneath.
      if (relabelDetectionId) return;

      // Confirm + advance to the next unconfirmed event.
      if (e.key === "Enter") {
        e.preventDefault();
        handleConfirmAndAdvance();
        return;
      }

      switch (e.key) {
        case " ":
          // Space toggles auto-play (cine-loop the event's frames).
          e.preventDefault();
          toggleAutoPlay();
          break;
        // No modifier = move across events; Shift = move within the event
        // (frames). Shift as the frame-level modifier mirrors Shift+scroll.
        // Event nav keeps the cine-loop running (session mode); a manual
        // frame step grabs control and stops it.
        case "ArrowLeft":
          e.preventDefault();
          if (e.shiftKey) {
            setAutoPlay(false);
            if (selectedFileIndex > 0) setSelectedFileIndex((i) => i - 1);
          } else {
            navigateEvent(adjacent?.previous_id);
          }
          break;
        case "ArrowRight":
          e.preventDefault();
          if (e.shiftKey) {
            setAutoPlay(false);
            if (selectedFileIndex < files.length - 1)
              setSelectedFileIndex((i) => i + 1);
          } else {
            navigateEvent(adjacent?.next_id);
          }
          break;
        case "p":
        case "P":
          // Toggle the focused video between its still and the real player.
          if (currentFile && isPlayableVideo(currentFile)) {
            e.preventDefault();
            if (viewMode === "video") {
              setViewMode("frame");
            } else {
              playFocusedVideo();
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
        case "b":
        case "B":
          e.preventDefault();
          setBoxesHidden((h) => !h);
          break;
        case "Escape":
          e.preventDefault();
          onClose();
          break;
      }
    };

    // Register in capture phase so our preventDefault on Enter fires
    // before any focused button's implicit Enter-activates-click. Without
    // capture, clicking the > nav button grabs focus, and the next Enter
    // would re-fire that button's onClick (handleNext) instead of going
    // through the case "Enter" branch (confirm + next unconfirmed).
    window.addEventListener("keydown", handleKeyDown, true);
    return () => window.removeEventListener("keydown", handleKeyDown, true);
  }, [
    isOpen,
    currentFile,
    handleConfirmAndAdvance,
    toggleAutoPlay,
    onClose,
    selectedFileIndex,
    files.length,
    flagMutation,
    viewMode,
    playFocusedVideo,
    navigateEvent,
    adjacent,
    relabelDetectionId,
  ]);

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
            <div className="flex flex-col items-center gap-1 px-1.5 py-2 bg-white border-r shrink-0">
              {/* Watching a video moved to a big center play button over the
                  focus (the universal pattern), so the rail no longer carries
                  a play control of its own — only the cine-loop below. */}
              {/* Image: brightness / contrast (seeing a dark IR animal). */}
              <ViewControls
                brightness={brightness}
                onBrightnessChange={setBrightness}
                contrast={contrast}
                onContrastChange={setContrast}
              />
              {/* Show / hide the AI boxes — toggle off to count the scene
                  yourself without the AI's boxes anchoring you. */}
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => setBoxesHidden((h) => !h)}
                title={boxesHidden ? "Show AI boxes (B)" : "Hide AI boxes (B)"}
              >
                {boxesHidden ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </Button>
              {/* Loop event — cine-loop the event's frames to see motion.
                  Videos show as their best frame here; watch a clip in full
                  via the center play button on the focus. The loop glyph
                  keeps it distinct from that video-play triangle. */}
              {files.length > 1 && (
                <Button
                  variant={autoPlay ? "default" : "ghost"}
                  size="icon"
                  className="h-8 w-8"
                  onClick={toggleAutoPlay}
                  title={autoPlay ? "Stop (Space)" : "Loop event (Space)"}
                >
                  {autoPlay ? (
                    <Pause className="h-4 w-4" />
                  ) : (
                    <Repeat className="h-4 w-4" />
                  )}
                </Button>
              )}
              {/* Flag for review — the one triage action worth its own key. */}
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
              {/* Everything else (zoom, like, download, reveal) is rarely
                  used here, so it lives behind one "more" menu. */}
              <Popover open={moreOpen} onOpenChange={setMoreOpen}>
                <PopoverTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    title="More"
                  >
                    <MoreVertical className="h-4 w-4" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent side="right" align="start" className="w-48 p-1">
                  <button
                    onClick={() => {
                      setMoreOpen(false);
                      favoriteMutation.mutate({
                        fileId: currentFile.id,
                        favorited: !currentFile.favorited,
                      });
                    }}
                    disabled={favoriteMutation.isPending}
                    className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-accent transition-colors"
                  >
                    <Heart
                      className={cn(
                        "h-4 w-4",
                        currentFile.favorited && "fill-[#882000] text-[#882000]"
                      )}
                    />
                    {currentFile.favorited ? "Unlike" : "Like"}
                  </button>
                  <button
                    onClick={() => {
                      setMoreOpen(false);
                      handleDownload();
                    }}
                    className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-accent transition-colors"
                  >
                    <Download className="h-4 w-4" />
                    {isPlayableVideo(currentFile)
                      ? "Download video"
                      : "Download image"}
                  </button>
                  <button
                    onClick={() => {
                      setMoreOpen(false);
                      revealInFolder(currentFile);
                    }}
                    className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-accent transition-colors"
                  >
                    <FolderOpen className="h-4 w-4" />
                    Open in file explorer
                  </button>
                </PopoverContent>
              </Popover>
            </div>
          )}

          {/* Center column: focused image + resizable filmstrip below. */}
          <div ref={centerColumnRef} className="flex-1 flex flex-col min-w-0">
            {/* Focused image / video — all tools act here. */}
            <div className="relative flex-1 flex items-center justify-center bg-black/95 min-h-0">
              {/* Restart cue — flashed centered when the auto-play loop wraps
                  back to the first frame. Self-hides via the CSS fade. */}
              {restartFlash > 0 && (
                <div
                  key={restartFlash}
                  className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center"
                  style={{ animation: "restart-flash 0.8s ease-out forwards" }}
                >
                  <RotateCcw className="h-20 w-20 text-white" />
                </div>
              )}
              {/* Position-in-event chip + gap since the previous frame —
                  glanceable while scrubbing, eyes on the focus. */}
              {files.length > 1 && (
                <div className="pointer-events-none absolute bottom-2 right-2 z-10 rounded-md bg-black/60 px-2.5 py-1 text-sm font-medium tabular-nums text-white/90">
                  {selectedFileIndex + 1} / {files.length}
                  {focusGap && (
                    <span className="text-white/60"> · {focusGap}</span>
                  )}
                </div>
              )}
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
                ) : currentFile.file_type === "video" &&
                  !(autoPlay && files.length > 1) ? (
                  // Video, frame mode: show the time-spaced filmstrip gallery
                  // (the play overlay below sits on top). Only an actively
                  // flipping cine-loop (autoPlay across >1 file) falls back to
                  // the best-frame still; a lone video keeps its filmstrip.
                  <VideoFilmstrip fileId={currentFile.id} />
                ) : (
                  // View-only on the Counts page: boxes show but aren't
                  // edited here (label/box cleanup lives on the Labels page).
                  <AnnotationCanvas
                    file={currentFile}
                    detectionThreshold={detectionThreshold}
                    selectedDetectionId={relabelDetectionId}
                    onSelectDetection={() => {}}
                    onRequestRelabel={(id) => {
                      setAutoPlay(false);
                      setRelabelDetectionId(id);
                    }}
                    drawMode={false}
                    onDrawModeChange={() => {}}
                    readOnly
                    onScrubFrame={handleScrubFrame}
                    imageFilter={imageFilter}
                    boxesHidden={boxesHidden}
                    exportFnRef={exportFnRef}
                  />
                )
              ) : (
                <div className="text-white/50">Loading...</div>
              )}

              {/* Big center play button over a focused video's still — the
                  universal "watch this clip" affordance. Hidden during the
                  cine-loop (frames are flipping) and once the player is up.
                  The wrapper is click-through so the still still pans/zooms;
                  only the button itself catches the click. */}
              {currentFile?.file_type === "video" &&
                viewMode !== "video" &&
                !autoPlay &&
                (isPlayableVideo(currentFile) ? (
                  <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
                    <button
                      type="button"
                      onClick={playFocusedVideo}
                      title="Watch this video (P)"
                      className="pointer-events-auto flex h-16 w-16 items-center justify-center rounded-full bg-black/55 ring-1 ring-white/40 transition hover:scale-105 hover:bg-black/75"
                    >
                      <Play className="h-7 w-7 translate-x-0.5 fill-white text-white" />
                    </button>
                  </div>
                ) : (
                  <div className="pointer-events-none absolute inset-x-0 bottom-12 z-10 flex justify-center">
                    <span className="rounded-md bg-black/60 px-2.5 py-1 text-xs text-white/80">
                      Video format not supported for playback
                    </span>
                  </div>
                ))}
            </div>

            {/* Headless relabel picker — opened by clicking a label pill on
                the focus image. The trigger is taken out of the flow
                (absolute, zero-size); the dialog is portaled by Radix, so
                the wrapper position doesn't matter. Mirrors the Labels page
                (DetectionDetailModal). */}
            <div
              aria-hidden="true"
              className="absolute h-0 w-0 overflow-hidden pointer-events-none"
            >
              <LabelPicker
                headless
                value={relabelDetection?.label ?? null}
                displayName={relabelDetection?.scientific_name ?? null}
                options={labelOptions}
                isLoading={labelOptionsLoading}
                projectId={projectId}
                forceOpen={relabelDetectionId !== null}
                onOpenChange={(open) => {
                  if (!open) setRelabelDetectionId(null);
                }}
                onSelect={(option) => {
                  if (relabelDetectionId == null) return;
                  relabelMutation.mutate({
                    id: relabelDetectionId,
                    label: option.label,
                    category: option.category,
                  });
                }}
              />
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
          <div className="w-80 bg-white border-l flex flex-col shrink-0">
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
              <div className="mx-3 mt-3 shrink-0 rounded-lg border bg-muted/40 px-3 py-2 space-y-0.5 text-xs text-muted-foreground">
                <div className="font-medium text-foreground">
                  {eventSummary.media}
                </div>
                {eventSummary.when && <div>{eventSummary.when}</div>}
                {event.site_name && <div>{event.site_name}</div>}
                {currentFile && (
                  <div className="truncate pt-0.5 text-[11px] text-muted-foreground/70">
                    Viewing {basename(currentFile.file_path)} (
                    {selectedFileIndex + 1} of {files.length})
                  </div>
                )}
              </div>
            )}

            {/* Event-level species + count editor (the ecological record,
                and the star of this modal). */}
            {event && (
              <EventCountPanel
                eventId={event.id}
                projectId={projectId}
                observations={event.observations}
                confirmed={event.confirmed}
                onConfirm={handleConfirmAndAdvance}
                labelOptions={labelOptions}
                labelOptionsLoading={labelOptionsLoading}
              />
            )}

            {/* Keyboard shortcuts */}
            <div className="mt-auto shrink-0 px-3 pb-2 relative">
              {showShortcuts && (
                <>
                <div className="fixed inset-0 z-40" onClick={() => setShowShortcuts(false)} />
                <div className="absolute bottom-10 right-6 mb-2 rounded-lg border bg-background shadow-lg px-4 py-3 z-50 whitespace-nowrap">
                  <div className="text-[11px] font-semibold text-muted-foreground mb-1">Shortcuts</div>
                  {[
                    ["Enter", "Confirm + next event"],
                    ["↑ ↓", "Select species row"],
                    ["0-9", "Set count (type fast for 12, 130…)"],
                    ["+ / −", "Adjust count by 1"],
                    ["A", "Add species"],
                    ["← →", "Prev / next event"],
                    ["Shift + ← →", "Prev / next frame"],
                    ["Space", "Loop event"],
                    ["Shift + scroll", "Scrub frames"],
                    ["Scroll", "Zoom the focus"],
                    ["Click", "Focus a thumbnail"],
                    ["P", "Watch focused video"],
                    ["F", "Flag / unflag"],
                    ["B", "Show / hide AI boxes"],
                    ["Esc", "Close"],
                  ].map(([key, action]) => (
                    <div key={key} className="flex items-center text-xs gap-3 h-7">
                      <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">{key.split("+").map((part, i, arr) => <span key={i}>{part}{i < arr.length - 1 && <span className="text-[#bbbbc1]">+</span>}</span>)}</code>
                      <span>{action}</span>
                    </div>
                  ))}
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
