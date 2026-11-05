/**
 * Event detail modal — the Counts-page event viewer.
 *
 * Center: one focused image (the annotation canvas / video player) where
 * every tool acts, above a resizable wrapping-grid filmstrip of the event's
 * frames; drag the divider to trade focus size for scanning room, and click
 * a thumbnail to focus it. Left: the shared tool rail (`ViewerToolRail`:
 * brightness/contrast, hide boxes, flag, like, download, explorer) plus
 * the cine-loop. Right: the event-level species
 * + count editor (EventCountPanel) with the single "Confirm" sign-off.
 * Per-detection label cleanup at scale lives on the Labels page.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  ChevronsRight,
  X,
  Play,
  Pause,
  Repeat,
  RotateCcw,
} from "lucide-react";
import { toast } from "sonner";
import { warmClipFrames } from "../../lib/clip-frames";
import { cardBoxes, nearestSampledFrame, speciesKey, steppedFrame } from "../../lib/track-utils";
import { ClipTimeline } from "./TrackTimeline";
import { ApiError } from "../../lib/api-client";
import { eventsApi } from "../../api/events";
import { detectionsApi } from "../../api/detections";
import { projectsApi } from "../../api/projects";
import { basename } from "../../lib/path-utils";
import { describeEventMedia } from "../../lib/event-media";
import {
  formatCameraDate,
  formatCameraTime,
  formatTimeOffset,
} from "../../lib/datetime";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog";
import type {
  EventFilterParams,
  EventObservationItem,
  FileWithDetections,
} from "../../api/types";
import { EventFilmstrip } from "./EventFilmstrip";
import { ShortcutKey } from "./LabelsKeyboardPopover";
import { ViewerToolRail } from "./ViewerToolRail";
import { useFileTriage, useImageAdjust } from "./viewer-tools";
import type { TileSize } from "./CropGrid";
import { AnnotationCanvas } from "./AnnotationCanvas";
import { EventCountPanel } from "./EventCountPanel";
import { LabelPicker } from "./LabelPicker";
import { VideoPlayer, isPlayableVideo } from "./VideoPlayer";
import { useLabelOptions } from "../../hooks/useLabelOptions";
import { fileQueryOptions } from "../../api/files";
import { useResizableHeight } from "../../hooks/useResizableHeight";
import { persistSetting, readSetting } from "../../lib/local-settings";
import { ResizeHandle } from "../ui/resize-handle";
import { invalidateLabelQueries } from "../../lib/invalidate-label-queries";

/** The Counts page's view settings: tile size and the filmstrip height. */
const COUNTS_SETTINGS = "addaxai:countsSettings";

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
  const { brightness, setBrightness, contrast, setContrast, imageFilter } =
    useImageAdjust();

  // Filmstrip view settings, persisted per user under one key: the
  // S/M/L thumbnail size and the filmstrip height a drag leaves behind.
  const [tileSize, _setTileSize] = useState<TileSize>(
    () => (readSetting(COUNTS_SETTINGS, "tileSize") as TileSize | undefined) ?? "M",
  );
  const setTileSize = useCallback((v: TileSize) => {
    _setTileSize(v);
    persistSetting(COUNTS_SETTINGS, v, "tileSize");
  }, []);
  const centerColumnRef = useRef<HTMLDivElement>(null);
  const filmstripMax = useCallback(
    () => (centerColumnRef.current?.getBoundingClientRect().height ?? 0) * 0.8,
    [],
  );
  const { height: filmstripHeight, startDrag: startDividerDrag } = useResizableHeight({
    storageKey: COUNTS_SETTINGS,
    field: "filmstripHeight",
    defaultHeight: 220,
    minHeight: 120,
    maxHeight: filmstripMax,
  });

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

  // Prefetch the next event, so moving forward through observations
  // feels instant.
  useEffect(() => {
    const nextId = adjacent?.next_id;
    if (!nextId || !isOpen) return;
    queryClient.prefetchQuery({
      queryKey: ["event", nextId],
      queryFn: () => eventsApi.get(nextId),
    });
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

  // The track picked on the timeline; the player dims the others.
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null);
  // The timeline's zoom, kept here so it survives the switch between the
  // still and the player, which mount the timeline afresh.
  const [timelineZoom, setTimelineZoom] = useState(0);
  // The frame shown as a still in frame mode (a bar click, a lane click,
  // Show on an unplayable clip), keyed by file so paging never shows
  // one clip's frame under another. Null means the landing frame: the
  // first species' MaxN frame, the frame this page is for, else the
  // cover. Until 2026-11-02 a clip opened on a grid of nine frames
  // spaced evenly through it, from before the timeline existed: nine
  // seeks into the clip per open, and by construction rarely the
  // moment that matters.
  const [still, setStill] = useState<{ fileId: string; frame: number } | null>(null);

  // On open or event change, focus the busiest file: the one with the
  // most detections (a photo's boxes, a clip's tracks), regardless of
  // species. Gives multi-species events one unambiguous landing file.
  // Threshold 0 on purpose: the pick is threshold-blind, as it always
  // was, and `passesDrawFilter` still leaves rejected boxes out.
  useEffect(() => {
    const fs = event?.files ?? [];
    const peakCount = (f: FileWithDetections) => cardBoxes(f, 0).length;
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
    setSelectedTrackId(null);
    setTimelineZoom(0);
    setStill(null);
    setPendingVideoExport(false);
    setRelabelDetectionId(null);
  }, [eventId, event?.id]);

  const files = useMemo(() => event?.files ?? [], [event?.files]);
  const currentFile = files[selectedFileIndex] as
    | FileWithDetections
    | undefined;
  // The event carries each file's cards (cover boxes, one box per
  // track, MaxN); the canvas at any frame, the player and the timeline
  // need every box, so the focused file is fetched in full, the way the
  // Files viewer does. Until it lands, the cards draw the first still.
  const { data: fullFile } = useQuery({
    ...fileQueryOptions(currentFile?.id ?? ""),
    enabled: !!currentFile,
  });
  const focused = fullFile && fullFile.id === currentFile?.id ? fullFile : currentFile;
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
  const detectionThreshold = project?.counting_threshold ?? 0;

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
      allDetections: focused?.detections ?? currentFile.detections,
    };
  }, [currentFile, focused]);

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
  // Flag/like via the shared rail hook; the ["file"] invalidation lives
  // in the hook, the event caches are this modal's own concern.
  const invalidateEventCaches = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["event", eventId] });
    queryClient.invalidateQueries({ queryKey: ["events"] });
  }, [queryClient, eventId]);
  const triage = useFileTriage(invalidateEventCaches);

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
      invalidateLabelQueries(queryClient);
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

  // Auto-play toggle. Turning it on restarts the playthrough from frame 0.
  const toggleAutoPlay = useCallback(() => {
    if (!autoPlayRef.current) {
      setSelectedFileIndex(0);
      setViewMode("frame");
    }
    setAutoPlay((on) => !on);
  }, []);

  // The Counts panel's "Show": go to the file the MaxN was counted on and,
  // for a video, to that frame: the player seeks there, a clip the browser
  // cannot play shows that frame as a still, decoded on request.
  const [seekRequest, setSeekRequest] = useState<{ frame: number; nonce: number; play?: boolean } | null>(null);
  const stillFrame = still && still.fileId === currentFile?.id ? still.frame : null;
  const selectTrack = useCallback(
    (trackId: string) => {
      setSelectedTrackId(trackId);
      const track = currentFile?.tracks.find((t) => t.id === trackId);
      if (track && currentFile) {
        setStill({ fileId: currentFile.id, frame: track.representative_frame_number });
      }
    },
    [currentFile],
  );
  // MaxN markers for the focused video: one per species row, at the
  // frame its count was made on. A Counts thing, so only this modal
  // passes them to the player.
  const markers = useMemo(
    () =>
      (event?.max_n_frames ?? [])
        .filter(
          (m) => m.file_id === currentFile?.id && m.max_n_frame_number != null && m.label != null,
        )
        // Keyed as the timeline keys its groups, so a marker finds its row.
        .map((m) => ({
          frame: m.max_n_frame_number as number,
          label: speciesKey(m),
          count: m.effective_count,
        })),
    [event?.max_n_frames, currentFile?.id],
  );
  // The frame on screen in frame mode: the still a person asked for,
  // else the landing frame (see `still`).
  const shownFrame = stillFrame ?? markers[0]?.frame ?? null;

  /** Shift+Arrow and Shift+wheel: the small unit inside the event. For
   *  a clip, the previous or next sampled frame of the focused clip
   *  (`steppedFrame`, the Files viewer's step, so stepping a clip is
   *  one thing on both pages; until 2026-11-03 this moved between the
   *  event's files, which on a one-clip event was nothing). For photos,
   *  the previous or next picture of the burst. A manual step takes
   *  over from the cine-loop. */
  /** Once a clip's still is on screen, warm the sampled frames either
   *  side, so Shift+Arrow around the MaxN frame does not wait at each
   *  step (`warmClipFrames`, the Files viewer's rule too). */
  const warmNeighbourFrames = useCallback(
    (loading: boolean) => {
      if (loading || !focused || focused.file_type !== "video") return;
      const here = shownFrame ?? focused.best_frame_number ?? 0;
      warmClipFrames(focused, [steppedFrame(focused, here, 1), steppedFrame(focused, here, -1)]);
    },
    [focused, shownFrame],
  );
  const stepWithinEvent = useCallback(
    (dir: 1 | -1) => {
      setAutoPlay(false);
      setViewMode("frame");
      if (focused && currentFile && focused.file_type === "video") {
        const from = shownFrame ?? focused.best_frame_number ?? 0;
        setStill({ fileId: currentFile.id, frame: steppedFrame(focused, from, dir) });
        return;
      }
      setSelectedFileIndex((i) => Math.max(0, Math.min(files.length - 1, i + dir)));
    },
    [focused, currentFile, shownFrame, files.length],
  );

  // Watch the focused video from the frame on screen, the Files viewer's
  // rule: a bar click then P plays from that animal, and a fresh open
  // plays from the MaxN moment. Until 2026-11-02 the player started at
  // 0:00 here while the Files viewer seeked. Stops the cine-loop first
  // (you've grabbed control of one clip).
  const playFocusedVideo = useCallback(() => {
    setAutoPlay(false);
    if (shownFrame != null) {
      setSeekRequest({ frame: shownFrame, nonce: Date.now(), play: true });
    }
    setViewMode("video");
  }, [shownFrame]);
  const showMaxN = useCallback(
    (obs: EventObservationItem) => {
      const index = files.findIndex((f) => f.id === obs.max_n_file_id);
      if (index < 0) return;
      setAutoPlay(false);
      setSelectedFileIndex(index);
      const file = files[index];
      if (file.file_type === "video" && obs.max_n_frame_number != null) {
        if (isPlayableVideo(file)) {
          setSeekRequest({ frame: obs.max_n_frame_number, nonce: Date.now() });
          setViewMode("video");
          return;
        }
        setStill({ fileId: file.id, frame: obs.max_n_frame_number });
        setSelectedTrackId(null);
      }
      setViewMode("frame");
    },
    [files],
  );

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
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }

      // Keys inside a popover (the rail's brightness sliders) belong to
      // the popover: without this, arrow keys on the slider also paged
      // between events.
      if (
        e.target instanceof HTMLElement &&
        e.target.closest("[data-radix-popper-content-wrapper]")
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
          if (e.shiftKey) stepWithinEvent(-1);
          else navigateEvent(adjacent?.previous_id);
          break;
        case "ArrowRight":
          e.preventDefault();
          if (e.shiftKey) stepWithinEvent(1);
          else navigateEvent(adjacent?.next_id);
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
          if (currentFile) triage.toggleFlag(currentFile);
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
    stepWithinEvent,
    triage,
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
        // Escape inside the event note closes the note (EventCountPanel
        // handles it), not the modal. Radix listens in the capture phase,
        // so this is the only place that can stop it.
        onEscapeKeyDown={(e) => {
          if (e.target instanceof HTMLTextAreaElement) e.preventDefault();
        }}
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
          {/* Left toolbar — the shared rail (how you look at the
              picture), plus this modal's own cine-loop. Watching a video
              moved to a big center play button over the focus (the
              universal pattern), so the rail carries no play control. */}
          {currentFile && (
            <div className="flex flex-col items-center gap-1 px-1.5 py-2 bg-white border-r shrink-0">
              <ViewerToolRail
                brightness={brightness}
                onBrightnessChange={setBrightness}
                contrast={contrast}
                onContrastChange={setContrast}
                boxesHidden={boxesHidden}
                onToggleBoxes={() => setBoxesHidden((h) => !h)}
                file={currentFile}
                triage={triage}
                downloadLabel={
                  isPlayableVideo(currentFile)
                    ? "Download video"
                    : "Download image"
                }
                onDownload={handleDownload}
              >
                {/* Loop event — cine-loop the event's frames to see
                    motion. The loop glyph keeps it distinct from the
                    video-play triangle. */}
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
              </ViewerToolRail>
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
                <div className="flex h-full w-full flex-col">
                <div className="relative min-h-0 w-full flex-1">
                {viewMode === "video" && isPlayableVideo(currentFile) ? (
                  <VideoPlayer
                    file={focused ?? currentFile}
                    detectionThreshold={detectionThreshold}
                    sourceVideoId={videoPlaybackProps?.sourceVideoId}
                    allDetections={videoPlaybackProps?.allDetections}
                    exportFnRef={exportFnRef}
                    autoExport={pendingVideoExport}
                    onAutoExportConsumed={() => setPendingVideoExport(false)}
                    boxesHidden={boxesHidden}
                    seekRequest={seekRequest}
                    selectedTrackId={selectedTrackId}
                    zoomStep={timelineZoom}
                    onZoomStep={setTimelineZoom}
                    onSelectTrack={selectTrack}
                    markers={markers}
                    surface="counts"
                  />
                ) : (
                  // View-only on the Counts page: boxes show but aren't
                  // edited here (label/box cleanup lives on the Labels page).
                  // A clip shows one frame, the MaxN frame first, with the
                  // timeline under it; the play overlay below sits on top.
                  <AnnotationCanvas
                    file={focused ?? currentFile}
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
                    onScrubFrame={(d) => stepWithinEvent(d > 0 ? 1 : -1)}
                    onLoadingChange={warmNeighbourFrames}
                    imageFilter={imageFilter}
                    boxesHidden={boxesHidden}
                    exportFnRef={exportFnRef}
                    frameNumber={
                      shownFrame != null && shownFrame !== currentFile.best_frame_number
                        ? shownFrame
                        : undefined
                    }
                  />
                )}
                </div>
                {/* The clip's tracks as bars under the filmstrip or the
                    still, with the MaxN markers: the same timeline the
                    player carries, so an AVI clip shows its detections
                    and the counted frame too. */}
                {currentFile.file_type === "video" && viewMode !== "video" && (
                  <ClipTimeline
                    file={focused ?? currentFile}
                    detectionThreshold={detectionThreshold}
                    currentFrame={shownFrame ?? currentFile.best_frame_number ?? 0}
                    selectedTrackId={selectedTrackId}
                    zoomStep={timelineZoom}
                    onZoomStep={setTimelineZoom}
                    markers={markers}
                    surface="counts"
                    onSelectTrack={selectTrack}
                    onSeek={(frame) => {
                      // Snapped to a sampled frame (`nearestSampledFrame`).
                      setStill({
                        fileId: currentFile.id,
                        frame: nearestSampledFrame(focused ?? currentFile, frame),
                      });
                      setSelectedTrackId(null);
                    }}
                  />
                )}
                </div>
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
            <ResizeHandle onMouseDown={startDividerDrag} title="Drag to resize the filmstrip" />

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
                notes={event.notes}
                onConfirm={handleConfirmAndAdvance}
                labelOptions={labelOptions}
                labelOptionsLoading={labelOptionsLoading}
                onShowMaxN={showMaxN}
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
                    ["R", "Change species"],
                    ["← / →", "Previous / next event"],
                    ["Shift + ← / →", "Previous / next frame"],
                    ["Space", "Loop event"],
                    ["Shift + scroll", "Previous / next frame"],
                    ["Scroll", "Zoom the picture"],
                    ["Click", "Focus a thumbnail"],
                    ["P", "Watch focused video"],
                    ["F", "Flag / unflag"],
                    ["B", "Show / hide AI boxes"],
                    ["Esc", "Close"],
                  ].map(([key, action]) => (
                    <div key={key} className="flex items-center text-xs gap-3 h-7">
                      <ShortcutKey keys={key} />
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
