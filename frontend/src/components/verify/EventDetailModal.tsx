/**
 * Event detail modal - full-screen event viewer with filmstrip navigation.
 *
 * Shows the selected event's images with interactive annotation canvas,
 * filmstrip for multi-file navigation, verification panel, and
 * event-to-event navigation.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { basename } from "../../lib/path-utils";
import {
  ChevronLeft,
  ChevronRight,
  ChevronsRight,
  X,
  Scale,
  Sun,
  Contrast,
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
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { useRevealInFolder } from "../../lib/file-reveal";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { Slider } from "../ui/slider";
import type { EventFilterParams, FileWithDetections } from "../../api/types";
import { EventFilmstrip } from "./EventFilmstrip";
import { AnnotationCanvas } from "./AnnotationCanvas";
import { EventCountPanel } from "./EventCountPanel";
import { FileVerificationPanel } from "./FileVerificationPanel";
import { LabelPicker } from "./LabelPicker";
import { VideoPlayer, isPlayableVideo } from "./VideoPlayer";
import { useLabelOptions, type LabelOption } from "../../hooks/useLabelOptions";
import { getSpeciesColor, getSpeciesTextColor } from "../../utils/species-colors";

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
  const [bulkSelection, setBulkSelection] = useState<Set<number>>(new Set());
  const [viewMode, setViewMode] = useState<"frame" | "video">("frame");
  // One-shot flag: set when Download is clicked on a video while in frame
  // view, so the VideoPlayer runs the annotated-video export once it mounts.
  const [pendingVideoExport, setPendingVideoExport] = useState(false);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [videoPopoverOpen, setVideoPopoverOpen] = useState(false);
  const [boxesHidden, setBoxesHidden] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [shortcutLabels, setShortcutLabels] = useState<Record<number, LabelOption>>({});
  const [openLabelPickerFor, setOpenLabelPickerFor] = useState<string | null>(null);
  const [localThreshold, setLocalThreshold] = useState<number | null>(null);
  const [brightness, setBrightness] = useState(50);
  const [contrast, setContrast] = useState(50);
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
  const fileNavRef = useRef<"forward" | "backward" | null>(null);

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

  // Compute MaxN file IDs set for navigation
  const maxNFileIds = useMemo(() => {
    if (!event?.max_n_frames) return new Set<string>();
    return new Set(event.max_n_frames.map((f) => f.file_id));
  }, [event?.max_n_frames]);

  // When event changes, open to the first unverified MaxN frame (in MaxN mode)
  // or the first file (in file mode). If navigating via file-level nav,
  // continue the sequential flow.
  useEffect(() => {
    // ←/→ always navigates by MaxN frame; this effect picks the
    // initial selectedFileIndex on event load. fileNavRef tells us
    // whether we arrived from the previous or next event.
    if (fileNavRef.current) {
      const dir = fileNavRef.current;
      fileNavRef.current = null;
      if (event?.files.length) {
        const maxNIds = new Set(event.max_n_frames?.map((f) => f.file_id) ?? []);
        if (dir === "backward") {
          // Land on the last MaxN frame in the new event.
          let found = false;
          for (let i = event.files.length - 1; i >= 0; i--) {
            if (maxNIds.has(event.files[i].id)) {
              setSelectedFileIndex(i);
              found = true;
              break;
            }
          }
          if (!found) setSelectedFileIndex(event.files.length - 1);
        } else {
          // Forward: first unverified MaxN, else first MaxN, else 0.
          let found = false;
          for (let i = 0; i < event.files.length; i++) {
            if (maxNIds.has(event.files[i].id) && !event.files[i].verified) {
              setSelectedFileIndex(i);
              found = true;
              break;
            }
          }
          if (!found) {
            for (let i = 0; i < event.files.length; i++) {
              if (maxNIds.has(event.files[i].id)) {
                setSelectedFileIndex(i);
                found = true;
                break;
              }
            }
          }
          if (!found) setSelectedFileIndex(0);
        }
      } else {
        setSelectedFileIndex(0);
      }
    } else if (!event) {
      setSelectedFileIndex(0);
    } else {
      // Fresh open: first unverified MaxN frame, else first MaxN, else 0.
      const maxNIds = new Set(event.max_n_frames?.map((f) => f.file_id) ?? []);
      let found = false;
      for (let i = 0; i < event.files.length; i++) {
        if (maxNIds.has(event.files[i].id) && !event.files[i].verified) {
          setSelectedFileIndex(i);
          found = true;
          break;
        }
      }
      if (!found) {
        for (let i = 0; i < event.files.length; i++) {
          if (maxNIds.has(event.files[i].id)) {
            setSelectedFileIndex(i);
            found = true;
            break;
          }
        }
      }
      if (!found) setSelectedFileIndex(0);
    }
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

  // MaxN frames for the currently selected file (for badge display)
  const currentFileMaxNFrames = useMemo(() => {
    if (!currentFile || !event?.max_n_frames) return [];
    return event.max_n_frames.filter((f) => f.file_id === currentFile.id);
  }, [currentFile, event?.max_n_frames]);

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

  // Calculate modal dimensions to tightly fit the image + UI panels.
  // Keep previous size while loading to avoid a resize flash between images.
  const lastModalStyle = useRef<{ width: number; height: number } | null>(null);
  const modalStyle = useMemo(() => {
    const TOOLBAR_W = 44;
    const PANEL_W = 320;
    const FILMSTRIP_H = files.length > 0 ? 96 : 0;
    const IMAGE_PAD = 16;

    const maxW = viewport.width * 0.95;
    const maxH = viewport.height * 0.95;

    if (!currentFile?.width_px || !currentFile?.height_px) {
      return lastModalStyle.current ?? { width: maxW, height: maxH };
    }

    const maxImgW = maxW - TOOLBAR_W - PANEL_W;
    const maxImgH = maxH - FILMSTRIP_H - IMAGE_PAD;

    // Fit image maintaining aspect ratio, capped at natural resolution
    const scale = Math.min(
      maxImgW / currentFile.width_px,
      maxImgH / currentFile.height_px,
      1
    );
    const imgDisplayW = currentFile.width_px * scale;
    const imgDisplayH = currentFile.height_px * scale;

    const style = {
      width: Math.round(imgDisplayW + TOOLBAR_W + PANEL_W),
      height: Math.round(imgDisplayH + FILMSTRIP_H + IMAGE_PAD),
    };
    lastModalStyle.current = style;
    return style;
  }, [currentFile?.width_px, currentFile?.height_px, files.length, viewport]);

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
      queryClient.invalidateQueries({ queryKey: ["file"] });
    },
  });


  // Favorite mutation
  const favoriteMutation = useMutation({
    mutationFn: () => {
      if (!currentFile) return Promise.resolve(null);
      return filesApi.update(currentFile.id, { favorited: !currentFile.favorited });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
      queryClient.invalidateQueries({ queryKey: ["events"] });
      queryClient.invalidateQueries({ queryKey: ["file"] });
    },
  });

  // Flag mutation — toggles flagged on the currently viewed file. Flag
  // lives at the file level; the Event card badge lights up if any file
  // in the event is flagged.
  const flagMutation = useMutation({
    mutationFn: () => {
      if (!currentFile) return Promise.resolve(null);
      return filesApi.update(currentFile.id, { flagged: !currentFile.flagged });
    },
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

  // Mark blank mutation: delete all detections + verify + advance
  const markBlankMutation = useMutation({
    mutationFn: async () => {
      if (!currentFile) return;
      await detectionsApi.deleteByFile(currentFile.id);
      await filesApi.update(currentFile.id, { verified: true });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
      queryClient.invalidateQueries({ queryKey: ["events"] });
      queryClient.invalidateQueries({ queryKey: ["file"] });
      queryClient.invalidateQueries({ queryKey: ["label-tree"] });
    },
  });

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

  // ←/→ navigates by MaxN frame across the event (and crosses to the
  // adjacent event at the boundary). Shift+←/→ in the keyboard handler
  // walks every frame inside the current event for the rare per-frame
  // inspection case.
  const nextUnverifiedFileIndex = useMemo(() => {
    for (let i = selectedFileIndex + 1; i < files.length; i++) {
      if (maxNFileIds.has(files[i].id) && !files[i].verified) return i;
    }
    return -1;
  }, [files, selectedFileIndex, maxNFileIds]);

  const handlePrev = useCallback(() => {
    for (let i = selectedFileIndex - 1; i >= 0; i--) {
      if (maxNFileIds.has(files[i].id)) {
        setSelectedFileIndex(i);
        return;
      }
    }
    // No earlier MaxN frame in this event — cross to previous event.
    if (adjacent?.previous_id) {
      fileNavRef.current = "backward";
      navigateEvent(adjacent.previous_id);
    }
  }, [selectedFileIndex, adjacent, navigateEvent, files, maxNFileIds]);

  const handleNext = useCallback(() => {
    for (let i = selectedFileIndex + 1; i < files.length; i++) {
      if (maxNFileIds.has(files[i].id)) {
        setSelectedFileIndex(i);
        return;
      }
    }
    // No later MaxN frame in this event — cross to next event.
    if (adjacent?.next_id) {
      fileNavRef.current = "forward";
      navigateEvent(adjacent.next_id);
    }
  }, [selectedFileIndex, files, adjacent, navigateEvent, maxNFileIds]);

  const handleNextUnverified = useCallback(() => {
    if (nextUnverifiedFileIndex >= 0) {
      setSelectedFileIndex(nextUnverifiedFileIndex);
    } else if (adjacent?.next_unverified_id) {
      fileNavRef.current = "forward";
      navigateEvent(adjacent.next_unverified_id);
    }
  }, [nextUnverifiedFileIndex, adjacent, navigateEvent]);

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

  // Verify and advance. With several files selected, bulk-verify them all;
  // otherwise verify the current file (if not already) and move to the next
  // unverified one. Shared by the Enter shortcut and the "Mark verified"
  // button so the two never drift apart.
  const handleVerifyAndNext = useCallback(() => {
    if (bulkSelection.size > 1) {
      const toVerify = [...bulkSelection]
        .map((i) => files[i])
        .filter((f) => f && !f.verified);
      if (toVerify.length > 0) {
        Promise.all(
          toVerify.map((f) => filesApi.update(f.id, { verified: true })),
        ).then(() => {
          queryClient.invalidateQueries({ queryKey: ["event", eventId] });
          queryClient.invalidateQueries({ queryKey: ["events"] });
          queryClient.invalidateQueries({ queryKey: ["file"] });
          setBulkSelection(new Set());
          handleNextUnverified();
        });
      } else {
        setBulkSelection(new Set());
        handleNextUnverified();
      }
    } else if (currentFile && !currentFile.verified) {
      verifyMutation.mutateAsync().then(() => handleNextUnverified());
    } else {
      handleNextUnverified();
    }
  }, [
    bulkSelection,
    files,
    queryClient,
    eventId,
    currentFile,
    verifyMutation,
    handleNextUnverified,
  ]);

  const handleFilmstripSelect = useCallback((index: number, shiftKey: boolean) => {
    if (shiftKey && files.length > 1) {
      const start = Math.min(selectedFileIndex, index);
      const end = Math.max(selectedFileIndex, index);
      const range = new Set<number>();
      for (let i = start; i <= end; i++) range.add(i);
      setBulkSelection(range);
    } else {
      setBulkSelection(new Set());
    }
    setSelectedFileIndex(index);
    setSelectedDetectionId(null);
  }, [selectedFileIndex, files.length]);

  const prevDisabled = (() => {
    const hasPrevMaxN = files.slice(0, selectedFileIndex).some((f) => maxNFileIds.has(f.id));
    return !hasPrevMaxN && !adjacent?.previous_id;
  })();
  const nextDisabled = (() => {
    const hasNextMaxN = files.slice(selectedFileIndex + 1).some((f) => maxNFileIds.has(f.id));
    return !hasNextMaxN && !adjacent?.next_id;
  })();
  const nextUnverifiedDisabled = nextUnverifiedFileIndex < 0 && !adjacent?.next_unverified_id;

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
          setBulkSelection(new Set());
          if (e.shiftKey) {
            // Navigate files within event, stop at boundary
            if (selectedFileIndex > 0) {
              setSelectedFileIndex((i) => i - 1);
              setSelectedDetectionId(null);
            }
          } else {
            handlePrev();
          }
          break;
        case "ArrowRight":
          e.preventDefault();
          setBulkSelection(new Set());
          if (e.shiftKey) {
            // Navigate files within event, stop at boundary
            if (selectedFileIndex < files.length - 1) {
              setSelectedFileIndex((i) => i + 1);
              setSelectedDetectionId(null);
            }
          } else {
            handleNext();
          }
          break;
        case "Enter":
          e.preventDefault();
          handleVerifyAndNext();
          break;
        case "e":
        case "E":
          e.preventDefault();
          if (currentFile) {
            markBlankMutation.mutateAsync().then(() => handleNextUnverified());
          }
          break;
        case "1": case "2": case "3": case "4": case "5": {
          const slot = parseInt(e.key);
          const label = shortcutLabels[slot];
          if (!label || !currentFile) break;
          e.preventDefault();
          Promise.all(
            filteredDetections.map((d) =>
              detectionsApi.update(d.id, {
                category: label.category,
                label: label.label,
              })
            )
          ).then(() => {
            queryClient.invalidateQueries({ queryKey: ["event", eventId] });
            queryClient.invalidateQueries({ queryKey: ["label-tree"] });
          });
          break;
        }
        case "Tab":
          if (selectedDetectionId) {
            e.preventDefault();
            setOpenLabelPickerFor(selectedDetectionId);
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
        case "v":
        case "V":
          e.preventDefault();
          verifyMutation.mutate();
          break;
        case "f":
        case "F":
          e.preventDefault();
          flagMutation.mutate();
          break;
        case "a":
        case "A":
          // Cmd+A / Ctrl+A: select all files (for bulk verify). Plain A
          // is unbound.
          if (e.metaKey || e.ctrlKey) {
            e.preventDefault();
            if (files.length > 1) {
              const all = new Set<number>();
              for (let i = 0; i < files.length; i++) all.add(i);
              setBulkSelection(all);
            }
          }
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
          if (bulkSelection.size > 0) {
            setBulkSelection(new Set());
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
    currentFile,
    drawMode,
    filteredDetections,
    handlePrev,
    handleNext,
    handleNextUnverified,
    handleVerifyAndNext,
    onClose,
    selectedDetectionId,
    selectedFileIndex,
    files.length,
    verifyMutation,
    flagMutation,
    markBlankMutation,
    deleteDetectionMutation,
    shortcutLabels,
    eventId,
    queryClient,
    bulkSelection,
    files,
    viewMode,
    sourceVideos,
  ]);

  // B key hold: momentarily hide boxes
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
          {/* Left toolbar */}
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
              {/* View threshold popover (local override; does not change
                  the project's detection_threshold). */}
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    title={`View threshold: ${(detectionThreshold * 100).toFixed(0)}%`}
                  >
                    <Scale className="h-4 w-4" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent side="right" className="w-48 p-3">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium">View threshold</span>
                      <div className="flex items-center gap-1">
                        <span className="text-xs text-muted-foreground tabular-nums">
                          {(detectionThreshold * 100).toFixed(0)}%
                        </span>
                        {localThreshold !== null && (
                          <button
                            onClick={() => setLocalThreshold(null)}
                            className="text-xs text-muted-foreground hover:text-foreground"
                            title={`Reset to project default (${(projectThreshold * 100).toFixed(0)}%)`}
                          >
                            <RotateCcw className="h-3 w-3" />
                          </button>
                        )}
                      </div>
                    </div>
                    <Slider
                      value={[detectionThreshold]}
                      onValueChange={([v]) => setLocalThreshold(v)}
                      min={0}
                      max={1}
                      step={0.05}
                    />
                  </div>
                </PopoverContent>
              </Popover>
              {/* Brightness popover */}
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    title={`Brightness: ${brightness}%`}
                  >
                    <Sun className="h-4 w-4" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent side="right" className="w-48 p-3">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium">Brightness</span>
                      <div className="flex items-center gap-1">
                        <span className="text-xs text-muted-foreground tabular-nums">
                          {brightness}%
                        </span>
                        {brightness !== 50 && (
                          <button
                            onClick={() => setBrightness(50)}
                            className="text-xs text-muted-foreground hover:text-foreground"
                            title="Reset to 50%"
                          >
                            <RotateCcw className="h-3 w-3" />
                          </button>
                        )}
                      </div>
                    </div>
                    <Slider
                      value={[brightness]}
                      onValueChange={([v]) => setBrightness(v)}
                      min={0}
                      max={100}
                      step={5}
                    />
                  </div>
                </PopoverContent>
              </Popover>
              {/* Contrast popover */}
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    title={`Contrast: ${contrast}%`}
                  >
                    <Contrast className="h-4 w-4" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent side="right" className="w-48 p-3">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium">Contrast</span>
                      <div className="flex items-center gap-1">
                        <span className="text-xs text-muted-foreground tabular-nums">
                          {contrast}%
                        </span>
                        {contrast !== 50 && (
                          <button
                            onClick={() => setContrast(50)}
                            className="text-xs text-muted-foreground hover:text-foreground"
                            title="Reset to 50%"
                          >
                            <RotateCcw className="h-3 w-3" />
                          </button>
                        )}
                      </div>
                    </div>
                    <Slider
                      value={[contrast]}
                      onValueChange={([v]) => setContrast(v)}
                      min={0}
                      max={100}
                      step={5}
                    />
                  </div>
                </PopoverContent>
              </Popover>
              <div className="w-6 border-t my-0.5" />
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => favoriteMutation.mutate()}
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
                onClick={() => flagMutation.mutate()}
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

          {/* Image area */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Main image/video with detections */}
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

            {/* Filmstrip */}
            {files.length > 0 && (
              <EventFilmstrip
                files={files}
                selectedIndex={selectedFileIndex}
                detectionThreshold={detectionThreshold}
                maxNFrames={event?.max_n_frames ?? []}
                onSelectIndex={handleFilmstripSelect}
                bulkSelection={bulkSelection}
              />
            )}
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
                  onClick={handlePrev}
                  title="Previous (←)"
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  disabled={nextDisabled}
                  onClick={handleNext}
                  title="Next (→)"
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  disabled={nextUnverifiedDisabled}
                  onClick={handleNextUnverified}
                  title="Next unverified"
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

            {/* Event-level species + count editor (the ecological record). */}
            {event && (
              <EventCountPanel
                eventId={event.id}
                projectId={projectId}
                observations={event.observations}
                verified={event.verified}
                labelOptions={labelOptions}
                labelOptionsLoading={labelOptionsLoading}
              />
            )}

            {/* File metadata card */}
            {currentFile && (
              <div className="mx-3 mt-2 rounded-lg border bg-muted/40">
                <div className="flex items-center gap-2 px-3 pt-3 pb-2">
                  <h3 className="text-sm font-semibold">
                    {currentFile.file_type === "video" ? "Video" : "Image"}
                  </h3>
                  {currentFileMaxNFrames.map((frame) => (
                    <span
                      key={frame.label}
                      className="text-[10px] leading-none font-medium px-1.5 py-0.5 rounded-sm capitalize"
                      style={{ backgroundColor: getSpeciesColor(frame.label_taxonomy_id || frame.label || ""), color: getSpeciesTextColor(frame.label_taxonomy_id || frame.label || "") }}
                    >
                      MaxN: {frame.label} ×{frame.effective_count}
                    </span>
                  ))}
                </div>
                <div className="px-3 pb-3 space-y-0.5 text-xs text-muted-foreground">
                  <div className="truncate">
                    {basename(currentFile.file_path)}
                  </div>
                  <div>
                    {formatCameraDate(currentFile.captured_at_local, { day: "numeric", month: "short", year: "numeric" }, "en-GB")}{" "}
                    {formatCameraTime(currentFile.captured_at_local, { hour: "2-digit", minute: "2-digit" }, "en-GB")}
                    {event?.site_name && ` · ${event.site_name}`}
                  </div>
                </div>
              </div>
            )}

            {/* Verification panel */}
            {currentFile && (
              <FileVerificationPanel
                key={currentFile.id}
                file={currentFile}
                projectId={projectId}
                detectionThreshold={detectionThreshold}
                labelOptions={labelOptions}
                labelOptionsLoading={labelOptionsLoading}
                selectedDetectionId={selectedDetectionId}
                onSelectDetection={setSelectedDetectionId}
                openLabelPickerFor={openLabelPickerFor}
                onLabelPickerOpenChange={(open) => {
                  if (!open) setOpenLabelPickerFor(null);
                }}
                pinnedOptions={Object.entries(shortcutLabels).map(([k, v]) => ({
                  key: Number(k),
                  option: v,
                }))}
                onVerify={handleVerifyAndNext}
                verifyPending={verifyMutation.isPending}
                onMutated={() => {
                  queryClient.invalidateQueries({ queryKey: ["event", eventId] });
                  queryClient.invalidateQueries({ queryKey: ["events"] });
                  queryClient.invalidateQueries({ queryKey: ["file"] });
                  queryClient.invalidateQueries({ queryKey: ["label-tree"] });
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
                    {/* Left column: navigation & selection */}
                    <div>
                      {[
                        ["← →", "Prev / next MaxN frame"],
                        ["Shift + ← →", "Prev / next frame in event"],
                        ["↑ ↓", "Select detection"],
                        ["Shift + Click", "Select file range"],
                        [navigator.platform.includes("Mac") ? "Cmd + A" : "Ctrl + A", "Select all files"],
                        ["Scroll", "Zoom in / out"],
                        ["P", "Toggle video / frame"],
                        ["B (hold)", "Hide boxes"],
                        ["Esc", "Deselect / close"],
                      ].map(([key, action]) => (
                        <div key={key} className="flex items-center text-xs gap-3 h-7">
                          <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">{key.split("+").map((part, i, arr) => <span key={i}>{part}{i < arr.length - 1 && <span className="text-[#bbbbc1]">+</span>}</span>)}</code>
                          <span>{action}</span>
                        </div>
                      ))}
                    </div>

                    {/* Right column: actions & label shortcuts */}
                    <div>
                      {[
                        ["Enter", "Verify + next unverified"],
                        ["V", "Verify (stay on file)"],
                        ["E", "Empty + next unverified"],
                        ["F", "Flag / unflag file"],
                        ["Tab", "Change label"],
                        ["D", "Draw a box"],
                        ["Del", "Delete detection"],
                      ].map(([key, action]) => (
                        <div key={key} className="flex items-center text-xs gap-3 h-7">
                          <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">{key.split("+").map((part, i, arr) => <span key={i}>{part}{i < arr.length - 1 && <span className="text-[#bbbbc1]">+</span>}</span>)}</code>
                          <span>{action}</span>
                        </div>
                      ))}

                      <div className="border-t my-2" />

                      {[1, 2, 3, 4, 5].map((n) => (
                        <div key={n} className="flex items-center text-xs gap-3 h-7">
                          <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">{n}</code>
                          <span>Change all to</span>
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
