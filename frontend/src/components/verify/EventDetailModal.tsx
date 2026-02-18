/**
 * Event detail modal - full-screen event viewer with filmstrip navigation.
 *
 * Shows the selected event's images with interactive annotation canvas,
 * filmstrip for multi-file navigation, verification panel, and
 * event-to-event navigation.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  ChevronsRight,
  X,
  Scale,
  Sun,
  Contrast,
  Pencil,
  SquarePlus,
  Download,
  Heart,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  FolderOpen,
} from "lucide-react";
import { eventsApi } from "../../api/events";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { projectsApi } from "../../api/projects";
import { cn } from "../../lib/utils";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Slider } from "../ui/slider";
import type { FileWithDetections } from "../../api/types";
import { EventFilmstrip } from "./EventFilmstrip";
import { AnnotationCanvas } from "./AnnotationCanvas";
import { FileVerificationPanel } from "./FileVerificationPanel";
import { LabelPicker } from "./LabelPicker";
import { useLabelOptions, type LabelOption } from "../../hooks/useLabelOptions";

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
  const [navScope, setNavScope] = useState<"event" | "file">("event");
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(
    null
  );
  const [drawMode, setDrawMode] = useState(false);
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

  // When event changes, open to the representative file — unless we got
  // here by stepping through files (file-level nav), in which case start
  // at the first file to continue the sequential flow.
  useEffect(() => {
    if (fileNavRef.current) {
      const dir = fileNavRef.current;
      fileNavRef.current = null;
      setSelectedFileIndex(
        dir === "backward" && event?.files.length ? event.files.length - 1 : 0
      );
    } else if (!event) {
      setSelectedFileIndex(0);
    } else {
      const repIdx = event.representative_file_id
        ? event.files.findIndex((f) => f.id === event.representative_file_id)
        : -1;
      setSelectedFileIndex(repIdx >= 0 ? repIdx : 0);
    }
    setSelectedDetectionId(null);
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

  // Compute most common detection label for smart draw defaults
  const defaultLabel = useMemo(() => {
    if (!event?.files)
      return { category: "animal", species: undefined as string | undefined };

    const labelCounts = new Map<
      string,
      { count: number; category: string; species: string | undefined }
    >();

    for (const f of event.files) {
      for (const d of f.detections) {
        if (d.confidence >= detectionThreshold) {
          const key = d.species || d.category;
          const existing = labelCounts.get(key);
          if (existing) {
            existing.count++;
          } else {
            labelCounts.set(key, {
              count: 1,
              category: d.category,
              species: d.species || undefined,
            });
          }
        }
      }
    }

    let best = { category: "animal", species: undefined as string | undefined };
    let bestCount = 0;
    for (const entry of labelCounts.values()) {
      if (entry.count > bestCount) {
        bestCount = entry.count;
        best = { category: entry.category, species: entry.species };
      }
    }

    return best;
  }, [event?.files, detectionThreshold]);

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
    },
  });

  // Hidden detections for "add box" (below threshold, min 0.2 confidence)
  const hiddenDetections = useMemo(() => {
    if (!currentFile) return [];
    return currentFile.detections
      .filter((d) => d.confidence < detectionThreshold && d.confidence >= 0.2)
      .sort((a, b) => b.confidence - a.confidence);
  }, [currentFile, detectionThreshold]);

  // Add box mutation - promote highest confidence hidden detection
  const addBoxMutation = useMutation({
    mutationFn: async () => {
      if (!currentFile || hiddenDetections.length === 0) return;
      const best = hiddenDetections[0];
      await detectionsApi.create({
        file_id: currentFile.id,
        category: best.category,
        bbox_x: best.bbox_x,
        bbox_y: best.bbox_y,
        bbox_width: best.bbox_width,
        bbox_height: best.bbox_height,
        species: best.species,
      });
      await detectionsApi.delete(best.id);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
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
    },
  });

  // Filtered detections for the current file (for Tab cycling)
  const filteredDetections = useMemo(() => {
    if (!currentFile) return [];
    return currentFile.detections.filter(
      (d) => d.confidence >= detectionThreshold
    );
  }, [currentFile, detectionThreshold]);

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

  // Context-aware navigation (file-level vs event-level)
  const nextUnverifiedFileIndex = useMemo(() => {
    for (let i = selectedFileIndex + 1; i < files.length; i++) {
      if (!files[i].verified) return i;
    }
    return -1;
  }, [files, selectedFileIndex]);

  const handlePrev = useCallback(() => {
    if (navScope === "file") {
      if (selectedFileIndex > 0) {
        setSelectedFileIndex((i) => i - 1);
      } else if (adjacent?.previous_id) {
        // At first file — advance to previous event
        fileNavRef.current = "backward";
        navigateEvent(adjacent.previous_id);
      }
    } else {
      navigateEvent(adjacent?.previous_id);
    }
  }, [navScope, selectedFileIndex, adjacent, navigateEvent]);

  const handleNext = useCallback(() => {
    if (navScope === "file") {
      if (selectedFileIndex < files.length - 1) {
        setSelectedFileIndex((i) => i + 1);
      } else if (adjacent?.next_id) {
        // At last file — advance to next event
        fileNavRef.current = "forward";
        navigateEvent(adjacent.next_id);
      }
    } else {
      navigateEvent(adjacent?.next_id);
    }
  }, [navScope, selectedFileIndex, files.length, adjacent, navigateEvent]);

  const handleNextUnverified = useCallback(() => {
    if (navScope === "file") {
      if (nextUnverifiedFileIndex >= 0) {
        setSelectedFileIndex(nextUnverifiedFileIndex);
      } else if (adjacent?.next_unverified_id) {
        // No more unverified files in event — jump to next unverified event
        fileNavRef.current = "forward";
        navigateEvent(adjacent.next_unverified_id);
      }
    } else {
      navigateEvent(adjacent?.next_unverified_id);
    }
  }, [navScope, nextUnverifiedFileIndex, adjacent, navigateEvent]);

  const prevDisabled =
    navScope === "file"
      ? selectedFileIndex === 0 && !adjacent?.previous_id
      : !adjacent?.previous_id;
  const nextDisabled =
    navScope === "file"
      ? selectedFileIndex >= files.length - 1 && !adjacent?.next_id
      : !adjacent?.next_id;
  const nextUnverifiedDisabled =
    navScope === "file"
      ? nextUnverifiedFileIndex < 0 && !adjacent?.next_unverified_id
      : !adjacent?.next_unverified_id;

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
          handlePrev();
          break;
        case "ArrowRight":
          e.preventDefault();
          handleNext();
          break;
        case "Enter":
          e.preventDefault();
          if (currentFile && !currentFile.verified) {
            verifyMutation.mutateAsync().then(() => handleNextUnverified());
          } else {
            handleNextUnverified();
          }
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
                species: label.species,
              })
            )
          ).then(() => {
            queryClient.invalidateQueries({ queryKey: ["event", eventId] });
          });
          break;
        }
        case "Tab":
          if (selectedDetectionId) {
            e.preventDefault();
            setOpenLabelPickerFor(selectedDetectionId);
          }
          break;
        case "v":
        case "V":
          e.preventDefault();
          verifyMutation.mutate();
          break;
        case "a":
        case "A":
          e.preventDefault();
          if (hiddenDetections.length > 0) {
            addBoxMutation.mutate();
          }
          break;
        case "x":
        case "X":
          if (selectedDetectionId) {
            e.preventDefault();
            deleteDetectionMutation.mutate(selectedDetectionId);
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
          if (selectedDetectionId) {
            setSelectedDetectionId(null);
          } else if (drawMode) {
            setDrawMode(false);
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
    currentFile,
    drawMode,
    filteredDetections,
    handlePrev,
    handleNext,
    handleNextUnverified,
    onClose,
    selectedDetectionId,
    verifyMutation,
    markBlankMutation,
    addBoxMutation,
    hiddenDetections,
    deleteDetectionMutation,
    shortcutLabels,
    eventId,
    queryClient,
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
                <Pencil className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => addBoxMutation.mutate()}
                disabled={hiddenDetections.length === 0 || addBoxMutation.isPending}
                title={
                  hiddenDetections.length > 0
                    ? `Add next AI detection (${hiddenDetections.length} below threshold)`
                    : "No hidden detections"
                }
              >
                <SquarePlus className="h-4 w-4" />
              </Button>
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
              {/* Detection threshold popover */}
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    title={`Detection threshold: ${(detectionThreshold * 100).toFixed(0)}%`}
                  >
                    <Scale className="h-4 w-4" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent side="right" className="w-48 p-3">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium">Detection threshold</span>
                      <span className="text-xs text-muted-foreground tabular-nums">
                        {(detectionThreshold * 100).toFixed(0)}%
                      </span>
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
                      <span className="text-xs text-muted-foreground tabular-nums">
                        {brightness}%
                      </span>
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
                      <span className="text-xs text-muted-foreground tabular-nums">
                        {contrast}%
                      </span>
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
                title={currentFile.favorited ? "Remove from favorites" : "Add to favorites"}
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
                onClick={() => exportFnRef.current?.()}
                title="Download image with annotations"
              >
                <Download className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => window.electronAPI?.showItemInFolder(currentFile.file_path)}
                title="Open in file explorer"
              >
                <FolderOpen className="h-4 w-4" />
              </Button>
            </div>
          )}

          {/* Image area */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Main image with detections */}
            <div className="flex-1 flex items-center justify-center bg-black/95 min-h-0 p-2 rounded-lg">
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
                  defaultCategory={defaultLabel.category}
                  defaultSpecies={defaultLabel.species}
                  boxesHidden={boxesHidden}
                  exportFnRef={exportFnRef}
                  zoomFnRef={zoomFnRef}
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
                detectionThreshold={detectionThreshold}
                onSelectIndex={(i) => {
                  setSelectedFileIndex(i);
                  setSelectedDetectionId(null);
                }}
              />
            )}
          </div>

          {/* Right sidebar: navigation + verification panel */}
          <div className="w-80 bg-white flex flex-col shrink-0">
            <div className="flex items-center justify-between px-3 py-1.5 shrink-0">
              <div className="flex items-center gap-0.5">
                {/* Nav scope selector */}
                <Select
                  value={navScope}
                  onValueChange={(v) => setNavScope(v as "event" | "file")}
                >
                  <SelectTrigger className="h-7 min-h-0 w-auto text-xs gap-1 px-2 py-0 mr-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="event">Navigate by event</SelectItem>
                    <SelectItem value="file">Navigate by file</SelectItem>
                  </SelectContent>
                </Select>
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
                  title="Next unverified (Enter)"
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

            {/* Verification panel */}
            {currentFile && (
              <FileVerificationPanel
                key={currentFile.id}
                file={currentFile}
                projectId={projectId}
                eventId={eventId!}
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
              />
            )}

            {/* Keyboard shortcuts */}
            <div className="mt-auto shrink-0 px-3 pb-2 relative">
              {showShortcuts && (
                <>
                <div className="fixed inset-0 z-40" onClick={() => setShowShortcuts(false)} />
                <div className="absolute bottom-10 right-6 w-[400px] mb-2 rounded-lg border bg-background shadow-lg px-4 py-3 z-50">
                  {[
                    ["Enter", "Verify + next"],
                    ["E", "Empty + next"],
                    ["← →", "Navigate"],
                    ["↑ ↓", "Select detection"],
                    ["Tab", "Change label"],
                    ["A", "Add box"],
                    ["D", "Toggle draw mode"],
                    ["X", "Delete detection"],
                    ["Scroll", "Zoom in / out"],
                    ["B (hold)", "Hide boxes"],
                    ["Esc", "Deselect / close"],
                  ].map(([key, action]) => (
                    <div key={key} className="flex items-center text-xs gap-3 h-7">
                      <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">{key}</code>
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
                      />
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
