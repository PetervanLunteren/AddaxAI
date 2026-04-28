/**
 * FileDetailModal - full-screen file viewer for the Files verify tab.
 *
 * Reuses the annotation stack (AnnotationCanvas, FileVerificationPanel,
 * LabelPicker, VideoPlayer, toolbar icons) from the Event detail modal,
 * but drops the filmstrip, the MaxN nav scope, and event-to-event
 * adjacency. Navigation is strictly file-to-file across the filtered
 * file list supplied by the Files tab.
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
  Flag,
  Heart,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  FolderOpen,
  CircleHelp,
  Play,
  Image as ImageIcon,
} from "lucide-react";
import { toast } from "sonner";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { projectsApi } from "../../api/projects";
import { cn } from "../../lib/utils";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { Slider } from "../ui/slider";
import type { EventFilterParams } from "../../api/types";
import { AnnotationCanvas } from "./AnnotationCanvas";
import { FileVerificationPanel } from "./FileVerificationPanel";
import { LabelPicker } from "./LabelPicker";
import { HelpSheet } from "./HelpSheet";
import { VideoPlayer, isPlayableVideo } from "./VideoPlayer";
import { useLabelOptions, type LabelOption } from "../../hooks/useLabelOptions";

interface FileDetailModalProps {
  fileId: string | null;
  projectId: string;
  isOpen: boolean;
  onClose: () => void;
  filters?: EventFilterParams;
}

export function FileDetailModal({
  fileId,
  projectId,
  isOpen,
  onClose,
  filters,
}: FileDetailModalProps) {
  const queryClient = useQueryClient();
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(
    null,
  );
  const [drawMode, setDrawMode] = useState(false);
  const [drawLabel, setDrawLabel] = useState<{
    category: string;
    label: string | undefined;
  } | null>(null);
  const [viewMode, setViewMode] = useState<"frame" | "video">("frame");
  const [boxesHidden, setBoxesHidden] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [shortcutLabels, setShortcutLabels] = useState<
    Record<number, LabelOption>
  >({});
  const [openLabelPickerFor, setOpenLabelPickerFor] = useState<string | null>(
    null,
  );
  const [helpOpen, setHelpOpen] = useState(false);
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

  // Fetch file data
  const { data: file } = useQuery({
    queryKey: ["file", fileId],
    queryFn: ({ signal }) => filesApi.get(fileId!, { signal }),
    enabled: !!fileId && isOpen,
  });

  // Fetch adjacent files for navigation within the filtered set
  const { data: adjacent } = useQuery({
    queryKey: ["file-adjacent", fileId, projectId, filters],
    queryFn: () =>
      filesApi.getAdjacentForVerify(fileId!, projectId, filters),
    enabled: !!fileId && isOpen,
  });

  // Fetch project for detection threshold + shortcut labels
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  });

  const { options: labelOptions, isLoading: labelOptionsLoading } =
    useLabelOptions(project?.classification_model_id ?? null, projectId);

  // Viewport resize tracking
  useEffect(() => {
    const handleResize = () =>
      setViewport({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Load shortcut label mappings from project
  useEffect(() => {
    if (project?.shortcut_labels) {
      const parsed: Record<number, LabelOption> = {};
      for (const [k, v] of Object.entries(project.shortcut_labels)) {
        parsed[Number(k)] = v as LabelOption;
      }
      setShortcutLabels(parsed);
    }
  }, [project?.shortcut_labels]);

  const updateShortcutLabels = useCallback(
    (
      updater: (
        prev: Record<number, LabelOption>,
      ) => Record<number, LabelOption>,
    ) => {
      setShortcutLabels((prev) => {
        const next = updater(prev);
        projectsApi.update(projectId, { shortcut_labels: next });
        return next;
      });
    },
    [projectId],
  );

  // Reset ephemeral state when the file changes
  useEffect(() => {
    setSelectedDetectionId(null);
    setViewMode("frame");
  }, [fileId]);

  const projectThreshold = project?.detection_threshold ?? 0;
  const detectionThreshold = localThreshold ?? projectThreshold;
  const imageFilter =
    brightness !== 50 || contrast !== 50
      ? `brightness(${brightness / 50}) contrast(${contrast / 50})`
      : undefined;

  // Default draw label: most common animal label on this file
  const defaultLabel = useMemo(() => {
    if (!file?.detections) return { category: "animal", label: undefined };
    const labelCounts = new Map<
      string,
      { count: number; category: string; label: string | undefined }
    >();
    for (const d of file.detections) {
      if (d.confidence >= detectionThreshold) {
        const key = d.label || d.category;
        const existing = labelCounts.get(key);
        if (existing) existing.count++;
        else
          labelCounts.set(key, {
            count: 1,
            category: d.category,
            label: d.label || undefined,
          });
      }
    }
    let best: { category: string; label: string | undefined } = {
      category: "animal",
      label: undefined,
    };
    let bestCount = 0;
    for (const entry of labelCounts.values()) {
      if (entry.count > bestCount) {
        bestCount = entry.count;
        best = { category: entry.category, label: entry.label };
      }
    }
    return best;
  }, [file?.detections, detectionThreshold]);
  const effectiveDrawLabel = drawLabel ?? defaultLabel;

  // Reset draw label when draw mode is toggled off
  useEffect(() => {
    if (!drawMode) setDrawLabel(null);
  }, [drawMode]);

  // Modal size: fit the image at its aspect ratio plus the fixed side chrome.
  const lastModalStyle = useRef<{ width: number; height: number } | null>(null);
  const modalStyle = useMemo(() => {
    const TOOLBAR_W = 44;
    const PANEL_W = 320;
    const IMAGE_PAD = 16;
    const maxW = viewport.width * 0.95;
    const maxH = viewport.height * 0.95;

    if (!file?.width_px || !file?.height_px) {
      return lastModalStyle.current ?? { width: maxW, height: maxH };
    }
    const maxImgW = maxW - TOOLBAR_W - PANEL_W;
    const maxImgH = maxH - IMAGE_PAD;
    const scale = Math.min(maxImgW / file.width_px, maxImgH / file.height_px, 1);
    const imgDisplayW = file.width_px * scale;
    const imgDisplayH = file.height_px * scale;
    const style = {
      width: Math.round(imgDisplayW + TOOLBAR_W + PANEL_W),
      height: Math.round(imgDisplayH + IMAGE_PAD),
    };
    lastModalStyle.current = style;
    return style;
  }, [file?.width_px, file?.height_px, viewport]);

  // Invalidations shared by all mutating actions
  const invalidateAfterMutation = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["file", fileId] });
    queryClient.invalidateQueries({ queryKey: ["files-for-verify"] });
    queryClient.invalidateQueries({ queryKey: ["files-count-for-verify"] });
    queryClient.invalidateQueries({ queryKey: ["files-verification-stats"] });
    queryClient.invalidateQueries({ queryKey: ["label-tree"] });
  }, [queryClient, fileId]);

  const verifyMutation = useMutation({
    mutationFn: () => {
      if (!file) return Promise.resolve(null);
      return filesApi.update(file.id, { verified: !file.verified });
    },
    onSuccess: invalidateAfterMutation,
  });

  const favoriteMutation = useMutation({
    mutationFn: () => {
      if (!file) return Promise.resolve(null);
      return filesApi.update(file.id, { favorited: !file.favorited });
    },
    onSuccess: invalidateAfterMutation,
  });

  const flagMutation = useMutation({
    mutationFn: () => {
      if (!file) return Promise.resolve(null);
      return filesApi.update(file.id, { flagged: !file.flagged });
    },
    onSuccess: invalidateAfterMutation,
  });

  // Hidden detections for "add box" — anything below the project
  // threshold but still in the DB. The detection worker already filters
  // at ingest (≥ 0.1 per detection_worker.py), so a floor of 0.05 here
  // is effectively "everything the model kept": surfacing every real
  // candidate without re-filtering arbitrary low-conf hits a second time.
  const hiddenDetections = useMemo(() => {
    if (!file) return [];
    return file.detections
      .filter(
        (d) => d.confidence < detectionThreshold && d.confidence >= 0.05,
      )
      .sort((a, b) => b.confidence - a.confidence);
  }, [file, detectionThreshold]);

  const addBoxMutation = useMutation({
    mutationFn: async () => {
      if (!file || hiddenDetections.length === 0) return;
      const best = hiddenDetections[0];
      await detectionsApi.create({
        file_id: file.id,
        category: best.category,
        bbox_x: best.bbox_x,
        bbox_y: best.bbox_y,
        bbox_width: best.bbox_width,
        bbox_height: best.bbox_height,
        label: best.label,
      });
      await detectionsApi.delete(best.id);
    },
    onSuccess: invalidateAfterMutation,
  });

  const handlePromoteHiddenBox = useCallback(() => {
    if (addBoxMutation.isPending) return;
    if (hiddenDetections.length === 0) {
      toast.info("Nothing to promote", {
        description:
          "This shortcut promotes the highest-confidence below-threshold AI box into a confirmed detection. The AI has no box below the project threshold for this image, so there is nothing to promote.",
      });
      return;
    }
    addBoxMutation.mutate();
  }, [addBoxMutation, hiddenDetections.length]);

  const markBlankMutation = useMutation({
    mutationFn: async () => {
      if (!file) return;
      await detectionsApi.deleteByFile(file.id);
      await filesApi.update(file.id, { verified: true });
    },
    onSuccess: invalidateAfterMutation,
  });

  // Filtered detections for Up/Down cycling (respects threshold)
  const filteredDetections = useMemo(() => {
    if (!file) return [];
    return file.detections.filter(
      (d) => d.confidence >= detectionThreshold,
    );
  }, [file, detectionThreshold]);

  const deleteDetectionMutation = useMutation({
    mutationFn: (id: string) => {
      const idx = filteredDetections.findIndex((d) => d.id === id);
      const next =
        filteredDetections[idx + 1] ?? filteredDetections[idx - 1] ?? null;
      return detectionsApi.delete(id).then(() => next);
    },
    onSuccess: (next) => {
      invalidateAfterMutation();
      setSelectedDetectionId(next?.id ?? null);
    },
  });

  // Navigation: fire a DOM event that the Files tab listens for.
  // Mirrors the pattern used by the Events tab modal.
  const navigateFile = useCallback((targetFileId: string | null | undefined) => {
    if (!targetFileId) return;
    window.dispatchEvent(
      new CustomEvent("navigate-file", { detail: targetFileId }),
    );
  }, []);

  const handlePrev = useCallback(() => {
    if (adjacent?.previous_id) navigateFile(adjacent.previous_id);
  }, [adjacent, navigateFile]);

  const handleNext = useCallback(() => {
    if (adjacent?.next_id) navigateFile(adjacent.next_id);
  }, [adjacent, navigateFile]);

  const handleNextUnverified = useCallback(() => {
    if (adjacent?.next_unverified_id) navigateFile(adjacent.next_unverified_id);
  }, [adjacent, navigateFile]);

  // Keyboard shortcuts
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      )
        return;
      if (helpOpen) return;

      switch (e.key) {
        case "ArrowUp":
          e.preventDefault();
          if (filteredDetections.length === 0) break;
          setSelectedDetectionId((prev) => {
            const idx = prev
              ? filteredDetections.findIndex((d) => d.id === prev)
              : -1;
            const next =
              idx <= 0 ? filteredDetections.length - 1 : idx - 1;
            return filteredDetections[next].id;
          });
          break;
        case "ArrowDown":
          e.preventDefault();
          if (filteredDetections.length === 0) break;
          setSelectedDetectionId((prev) => {
            const idx = prev
              ? filteredDetections.findIndex((d) => d.id === prev)
              : -1;
            const next =
              idx < 0 || idx >= filteredDetections.length - 1 ? 0 : idx + 1;
            return filteredDetections[next].id;
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
          if (file && !file.verified) {
            verifyMutation.mutateAsync().then(() => handleNextUnverified());
          } else {
            handleNextUnverified();
          }
          break;
        case "e":
        case "E":
          e.preventDefault();
          if (file) {
            markBlankMutation.mutateAsync().then(() => handleNextUnverified());
          }
          break;
        case "1":
        case "2":
        case "3":
        case "4":
        case "5": {
          const slot = parseInt(e.key);
          const label = shortcutLabels[slot];
          if (!label || !file) break;
          e.preventDefault();
          Promise.all(
            filteredDetections.map((d) =>
              detectionsApi.update(d.id, {
                category: label.category,
                label: label.label,
              }),
            ),
          ).then(invalidateAfterMutation);
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
          if (file && isPlayableVideo(file)) {
            e.preventDefault();
            setViewMode((v) => (v === "video" ? "frame" : "video"));
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
          if (e.metaKey || e.ctrlKey) break;
          e.preventDefault();
          handlePromoteHiddenBox();
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
          if (selectedDetectionId) setSelectedDetectionId(null);
          else if (drawMode) setDrawMode(false);
          else onClose();
          break;
      }
    };

    // Register in capture phase so our preventDefault on Enter fires
    // before any focused button's implicit Enter-activates-click. Without
    // capture, clicking the > nav button grabs focus, and the next Enter
    // would re-fire that button's onClick (handleNext) instead of going
    // through the case "Enter" branch below (verify + handleNextUnverified).
    window.addEventListener("keydown", handleKeyDown, true);
    return () => window.removeEventListener("keydown", handleKeyDown, true);
  }, [
    isOpen,
    helpOpen,
    file,
    drawMode,
    filteredDetections,
    handlePrev,
    handleNext,
    handleNextUnverified,
    onClose,
    selectedDetectionId,
    verifyMutation,
    flagMutation,
    markBlankMutation,
    handlePromoteHiddenBox,
    deleteDetectionMutation,
    shortcutLabels,
    invalidateAfterMutation,
  ]);

  // B key hold: momentarily hide boxes
  useEffect(() => {
    if (!isOpen) return;
    const down = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      )
        return;
      if (helpOpen) return;
      if ((e.key === "b" || e.key === "B") && !e.repeat) setBoxesHidden(true);
    };
    const up = (e: KeyboardEvent) => {
      if (e.key === "b" || e.key === "B") setBoxesHidden(false);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [isOpen, helpOpen]);

  const prevDisabled = !adjacent?.previous_id;
  const nextDisabled = !adjacent?.next_id;
  const nextUnverifiedDisabled = !adjacent?.next_unverified_id;

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
        <DialogTitle className="sr-only">File detail viewer</DialogTitle>

        <div className="flex flex-1 min-h-0 overflow-hidden">
          {/* Left toolbar */}
          {file && (
            <div className="flex flex-col items-center gap-1 px-1.5 py-2 bg-white shrink-0">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => setHelpOpen(true)}
                title="Help"
              >
                <CircleHelp className="h-4 w-4" />
              </Button>
              <div className="w-6 border-t my-0.5" />
              <Button
                variant={drawMode ? "default" : "ghost"}
                size="icon"
                className="h-8 w-8"
                onClick={() => setDrawMode(!drawMode)}
                title="Draw new box (D)"
              >
                <Pencil className="h-4 w-4" />
              </Button>
              {drawMode && (
                <div
                  className="[&_button]:h-8 [&_button]:w-8 [&_button]:p-0 [&_button]:justify-center [&_svg]:opacity-100"
                  title="Label for new boxes"
                >
                  <LabelPicker
                    value={
                      effectiveDrawLabel.label || effectiveDrawLabel.category
                    }
                    onSelect={(option) =>
                      setDrawLabel({
                        category: option.category,
                        label: option.label ?? undefined,
                      })
                    }
                    options={labelOptions}
                    isLoading={labelOptionsLoading}
                    pinnedOptions={Object.entries(shortcutLabels).map(
                      ([k, v]) => ({
                        key: Number(k),
                        option: v,
                      }),
                    )}
                    hideDot
                    hideLabel
                    projectId={projectId}
                  />
                </div>
              )}
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={handlePromoteHiddenBox}
                disabled={addBoxMutation.isPending}
                title={
                  hiddenDetections.length > 0
                    ? `Promote highest below-threshold AI box (${hiddenDetections.length} candidate${hiddenDetections.length === 1 ? "" : "s"})`
                    : "No hidden detections to promote"
                }
              >
                <SquarePlus className="h-4 w-4" />
              </Button>
              {file.source_video_id != null && (
                <Button
                  variant={viewMode === "video" ? "default" : "ghost"}
                  size="icon"
                  className="h-8 w-8"
                  onClick={() =>
                    setViewMode(viewMode === "video" ? "frame" : "video")
                  }
                  disabled={!isPlayableVideo(file)}
                  title={
                    !isPlayableVideo(file)
                      ? "Video format not supported for browser playback"
                      : viewMode === "video"
                        ? "View frame"
                        : "Play video"
                  }
                >
                  {viewMode === "video" ? (
                    <ImageIcon className="h-4 w-4" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                </Button>
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
                      <span className="text-xs font-medium">
                        View threshold
                      </span>
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
                title={file.favorited ? "Unlike" : "Like"}
              >
                <Heart
                  className={cn(
                    "h-4 w-4",
                    file.favorited && "fill-[#882000] text-[#882000]",
                  )}
                />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => flagMutation.mutate()}
                disabled={flagMutation.isPending}
                title={file.flagged ? "Remove flag" : "Flag for review (F)"}
              >
                <Flag
                  className={cn(
                    "h-4 w-4",
                    file.flagged && "fill-[#71b7ba] text-[#71b7ba]",
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
                onClick={() =>
                  window.electronAPI?.showItemInFolder(file.file_path)
                }
                title="Open in file explorer"
              >
                <FolderOpen className="h-4 w-4" />
              </Button>
            </div>
          )}

          {/* Main image/video */}
          <div className="flex-1 flex flex-col min-w-0">
            <div className="flex-1 flex items-center justify-center bg-black/95 min-h-0 p-2 rounded-lg">
              {file ? (
                viewMode === "video" && isPlayableVideo(file) ? (
                  <VideoPlayer
                    file={file}
                    detectionThreshold={detectionThreshold}
                    sourceVideoId={file.source_video_id ?? file.id}
                    allDetections={file.detections}
                    exportFnRef={exportFnRef}
                  />
                ) : (
                  <AnnotationCanvas
                    file={file}
                    detectionThreshold={detectionThreshold}
                    selectedDetectionId={selectedDetectionId}
                    onSelectDetection={setSelectedDetectionId}
                    drawMode={drawMode}
                    onDrawModeChange={setDrawMode}
                    onMutated={invalidateAfterMutation}
                    imageFilter={imageFilter}
                    defaultCategory={effectiveDrawLabel.category}
                    defaultLabel={effectiveDrawLabel.label}
                    boxesHidden={boxesHidden}
                    exportFnRef={exportFnRef}
                    zoomFnRef={zoomFnRef}
                  />
                )
              ) : (
                <div className="text-white/50">Loading...</div>
              )}
            </div>
          </div>

          {/* Right sidebar */}
          <div className="w-80 bg-white flex flex-col shrink-0">
            <div className="flex items-center justify-between px-3 py-1.5 shrink-0">
              <div className="flex items-center gap-0.5">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  disabled={prevDisabled}
                  onClick={handlePrev}
                  title="Previous file (←)"
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  disabled={nextDisabled}
                  onClick={handleNext}
                  title="Next file (→)"
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
                {adjacent && (
                  <span className="text-xs text-muted-foreground ml-2">
                    {adjacent.current_index + 1} / {adjacent.total_count}
                  </span>
                )}
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

            {/* File metadata card */}
            {file && (
              <div className="mx-3 mt-2 rounded-lg border bg-muted/40">
                <div className="flex items-center gap-2 px-3 pt-3 pb-2">
                  <h3 className="text-sm font-semibold">
                    {file.source_video_id != null ? "Video frame" : "Image"}
                  </h3>
                </div>
                <div className="px-3 pb-3 space-y-0.5 text-xs text-muted-foreground">
                  <div className="truncate">
                    {file.source_video_id != null
                      ? file.file_path.split("/").slice(-2, -1)[0]
                      : file.file_path.split("/").pop()}
                    {file.source_video_id != null &&
                      file.source_frame_number != null && (
                        <span> · frame {file.source_frame_number}</span>
                      )}
                  </div>
                  <div>
                    {formatCameraDate(
                      file.captured_at_local,
                      { day: "numeric", month: "short", year: "numeric" },
                      "en-GB",
                    )}{" "}
                    {formatCameraTime(
                      file.captured_at_local,
                      {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      },
                      "en-GB",
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Verification panel */}
            {file && (
              <FileVerificationPanel
                key={file.id}
                file={file}
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
                onDraw={() => setDrawMode(true)}
                onAddBox={handlePromoteHiddenBox}
                canAddBox={!addBoxMutation.isPending}
                onMutated={invalidateAfterMutation}
              />
            )}

            {/* Keyboard shortcuts */}
            <div className="mt-auto shrink-0 px-3 pb-2 relative">
              {showShortcuts && (
                <>
                  <div
                    className="fixed inset-0 z-40"
                    onClick={() => setShowShortcuts(false)}
                  />
                  <div className="absolute bottom-10 right-6 mb-2 rounded-lg border bg-background shadow-lg px-4 py-3 z-50 whitespace-nowrap">
                    <div className="flex gap-8">
                      <div>
                        {[
                          ["← →", "Navigate files"],
                          ["↑ ↓", "Select detection"],
                          ["Scroll", "Zoom in / out"],
                          ["P", "Toggle video / frame"],
                          ["B (hold)", "Hide boxes"],
                          ["Esc", "Deselect / close"],
                        ].map(([key, action]) => (
                          <div
                            key={key}
                            className="flex items-center text-xs gap-3 h-7"
                          >
                            <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">
                              {key}
                            </code>
                            <span>{action}</span>
                          </div>
                        ))}
                      </div>

                      <div>
                        {[
                          ["Enter", "Verify + next unverified"],
                          ["E", "Empty + next unverified"],
                          ["Tab", "Change label"],
                          ["A", "Promote highest below-threshold box"],
                          ["D", "Toggle draw mode"],
                          ["Del", "Delete detection"],
                        ].map(([key, action]) => (
                          <div
                            key={key}
                            className="flex items-center text-xs gap-3 h-7"
                          >
                            <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">
                              {key}
                            </code>
                            <span>{action}</span>
                          </div>
                        ))}

                        <div className="border-t my-2" />

                        {[1, 2, 3, 4, 5].map((n) => (
                          <div
                            key={n}
                            className="flex items-center text-xs gap-3 h-7"
                          >
                            <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">
                              {n}
                            </code>
                            <span>Change all to</span>
                            <LabelPicker
                              value={shortcutLabels[n]?.value ?? null}
                              onSelect={(option) =>
                                updateShortcutLabels((prev) => ({
                                  ...prev,
                                  [n]: option,
                                }))
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
      <HelpSheet open={helpOpen} onOpenChange={setHelpOpen} />
    </Dialog>
  );
}
