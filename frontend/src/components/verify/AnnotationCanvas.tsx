/**
 * Interactive annotation canvas using react-konva.
 *
 * Supports drawing, moving, resizing, and deleting bounding boxes.
 * Replaces the SVG overlay when verify mode is active.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Stage, Layer, Rect, Text, Image as KonvaImage, Transformer, Shape, Group } from "react-konva";
import { useMutation } from "@tanstack/react-query";
import { ImageOff } from "lucide-react";
import { detectionsApi } from "../../api/detections";
import { reportMissingMedia } from "../../hooks/useBrokenDeployments";
import { getDetectionColor, shouldDrawBbox } from "../../lib/detection-utils";
import { basename } from "../../lib/path-utils";
import {
  roundedRectPath,
  computePillLayout,
  placePill,
  PILL_PAD_Y,
  LINE_GAP,
  FONT,
  TEXT_START_X,
  BBOX_CORNER_RADIUS,
  BBOX_OPACITY,
  DIM_FILL,
  PILL_BG,
} from "../../lib/detection-overlay";
import type { FileWithDetections, DetectionResponse } from "../../api/types";
import { useSpeciesColorsVersion } from "../../utils/species-colors";

interface AnnotationCanvasProps {
  file: FileWithDetections;
  detectionThreshold: number;
  /** Draw only the boxes a person put there, ignoring `detectionThreshold`
   *  and every box the detector found. The empties viewer sets it: that
   *  page says the picture is empty, so a machine box on it argues with
   *  the page. See `shouldDrawBbox`. */
  humanDrawnOnly?: boolean;
  selectedDetectionId: string | null;
  onSelectDetection: (id: string | null) => void;
  /** Fired when the user clicks a box's label pill — opens the relabel
   *  dialog for that detection. The selected box is highlighted so it's
   *  clear which one is being edited. */
  onRequestRelabel?: (id: string) => void;
  drawMode: boolean;
  onDrawModeChange: (active: boolean) => void;
  /**
   * Called after a successful create / update / delete mutation. The parent
   * is responsible for invalidating its own query keys (events vs files vs
   * grid lists). The canvas itself does not know which keys to touch.
   */
  onMutated?: () => void;
  /** Fired with the new detection's id right after a box is drawn, on
   *  top of `onMutated`. Separate because the owner may want to follow
   *  a fresh box up (asking which species it is), which is not true of
   *  a move, a resize or a delete. */
  onCreated?: (detectionId: string) => void;
  /** Shift+wheel steps to the previous/next frame instead of zooming.
   *  `delta` is the wheel deltaY (negative = up/previous). */
  onScrubFrame?: (delta: number) => void;
  /** View-only: boxes render but are not interactive (no drag / select /
   *  relabel), so a drag anywhere pans the zoomed image. Used by the Counts
   *  modal, where label/box editing lives on the Labels page. */
  readOnly?: boolean;
  imageFilter?: string;
  defaultCategory?: string;
  defaultLabel?: string;
  boxesHidden?: boolean;
  exportFnRef?: React.MutableRefObject<(() => void) | null>;
  zoomFnRef?: React.MutableRefObject<{
    zoomIn: () => void;
    zoomOut: () => void;
    resetZoom: () => void;
    getZoom: () => number;
  } | null>;
}

interface DrawingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export function AnnotationCanvas({
  file,
  detectionThreshold,
  humanDrawnOnly,
  selectedDetectionId,
  onSelectDetection,
  onRequestRelabel,
  drawMode,
  onDrawModeChange,
  onMutated,
  onCreated,
  onScrubFrame,
  readOnly,
  imageFilter,
  defaultCategory,
  defaultLabel,
  boxesHidden,
  exportFnRef,
  zoomFnRef,
}: AnnotationCanvasProps) {
  // Repaint when the project's colour map lands or changes.
  useSpeciesColorsVersion();
  const stageRef = useRef<any>(null);
  const transformerRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [loading, setLoading] = useState(true);
  // The picture is gone from disk (a moved / disconnected deployment
  // folder). Without this the stage stayed black and the boxes drew on
  // top of nothing, so the file read as an all-black photo the AI had
  // somehow found animals in.
  const [imageFailed, setImageFailed] = useState(false);
  const [stageSize, setStageSize] = useState({ width: 800, height: 600 });
  const [drawingBox, setDrawingBox] = useState<DrawingBox | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [stagePos, setStagePos] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const lastPanPosRef = useRef({ x: 0, y: 0 });

  const imageUrl = `/api/files/${file.id}/image`;
  const imgWidth = file.width_px || 1;
  const imgHeight = file.height_px || 1;

  // For video files: only show detections whose frame_number matches
  // the best frame the canvas is actually rendering. Event-level
  // observations (no bbox) never draw. `shouldDrawBbox` centralises
  // both gates.
  const filteredDetections = file.detections.filter((d) =>
    shouldDrawBbox(d, file, detectionThreshold, { humanDrawnOnly }),
  );

  // Update stage size based on container. Defined before the image-load
  // effect below because that effect lists it as a dependency (a const from
  // useCallback is in the temporal dead zone until its own line runs).
  const updateStageSize = useCallback(
    (naturalWidth: number, naturalHeight: number) => {
      if (!containerRef.current) return;
      const containerWidth = containerRef.current.clientWidth;
      const containerHeight = containerRef.current.clientHeight;

      const scaleX = containerWidth / naturalWidth;
      const scaleY = containerHeight / naturalHeight;
      const scale = Math.min(scaleX, scaleY, 1);

      setStageSize({
        width: naturalWidth * scale,
        height: naturalHeight * scale,
      });
    },
    []
  );

  // Load image. Keep the previous image on screen while the next one loads
  // (so rapid navigation / auto-play doesn't flash black between frames), and
  // hide the boxes while `loading` so the old image is never shown with the
  // new file's detections. Stale loads (A → B → C where B's onload fires after
  // C has started) are ignored via `cancelled`.
  useEffect(() => {
    let cancelled = false;
    // Clear the previous file's verdict up front, so stepping from a
    // missing file to a present one does not keep the placeholder.
    setImageFailed(false);
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    const settle = () => {
      if (cancelled) return;
      setImage(img);
      updateStageSize(img.naturalWidth, img.naturalHeight);
      setImageFailed(false);
      setLoading(false);
    };
    img.onload = settle;
    img.onerror = () => {
      if (cancelled) return;
      setLoading(false);
      setImageFailed(true);
      reportMissingMedia(file.deployment_id);
    };
    img.src = imageUrl;
    // Auto-play prefetches the next frame, so by the time we get here the
    // image is often already cached and decoded. A cached image is `complete`
    // as soon as `src` is set, and in that case `onload` may never fire —
    // which would leave `loading` stuck true and the boxes never rendered
    // (the intermittent "no boxes at all" during a loop). Resolve it
    // synchronously instead: image + boxes swap together, with no hidden-box
    // gap. Only mark `loading` when the bitmap genuinely isn't ready yet.
    if (img.complete && img.naturalWidth > 0) {
      settle();
    } else {
      setLoading(true);
    }
    return () => {
      cancelled = true;
    };
  }, [imageUrl, updateStageSize, file.deployment_id]);

  // Resize observer
  useEffect(() => {
    if (!containerRef.current || !image) return;
    const observer = new ResizeObserver(() => {
      updateStageSize(image.naturalWidth, image.naturalHeight);
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [image, updateStageSize]);

  // Reset zoom when file changes
  useEffect(() => {
    setZoom(1);
    setStagePos({ x: 0, y: 0 });
  }, [file.id]);

  // Update transformer when selection changes
  useEffect(() => {
    if (!transformerRef.current || !stageRef.current) return;

    const stage = stageRef.current;
    if (selectedDetectionId) {
      const node = stage.findOne(`#det-${selectedDetectionId}`);
      if (node) {
        transformerRef.current.nodes([node]);
        transformerRef.current.getLayer()?.batchDraw();
        return;
      }
    }
    transformerRef.current.nodes([]);
    transformerRef.current.getLayer()?.batchDraw();
  }, [selectedDetectionId, filteredDetections]);

  // Pulse selected detection bbox on selection change
  useEffect(() => {
    if (!selectedDetectionId || !stageRef.current) return;
    const node = stageRef.current.findOne(`#det-${selectedDetectionId}`);
    if (!node) return;
    node.to({ opacity: 1, duration: 0.15, onFinish: () => {
      node.to({ opacity: 0.5, duration: 0.25 });
    }});
  }, [selectedDetectionId]);

  // Register export function for download button
  useEffect(() => {
    if (!exportFnRef) return;
    exportFnRef.current = () => {
      const stage = stageRef.current;
      if (!stage) return;
      // Save current transform and reset for clean export
      const savedScale = { x: stage.scaleX(), y: stage.scaleY() };
      const savedPos = { x: stage.x(), y: stage.y() };
      stage.scale({ x: 1, y: 1 });
      stage.position({ x: 0, y: 0 });

      const transformer = transformerRef.current;
      const wasVisible = transformer?.visible();
      transformer?.visible(false);
      // The annotated download always carries the labels and the spotlight
      // dim. Labels are already on screen everywhere now; the spotlight is
      // still hidden on screen in read-only (Counts) mode, so force it on
      // just for the export and restore it after.
      const pills = stage.find(".label-pill");
      const spots = stage.find(".spotlight");
      [...pills, ...spots].forEach((n: any) => n.visible(true));
      stage.batchDraw();

      const pixelRatio = Math.max(1, imgWidth / stage.width());
      const dataUrl = stage.toDataURL({ pixelRatio });

      // Restore transform, transformer, and on-screen annotation visibility:
      // pills stay shown, the spotlight reverts to its read-only state.
      transformer?.visible(wasVisible ?? true);
      pills.forEach((n: any) => n.visible(true));
      spots.forEach((n: any) => n.visible(!readOnly));
      stage.scale(savedScale);
      stage.position(savedPos);
      stage.batchDraw();

      // Trigger download
      const fileName =
        basename(file.file_path).replace(/\.[^.]+$/, "") || "image";
      const link = document.createElement("a");
      link.download = `${fileName}_annotated.png`;
      link.href = dataUrl;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };
  }, [exportFnRef, image, file.file_path, imgWidth]);

  // Register zoom functions for external buttons
  useEffect(() => {
    if (!zoomFnRef) return;
    const zoomBy = (direction: 1 | -1) => {
      const factor = 1.3;
      const oldScale = zoom;
      const newScale = Math.min(
        5,
        Math.max(1, oldScale * (direction > 0 ? factor : 1 / factor))
      );
      if (newScale === 1) {
        setZoom(1);
        setStagePos({ x: 0, y: 0 });
        return;
      }
      // Zoom toward center of stage
      const cx = stageSize.width / 2;
      const cy = stageSize.height / 2;
      const mousePointTo = {
        x: (cx - stagePos.x) / oldScale,
        y: (cy - stagePos.y) / oldScale,
      };
      const newPos = clampPos(
        { x: cx - mousePointTo.x * newScale, y: cy - mousePointTo.y * newScale },
        newScale
      );
      setZoom(newScale);
      setStagePos(newPos);
    };
    zoomFnRef.current = {
      zoomIn: () => zoomBy(1),
      zoomOut: () => zoomBy(-1),
      resetZoom: () => { setZoom(1); setStagePos({ x: 0, y: 0 }); },
      getZoom: () => zoom,
    };
  });

  // Keyboard shortcut: D to toggle draw mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      if (e.key === "d" || e.key === "D") {
        e.preventDefault();
        onDrawModeChange(!drawMode);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [drawMode, onDrawModeChange]);

  // Simpler conversion: just scale normalized -> stage pixels
  const normToPixel = (v: number, dim: number) =>
    v * (dim === imgWidth ? stageSize.width : stageSize.height);
  const pixelToNorm = (v: number, dim: number) =>
    v / (dim === imgWidth ? stageSize.width : stageSize.height);

  // Create detection mutation. The drawn box is anchored to whatever
  // pixels the canvas is rendering — for videos that's the best frame,
  // so we stamp `frame_number = best_frame_number` on the new row.
  // Without this, `shouldDrawBbox` filters the fresh detection out of
  // the overlay (null !== best_frame_number) and the user sees the
  // box disappear as soon as they release the mouse.
  const createMutation = useMutation({
    mutationFn: (box: DrawingBox) => {
      const normX = pixelToNorm(box.x, imgWidth);
      const normY = pixelToNorm(box.y, imgHeight);
      const normW = pixelToNorm(box.width, imgWidth);
      const normH = pixelToNorm(box.height, imgHeight);
      return detectionsApi.create({
        file_id: file.id,
        category: defaultCategory || "animal",
        bbox_x: Math.max(0, Math.min(1, normX)),
        bbox_y: Math.max(0, Math.min(1, normY)),
        bbox_width: Math.max(0, Math.min(1, normW)),
        bbox_height: Math.max(0, Math.min(1, normH)),
        label: defaultLabel,
        frame_number:
          file.file_type === "video" ? file.best_frame_number ?? null : null,
      });
    },
    onSuccess: (created) => {
      onMutated?.();
      onCreated?.(created.id);
    },
  });

  // Update detection mutation (for move/resize)
  const updateMutation = useMutation({
    mutationFn: ({
      id,
      x,
      y,
      width,
      height,
    }: {
      id: string;
      x: number;
      y: number;
      width: number;
      height: number;
    }) => {
      return detectionsApi.update(id, {
        bbox_x: Math.max(0, Math.min(1, pixelToNorm(x, imgWidth))),
        bbox_y: Math.max(0, Math.min(1, pixelToNorm(y, imgHeight))),
        bbox_width: Math.max(0, Math.min(1, pixelToNorm(width, imgWidth))),
        bbox_height: Math.max(0, Math.min(1, pixelToNorm(height, imgHeight))),
      });
    },
    onSuccess: () => onMutated?.(),
  });

  // Convert screen pointer position to stage coordinates (accounting for zoom/pan)
  const getStagePointerPos = () => {
    const pointer = stageRef.current?.getPointerPosition();
    if (!pointer) return null;
    return {
      x: (pointer.x - stagePos.x) / zoom,
      y: (pointer.y - stagePos.y) / zoom,
    };
  };

  // Clamp pan position so image stays visible
  const clampPos = (pos: { x: number; y: number }, z: number) => ({
    x: Math.min(0, Math.max(pos.x, stageSize.width * (1 - z))),
    y: Math.min(0, Math.max(pos.y, stageSize.height * (1 - z))),
  });

  // Plain wheel = zoom; Shift+wheel = step to the prev/next frame.
  const handleWheel = (e: any) => {
    e.evt.preventDefault();
    if (e.evt.shiftKey && onScrubFrame) {
      // Holding Shift makes the browser remap the wheel to the horizontal
      // axis (deltaY becomes 0, the value lands in deltaX), so scrub on
      // whichever axis actually carries the scroll.
      const d =
        Math.abs(e.evt.deltaY) >= Math.abs(e.evt.deltaX)
          ? e.evt.deltaY
          : e.evt.deltaX;
      onScrubFrame(d);
      return;
    }
    const stage = stageRef.current;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const oldScale = zoom;
    const mousePointTo = {
      x: (pointer.x - stagePos.x) / oldScale,
      y: (pointer.y - stagePos.y) / oldScale,
    };

    const direction = e.evt.deltaY > 0 ? -1 : 1;
    const factor = 1.1;
    const newScale = Math.min(
      5,
      Math.max(1, oldScale * (direction > 0 ? factor : 1 / factor))
    );

    if (newScale === 1) {
      setZoom(1);
      setStagePos({ x: 0, y: 0 });
      return;
    }

    const newPos = clampPos(
      {
        x: pointer.x - mousePointTo.x * newScale,
        y: pointer.y - mousePointTo.y * newScale,
      },
      newScale
    );
    setZoom(newScale);
    setStagePos(newPos);
  };

  // Mouse handlers for drawing and panning
  const handleMouseDown = (e: any) => {
    const target = e.target;
    const isEmptyArea =
      target === stageRef.current || target.className === "Image";

    if (drawMode) {
      const pos = getStagePointerPos();
      if (!pos) return;
      setIsDrawing(true);
      setDrawingBox({ x: pos.x, y: pos.y, width: 0, height: 0 });
      return;
    }

    if (isEmptyArea) {
      onSelectDetection(null);
      if (zoom > 1) {
        setIsPanning(true);
        const pointer = stageRef.current.getPointerPosition();
        if (pointer) lastPanPosRef.current = { x: pointer.x, y: pointer.y };
      }
    }
  };

  const handleMouseMove = () => {
    if (isPanning) {
      const pointer = stageRef.current?.getPointerPosition();
      if (!pointer) return;
      const dx = pointer.x - lastPanPosRef.current.x;
      const dy = pointer.y - lastPanPosRef.current.y;
      lastPanPosRef.current = { x: pointer.x, y: pointer.y };
      setStagePos((prev) => clampPos({ x: prev.x + dx, y: prev.y + dy }, zoom));
      return;
    }

    if (!isDrawing || !drawingBox) return;
    const pos = getStagePointerPos();
    if (!pos) return;
    setDrawingBox({
      ...drawingBox,
      width: pos.x - drawingBox.x,
      height: pos.y - drawingBox.y,
    });
  };

  const handleMouseUp = () => {
    if (isPanning) {
      setIsPanning(false);
      return;
    }

    if (!isDrawing || !drawingBox) return;
    setIsDrawing(false);

    // Normalize negative dimensions
    const box = {
      x: drawingBox.width < 0 ? drawingBox.x + drawingBox.width : drawingBox.x,
      y:
        drawingBox.height < 0
          ? drawingBox.y + drawingBox.height
          : drawingBox.y,
      width: Math.abs(drawingBox.width),
      height: Math.abs(drawingBox.height),
    };

    // Minimum size check (at least 10px in stage coords)
    if (box.width > 10 && box.height > 10) {
      createMutation.mutate(box);
      // One-shot: a committed box returns the canvas to normal so the
      // common "add one box" case costs no extra exit click. Consistent
      // with the other one-shot create-actions (Add observation). Draw
      // several by re-pressing D / the toolbar button. A stray sub-minimum
      // click below is NOT a commit, so it leaves draw mode armed.
      onDrawModeChange(false);
    }

    setDrawingBox(null);
  };

  // Handle drag end for moving boxes
  const handleDragEnd = (detection: DetectionResponse, e: any) => {
    const node = e.target;
    updateMutation.mutate({
      id: detection.id,
      x: node.x(),
      y: node.y(),
      width: node.width() * node.scaleX(),
      height: node.height() * node.scaleY(),
    });
  };

  // Handle transform end for resizing
  const handleTransformEnd = (detection: DetectionResponse, e: any) => {
    const node = e.target;
    const newWidth = node.width() * node.scaleX();
    const newHeight = node.height() * node.scaleY();
    node.scaleX(1);
    node.scaleY(1);

    updateMutation.mutate({
      id: detection.id,
      x: node.x(),
      y: node.y(),
      width: newWidth,
      height: newHeight,
    });
  };

  // Nothing to annotate when the picture is gone, so the whole stage is
  // replaced rather than left black with boxes floating on it. The banner
  // on Labels / Counts carries the "why" and the way to fix it; this only
  // has to stop the file reading as a corrupted photo.
  if (imageFailed) {
    return (
      <div
        ref={containerRef}
        className="relative w-full h-full flex flex-col items-center justify-center gap-3 bg-neutral-200 dark:bg-neutral-800"
      >
        <ImageOff className="h-10 w-10 text-neutral-400 dark:text-neutral-500" />
        <p className="text-sm text-neutral-500 dark:text-neutral-400">
          This picture can't be found on disk
        </p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full flex items-center justify-center"
      style={{
        cursor: drawMode
          ? "crosshair"
          : isPanning
            ? "grabbing"
            : zoom > 1
              ? "grab"
              : "default",
        filter: imageFilter,
      }}
    >
      {/* Draw mode indicator */}
      {drawMode && (
        <div className="absolute top-2 left-2 z-10 bg-primary text-white text-xs px-2 py-1 rounded">
          Drawing mode - click and drag to draw a box (Esc to cancel)
        </div>
      )}

      <Stage
        ref={stageRef}
        width={stageSize.width}
        height={stageSize.height}
        scaleX={zoom}
        scaleY={zoom}
        x={stagePos.x}
        y={stagePos.y}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        {/* Background image layer */}
        <Layer>
          {image && (
            <KonvaImage
              image={image}
              width={stageSize.width}
              height={stageSize.height}
            />
          )}
        </Layer>

        {/* Detections layer */}
        <Layer>
          {/* Spotlight dim overlay — darkens everything outside bounding
              boxes. Hidden on screen in read-only (Counts) mode so the focus
              is a clean study view, but still drawn into the download. */}
          {!loading && !boxesHidden && filteredDetections.length > 0 && (
            <Shape
              name="spotlight"
              visible={!readOnly}
              sceneFunc={(context) => {
                const ctx = (context as any)._context as CanvasRenderingContext2D;
                ctx.save();
                // Dark overlay over entire stage
                ctx.fillStyle = DIM_FILL;
                ctx.fillRect(0, 0, stageSize.width, stageSize.height);
                // Erase box regions — overlapping boxes merge cleanly
                ctx.globalCompositeOperation = "destination-out";
                ctx.fillStyle = "black";
                for (const det of filteredDetections) {
                  ctx.beginPath();
                  roundedRectPath(
                    ctx,
                    normToPixel(det.bbox_x, imgWidth),
                    normToPixel(det.bbox_y, imgHeight),
                    normToPixel(det.bbox_width, imgWidth),
                    normToPixel(det.bbox_height, imgHeight),
                    BBOX_CORNER_RADIUS,
                  );
                  ctx.fill();
                }
                ctx.restore();
              }}
              listening={false}
            />
          )}

          {!loading && !boxesHidden && filteredDetections.map((detection) => {
            const x = normToPixel(detection.bbox_x, imgWidth);
            const y = normToPixel(detection.bbox_y, imgHeight);
            const w = normToPixel(detection.bbox_width, imgWidth);
            const h = normToPixel(detection.bbox_height, imgHeight);
            const color = getDetectionColor(detection);
            const isSelected = selectedDetectionId === detection.id;
            const pill = computePillLayout(detection);
            // Read-only focus uses a slightly bolder colored line so it
            // stays dominant.
            const colorW = readOnly
              ? isSelected
                ? 2.9
                : 2.25
              : isSelected
                ? 3
                : 2;
            // In read-only mode keep the line a constant screen width by
            // dividing out the zoom, so zooming into a small animal gives a
            // thin border (not one scaled up to eat the pixels).
            const zDiv = readOnly ? zoom : 1;
            const lineW = colorW / zDiv;

            const { x: pillX, y: pillY } = placePill(
              { x, y, width: w, height: h },
              { width: pill.pillWidth, height: pill.pillHeight },
              stageSize,
            );

            return (
              <React.Fragment key={detection.id}>
                {/* Bounding box (stroke only, rounded) */}
                <Rect
                  id={`det-${detection.id}`}
                  x={x}
                  y={y}
                  width={w}
                  height={h}
                  stroke={color}
                  strokeWidth={lineW}
                  opacity={readOnly ? 1 : BBOX_OPACITY}
                  fill="transparent"
                  cornerRadius={BBOX_CORNER_RADIUS}
                  listening={!readOnly}
                  draggable={!drawMode && !readOnly}
                  onClick={() => onSelectDetection(detection.id)}
                  onTap={() => onSelectDetection(detection.id)}
                  onDragEnd={(e) => handleDragEnd(detection, e)}
                  onTransformEnd={(e) => handleTransformEnd(detection, e)}
                />
                {/* Label pill — click to relabel this box in place. Shown in
                    read-only (Counts) mode too: when skimming or looping an
                    event you can't cross-reference box colours against a
                    legend at speed, so the per-box species is what makes the
                    scene readable. The show/hide AI overlays toggle (B) clears
                    them along with the boxes when a frame gets too busy. */}
                <Group
                  name="label-pill"
                  visible={true}
                  x={pillX}
                  y={pillY}
                  listening={!drawMode}
                  onClick={() => {
                    onSelectDetection(detection.id);
                    onRequestRelabel?.(detection.id);
                  }}
                  onTap={() => {
                    onSelectDetection(detection.id);
                    onRequestRelabel?.(detection.id);
                  }}
                  onMouseEnter={(e) => {
                    const c = e.target.getStage()?.container();
                    if (c) c.style.cursor = "pointer";
                  }}
                  onMouseLeave={(e) => {
                    const c = e.target.getStage()?.container();
                    if (c) c.style.cursor = drawMode ? "crosshair" : "default";
                  }}
                >
                  <Rect
                    width={pill.pillWidth}
                    height={pill.pillHeight}
                    fill={PILL_BG}
                    cornerRadius={BBOX_CORNER_RADIUS}
                  />
                  <Text
                    x={TEXT_START_X}
                    y={PILL_PAD_Y}
                    text={pill.categoryText}
                    fill="white"
                    fontSize={FONT}
                  />
                  {pill.hasLabel && (
                    <Text
                      x={TEXT_START_X}
                      y={PILL_PAD_Y + FONT + LINE_GAP}
                      text={pill.labelText}
                      fill="white"
                      fontSize={FONT}
                    />
                  )}
                </Group>
              </React.Fragment>
            );
          })}

          {/* Drawing box preview */}
          {drawingBox && (
            <Rect
              x={drawingBox.x}
              y={drawingBox.y}
              width={drawingBox.width}
              height={drawingBox.height}
              stroke="#0f6064"
              strokeWidth={2}
              dash={[5, 5]}
              listening={false}
            />
          )}

          {/* Transformer for selected box */}
          <Transformer
            ref={transformerRef}
            visible={!boxesHidden}
            rotateEnabled={false}
            keepRatio={false}
            enabledAnchors={[
              "top-left",
              "top-right",
              "bottom-left",
              "bottom-right",
              "middle-left",
              "middle-right",
              "top-center",
              "bottom-center",
            ]}
            borderStroke="#0f6064"
            anchorFill="#0f6064"
            anchorSize={8}
          />
        </Layer>
      </Stage>
    </div>
  );
}
