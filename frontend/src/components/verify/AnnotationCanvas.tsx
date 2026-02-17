/**
 * Interactive annotation canvas using react-konva.
 *
 * Supports drawing, moving, resizing, and deleting bounding boxes.
 * Replaces the SVG overlay when verify mode is active.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Stage, Layer, Rect, Text, Image as KonvaImage, Transformer } from "react-konva";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { detectionsApi } from "../../api/detections";
import { getCategoryColor } from "../../lib/detection-utils";
import type { FileWithDetections, DetectionResponse } from "../../api/types";

interface AnnotationCanvasProps {
  file: FileWithDetections;
  detectionThreshold: number;
  eventId: string;
  selectedDetectionId: string | null;
  onSelectDetection: (id: string | null) => void;
  drawMode: boolean;
  onDrawModeChange: (active: boolean) => void;
  imageFilter?: string;
  defaultCategory?: string;
  defaultSpecies?: string;
  exportFnRef?: React.MutableRefObject<(() => void) | null>;
  zoomFnRef?: React.MutableRefObject<{
    zoomIn: () => void;
    zoomOut: () => void;
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
  eventId,
  selectedDetectionId,
  onSelectDetection,
  drawMode,
  onDrawModeChange,
  imageFilter,
  defaultCategory,
  defaultSpecies,
  exportFnRef,
  zoomFnRef,
}: AnnotationCanvasProps) {
  const queryClient = useQueryClient();
  const stageRef = useRef<any>(null);
  const transformerRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
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

  const filteredDetections = file.detections.filter(
    (d) => d.confidence >= detectionThreshold
  );

  // Load image
  useEffect(() => {
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;
    img.onload = () => {
      setImage(img);
      updateStageSize(img.naturalWidth, img.naturalHeight);
    };
  }, [imageUrl]);

  // Update stage size based on container
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
      stage.batchDraw();

      const pixelRatio = Math.max(1, imgWidth / stage.width());
      const dataUrl = stage.toDataURL({ pixelRatio });

      // Restore transform and transformer
      transformer?.visible(wasVisible ?? true);
      stage.scale(savedScale);
      stage.position(savedPos);
      stage.batchDraw();

      // Trigger download
      const fileName =
        file.file_path.split("/").pop()?.replace(/\.[^.]+$/, "") || "image";
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
      getZoom: () => zoom,
    };
  });

  // Keyboard shortcuts
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

      if (
        (e.key === "Delete" || e.key === "Backspace") &&
        selectedDetectionId
      ) {
        e.preventDefault();
        deleteMutation.mutate(selectedDetectionId);
      }

      if (e.key === "Escape") {
        onDrawModeChange(false);
        onSelectDetection(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedDetectionId, onSelectDetection, drawMode, onDrawModeChange]);

  // Scale factor from stage to normalized coordinates
  const scaleX = stageSize.width > 0 ? imgWidth / stageSize.width : 1;
  const scaleY = stageSize.height > 0 ? imgHeight / stageSize.height : 1;

  // Convert normalized coords to stage pixels
  const toStage = (normX: number, normY: number, normW: number, normH: number) => ({
    x: (normX / scaleX) * (stageSize.width / imgWidth) * imgWidth,
    y: (normY / scaleY) * (stageSize.height / imgHeight) * imgHeight,
    width: (normW / scaleX) * (stageSize.width / imgWidth) * imgWidth,
    height: (normH / scaleY) * (stageSize.height / imgHeight) * imgHeight,
  });

  // Simpler conversion: just scale normalized -> stage pixels
  const normToPixel = (v: number, dim: number) =>
    v * (dim === imgWidth ? stageSize.width : stageSize.height);
  const pixelToNorm = (v: number, dim: number) =>
    v / (dim === imgWidth ? stageSize.width : stageSize.height);

  // Create detection mutation
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
        species: defaultSpecies,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
      onDrawModeChange(false);
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
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
    },
  });

  // Delete detection mutation
  const deleteMutation = useMutation({
    mutationFn: (id: string) => detectionsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
      onSelectDetection(null);
    },
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

  // Zoom via scroll wheel / trackpad pinch
  const handleWheel = (e: any) => {
    e.evt.preventDefault();
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
        <div className="absolute top-2 left-2 z-10 bg-blue-500 text-white text-xs px-2 py-1 rounded">
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
          {filteredDetections.map((detection) => {
            const x = normToPixel(detection.bbox_x, imgWidth);
            const y = normToPixel(detection.bbox_y, imgHeight);
            const w = normToPixel(detection.bbox_width, imgWidth);
            const h = normToPixel(detection.bbox_height, imgHeight);
            const color = getCategoryColor(detection.category);
            const isSelected = selectedDetectionId === detection.id;

            const hasSpecies = !!detection.species;
            const labelHeight = hasSpecies ? 32 : 18;
            const label = hasSpecies
              ? `${detection.category} ${(detection.confidence * 100).toFixed(0)}%\n${detection.species} ${((detection.species_confidence ?? detection.confidence) * 100).toFixed(0)}%`
              : `${detection.category} ${(detection.confidence * 100).toFixed(0)}%`;

            return (
              <React.Fragment key={detection.id}>
                {/* Label background */}
                <Rect
                  x={x}
                  y={y - labelHeight}
                  width={Math.max(w, 60)}
                  height={labelHeight}
                  fill={color}
                  opacity={0.8}
                  listening={false}
                />
                {/* Label text */}
                <Text
                  x={x + 3}
                  y={y - labelHeight + 2}
                  text={label}
                  fill="white"
                  fontSize={12}
                  fontStyle="bold"
                  lineHeight={1.2}
                  listening={false}
                />
                {/* Bounding box */}
                <Rect
                  id={`det-${detection.id}`}
                  x={x}
                  y={y}
                  width={w}
                  height={h}
                  stroke={color}
                  strokeWidth={isSelected ? 3 : 2}
                  fill={color}
                  opacity={0.2}
                  draggable={!drawMode}
                  onClick={() => onSelectDetection(detection.id)}
                  onTap={() => onSelectDetection(detection.id)}
                  onDragEnd={(e) => handleDragEnd(detection, e)}
                  onTransformEnd={(e) => handleTransformEnd(detection, e)}
                />
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
              stroke="rgb(34, 197, 94)"
              strokeWidth={2}
              dash={[5, 5]}
              listening={false}
            />
          )}

          {/* Transformer for selected box */}
          <Transformer
            ref={transformerRef}
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
            borderStroke="#3b82f6"
            anchorFill="#3b82f6"
            anchorSize={8}
          />
        </Layer>
      </Stage>
    </div>
  );
}
