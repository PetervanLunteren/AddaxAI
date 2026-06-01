/**
 * DetectionDetailModal - centered two-panel dialog showing full detection details.
 *
 * Left panel: source image with bbox overlay (dark background).
 * Right panel: crop, metadata, label agreement, and action buttons.
 * Supports prev/next navigation and verify-and-advance (Enter) for rapid review.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Ban, Check, Tag, ChevronLeft, ChevronRight, ChevronsRight, X } from "lucide-react";
import { basename } from "../../lib/path-utils";
import { toast } from "sonner";
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog";
import { Button } from "../ui/button";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { observationsApi } from "../../api/observations";
import { API_BASE_URL } from "../../lib/api-client";
import { cn } from "../../lib/utils";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { getDetectionDisplayName } from "../../lib/detection-utils";
import {
  computePillLayout,
  svgRoundedRectPath,
  PILL_PAD_X, PILL_PAD_Y, DOT_R, LINE_GAP,
  FONT_SM, FONT_LG, TEXT_START_X,
  BBOX_STROKE_WIDTH, BBOX_OPACITY, BBOX_CORNER_RADIUS,
  DIM_FILL, PILL_BG,
} from "../../lib/detection-overlay";
import type { DetectionSummary } from "../../api/types";
import type { LabelOption } from "../../hooks/useLabelOptions";
import { LabelPicker } from "./LabelPicker";

interface DetectionDetailModalProps {
  detection: DetectionSummary | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onActionComplete: () => void;
  onRelabel?: (detectionId: string, label: string, category: string) => void;
  /** Optimistic mark-false callback so parent can patch local state before navigating. */
  onMarkFalse?: (detectionId: string) => void;
  /** Optimistic verify callback so parent can patch local state before navigating. */
  onVerify?: (detectionId: string, verified?: boolean) => void;
  /** Navigate to adjacent detection. Return false if at boundary. */
  onNavigate?: (direction: "prev" | "next" | "nextUnverified") => boolean;
  /** Current position, e.g. "3 / 48" */
  position?: string;
  /** Project ID for fetching nearest neighbors. */
  projectId?: string;
  /** Available label options for the relabel picker. */
  labelOptions?: LabelOption[];
  labelOptionsLoading?: boolean;
}

export function DetectionDetailModal({
  detection,
  open,
  onOpenChange,
  onActionComplete,
  onRelabel,
  onMarkFalse,
  onVerify,
  onNavigate,
  position,
  projectId,
  labelOptions = [],
  labelOptionsLoading = false,
}: DetectionDetailModalProps) {
  const queryClient = useQueryClient();
  const [viewport, setViewport] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });
  const [forceOpenPicker, setForceOpenPicker] = useState(false);

  // Source-image zoom (scroll wheel) + pan (drag while zoomed). The
  // image-detail modals zoom via a Konva stage, but this modal shows a
  // plain <img> + SVG overlay, so we scale their shared wrapper with a
  // CSS transform instead. Same scroll-to-zoom interaction, lighter
  // mechanism. The SVG sits exactly over the image, so scaling the
  // wrapper keeps the bbox aligned for free.
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const imagePanelRef = useRef<HTMLDivElement | null>(null);
  const panStartRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const wheelTeardownRef = useRef<(() => void) | null>(null);

  // Callback ref: attach the wheel listener the instant the panel DOM
  // node mounts (and detach on unmount). A native listener is required
  // because React makes its synthetic onWheel passive, so e.preventDefault
  // there is a no-op and the page would scroll instead of zooming. Doing
  // this via useEffect+ref proved racy with the dialog portal, so the
  // callback ref guarantees the node exists when we bind.
  const attachImagePanel = useCallback((node: HTMLDivElement | null) => {
    wheelTeardownRef.current?.();
    wheelTeardownRef.current = null;
    imagePanelRef.current = node;
    if (!node) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = node.getBoundingClientRect();
      const cx = e.clientX - rect.left - rect.width / 2;
      const cy = e.clientY - rect.top - rect.height / 2;
      setZoom((z0) => {
        const factor = e.deltaY > 0 ? 0.9 : 1.1;
        const z1 = Math.min(5, Math.max(1, z0 * factor));
        if (z1 === z0) return z0;
        setPan((p0) => {
          if (z1 === 1) return { x: 0, y: 0 };
          const ratio = z1 / z0;
          return { x: cx - (cx - p0.x) * ratio, y: cy - (cy - p0.y) * ratio };
        });
        return z1;
      });
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    wheelTeardownRef.current = () => node.removeEventListener("wheel", onWheel);
  }, []);

  // Track viewport size for responsive modal sizing
  useEffect(() => {
    const handleResize = () => {
      setViewport({ width: window.innerWidth, height: window.innerHeight });
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Reset zoom/pan when the detection changes or the modal closes.
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [detection?.detection_id, open]);

  // Drag to pan while zoomed in. Global listeners so the drag survives
  // the cursor leaving the panel. Pan is clamped to the scaled overflow
  // so the image can't be dragged out of its frame.
  useEffect(() => {
    if (!isPanning) return;
    const onMove = (e: PointerEvent) => {
      const start = panStartRef.current;
      if (!start) return;
      let nx = start.panX + (e.clientX - start.x);
      let ny = start.panY + (e.clientY - start.y);
      const rect = imagePanelRef.current?.getBoundingClientRect();
      if (rect) {
        const maxX = (rect.width / 2) * (zoom - 1);
        const maxY = (rect.height / 2) * (zoom - 1);
        nx = Math.max(-maxX, Math.min(maxX, nx));
        ny = Math.max(-maxY, Math.min(maxY, ny));
      }
      setPan({ x: nx, y: ny });
    };
    const onUp = () => {
      setIsPanning(false);
      panStartRef.current = null;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [isPanning, zoom]);

  // Load full file data to get image dimensions and detection bbox
  const { data: fileData } = useQuery({
    queryKey: ["file", detection?.file_id],
    queryFn: () => filesApi.get(detection!.file_id),
    enabled: open && !!detection?.file_id,
  });

  // Fetch 10 nearest neighbors for the Label Agreement thumbnails
  const { data: neighborsData } = useQuery({
    queryKey: ["detection-neighbors", detection?.detection_id],
    queryFn: () =>
      observationsApi.search(projectId!, {
        anchor_detection_id: detection!.detection_id,
        limit: 11,
      }),
    enabled: open && !!detection?.detection_id && !!projectId,
  });

  const verifyMutation = useMutation({
    mutationFn: () =>
      detectionsApi.verify(detection!.detection_id, !detection!.verified),
    onSuccess: () => {
      onVerify?.(detection!.detection_id, !detection!.verified);
      onActionComplete();
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const relabelMutation = useMutation({
    mutationFn: ({ label, category }: { label: string | null; category: string }) =>
      detectionsApi.bulkRelabel([detection!.detection_id], label, category),
    onSuccess: (_, { label, category }) => {
      onRelabel?.(detection!.detection_id, label ?? category, category);
      onActionComplete();
      onNavigate?.("nextUnverified");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const markFalseMutation = useMutation({
    mutationFn: () =>
      detectionsApi.bulkRelabel([detection!.detection_id], "false detection", undefined),
    onSuccess: () => {
      onMarkFalse?.(detection!.detection_id);
      onActionComplete();
      onNavigate?.("nextUnverified");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  // Verify current detection and advance to next unverified
  const handleVerifyAndAdvance = useCallback(() => {
    if (!detection || detection.verified) {
      // Already verified — just advance
      onNavigate?.("nextUnverified");
      return;
    }
    // Patch local state immediately so navigation sees updated verified status
    onVerify?.(detection.detection_id);
    onNavigate?.("nextUnverified");
    // Fire API call in background
    detectionsApi
      .verify(detection.detection_id, true)
      .then(() => onActionComplete())
      .catch((err: Error) => toast.error(err.message));
  }, [detection, onActionComplete, onNavigate, onVerify]);

  // Keyboard navigation while modal is open
  useEffect(() => {
    if (!open || !onNavigate) return;

    function handleKeyDown(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      const key = e.key.toLowerCase();

      if (e.key === "ArrowLeft") {
        e.preventDefault();
        onNavigate!("prev");
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        onNavigate!("next");
      } else if (e.key === "Enter") {
        e.preventDefault();
        handleVerifyAndAdvance();
      } else if (key === "x") {
        e.preventDefault();
        if (!markFalseMutation.isPending) markFalseMutation.mutate();
      } else if (key === "r") {
        e.preventDefault();
        setForceOpenPicker(true);
      } else if (key === "a") {
        e.preventDefault();
        if (
          detection?.neighbor_top_label &&
          detection.neighbor_top_label !== detection.label &&
          !relabelMutation.isPending
        ) {
          relabelMutation.mutate({
            label: detection.neighbor_top_label,
            category: detection.category,
          });
        }
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    open,
    onNavigate,
    handleVerifyAndAdvance,
    markFalseMutation,
    relabelMutation,
    detection,
    onOpenChange,
  ]);

  // Calculate modal dimensions to tightly fit the image + sidebar panel.
  // Keep previous size while loading to avoid a resize flash between images.
  const PANEL_W = 320;
  const IMAGE_PAD = 16;
  const lastModalStyle = useRef<{ width: number; height: number } | null>(null);
  const modalStyle = useMemo(() => {
    const maxW = viewport.width * 0.95;
    const maxH = viewport.height * 0.95;

    if (!fileData?.width_px || !fileData?.height_px) {
      return lastModalStyle.current ?? { width: maxW, height: maxH };
    }

    const maxImgW = maxW - PANEL_W;
    const maxImgH = maxH - IMAGE_PAD;

    const scale = Math.min(
      maxImgW / fileData.width_px,
      maxImgH / fileData.height_px,
      1
    );
    const imgDisplayW = fileData.width_px * scale;
    const imgDisplayH = fileData.height_px * scale;

    const style = {
      width: Math.round(imgDisplayW + PANEL_W),
      height: Math.round(imgDisplayH + IMAGE_PAD),
    };
    lastModalStyle.current = style;
    return style;
  }, [fileData?.width_px, fileData?.height_px, viewport]);

  if (!detection) return null;

  // Find the matching detection in file data for bbox
  const fullDetection = fileData?.detections.find(
    (d) => d.id === detection.detection_id
  );

  const imgW = fileData?.width_px || 1;
  const imgH = fileData?.height_px || 1;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="flex flex-col p-3 pr-0 gap-0 overflow-hidden [&>button.absolute]:hidden"
        style={{
          width: modalStyle.width,
          height: modalStyle.height,
          maxWidth: "95vw",
          maxHeight: "95vh",
        }}
        onOpenAutoFocus={(e) => e.preventDefault()}
        aria-describedby={undefined}
      >
        <DialogTitle className="sr-only">
          {getDetectionDisplayName(detection)} detection detail
        </DialogTitle>

        <div className="flex flex-1 min-h-0 overflow-hidden">
          {/* Left panel — source image with bbox. Scroll to zoom, drag
              to pan when zoomed, double-click to reset. */}
          <div
            ref={attachImagePanel}
            className="flex-1 flex select-none items-center justify-center overflow-hidden bg-black/95 min-h-0 p-2 rounded-lg"
            style={{
              cursor: zoom > 1 ? (isPanning ? "grabbing" : "grab") : "default",
            }}
            onPointerDown={(e) => {
              if (zoom <= 1) return;
              // Stop the browser's native image drag-and-drop, which
              // otherwise swallows the pointerup and leaves the pan
              // stuck "held" after release.
              e.preventDefault();
              panStartRef.current = {
                x: e.clientX,
                y: e.clientY,
                panX: pan.x,
                panY: pan.y,
              };
              setIsPanning(true);
            }}
            onDoubleClick={() => {
              setZoom(1);
              setPan({ x: 0, y: 0 });
            }}
          >
            <div
              className="relative inline-flex"
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: "center center",
              }}
            >
              <img
                src={`${API_BASE_URL}/api/files/${detection.file_id}/image`}
                alt="Source image"
                draggable={false}
                className="max-w-full max-h-full object-contain"
              />
              {fullDetection && fullDetection.bbox_x !== null && (() => {
                // Event-level observations carry no bbox and never reach
                // this modal (the Observations grid excludes them), but
                // the type system can't see that — the guard above keeps
                // the canvas math typesafe.
                const s = Math.max(imgW, imgH) / 1000;
                const pill = computePillLayout(fullDetection);
                const bx = fullDetection.bbox_x * imgW;
                const by = (fullDetection.bbox_y ?? 0) * imgH;
                const bw = (fullDetection.bbox_width ?? 0) * imgW;
                const bh = (fullDetection.bbox_height ?? 0) * imgH;
                const pillH = pill.pillHeight * s;
                const pillY = by - pillH > 0 ? by - pillH : by;
                const spotlightPath =
                  `M0,0H${imgW}V${imgH}H0Z` +
                  svgRoundedRectPath(bx, by, bw, bh, BBOX_CORNER_RADIUS * s);

                return (
                  <svg
                    className="absolute inset-0 w-full h-full pointer-events-none"
                    viewBox={`0 0 ${imgW} ${imgH}`}
                    preserveAspectRatio="xMidYMid meet"
                  >
                    {/* Spotlight dim overlay */}
                    <path fillRule="evenodd" d={spotlightPath} fill={DIM_FILL} />

                    {/* Rounded bbox */}
                    <rect
                      x={bx} y={by} width={bw} height={bh}
                      rx={BBOX_CORNER_RADIUS * s}
                      fill="none"
                      stroke={pill.color}
                      strokeWidth={BBOX_STROKE_WIDTH * s}
                      opacity={BBOX_OPACITY}
                    />

                    {/* Pill label */}
                    <g transform={`translate(${bx}, ${pillY}) scale(${s})`}>
                      <rect
                        x={0} y={0}
                        width={pill.pillWidth} height={pill.pillHeight}
                        rx={BBOX_CORNER_RADIUS}
                        fill={PILL_BG}
                      />
                      <circle
                        cx={PILL_PAD_X + DOT_R}
                        cy={pill.pillHeight / 2}
                        r={DOT_R}
                        fill={pill.color}
                      />
                      {pill.hasLabel ? (
                        <>
                          <text
                            x={TEXT_START_X} y={PILL_PAD_Y}
                            fill="rgba(255,255,255,0.7)"
                            fontSize={FONT_SM}
                            fontFamily="Arial, sans-serif"
                            dominantBaseline="hanging"
                          >
                            {pill.categoryText}
                          </text>
                          <text
                            x={TEXT_START_X} y={PILL_PAD_Y + FONT_SM + LINE_GAP}
                            fill="white"
                            fontSize={FONT_LG}
                            fontWeight="bold"
                            fontFamily="Arial, sans-serif"
                            dominantBaseline="hanging"
                          >
                            {pill.labelText}
                          </text>
                        </>
                      ) : (
                        <text
                          x={TEXT_START_X} y={PILL_PAD_Y}
                          fill="white"
                          fontSize={FONT_LG}
                          fontWeight="bold"
                          fontFamily="Arial, sans-serif"
                          dominantBaseline="hanging"
                        >
                          {pill.categoryText}
                        </text>
                      )}
                    </g>
                  </svg>
                );
              })()}
            </div>
          </div>

          {/* Right panel — sidebar */}
          <div className="w-80 bg-white flex flex-col shrink-0">
            {/* Navigation header — pinned */}
            <div className="flex items-center justify-between px-3 py-1.5 shrink-0">
              <div className="flex items-center gap-0.5">
                {position && (
                  <span className="text-xs text-muted-foreground mr-1">
                    {position}
                  </span>
                )}
                {onNavigate && (
                  <>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => onNavigate("prev")}
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => onNavigate("next")}
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => onNavigate("nextUnverified")}
                      title="Next unverified"
                    >
                      <ChevronsRight className="h-4 w-4" />
                    </Button>
                  </>
                )}
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={() => onOpenChange(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Scrollable area for all cards */}
            <div className="flex-1 min-h-0 overflow-y-auto">

            {/* Card: Image metadata */}
              {fileData && (
                <div className="mx-3 mt-3 rounded-lg border bg-muted/40">
                  <h3 className="px-3 pt-3 pb-2 text-sm font-semibold">
                    {fileData.file_type === "video" ? "Video" : "Image"}
                  </h3>
                  <div className="px-3 pb-3 space-y-0.5 text-xs text-muted-foreground">
                    <div className="truncate">
                      {basename(fileData.file_path)}
                      {fileData.file_type === "video" && detection.frame_number != null && (
                        <span> · frame {detection.frame_number}</span>
                      )}
                    </div>
                    <div>
                      {formatCameraDate(detection.captured_at_local, { day: "numeric", month: "short", year: "numeric" }, "en-GB")}{" "}
                      {formatCameraTime(detection.captured_at_local, { hour: "2-digit", minute: "2-digit" }, "en-GB")}
                      {detection.site_name && ` · ${detection.site_name}`}
                    </div>
                    {detection.similarity != null && (
                      <div>Similarity: {(detection.similarity * 100).toFixed(1)}%</div>
                    )}
                    {detection.distance_to_centroid != null &&
                      detection.distance_to_centroid !== Infinity && (
                        <div>Distance: {detection.distance_to_centroid.toFixed(3)}</div>
                      )}
                  </div>
                </div>
              )}

              {/* Label Agreement */}
              {!detection.verified && detection.neighbor_agreement != null && (() => {
                const count = Math.round(detection.neighbor_agreement * 10);
                const pct = detection.neighbor_agreement * 100;
                const hasSuggestion =
                  detection.neighbor_top_label &&
                  detection.neighbor_top_label !== detection.label;
                return (
                  <div className="mx-3 mt-3 rounded-lg border bg-muted/40">
                    <h3 className="px-3 pt-3 pb-2 text-sm font-semibold">Label agreement</h3>
                    <div className="px-3 pb-3 space-y-2">
                      <div className="relative h-3 w-full overflow-hidden rounded-full flex">
                        <div style={{ width: `${pct}%`, backgroundColor: "#0f6064" }} className="h-full transition-all duration-500 ease-out" />
                        <div style={{ width: `${100 - pct}%`, backgroundColor: "#882000" }} className="h-full transition-all duration-500 ease-out" />
                      </div>
                      <p className="text-xs text-muted-foreground text-center">
                        {count}/10 neighbors agree
                      </p>
                      {hasSuggestion && (
                        <p className="text-sm text-amber-600 dark:text-amber-400">
                          Neighbors suggest:{" "}
                          <span className="font-semibold capitalize">
                            {detection.neighbor_top_display_name ||
                              detection.neighbor_top_label}
                          </span>
                        </p>
                      )}
                      {/* Neighbor thumbnails */}
                      {neighborsData?.results && neighborsData.results.length > 0 && (
                        <div className="grid grid-cols-5 gap-1.5 pt-1">
                          {neighborsData.results
                            .filter((n) => n.detection_id !== detection.detection_id)
                            .slice(0, 10)
                            .map((n) => {
                            const agrees = n.label === detection.label;
                            return (
                              <div key={n.detection_id} className="space-y-0.5">
                                <img
                                  src={`${API_BASE_URL}${n.crop_url}`}
                                  alt={getDetectionDisplayName(n)}
                                  className={cn(
                                    "w-full aspect-square object-cover rounded border-2",
                                    agrees ? "border-[#0f6064]" : "border-[#882000]"
                                  )}
                                />
                                <p className="text-[9px] text-muted-foreground truncate text-center capitalize">
                                  {getDetectionDisplayName(n)}
                                </p>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })()}
            </div> {/* end scrollable area */}

            {/* Bottom pinned: action buttons.
                Mirrors the floating BulkActionBar's vocabulary, icons,
                and shortcuts so the muscle memory carries over from the
                grid into the modal. Each action operates on this single
                detection. */}
            <div className="px-3 py-3 space-y-2 shrink-0">
              {detection.neighbor_top_label &&
                detection.neighbor_top_label !== detection.label && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full justify-center"
                    disabled={relabelMutation.isPending}
                    onClick={() =>
                      relabelMutation.mutate({
                        label: detection.neighbor_top_label!,
                        category: detection.category,
                      })
                    }
                  >
                    <Tag className="h-4 w-4 mr-1" />
                    Accept &ldquo;
                    {detection.neighbor_top_display_name ||
                      detection.neighbor_top_label}
                    &rdquo;
                    <kbd className="ml-1.5 text-[10px] font-sans text-muted-foreground/60 border border-border/60 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(0,0,0,0.08)] leading-none">
                      A
                    </kbd>
                  </Button>
                )}

              <Button
                variant="outline"
                size="sm"
                className="w-full justify-center"
                onClick={() => setForceOpenPicker(true)}
              >
                <Tag className="h-4 w-4 mr-1" />
                Relabel
                <kbd className="ml-1.5 text-[10px] font-sans text-muted-foreground/60 border border-border/60 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(0,0,0,0.08)] leading-none">
                  R
                </kbd>
              </Button>
              {/* Hidden LabelPicker. The Relabel button above is the
                  visible trigger; this picker's own trigger is taken
                  out of the flow with absolute+zero-size so it doesn't
                  steal a `space-y-2` slot. The popup dialog is portaled
                  by Radix, so the wrapper position doesn't constrain it. */}
              <div
                aria-hidden="true"
                className="absolute h-0 w-0 overflow-hidden pointer-events-none"
              >
                <LabelPicker
                  value={detection.label}
                  displayName={detection.display_name}
                  onSelect={(option) => {
                    relabelMutation.mutate({
                      label: option.label,
                      category: option.category,
                    });
                  }}
                  options={labelOptions}
                  isLoading={labelOptionsLoading}
                  forceOpen={forceOpenPicker}
                  onOpenChange={(open) => {
                    if (!open) setForceOpenPicker(false);
                  }}
                  projectId={projectId}
                />
              </div>

              <Button
                variant="outline"
                size="sm"
                className="w-full justify-center"
                onClick={() => markFalseMutation.mutate()}
                disabled={markFalseMutation.isPending}
              >
                <Ban className="h-4 w-4 mr-1" />
                Mark false
                <kbd className="ml-1.5 text-[10px] font-sans text-muted-foreground/60 border border-border/60 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(0,0,0,0.08)] leading-none">
                  X
                </kbd>
              </Button>

              <Button
                className="w-full justify-center"
                size="sm"
                onClick={
                  detection.verified
                    ? () => verifyMutation.mutate()
                    : handleVerifyAndAdvance
                }
                disabled={verifyMutation.isPending}
                variant={detection.verified ? "outline" : "default"}
              >
                <Check className="h-4 w-4 mr-1" />
                {detection.verified ? "Unverify" : "Verify"} &ldquo;
                {getDetectionDisplayName(detection)}
                &rdquo;
                {!detection.verified && (
                  <kbd className="ml-1.5 text-[10px] font-sans text-primary-foreground/60 border border-primary-foreground/30 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(255,255,255,0.1)] leading-none">
                    ⏎
                  </kbd>
                )}
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
