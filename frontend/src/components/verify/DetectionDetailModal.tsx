/**
 * DetectionDetailModal - centered two-panel dialog showing full detection details.
 *
 * Left panel: source image with bbox overlay (dark background).
 * Right panel: crop, metadata, label agreement, and action buttons.
 * Supports prev/next navigation and verify-and-advance (Enter) for rapid review.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Ban, Check, CircleHelp, ImageOff, Tag, Play } from "lucide-react";
import { toast } from "sonner";
import { Button } from "../ui/button";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { labelsApi } from "../../api/labels";
import { eventsApi } from "../../api/events";
import { projectsApi } from "../../api/projects";
import { API_BASE_URL } from "../../lib/api-client";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { getDetectionDisplayName } from "../../lib/detection-utils";
import { reportMissingMedia } from "../../hooks/useBrokenDeployments";
import { FrameThumbnail } from "./FrameThumbnail";
import { ContextCard } from "./ContextCard";
import {
  computePillLayout,
  placePill,
  svgRoundedRectPath,
  PILL_PAD_Y, LINE_GAP,
  FONT, TEXT_START_X,
  BBOX_STROKE_WIDTH, BBOX_OPACITY, BBOX_CORNER_RADIUS,
  DIM_FILL, PILL_BG,
} from "../../lib/detection-overlay";
import type { DetectionSummary } from "../../api/types";
import type { LabelOption } from "../../hooks/useLabelOptions";
import { LabelPicker } from "./LabelPicker";
import { DetailCard, VerifyDetailShell } from "./VerifyDetailShell";
import { FileLocation } from "./FileLocation";
import { useSpeciesColorsVersion } from "../../utils/species-colors";

interface DetectionDetailModalProps {
  detection: DetectionSummary | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onActionComplete: () => void;
  onRelabel?: (detectionId: string, label: string, category: string) => void;
  /** Optimistic mark-false callback so parent can patch local state before navigating. */
  onMarkFalse?: (detectionId: string) => void;
  /** Optimistic mark-unknown callback (real observation, keeps category). */
  onMarkUnknown?: (detectionId: string) => void;
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
  onMarkUnknown,
  onVerify,
  onNavigate,
  position,
  projectId,
  labelOptions = [],
  labelOptionsLoading = false,
}: DetectionDetailModalProps) {
  // Repaint when the project's colour map lands or changes.
  useSpeciesColorsVersion();
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
  const [imageFailed, setImageFailed] = useState(false);
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
  // `imageFailed` rides along: stepping from a missing file to a present
  // one must not keep showing the placeholder.
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setImageFailed(false);
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

  // Load full file data to get image dimensions and detection bbox.
  const { data: fileData } = useQuery({
    queryKey: ["file", detection?.file_id],
    queryFn: () => filesApi.get(detection!.file_id),
    enabled: open && !!detection?.file_id,
  });

  // Load the detection's event for the chronological-context card. Reuses
  // the same endpoint the Counts modal uses, so it shows every frame of
  // the event regardless of embeddings. Absent when event clustering
  // hasn't run (event_id is null).
  const { data: eventData } = useQuery({
    queryKey: ["event", detection?.event_id],
    queryFn: () => eventsApi.get(detection!.event_id!),
    enabled: open && !!detection?.event_id,
  });
  const eventFiles = eventData?.files ?? [];

  // Project detection threshold for the context thumbnails' box overlay.
  // Shared query key with the rest of the app, so usually a cache hit.
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: open && !!projectId,
  });
  const detectionThreshold = project?.counting_threshold ?? 0;

  // Fetch 10 nearest neighbors for the Label Agreement thumbnails
  const { data: neighborsData } = useQuery({
    queryKey: ["detection-neighbors", detection?.detection_id],
    queryFn: () =>
      labelsApi.search(projectId!, {
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

  const markUnknownMutation = useMutation({
    mutationFn: () =>
      detectionsApi.bulkRelabel([detection!.detection_id], "unknown", undefined),
    onSuccess: () => {
      onMarkUnknown?.(detection!.detection_id);
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
      } else if (key === "u") {
        e.preventDefault();
        if (!markUnknownMutation.isPending) markUnknownMutation.mutate();
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
    markUnknownMutation,
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
    <VerifyDetailShell
      open={open}
      onOpenChange={onOpenChange}
      title={`${getDetectionDisplayName(detection)} detection detail`}
      width={modalStyle.width}
      height={modalStyle.height}
      position={position}
      onNavigate={onNavigate}
      // Scroll to zoom, drag to pan when zoomed, double-click to reset.
      // The shell owns the panel's classes; the behaviour stays here.
      imagePanelProps={{
        ref: attachImagePanel,
        style: {
          cursor: zoom > 1 ? (isPanning ? "grabbing" : "grab") : "default",
        },
        onPointerDown: (e) => {
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
        },
        onDoubleClick: () => {
          setZoom(1);
          setPan({ x: 0, y: 0 });
        },
      }}
      image={
            <div
              className="relative inline-flex"
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: "center center",
              }}
            >
              {imageFailed ? (
                // The file is gone from disk (a moved / disconnected
                // deployment folder). Without this the browser drew its
                // own broken-image glyph with the alt text on black, and
                // the box overlay below drew against a zero-sized image.
                <div className="flex flex-col items-center justify-center gap-3 px-16 py-24">
                  <ImageOff className="h-10 w-10 text-neutral-400" />
                  <p className="text-sm text-neutral-400">
                    This picture can't be found on disk
                  </p>
                </div>
              ) : (
                <img
                  src={`${API_BASE_URL}/api/files/${detection.file_id}/image`}
                  alt="Source image"
                  draggable={false}
                  className="max-w-full max-h-full object-contain"
                  onError={() => {
                    setImageFailed(true);
                    reportMissingMedia(detection.deployment_id);
                  }}
                />
              )}
              {!imageFailed && fullDetection && fullDetection.bbox_x !== null && (() => {
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
                const { x: pillX, y: pillY } = placePill(
                  { x: bx, y: by, width: bw, height: bh },
                  { width: pill.pillWidth * s, height: pill.pillHeight * s },
                  { width: imgW, height: imgH },
                );
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
                    <g transform={`translate(${pillX}, ${pillY}) scale(${s})`}>
                      <rect
                        x={0} y={0}
                        width={pill.pillWidth} height={pill.pillHeight}
                        rx={BBOX_CORNER_RADIUS}
                        fill={PILL_BG}
                      />
                      <text
                        x={TEXT_START_X} y={PILL_PAD_Y}
                        fill="white"
                        fontSize={FONT}
                        fontFamily="Arial, sans-serif"
                        dominantBaseline="hanging"
                      >
                        {pill.categoryText}
                      </text>
                      {pill.hasLabel && (
                        <text
                          x={TEXT_START_X} y={PILL_PAD_Y + FONT + LINE_GAP}
                          fill="white"
                          fontSize={FONT}
                          fontFamily="Arial, sans-serif"
                          dominantBaseline="hanging"
                        >
                          {pill.labelText}
                        </text>
                      )}
                    </g>
                  </svg>
                );
              })()}
            </div>
      }
      details={
        <>

            {/* Card: Image metadata */}
              {fileData && (
                <DetailCard
                  title={fileData.file_type === "video" ? "Video" : "Image"}
                >
                  <div className="space-y-0.5 text-xs text-muted-foreground">
                    <FileLocation
                      filePath={fileData.file_path}
                      suffix={
                        fileData.file_type === "video" &&
                        detection.frame_number != null && (
                          <span> · frame {detection.frame_number}</span>
                        )
                      }
                    />
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
                </DetailCard>
              )}

              {/* Event context: the event's other frames, so a close-up
                  crop can be read in sequence. Hover-only — nothing is
                  opened, so the big image on the left always stays on the
                  crop being verified. */}
              {eventFiles.length > 1 && (
                <ContextCard
                  title="Event context"
                  caption="Other frames in this event"
                  columns={4}
                  items={eventFiles.map((file) => {
                    const isCurrent = file.id === detection.file_id;
                    return {
                      key: file.id,
                      borderClassName: isCurrent
                        ? "border-primary"
                        : "border-transparent",
                      tile: (
                        <>
                          <FrameThumbnail
                            fileId={file.id}
                            file={file}
                            detectionThreshold={detectionThreshold}
                          />
                          {file.file_type === "video" && (
                            <span className="pointer-events-none absolute bottom-0.5 left-0.5 flex items-center justify-center rounded-full bg-black/60 p-0.5">
                              <Play className="h-2.5 w-2.5 fill-white text-white" />
                            </span>
                          )}
                        </>
                      ),
                      preview: (
                        <>
                          <div className="relative aspect-[4/3] w-full overflow-hidden rounded">
                            <FrameThumbnail
                              fileId={file.id}
                              file={file}
                              detectionThreshold={detectionThreshold}
                            />
                          </div>
                          <p className="mt-1.5 px-0.5 text-[11px] text-muted-foreground">
                            {formatCameraTime(
                              file.captured_at_local,
                              { hour: "2-digit", minute: "2-digit", second: "2-digit" },
                              "en-GB",
                            )}
                            {isCurrent && " · the crop you're verifying"}
                          </p>
                        </>
                      ),
                    };
                  })}
                />
              )}

              {/* Similarity context: the crop's look-alike neighbours,
                  mirroring the Similarity sort. Meter shows how many
                  neighbours share the label; thumbnails enlarge on hover
                  like the Event context card. */}
              {!detection.verified && detection.neighbor_agreement != null && (() => {
                const count = Math.round(detection.neighbor_agreement * 10);
                const pct = detection.neighbor_agreement * 100;
                const hasSuggestion =
                  detection.neighbor_top_label &&
                  detection.neighbor_top_label !== detection.label;
                const header = (
                  <>
                    {/* Calm confidence meter: a single teal fill over a
                        muted track. Disagreement is just the unfilled
                        remainder, not a red "wrong" signal. */}
                    <div className="relative h-3 w-full overflow-hidden rounded-full bg-muted-foreground/15">
                      <div style={{ width: `${pct}%`, backgroundColor: "#0f6064" }} className="h-full transition-all duration-500 ease-out" />
                    </div>
                    <p className="text-xs text-muted-foreground text-center">
                      {count} of 10 similar crops share this label
                    </p>
                    {hasSuggestion && (
                      <p className="text-sm text-muted-foreground">
                        Most look like:{" "}
                        <span className="font-semibold capitalize text-foreground">
                          {detection.neighbor_top_scientific_name ||
                            detection.neighbor_top_label}
                        </span>
                      </p>
                    )}
                  </>
                );
                const items = (neighborsData?.results ?? [])
                  .filter((n) => n.detection_id !== detection.detection_id)
                  .slice(0, 10)
                  .map((n) => {
                    const agrees = n.label === detection.label;
                    const crop = (
                      <img
                        src={`${API_BASE_URL}${n.crop_url}`}
                        alt={getDetectionDisplayName(n)}
                        className="h-full w-full object-cover"
                      />
                    );
                    return {
                      key: n.detection_id,
                      borderClassName: agrees
                        ? "border-[#0f6064]"
                        : "border-muted-foreground/30",
                      tile: crop,
                      preview: (
                        <>
                          <div className="relative aspect-[4/3] w-full overflow-hidden rounded">
                            {crop}
                          </div>
                          <p className="mt-1.5 px-0.5 text-[11px] capitalize text-muted-foreground">
                            {getDetectionDisplayName(n)}
                          </p>
                        </>
                      ),
                    };
                  });
                return (
                  <ContextCard
                    title="Similarity context"
                    caption="Similar-looking crops"
                    columns={5}
                    header={header}
                    items={items}
                  />
                );
              })()}
        </>
      }
      // Mirrors the floating BulkActionBar's vocabulary, icons and
      // shortcuts so the muscle memory carries over from the grid into
      // the modal. Each action operates on this single detection.
      actions={
        <>
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
                    {detection.neighbor_top_scientific_name ||
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
                  displayName={detection.scientific_name}
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
                variant="outline"
                size="sm"
                className="w-full justify-center"
                onClick={() => markUnknownMutation.mutate()}
                disabled={markUnknownMutation.isPending}
                title="Mark as an unidentifiable animal and verify"
              >
                <CircleHelp className="h-4 w-4 mr-1" />
                Unknown
                <kbd className="ml-1.5 text-[10px] font-sans text-muted-foreground/60 border border-border/60 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(0,0,0,0.08)] leading-none">
                  U
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
        </>
      }
    />
  );
}
