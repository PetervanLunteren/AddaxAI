/**
 * DetectionDetailModal - centered two-panel dialog showing full detection details.
 *
 * Left panel: source image with bbox overlay (dark background).
 * Right panel: crop, metadata, label agreement, and action buttons.
 * Supports prev/next navigation and verify-and-advance (Enter) for rapid review.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Search, Tag, ChevronLeft, ChevronRight, ChevronsRight, X } from "lucide-react";
import { toast } from "sonner";
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { similarityApi } from "../../api/similarity";
import { API_BASE_URL } from "../../lib/api-client";
import { cn } from "../../lib/utils";
import {
  computePillLayout,
  svgRoundedRectPath,
  PILL_PAD_X, PILL_PAD_Y, DOT_R, LINE_GAP,
  FONT_SM, FONT_LG, TEXT_START_X,
  BBOX_STROKE_WIDTH, BBOX_OPACITY, BBOX_CORNER_RADIUS,
  DIM_FILL, PILL_BG,
} from "../../lib/detection-overlay";
import type { DetectionSummary } from "../../api/types";

interface DetectionDetailModalProps {
  detection: DetectionSummary | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onFindSimilar: (detectionId: string) => void;
  onActionComplete: () => void;
  onRelabel?: (detectionId: string, label: string, category: string) => void;
  /** Optimistic verify callback so parent can patch local state before navigating. */
  onVerify?: (detectionId: string, verified?: boolean) => void;
  /** Navigate to adjacent detection. Return false if at boundary. */
  onNavigate?: (direction: "prev" | "next" | "nextUnverified") => boolean;
  /** Current position, e.g. "3 / 48" */
  position?: string;
  /** Project ID for fetching nearest neighbors. */
  projectId?: string;
}

export function DetectionDetailModal({
  detection,
  open,
  onOpenChange,
  onFindSimilar,
  onActionComplete,
  onRelabel,
  onVerify,
  onNavigate,
  position,
  projectId,
}: DetectionDetailModalProps) {
  const queryClient = useQueryClient();
  const [viewport, setViewport] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  // Track viewport size for responsive modal sizing
  useEffect(() => {
    const handleResize = () => {
      setViewport({ width: window.innerWidth, height: window.innerHeight });
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

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
      similarityApi.search(projectId!, {
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
    mutationFn: ({ label, category }: { label: string; category: string }) =>
      detectionsApi.bulkRelabel([detection!.detection_id], label, category),
    onSuccess: (_, { label, category }) => {
      onRelabel?.(detection!.detection_id, label, category);
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

      if (e.key === "ArrowLeft") {
        e.preventDefault();
        onNavigate!("prev");
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        onNavigate!("next");
      } else if (e.key === "Enter") {
        e.preventDefault();
        handleVerifyAndAdvance();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onNavigate, handleVerifyAndAdvance]);

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
          {detection.label || detection.category} detection detail
        </DialogTitle>

        <div className="flex flex-1 min-h-0 overflow-hidden">
          {/* Left panel — source image with bbox */}
          <div className="flex-1 flex items-center justify-center bg-black/95 min-h-0 p-2 rounded-lg">
            <div className="relative inline-flex">
              <img
                src={`${API_BASE_URL}/api/files/${detection.file_id}/image`}
                alt="Source image"
                className="max-w-full max-h-full object-contain"
              />
              {fullDetection && (() => {
                const s = Math.max(imgW, imgH) / 1000;
                const pill = computePillLayout(fullDetection);
                const bx = fullDetection.bbox_x * imgW;
                const by = fullDetection.bbox_y * imgH;
                const bw = fullDetection.bbox_width * imgW;
                const bh = fullDetection.bbox_height * imgH;
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

            {/* Card 1: Crop + label info */}
            <div className="mx-3 mt-2 rounded-lg border bg-muted/40">
              <h3 className="px-3 pt-3 pb-2 text-sm font-semibold">Detection</h3>
              <div className="px-3 pb-3 space-y-2">
                <img
                  src={`${API_BASE_URL}${detection.crop_url}`}
                  alt="Crop"
                  className="w-full aspect-square max-h-[150px] max-w-[150px] mx-auto object-cover rounded-lg border"
                />
                <div className="flex items-center justify-center gap-2">
                  {fullDetection && (() => {
                    const p = computePillLayout(fullDetection);
                    const ps = 1.2;
                    return (
                      <svg width={p.pillWidth * ps} height={p.pillHeight * ps} viewBox={`0 0 ${p.pillWidth} ${p.pillHeight}`}>
                        <rect
                          x={0} y={0}
                          width={p.pillWidth} height={p.pillHeight}
                          rx={BBOX_CORNER_RADIUS}
                          fill={PILL_BG}
                        />
                        <circle
                          cx={PILL_PAD_X + DOT_R}
                          cy={p.pillHeight / 2}
                          r={DOT_R}
                          fill={p.color}
                        />
                        {p.hasLabel ? (
                          <>
                            <text
                              x={TEXT_START_X} y={PILL_PAD_Y}
                              fill="rgba(255,255,255,0.7)"
                              fontSize={FONT_SM}
                              fontFamily="Arial, sans-serif"
                              dominantBaseline="hanging"
                            >
                              {p.categoryText}
                            </text>
                            <text
                              x={TEXT_START_X} y={PILL_PAD_Y + FONT_SM + LINE_GAP}
                              fill="white"
                              fontSize={FONT_LG}
                              fontWeight="bold"
                              fontFamily="Arial, sans-serif"
                              dominantBaseline="hanging"
                            >
                              {p.labelText}
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
                            {p.categoryText}
                          </text>
                        )}
                      </svg>
                    );
                  })()}
                  {detection.verified && (
                    <Badge variant="default" className="text-[10px]">
                      Verified
                    </Badge>
                  )}
                </div>
              </div>
            </div>

            {/* Card 2: Image metadata */}
              {fileData && (
                <div className="mx-3 mt-3 rounded-lg border bg-muted/40">
                  <h3 className="px-3 pt-3 pb-2 text-sm font-semibold">
                    {fileData.file_type === "frame" ? "Video" : "Image"}
                  </h3>
                  <div className="px-3 pb-3 space-y-0.5 text-xs text-muted-foreground">
                    <div className="truncate">
                      {fileData.file_type === "frame"
                        ? fileData.file_path.split("/").slice(-2, -1)[0]
                        : fileData.file_path.split("/").pop()}
                      {fileData.file_type === "frame" && fileData.source_frame_number != null && (
                        <span> · frame {fileData.source_frame_number}</span>
                      )}
                    </div>
                    <div>
                      {detection.timestamp &&
                        new Date(detection.timestamp).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}{" "}
                      {detection.timestamp &&
                        new Date(detection.timestamp).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}
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
                    <h3 className="px-3 pt-3 pb-2 text-sm font-semibold">Label Agreement</h3>
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
                            {detection.neighbor_top_label}
                          </span>
                        </p>
                      )}
                      {hasSuggestion && (
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={relabelMutation.isPending}
                          onClick={() =>
                            relabelMutation.mutate({
                              label: detection.neighbor_top_label!,
                              category: detection.category,
                            })
                          }
                        >
                          <Tag className="h-4 w-4 mr-2" />
                          Accept &ldquo;{detection.neighbor_top_label}&rdquo;
                        </Button>
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
                                  alt={n.label || n.category}
                                  className={cn(
                                    "w-full aspect-square object-cover rounded border-2",
                                    agrees ? "border-[#0f6064]" : "border-[#882000]"
                                  )}
                                />
                                <p className="text-[9px] text-muted-foreground truncate text-center capitalize">
                                  {n.label || n.category}
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

            {/* Bottom pinned: action buttons */}
            <div className="px-3 py-3 space-y-2 shrink-0">
              <Button
                className="w-full"
                size="sm"
                onClick={detection.verified ? () => verifyMutation.mutate() : handleVerifyAndAdvance}
                disabled={verifyMutation.isPending}
                variant={detection.verified ? "outline" : "default"}
              >
                <Check className="h-4 w-4 mr-2" />
                {detection.verified ? "Unverify" : "Mark verified"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="w-full"
                onClick={() => {
                  onFindSimilar(detection.detection_id);
                  onOpenChange(false);
                }}
              >
                <Search className="h-4 w-4 mr-2" />
                Find similar
              </Button>
            </div>
            {onNavigate && (
              <div className="shrink-0 px-3 pb-2">
                <p className="text-[11px] text-center text-muted-foreground">
                  ← → navigate &middot; Enter verify &amp; next
                </p>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
