/**
 * DetectionDetailModal - centered two-panel dialog showing full detection details.
 *
 * Left panel: source image with bbox overlay (dark background).
 * Right panel: crop, metadata, label agreement, and action buttons.
 * Supports prev/next navigation and verify-and-advance (Enter) for rapid review.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Search, Tag, ChevronLeft, ChevronRight, X } from "lucide-react";
import { toast } from "sonner";
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Progress } from "../ui/progress";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { API_BASE_URL } from "../../lib/api-client";
import { getCategoryColor } from "../../lib/detection-utils";
import type { DetectionSummary } from "../../api/types";

interface DetectionDetailModalProps {
  detection: DetectionSummary | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onFindSimilar: (detectionId: string) => void;
  onActionComplete: () => void;
  onRelabel?: (detectionId: string, species: string, category: string) => void;
  /** Optimistic verify callback so parent can patch local state before navigating. */
  onVerify?: (detectionId: string) => void;
  /** Navigate to adjacent detection. Return false if at boundary. */
  onNavigate?: (direction: "prev" | "next" | "nextUnverified") => boolean;
  /** Current position, e.g. "3 / 48" */
  position?: string;
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

  const verifyMutation = useMutation({
    mutationFn: () =>
      detectionsApi.verify(detection!.detection_id, !detection!.verified),
    onSuccess: () => {
      toast.success(
        detection!.verified ? "Detection unverified" : "Detection verified"
      );
      onActionComplete();
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const relabelMutation = useMutation({
    mutationFn: ({ species, category }: { species: string; category: string }) =>
      detectionsApi.bulkRelabel([detection!.detection_id], species, category),
    onSuccess: (_, { species, category }) => {
      toast.success(`Relabeled to "${species}"`);
      onRelabel?.(detection!.detection_id, species, category);
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
          {detection.species || detection.category} detection detail
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
              {fullDetection && (
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  viewBox={`0 0 ${imgW} ${imgH}`}
                  preserveAspectRatio="none"
                >
                  <rect
                    x={fullDetection.bbox_x * imgW}
                    y={fullDetection.bbox_y * imgH}
                    width={fullDetection.bbox_width * imgW}
                    height={fullDetection.bbox_height * imgH}
                    fill="none"
                    stroke={getCategoryColor(fullDetection.category)}
                    strokeWidth={Math.max(imgW, imgH) * 0.003}
                    rx={2}
                  />
                </svg>
              )}
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

            {/* Card 1: Crop + species info */}
            <div className="mx-3 mt-2 rounded-lg border bg-muted/40">
              <h3 className="px-3 pt-3 pb-2 text-sm font-semibold">Detection</h3>
              <div className="px-3 pb-3 space-y-2">
                <img
                  src={`${API_BASE_URL}${detection.crop_url}`}
                  alt="Crop"
                  className="w-full aspect-square object-cover rounded-lg border"
                />
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm capitalize">
                      {detection.category} ({(detection.confidence * 100).toFixed(0)}%)
                    </span>
                    {detection.verified && (
                      <Badge variant="default" className="text-[10px]">
                        Verified
                      </Badge>
                    )}
                  </div>
                  {detection.species && (
                    <div className="font-medium capitalize">
                      {detection.species} ({detection.species_confidence != null ? `${(detection.species_confidence * 100).toFixed(0)}%` : "—"})
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Card 2: Image metadata (scrollable) */}
            <div className="flex-1 min-h-0 overflow-y-auto">
              {fileData && (
                <div className="mx-3 mt-3 rounded-lg border bg-muted/40">
                  <h3 className="px-3 pt-3 pb-2 text-sm font-semibold">Image</h3>
                  <div className="px-3 pb-3 space-y-0.5 text-xs text-muted-foreground">
                    <div className="truncate">
                      {fileData.file_path.split("/").pop()}
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
              {detection.neighbor_agreement != null && (() => {
                const count = Math.round(detection.neighbor_agreement * 10);
                const pct = detection.neighbor_agreement * 100;
                const hasSuggestion =
                  detection.neighbor_top_label &&
                  detection.neighbor_top_label !== detection.species;
                // Interpolate between bad (#882000) and good (#0f6064)
                const t = detection.neighbor_agreement;
                const r = Math.round(0x88 + (0x0f - 0x88) * t);
                const g = Math.round(0x20 + (0x60 - 0x20) * t);
                const b = Math.round(0x00 + (0x64 - 0x00) * t);
                const barColor = `rgb(${r}, ${g}, ${b})`;

                return (
                  <div className="mx-3 mt-3 rounded-lg border bg-muted/40">
                    <h3 className="px-3 pt-3 pb-2 text-sm font-semibold">Label Agreement</h3>
                    <div className="px-3 pb-3 space-y-2">
                      <Progress value={pct} className="h-3" barColor={barColor} />
                      <p className="text-sm text-muted-foreground">
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
                              species: detection.neighbor_top_label!,
                              category: detection.category,
                            })
                          }
                        >
                          <Tag className="h-4 w-4 mr-2" />
                          Accept &ldquo;{detection.neighbor_top_label}&rdquo;
                        </Button>
                      )}
                    </div>
                  </div>
                );
              })()}
            </div>

            {/* Bottom pinned: action buttons */}
            <div className="px-3 py-3 space-y-2 shrink-0">
              <div className="flex gap-2">
                <Button
                  className="flex-1"
                  size="sm"
                  onClick={() => verifyMutation.mutate()}
                  disabled={verifyMutation.isPending}
                  variant={detection.verified ? "outline" : "default"}
                >
                  <Check className="h-4 w-4 mr-2" />
                  {detection.verified ? "Unverify" : "Verify"}
                </Button>
                {onNavigate && !detection.verified && (
                  <Button
                    className="flex-1"
                    size="sm"
                    onClick={handleVerifyAndAdvance}
                  >
                    <Check className="h-4 w-4 mr-2" />
                    Verify & Next
                  </Button>
                )}
              </div>
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
