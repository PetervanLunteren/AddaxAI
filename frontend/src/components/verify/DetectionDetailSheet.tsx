/**
 * DetectionDetailSheet - right-side slide-out panel showing full detection details.
 *
 * Shows the source image with bbox overlay, detection metadata, and action buttons.
 * Supports prev/next navigation and verify-and-advance (Enter) for rapid review.
 */

import { useCallback, useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Search, Tag, ChevronLeft, ChevronRight } from "lucide-react";
import { toast } from "sonner";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "../ui/sheet";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Progress } from "../ui/progress";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { API_BASE_URL } from "../../lib/api-client";
import { getDetectionColor } from "../../lib/detection-utils";
import type { DetectionSummary } from "../../api/types";

interface DetectionDetailSheetProps {
  detection: DetectionSummary | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onFindSimilar: (detectionId: string) => void;
  onActionComplete: () => void;
  onRelabel?: (detectionId: string, label: string, category: string) => void;
  /** Optimistic verify callback so parent can patch local state before navigating. */
  onVerify?: (detectionId: string) => void;
  /** Navigate to adjacent detection. Return false if at boundary. */
  onNavigate?: (direction: "prev" | "next" | "nextUnverified") => boolean;
  /** Current position, e.g. "3 / 48" */
  position?: string;
}

export function DetectionDetailSheet({
  detection,
  open,
  onOpenChange,
  onFindSimilar,
  onActionComplete,
  onRelabel,
  onVerify,
  onNavigate,
  position,
}: DetectionDetailSheetProps) {
  const queryClient = useQueryClient();

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

  // Keyboard navigation while sheet is open
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

  if (!detection) return null;

  // Find the matching detection in file data for bbox
  const fullDetection = fileData?.detections.find(
    (d) => d.id === detection.detection_id
  );

  const imgW = fileData?.width_px || 1;
  const imgH = fileData?.height_px || 1;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-lg overflow-y-auto">
        <SheetHeader>
          <div className="flex items-center justify-between">
            <SheetTitle className="capitalize">
              {detection.label || detection.category}
            </SheetTitle>
            {onNavigate && (
              <div className="flex items-center gap-1">
                {position && (
                  <span className="text-xs text-muted-foreground mr-1">
                    {position}
                  </span>
                )}
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
              </div>
            )}
          </div>
        </SheetHeader>

        <div className="space-y-4 mt-4">
          {/* Source image with bbox overlay */}
          <div className="relative bg-muted rounded-lg overflow-hidden">
            <img
              src={`${API_BASE_URL}/api/files/${detection.file_id}/image`}
              alt="Source image"
              className="w-full h-auto"
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
                  stroke={getDetectionColor(fullDetection)}
                  strokeWidth={Math.max(imgW, imgH) * 0.003}
                  rx={2}
                />
              </svg>
            )}
          </div>

          {/* Crop thumbnail */}
          <div className="flex items-center gap-3">
            <img
              src={`${API_BASE_URL}${detection.crop_url}`}
              alt="Crop"
              className="h-20 w-20 rounded-lg object-cover border"
            />
            <div className="space-y-1">
              <div>
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
                {detection.label && (
                  <div className="font-medium capitalize">
                    {detection.label} ({detection.label_confidence != null ? `${(detection.label_confidence * 100).toFixed(0)}%` : "—"})
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Metadata */}
          <div className="grid grid-cols-2 gap-2 text-sm">
            {detection.site_name && (
              <div>
                <span className="text-muted-foreground">Site</span>
                <p>{detection.site_name}</p>
              </div>
            )}
            {detection.timestamp && (
              <div>
                <span className="text-muted-foreground">Date & time</span>
                <p>{new Date(detection.timestamp).toLocaleString()}</p>
              </div>
            )}
            {detection.similarity != null && (
              <div>
                <span className="text-muted-foreground">Similarity</span>
                <p>{(detection.similarity * 100).toFixed(1)}%</p>
              </div>
            )}
            {detection.distance_to_centroid != null &&
              detection.distance_to_centroid !== Infinity && (
                <div>
                  <span className="text-muted-foreground">Distance</span>
                  <p>{detection.distance_to_centroid.toFixed(3)}</p>
                </div>
              )}
          </div>

          {/* Label Agreement */}
          {!detection.verified && detection.neighbor_agreement != null && (() => {
            const count = Math.round(detection.neighbor_agreement * 10);
            const pct = detection.neighbor_agreement * 100;
            const hasSuggestion =
              detection.neighbor_top_label &&
              detection.neighbor_top_label !== detection.label;
            const barColorClass =
              detection.neighbor_agreement >= 0.7
                ? "[&>div]:bg-green-500"
                : detection.neighbor_agreement >= 0.3
                  ? "[&>div]:bg-amber-500"
                  : "[&>div]:bg-red-500";

            return (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Label Agreement</h4>
                <Progress value={pct} className={`h-3 ${barColorClass}`} />
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
                {detection.classification_method && (
                  <p className="text-xs text-muted-foreground">
                    Labeled by {detection.classification_method}
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
              </div>
            );
          })()}

          {/* Actions */}
          <div className="flex flex-col gap-2 pt-2">
            <div className="flex gap-2">
              <Button
                className="flex-1"
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
                  onClick={handleVerifyAndAdvance}
                >
                  <Check className="h-4 w-4 mr-2" />
                  Verify & Next
                </Button>
              )}
            </div>
            <Button
              variant="outline"
              onClick={() => {
                onFindSimilar(detection.detection_id);
                onOpenChange(false);
              }}
            >
              <Search className="h-4 w-4 mr-2" />
              Find similar
            </Button>
            {onNavigate && (
              <p className="text-[11px] text-center text-muted-foreground">
                ← → navigate &middot; Enter verify &amp; next
              </p>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
