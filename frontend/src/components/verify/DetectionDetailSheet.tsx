/**
 * DetectionDetailSheet - right-side slide-out panel showing full detection details.
 *
 * Shows the source image with bbox overlay, detection metadata, and action buttons.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Search, Tag, ExternalLink } from "lucide-react";
import { toast } from "sonner";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "../ui/sheet";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { API_BASE_URL } from "../../lib/api-client";
import { getCategoryColor } from "../../lib/detection-utils";
import type { DetectionSummary } from "../../api/types";

interface DetectionDetailSheetProps {
  detection: DetectionSummary | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onFindSimilar: (detectionId: string) => void;
  onActionComplete: () => void;
}

export function DetectionDetailSheet({
  detection,
  open,
  onOpenChange,
  onFindSimilar,
  onActionComplete,
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
      toast.success(
        detection!.verified ? "Detection unverified" : "Detection verified"
      );
      onActionComplete();
    },
    onError: (err: Error) => toast.error(err.message),
  });

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
          <SheetTitle className="capitalize">
            {detection.species || detection.category}
          </SheetTitle>
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
                  stroke={getCategoryColor(fullDetection.category)}
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
                {detection.species && (
                  <div className="font-medium capitalize">
                    {detection.species} ({detection.species_confidence != null ? `${(detection.species_confidence * 100).toFixed(0)}%` : "—"})
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
            {detection.neighbor_agreement != null && (
              <div>
                <span className="text-muted-foreground">Neighbor agreement</span>
                <p>{Math.round(detection.neighbor_agreement * 10)}/10 agree</p>
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

          {/* Actions */}
          <div className="flex flex-col gap-2 pt-2">
            <Button
              onClick={() => verifyMutation.mutate()}
              disabled={verifyMutation.isPending}
              variant={detection.verified ? "outline" : "default"}
            >
              <Check className="h-4 w-4 mr-2" />
              {detection.verified ? "Unverify" : "Verify"}
            </Button>
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
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
