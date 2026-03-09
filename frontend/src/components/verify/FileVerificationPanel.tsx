/**
 * File verification panel - side panel for verify mode.
 *
 * Shows detection list, species/category editing, draw button, notes,
 * and verification controls.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Pencil, SquarePlus, Trash2 } from "lucide-react";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Textarea } from "../ui/textarea";
import { cn } from "../../lib/utils";
import type { FileWithDetections, DetectionResponse } from "../../api/types";
import type { LabelOption } from "../../hooks/useLabelOptions";

interface PinnedOption {
  key: number;
  option: LabelOption;
}
import { LabelPicker } from "./LabelPicker";

interface FileVerificationPanelProps {
  file: FileWithDetections;
  projectId: string;
  eventId: string;
  detectionThreshold: number;
  labelOptions: LabelOption[];
  labelOptionsLoading: boolean;
  selectedDetectionId: string | null;
  onSelectDetection: (id: string | null) => void;
  openLabelPickerFor?: string | null;
  onLabelPickerOpenChange?: (open: boolean) => void;
  pinnedOptions?: PinnedOption[];
  onDraw?: () => void;
  onAddBox?: () => void;
  canAddBox?: boolean;
}

export function FileVerificationPanel({
  file,
  projectId,
  eventId,
  detectionThreshold,
  labelOptions,
  labelOptionsLoading,
  selectedDetectionId,
  onSelectDetection,
  openLabelPickerFor,
  onLabelPickerOpenChange,
  pinnedOptions,
  onDraw,
  onAddBox,
  canAddBox,
}: FileVerificationPanelProps) {
  const queryClient = useQueryClient();
  const [notes, setNotes] = useState(file.notes ?? "");
  const [showNotes, setShowNotes] = useState(false);

  // Verify mutation
  const verifyMutation = useMutation({
    mutationFn: () =>
      filesApi.update(file.id, { verified: !file.verified }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
      queryClient.invalidateQueries({ queryKey: ["events"] });
    },
  });

  // Notes mutation
  const notesMutation = useMutation({
    mutationFn: () => filesApi.update(file.id, { notes }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
    },
  });

  // Delete detection mutation
  const deleteMutation = useMutation({
    mutationFn: (detectionId: string) => detectionsApi.delete(detectionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
      onSelectDetection(null);
    },
  });

  // Update label mutation (sets both category and species in one call)
  const updateLabelMutation = useMutation({
    mutationFn: ({
      detectionId,
      category,
      species,
    }: {
      detectionId: string;
      category: string;
      species: string | null;
    }) => detectionsApi.update(detectionId, { category, species }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
    },
  });

  const isVideo = file.file_type === "video" || (file.file_type === "frame" && file.source_video_id != null);

  const filteredDetections = useMemo(() => {
    let dets = file.detections.filter((d) => d.confidence >= detectionThreshold);
    // For videos, only show detections from the best frame
    if (isVideo && file.best_frame_number != null) {
      dets = dets.filter((d) => d.frame_number === file.best_frame_number);
    }
    return dets;
  }, [file.detections, detectionThreshold, isVideo, file.best_frame_number]);

  const groupedDetections = useMemo(() => {
    const counts = new Map<string, number>();
    for (const d of filteredDetections) {
      const label = d.species || d.category;
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    return counts;
  }, [filteredDetections]);

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-y-auto">
      {/* Detections card */}
      <div className="relative mx-3 mt-3 rounded-lg border bg-muted/40 flex flex-col min-h-0 flex-1">
        {file.verified ? (
          <>
            {/* Verified badge */}
            <div className="absolute -top-1.5 -right-1.5 bg-primary rounded-full p-0.5">
              <Check className="h-3 w-3 text-primary-foreground" />
            </div>

            {/* Header */}
            <div className="px-3 pt-3 pb-2">
              <div className="flex items-center gap-1.5">
                <h3 className="text-sm font-semibold">Detections</h3>
                <Badge variant="outline" className="text-xs">
                  {filteredDetections.length}
                </Badge>
              </div>
            </div>

            {/* Grouped summary */}
            <div className="flex-1 overflow-y-auto px-3 space-y-1">
              {[...groupedDetections.entries()].map(([label, count]) => (
                <div
                  key={label}
                  className="flex items-center justify-between rounded border p-2 text-sm"
                  style={{ backgroundColor: "#e7efef" }}
                >
                  <span>{label}</span>
                  <span className="text-muted-foreground">&times; {count}</span>
                </div>
              ))}

              {filteredDetections.length === 0 && (
                <div className="text-center text-xs text-muted-foreground py-4">
                  No detections
                </div>
              )}
            </div>

            {/* Verified date + Edit button */}
            <div className="px-3 py-3 space-y-2">
              {file.verified_at && (
                <p className="text-xs text-muted-foreground">
                  Verified on{" "}
                  {new Date(file.verified_at).toLocaleDateString(undefined, {
                    month: "short",
                    day: "numeric",
                  })}{" "}
                  at{" "}
                  {new Date(file.verified_at).toLocaleTimeString(undefined, {
                    hour: "numeric",
                    minute: "2-digit",
                  })}
                </p>
              )}
              <Button
                onClick={() => verifyMutation.mutate()}
                disabled={verifyMutation.isPending}
                variant="outline"
                className="w-full"
                size="sm"
              >
                Edit
              </Button>
            </div>
          </>
        ) : (
          <>
            {/* Header */}
            <div className="px-3 pt-3 pb-2">
              <div className="flex items-center gap-1.5">
                <h3 className="text-sm font-semibold">Detections</h3>
                <Badge variant="outline" className="text-xs">
                  {filteredDetections.length}
                </Badge>
              </div>
            </div>

            {/* Detection list */}
            <div className="flex-1 overflow-y-auto px-3 space-y-1">
              {filteredDetections.map((detection) => (
                <DetectionItem
                  key={detection.id}
                  detection={detection}
                  isSelected={selectedDetectionId === detection.id}
                  onSelect={() =>
                    onSelectDetection(
                      selectedDetectionId === detection.id ? null : detection.id
                    )
                  }
                  onDelete={() => deleteMutation.mutate(detection.id)}
                  onUpdateLabel={(option) =>
                    updateLabelMutation.mutate({
                      detectionId: detection.id,
                      category: option.category,
                      species: option.species,
                    })
                  }
                  labelOptions={labelOptions}
                  labelOptionsLoading={labelOptionsLoading}
                  forceOpenPicker={openLabelPickerFor === detection.id}
                  onPickerOpenChange={onLabelPickerOpenChange}
                  pinnedOptions={pinnedOptions}
                  projectId={projectId}
                />
              ))}

              {filteredDetections.length === 0 && (
                <div className="text-center py-4 space-y-2">
                  <p className="text-xs text-muted-foreground">No detections found.</p>
                  <div className="flex items-center justify-center gap-2">
                    <Button variant="outline" size="sm" onClick={onDraw}>
                      <Pencil className="h-3.5 w-3.5 mr-1.5" />
                      Draw
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={onAddBox}
                      disabled={!canAddBox}
                      title={canAddBox ? "Promote a hidden AI detection" : "No hidden detections available"}
                    >
                      <SquarePlus className="h-3.5 w-3.5 mr-1.5" />
                      Add
                    </Button>
                  </div>
                </div>
              )}
            </div>

            {/* Verify button */}
            <div className="px-3 py-3">
              <Button
                onClick={() => verifyMutation.mutate()}
                disabled={verifyMutation.isPending}
                className="w-full"
                size="sm"
              >
                <Check className="h-4 w-4 mr-2" />
                Mark verified
              </Button>
            </div>
          </>
        )}
      </div>

      {/* Notes */}
      <div className="px-3 py-2">
        {showNotes ? (
          <div className="space-y-1.5">
            <Textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Add notes about this image..."
              className="text-sm resize-none h-24"
            />
            <div className="flex justify-end">
              <button
                onClick={() => {
                  if (notes !== (file.notes ?? "")) {
                    notesMutation.mutate();
                  }
                  setShowNotes(false);
                }}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                Done
              </button>
            </div>
          </div>
        ) : notes ? (
          <button
            onClick={() => setShowNotes(true)}
            className="text-left w-full rounded-lg border bg-muted/40 px-3 py-2"
          >
            <span className="text-xs text-muted-foreground">Notes</span>
            <p className="text-sm truncate">{notes}</p>
          </button>
        ) : (
          <button
            onClick={() => setShowNotes(true)}
            className="text-sm text-muted-foreground hover:text-foreground"
          >
            + Add notes
          </button>
        )}
      </div>

    </div>
  );
}

function DetectionItem({
  detection,
  isSelected,
  onSelect,
  onDelete,
  onUpdateLabel,
  labelOptions,
  labelOptionsLoading,
  forceOpenPicker,
  onPickerOpenChange,
  pinnedOptions,
  projectId,
}: {
  detection: DetectionResponse;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onUpdateLabel: (option: LabelOption) => void;
  labelOptions: LabelOption[];
  labelOptionsLoading: boolean;
  forceOpenPicker?: boolean;
  onPickerOpenChange?: (open: boolean) => void;
  pinnedOptions?: PinnedOption[];
  projectId?: string;
}) {
  const currentLabel = detection.species || detection.category;
  const itemRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isSelected && itemRef.current) {
      itemRef.current.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }, [isSelected]);

  return (
    <div
      ref={itemRef}
      className={cn(
        "rounded border p-2 text-sm cursor-pointer transition-colors",
        isSelected ? "border-primary bg-primary/10" : "hover:bg-gray-50"
      )}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <LabelPicker
          value={currentLabel}
          onSelect={onUpdateLabel}
          options={labelOptions}
          isLoading={labelOptionsLoading}
          forceOpen={forceOpenPicker}
          onOpenChange={onPickerOpenChange}
          pinnedOptions={pinnedOptions}
          projectId={projectId}
        />
        <div className="flex items-center gap-1 ml-auto">
          <span className="text-muted-foreground text-xs tabular-nums">
            {(((detection.species ? detection.species_confidence : null) ?? detection.confidence) * 100).toFixed(0)}%
          </span>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className="text-muted-foreground hover:text-[#882000] p-0.5"
          title="Delete detection"
        >
          <Trash2 className="h-3 w-3" />
        </button>
        </div>
      </div>
    </div>
  );
}
