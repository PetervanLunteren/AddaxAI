/**
 * File verification panel - side panel for verify mode.
 *
 * Shows detection list, label/category editing, draw button, notes,
 * and verification controls.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Check, Binoculars, SquareDashed, SquarePlus, Trash2 } from "lucide-react";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Textarea } from "../ui/textarea";
import { cn } from "../../lib/utils";
import { getDetectionDisplayName } from "../../lib/detection-utils";
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
  detectionThreshold: number;
  labelOptions: LabelOption[];
  labelOptionsLoading: boolean;
  selectedDetectionId: string | null;
  onSelectDetection: (id: string | null) => void;
  openLabelPickerFor?: string | null;
  onLabelPickerOpenChange?: (open: boolean) => void;
  pinnedOptions?: PinnedOption[];
  onAddBox?: () => void;
  canAddBox?: boolean;
  /**
   * Called after a successful verify / notes / delete / relabel mutation.
   * The parent owns its query keys (events vs files vs grid lists) and
   * decides what to invalidate. Mirrors AnnotationCanvas's onMutated.
   */
  onMutated?: () => void;
}

export function FileVerificationPanel({
  file,
  projectId,
  detectionThreshold,
  labelOptions,
  labelOptionsLoading,
  selectedDetectionId,
  onSelectDetection,
  openLabelPickerFor,
  onLabelPickerOpenChange,
  pinnedOptions,
  onAddBox,
  canAddBox,
  onMutated,
}: FileVerificationPanelProps) {
  const [notes, setNotes] = useState(file.notes ?? "");
  const [showNotes, setShowNotes] = useState(false);

  // Verify mutation
  const verifyMutation = useMutation({
    mutationFn: () =>
      filesApi.update(file.id, { verified: !file.verified }),
    onSuccess: () => onMutated?.(),
  });

  // Notes mutation
  const notesMutation = useMutation({
    mutationFn: () => filesApi.update(file.id, { notes }),
    onSuccess: () => onMutated?.(),
  });

  // Delete detection mutation
  const deleteMutation = useMutation({
    mutationFn: (detectionId: string) => detectionsApi.delete(detectionId),
    onSuccess: () => {
      onMutated?.();
      onSelectDetection(null);
    },
  });

  // Update label mutation (sets both category and label in one call)
  const updateLabelMutation = useMutation({
    mutationFn: ({
      detectionId,
      category,
      label,
    }: {
      detectionId: string;
      category: string;
      label: string | null;
    }) => detectionsApi.update(detectionId, { category, label }),
    onSuccess: () => onMutated?.(),
  });

  const isVideo = file.file_type === "video";

  const filteredDetections = useMemo(() => {
    return file.detections.filter((d) => {
      if (d.confidence < detectionThreshold) return false;
      // Event-level observations (no bbox) always surface in the list
      // regardless of frame, because they live at clip level rather
      // than frame level. They render with an "observation" icon
      // instead of a bbox indicator.
      if (d.bbox_x === null) return true;
      // Bboxed detections on videos: only the ones from the best
      // frame are shown, matching what the canvas overlay paints. The
      // others exist in the data but stay out of the verify list for
      // now; surfacing them is a separate follow-up.
      if (isVideo && file.best_frame_number != null) {
        return d.frame_number === file.best_frame_number;
      }
      return true;
    });
  }, [file.detections, detectionThreshold, isVideo, file.best_frame_number]);

  // Build lookup from labelOptions for taxonomy captions
  const labelOptionsByValue = useMemo(() => {
    const map = new Map<string, { displayName: string; caption: string | null; commonName: string | null }>();
    for (const opt of labelOptions) {
      map.set(opt.value, {
        displayName: opt.displayName,
        caption: opt.taxonomyCaption ?? null,
        commonName: opt.label,
      });
    }
    return map;
  }, [labelOptions]);

  const groupedDetections = useMemo(() => {
    const groups = new Map<string, {
      count: number;
      displayName: string;
      commonName: string | null;
      caption: string | null;
    }>();
    for (const d of filteredDetections) {
      const label = d.label || d.category;
      const existing = groups.get(label);
      if (existing) {
        existing.count += 1;
      } else {
        const opt = labelOptionsByValue.get(label);
        groups.set(label, {
          count: 1,
          displayName: d.display_name || opt?.displayName || getDetectionDisplayName(d),
          commonName: opt?.commonName ?? null,
          caption: opt?.caption ?? null,
        });
      }
    }
    return groups;
  }, [filteredDetections, labelOptionsByValue]);

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
              {[...groupedDetections.entries()].map(([label, { count, displayName }]) => (
                <div
                  key={label}
                  className="flex items-center justify-between rounded border p-2 text-sm"
                  style={{ backgroundColor: "#e7efef" }}
                >
                  <span className="truncate">{displayName}</span>
                  <span className="text-muted-foreground shrink-0">&times; {count}</span>
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
              {file.verified_at_utc && (
                <p className="text-xs text-muted-foreground">
                  Verified on{" "}
                  {new Date(file.verified_at_utc).toLocaleDateString(undefined, {
                    month: "short",
                    day: "numeric",
                  })}{" "}
                  at{" "}
                  {new Date(file.verified_at_utc).toLocaleTimeString(undefined, {
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
                      label: option.label,
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
                  <p className="text-xs text-muted-foreground">
                    No detections found.
                  </p>
                  {canAddBox && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={onAddBox}
                      title="Promote highest below-threshold AI box"
                    >
                      <SquarePlus className="h-3.5 w-3.5 mr-1.5" />
                      Promote hidden box
                    </Button>
                  )}
                </div>
              )}
            </div>

            {/* "Draw box" / "Add observation" create-actions live in
                the modal toolbar (Pencil / Binoculars icons) with
                tooltips and keyboard shortcuts (D / N). The
                AddObservationPopover that listens for N is wired
                there too — see FileDetailModal / EventDetailModal. */}

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
  const currentLabel = detection.label || detection.category;
  const currentDisplayName = getDetectionDisplayName(detection);
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
      <div className="flex items-center justify-between gap-1">
        {/* Origin marker: every row tells the user where the
            detection came from. Bboxed rows get a scan icon (the same
            shape the canvas paints); bbox-less event-level
            observations get an eye icon ("seen, no box"). Symmetric
            visual contract — every row signals its origin, not just
            the unusual one. */}
        {detection.bbox_x === null ? (
          <Binoculars
            className="h-3.5 w-3.5 text-muted-foreground shrink-0"
            aria-label="Observation (no bounding box)"
          />
        ) : (
          <SquareDashed
            className="h-3.5 w-3.5 text-muted-foreground shrink-0"
            aria-label="Detection with bounding box"
          />
        )}
        <LabelPicker
          value={currentLabel}
          displayName={currentDisplayName}
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
            {(((detection.label ? detection.label_confidence : null) ?? detection.confidence) * 100).toFixed(0)}%
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
