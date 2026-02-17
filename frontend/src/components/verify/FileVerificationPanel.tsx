/**
 * File verification panel - side panel for verify mode.
 *
 * Shows detection list, species/category editing, draw button, notes,
 * and verification controls.
 */

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Trash2 } from "lucide-react";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Textarea } from "../ui/textarea";
import { cn } from "../../lib/utils";
import type { FileWithDetections, DetectionResponse } from "../../api/types";
import type { LabelOption } from "../../hooks/useLabelOptions";
import { LabelPicker } from "./LabelPicker";

interface FileVerificationPanelProps {
  file: FileWithDetections;
  projectId: string;
  eventId: string;
  detectionThreshold: number;
  labelOptions: LabelOption[];
  labelOptionsLoading: boolean;
}

export function FileVerificationPanel({
  file,
  projectId,
  eventId,
  detectionThreshold,
  labelOptions,
  labelOptionsLoading,
}: FileVerificationPanelProps) {
  const queryClient = useQueryClient();
  const [notes, setNotes] = useState(file.notes ?? "");
  const [showNotes, setShowNotes] = useState(!!file.notes);
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(
    null
  );

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
      setSelectedDetectionId(null);
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

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-y-auto">
      {/* Header */}
      <div className="p-3">
        <div className="flex items-center gap-1.5">
          <h3 className="text-sm font-semibold">Detections</h3>
          <Badge variant="outline" className="text-xs">
            {file.detections.filter((d) => d.confidence >= detectionThreshold).length}
          </Badge>
        </div>
      </div>

      {/* Detection list */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {file.detections.filter((d) => d.confidence >= detectionThreshold).map((detection) => (
          <DetectionItem
            key={detection.id}
            detection={detection}
            isSelected={selectedDetectionId === detection.id}
            onSelect={() =>
              setSelectedDetectionId(
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
          />
        ))}

        {file.detections.length === 0 && (
          <div className="text-center text-xs text-muted-foreground py-4">
            No detections. Draw a box to add one.
          </div>
        )}
      </div>

      {/* Notes */}
      <div className="px-3 pb-2">
        {showNotes ? (
          <div className="space-y-1.5">
            <Textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Add notes about this image..."
              className="text-sm resize-none h-24"
              autoFocus
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
            className="text-left w-full"
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

      {/* Verify button */}
      <div className="p-3">
        <Button
          onClick={() => verifyMutation.mutate()}
          disabled={verifyMutation.isPending}
          variant={file.verified ? "outline" : "default"}
          className="w-full"
          size="sm"
        >
          <Check className="h-4 w-4 mr-2" />
          {file.verified ? "Mark unverified" : "Mark verified"}
        </Button>
        <p className="text-xs text-center text-muted-foreground mt-1">
          Press V to toggle
        </p>
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
}: {
  detection: DetectionResponse;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onUpdateLabel: (option: LabelOption) => void;
  labelOptions: LabelOption[];
  labelOptionsLoading: boolean;
}) {
  const currentLabel = detection.species || detection.category;

  return (
    <div
      className={cn(
        "rounded border p-2 text-sm cursor-pointer transition-colors",
        isSelected ? "border-blue-500 bg-blue-50" : "hover:bg-gray-50"
      )}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <LabelPicker
            value={currentLabel}
            onSelect={onUpdateLabel}
            options={labelOptions}
            isLoading={labelOptionsLoading}
          />
          <span className="text-muted-foreground text-xs">
            {(detection.confidence * 100).toFixed(0)}%
          </span>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className="text-muted-foreground hover:text-red-500 p-0.5"
          title="Delete detection"
        >
          <Trash2 className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
}
