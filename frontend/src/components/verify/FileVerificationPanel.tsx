/**
 * File verification panel - side panel for verify mode.
 *
 * Shows detection list, species/category editing, draw button, notes,
 * and verification controls.
 */

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Trash2, Plus, Pencil } from "lucide-react";
import { filesApi } from "../../api/files";
import { detectionsApi } from "../../api/detections";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Textarea } from "../ui/textarea";
import { cn } from "../../lib/utils";
import { getCategoryColor } from "../../lib/detection-utils";
import type { FileWithDetections, DetectionResponse } from "../../api/types";

interface FileVerificationPanelProps {
  file: FileWithDetections;
  projectId: string;
  eventId: string;
  drawMode: boolean;
  onDrawModeChange: (active: boolean) => void;
  detectionThreshold: number;
}

export function FileVerificationPanel({
  file,
  projectId,
  eventId,
  drawMode,
  onDrawModeChange,
  detectionThreshold,
}: FileVerificationPanelProps) {
  const queryClient = useQueryClient();
  const [notes, setNotes] = useState(file.notes ?? "");
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

  // Update species mutation
  const updateSpeciesMutation = useMutation({
    mutationFn: ({
      detectionId,
      species,
    }: {
      detectionId: string;
      species: string;
    }) => detectionsApi.update(detectionId, { species }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
    },
  });

  // Update category mutation
  const updateCategoryMutation = useMutation({
    mutationFn: ({
      detectionId,
      category,
    }: {
      detectionId: string;
      category: string;
    }) => detectionsApi.update(detectionId, { category }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["event", eventId] });
    },
  });

  return (
    <div className="w-80 border-l bg-white flex flex-col shrink-0 overflow-y-auto">
      {/* Header with draw button */}
      <div className="p-3 border-b">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold">Detections</h3>
          <Badge variant="outline" className="text-xs">
            {file.detections.filter((d) => d.confidence >= detectionThreshold).length}
          </Badge>
        </div>
        <Button
          variant={drawMode ? "default" : "outline"}
          size="sm"
          className="w-full"
          onClick={() => onDrawModeChange(!drawMode)}
        >
          {drawMode ? (
            <>
              <Pencil className="h-3.5 w-3.5 mr-1.5" />
              Drawing... (click to stop)
            </>
          ) : (
            <>
              <Plus className="h-3.5 w-3.5 mr-1.5" />
              Draw new box
            </>
          )}
        </Button>
        <p className="text-xs text-center text-muted-foreground mt-1">
          Press D to toggle
        </p>
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
            onUpdateSpecies={(species) =>
              updateSpeciesMutation.mutate({
                detectionId: detection.id,
                species,
              })
            }
            onUpdateCategory={(category) =>
              updateCategoryMutation.mutate({
                detectionId: detection.id,
                category,
              })
            }
          />
        ))}

        {file.detections.length === 0 && (
          <div className="text-center text-xs text-muted-foreground py-4">
            No detections. Draw a box to add one.
          </div>
        )}
      </div>

      {/* Notes */}
      <div className="p-3 border-t space-y-2">
        <label className="text-xs font-medium text-muted-foreground">
          Notes
        </label>
        <Textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Add notes..."
          className="text-sm resize-none h-20"
          onBlur={() => {
            if (notes !== (file.notes ?? "")) {
              notesMutation.mutate();
            }
          }}
        />
      </div>

      {/* Verify button */}
      <div className="p-3 border-t">
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

const CATEGORIES = ["animal", "person", "vehicle"] as const;

function DetectionItem({
  detection,
  isSelected,
  onSelect,
  onDelete,
  onUpdateSpecies,
  onUpdateCategory,
}: {
  detection: DetectionResponse;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onUpdateSpecies: (species: string) => void;
  onUpdateCategory: (category: string) => void;
}) {
  const [editingSpecies, setEditingSpecies] = useState(false);
  const [speciesValue, setSpeciesValue] = useState(detection.species ?? "");
  const color = getCategoryColor(detection.category);

  return (
    <div
      className={cn(
        "rounded border p-2 text-sm cursor-pointer transition-colors",
        isSelected ? "border-blue-500 bg-blue-50" : "hover:bg-gray-50"
      )}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className="w-2.5 h-2.5 rounded-full shrink-0"
            style={{ backgroundColor: color }}
          />
          {/* Category selector */}
          <select
            value={detection.category}
            onChange={(e) => {
              e.stopPropagation();
              onUpdateCategory(e.target.value);
            }}
            onClick={(e) => e.stopPropagation()}
            className="text-sm font-medium capitalize bg-transparent border-none p-0 cursor-pointer focus:ring-0 focus:outline-none"
          >
            {CATEGORIES.map((cat) => (
              <option key={cat} value={cat} className="capitalize">
                {cat}
              </option>
            ))}
          </select>
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

      {/* Species */}
      <div className="mt-1">
        {editingSpecies ? (
          <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
            <input
              type="text"
              value={speciesValue}
              onChange={(e) => setSpeciesValue(e.target.value)}
              className="flex-1 text-xs border rounded px-1.5 py-0.5"
              placeholder="Species name"
              autoFocus
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  onUpdateSpecies(speciesValue);
                  setEditingSpecies(false);
                }
                if (e.key === "Escape") {
                  setEditingSpecies(false);
                  setSpeciesValue(detection.species ?? "");
                }
              }}
              onBlur={() => {
                if (speciesValue !== (detection.species ?? "")) {
                  onUpdateSpecies(speciesValue);
                }
                setEditingSpecies(false);
              }}
            />
          </div>
        ) : (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setEditingSpecies(true);
            }}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            {detection.species ? (
              <span className="capitalize">{detection.species}</span>
            ) : (
              <span className="italic">+ Add species</span>
            )}
            {detection.species_confidence != null && (
              <span className="ml-1">
                ({(detection.species_confidence * 100).toFixed(0)}%)
              </span>
            )}
          </button>
        )}
      </div>
    </div>
  );
}
