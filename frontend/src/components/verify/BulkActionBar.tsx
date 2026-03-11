/**
 * BulkActionBar - floating bar for bulk operations on selected detections.
 *
 * Appears when one or more detections are selected.
 * Actions: Verify, Relabel, Find similar, Deselect all.
 */

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Tag, Search, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "../ui/button";
import { detectionsApi } from "../../api/detections";
import { LabelPicker } from "./LabelPicker";
import type { LabelOption } from "../../hooks/useLabelOptions";

interface BulkActionBarProps {
  selectedIds: Set<string>;
  onDeselectAll: () => void;
  onFindSimilar: (detectionId: string) => void;
  labelOptions: LabelOption[];
  labelOptionsLoading: boolean;
  onActionComplete: () => void;
  onRelabel?: (ids: string[], label: string | null, category: string) => void;
  onVerify?: (ids: string[]) => void;
  /** Number of selected detections that have a neighbor label suggestion. */
  suggestionCount?: number;
  /** Accept neighbor suggestions for selected detections. */
  onAcceptSuggestions?: () => void;
  projectId?: string;
}

export function BulkActionBar({
  selectedIds,
  onDeselectAll,
  onFindSimilar,
  labelOptions,
  labelOptionsLoading,
  onActionComplete,
  onRelabel,
  onVerify,
  suggestionCount = 0,
  onAcceptSuggestions,
  projectId,
}: BulkActionBarProps) {
  const queryClient = useQueryClient();
  const [relabelOpen, setRelabelOpen] = useState(false);
  const count = selectedIds.size;
  const ids = Array.from(selectedIds);

  const verifyMutation = useMutation({
    mutationFn: () => detectionsApi.bulkVerify(ids, true),
    onSuccess: (_data) => {
      if (onVerify) {
        onVerify(ids);
      } else {
        onActionComplete();
      }
      onDeselectAll();
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const relabelMutation = useMutation({
    mutationFn: (opt: LabelOption) =>
      detectionsApi.bulkRelabel(ids, opt.label, opt.category),
    onSuccess: (_data, opt) => {
      if (onRelabel) {
        onRelabel(ids, opt.label, opt.category);
      } else {
        onActionComplete();
      }
      onDeselectAll();
    },
    onError: (err: Error) => toast.error(err.message),
  });

  if (count === 0) return null;

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 bg-background/95 backdrop-blur border rounded-xl shadow-lg px-4 py-2.5 flex items-center gap-3">
      <span className="text-sm font-medium min-w-[80px]">
        {count} selected
      </span>

      <Button
        variant="default"
        size="sm"
        onClick={() => verifyMutation.mutate()}
        disabled={verifyMutation.isPending}
      >
        <Check className="h-4 w-4 mr-1" />
        Verify
      </Button>

      <div className="relative">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setRelabelOpen(!relabelOpen)}
        >
          <Tag className="h-4 w-4 mr-1" />
          Relabel
        </Button>
        <div className="absolute bottom-full mb-2 left-0 z-50">
          <LabelPicker
            value=""
            onSelect={(opt) => {
              relabelMutation.mutate(opt);
            }}
            options={labelOptions}
            isLoading={labelOptionsLoading}
            forceOpen={relabelOpen}
            onOpenChange={(open) => {
              if (!open) setRelabelOpen(false);
            }}
            projectId={projectId}
          />
        </div>
      </div>

      <Button
        variant="outline"
        size="sm"
        onClick={() => {
          const first = selectedIds.values().next().value;
          if (first) onFindSimilar(first);
        }}
      >
        <Search className="h-4 w-4 mr-1" />
        Find similar
      </Button>

      {suggestionCount > 0 && onAcceptSuggestions && (
        <Button
          variant="outline"
          size="sm"
          onClick={onAcceptSuggestions}
        >
          <Tag className="h-4 w-4 mr-1" />
          Accept {suggestionCount} suggestion{suggestionCount !== 1 ? "s" : ""}
        </Button>
      )}

      <Button variant="ghost" size="sm" onClick={onDeselectAll}>
        <X className="h-4 w-4 mr-1" />
        Deselect
      </Button>
    </div>
  );
}
