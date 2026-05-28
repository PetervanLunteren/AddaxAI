/**
 * BulkActionBar - floating bar for bulk operations on selected detections.
 *
 * Appears when one or more detections are selected.
 * Actions: Verify, Mark false, Match majority, Relabel, Deselect.
 */

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Ban, Check, CheckCheck, Tag, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "../ui/button";
import { detectionsApi } from "../../api/detections";
import { LabelPicker } from "./LabelPicker";
import type { LabelOption } from "../../hooks/useLabelOptions";

interface BulkActionBarProps {
  selectedIds: Set<string>;
  onDeselectAll: () => void;
  labelOptions: LabelOption[];
  labelOptionsLoading: boolean;
  onActionComplete: () => void;
  onRelabel?: (ids: string[], label: string | null, category: string, displayName: string) => void;
  onVerify?: (ids: string[]) => void;
  onMarkFalse?: (ids: string[]) => void;
  /** Relabel the selection to its most common label and verify. The
   *  parent owns the mode-finding + API call because it has the full
   *  detection metadata; this prop just wires the button. */
  onMatchMajority?: (ids: string[]) => void;
  projectId?: string;
  /** Controlled state for the relabel picker (keyboard shortcut). */
  relabelOpen?: boolean;
  onRelabelOpenChange?: (open: boolean) => void;
}

export function BulkActionBar({
  selectedIds,
  onDeselectAll,
  labelOptions,
  labelOptionsLoading,
  onActionComplete,
  onRelabel,
  onVerify,
  onMarkFalse,
  onMatchMajority,
  projectId,
  relabelOpen: relabelOpenProp,
  onRelabelOpenChange,
}: BulkActionBarProps) {
  const queryClient = useQueryClient();
  const [relabelOpenLocal, setRelabelOpenLocal] = useState(false);
  const relabelOpen = relabelOpenProp ?? relabelOpenLocal;
  const setRelabelOpen = onRelabelOpenChange ?? setRelabelOpenLocal;
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

  const markFalseMutation = useMutation({
    mutationFn: () => detectionsApi.bulkRelabel(ids, "false detection", undefined),
    onSuccess: () => {
      if (onMarkFalse) {
        onMarkFalse(ids);
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
        onRelabel(ids, opt.label, opt.category, opt.displayName);
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
        <kbd className="ml-1.5 text-[10px] font-sans text-primary-foreground/60 border border-primary-foreground/30 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(255,255,255,0.1)] leading-none">⏎</kbd>
      </Button>

      <Button
        variant="outline"
        size="sm"
        onClick={() => markFalseMutation.mutate()}
        disabled={markFalseMutation.isPending}
      >
        <Ban className="h-4 w-4 mr-1" />
        Mark false
        <kbd className="ml-1.5 text-[10px] font-sans text-muted-foreground/60 border border-border/60 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(0,0,0,0.08)] leading-none">X</kbd>
      </Button>

      {onMatchMajority && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => onMatchMajority(ids)}
        >
          <CheckCheck className="h-4 w-4 mr-1" />
          Match majority
          <kbd className="ml-1.5 text-[10px] font-sans text-muted-foreground/60 border border-border/60 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(0,0,0,0.08)] leading-none">M</kbd>
        </Button>
      )}

      <div className="relative">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setRelabelOpen(!relabelOpen)}
        >
          <Tag className="h-4 w-4 mr-1" />
          Relabel
          <kbd className="ml-1.5 text-[10px] font-sans text-muted-foreground/60 border border-border/60 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(0,0,0,0.08)] leading-none">R</kbd>
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
            headless
          />
        </div>
      </div>

      <Button variant="outline" size="sm" onClick={onDeselectAll}>
        <X className="h-4 w-4 mr-1" />
        Deselect
        <kbd className="ml-1.5 text-[10px] font-sans text-muted-foreground/60 border border-border/60 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(0,0,0,0.08)] leading-none">Esc</kbd>
      </Button>
    </div>
  );
}
