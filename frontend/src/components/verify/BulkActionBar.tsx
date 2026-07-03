/**
 * BulkActionBar - floating bar for bulk operations on selected detections.
 *
 * Appears when one or more detections are selected.
 * Actions: Verify, Mark false, Match majority, Relabel, Deselect.
 */

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Ban, Check, CheckCheck, CircleHelp, Tag, X } from "lucide-react";
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
  /** Mark the selection as "unknown" (a real, unidentifiable observation)
   *  and verify, so it leaves the queue but stays countable. */
  onMarkUnknown?: (ids: string[]) => void;
  /** Relabel the selection to its most common label and verify. The
   *  parent owns the mode-finding + API call because it has the full
   *  detection metadata; this prop just wires the button. */
  onMatchMajority?: (ids: string[]) => void;
  /** Display label of the selection's majority, shown on the
   *  Match-majority button ("Set all to Corvus"). Null when the
   *  selection carries no labels, in which case the button hides. */
  majorityLabel?: string | null;
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
  onMarkUnknown,
  onMatchMajority,
  majorityLabel,
  projectId,
  relabelOpen: relabelOpenProp,
  onRelabelOpenChange,
}: BulkActionBarProps) {
  const [relabelOpenLocal, setRelabelOpenLocal] = useState(false);
  const relabelOpen = relabelOpenProp ?? relabelOpenLocal;
  const setRelabelOpen = onRelabelOpenChange ?? setRelabelOpenLocal;
  const count = selectedIds.size;
  const ids = Array.from(selectedIds);

  const verifyMutation = useMutation({
    mutationFn: () => detectionsApi.bulkVerify(ids, true),
    onSuccess: () => {
      if (onVerify) {
        // Parent advances the selection to the next card; don't clear here.
        onVerify(ids);
      } else {
        onActionComplete();
        onDeselectAll();
      }
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
        onDeselectAll();
      }
    },
    onError: (err: Error) => toast.error(err.message),
  });

  // "Unknown" is a real observation (unlike "false detection"): keep the
  // category, just relabel to unknown and verify so it leaves the queue.
  const markUnknownMutation = useMutation({
    mutationFn: () => detectionsApi.bulkRelabel(ids, "unknown", undefined),
    onSuccess: () => {
      if (onMarkUnknown) {
        onMarkUnknown(ids);
      } else {
        onActionComplete();
        onDeselectAll();
      }
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
        onDeselectAll();
      }
    },
    onError: (err: Error) => toast.error(err.message),
  });

  if (count === 0) return null;

  return (
    <>
      {/* Gradient blur scrim behind the floating bar, so the grid crops
          fade out under it instead of butting hard against the pill.
          Mirrors the folder-run stepper's blurred Back / Continue bar.
          Non-interactive; the mask fades the blur upward so there's no
          hard blur edge. */}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-x-0 bottom-0 z-40 h-28 bg-gradient-to-t from-background via-background/60 to-transparent backdrop-blur-sm [mask-image:linear-gradient(to_top,black,transparent)] [-webkit-mask-image:linear-gradient(to_top,black,transparent)]"
      />
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

      <Button
        variant="outline"
        size="sm"
        onClick={() => markUnknownMutation.mutate()}
        disabled={markUnknownMutation.isPending}
        title="Mark as an unidentifiable animal and verify"
      >
        <CircleHelp className="h-4 w-4 mr-1" />
        Unknown
        <kbd className="ml-1.5 text-[10px] font-sans text-muted-foreground/60 border border-border/60 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(0,0,0,0.08)] leading-none">U</kbd>
      </Button>

      {onMatchMajority && majorityLabel && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => onMatchMajority(ids)}
          title={`Relabel all ${count} to ${majorityLabel} and verify`}
        >
          <CheckCheck className="h-4 w-4 mr-1 shrink-0" />
          <span className="truncate max-w-[180px]">
            Set all to <span className="capitalize">{majorityLabel}</span>
          </span>
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

      <Button
        variant="outline"
        size="sm"
        className="w-9 px-0"
        onClick={onDeselectAll}
        title="Deselect (Esc)"
        aria-label="Deselect"
      >
        <X className="h-4 w-4" />
      </Button>
      </div>
    </>
  );
}
