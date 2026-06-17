/**
 * Event count panel — the Counts-page count editor and the star of the
 * modal. The right-hand panel of EventDetailModal.
 *
 * Owns everything about the event's counts: the species rows (effective
 * count = human override, else AI MaxN), the +/- steppers and inline count
 * input, removing a species, the event-wide "reset to AI", and the
 * keyboard accelerators (up/down to pick a row, a digit to set its count).
 * Confirm+advance is handed in from the modal (`onConfirm`) so the Enter key
 * and this button share one path.
 *
 * Rows with an effective count of 0 are hidden: that is how a species is
 * "removed" (a human-added row is deleted; an AI row keeps its boxes but its
 * count drops to 0, which survives a MaxN recompute and leaves the
 * ecological exports). Re-add via the picker or "Reset to AI".
 */

import { useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Minus, Plus, RotateCcw, X } from "lucide-react";
import { toast } from "sonner";
import { eventsApi } from "../../api/events";
import { cn } from "../../lib/utils";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import { getSpeciesColor } from "../../utils/species-colors";
import { LabelPicker } from "./LabelPicker";
import type { LabelOption } from "../../hooks/useLabelOptions";
import type { EventObservationItem } from "../../api/types";

// How long a typed digit stays "open" to be extended by the next one, so
// "1" then "2" within the window sets 12 instead of 2. Single digits still
// apply instantly; the next digit just revises the number while it's fresh.
const DIGIT_WINDOW_MS = 700;

interface EventCountPanelProps {
  eventId: string;
  projectId: string;
  /** Full observation list (including hidden count-0 rows). */
  observations: EventObservationItem[];
  confirmed: boolean;
  /** Confirm the event and jump to the next unconfirmed one (modal owns it,
   *  shared with the Enter key). */
  onConfirm: () => void;
  labelOptions: LabelOption[];
  labelOptionsLoading: boolean;
}

export function EventCountPanel({
  eventId,
  projectId,
  observations,
  confirmed,
  onConfirm,
  labelOptions,
  labelOptionsLoading,
}: EventCountPanelProps) {
  const queryClient = useQueryClient();
  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["event", eventId] });
    queryClient.invalidateQueries({ queryKey: ["events"] });
  };
  const onError = (verb: string) => (e: Error) =>
    toast.error(`Could not ${verb}`, { description: e.message });

  const setCount = useMutation({
    mutationFn: ({ obsId, count }: { obsId: string; count: number }) =>
      eventsApi.setObservationCount(eventId, obsId, count),
    onSuccess: invalidate,
    onError: onError("set the count"),
  });
  const addSpecies = useMutation({
    mutationFn: (opt: LabelOption) =>
      eventsApi.addObservation(eventId, {
        category: opt.category,
        count: 1,
        label: opt.label ?? null,
      }),
    onSuccess: invalidate,
    onError: onError("add the species"),
  });
  const removeObs = useMutation({
    mutationFn: (obsId: string) => eventsApi.deleteObservation(eventId, obsId),
    onSuccess: invalidate,
    onError: onError("remove the species"),
  });
  const resetCounts = useMutation({
    mutationFn: () => eventsApi.resetCounts(eventId),
    onSuccess: invalidate,
    onError: onError("reset the counts"),
  });
  const busy =
    setCount.isPending ||
    addSpecies.isPending ||
    removeObs.isPending ||
    resetCounts.isPending;

  // Visible rows = species actually present (count > 0). A count-0 row is a
  // removed species, hidden from the editor.
  const visible = observations.filter((o) => o.effective_count > 0);
  // Any override, count-0 removal, or human-only row counts as a human edit
  // (so "Reset to AI" lights up).
  const hasHumanEdits = observations.some(
    (o) => o.max_n === 0 || o.effective_count !== o.max_n,
  );

  // Set a count, treating 0 as "remove": an AI row goes to count 0 (survives
  // recompute), a human-only row is deleted outright.
  const applyCount = (obs: EventObservationItem, count: number) => {
    const next = Math.max(0, count);
    if (next === 0 && obs.max_n === 0) {
      removeObs.mutate(obs.id);
    } else {
      setCount.mutate({ obsId: obs.id, count: next });
    }
  };

  const [addOpen, setAddOpen] = useState(false);

  // Active row for the keyboard accelerators. Refs keep the window listener
  // stable while still reading the latest values.
  const [activeIndex, setActiveIndex] = useState(0);
  const visibleRef = useRef(visible);
  visibleRef.current = visible;
  const activeRef = useRef(0);
  activeRef.current = activeIndex;
  // Digit accumulation for multi-digit count entry (see DIGIT_WINDOW_MS).
  const digitBufferRef = useRef<{ obsId: string; digits: string } | null>(null);
  const digitTimerRef = useRef<number | null>(null);

  // Keep the active row in range and reset it when the event changes.
  useEffect(() => {
    setActiveIndex(0);
    if (digitTimerRef.current) clearTimeout(digitTimerRef.current);
    digitTimerRef.current = null;
    digitBufferRef.current = null;
  }, [eventId]);
  useEffect(() => {
    if (activeIndex > visible.length - 1) {
      setActiveIndex(Math.max(0, visible.length - 1));
    }
  }, [visible.length, activeIndex]);

  // up/down pick a species row; digits set the active row's count (type fast
  // for multi-digit); + / - nudge it by one. Bound only while the panel is
  // mounted (i.e. the modal is open). Editing keys are ignored while typing in
  // an input (the count field, the picker).
  useEffect(() => {
    const clearDigitBuffer = () => {
      if (digitTimerRef.current) clearTimeout(digitTimerRef.current);
      digitTimerRef.current = null;
      digitBufferRef.current = null;
    };
    const handler = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }
      // Add a species — works even with no rows yet (that's when you most
      // need it), so handle it before the empty-list guard.
      if (e.key === "a" || e.key === "A") {
        e.preventDefault();
        clearDigitBuffer();
        setAddOpen(true);
        return;
      }
      const rows = visibleRef.current;
      if (rows.length === 0) return;
      if (e.key === "ArrowUp") {
        e.preventDefault();
        clearDigitBuffer();
        setActiveIndex((i) => (i <= 0 ? rows.length - 1 : i - 1));
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        clearDigitBuffer();
        setActiveIndex((i) => (i >= rows.length - 1 ? 0 : i + 1));
      } else if (e.key >= "0" && e.key <= "9") {
        const obs = rows[activeRef.current];
        if (!obs) return;
        e.preventDefault();
        // Extend the open buffer for this row, else start fresh. Cap at 4
        // digits (9999) so a key-mash can't request an absurd count.
        const buf = digitBufferRef.current;
        const prev = buf && buf.obsId === obs.id ? buf.digits : "";
        const digits = (prev + e.key).slice(0, 4);
        digitBufferRef.current = { obsId: obs.id, digits };
        if (digitTimerRef.current) clearTimeout(digitTimerRef.current);
        digitTimerRef.current = window.setTimeout(clearDigitBuffer, DIGIT_WINDOW_MS);
        applyCount(obs, Number(digits));
      } else if (e.key === "+" || e.key === "=" || e.key === "-") {
        const obs = rows[activeRef.current];
        if (!obs) return;
        e.preventDefault();
        // Nudge from the in-progress number if one is open, else the count.
        const buf = digitBufferRef.current;
        const base =
          buf && buf.obsId === obs.id ? Number(buf.digits) : obs.effective_count;
        clearDigitBuffer();
        applyCount(obs, base + (e.key === "-" ? -1 : 1));
      }
    };
    window.addEventListener("keydown", handler);
    return () => {
      window.removeEventListener("keydown", handler);
      if (digitTimerRef.current) clearTimeout(digitTimerRef.current);
    };
    // applyCount is stable enough; rows/active come from refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="mx-3 mt-3 mb-2 flex min-h-0 flex-1 flex-col rounded-lg border bg-muted/40">
      <div className="flex shrink-0 items-center gap-1.5 px-3 pt-3 pb-2">
        <h3 className="text-sm font-semibold">Counts</h3>
        <Badge variant="outline" className="text-xs">
          {visible.length}
        </Badge>
        {/* Persistent confirmed cue: clearer than the Confirm button's
            fill change, and uses the same teal/check as the grid badge. */}
        {confirmed && (
          <span
            className="ml-auto inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium text-white"
            style={{ backgroundColor: "#0f6064" }}
          >
            <Check className="h-3 w-3" strokeWidth={3} />
            Confirmed
          </span>
        )}
      </div>

      {/* Scrollable species list; the footer (reset + confirm) stays pinned
          below so a long list never pushes Confirm off-screen. */}
      <div className="min-h-0 flex-1 overflow-y-auto px-3 pt-1 space-y-1">
        {visible.map((obs, index) => {
          const name = resolveSpeciesName({
            common_name: obs.common_name,
            scientific_name: obs.scientific_name,
            label: obs.label,
            category: obs.category,
          });
          const colorKey = obs.label_taxonomy_id || obs.label || obs.category;
          return (
            <div
              key={obs.id}
              onMouseDown={() => setActiveIndex(index)}
              className={cn(
                "flex items-center justify-between gap-2 rounded border bg-white px-2 py-1.5 text-sm",
                index === activeIndex && "ring-2 ring-primary/40",
              )}
            >
              <span className="flex items-center gap-1.5 truncate">
                <span
                  className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm"
                  style={{ backgroundColor: getSpeciesColor(colorKey) }}
                />
                <span className="truncate" title={name}>
                  {name}
                </span>
              </span>
              <span className="flex shrink-0 items-center gap-1">
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  disabled={busy}
                  title="Decrease"
                  onClick={() => applyCount(obs, obs.effective_count - 1)}
                >
                  <Minus className="h-3.5 w-3.5" />
                </Button>
                <input
                  key={obs.effective_count}
                  type="text"
                  inputMode="numeric"
                  defaultValue={obs.effective_count}
                  disabled={busy}
                  className="w-9 rounded border bg-white px-1 py-0.5 text-center text-sm tabular-nums focus:outline-none focus:ring-1 focus:ring-primary"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") e.currentTarget.blur();
                  }}
                  onBlur={(e) => {
                    const n = parseInt(e.currentTarget.value, 10);
                    if (Number.isFinite(n) && n !== obs.effective_count) {
                      applyCount(obs, n);
                    }
                  }}
                />
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  disabled={busy}
                  title="Increase"
                  onClick={() => applyCount(obs, obs.effective_count + 1)}
                >
                  <Plus className="h-3.5 w-3.5" />
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6 text-muted-foreground"
                  disabled={busy}
                  title="Remove species"
                  onClick={() => applyCount(obs, 0)}
                >
                  <X className="h-3.5 w-3.5" />
                </Button>
              </span>
            </div>
          );
        })}

        {visible.length === 0 && (
          <div className="py-3 text-center text-xs text-muted-foreground">
            No species recorded
          </div>
        )}

        {/* Add a species the AI missed. Full-width target (its own button
            drives a headless picker). */}
        <button
          onClick={() => setAddOpen(true)}
          disabled={busy}
          className="flex w-full items-center justify-center gap-1.5 rounded border border-dashed px-2 py-1.5 text-sm text-muted-foreground hover:border-primary/50 hover:text-foreground disabled:opacity-50"
        >
          <Plus className="h-3.5 w-3.5" />
          Add species
        </button>
        <LabelPicker
          headless
          forceOpen={addOpen}
          onOpenChange={setAddOpen}
          value={null}
          onSelect={(option) => addSpecies.mutate(option)}
          options={labelOptions}
          isLoading={labelOptionsLoading}
          projectId={projectId}
        />
      </div>

      {/* Footer: reset to the AI proposal, then the primary Confirm. Pinned
          to the panel bottom so Confirm holds a stable position. */}
      <div className="shrink-0 border-t px-3 py-2 space-y-2">
        {hasHumanEdits && (
          <Button
            variant="ghost"
            onClick={() => resetCounts.mutate()}
            disabled={busy}
            className="w-full gap-1.5 text-muted-foreground"
          >
            <RotateCcw className="h-4 w-4" />
            Reset to AI
          </Button>
        )}
        <Button
          variant={confirmed ? "outline" : "default"}
          onClick={onConfirm}
          className="w-full gap-1.5"
        >
          <Check className="h-4 w-4" />
          {confirmed ? "Confirmed" : "Confirm"}
        </Button>
      </div>
    </div>
  );
}
