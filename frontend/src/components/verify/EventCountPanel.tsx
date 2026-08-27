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
 *
 * A row is one cohort. Its second line holds sex, life stage and behaviour
 * for the individuals on that row; Split makes a second row of the same
 * species so 4 males and 2 females can be two rows with their own counts.
 * The species total is the sum. A note is about the visit, so there is one
 * per event, under the list.
 */

import { useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Copy, Minus, Plus, RotateCcw, X } from "lucide-react";
import { toast } from "sonner";
import { eventsApi } from "../../api/events";
import { cn } from "../../lib/utils";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Textarea } from "../ui/textarea";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import { getSpeciesColor } from "../../utils/species-colors";
import { getCategoryColor } from "../../lib/detection-utils";
import { OBSERVATION_ATTRIBUTES } from "../../lib/observation-attributes";
import { LabelPicker } from "./LabelPicker";
import type { LabelOption } from "../../hooks/useLabelOptions";
import type {
  EventObservationItem,
  ObservationAttributesPatch,
} from "../../api/types";
import { useSpeciesColorsVersion } from "../../utils/species-colors";

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
  /** The event's free text (Event.notes). */
  notes: string | null;
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
  notes,
  onConfirm,
  labelOptions,
  labelOptionsLoading,
}: EventCountPanelProps) {
  // Repaint when the project's colour map lands or changes.
  useSpeciesColorsVersion();
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
  const relabelSpecies = useMutation({
    mutationFn: ({ obsId, opt }: { obsId: string; opt: LabelOption }) =>
      eventsApi.relabelObservation(eventId, obsId, {
        category: opt.category,
        label: opt.label ?? null,
      }),
    onSuccess: invalidate,
    onError: onError("change the species"),
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
  const setAttributes = useMutation({
    mutationFn: ({
      obsId,
      patch,
    }: {
      obsId: string;
      patch: ObservationAttributesPatch;
    }) => eventsApi.setObservationAttributes(eventId, obsId, patch),
    onSuccess: invalidate,
    onError: onError("save the details"),
  });
  const splitObs = useMutation({
    mutationFn: (obsId: string) => eventsApi.splitObservation(eventId, obsId),
    onSuccess: invalidate,
    onError: onError("split the row"),
  });
  const setNotes = useMutation({
    mutationFn: (next: string | null) => eventsApi.setNotes(eventId, next),
    onSuccess: invalidate,
    onError: onError("save the note"),
  });
  // While a request runs the buttons are disabled but not dimmed
  // (`disabled:opacity-100`): with three selects per row the default 50%
  // fade made the whole list blink on every edit. The selects are not
  // guarded at all, and neither is the count box: a browser greys a
  // disabled <select> and <input> itself, and a change during another
  // request is just one more PATCH.
  const busy =
    setCount.isPending ||
    addSpecies.isPending ||
    relabelSpecies.isPending ||
    removeObs.isPending ||
    resetCounts.isPending ||
    setAttributes.isPending ||
    splitObs.isPending;

  // Visible rows = species actually present (count > 0). A count-0 row is a
  // removed species, hidden from the editor.
  const visible = observations.filter((o) => o.effective_count > 0);
  // Any override, count-0 removal, human-only row or demographic counts
  // as a human edit (so "Reset to AI" lights up). Not the note: reset is
  // about the counts and leaves it alone.
  const hasHumanEdits = observations.some(
    (o) =>
      o.max_n === 0 ||
      o.effective_count !== o.max_n ||
      o.sex !== null ||
      o.life_stage !== null ||
      o.behavior !== null,
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
  // The note is collapsed by default (a preview, or "+ Add notes") and
  // opens into a textarea. Closes again when the event changes. The text
  // is a local draft while open: binding the textarea to the saved value
  // remounted it on every save, which put the caret at the start and
  // pulled focus back from whatever the user clicked next.
  const [notesExpanded, setNotesExpanded] = useState(false);
  const [draft, setDraft] = useState("");
  const openNotes = () => {
    setDraft(notes ?? "");
    setNotesExpanded(true);
  };
  useEffect(() => setNotesExpanded(false), [eventId]);
  // Row whose species is being changed. Clicking a row's name opens the
  // picker; picking a species relabels the row (count-level: the count moves
  // to the target species, summing if it already has a row). Null = closed.
  const [relabelObs, setRelabelObs] = useState<EventObservationItem | null>(
    null,
  );

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
  // for multi-digit); + / - nudge it by one; R relabels the active row. Bound
  // only while the panel is mounted (i.e. the modal is open). Editing keys are
  // ignored while typing in an input (the count field, the picker, the note)
  // or while a select has focus (its own arrows pick an option).
  useEffect(() => {
    const clearDigitBuffer = () => {
      if (digitTimerRef.current) clearTimeout(digitTimerRef.current);
      digitTimerRef.current = null;
      digitBufferRef.current = null;
    };
    const handler = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
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
      // Relabel the active row — same key ("R") the Labels page binds to
      // its relabel picker, so the muscle memory carries over.
      if (e.key === "r" || e.key === "R") {
        const obs = rows[activeRef.current];
        if (!obs) return;
        e.preventDefault();
        clearDigitBuffer();
        setRelabelObs(obs);
        return;
      }
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
          // Species colour when the observation names a species, the
          // category colour for a bare person / vehicle / animal.
          const speciesKey = obs.label_taxonomy_id || obs.label;
          const swatch = speciesKey
            ? getSpeciesColor(speciesKey)
            : getCategoryColor(obs.category);
          return (
            <div
              key={obs.id}
              onMouseDown={() => setActiveIndex(index)}
              onFocusCapture={() => setActiveIndex(index)}
              className={cn(
                "flex flex-col gap-1 rounded border bg-white px-2 py-1.5 text-sm",
                index === activeIndex && "ring-2 ring-primary/40",
              )}
            >
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-1.5 truncate">
                <span
                  className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm"
                  style={{ backgroundColor: swatch }}
                />
                {/* Click the name to change the species (count carries over).
                    No button: keeps the row uncluttered, the label itself is
                    the target. */}
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => setRelabelObs(obs)}
                  className="-mx-1 -my-0.5 truncate rounded px-1 py-0.5 text-left transition-colors hover:bg-accent disabled:hover:bg-transparent"
                  title={`${name} — click to change species`}
                >
                  {name}
                </button>
              </span>
              <span className="flex shrink-0 items-center gap-1">
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6 disabled:opacity-100"
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
                  className="h-6 w-6 disabled:opacity-100"
                  disabled={busy}
                  title="Increase"
                  onClick={() => applyCount(obs, obs.effective_count + 1)}
                >
                  <Plus className="h-3.5 w-3.5" />
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6 text-muted-foreground disabled:opacity-100"
                  disabled={busy}
                  title="Split into two rows"
                  onClick={() => splitObs.mutate(obs.id)}
                >
                  <Copy className="h-3.5 w-3.5" />
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6 text-muted-foreground disabled:opacity-100"
                  disabled={busy}
                  title="Remove species"
                  onClick={() => applyCount(obs, 0)}
                >
                  <X className="h-3.5 w-3.5" />
                </Button>
              </span>
            </div>
            {/* Line 2: what the individuals on this row are. Empty means
                unknown and is sent as null. Native selects: three per row,
                no portal, and their own arrow keys stay their own. */}
            <div className="flex items-center gap-1">
              {OBSERVATION_ATTRIBUTES.map(({ field, label, options }) => (
                <select
                  key={field}
                  value={obs[field] ?? ""}
                  title={label}
                  aria-label={label}
                  className={cn(
                    "h-6 min-w-0 rounded border bg-white px-1 text-xs focus:outline-none focus:ring-1 focus:ring-primary",
                    field === "behavior" ? "flex-[1.5]" : "flex-1",
                    obs[field] === null && "text-muted-foreground",
                  )}
                  onChange={(e) =>
                    setAttributes.mutate({
                      obsId: obs.id,
                      patch: { [field]: e.currentTarget.value || null },
                    })
                  }
                >
                  <option value="">{label}</option>
                  {options.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              ))}
            </div>
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
          className="flex w-full items-center justify-center gap-1.5 rounded border border-dashed px-2 py-1.5 text-sm text-muted-foreground hover:border-primary/50 hover:text-foreground"
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
        {/* Relabel a row: opened by clicking a species name. Shares the same
            picker; the count carries to whatever species is chosen. */}
        <LabelPicker
          headless
          forceOpen={relabelObs !== null}
          onOpenChange={(open) => {
            if (!open) setRelabelObs(null);
          }}
          value={relabelObs ? relabelObs.label ?? relabelObs.category : null}
          onSelect={(option) => {
            if (relabelObs) {
              relabelSpecies.mutate({ obsId: relabelObs.id, opt: option });
            }
          }}
          options={labelOptions}
          isLoading={labelOptionsLoading}
          projectId={projectId}
        />
      </div>

      {/* Footer: the event's note, reset to the AI proposal, then the
          primary Confirm. Pinned to the panel bottom so Confirm holds a
          stable position. The note follows AddaxAI Connect: collapsed it
          is an "Add notes" link or a two-line preview, expanded it is a
          textarea with Done. Done (or leaving the field) saves; Escape
          closes it without saving. */}
      <div className="shrink-0 border-t px-3 py-2 space-y-2">
        {notesExpanded ? (
          <div className="rounded-md border bg-white p-2">
            <Textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Add notes about this event..."
              aria-label="Notes on this event"
              autoFocus
              maxLength={2000}
              className="min-h-[80px] resize-none border-0 bg-transparent px-1 py-1 text-sm focus-visible:ring-0 focus-visible:ring-offset-0"
              onKeyDown={(e) => {
                if (e.key === "Escape") {
                  // Closes the note only; the modal's DialogContent keeps
                  // Escape from a textarea from closing the modal.
                  setDraft(notes ?? "");
                  setNotesExpanded(false);
                }
              }}
              onBlur={() => {
                const next = draft.trim() || null;
                if (next !== notes) setNotes.mutate(next);
              }}
            />
            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => setNotesExpanded(false)}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                Done
              </button>
            </div>
          </div>
        ) : notes ? (
          <button
            type="button"
            onClick={openNotes}
            className="w-full rounded-md border bg-muted/30 p-2 text-left transition-colors hover:bg-muted/50"
          >
            <p className="mb-0.5 text-xs text-muted-foreground">Notes</p>
            <p className="line-clamp-2 whitespace-pre-line text-sm">{notes}</p>
          </button>
        ) : (
          <button
            type="button"
            onClick={openNotes}
            className="text-sm text-muted-foreground transition-colors hover:text-foreground"
          >
            + Add notes
          </button>
        )}
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
